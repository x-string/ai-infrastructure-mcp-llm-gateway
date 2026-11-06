"""Relaying the upstream stream to the client, redacted (Task 3).

The client receives chunks the gateway builds itself, in this order:

1. a role chunk, sent as soon as the response starts;
2. content and refusal chunks, each time a redactor releases text (one redactor per channel);
3. at the end, the redactors' last text, one finish chunk, a usage chunk if the client asked for
   ``stream_options.include_usage`` and the provider reported usage, then ``data: [DONE]``.

Every chunk carries the same ``id``, ``created`` and ``model``. If anything fails once the
response has started, the held-back text is discarded (it may be the start of PII), and the
stream ends with one ``data: {"error": ...}`` frame and no ``[DONE]``: the OpenAI SDKs raise
``APIError`` for it. A stream running longer than its maximum duration ends the same way.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Protocol

import anyio

from ai_gateway.llm_gateway.events import Channel, KeepAlive, StreamEnd, TextDelta, Usage
from ai_gateway.llm_gateway.redaction import REDACTED, RedactionRules, StreamRedactor
from ai_gateway.llm_gateway.sse import DONE, encode_comment, encode_json
from ai_gateway.llm_gateway.upstream import StreamItem, UpstreamError

logger = logging.getLogger(__name__)


class ItemSource(Protocol):
    def items(self) -> AsyncIterator[StreamItem]: ...

    async def aclose(self) -> None: ...


@dataclass
class StreamRecord:
    """What happened to one stream, for the log line written when it ends."""

    request_id: str
    started: float = field(default_factory=time.perf_counter)
    outcome: str = "client_disconnected"
    finish_reason: str = "-"
    error_code: str = "-"
    detail: str = "-"
    redactions: int = 0

    def emit(self) -> None:
        logger.info(
            "request_id=%s outcome=%s finish=%s error=%s detail=%s redactions=%d duration_ms=%.1f",
            self.request_id,
            self.outcome,
            self.finish_reason,
            self.error_code,
            self.detail,
            self.redactions,
            (time.perf_counter() - self.started) * 1000,
        )


class ChunkWriter:
    """Builds the outgoing chunks, all with the same id, creation time and model."""

    def __init__(self, model: str) -> None:
        self.id = f"chatcmpl-{secrets.token_urlsafe(18)}"
        self.created = int(time.time())
        self.model = model

    def delta(self, delta: dict[str, str], finish_reason: str | None = None) -> bytes:
        choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
        return encode_json(self._chunk([choice]))

    def usage(self, usage: Usage) -> bytes:
        return encode_json(
            self._chunk(
                [],
                usage={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
            )
        )

    def _chunk(self, choices: list[dict], **extra: object) -> dict[str, object]:
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices,
            **extra,
        }


def error_type(status: int) -> str:
    return {400: "invalid_request_error", 429: "rate_limit_error"}.get(status, "server_error")


def error_body(error: UpstreamError) -> dict[str, object]:
    """The OpenAI-style error object: a generic message, never the provider's own text."""
    return {
        "error": {
            "message": error.message,
            "type": error_type(error.status),
            "code": error.code,
            "param": None,
        }
    }


async def relay(
    source: ItemSource,
    *,
    model: str,
    rules: RedactionRules,
    include_usage: bool,
    max_seconds: float,
    keepalive_seconds: float,
    record: StreamRecord,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[bytes]:
    """The client's stream: redacted chunks rebuilt from the upstream's items."""
    writer = ChunkWriter(model)
    redactors = {channel: StreamRedactor(rules) for channel in Channel}
    deadline = clock() + max_seconds
    items = source.items()
    try:
        yield writer.delta({"role": "assistant", "content": ""})
        last_sent = clock()
        while True:
            try:
                # The deadline covers each wait for the upstream, never a yield to the client.
                with anyio.fail_after(max(deadline - clock(), 0)):
                    item = await anext(items)
            except TimeoutError:
                raise UpstreamError(
                    504, "upstream_timeout", retryable=True, detail="maximum stream duration"
                ) from None
            if isinstance(item, TextDelta):
                released = redactors[item.channel].feed(item.text)
                if released:
                    record.redactions += released.count(REDACTED)
                    yield writer.delta({item.channel.value: released})
                    last_sent = clock()
            elif isinstance(item, KeepAlive):
                if clock() - last_sent >= keepalive_seconds:
                    yield encode_comment("keep-alive")
                    last_sent = clock()
            elif isinstance(item, StreamEnd):
                for channel, redactor in redactors.items():
                    rest = redactor.flush()
                    if rest:
                        record.redactions += rest.count(REDACTED)
                        yield writer.delta({channel.value: rest})
                yield writer.delta({}, item.finish_reason)
                if include_usage and item.usage is not None:
                    yield writer.usage(item.usage)
                yield DONE
                record.outcome, record.finish_reason = "completed", item.finish_reason
                return
    except UpstreamError as error:
        _fail(redactors, record, error.code, error.detail)
        yield encode_json(error_body(error))
    except Exception:
        logger.exception("Unexpected error while relaying stream %s", record.request_id)
        _fail(redactors, record, "internal_error", "unexpected error")
        yield encode_json(
            {
                "error": {
                    "message": "Internal error",
                    "type": "server_error",
                    "code": "internal_error",
                    "param": None,
                }
            }
        )
    finally:
        with anyio.CancelScope(shield=True):
            await items.aclose()
            await source.aclose()


def _fail(
    redactors: dict[Channel, StreamRedactor], record: StreamRecord, code: str, detail: str
) -> None:
    for redactor in redactors.values():
        redactor.discard()  # fail closed: held-back text may be the start of PII
    record.outcome, record.error_code, record.detail = "upstream_error", code, detail
