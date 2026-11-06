"""Model providers for Task 4: one attempt against an OpenAI-compatible endpoint, under a deadline.

An attempt's deadline is total. Connecting, sending, waiting for the headers and reading the whole
body must all finish within it (3,000 ms for the primary by default). httpx2's per-read timeout is
off, so a provider that drips its body slowly cannot keep resetting a timer. Connecting has its own
shorter timeout, so an unreachable provider is recognised as one that never received the request,
which matters for charging. The body is parsed and validated only after the deadline block.

Every attempt ends in an ``Outcome``. The router fails over on the ones that another provider
might not share (rate limited, timed out, unreachable, a 5xx, a broken response) but not on a
request the provider rejected, which the next one would reject too.

A successful response is rebuilt as a plain OpenAI ``chat.completion`` rather than passed through.
Logs and errors carry the status and the kind of failure, never a body, which can echo the prompt.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Mapping
from contextlib import aclosing
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, TypeGuard

import anyio
import httpx2

CONNECT_TIMEOUT_SECONDS: Final = 1.0
MAX_RESPONSE_BYTES: Final = 4 << 20
USER_AGENT: Final = "ai-gateway-model-router"
_MAX_TOKENS: Final = 10**9
_FINISH_REASONS: Final = {
    "stop": "stop",
    "eos": "stop",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "length": "length",
    "max_tokens": "length",
    "content_filter": "content_filter",
}


@dataclass(frozen=True, slots=True)
class Provider:
    name: str
    url: str  # the full chat completions URL
    deadline_seconds: float
    api_key: str | None = field(default=None, repr=False)
    model: str | None = None  # sent instead of the client's model name, when set


class Outcome(StrEnum):
    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"  # 429
    TIMEOUT = "timeout"  # the total deadline passed
    UNREACHABLE = "unreachable"  # never connected: the provider did not see the request
    SERVER_ERROR = "server_error"  # 5xx
    BROKEN = "broken"  # the connection failed after sending, or the response is invalid
    REJECTED = "rejected"  # 400, 413, 422: the request itself is the problem
    MISCONFIGURED = "misconfigured"  # 401, 403, 404 and other 4xx: the gateway's setup is


FAILS_OVER: Final = frozenset(
    {Outcome.RATE_LIMITED, Outcome.TIMEOUT, Outcome.UNREACHABLE, Outcome.SERVER_ERROR,
     Outcome.BROKEN}
)  # fmt: skip
# The provider may have processed the prompt, so the attempt is charged its prompt estimate.
CHARGES_PROMPT: Final = frozenset({Outcome.TIMEOUT, Outcome.SERVER_ERROR, Outcome.BROKEN})


@dataclass(frozen=True, slots=True)
class Completion:
    body: dict[str, object]  # the rebuilt chat.completion
    total_tokens: int | None  # the provider's count; None when missing or not believable


@dataclass(frozen=True, slots=True)
class Attempt:
    provider: str
    outcome: Outcome
    status: int | None
    elapsed_ms: int
    detail: str  # for logs: never a body
    completion: Completion | None = None


class _TooLarge(Exception):
    pass


async def attempt(
    client: httpx2.AsyncClient, provider: Provider, payload: Mapping[str, object]
) -> Attempt:
    """Send ``payload`` to ``provider`` and classify what happened, within its deadline."""
    body = dict(payload)
    if provider.model:
        body["model"] = provider.model
    deadline = provider.deadline_seconds
    request = client.build_request(
        "POST",
        provider.url,
        content=json.dumps(body, ensure_ascii=False, allow_nan=False).encode(),
        headers=_headers(provider),
        timeout=httpx2.Timeout(
            connect=min(CONNECT_TIMEOUT_SECONDS, deadline / 2), read=None, write=None, pool=deadline
        ),
    )
    started = time.monotonic()
    status: int | None = None

    def done(outcome: Outcome, detail: str, completion: Completion | None = None) -> Attempt:
        elapsed = math.ceil((time.monotonic() - started) * 1000)
        return Attempt(provider.name, outcome, status, elapsed, detail, completion)

    try:
        with anyio.fail_after(deadline):
            response = await client.send(request, stream=True)
            status = response.status_code
            try:
                raw = await _read_body(response)
            finally:
                with anyio.CancelScope(shield=True):
                    await response.aclose()
    except TimeoutError:
        return done(Outcome.TIMEOUT, f"no complete response within {deadline:g} s")
    except (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.UnsupportedProtocol) as exc:
        return done(Outcome.UNREACHABLE, type(exc).__name__)
    except httpx2.PoolTimeout:
        return done(Outcome.UNREACHABLE, "connection pool full")
    except httpx2.RequestError as exc:
        return done(Outcome.BROKEN, type(exc).__name__)
    except _TooLarge:
        return done(Outcome.BROKEN, f"response over {MAX_RESPONSE_BYTES} bytes")

    if status == 200:
        completion = parse_completion(raw, str(body.get("model", "")))
        if completion is None:
            return done(Outcome.BROKEN, "HTTP 200 with an invalid completion")
        return done(Outcome.SUCCESS, "HTTP 200", completion)
    if status == 429:
        return done(Outcome.RATE_LIMITED, "HTTP 429")
    if status in (400, 413, 422):
        return done(Outcome.REJECTED, f"HTTP {status}")
    if status >= 500:
        return done(Outcome.SERVER_ERROR, f"HTTP {status}")
    if status >= 400:
        return done(Outcome.MISCONFIGURED, f"HTTP {status}")
    return done(Outcome.BROKEN, f"HTTP {status}")


def parse_completion(raw: bytes, model: str) -> Completion | None:
    """Rebuild a ``chat.completion`` from the provider's body, or None if it is not one."""
    try:
        data = json.loads(raw, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
        return None
    content, refusal = message.get("content"), message.get("refusal")
    if not isinstance(content, str | None) or not isinstance(refusal, str | None):
        return None
    if content is None and refusal is None:
        return None
    finish = _FINISH_REASONS.get(str(choices[0].get("finish_reason")))
    if finish is None:
        return None
    rebuilt_message: dict[str, object] = {"role": "assistant", "content": content}
    if refusal is not None:
        rebuilt_message["refusal"] = refusal
    usage = _usage(data.get("usage"))
    body: dict[str, object] = {
        "id": _identifier(data.get("id")) or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": data["created"] if _is_timestamp(data.get("created")) else int(time.time()),
        "model": _identifier(data.get("model")) or model,
        "choices": [{"index": 0, "message": rebuilt_message, "finish_reason": finish}],
    }
    if usage is not None:
        body["usage"] = usage
    return Completion(body, None if usage is None else usage["total_tokens"])


async def _read_body(response: httpx2.Response) -> bytes:
    body = bytearray()
    async with aclosing(response.aiter_bytes()) as chunks:
        async for chunk in chunks:
            body += chunk
            if len(body) > MAX_RESPONSE_BYTES:
                raise _TooLarge
    return bytes(body)


def _headers(provider: Provider) -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        "accept": "application/json",
        "accept-encoding": "identity",
        "user-agent": USER_AGENT,
    }
    if provider.api_key:
        headers["authorization"] = f"Bearer {provider.api_key}"
    return headers


def _usage(value: object) -> dict[str, int] | None:
    """The provider's token counts, if they are believable."""
    if not isinstance(value, dict):
        return None
    prompt, completion, total = (
        value.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    if _is_count(prompt) and _is_count(completion) and _is_count(total):
        if total >= prompt + completion:
            return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
    return None


def _is_count(value: object) -> TypeGuard[int]:
    return type(value) is int and 0 <= value < _MAX_TOKENS


def _is_timestamp(value: object) -> TypeGuard[int]:
    return type(value) is int and 0 < value < 10**11  # Unix seconds


def _identifier(value: object) -> str | None:
    if isinstance(value, str) and 0 < len(value) <= 128 and value.isprintable():
        return value
    return None


def _reject_constant(name: str) -> object:
    raise ValueError(f"{name} is not JSON")
