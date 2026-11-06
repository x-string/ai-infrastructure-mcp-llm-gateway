"""Upstream ``chat.completion.chunk`` events, normalized into what the gateway relays (Task 3).

The gateway never forwards upstream chunks. Every chunk it sends is rebuilt from these events, so
a field it does not understand can never reach the client. The rules are an allowlist, written
for OpenAI and the OpenAI-compatible providers whose streams the tests replay:

* ``content`` and ``refusal`` are the only text channels. ``content`` may also be a list of typed
  parts (Mistral): text parts are relayed; thinking and reference parts carry nothing to relay.
* Reasoning fields (``reasoning``, ``reasoning_content`` and similar) are dropped, never relayed.
* ``tool_calls``, ``function_call``, ``audio`` and ``images`` are refused: the request allowlist
  forbids them, and their contents could not be redacted as text.
* Known metadata (``role``, ``annotations``, ...) is ignored. Any other non-empty delta field is
  refused, since it could be a text channel the gateway does not know about.
* An error object (top-level ``error``, ``object: "error"``, Groq's ``x_groq.error``), an
  ``event: error`` or an error finish reason (``abort``, ...) means the generation failed.
* Finish reasons are mapped onto OpenAI's ``stop``, ``length`` and ``content_filter``. A later
  chunk may upgrade one (Azure reports a content filter after ``stop``), so the result is only
  final at the end. Text after a finish reason, or an unknown finish reason, is refused.
* A stream succeeds only once a finish reason arrived. ``[DONE]`` alone does not prove it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, NoReturn

from ai_gateway.llm_gateway.sse import ServerSentEvent

FinishReason = Literal["stop", "length", "content_filter"]

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "eos": "stop",  # Together
    "end_turn": "stop",
    "stop_sequence": "stop",
    "repetition": "stop",  # vLLM: repetitive output detected, generation stopped
    "length": "length",
    "max_tokens": "length",
    "model_length": "length",
    "content_filter": "content_filter",
}
_FAILED_FINISH_REASONS = frozenset({"error", "abort", "aborted", "insufficient_system_resource"})
_SEVERITY: dict[FinishReason, int] = {"stop": 0, "length": 1, "content_filter": 2}

_DROPPED = frozenset(
    {"reasoning", "reasoning_content", "reasoning_details", "thinking_blocks", "reasoning_items"}
)
_REFUSED = frozenset({"tool_calls", "function_call", "audio", "images"})
_IGNORED = frozenset(
    {
        "role",
        "annotations",
        "citations",
        "executed_tools",
        "index",
        "internal_content",
        "metadata",
        "provider_specific_fields",
        "token_id",
        "tool_call_id",
    }
)
_SAFE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")
_MAX_TOKENS = 10**9


class Channel(StrEnum):
    CONTENT = "content"
    REFUSAL = "refusal"


@dataclass(frozen=True, slots=True)
class TextDelta:
    channel: Channel
    text: str


@dataclass(frozen=True, slots=True)
class KeepAlive:
    """The upstream is alive but sent nothing to relay (a comment or an empty event)."""


KEEP_ALIVE = KeepAlive()


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class StreamEnd:
    finish_reason: FinishReason
    usage: Usage | None


class StreamError(Exception):
    """The stream cannot be relayed. ``detail`` is written by the gateway and safe to log."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ReportedError(StreamError):
    """The upstream reported that the generation failed."""


class ProtocolViolation(StreamError):
    """The upstream sent something the gateway cannot vouch for."""


class StreamNormalizer:
    """Turns the upstream's events, one by one, into text deltas and a final result."""

    __slots__ = ("_finish", "_usage", "_done")

    def __init__(self) -> None:
        self._finish: FinishReason | None = None
        self._usage: Usage | None = None
        self._done = False

    @property
    def done(self) -> bool:
        """``[DONE]`` arrived: nothing after it is read."""
        return self._done

    def accept(self, event: ServerSentEvent) -> list[TextDelta | KeepAlive]:
        if self._done:
            raise ProtocolViolation("event after [DONE]")
        if event.event == "error":
            raise ReportedError("error event")
        if event.event != "message":
            raise ProtocolViolation(f"unexpected event type {_safe(event.event)}")
        data = event.data.strip()
        if not data:
            return [KEEP_ALIVE]
        if data == "[DONE]":
            self._done = True
            return []
        return self._chunk(_parse_json(event.data))

    def end(self) -> StreamEnd:
        """The stream is over: succeed only if a finish reason arrived."""
        if self._finish is None:
            raise ProtocolViolation("stream ended before a finish reason")
        return StreamEnd(self._finish, self._usage)

    def _chunk(self, payload: object) -> list[TextDelta | KeepAlive]:
        if not isinstance(payload, dict):
            raise ProtocolViolation("data is not a JSON object")
        if payload.get("error") is not None:
            raise ReportedError("error object")
        if payload.get("object") == "error" or payload.get("type") == "error":
            raise ReportedError("error object")
        if payload.get("object") not in (None, "", "chat.completion.chunk"):
            raise ProtocolViolation(f"unexpected object {_safe(payload.get('object'))}")
        groq = payload.get("x_groq")
        if isinstance(groq, dict):
            if groq.get("error") is not None:
                raise ReportedError("x_groq error")
            self._take_usage(groq.get("usage"))
        self._take_usage(payload.get("usage"))
        choices = payload.get("choices")
        if choices is None:
            return []
        if not isinstance(choices, list):
            raise ProtocolViolation("choices is not a list")
        out: list[TextDelta | KeepAlive] = []
        for choice in choices:
            out += self._choice(choice)
        return out

    def _choice(self, choice: object) -> list[TextDelta]:
        if not isinstance(choice, dict):
            raise ProtocolViolation("choice is not an object")
        index = choice.get("index", 0)
        if type(index) is not int or index != 0:
            raise ProtocolViolation("unexpected choice index")  # the request allows n=1 only
        self._take_usage(choice.get("usage"))
        texts: list[TextDelta] = []
        delta = choice.get("delta")
        if delta is not None:
            if not isinstance(delta, dict):
                raise ProtocolViolation("delta is not an object")
            texts = _delta_texts(delta)
        if texts and self._finish is not None:
            raise ProtocolViolation("text after a finish reason")
        finish = choice.get("finish_reason")
        if finish is not None:
            self._take_finish(finish)
        return texts

    def _take_finish(self, value: object) -> None:
        if not isinstance(value, str):
            raise ProtocolViolation("finish reason is not a string")
        if value in _FAILED_FINISH_REASONS:
            raise ReportedError(f"finish reason {value}")
        if value in ("tool_calls", "function_call"):
            raise ProtocolViolation(f"finish reason {value} without tools")
        mapped = _FINISH_REASONS.get(value)
        if mapped is None:
            raise ProtocolViolation(f"unknown finish reason {_safe(value)}")
        if self._finish is None or _SEVERITY[mapped] > _SEVERITY[self._finish]:
            self._finish = mapped

    def _take_usage(self, value: object) -> None:
        if not isinstance(value, dict):
            return
        counts = [value.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if all(type(count) is int and 0 <= count < _MAX_TOKENS for count in counts):
            self._usage = Usage(*counts)


def _delta_texts(delta: dict[str, object]) -> list[TextDelta]:
    texts: list[TextDelta] = []
    for key, value in delta.items():
        if key == "content":
            texts += _content_texts(value)
        elif key == "refusal":
            if value is not None and value != "":
                texts.append(TextDelta(Channel.REFUSAL, _text(value, "refusal")))
        elif key in _REFUSED:
            if not _empty(value):
                raise ProtocolViolation(f"unsupported output channel {key}")
        elif key in _DROPPED or key in _IGNORED:
            continue
        elif not _empty(value):
            raise ProtocolViolation(f"unknown delta field {_safe(key)}")
    return texts


def _content_texts(value: object) -> list[TextDelta]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [TextDelta(Channel.CONTENT, _text(value, "content"))]
    if not isinstance(value, list):
        raise ProtocolViolation("content is not a string or a list")
    texts: list[TextDelta] = []
    for part in value:
        if not isinstance(part, dict):
            raise ProtocolViolation("content part is not an object")
        kind = part.get("type")
        if kind == "text":
            text = _text(part.get("text"), "content part")
            if text:
                texts.append(TextDelta(Channel.CONTENT, text))
        elif kind not in ("thinking", "reference"):
            raise ProtocolViolation(f"unsupported content part {_safe(kind)}")
    return texts


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ProtocolViolation(f"{field} is not a string")
    try:
        value.encode()
    except UnicodeEncodeError:  # a lone surrogate, which JSON "\ud800" escapes can produce
        raise ProtocolViolation(f"{field} is not valid Unicode") from None
    return value


def _empty(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _parse_json(data: str) -> object:
    try:
        return json.loads(data, parse_constant=_reject_constant)
    except (ValueError, RecursionError):  # also the int-digit limit and NaN/Infinity
        raise ProtocolViolation("invalid JSON") from None


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is not JSON")


def _safe(value: object) -> str:
    """A name from the upstream, quoted for logs, unless it could carry data."""
    if isinstance(value, str) and _SAFE_NAME.fullmatch(value):
        return repr(value)
    return "<redacted>"
