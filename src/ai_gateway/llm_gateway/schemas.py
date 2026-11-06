"""The client request (Task 3): OpenAI chat completions, restricted to what can be redacted.

Every response must pass through the redactor, so the gateway only accepts requests whose output
is text it can see: ``stream`` must be true (or absent), there is one choice, and parameters that
open another output channel are refused, such as ``tools`` (arguments arrive as JSON text with
escapes), ``logprobs`` (alternative tokens), ``audio`` and ``n`` greater than 1. Any other
unknown parameter is refused too, so a new channel cannot slip through.

The request sent upstream is rebuilt from the validated model, never forwarded raw.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

_SAFE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")

# Parameters that would add an output channel the redactor cannot see.
REFUSED_PARAMETERS = {
    "tools": "tool calls are not supported: their arguments cannot be redacted as text",
    "tool_choice": "tool calls are not supported",
    "parallel_tool_calls": "tool calls are not supported",
    "functions": "function calls are not supported: their arguments cannot be redacted as text",
    "function_call": "function calls are not supported",
    "logprobs": "logprobs are not supported: alternative tokens would bypass redaction",
    "top_logprobs": "logprobs are not supported: alternative tokens would bypass redaction",
    "audio": "audio output is not supported: it cannot be redacted",
    "modalities": "only text output is supported",
    "prediction": "predicted outputs are not supported",
    "web_search_options": "web search is not supported",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class TextPart(_Strict):
    type: Literal["text"]
    text: Annotated[str, Field(max_length=1 << 20)]


class Message(_Strict):
    role: Literal["system", "developer", "user", "assistant"]
    content: (
        Annotated[str, Field(max_length=1 << 20)]
        | Annotated[list[TextPart], Field(min_length=1, max_length=256)]
    )
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")] | None = None


class StreamOptions(_Strict):
    include_usage: bool = False


class ResponseFormat(_Strict):
    type: Literal["text", "json_object"]


_Penalty = Annotated[float, Field(ge=-2, le=2, allow_inf_nan=False)]
_Tokens = Annotated[int, Field(ge=1, le=1_000_000)]


class ChatRequest(_Strict):
    model: Annotated[str, Field(pattern=r"^[\x21-\x7e]{1,256}$")]
    messages: Annotated[list[Message], Field(min_length=1, max_length=1024)]
    stream: bool = True
    stream_options: StreamOptions | None = None
    max_tokens: _Tokens | None = None
    max_completion_tokens: _Tokens | None = None
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] | None = None
    top_p: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    presence_penalty: _Penalty | None = None
    frequency_penalty: _Penalty | None = None
    stop: (
        Annotated[str, Field(max_length=256)]
        | Annotated[list[Annotated[str, Field(max_length=256)]], Field(max_length=4)]
        | None
    ) = None
    seed: int | None = None
    user: Annotated[str, Field(max_length=256)] | None = None
    n: Literal[1] | None = None
    response_format: ResponseFormat | None = None
    reasoning_effort: Annotated[str, Field(pattern=r"^[a-z]{1,16}$")] | None = None
    service_tier: Annotated[str, Field(pattern=r"^[a-z]{1,16}$")] | None = None

    @model_validator(mode="after")
    def _streaming_only(self) -> ChatRequest:
        if not self.stream:
            raise ValueError("stream must be true: every response is streamed through the redactor")
        return self

    @property
    def include_usage(self) -> bool:
        return self.stream_options is not None and self.stream_options.include_usage

    def upstream_payload(self) -> dict[str, object]:
        """The request for the provider, rebuilt from the validated fields."""
        payload = self.model_dump(exclude_none=True, exclude={"stream", "stream_options"})
        payload["stream"] = True
        if self.include_usage:
            payload["stream_options"] = {"include_usage": True}
        return payload


class RequestError(Exception):
    """The request is invalid. ``param`` names the first offending field; nothing echoes input."""

    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.param = param


def parse_request(payload: object) -> ChatRequest:
    """Validate a decoded JSON body, explaining the first problem without echoing any value."""
    if not isinstance(payload, dict):
        raise RequestError("The request body must be a JSON object.")
    for name, reason in REFUSED_PARAMETERS.items():
        if name in payload and not _harmless(name, payload[name]):
            raise RequestError(f"{name}: {reason}", name)
    # What is left of them asks for nothing, so it is dropped rather than forwarded.
    payload = {key: value for key, value in payload.items() if key not in REFUSED_PARAMETERS}
    try:
        return ChatRequest.model_validate(payload)
    except ValidationError as exc:
        error = exc.errors(include_url=False, include_input=False, include_context=False)[0]
        # Unions add the branch tried ("constrained-str", "list[TextPart]") to the location, and
        # their first error only describes that branch, so neither is reported.
        location = [part for part in error["loc"] if not _is_union_tag(part)]
        param = ".".join(_safe(part) for part in location) or None
        if error["type"] == "extra_forbidden":
            raise RequestError(f"{param}: this parameter is not supported", param) from None
        if len(location) < len(error["loc"]):
            message = "invalid value"
        else:
            message = error["msg"].removeprefix("Value error, ")
        raise RequestError(f"{param}: {message}" if param else message, param) from None


def _harmless(name: str, value: object) -> bool:
    """Values of refused parameters that ask for nothing: ``modalities: ["text"]`` and the like."""
    if name == "modalities":
        return value == ["text"]
    if name == "tool_choice":
        return value in (None, "none")
    return value is None or value is False or value == []


def _is_union_tag(part: object) -> bool:
    return isinstance(part, str) and ("[" in part or "-" in part)


def _safe(part: object) -> str:
    text = str(part)
    return text if isinstance(part, int) or _SAFE_NAME.fullmatch(text) else "<redacted>"
