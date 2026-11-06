"""The request the Task 4 router accepts: a strict, non-streaming, text-only chat completion.

Anything outside this model is refused with 400 rather than passed on: tools, images, streaming,
several choices. A request may set ``max_tokens`` or ``max_completion_tokens`` (not both); without
either, the router's default applies. Whichever it is, the cap is always sent to the provider, so
the reservation is a real upper bound.

Validation errors name the fields and the kind of problem, never the input values, and a field
name that could carry data is shown as ``<redacted>``.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_SAFE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
MAX_REPORTED_ERRORS = 5

Penalty = Annotated[float, Field(ge=-2, le=2, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Message(StrictModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: Annotated[str, Field(max_length=500_000)]
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")] | None = None


class ChatCompletionRequest(StrictModel):
    model: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,127}$")]
    messages: Annotated[list[Message], Field(min_length=1, max_length=256)]
    max_tokens: Annotated[int, Field(ge=1)] | None = None
    max_completion_tokens: Annotated[int, Field(ge=1)] | None = None
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] | None = None
    top_p: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] | None = None
    presence_penalty: Penalty | None = None
    frequency_penalty: Penalty | None = None
    stop: (
        Annotated[str, Field(min_length=1, max_length=64)]
        | Annotated[list[Annotated[str, Field(min_length=1, max_length=64)]], Field(max_length=4)]
        | None
    ) = None
    seed: int | None = None
    user: Annotated[str, Field(max_length=256)] | None = None
    n: Literal[1] | None = None
    stream: Literal[False] | None = None

    def upstream_payload(self, max_tokens: int) -> dict[str, object]:
        """The exact body sent to a provider, with the enforced completion cap."""
        cap_field = "max_completion_tokens" if self.max_completion_tokens else "max_tokens"
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [m.model_dump(exclude_none=True) for m in self.messages],
            cap_field: max_tokens,
        }
        for name in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "stop"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.user is not None:
            payload["user"] = self.user
        payload["stream"] = False
        return payload


def describe(error: ValidationError) -> str:
    """``Invalid request: messages.0.role (literal_error); tools (extra_forbidden)``."""
    problems = []
    for detail in error.errors(include_url=False, include_input=False)[:MAX_REPORTED_ERRORS]:
        path = ".".join(_location_part(part) for part in detail["loc"]) or "body"
        problems.append(f"{path} ({detail['type']})")
    return "Invalid request: " + "; ".join(problems) + "."


def _location_part(part: object) -> str:
    if isinstance(part, int):
        return str(part)
    if isinstance(part, str) and _SAFE_NAME.fullmatch(part):
        return part
    return "<redacted>"
