"""Configuration for the streaming PII-redaction gateway (Task 3), read from environment variables.

| Variable | Default | Meaning |
|---|---|---|
| ``T3_UPSTREAM_URL`` | ``http://127.0.0.1:9100/v1/chat/completions`` | the provider's URL |
| ``T3_UPSTREAM_API_KEY`` | unset | the provider's key (``Authorization: Bearer``), never logged |
| ``T3_CARD_LUHN`` | ``0`` | ``1``: redact only cards that start with 2-6 and pass Luhn |
| ``T3_BARE_SSN`` | ``1`` | ``0``: leave bare nine-digit numbers alone |
| ``T3_MAX_STREAMS`` | ``100`` | concurrent streams; more are answered with 503 |
| ``T3_MAX_STREAM_SECONDS`` | ``600`` | a stream still running after this ends with an error |
| ``T3_FIRST_EVENT_TIMEOUT`` | ``60`` | seconds from the request to the provider's first event |
| ``T3_READ_TIMEOUT`` | ``120`` | the longest silence allowed from the provider, in seconds |
| ``T3_HOST`` / ``T3_PORT`` | ``127.0.0.1`` / ``8300`` | where the gateway listens |
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

from ai_gateway.llm_gateway.redaction import RedactionRules
from ai_gateway.llm_gateway.upstream import UpstreamConfig

DEFAULT_UPSTREAM_URL = "http://127.0.0.1:9100/v1/chat/completions"
_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}


@dataclass(frozen=True)
class GatewaySettings:
    upstream: UpstreamConfig
    rules: RedactionRules = RedactionRules()
    max_streams: int = 100
    max_stream_seconds: float = 600.0
    max_body_bytes: int = 1 << 20
    keepalive_seconds: float = 15.0  # relay the provider's keep-alives at most this often
    host: str = "127.0.0.1"
    port: int = 8300

    def __post_init__(self) -> None:
        if self.max_streams < 1:
            raise ValueError("T3_MAX_STREAMS must be at least 1")
        if not 0 < self.max_stream_seconds < math.inf:
            raise ValueError("T3_MAX_STREAM_SECONDS must be a positive number")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> GatewaySettings:
        env = os.environ if environ is None else environ
        max_streams = _integer(env, "T3_MAX_STREAMS", 100)
        upstream = UpstreamConfig(
            env.get("T3_UPSTREAM_URL") or DEFAULT_UPSTREAM_URL,
            api_key=env.get("T3_UPSTREAM_API_KEY") or None,
            first_event_timeout=_number(env, "T3_FIRST_EVENT_TIMEOUT", 60.0),
            read_timeout=_number(env, "T3_READ_TIMEOUT", 120.0),
            max_connections=max_streams,
        )
        return cls(
            upstream=upstream,
            rules=RedactionRules(
                bare_ssn=_flag(env, "T3_BARE_SSN", True),
                card_luhn=_flag(env, "T3_CARD_LUHN", False),
            ),
            max_streams=max_streams,
            max_stream_seconds=_number(env, "T3_MAX_STREAM_SECONDS", 600.0),
            host=env.get("T3_HOST") or "127.0.0.1",
            port=_integer(env, "T3_PORT", 8300),
        )


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = (env.get(name) or "").strip().lower()
    if not value:
        return default
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be 1 or 0")


def _number(env: Mapping[str, str], name: str, default: float) -> float:
    value = env.get(name)
    if not value:
        return default
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"{name} must be a number of seconds") from None
    if not 0 < number < math.inf:
        raise ValueError(f"{name} must be a positive number of seconds")
    return number


def _integer(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be a whole number") from None
