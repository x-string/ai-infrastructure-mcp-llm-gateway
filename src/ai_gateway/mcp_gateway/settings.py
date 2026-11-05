"""Configuration for the MCP security gateway, read from environment variables.

| Variable | Default | Meaning |
|---|---|---|
| ``GATEWAY_JWT_SECRET`` | required | HS256 secret, at least 32 bytes |
| ``GATEWAY_JWT_AUDIENCE`` | ``mcp-gateway`` | expected ``aud`` claim |
| ``GATEWAY_JWT_ISSUER`` | unset | expected ``iss`` claim, checked when set |
| ``GATEWAY_DOWNSTREAM_URL`` | ``http://127.0.0.1:8001/mcp`` | the MCP server behind the gateway |
| ``GATEWAY_ALLOWED_ORIGINS`` | none | comma-separated browser origins allowed to call the gateway |
| ``GATEWAY_HOST`` / ``GATEWAY_PORT`` | ``127.0.0.1`` / ``8000`` | where the gateway listens |
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

from ai_gateway.mcp_gateway.auth import TokenSettings

DEFAULT_DOWNSTREAM_URL: Final = "http://127.0.0.1:8001/mcp"


def token_settings_from_env(environ: Mapping[str, str]) -> TokenSettings:
    secret = environ.get("GATEWAY_JWT_SECRET", "")
    if not secret:
        raise ValueError("GATEWAY_JWT_SECRET is required: an HS256 secret of at least 32 bytes")
    return TokenSettings(
        secret=secret,
        audience=environ.get("GATEWAY_JWT_AUDIENCE") or "mcp-gateway",
        issuer=environ.get("GATEWAY_JWT_ISSUER") or None,
    )


@dataclass(frozen=True)
class GatewaySettings:
    token: TokenSettings
    downstream_url: str = DEFAULT_DOWNSTREAM_URL
    allowed_origins: frozenset[str] = frozenset()
    max_body_bytes: int = 1024 * 1024
    connect_timeout_seconds: float = 3.0
    header_timeout_seconds: float = 120.0  # how long a tool may run before its reply starts
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        parts = urlsplit(self.downstream_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("GATEWAY_DOWNSTREAM_URL must be an http:// or https:// URL")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> GatewaySettings:
        env = os.environ if environ is None else environ
        origins = env.get("GATEWAY_ALLOWED_ORIGINS", "")
        return cls(
            token=token_settings_from_env(env),
            downstream_url=env.get("GATEWAY_DOWNSTREAM_URL") or DEFAULT_DOWNSTREAM_URL,
            allowed_origins=frozenset(o.strip() for o in origins.split(",") if o.strip()),
            host=env.get("GATEWAY_HOST") or "127.0.0.1",
            port=int(env.get("GATEWAY_PORT") or 8000),
        )
