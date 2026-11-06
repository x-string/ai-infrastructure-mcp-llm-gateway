"""Configuration for the Task 4 model router, read from environment variables.

| Variable | Default | Meaning |
|---|---|---|
| ``T4_DB_PATH`` | ``t4-router.db`` | SQLite file with tenants, keys and token usage |
| ``T4_PRIMARY_URL`` | ``http://127.0.0.1:9201/v1/chat/completions`` | primary provider |
| ``T4_PRIMARY_API_KEY`` / ``T4_PRIMARY_MODEL`` | unset | its bearer key / model name |
| ``T4_PRIMARY_DEADLINE_MS`` | ``3000`` | total deadline for the primary's whole response |
| ``T4_SECONDARY_URL`` | ``http://127.0.0.1:9202/v1/chat/completions`` | backup provider |
| ``T4_SECONDARY_API_KEY`` / ``T4_SECONDARY_MODEL`` | unset | its bearer key / model name |
| ``T4_SECONDARY_DEADLINE_MS`` | ``15000`` | total deadline for the backup |
| ``T4_DEFAULT_MAX_TOKENS`` | ``1024`` | completion cap when a request sets none |
| ``T4_MAX_OUTPUT_TOKENS`` | ``4096`` | the largest cap a request may ask for |
| ``T4_MAX_BODY_BYTES`` | ``1048576`` | request body limit |
| ``T4_HOST`` / ``T4_PORT`` | ``127.0.0.1`` / ``8300`` | where the router listens |

Set ``T4_SECONDARY_URL`` to an empty string to run with the primary only.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

from ai_gateway.model_router.providers import Provider
from ai_gateway.model_router.storage import DEFAULT_DB_PATH

DEFAULT_PRIMARY_URL: Final = "http://127.0.0.1:9201/v1/chat/completions"
DEFAULT_SECONDARY_URL: Final = "http://127.0.0.1:9202/v1/chat/completions"
DEFAULT_PROVIDERS: Final = (
    Provider("primary", DEFAULT_PRIMARY_URL, 3.0),
    Provider("secondary", DEFAULT_SECONDARY_URL, 15.0),
)


@dataclass(frozen=True)
class RouterSettings:
    db_path: str = DEFAULT_DB_PATH
    providers: tuple[Provider, ...] = DEFAULT_PROVIDERS
    default_max_tokens: int = 1024
    max_output_tokens: int = 4096
    max_body_bytes: int = 1 << 20
    host: str = "127.0.0.1"
    port: int = 8300

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("at least one provider is required")
        for provider in self.providers:
            parts = urlsplit(provider.url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise ValueError(f"the {provider.name} URL must be an http:// or https:// URL")
            if not 0 < provider.deadline_seconds <= 600:
                raise ValueError(f"the {provider.name} deadline must be between 1 ms and 600 s")
        if not 1 <= self.default_max_tokens <= self.max_output_tokens:
            raise ValueError("T4_DEFAULT_MAX_TOKENS must be between 1 and T4_MAX_OUTPUT_TOKENS")
        if self.max_body_bytes < 1024:
            raise ValueError("T4_MAX_BODY_BYTES must be at least 1024")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> RouterSettings:
        env = os.environ if environ is None else environ
        providers = [_provider(env, "primary", DEFAULT_PRIMARY_URL, 3000)]
        if env.get("T4_SECONDARY_URL", DEFAULT_SECONDARY_URL):
            providers.append(_provider(env, "secondary", DEFAULT_SECONDARY_URL, 15000))
        return cls(
            db_path=env.get("T4_DB_PATH") or DEFAULT_DB_PATH,
            providers=tuple(providers),
            default_max_tokens=_integer(env, "T4_DEFAULT_MAX_TOKENS", 1024),
            max_output_tokens=_integer(env, "T4_MAX_OUTPUT_TOKENS", 4096),
            max_body_bytes=_integer(env, "T4_MAX_BODY_BYTES", 1 << 20),
            host=env.get("T4_HOST") or "127.0.0.1",
            port=_integer(env, "T4_PORT", 8300),
        )


def _provider(env: Mapping[str, str], name: str, url: str, deadline_ms: int) -> Provider:
    prefix = f"T4_{name.upper()}_"
    return Provider(
        name=name,
        url=env.get(prefix + "URL") or url,
        deadline_seconds=_integer(env, prefix + "DEADLINE_MS", deadline_ms) / 1000,
        api_key=env.get(prefix + "API_KEY") or None,
        model=env.get(prefix + "MODEL") or None,
    )


def _integer(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if not value:
        return default
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"{name} must be a whole number")
    return int(value)
