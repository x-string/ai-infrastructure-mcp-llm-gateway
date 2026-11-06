"""Task 4: configuration from the environment, and the mock provider's modes."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from ai_gateway.model_router.mock_provider import LEAKY_ERROR, MODES, create_app
from ai_gateway.model_router.settings import DEFAULT_PROVIDERS, RouterSettings

# --- settings -------------------------------------------------------------------------------------


def test_defaults_are_the_brief_and_bind_to_localhost() -> None:
    settings = RouterSettings.from_env({})
    assert settings.providers == DEFAULT_PROVIDERS
    assert [p.deadline_seconds for p in settings.providers] == [3.0, 15.0]
    assert (settings.host, settings.port, settings.db_path) == ("127.0.0.1", 8300, "t4-router.db")
    assert (settings.default_max_tokens, settings.max_output_tokens) == (1024, 4096)


def test_everything_can_be_configured() -> None:
    settings = RouterSettings.from_env(
        {
            "T4_DB_PATH": "/tmp/x.db",
            "T4_PRIMARY_URL": "https://api.example.com/v1/chat/completions",
            "T4_PRIMARY_API_KEY": "sk-primary",
            "T4_PRIMARY_MODEL": "gpt-x",
            "T4_PRIMARY_DEADLINE_MS": "250",
            "T4_SECONDARY_URL": "http://127.0.0.1:9999/v1/chat/completions",
            "T4_SECONDARY_DEADLINE_MS": "5000",
            "T4_DEFAULT_MAX_TOKENS": "256",
            "T4_MAX_OUTPUT_TOKENS": "512",
            "T4_PORT": "9000",
        }
    )
    primary, secondary = settings.providers
    assert (primary.url, primary.api_key, primary.model) == (
        "https://api.example.com/v1/chat/completions",
        "sk-primary",
        "gpt-x",
    )
    assert (primary.deadline_seconds, secondary.deadline_seconds) == (0.25, 5.0)
    assert "sk-primary" not in repr(settings)
    assert (settings.db_path, settings.port, settings.default_max_tokens) == (
        "/tmp/x.db",
        9000,
        256,
    )


def test_an_empty_secondary_url_means_no_failover() -> None:
    assert [p.name for p in RouterSettings.from_env({"T4_SECONDARY_URL": ""}).providers] == [
        "primary"
    ]


@pytest.mark.parametrize(
    "env",
    [
        {"T4_PRIMARY_URL": "ftp://x/y"},
        {"T4_PRIMARY_URL": "not a url"},
        {"T4_PRIMARY_DEADLINE_MS": "0"},
        {"T4_PRIMARY_DEADLINE_MS": "-5"},
        {"T4_PRIMARY_DEADLINE_MS": "3s"},
        {"T4_DEFAULT_MAX_TOKENS": "5000"},
        {"T4_MAX_BODY_BYTES": "10"},
        {"T4_PORT": "١٢"},
    ],
)
def test_invalid_configuration_stops_the_router_at_startup(env: dict) -> None:
    with pytest.raises(ValueError):
        RouterSettings.from_env(env)


# --- the mock provider ----------------------------------------------------------------------------

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi there"}], "max_tokens": 5}


def ask(mode: str, **kwargs: object):
    with TestClient(create_app(mode, 0.05, "p", **kwargs)) as client:
        return client.post("/v1/chat/completions", json=BODY)


def test_ok_echoes_the_last_user_message_with_consistent_usage() -> None:
    answer = ask("ok").json()
    assert answer["choices"][0]["message"]["content"] == "[p] hi there"
    usage = answer["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"] > 0


@pytest.mark.parametrize(
    ("mode", "status_code"),
    [("rate_limited", 429), ("error", 500), ("malformed", 200), ("slow", 200), ("slow_drip", 200)],
)
def test_modes(mode: str, status_code: int) -> None:
    response = ask(mode)
    assert response.status_code == status_code
    if mode == "rate_limited":
        assert response.headers["retry-after"] == "1"
    if mode == "error":
        assert LEAKY_ERROR in response.text
    if mode == "malformed":
        with pytest.raises(json.JSONDecodeError):
            response.json()
    if mode in ("slow", "slow_drip"):
        assert response.json()["usage"]["total_tokens"] > 0


def test_no_usage_mode_and_the_control_endpoints() -> None:
    assert "usage" not in ask("no_usage").json()
    with TestClient(create_app("ok", 0, "p", control=True)) as client:
        client.post("/v1/chat/completions", json=BODY)
        assert client.get("/_calls").json() == {"calls": 1, "mode": "ok"}
        assert client.put("/_mode", json={"mode": "error"}).status_code == 200
        assert client.post("/v1/chat/completions", json=BODY).status_code == 500
        assert client.put("/_mode", json={"mode": "nope"}).status_code == 400
    with TestClient(create_app("ok")) as client:
        assert client.get("/_calls").status_code == 404  # control endpoints are opt-in


def test_unknown_modes_are_refused() -> None:
    assert "slow_drip" in MODES
    with pytest.raises(ValueError):
        create_app("chaos")
