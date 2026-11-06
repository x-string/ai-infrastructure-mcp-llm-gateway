"""Task 3: configuration from environment variables, refused at startup when invalid."""

from __future__ import annotations

import pytest

from ai_gateway.llm_gateway.redaction import RedactionRules
from ai_gateway.llm_gateway.settings import DEFAULT_UPSTREAM_URL, GatewaySettings


def test_defaults_point_at_the_local_mock_provider() -> None:
    settings = GatewaySettings.from_env({})
    assert settings.upstream.url == DEFAULT_UPSTREAM_URL
    assert settings.upstream.api_key is None
    assert settings.rules == RedactionRules(bare_ssn=True, card_luhn=False)
    assert (settings.host, settings.port, settings.max_streams) == ("127.0.0.1", 8200, 100)
    assert settings.upstream.max_connections == settings.max_streams


def test_every_variable_is_read() -> None:
    settings = GatewaySettings.from_env(
        {
            "T3_UPSTREAM_URL": "https://api.example.com/v1/chat/completions",
            "T3_UPSTREAM_API_KEY": "sk-example",
            "T3_CARD_LUHN": "1",
            "T3_BARE_SSN": "false",
            "T3_MAX_STREAMS": "7",
            "T3_MAX_STREAM_SECONDS": "30",
            "T3_FIRST_EVENT_TIMEOUT": "5",
            "T3_READ_TIMEOUT": "9.5",
            "T3_HOST": "0.0.0.0",
            "T3_PORT": "9000",
        }
    )
    assert settings.upstream.url == "https://api.example.com/v1/chat/completions"
    assert settings.upstream.api_key == "sk-example"
    assert settings.rules == RedactionRules(bare_ssn=False, card_luhn=True)
    assert (settings.max_streams, settings.upstream.max_connections) == (7, 7)
    assert settings.max_stream_seconds == 30
    assert (settings.upstream.first_event_timeout, settings.upstream.read_timeout) == (5, 9.5)
    assert (settings.host, settings.port) == ("0.0.0.0", 9000)
    assert "sk-example" not in repr(settings)


@pytest.mark.parametrize(
    "env",
    [
        {"T3_UPSTREAM_URL": "ftp://example.com"},
        {"T3_CARD_LUHN": "maybe"},
        {"T3_MAX_STREAMS": "0"},
        {"T3_MAX_STREAMS": "many"},
        {"T3_MAX_STREAM_SECONDS": "-1"},
        {"T3_FIRST_EVENT_TIMEOUT": "soon"},
        {"T3_READ_TIMEOUT": "inf"},
        {"T3_PORT": "http"},
    ],
)
def test_invalid_values_are_refused_at_startup(env: dict) -> None:
    with pytest.raises(ValueError):
        GatewaySettings.from_env(env)
