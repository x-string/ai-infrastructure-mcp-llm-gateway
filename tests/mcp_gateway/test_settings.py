"""Task 2: gateway configuration and the development token CLI."""

from __future__ import annotations

import pytest

from ai_gateway.mcp_gateway import mint_jwt
from ai_gateway.mcp_gateway.auth import Role, TokenVerifier
from ai_gateway.mcp_gateway.settings import GatewaySettings, token_settings_from_env

SECRET = "gateway-test-secret-0123456789abcdef"


def test_secret_is_required() -> None:
    with pytest.raises(ValueError, match="GATEWAY_JWT_SECRET"):
        GatewaySettings.from_env({})


def test_short_secret_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        GatewaySettings.from_env({"GATEWAY_JWT_SECRET": "short"})


def test_defaults_are_local_and_closed() -> None:
    settings = GatewaySettings.from_env({"GATEWAY_JWT_SECRET": SECRET})
    assert settings.host == "127.0.0.1"
    assert settings.downstream_url == "http://127.0.0.1:8001/mcp"
    assert settings.allowed_origins == frozenset()
    assert settings.token.audience == "mcp-gateway"


def test_environment_overrides() -> None:
    settings = GatewaySettings.from_env(
        {
            "GATEWAY_JWT_SECRET": SECRET,
            "GATEWAY_JWT_AUDIENCE": "tools",
            "GATEWAY_JWT_ISSUER": "https://idp.example",
            "GATEWAY_DOWNSTREAM_URL": "https://mcp.internal:9443/mcp",
            "GATEWAY_ALLOWED_ORIGINS": "https://a.example, https://b.example",
            "GATEWAY_PORT": "9000",
        }
    )
    assert settings.token.issuer == "https://idp.example"
    assert settings.token.audience == "tools"
    assert settings.downstream_url == "https://mcp.internal:9443/mcp"
    assert settings.allowed_origins == {"https://a.example", "https://b.example"}
    assert settings.port == 9000


@pytest.mark.parametrize("url", ["ftp://host/mcp", "not a url", "http:///mcp"])
def test_downstream_url_must_be_http(url: str) -> None:
    with pytest.raises(ValueError, match="GATEWAY_DOWNSTREAM_URL"):
        GatewaySettings.from_env({"GATEWAY_JWT_SECRET": SECRET, "GATEWAY_DOWNSTREAM_URL": url})


def test_mint_jwt_prints_a_token_the_gateway_accepts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GATEWAY_JWT_SECRET", SECRET)
    mint_jwt.main(["--role", "viewer", "--subject", "alice"])
    token = capsys.readouterr().out.strip()

    verifier = TokenVerifier(token_settings_from_env({"GATEWAY_JWT_SECRET": SECRET}))
    principal = verifier.authenticate([f"Bearer {token}"])
    assert (principal.subject, principal.role) == ("alice", Role.VIEWER)
