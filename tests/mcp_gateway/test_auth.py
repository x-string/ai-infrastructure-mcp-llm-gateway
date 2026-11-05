"""Task 2: bearer-token authentication (signed JWT, HS256 only)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import warnings
from collections.abc import Callable
from typing import Any

import jwt
import pytest
from jwt.warnings import InsecureKeyLengthWarning

from ai_gateway.mcp_gateway.auth import (
    AuthenticationError,
    Principal,
    Role,
    TokenSettings,
    TokenVerifier,
    issue_token,
)

SECRET = "gateway-test-secret-0123456789abcdef"
SETTINGS = TokenSettings(secret=SECRET, audience="mcp-gateway")
VERIFIER = TokenVerifier(SETTINGS)
DROP = object()


def now() -> int:
    # Read the clock when a token is built, not at import: the verifier checks the time claims
    # against the clock when the test runs, which can be long after collection.
    return int(time.time())


def claims(**overrides: Any) -> dict[str, Any]:
    issued_at = now()
    base: dict[str, Any] = {
        "sub": "alice",
        "role": "viewer",
        "aud": "mcp-gateway",
        "iat": issued_at,
        "exp": issued_at + 300,
    }
    base.update(overrides)
    return {key: value for key, value in base.items() if value is not DROP}


def sign(payload: dict[str, Any], secret: str = SECRET, algorithm: str = "HS256") -> str:
    return jwt.encode(payload, secret, algorithm=algorithm)


def sign_hs512(payload: dict[str, Any]) -> str:
    """HS512 with the gateway's (HS256-sized) secret: a token signed with the wrong algorithm."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InsecureKeyLengthWarning)
        return sign(payload, algorithm="HS512")


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def forged(header: dict[str, Any], payload: dict[str, Any], *, signed: bool) -> str:
    """Hand-built token, for algorithm-confusion cases the JWT library refuses to produce."""
    signing_input = f"{b64url(json.dumps(header).encode())}.{b64url(json.dumps(payload).encode())}"
    signature = hmac.new(SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{b64url(signature) if signed else ''}"


def bearer(token: str) -> list[str]:
    return [f"Bearer {token}"]


def failure(headers: list[str], verifier: TokenVerifier = VERIFIER) -> AuthenticationError:
    with pytest.raises(AuthenticationError) as excinfo:
        verifier.authenticate(headers)
    return excinfo.value


# --- configuration -------------------------------------------------------------------------------


def test_short_secrets_are_refused_at_startup() -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        TokenSettings(secret="too-short")


# --- valid tokens --------------------------------------------------------------------------------


@pytest.mark.parametrize("role", [Role.ADMIN, Role.VIEWER])
def test_issued_tokens_authenticate_with_their_role(role: Role) -> None:
    token = issue_token(SETTINGS, "alice", role)
    assert VERIFIER.authenticate(bearer(token)) == Principal(subject="alice", role=role)


def test_scheme_is_case_insensitive() -> None:
    assert VERIFIER.authenticate([f"bearer {sign(claims())}"]).role is Role.VIEWER


def test_small_clock_skew_is_tolerated() -> None:
    token = sign(claims(exp=now() - 5))  # within the 10-second leeway
    assert VERIFIER.authenticate(bearer(token)).subject == "alice"


def test_issuer_is_checked_when_configured() -> None:
    verifier = TokenVerifier(TokenSettings(secret=SECRET, audience="mcp-gateway", issuer="idp"))
    assert verifier.authenticate(bearer(sign(claims(iss="idp")))).subject == "alice"
    assert failure(bearer(sign(claims(iss="evil"))), verifier).token_present is True
    assert failure(bearer(sign(claims())), verifier).token_present is True  # iss missing


# --- no usable credentials: 401 challenge without an error code (RFC 6750 3.1) -------------------


@pytest.mark.parametrize(
    "headers",
    [[], ["Basic YWxpY2U6c2VjcmV0"], ["Bearer"], ["Bearer    "], ["Token abc"]],
    ids=["missing", "basic-scheme", "bearer-without-token", "bearer-blank", "other-scheme"],
)
def test_missing_credentials_get_a_plain_challenge(headers: list[str]) -> None:
    assert failure(headers).token_present is False


# --- invalid tokens: 401 with error="invalid_token" ----------------------------------------------

# Each case builds its headers when the test runs, so time-based claims stay relative to the clock.
INVALID_TOKENS = [
    pytest.param(lambda: ["Bearer a.b.c", "Bearer a.b.c"], id="two-authorization-headers"),
    pytest.param(lambda: bearer("not-a-jwt"), id="garbage"),
    pytest.param(lambda: ["Bearer abc def"], id="space-in-token"),
    pytest.param(lambda: bearer("a" * 9000), id="oversized"),
    pytest.param(
        lambda: bearer(sign(claims(), secret="another-secret-0123456789abcdefgh")), id="bad-sig"
    ),
    pytest.param(lambda: bearer(sign(claims(exp=now() - 60))), id="expired"),
    pytest.param(lambda: bearer(sign(claims(nbf=now() + 60))), id="not-yet-valid"),
    pytest.param(lambda: bearer(sign(claims(iat=now() + 60))), id="issued-in-future"),
    pytest.param(lambda: bearer(sign(claims(exp=DROP))), id="missing-exp"),
    pytest.param(lambda: bearer(sign(claims(iat=DROP))), id="missing-iat"),
    pytest.param(lambda: bearer(sign(claims(sub=DROP))), id="missing-sub"),
    pytest.param(lambda: bearer(sign(claims(aud=DROP))), id="missing-aud"),
    pytest.param(lambda: bearer(sign(claims(role=DROP))), id="missing-role"),
    pytest.param(lambda: bearer(sign(claims(aud="another-service"))), id="wrong-audience"),
    pytest.param(lambda: bearer(sign(claims(sub=123))), id="non-string-subject"),
    pytest.param(lambda: bearer(sign(claims(role="root"))), id="unknown-role"),
    pytest.param(lambda: bearer(sign(claims(role="ADMIN"))), id="role-wrong-case"),
    pytest.param(lambda: bearer(sign(claims(role=["admin"]))), id="role-list"),
    pytest.param(lambda: bearer(sign(claims(role=1))), id="role-number"),
    pytest.param(lambda: bearer(sign_hs512(claims())), id="hs512"),
    pytest.param(
        lambda: bearer(forged({"alg": "none"}, claims(role="admin"), signed=False)), id="alg-none"
    ),
    pytest.param(
        lambda: bearer(forged({"alg": "RS256"}, claims(role="admin"), signed=True)), id="rs256"
    ),
]


@pytest.mark.parametrize("make_headers", INVALID_TOKENS)
def test_invalid_tokens_are_rejected(make_headers: Callable[[], list[str]]) -> None:
    error = failure(make_headers())
    assert error.token_present is True
    assert SECRET not in error.reason
