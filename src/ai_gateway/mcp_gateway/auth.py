"""Bearer-token authentication for the MCP security gateway.

Tokens are JWTs signed with HS256 and a shared secret of at least 32 bytes. The role is read
from the ``role`` claim. Verification pins the algorithm (no ``none``, no algorithm confusion),
requires ``exp``, ``iat``, ``sub``, ``aud`` and ``role`` (plus ``iss`` when an issuer is
configured), and allows a small clock skew.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import jwt

ALGORITHM: Final = "HS256"
MIN_SECRET_BYTES: Final = 32
MAX_TOKEN_CHARS: Final = 8192
_TOKEN68 = re.compile(r"[A-Za-z0-9\-._~+/]+=*")


class Role(StrEnum):
    ADMIN = "admin"
    VIEWER = "viewer"


@dataclass(frozen=True)
class Principal:
    subject: str
    role: Role


class AuthenticationError(Exception):
    """Authentication failed.

    ``token_present`` decides the 401 challenge. RFC 6750 adds ``error="invalid_token"`` only when
    a bearer token was actually sent.
    """

    def __init__(self, reason: str, *, token_present: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.token_present = token_present


@dataclass(frozen=True)
class TokenSettings:
    secret: str
    audience: str = "mcp-gateway"
    issuer: str | None = None
    leeway_seconds: float = 10.0

    def __post_init__(self) -> None:
        if len(self.secret.encode("utf-8")) < MIN_SECRET_BYTES:
            raise ValueError(f"the JWT secret must be at least {MIN_SECRET_BYTES} bytes")

    @property
    def required_claims(self) -> list[str]:
        claims = ["exp", "iat", "sub", "aud", "role"]
        if self.issuer is not None:
            claims.append("iss")
        return claims


_REASONS: Final[tuple[tuple[type[jwt.PyJWTError], str], ...]] = (
    (jwt.ExpiredSignatureError, "token has expired"),
    (jwt.ImmatureSignatureError, "token is not valid yet"),
    (jwt.InvalidAudienceError, "token audience is not accepted"),
    (jwt.InvalidIssuerError, "token issuer is not accepted"),
    (jwt.InvalidAlgorithmError, "token algorithm is not allowed"),
    (jwt.InvalidSignatureError, "token signature is invalid"),
    (jwt.DecodeError, "token is malformed"),
)


def _describe(exc: jwt.InvalidTokenError) -> str:
    if isinstance(exc, jwt.MissingRequiredClaimError):
        return f"token is missing the {exc.claim!r} claim"
    for error_type, reason in _REASONS:
        if isinstance(exc, error_type):
            return reason
    return "token is invalid"


def _bearer_token(authorization: Sequence[str]) -> str:
    if len(authorization) > 1:
        raise AuthenticationError("multiple Authorization headers", token_present=True)
    if not authorization:
        raise AuthenticationError("bearer token required", token_present=False)
    scheme, _, token = authorization[0].strip().partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise AuthenticationError("bearer token required", token_present=False)
    if len(token) > MAX_TOKEN_CHARS or not _TOKEN68.fullmatch(token):
        raise AuthenticationError("token is malformed", token_present=True)
    return token


class TokenVerifier:
    def __init__(self, settings: TokenSettings) -> None:
        self._settings = settings

    def authenticate(self, authorization: Sequence[str]) -> Principal:
        """Verify the ``Authorization`` header values and return the caller's identity and role."""
        token = _bearer_token(authorization)
        settings = self._settings
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                settings.secret,
                algorithms=[ALGORITHM],
                audience=settings.audience,
                issuer=settings.issuer,
                leeway=settings.leeway_seconds,
                options={"require": settings.required_claims},
            )
        except jwt.InvalidTokenError as exc:
            raise AuthenticationError(_describe(exc), token_present=True) from None

        role = claims["role"]
        if not isinstance(role, str) or role not in {r.value for r in Role}:
            raise AuthenticationError("token role is not recognized", token_present=True)
        subject = claims["sub"]
        if not isinstance(subject, str) or not subject:
            raise AuthenticationError("token subject is invalid", token_present=True)
        return Principal(subject=subject, role=Role(role))


def issue_token(
    settings: TokenSettings,
    subject: str,
    role: Role,
    *,
    ttl_seconds: int = 3600,
    now: float | None = None,
) -> str:
    """Mint a token the verifier accepts.

    For development and tests only. Real tokens come from an identity provider.
    """
    issued_at = int(time.time() if now is None else now)
    claims: dict[str, Any] = {
        "sub": subject,
        "role": role.value,
        "aud": settings.audience,
        "iat": issued_at,
        "exp": issued_at + ttl_seconds,
    }
    if settings.issuer is not None:
        claims["iss"] = settings.issuer
    return jwt.encode(claims, settings.secret, algorithm=ALGORITHM)
