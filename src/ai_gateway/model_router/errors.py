"""The one error payload every Task 4 response uses, and the errors the router raises.

Every error, from a bad request to an unexpected exception, has the same shape::

    {"error": {"type": "...", "code": "...", "message": "...", "request_id": "req_..."}}

``message`` is always written by the gateway. Upstream bodies, prompts, exception text and stack
traces never appear in a response.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

INVALID_REQUEST: Final = "invalid_request_error"
AUTHENTICATION: Final = "authentication_error"
RATE_LIMIT: Final = "rate_limit_error"
UPSTREAM: Final = "upstream_error"
SERVER: Final = "server_error"


class GatewayError(Exception):
    """An error as the client sees it: status, type, stable code, safe message, extra headers."""

    def __init__(
        self,
        status: int,
        error_type: str,
        code: str,
        message: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.code = code
        self.message = message
        self.headers = dict(headers or {})


def error_body(error_type: str, code: str, message: str, request_id: str) -> dict[str, object]:
    return {
        "error": {"type": error_type, "code": code, "message": message, "request_id": request_id}
    }


# Errors for HTTP statuses that Starlette raises itself (routing, methods).
HTTP_STATUS_ERRORS: Final[dict[int, tuple[str, str, str]]] = {
    400: (INVALID_REQUEST, "invalid_request", "The request is invalid."),
    401: (AUTHENTICATION, "invalid_api_key", "The API key is missing or invalid."),
    404: (INVALID_REQUEST, "not_found", "There is nothing at this path."),
    405: (INVALID_REQUEST, "method_not_allowed", "This method is not allowed on this path."),
    413: (INVALID_REQUEST, "body_too_large", "The request body is too large."),
    415: (INVALID_REQUEST, "unsupported_media_type", "The request body must be application/json."),
    422: (INVALID_REQUEST, "invalid_request", "The request is invalid."),
    429: (RATE_LIMIT, "rate_limited", "Too many requests."),
    500: (SERVER, "internal_error", "The gateway failed to handle the request."),
    502: (UPSTREAM, "upstream_error", "The model provider failed."),
    503: (SERVER, "unavailable", "The gateway is temporarily unavailable."),
    504: (UPSTREAM, "upstream_timeout", "The model provider did not respond in time."),
}


def for_status(status: int, *, headers: Mapping[str, str] | None = None) -> GatewayError:
    error_type, code, message = HTTP_STATUS_ERRORS.get(
        status, (SERVER if status >= 500 else INVALID_REQUEST, "http_error", "The request failed.")
    )
    return GatewayError(status, error_type, code, message, headers=headers)


def invalid_request(message: str, code: str = "invalid_request") -> GatewayError:
    return GatewayError(400, INVALID_REQUEST, code, message)


def body_too_large(limit: int) -> GatewayError:
    return GatewayError(
        413, INVALID_REQUEST, "body_too_large", f"The request body exceeds {limit} bytes."
    )


def unsupported_media_type() -> GatewayError:
    return GatewayError(
        415,
        INVALID_REQUEST,
        "unsupported_media_type",
        "The request body must be application/json.",
    )


def missing_api_key() -> GatewayError:
    return GatewayError(
        401,
        AUTHENTICATION,
        "missing_api_key",
        "Send the tenant API key as Authorization: Bearer <key>.",
        headers={"WWW-Authenticate": 'Bearer realm="model-router"'},
    )


def invalid_api_key() -> GatewayError:
    return GatewayError(
        401,
        AUTHENTICATION,
        "invalid_api_key",
        "The API key is invalid.",
        headers={"WWW-Authenticate": 'Bearer realm="model-router", error="invalid_token"'},
    )


def request_too_large(cost: int, limit: int) -> GatewayError:
    return GatewayError(
        400,
        INVALID_REQUEST,
        "request_too_large",
        f"The request may use up to {cost} tokens, more than the tenant limit of {limit} "
        "tokens per minute. Send a shorter prompt or a lower max_tokens.",
    )


def rate_limited(retry_after: int, limit: int, remaining: int) -> GatewayError:
    return GatewayError(
        429,
        RATE_LIMIT,
        "tenant_token_limit_exceeded",
        f"The tenant limit of {limit} tokens per minute is reached; retry after {retry_after} s.",
        headers={
            "Retry-After": str(retry_after),
            **rate_limit_headers(limit, remaining),
        },
    )


def rate_limit_headers(limit: int, remaining: int) -> dict[str, str]:
    return {
        "X-RateLimit-Limit-Tokens": str(limit),
        "X-RateLimit-Remaining-Tokens": str(max(remaining, 0)),
    }


def limiter_unavailable() -> GatewayError:
    return GatewayError(
        503,
        SERVER,
        "limiter_unavailable",
        "The rate limiter is temporarily unavailable; the request was not sent.",
        headers={"Retry-After": "1"},
    )


def upstream_timeout() -> GatewayError:
    return GatewayError(
        504, UPSTREAM, "upstream_timeout", "The model providers did not respond in time."
    )


def upstream_unavailable() -> GatewayError:
    return GatewayError(
        503,
        UPSTREAM,
        "upstream_unavailable",
        "The model providers are rate limiting requests. Try again later.",
    )


def upstream_failed() -> GatewayError:
    return GatewayError(502, UPSTREAM, "upstream_error", "The model providers failed to answer.")


def upstream_rejected() -> GatewayError:
    return GatewayError(
        400,
        INVALID_REQUEST,
        "upstream_rejected_request",
        "The model provider rejected the request.",
    )


def upstream_misconfigured() -> GatewayError:
    return GatewayError(
        502,
        UPSTREAM,
        "upstream_configuration_error",
        "The gateway is not configured correctly for the model provider.",
    )


def internal_error() -> GatewayError:
    return for_status(500)
