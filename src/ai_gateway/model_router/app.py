"""The Task 4 model router: ``POST /v1/chat/completions`` with token limits and failover.

A request goes through these steps, each of which can answer with the standard error payload:

1. a request id (``req_`` + uuid4, generated here) is attached, and returned as ``X-Request-ID``
   on every response, errors and crashes included;
2. the tenant API key is checked (401) before the body is read;
3. the body must be JSON (415), within the size limit (413), and valid against the strict
   request model (400, naming the fields but never echoing input);
4. the router admits the request against the tenant's token window (429 with ``Retry-After``),
   tries the providers with failover, and settles the tokens.

Success returns the completion with ``X-Gateway-Provider`` and ``X-RateLimit-*`` headers.
Unexpected exceptions become a generic 500. No response ever carries a stack trace, an upstream
body or the prompt.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Final

import anyio
import anyio.to_thread
import httpx2
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ai_gateway.model_router import errors
from ai_gateway.model_router.limiter import TokenLimiter
from ai_gateway.model_router.router import Router
from ai_gateway.model_router.schema import ChatCompletionRequest, describe
from ai_gateway.model_router.settings import RouterSettings
from ai_gateway.model_router.storage import Database, StorageError
from ai_gateway.model_router.tenants import Tenant, find_tenant

PURGE_INTERVAL_SECONDS: Final = 60.0
_KEY: Final = re.compile(r"[A-Za-z0-9._~+/-]{1,256}=*")
_CLIENT_REQUEST_ID: Final = re.compile(r"[\x21-\x7e]{1,128}")

logger = logging.getLogger("ai_gateway.model_router")


class RequestIdMiddleware:
    """Outermost layer: every response, including Starlette's own 500, gets ``X-Request-ID``."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = f"req_{uuid.uuid4().hex}"
        scope.setdefault("state", {})["request_id"] = request_id
        client_id = _header(scope, b"x-request-id")
        if client_id is not None:
            logger.info(
                "request %s carries client id %s",
                request_id,
                client_id if _CLIENT_REQUEST_ID.fullmatch(client_id) else "<invalid>",
            )

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k != b"x-request-id"]
                message = {**message, "headers": [*headers, (b"x-request-id", request_id.encode())]}
            await send(message)

        await self.app(scope, receive, send_with_id)


def create_app(
    settings: RouterSettings,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Callable[[], float] = time.time,
) -> ASGIApp:
    """The router as an ASGI app. ``transport`` and ``clock`` are for tests."""
    database = Database(settings.db_path)
    limiter = TokenLimiter(database, clock=clock)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await anyio.to_thread.run_sync(database.open)
        await limiter.purge()
        client = httpx2.AsyncClient(
            transport=transport,
            timeout=httpx2.Timeout(connect=1.0, read=None, write=None, pool=5.0),
            limits=httpx2.Limits(max_connections=200, max_keepalive_connections=50),
            trust_env=False,
            follow_redirects=False,
        )
        app.state.router = Router(limiter, client, settings.providers)
        try:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(_purge_forever, limiter)
                yield
                tasks.cancel_scope.cancel()
        finally:
            await client.aclose()
            await anyio.to_thread.run_sync(database.close)  # waits for queued settlements

    api = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    api.add_exception_handler(errors.GatewayError, _gateway_error)
    api.add_exception_handler(StarletteHTTPException, _http_error)
    api.add_exception_handler(RequestValidationError, _validation_error)
    api.add_exception_handler(Exception, _unexpected_error)

    @api.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        tenant = await _authenticate(request, database)
        if not _is_json(request.headers.get("content-type")):
            raise errors.unsupported_media_type()
        body = _parse(await _read_body(request, settings.max_body_bytes))
        if body.max_tokens is not None and body.max_completion_tokens is not None:
            raise errors.invalid_request(
                "Set max_tokens or max_completion_tokens, not both.", "conflicting_max_tokens"
            )
        max_tokens = body.max_tokens or body.max_completion_tokens or settings.default_max_tokens
        if max_tokens > settings.max_output_tokens:
            raise errors.invalid_request(
                f"max_tokens may be at most {settings.max_output_tokens}.", "max_tokens_too_large"
            )
        routed = await request.app.state.router.complete(
            tenant=tenant,
            request_id=request.state.request_id,
            payload=body.upstream_payload(max_tokens),
            max_tokens=max_tokens,
        )
        headers = {
            "X-Gateway-Provider": routed.provider,
            **errors.rate_limit_headers(routed.limit, routed.remaining),
        }
        return JSONResponse(routed.body, headers=headers)

    @api.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return RequestIdMiddleware(api)


async def _authenticate(request: Request, database: Database) -> Tenant:
    values = request.headers.getlist("authorization")
    if not values:
        raise errors.missing_api_key()
    if len(values) > 1:
        raise errors.invalid_api_key()
    scheme, _, key = values[0].strip().partition(" ")
    key = key.strip()
    if scheme.lower() != "bearer" or not key:
        raise errors.missing_api_key()
    if not _KEY.fullmatch(key):
        raise errors.invalid_api_key()
    try:
        tenant = await database.run(partial(find_tenant, key=key))
    except (sqlite3.Error, StorageError) as exc:
        logger.error("tenant lookup failed: %s", type(exc).__name__)
        raise errors.limiter_unavailable() from None
    if tenant is None:
        raise errors.invalid_api_key()
    return tenant


def _is_json(content_type: str | None) -> bool:
    media_type = (content_type or "").partition(";")[0].strip().lower()
    return media_type == "application/json"


async def _read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise errors.body_too_large(limit)
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise errors.body_too_large(limit)
    return bytes(body)


def _parse(raw: bytes) -> ChatCompletionRequest:
    try:
        data = json.loads(raw.decode(), object_pairs_hook=_no_duplicates, parse_constant=_no_nan)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise errors.invalid_request(
            "The request body is not valid JSON.", "invalid_json"
        ) from None
    try:
        return ChatCompletionRequest.model_validate(data)
    except ValidationError as exc:
        raise errors.invalid_request(describe(exc)) from None


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate key")
    return result


def _no_nan(name: str) -> object:
    raise ValueError(f"{name} is not JSON")


async def _purge_forever(limiter: TokenLimiter) -> None:
    while True:
        await anyio.sleep(PURGE_INTERVAL_SECONDS)
        try:
            await limiter.purge()
        except Exception as exc:
            logger.warning("purging expired usage failed: %s", type(exc).__name__)


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "req_unknown")


def _respond(request: Request, error: errors.GatewayError) -> JSONResponse:
    body = errors.error_body(error.error_type, error.code, error.message, _request_id(request))
    return JSONResponse(body, status_code=error.status, headers=error.headers)


async def _gateway_error(request: Request, exc: errors.GatewayError) -> JSONResponse:
    return _respond(request, exc)


async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _respond(request, errors.for_status(exc.status_code, headers=exc.headers))


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    paths = [".".join(str(p) for p in e.get("loc", ())) for e in exc.errors()][:5]
    return _respond(request, errors.invalid_request(f"Invalid request: {', '.join(paths)}."))


async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("request %s failed: %s", _request_id(request), type(exc).__name__, exc_info=exc)
    return _respond(request, errors.internal_error())
