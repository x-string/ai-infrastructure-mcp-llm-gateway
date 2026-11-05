"""HTTP layer of the MCP security gateway (Task 2).

One endpoint, ``POST /mcp``. Every request passes these steps in order:

1. Origin check (403). The MCP spec requires it to block DNS rebinding.
2. Authentication (401): ``Authorization: Bearer <JWT>``.
3. Content type (415) and body size (413, enforced while the body is read).
4. Strict JSON-RPC parsing and, for 2026-07-28 clients, the header/body check.
5. Authorization. Denied requests are answered here and never forwarded.
6. Forwarding. The rebuilt message goes to the downstream server and the reply streams back.

Every decision is written to the ``ai_gateway.mcp_gateway.audit`` log; tokens are never logged.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Final

import httpx2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from mcp.types import INTERNAL_ERROR
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from ai_gateway.mcp_gateway import jsonrpc
from ai_gateway.mcp_gateway.auth import AuthenticationError, TokenVerifier
from ai_gateway.mcp_gateway.policy import Deny, authorize
from ai_gateway.mcp_gateway.proxy import ClientDisconnected, DownstreamError, forward
from ai_gateway.mcp_gateway.settings import GatewaySettings

logger = logging.getLogger(__name__)
audit_log = logging.getLogger("ai_gateway.mcp_gateway.audit")

REALM: Final = "mcp-gateway"
CLIENT_CLOSED_REQUEST: Final = 499
_UNPRINTABLE = re.compile(r"[^\x21-\x7e]")


def _loggable(value: str) -> str:
    """Keep audit lines single-line and bounded, whatever a client sends."""
    return _UNPRINTABLE.sub("?", value)[:128] or "-"


@dataclass
class _AuditRecord:
    request_id: str
    started: float = field(default_factory=time.perf_counter)
    subject: str = "-"
    role: str = "-"
    method: str = "-"
    tool: str = "-"
    decision: str = "-"
    code: int | None = None

    def emit(self, status: int) -> None:
        audit_log.info(
            "request_id=%s subject=%s role=%s method=%s tool=%s decision=%s code=%s status=%d "
            "duration_ms=%.1f",
            self.request_id,
            _loggable(self.subject),
            self.role,
            _loggable(self.method),
            _loggable(self.tool),
            self.decision,
            "-" if self.code is None else self.code,
            status,
            (time.perf_counter() - self.started) * 1000,
        )


def _is_json(content_type: str | None) -> bool:
    if content_type is None:
        return False
    return content_type.split(";", 1)[0].strip().lower() == "application/json"


def _http_error(status: int, error: str, description: str) -> JSONResponse:
    return JSONResponse({"error": error, "error_description": description}, status_code=status)


def _unauthorized(exc: AuthenticationError) -> JSONResponse:
    """401 with an RFC 6750 challenge, adding ``error="invalid_token"`` only if a token was sent."""
    challenge = f'Bearer realm="{REALM}"'
    if exc.token_present:
        challenge += f', error="invalid_token", error_description="{exc.reason}"'
        body = {"error": "invalid_token", "error_description": exc.reason}
    else:
        body = {"error": "unauthorized", "error_description": exc.reason}
    return JSONResponse(body, status_code=401, headers={"www-authenticate": challenge})


def _rpc_error(
    code: int,
    message: str,
    *,
    request_id: jsonrpc.RequestId | None,
    status: int,
    data: Any = None,
    include_id: bool = True,
) -> Response:
    body = jsonrpc.error_body(
        code, message, request_id=request_id, data=data, include_id=include_id
    )
    return Response(content=body, status_code=status, media_type="application/json")


def create_app(
    settings: GatewaySettings, *, transport: httpx2.AsyncBaseTransport | None = None
) -> FastAPI:
    """Build the gateway. Tests pass ``transport`` to stand in for the downstream server."""
    verifier = TokenVerifier(settings.token)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx2.Timeout(
            connect=settings.connect_timeout_seconds, read=None, write=10.0, pool=3.0
        )
        async with httpx2.AsyncClient(
            transport=transport, timeout=timeout, trust_env=False, follow_redirects=False
        ) as client:
            app.state.downstream = client
            yield

    app = FastAPI(
        title="MCP security gateway",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=settings.max_body_bytes)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> Response:
        return _rpc_error(INTERNAL_ERROR, "Internal error", request_id=None, status=500)

    async def handle(request: Request, record: _AuditRecord) -> Response:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            record.decision = "forbidden_origin"
            return _http_error(403, "forbidden_origin", "this origin is not allowed")

        try:
            principal = verifier.authenticate(request.headers.getlist("authorization"))
        except AuthenticationError as exc:
            record.decision = "unauthenticated"
            return _unauthorized(exc)
        record.subject, record.role = principal.subject, principal.role.value

        if not _is_json(request.headers.get("content-type")):
            record.decision = "unsupported_media_type"
            return _http_error(
                415, "unsupported_media_type", "Content-Type must be application/json"
            )

        body = await request.body()  # RequestBodyLimitMiddleware answers 413 past the limit
        raw_headers = [(k.decode("latin-1"), v.decode("latin-1")) for k, v in request.headers.raw]
        modern = jsonrpc.is_modern(raw_headers)

        message: jsonrpc.Message | None = None
        try:
            message = jsonrpc.parse_message(body)
            record.method = message.method
            jsonrpc.check_routing_headers(raw_headers, message)
        except jsonrpc.JsonRpcError as exc:
            record.decision, record.code = "reject", exc.code
            return _rpc_error(
                exc.code,
                exc.message,
                request_id=exc.request_id,
                data=exc.data,
                status=jsonrpc.http_status_for(exc.code, modern=modern),
                include_id=not isinstance(message, jsonrpc.Notification),
            )

        is_notification = isinstance(message, jsonrpc.Notification)
        request_id = None if is_notification else message.id
        tool = (message.params or {}).get("name") if message.method == "tools/call" else None
        if isinstance(tool, str):
            record.tool = tool

        decision = authorize(principal, message)
        if isinstance(decision, Deny):
            record.decision, record.code = "deny", decision.code
            return _rpc_error(
                decision.code,
                decision.message,
                request_id=request_id,
                data=decision.data,
                status=400
                if is_notification
                else jsonrpc.http_status_for(decision.code, modern=modern),
                include_id=not is_notification,
            )

        record.decision = "allow"
        try:
            return await forward(
                request.app.state.downstream,
                settings.downstream_url,
                message,
                raw_headers,
                request.receive,
                header_timeout=settings.header_timeout_seconds,
            )
        except DownstreamError as exc:
            record.decision = "downstream_error"
            return _rpc_error(
                INTERNAL_ERROR,
                exc.message,
                request_id=request_id,
                status=exc.http_status,
                include_id=not is_notification,
            )
        except ClientDisconnected:
            record.decision = "client_disconnected"
            return Response(status_code=CLIENT_CLOSED_REQUEST)
        except Exception:
            logger.exception("Unexpected error while forwarding request %s", record.request_id)
            record.decision = "error"
            return _rpc_error(
                INTERNAL_ERROR,
                "Internal error",
                request_id=request_id,
                status=500,
                include_id=not is_notification,
            )

    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        record = _AuditRecord(request_id=uuid.uuid4().hex)
        response = await handle(request, record)
        response.headers["x-request-id"] = record.request_id
        record.emit(response.status_code)
        return response

    return app
