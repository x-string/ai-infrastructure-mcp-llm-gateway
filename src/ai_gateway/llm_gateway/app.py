"""HTTP layer of the streaming PII-redaction gateway (Task 3).

``POST /v1/chat/completions`` takes an OpenAI chat completions request and answers with an
OpenAI-compatible event stream whose text has been redacted. A request passes these steps:

1. ``Content-Type: application/json`` (415) and the body size limit (413).
2. Strict JSON, then the request allowlist (400): only parameters whose output is redactable text.
3. A free stream slot (503 when ``T3_MAX_STREAMS`` streams are running).
4. The upstream request, rebuilt from the validated fields, and its first event (see
   ``upstream.py`` for the error statuses). If the client disconnects meanwhile, it is cancelled.
5. The redacted relay (``streaming.py``), until ``[DONE]``, an error frame, or a disconnect.

Errors use OpenAI's shape, ``{"error": {"message", "type", "code", "param"}}``, with a generic
message. Every response carries ``X-Request-ID``; nothing from the prompt or the response is
logged.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager

import anyio
import httpx2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from ai_gateway.llm_gateway.schemas import RequestError, parse_request
from ai_gateway.llm_gateway.settings import GatewaySettings
from ai_gateway.llm_gateway.streaming import StreamRecord, error_body, relay
from ai_gateway.llm_gateway.upstream import (
    UpstreamError,
    UpstreamStream,
    create_client,
    open_stream,
)

logger = logging.getLogger(__name__)

CLIENT_CLOSED_REQUEST = 499
_STATUS_CODES = {
    404: ("not_found", "Not found."),
    405: ("method_not_allowed", "Method not allowed."),
    413: ("request_too_large", "The request body is too large."),
}


def error_response(
    status: int,
    code: str,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body = {"error": {"message": message, "type": error_type, "code": code, "param": param}}
    return JSONResponse(body, status_code=status, headers=dict(headers or {}))


def upstream_error_response(error: UpstreamError) -> JSONResponse:
    headers = {"x-should-retry": "true" if error.retryable else "false"}
    if error.retry_after is not None:
        headers["retry-after"] = str(error.retry_after)
    return JSONResponse(error_body(error), status_code=error.status, headers=headers)


class ClientDisconnected(Exception):
    """The client went away before the upstream's first event arrived."""


class RelayResponse(StreamingResponse):
    """Streams the relay, and always closes it, the upstream and the stream slot.

    Starlette never closes a streaming body iterator, so the cleanup lives here, shielded so that
    a cancelled request still releases its upstream connection.
    """

    def __init__(
        self,
        content: AsyncIterator[bytes],
        upstream: UpstreamStream,
        record: StreamRecord,
        release: Callable[[], None],
    ) -> None:
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
                "x-request-id": record.request_id,
            },
        )
        self._upstream = upstream
        self._record = record
        self._release = release

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()  # type: ignore[attr-defined]
                finally:
                    await self._upstream.aclose()
            self._release()
            self._record.emit()


async def _open_unless_disconnected(
    client: httpx2.AsyncClient, settings: GatewaySettings, payload: dict, receive: Receive
) -> UpstreamStream:
    """Open the upstream stream, cancelling it if the client disconnects first.

    Nothing reads the ASGI ``receive`` channel while the endpoint waits for the first event, so a
    disconnect would go unnoticed and the provider would keep generating for nobody.
    """
    stream: UpstreamStream | None = None
    failure: Exception | None = None
    async with anyio.create_task_group() as tg:

        async def cancel_on_disconnect() -> None:
            while (await receive())["type"] != "http.disconnect":
                pass
            tg.cancel_scope.cancel()

        tg.start_soon(cancel_on_disconnect)
        try:
            stream = await open_stream(client, settings.upstream, payload)
        except Exception as exc:  # raised after the task group, so it is not wrapped in a group
            failure = exc
        tg.cancel_scope.cancel()
    if failure is not None:
        raise failure
    if stream is None:
        raise ClientDisconnected
    return stream


def create_app(
    settings: GatewaySettings, *, transport: httpx2.AsyncBaseTransport | None = None
) -> FastAPI:
    """Build the gateway. Tests pass ``transport`` to stand in for the provider."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with create_client(settings.upstream, transport=transport) as client:
            app.state.upstream = client
            app.state.slots = anyio.Semaphore(settings.max_streams)
            yield

    app = FastAPI(
        title="Streaming PII-redaction gateway",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=settings.max_body_bytes)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        code, message = _STATUS_CODES.get(exc.status_code, ("http_error", "Request failed."))
        return error_response(exc.status_code, code, message)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> Response:
        return error_response(500, "internal_error", "Internal error", error_type="server_error")

    async def handle(request: Request, record: StreamRecord) -> Response:
        media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
        if media_type != "application/json":
            record.outcome = "rejected"
            return error_response(
                415, "unsupported_media_type", "Content-Type must be application/json."
            )
        body = await request.body()  # RequestBodyLimitMiddleware raises 413 past the limit
        try:
            chat = parse_request(_load_json(body))
        except RequestError as exc:
            record.outcome, record.detail = "rejected", exc.param or "-"
            return error_response(400, "invalid_request", exc.message, param=exc.param)

        slots: anyio.Semaphore = request.app.state.slots
        try:
            slots.acquire_nowait()
        except anyio.WouldBlock:
            record.outcome = "overloaded"
            return error_response(
                503,
                "gateway_overloaded",
                "The gateway is handling too many streams. Try again later.",
                error_type="server_error",
                headers={"retry-after": "1", "x-should-retry": "true"},
            )
        try:
            upstream = await _open_unless_disconnected(
                request.app.state.upstream, settings, chat.upstream_payload(), request.receive
            )
        except UpstreamError as exc:
            slots.release()
            record.outcome, record.error_code, record.detail = (
                "upstream_error",
                exc.code,
                exc.detail,
            )
            return upstream_error_response(exc)
        except ClientDisconnected:
            slots.release()
            return Response(status_code=CLIENT_CLOSED_REQUEST)
        except BaseException:
            slots.release()
            raise
        return RelayResponse(
            relay(
                upstream,
                model=chat.model,
                rules=settings.rules,
                include_usage=chat.include_usage,
                max_seconds=settings.max_stream_seconds,
                keepalive_seconds=settings.keepalive_seconds,
                record=record,
            ),
            upstream,
            record,
            slots.release,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        record = StreamRecord(request_id=uuid.uuid4().hex)
        try:
            response = await handle(request, record)
        except StarletteHTTPException as exc:  # the body size limit, raised while reading
            record.outcome = "rejected"
            code, message = _STATUS_CODES.get(exc.status_code, ("http_error", "Request failed."))
            response = error_response(exc.status_code, code, message)
        response.headers["x-request-id"] = record.request_id
        if not isinstance(response, RelayResponse):
            record.emit()
        return response

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _load_json(body: bytes) -> object:
    try:
        return json.loads(body, parse_constant=_reject_constant)
    except (ValueError, RecursionError):  # includes UnicodeDecodeError and huge integers
        raise RequestError("The request body is not valid JSON.") from None


def _reject_constant(name: str) -> None:
    raise ValueError(f"{name} is not JSON")
