"""Forwarding to the downstream MCP server, with the reply streamed back unbuffered."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import anyio
import httpx2
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from ai_gateway.mcp_gateway.jsonrpc import Message, encode_message

_RELAYED_RESPONSE_HEADERS: Final = frozenset({"content-type", "cache-control", "x-accel-buffering"})


class DownstreamError(Exception):
    """The downstream server could not be reached or did not start answering in time."""

    def __init__(self, http_status: int, message: str) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.message = message


class ClientDisconnected(Exception):
    """The client went away before the downstream server answered."""


def downstream_request_headers(raw_headers: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Headers for the downstream request: content type, ``Accept`` and the ``Mcp-*`` headers.

    The client's ``Authorization`` header is never forwarded, because the MCP spec forbids token
    passthrough. Cookies, ``Origin`` and proxy headers are dropped too.
    """
    headers = [("content-type", "application/json"), ("accept-encoding", "identity")]
    for name, value in raw_headers:
        key = name.lower()
        if key == "accept" or key.startswith("mcp-"):
            headers.append((key, value))
    return headers


def relayed_response_headers(upstream: httpx2.Response) -> dict[str, str]:
    return {
        name.lower(): value
        for name, value in upstream.headers.items()
        if name.lower() in _RELAYED_RESPONSE_HEADERS or name.lower().startswith("mcp-")
    }


class UpstreamResponse(StreamingResponse):
    """Relays the downstream body as it arrives, and always closes the downstream response.

    Starlette never closes a streaming body iterator, and background tasks are skipped when the
    client disconnects on newer ASGI servers, so the cleanup lives here.
    """

    def __init__(self, upstream: httpx2.Response) -> None:
        # aiter_bytes, not aiter_raw: it also serves a body that is already loaded, and it decodes
        # any Content-Encoding the downstream applied (that header is not relayed).
        super().__init__(
            upstream.aiter_bytes(),
            status_code=upstream.status_code,
            headers=relayed_response_headers(upstream),
        )
        self._upstream = upstream

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self._upstream.aclose()


async def forward(
    client: httpx2.AsyncClient,
    url: str,
    message: Message,
    raw_headers: Sequence[tuple[str, str]],
    receive: Receive,
    *,
    header_timeout: float,
) -> UpstreamResponse:
    """Send the rebuilt ``message`` downstream and return a response that relays the reply."""
    request = client.build_request(
        "POST",
        url,
        content=encode_message(message),
        headers=downstream_request_headers(raw_headers),
    )
    return UpstreamResponse(
        await _send_unless_disconnected(client, request, receive, header_timeout)
    )


async def _send_unless_disconnected(
    client: httpx2.AsyncClient, request: httpx2.Request, receive: Receive, header_timeout: float
) -> httpx2.Response:
    """Send ``request``, but give up if the client disconnects before the reply starts.

    While the endpoint waits for the downstream headers, nothing reads the ASGI ``receive`` channel,
    so a disconnect would go unnoticed and the tool would keep running. For 2026-07-28 servers,
    closing the request is the cancellation signal, so it has to be passed on.
    """
    response: httpx2.Response | None = None
    failure: Exception | None = None
    async with anyio.create_task_group() as tg:

        async def cancel_on_disconnect() -> None:
            while (await receive())["type"] != "http.disconnect":
                pass
            tg.cancel_scope.cancel()

        tg.start_soon(cancel_on_disconnect)
        try:
            with anyio.fail_after(header_timeout):
                response = await client.send(request, stream=True)
        except (TimeoutError, httpx2.TimeoutException):
            failure = DownstreamError(504, "Upstream MCP server timed out")
        except httpx2.HTTPError:
            failure = DownstreamError(502, "Upstream MCP server unavailable")
        except Exception as exc:
            failure = exc
        tg.cancel_scope.cancel()

    if failure is not None:
        raise failure
    if response is None:
        raise ClientDisconnected
    return response
