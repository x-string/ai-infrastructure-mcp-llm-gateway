"""A downstream MCP server for the Task 2 gateway, built on the official SDK.

``t2-mock-downstream`` serves ordinary tools and ``admin_*`` tools over Streamable HTTP in stateless
mode, so the gateway's authorization can be shown end to end. It has no authentication of its own:
bind it to localhost, where the gateway is its only way in.

Options:

* ``--port`` (default 8001) and ``--host`` (default 127.0.0.1)
* ``--sse`` to answer with event streams instead of single JSON responses
* ``--expose-calls`` to add ``GET /_calls``, a log of tool activity read by the end-to-end tests
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from collections.abc import Sequence
from typing import Any, Final

import anyio
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

MAX_SLEEP_SECONDS: Final = 60.0
LOCAL_HOSTS: Final = ["127.0.0.1:*", "localhost:*", "[::1]:*"]


class CallLog:
    """Thread-safe record of tool activity: which tools started, finished or were cancelled."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, str]] = []

    def record(self, tool: str, event: str) -> None:
        with self._lock:
            self._events.append({"tool": tool, "event": event})

    def snapshot(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._events)


def build_server(calls: CallLog) -> MCPServer:
    mcp = MCPServer(
        "demo-downstream",
        instructions=(
            "Demo tools behind the MCP security gateway. "
            "Tools whose names start with admin_ need the admin role at the gateway."
        ),
    )

    @mcp.tool()
    def echo(text: str) -> str:
        """Return the text unchanged."""
        calls.record("echo", "finished")
        return text

    @mcp.tool()
    def get_status() -> dict[str, str]:
        """Report that the downstream service is healthy."""
        calls.record("get_status", "finished")
        return {"status": "ok"}

    @mcp.tool()
    async def slow_echo(text: str, seconds: float) -> str:
        """Wait up to 60 seconds, then return the text. Used by timeout and cancellation tests."""
        calls.record("slow_echo", "started")
        try:
            await anyio.sleep(min(max(seconds, 0.0), MAX_SLEEP_SECONDS))
        except anyio.get_cancelled_exc_class():
            calls.record("slow_echo", "cancelled")
            raise
        calls.record("slow_echo", "finished")
        return text

    @mcp.tool()
    def admin_reset_key(key_id: str) -> str:
        """Rotate an API key. Admin only."""
        calls.record("admin_reset_key", "finished")
        return f"Key {key_id} rotated"

    @mcp.tool()
    def admin_delete_user(user_id: str) -> str:
        """Delete a user account. Admin only."""
        calls.record("admin_delete_user", "finished")
        return f"User {user_id} deleted"

    return mcp


def create_app(
    *, sse: bool = False, expose_calls: bool = False, calls: CallLog | None = None
) -> Starlette:
    log = calls if calls is not None else CallLog()
    app = build_server(log).streamable_http_app(
        stateless_http=True,
        json_response=not sse,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=LOCAL_HOSTS,
            allowed_origins=[],
        ),
    )
    if expose_calls:

        async def list_calls(request: Request) -> JSONResponse:
            return JSONResponse(log.snapshot())

        app.router.routes.append(Route("/_calls", list_calls, methods=["GET"]))
    return app


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="t2-mock-downstream", description="Demo MCP server behind the Task 2 gateway."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--sse", action="store_true", help="answer with event streams")
    parser.add_argument("--expose-calls", action="store_true", help="add GET /_calls (tests)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app: Any = create_app(sse=args.sse, expose_calls=args.expose_calls)
    uvicorn.run(app, host=args.host, port=args.port, server_header=False, log_config=None)


if __name__ == "__main__":
    main()
