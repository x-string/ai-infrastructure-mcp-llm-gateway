"""Task 2: the gateway's HTTP contract, against a fake downstream that records every request.

Every rejection test also asserts that the downstream server was never called.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import anyio
import httpx2
import pytest
from fastapi.testclient import TestClient
from mcp.types import (
    HEADER_MISMATCH,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
)

from ai_gateway.mcp_gateway.app import create_app
from ai_gateway.mcp_gateway.auth import Role, TokenSettings, issue_token
from ai_gateway.mcp_gateway.settings import GatewaySettings

SECRET = "gateway-test-secret-0123456789abcdef"
SETTINGS = GatewaySettings(
    token=TokenSettings(secret=SECRET),
    downstream_url="http://downstream.test:8001/mcp",
    allowed_origins=frozenset({"https://app.example"}),
)
ADMIN = issue_token(SETTINGS.token, "root", Role.ADMIN)
VIEWER = issue_token(SETTINGS.token, "alice", Role.VIEWER)
MODERN_META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
}


class FakeDownstream:
    """Stands in for the downstream MCP server and records every request it receives."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.respond: Callable[[httpx2.Request], httpx2.Response] = self.echo

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.respond(request)

    @staticmethod
    def echo(request: httpx2.Request) -> httpx2.Response:
        message = json.loads(request.content)
        return httpx2.Response(
            200,
            json={"jsonrpc": "2.0", "id": message.get("id"), "result": {"echo": message["method"]}},
            headers={"mcp-session-id": "session-1", "set-cookie": "tracking=1", "x-internal": "1"},
        )


@pytest.fixture
def downstream() -> FakeDownstream:
    return FakeDownstream()


@pytest.fixture
def gateway(downstream: FakeDownstream) -> Iterator[TestClient]:
    app = create_app(SETTINGS, transport=httpx2.MockTransport(downstream.handler))
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def rpc(method: str, *, request_id: int | str = 1, params: dict[str, Any] | None = None) -> dict:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def call(name: str, *, request_id: int | str = 1) -> dict[str, Any]:
    return rpc("tools/call", request_id=request_id, params={"name": name, "arguments": {}})


def headers_for(token: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    headers.update(extra or {})
    return headers


def post(
    gateway: TestClient,
    body: dict[str, Any] | bytes,
    *,
    token: str | None = VIEWER,
    headers: dict[str, str] | None = None,
) -> httpx2.Response:
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return gateway.post("/mcp", content=content, headers=headers_for(token, headers))


def modern(method: str, *, name: str | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """A 2026-07-28 request: envelope metadata in the body, routing fields mirrored in headers."""
    params: dict[str, Any] = {"_meta": MODERN_META}
    headers = {"mcp-protocol-version": "2026-07-28", "mcp-method": method}
    if name is not None:
        params |= {"name": name, "arguments": {}}
        headers["mcp-name"] = name
    return rpc(method, params=params), headers


# --- forwarding -----------------------------------------------------------------------------------


def test_tools_list_is_forwarded_and_the_reply_relayed(
    gateway: TestClient, downstream: FakeDownstream
) -> None:
    response = post(gateway, rpc("tools/list"))

    assert response.status_code == 200
    assert response.json() == {"jsonrpc": "2.0", "id": 1, "result": {"echo": "tools/list"}}
    (forwarded,) = downstream.requests
    assert (forwarded.method, str(forwarded.url)) == ("POST", "http://downstream.test:8001/mcp")


def test_forwarded_body_is_rebuilt_from_what_was_authorized(
    gateway: TestClient, downstream: FakeDownstream
) -> None:
    pretty = (
        b'{\n  "params": {"name": "get_status"},\n  "method": "tools/call",'
        b'\n  "id": 4,\n  "jsonrpc": "2.0"\n}'
    )
    post(gateway, pretty)
    assert downstream.requests[0].content == (
        b'{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"get_status"}}'
    )


def test_only_protocol_headers_are_forwarded(
    gateway: TestClient, downstream: FakeDownstream
) -> None:
    extra = {
        "mcp-protocol-version": "2025-11-25",
        "mcp-session-id": "session-1",
        "cookie": "session=abc",
        "origin": "https://app.example",
        "x-forwarded-for": "10.0.0.1",
        "x-custom": "1",
    }
    post(gateway, rpc("tools/list"), headers=extra)

    forwarded = downstream.requests[0].headers
    assert "authorization" not in forwarded  # the client's token is never passed through
    for dropped in ("cookie", "origin", "x-forwarded-for", "x-custom"):
        assert dropped not in forwarded
    assert forwarded["mcp-protocol-version"] == "2025-11-25"
    assert forwarded["mcp-session-id"] == "session-1"
    assert forwarded["content-type"] == "application/json"
    assert forwarded["accept"] == "application/json, text/event-stream"


def test_response_headers_are_filtered(gateway: TestClient) -> None:
    response = post(gateway, rpc("tools/list"))
    assert response.headers["mcp-session-id"] == "session-1"
    assert "set-cookie" not in response.headers
    assert "x-internal" not in response.headers
    assert response.headers["x-request-id"]


def test_admin_tool_call_by_an_admin_is_forwarded(
    gateway: TestClient, downstream: FakeDownstream
) -> None:
    response = post(gateway, call("admin_reset_key"), token=ADMIN)
    assert response.json()["result"] == {"echo": "tools/call"}
    assert len(downstream.requests) == 1


def test_allowed_notification_is_forwarded(gateway: TestClient, downstream: FakeDownstream) -> None:
    downstream.respond = lambda request: httpx2.Response(202)
    response = post(gateway, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert response.status_code == 202
    assert len(downstream.requests) == 1


def test_event_stream_is_relayed_as_it_arrives(
    gateway: TestClient, downstream: FakeDownstream
) -> None:
    first_event_read = threading.Event()

    async def events() -> AsyncIterator[bytes]:
        yield b'data: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}\n\n'
        # Hold the final result until the client has read the first event.
        await anyio.to_thread.run_sync(first_event_read.wait, 10, abandon_on_cancel=True)
        yield b'data: {"jsonrpc":"2.0","id":1,"result":{}}\n\n'

    downstream.respond = lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content=events()
    )
    content = json.dumps(call("long_task")).encode()
    with gateway.stream("POST", "/mcp", content=content, headers=headers_for(VIEWER)) as response:
        assert response.headers["content-type"] == "text/event-stream"
        chunks = response.iter_bytes()
        assert b"notifications/progress" in next(chunks)  # relayed while the result was pending
        first_event_read.set()
        assert b'"result":{}' in b"".join(chunks)


# --- the brief's rule: admin_* tools need the admin role ------------------------------------------


@pytest.mark.parametrize("name", ["admin_reset_key", "ADMIN_reset_key", "Admin_Delete_User"])
def test_admin_tool_call_by_a_viewer_is_answered_by_the_gateway(
    gateway: TestClient, downstream: FakeDownstream, name: str
) -> None:
    response = post(gateway, call(name, request_id="call-9"))

    assert response.status_code == 200
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": "call-9",
        "error": {
            "code": -32001,
            "message": "Unauthorized Tool Call",
            "data": {"reason": "forbidden_tool", "tool": name, "required_role": "admin"},
        },
    }
    assert downstream.requests == []


# --- everything rejected stays away from the downstream server --------------------------------

REJECTED = [
    pytest.param(b"not json", 400, PARSE_ERROR, None, id="not-json"),
    pytest.param(
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        b'"params":{"name":"get_status","name":"admin_reset_key"}}',
        400,
        PARSE_ERROR,
        None,
        id="duplicate-key-smuggling",
    ),
    pytest.param(
        b'[{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key"}}]',
        400,
        INVALID_REQUEST,
        None,
        id="batch",
    ),
    pytest.param(
        b'{"jsonrpc":"2.0","id":7,"result":{},"method":"tools/call",'
        b'"params":{"name":"admin_reset_key","arguments":{}}}',
        400,
        INVALID_REQUEST,
        7,
        id="response-smuggling",
    ),
    pytest.param(
        json.dumps(rpc("resources/list")).encode(), 200, METHOD_NOT_FOUND, 1, id="unlisted"
    ),
    pytest.param(
        json.dumps(call(" admin_reset_key")).encode(), 200, INVALID_PARAMS, 1, id="padded"
    ),
]


@pytest.mark.parametrize(("body", "status", "code", "request_id"), REJECTED)
def test_rejected_requests_never_reach_the_downstream(
    gateway: TestClient,
    downstream: FakeDownstream,
    body: bytes,
    status: int,
    code: int,
    request_id: int | None,
) -> None:
    response = post(gateway, body)
    assert response.status_code == status
    assert (response.json()["error"]["code"], response.json()["id"]) == (code, request_id)
    assert downstream.requests == []


def test_tool_call_sent_as_a_notification_is_rejected_without_an_id(
    gateway: TestClient, downstream: FakeDownstream
) -> None:
    notification = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "admin_reset_key"}}
    response = post(gateway, notification)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == INVALID_REQUEST
    assert "id" not in response.json()
    assert downstream.requests == []


# --- 2026-07-28 (modern) clients ------------------------------------------------------------------


def test_modern_matching_request_is_forwarded_with_its_routing_headers(
    gateway: TestClient, downstream: FakeDownstream
) -> None:
    body, headers = modern("tools/call", name="get_status")
    assert post(gateway, body, headers=headers).status_code == 200
    forwarded = downstream.requests[0].headers
    assert (forwarded["mcp-method"], forwarded["mcp-name"]) == ("tools/call", "get_status")


def test_modern_header_body_mismatch_is_rejected(
    gateway: TestClient, downstream: FakeDownstream
) -> None:
    body, headers = modern("tools/call", name="admin_reset_key")
    headers["mcp-name"] = "get_status"  # headers say one tool, the body another
    response = post(gateway, body, headers=headers, token=ADMIN)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == HEADER_MISMATCH
    assert downstream.requests == []


def test_modern_unlisted_method_is_404(gateway: TestClient, downstream: FakeDownstream) -> None:
    body, headers = modern("resources/list")
    response = post(gateway, body, headers=headers)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == METHOD_NOT_FOUND
    assert downstream.requests == []


def test_modern_viewer_admin_call_still_gets_200_with_minus_32001(gateway: TestClient) -> None:
    body, headers = modern("tools/call", name="admin_reset_key")
    response = post(gateway, body, headers=headers)
    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32001


# --- authentication -------------------------------------------------------------------------------


def test_missing_token_gets_a_plain_bearer_challenge(
    gateway: TestClient, downstream: FakeDownstream
) -> None:
    response = post(gateway, rpc("tools/list"), token=None)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer realm="mcp-gateway"'
    assert response.json() == {
        "error": "unauthorized",
        "error_description": "bearer token required",
    }
    assert downstream.requests == []


def test_invalid_token_gets_an_invalid_token_challenge(
    gateway: TestClient, downstream: FakeDownstream
) -> None:
    other = TokenSettings(secret="a-different-secret-0123456789abcdef")
    forged_admin = issue_token(other, "mallory", Role.ADMIN)
    response = post(gateway, call("admin_reset_key"), token=forged_admin)

    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith(
        'Bearer realm="mcp-gateway", error="invalid_token"'
    )
    assert response.json()["error"] == "invalid_token"
    assert downstream.requests == []


def test_authentication_comes_before_body_parsing(gateway: TestClient) -> None:
    assert post(gateway, b"not json", token=None).status_code == 401


# --- transport-level checks ---------------------------------------------------------------------


def test_foreign_origin_is_forbidden(gateway: TestClient, downstream: FakeDownstream) -> None:
    response = post(gateway, rpc("tools/list"), headers={"origin": "https://evil.example"})
    assert response.status_code == 403
    assert downstream.requests == []


def test_allowed_origin_is_accepted(gateway: TestClient) -> None:
    response = post(gateway, rpc("tools/list"), headers={"origin": "https://app.example"})
    assert response.status_code == 200


def test_non_json_content_type_is_rejected(gateway: TestClient, downstream: FakeDownstream) -> None:
    response = post(gateway, rpc("tools/list"), headers={"content-type": "text/plain"})
    assert response.status_code == 415
    assert downstream.requests == []


def test_oversized_body_is_rejected(gateway: TestClient, downstream: FakeDownstream) -> None:
    response = post(gateway, b"{" + b" " * (1024 * 1024) + b"}")
    assert response.status_code == 413
    assert downstream.requests == []


@pytest.mark.parametrize("method", ["GET", "DELETE", "PUT"])
def test_only_post_is_supported(gateway: TestClient, method: str) -> None:
    response = gateway.request(method, "/mcp", headers=headers_for(VIEWER))
    assert response.status_code == 405


# --- downstream failures never leak details ----------------------------------------------------


def raise_error(error: Exception) -> Callable[[httpx2.Request], httpx2.Response]:
    def respond(request: httpx2.Request) -> httpx2.Response:
        raise error

    return respond


@pytest.mark.parametrize(
    ("error", "status", "message"),
    [
        (httpx2.ConnectError("[Errno 61] refused by 10.1.2.3:8001"), 502, "unavailable"),
        (httpx2.RemoteProtocolError("peer closed 10.1.2.3"), 502, "unavailable"),
        (httpx2.ConnectTimeout("10.1.2.3 timed out"), 504, "timed out"),
    ],
)
def test_downstream_failures_are_sanitized(
    gateway: TestClient, downstream: FakeDownstream, error: Exception, status: int, message: str
) -> None:
    downstream.respond = raise_error(error)
    response = post(gateway, rpc("tools/list", request_id=5))

    assert response.status_code == status
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": 5,
        "error": {"code": -32603, "message": f"Upstream MCP server {message}"},
    }
    assert "10.1.2.3" not in response.text


def test_unexpected_errors_are_a_generic_internal_error(
    gateway: TestClient, downstream: FakeDownstream
) -> None:
    downstream.respond = raise_error(RuntimeError("password=hunter2"))
    response = post(gateway, rpc("tools/list", request_id=6))

    assert response.status_code == 500
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": 6,
        "error": {"code": -32603, "message": "Internal error"},
    }
    assert "hunter2" not in response.text


# --- audit log ------------------------------------------------------------------------------------


def test_decisions_are_audited_without_the_token(
    gateway: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="ai_gateway.mcp_gateway.audit"):
        post(gateway, call("admin_reset_key"))

    line = caplog.records[-1].getMessage()
    for field in ("subject=alice", "role=viewer", "method=tools/call", "tool=admin_reset_key"):
        assert field in line
    assert "decision=deny" in line and "code=-32001" in line
    assert VIEWER not in caplog.text
