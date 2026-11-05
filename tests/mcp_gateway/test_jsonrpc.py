"""Task 2: strict JSON-RPC parsing, canonical forwarding and routing-header checks."""

from __future__ import annotations

import base64
import json

import pytest
from mcp.types import (
    HEADER_MISMATCH,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    UNSUPPORTED_PROTOCOL_VERSION,
)

from ai_gateway.mcp_gateway.jsonrpc import (
    JsonRpcError,
    Notification,
    Request,
    check_routing_headers,
    encode_message,
    error_body,
    http_status_for,
    is_modern,
    parse_message,
)


def rejected(body: bytes) -> JsonRpcError:
    with pytest.raises(JsonRpcError) as excinfo:
        parse_message(body)
    return excinfo.value


# --- valid messages ------------------------------------------------------------------------------


def test_request_is_parsed_with_params_intact() -> None:
    params = {"name": "echo", "arguments": {"text": "hi"}, "_meta": {"progressToken": 1}}
    body = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": params})
    assert parse_message(body.encode()) == Request(id=7, method="tools/call", params=params)


def test_notification_is_parsed() -> None:
    body = b'{"jsonrpc": "2.0", "method": "notifications/initialized"}'
    assert parse_message(body) == Notification(method="notifications/initialized", params=None)


@pytest.mark.parametrize("request_id", ["req-1", 0, 2**70], ids=["string", "zero", "huge"])
def test_string_and_integer_ids_are_kept_exactly(request_id: str | int) -> None:
    body = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": "ping"}).encode()
    assert parse_message(body) == Request(id=request_id, method="ping", params=None)


# --- -32700 Parse error: anything that is not plain, unambiguous JSON -------------------------

PARSE_ERRORS = [
    pytest.param(b"", id="empty"),
    pytest.param(b"not json", id="not-json"),
    pytest.param(b"\xff\xfe{}", id="invalid-utf8"),
    pytest.param("﻿{}".encode(), id="utf8-bom"),
    pytest.param('{"jsonrpc": "2.0"}'.encode("utf-16"), id="utf16"),
    pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"ping","x":NaN}', id="nan"),
    pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"ping","x":-Infinity}', id="infinity"),
    pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"ping","x":1e309}', id="overflow"),
    pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"ping","method":"tools/list"}', id="dup-key"),
    pytest.param(
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        b'"params":{"name":"get_status","name":"admin_reset_key"}}',
        id="dup-nested-key",
    ),
    pytest.param(b"[" * 50_000 + b"]" * 50_000, id="deep-nesting"),
    pytest.param(b'{"jsonrpc":"2.0","id":' + b"1" * 5_000 + b',"method":"ping"}', id="5000-digit"),
    pytest.param(b'{"jsonrpc":"2.0","id":"\\ud800","method":"ping"}', id="lone-surrogate"),
]


@pytest.mark.parametrize("body", PARSE_ERRORS)
def test_unparseable_bodies_are_parse_errors(body: bytes) -> None:
    error = rejected(body)
    assert (error.code, error.request_id) == (PARSE_ERROR, None)


# --- -32600 Invalid Request: the body must be exactly one request or notification -------------

INVALID_REQUESTS = [
    pytest.param(b'[{"jsonrpc":"2.0","id":1,"method":"ping"}]', None, id="batch"),
    pytest.param(b"[]", None, id="empty-batch"),
    pytest.param(b'"tools/list"', None, id="string"),
    pytest.param(b"42", None, id="number"),
    pytest.param(b'{"jsonrpc":"2.0","id":null,"method":"ping"}', None, id="id-null"),
    pytest.param(b'{"jsonrpc":"2.0","id":true,"method":"ping"}', None, id="id-bool"),
    pytest.param(b'{"jsonrpc":"2.0","id":1.0,"method":"ping"}', None, id="id-float"),
    pytest.param(b'{"jsonrpc":"2.0","id":[1],"method":"ping"}', None, id="id-list"),
    # The SDK downstream ignores unknown keys and would execute this as a tools/call.
    pytest.param(
        b'{"jsonrpc":"2.0","id":7,"result":{},"method":"tools/call",'
        b'"params":{"name":"admin_reset_key","arguments":{}}}',
        7,
        id="response-smuggling-a-call",
    ),
    pytest.param(
        b'{"jsonrpc":"2.0","id":7,"error":{"code":1,"message":"x"},"method":"tools/call"}',
        7,
        id="error-smuggling-a-call",
    ),
    pytest.param(b'{"jsonrpc":"2.0","id":7,"result":{}}', 7, id="client-response"),
    pytest.param(b'{"jsonrpc":"2.0","id":7,"method":"ping","extra":1}', 7, id="unknown-member"),
    pytest.param(b'{"id":7,"method":"ping"}', 7, id="missing-jsonrpc"),
    pytest.param(b'{"jsonrpc":"1.0","id":7,"method":"ping"}', 7, id="jsonrpc-1.0"),
    pytest.param(b'{"jsonrpc":"2.0","id":7}', 7, id="missing-method"),
    pytest.param(b'{"jsonrpc":"2.0","id":7,"method":42}', 7, id="method-not-string"),
    pytest.param(b'{"jsonrpc":"2.0","id":7,"method":""}', 7, id="empty-method"),
    pytest.param(b'{"jsonrpc":"2.0","method":"ping","params":[1]}', None, id="notify-params-list"),
]


@pytest.mark.parametrize(("body", "request_id"), INVALID_REQUESTS)
def test_malformed_envelopes_are_invalid_requests(body: bytes, request_id: int | None) -> None:
    error = rejected(body)
    assert (error.code, error.request_id) == (INVALID_REQUEST, request_id)


@pytest.mark.parametrize("params", [b'["admin_reset_key"]', b"null", b'"x"'])
def test_request_params_must_be_an_object(params: bytes) -> None:
    error = rejected(b'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":' + params + b"}")
    assert (error.code, error.request_id) == (INVALID_PARAMS, 3)


def test_rejections_never_echo_the_body() -> None:
    error = rejected(b'{"jsonrpc":"2.0","id":1,"method":"ping","ssn":"123-45-6789","x":NaN}')
    assert "6789" not in f"{error} {error.reason} {error.data}"


# --- forwarding: the body is rebuilt from the parsed fields --------------------------------------


def test_forwarded_body_is_rebuilt_from_the_parsed_fields() -> None:
    raw = (
        b'{ "params": {"name": "echo", "arguments": {"text": "caf\\u00e9"}},'
        b'  "method": "tools/call", "id": 5, "jsonrpc": "2.0" }'
    )
    assert encode_message(parse_message(raw)) == (
        b'{"jsonrpc":"2.0","id":5,"method":"tools/call",'
        b'"params":{"name":"echo","arguments":{"text":"caf\\u00e9"}}}'
    )


def test_forwarded_notification_has_no_id() -> None:
    message = parse_message(b'{"method":"notifications/initialized","jsonrpc":"2.0"}')
    assert encode_message(message) == b'{"jsonrpc":"2.0","method":"notifications/initialized"}'


# --- error bodies and HTTP statuses --------------------------------------------------------------


def test_error_body_with_unknown_id() -> None:
    body = error_body(PARSE_ERROR, "Parse error", request_id=None, data={"reason": "bad"})
    assert json.loads(body) == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": PARSE_ERROR, "message": "Parse error", "data": {"reason": "bad"}},
    }


def test_error_body_for_a_notification_has_no_id() -> None:
    body = json.loads(error_body(METHOD_NOT_FOUND, "Method not found", include_id=False))
    assert "id" not in body


def test_error_body_is_ascii_json() -> None:
    body = error_body(-32001, "Unauthorized Tool Call", request_id="réq-1")
    body.decode("ascii")  # raises if anything is not escaped
    assert json.loads(body)["id"] == "réq-1"


@pytest.mark.parametrize(
    ("code", "modern", "status"),
    [
        (PARSE_ERROR, False, 400),
        (PARSE_ERROR, True, 400),
        (INVALID_REQUEST, False, 400),
        (INVALID_REQUEST, True, 400),
        (HEADER_MISMATCH, True, 400),
        (INVALID_PARAMS, False, 200),
        (INVALID_PARAMS, True, 400),
        (METHOD_NOT_FOUND, False, 200),
        (METHOD_NOT_FOUND, True, 404),
        (-32001, False, 200),
        (-32001, True, 200),
    ],
)
def test_http_status_follows_the_protocol_era(code: int, modern: bool, status: int) -> None:
    assert http_status_for(code, modern=modern) == status


# --- routing headers (2026-07-28 clients mirror method and tool name in headers) ----------------

MODERN_META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
}


def modern_call(name: str) -> Request:
    return Request(
        id=1, method="tools/call", params={"name": name, "arguments": {}, "_meta": MODERN_META}
    )


def modern_headers(
    *, method: str = "tools/call", name: str = "echo", version: str = "2026-07-28"
) -> list[tuple[str, str]]:
    return [("MCP-Protocol-Version", version), ("Mcp-Method", method), ("Mcp-Name", name)]


def test_era_is_modern_only_for_non_handshake_versions() -> None:
    assert is_modern([]) is False
    assert is_modern([("mcp-protocol-version", "2025-11-25")]) is False
    assert is_modern([("MCP-Protocol-Version", "2026-07-28")]) is True
    assert is_modern([("mcp-protocol-version", "2099-01-01")]) is True
    assert (
        is_modern([("mcp-protocol-version", "2025-06-18"), ("mcp-protocol-version", "x")]) is False
    )


def test_legacy_requests_skip_header_checks() -> None:
    check_routing_headers([("Mcp-Method", "tools/list")], modern_call("echo"))


def test_matching_modern_headers_pass() -> None:
    check_routing_headers(modern_headers(), modern_call("echo"))


def test_base64_encoded_tool_name_header_passes() -> None:
    encoded = "=?base64?" + base64.b64encode(b"echo").decode() + "?="
    check_routing_headers(modern_headers(name=encoded), modern_call("echo"))


@pytest.mark.parametrize(
    ("headers", "code"),
    [
        pytest.param(modern_headers(name="echo"), HEADER_MISMATCH, id="tool-name-mismatch"),
        pytest.param(modern_headers(method="tools/list"), HEADER_MISMATCH, id="method-mismatch"),
        pytest.param(
            [*modern_headers(name="admin_reset_key"), ("mcp-name", "echo")],
            HEADER_MISMATCH,
            id="duplicated-name-header",
        ),
        pytest.param(
            modern_headers(name="admin_reset_key", version="2099-01-01"),
            HEADER_MISMATCH,
            id="version-header-disagrees-with-body",
        ),
    ],
)
def test_modern_header_body_disagreement_is_rejected(
    headers: list[tuple[str, str]], code: int
) -> None:
    with pytest.raises(JsonRpcError) as excinfo:
        check_routing_headers(headers, modern_call("admin_reset_key"))
    assert (excinfo.value.code, excinfo.value.request_id) == (code, 1)


def test_modern_request_without_envelope_meta_is_invalid_params() -> None:
    request = Request(id=2, method="tools/call", params={"name": "echo", "arguments": {}})
    with pytest.raises(JsonRpcError) as excinfo:
        check_routing_headers(modern_headers(), request)
    assert (excinfo.value.code, excinfo.value.request_id) == (INVALID_PARAMS, 2)


def test_unsupported_modern_version_is_rejected() -> None:
    meta = {**MODERN_META, "io.modelcontextprotocol/protocolVersion": "2099-01-01"}
    request = Request(id=3, method="tools/call", params={"name": "echo", "_meta": meta})
    with pytest.raises(JsonRpcError) as excinfo:
        check_routing_headers(modern_headers(version="2099-01-01"), request)
    assert excinfo.value.code == UNSUPPORTED_PROTOCOL_VERSION
