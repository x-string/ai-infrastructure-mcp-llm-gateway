"""Task 1: the validating stdio transport, tested without a subprocess."""

from __future__ import annotations

import io

import anyio
import pytest
from mcp.types import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    PARSE_ERROR,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
)

from ai_gateway.mcp_server.stdio_transport import (
    LINE_TOO_LONG,
    Forward,
    PendingRequests,
    Reply,
    parse_line,
    read_lines,
)


def error_of(outcome: Forward | Reply | None) -> tuple[int, object]:
    assert isinstance(outcome, Reply), outcome
    return outcome.error.error.code, outcome.error.id


# --- lines the transport ignores ---------------------------------------------------------------


@pytest.mark.parametrize("line", [b"\n", b"   \r\n", b"\t\n"], ids=["newline", "spaces", "tab"])
def test_blank_lines_are_ignored(line: bytes) -> None:
    assert parse_line(line) is None


def test_malformed_notifications_are_dropped_without_a_reply() -> None:
    # JSON-RPC forbids answering notifications, even invalid ones.
    line = b'{"jsonrpc": "2.0", "method": "notifications/initialized", "params": [1]}\n'
    assert parse_line(line) is None


# --- -32700 Parse error ------------------------------------------------------------------------

PARSE_ERRORS = [
    pytest.param(b"this is not json\n", id="not-json"),
    pytest.param(b'{"jsonrpc": "2.0", "id": 1,\n', id="truncated"),
    pytest.param(b"\xff\xfe{}\n", id="invalid-utf8"),
    pytest.param('\ufeff{"jsonrpc":"2.0","id":1,"method":"ping"}\n'.encode(), id="utf8-bom"),
    pytest.param(
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":'
        b'{"name":"trigger_refund","arguments":{"amount":NaN}}}\n',
        id="nan-literal",
    ),
    pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"x","amount":Infinity}\n', id="infinity"),
    pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"ping","method":"tools/list"}\n', id="dup-key"),
    pytest.param(b'{"jsonrpc":"2.0","id":"\\ud800","method":"ping"}\n', id="lone-surrogate"),
    pytest.param(b"[" * 100_000 + b"]" * 100_000 + b"\n", id="deep-nesting"),
    pytest.param(
        b'{"jsonrpc":"2.0","id":' + b"1" * 5_000 + b',"method":"ping"}\n', id="5000-digit"
    ),
]


@pytest.mark.parametrize("line", PARSE_ERRORS)
def test_unparseable_lines_are_parse_errors_with_null_id(line: bytes) -> None:
    assert error_of(parse_line(line)) == (PARSE_ERROR, None)


def test_error_replies_never_echo_the_input() -> None:
    outcome = parse_line(b'{"ssn": "123-45-6789", "card": NaN}\n')
    assert isinstance(outcome, Reply)
    assert "6789" not in outcome.error.model_dump_json()


# --- -32600 Invalid Request ----------------------------------------------------------------------

INVALID_REQUESTS = [
    pytest.param(b'[{"jsonrpc":"2.0","id":1,"method":"ping"}]', None, id="batch"),
    pytest.param(b"[]", None, id="empty-batch"),
    pytest.param(b'"ping"', None, id="string"),
    pytest.param(b"42", None, id="number"),
    pytest.param(b'{"jsonrpc":"2.0","id":null,"method":"tools/list"}', None, id="id-null"),
    pytest.param(b'{"jsonrpc":"2.0","id":true,"method":"tools/list"}', None, id="id-bool"),
    pytest.param(b'{"jsonrpc":"2.0","id":1.5,"method":"tools/list"}', None, id="id-float"),
    pytest.param(b'{"jsonrpc":"2.0","id":{},"method":"tools/list"}', None, id="id-object"),
    pytest.param(b'{"id":7,"method":"ping"}', 7, id="no-jsonrpc"),
    pytest.param(b'{"jsonrpc":"1.0","id":7,"method":"ping"}', 7, id="jsonrpc-1.0"),
    pytest.param(b'{"jsonrpc":"2.0","id":7,"method":42}', 7, id="method-not-string"),
    pytest.param(
        b'{"jsonrpc":"2.0","id":7,"method":"ping","result":{}}', 7, id="request-and-response"
    ),
    pytest.param(b'{"jsonrpc":"2.0","id":"r-7"}', "r-7", id="neither"),
]


@pytest.mark.parametrize(("line", "expected_id"), INVALID_REQUESTS)
def test_malformed_envelopes_are_invalid_requests(line: bytes, expected_id: object) -> None:
    assert error_of(parse_line(line)) == (INVALID_REQUEST, expected_id)


def test_request_params_must_be_an_object() -> None:
    line = b'{"jsonrpc":"2.0","id":9,"method":"tools/call","params":[1, 2]}\n'
    assert error_of(parse_line(line)) == (INVALID_PARAMS, 9)


# --- messages that are forwarded to the server -------------------------------------------------


def test_valid_request_is_forwarded() -> None:
    outcome = parse_line(b'{"jsonrpc":"2.0","id":"a-1","method":"tools/list"}\n')
    assert isinstance(outcome, Forward)
    assert isinstance(outcome.message, JSONRPCRequest)
    assert outcome.message.id == "a-1"


def test_valid_notification_is_forwarded() -> None:
    outcome = parse_line(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
    assert isinstance(outcome, Forward)
    assert isinstance(outcome.message, JSONRPCNotification)


def test_client_response_is_forwarded() -> None:
    outcome = parse_line(b'{"jsonrpc":"2.0","id":3,"result":{}}\n')
    assert isinstance(outcome, Forward)
    assert isinstance(outcome.message, JSONRPCResponse)


def test_out_of_range_numbers_are_left_to_the_tool_schema() -> None:
    # 1e309 is valid JSON syntax. trigger_refund's own validation rejects it later with -32602.
    line = (
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"trigger_refund",'
        b'"arguments":{"customer_id":"CUST-00042","amount":1e309,"reason":"duplicate charge"}}}\n'
    )
    assert isinstance(parse_line(line), Forward)


# --- line reader ---------------------------------------------------------------------------------


async def test_read_lines_flags_oversized_lines_and_keeps_going() -> None:
    stream = io.BytesIO(b'{"a":1}\n' + b"x" * 40 + b"\n" + b"y" * 16 + b"\n" + b'{"b":2}')
    lines = [line async for line in read_lines(stream, max_bytes=16)]
    # 16 bytes plus a newline still fits; 40 bytes does not; a final line may lack the newline.
    assert lines == [b'{"a":1}\n', LINE_TOO_LONG, b"y" * 16 + b"\n", b'{"b":2}']


# --- pending-request tracking --------------------------------------------------------------------


async def test_pending_requests_wait_until_every_request_is_answered() -> None:
    pending = PendingRequests()
    pending.add(1)
    pending.add("x")
    pending.discard(1)

    with anyio.move_on_after(0.05) as scope:
        await pending.wait_until_empty()
    assert scope.cancelled_caught  # "x" is still unanswered

    pending.discard("x")
    with anyio.fail_after(1):
        await pending.wait_until_empty()
