"""Task 1 end to end: the real server in a subprocess, spoken to over real stdin/stdout pipes.

These tests check what an MCP host actually sees: every stdout line is a JSON-RPC message, logs go
to stderr, malformed input gets a standard error instead of silence, piped input gets every reply,
and the process exits cleanly when stdin closes.

Control run against the SDK's plain ``stdio_server()`` with the same piped input: 11 of 20 replies,
no reply to malformed lines, and stray writes on stdout. This transport: 20 of 20, both errors, and
a clean stdout.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest
from mcp import Client, MCPError, StdioServerParameters
from mcp.types import INVALID_PARAMS, jsonrpc_message_adapter

from tests.mcp_server.edge_cases import (
    INVALID_AMOUNTS,
    INVALID_CUSTOMER_IDS,
    INVALID_REASONS,
    VALID_AMOUNTS,
    VALID_CUSTOMER_IDS,
    VALID_REASONS,
)

pytestmark = pytest.mark.e2e

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_MODULE = "ai_gateway.mcp_server"
TIMEOUT = 15.0

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"},
    },
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}
VALID_REFUND = {"customer_id": "CUST-00042", "amount": 25.0, "reason": "Duplicate charge"}


def tool_call(request_id: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


class ServerProcess:
    """A server subprocess with background readers, so a chatty stderr can never block it."""

    def __init__(self, module: str = SERVER_MODULE) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", module],
            cwd=PROJECT_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.stdout_lines: list[bytes] = []
        self._stderr_chunks: list[bytes] = []
        self._messages: queue.Queue[dict[str, Any] | None] = queue.Queue()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.stdout_lines.append(line)
            try:
                self._messages.put(json.loads(line))
            except ValueError:
                self._messages.put({"unparseable_stdout_line": line.decode(errors="replace")})
        self._messages.put(None)

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for chunk in self.proc.stderr:
            self._stderr_chunks.append(chunk)

    @property
    def stderr(self) -> str:
        return b"".join(self._stderr_chunks).decode(errors="replace")

    def send_line(self, line: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + b"\n")
        self.proc.stdin.flush()

    def send(self, message: dict[str, Any]) -> None:
        self.send_line(json.dumps(message).encode())

    def receive(self) -> dict[str, Any]:
        message = self._messages.get(timeout=TIMEOUT)
        assert message is not None, f"server closed stdout. stderr:\n{self.stderr}"
        return message

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        self.send(message)
        return self.receive()

    def close_stdin_and_wait(self) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        return self.proc.wait(timeout=TIMEOUT)

    def remaining_messages(self) -> list[dict[str, Any]]:
        messages = []
        while (message := self._messages.get(timeout=TIMEOUT)) is not None:
            messages.append(message)
        return messages

    def assert_stdout_is_pure_jsonrpc(self) -> None:
        for line in self.stdout_lines:
            jsonrpc_message_adapter.validate_json(line)  # raises on anything that is not JSON-RPC

    def __enter__(self) -> ServerProcess:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=TIMEOUT)


def test_full_session_keeps_stdout_pure_and_exits_cleanly() -> None:
    with ServerProcess() as server:
        initialized = server.request(INITIALIZE)
        assert initialized["result"]["protocolVersion"] == "2025-11-25"
        assert initialized["result"]["serverInfo"]["name"] == "customer-records"
        server.send(INITIALIZED)

        tools = server.request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert {tool["name"] for tool in tools["result"]["tools"]} == {
            "get_customer_record",
            "trigger_refund",
        }

        record = server.request(tool_call(2, "get_customer_record", {"customer_id": "CUST-00042"}))
        assert record["result"]["structuredContent"]["name"] == "Ada Lovelace"

        invalid = server.request(tool_call(3, "trigger_refund", {**VALID_REFUND, "amount": -5}))
        assert invalid["id"] == 3
        assert invalid["error"]["code"] == -32602
        assert invalid["error"]["data"]["errors"][0]["field"] == "amount"

        unknown_tool = server.request(tool_call(4, "drop_tables", {}))
        assert unknown_tool["error"] == {"code": -32602, "message": "Unknown tool: drop_tables"}

        unknown_method = server.request({"jsonrpc": "2.0", "id": 5, "method": "resources/list"})
        assert unknown_method["error"]["code"] == -32601

        assert server.request({"jsonrpc": "2.0", "id": 6, "method": "ping"})["result"] == {}

        assert server.close_stdin_and_wait() == 0

    server.assert_stdout_is_pure_jsonrpc()
    assert "Traceback" not in server.stderr


def test_malformed_input_gets_standard_errors_instead_of_silence() -> None:
    with ServerProcess() as server:
        server.request(INITIALIZE)
        server.send(INITIALIZED)

        def reply_to(line: bytes) -> dict[str, Any]:
            server.send_line(line)
            return server.receive()

        assert reply_to(b"this is not json")["error"]["code"] == -32700
        nan_call = b'{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"amount":NaN}}'
        assert reply_to(nan_call)["error"]["code"] == -32700
        assert (
            reply_to(b'{"jsonrpc":"2.0","id":1,"id":2,"method":"ping"}')["error"]["code"] == -32700
        )
        batch = reply_to(b'[{"jsonrpc":"2.0","id":11,"method":"ping"}]')
        assert (batch["id"], batch["error"]["code"]) == (None, -32600)
        for bad_id in (b"null", b"true", b"1.5"):
            reply = reply_to(b'{"jsonrpc":"2.0","id":' + bad_id + b',"method":"tools/list"}')
            assert (reply["id"], reply["error"]["code"]) == (None, -32600)
        reply = reply_to(b'{"jsonrpc":"2.0","id":12,"method":"tools/call","params":[1]}')
        assert (reply["id"], reply["error"]["code"]) == (12, -32602)
        reply = reply_to(b"x" * (1024 * 1024 + 1))
        assert (reply["id"], reply["error"]["code"]) == (None, -32600)

        server.send_line(b"")  # a blank line gets no reply...
        pong = server.request({"jsonrpc": "2.0", "id": 13, "method": "ping"})
        assert pong == {
            "jsonrpc": "2.0",
            "id": 13,
            "result": {},
        }  # ...so the next reply is the ping

        assert server.close_stdin_and_wait() == 0

    server.assert_stdout_is_pure_jsonrpc()


NON_STANDARD_JSON = {
    "NaN",
    "Infinity",
    "-Infinity",
}  # not JSON at all: the transport answers -32700

EDGE_CASE_TABLE = [
    *(("get_customer_record", "customer_id", case, False) for case in VALID_CUSTOMER_IDS),
    *(("get_customer_record", "customer_id", case, True) for case in INVALID_CUSTOMER_IDS),
    *(("trigger_refund", "amount", case, False) for case in VALID_AMOUNTS),
    *(("trigger_refund", "amount", case, True) for case in INVALID_AMOUNTS),
    *(("trigger_refund", "reason", case, False) for case in VALID_REASONS),
    *(("trigger_refund", "reason", case, True) for case in INVALID_REASONS),
]


def arguments_text(tool: str, field: str, value_json: str) -> str:
    """Raw JSON text for a tool's arguments, with one field's value spliced in verbatim."""
    fields = {"customer_id": '"CUST-00042"'}
    if tool == "trigger_refund":
        fields |= {"amount": "25.0", "reason": '"Duplicate charge on invoice INV-2041"'}
    fields[field] = value_json
    return "{" + ",".join(f'"{key}":{value}' for key, value in fields.items()) + "}"


def test_every_shared_edge_case_behaves_the_same_over_the_wire() -> None:
    with ServerProcess() as server:
        server.request(INITIALIZE)
        server.send(INITIALIZED)

        for request_id, (tool, field, case, invalid) in enumerate(EDGE_CASE_TABLE, start=100):
            label = f"{tool}.{field} {case.id}"
            envelope = f'{{"jsonrpc":"2.0","id":{request_id},"method":"tools/call","params":'
            params = f'{{"name":"{tool}","arguments":{arguments_text(tool, field, case.json)}}}'
            server.send_line((envelope + params + "}").encode())
            reply = server.receive()

            if case.json in NON_STANDARD_JSON:
                assert (reply["id"], reply["error"]["code"]) == (None, -32700), label
            elif invalid:
                assert reply["id"] == request_id, label
                assert reply["error"]["code"] == -32602, label
                assert [e["field"] for e in reply["error"]["data"]["errors"]] == [field], label
            else:
                assert reply["id"] == request_id, label
                assert "result" in reply, label  # valid input (an unknown customer is isError)

        assert server.close_stdin_and_wait() == 0

    server.assert_stdout_is_pure_jsonrpc()


def test_tool_call_before_initialize_is_rejected() -> None:
    with ServerProcess() as server:
        reply = server.request(tool_call(1, "get_customer_record", {"customer_id": "CUST-00042"}))
        assert reply["error"] == {
            "code": -32602,
            "message": "Invalid request parameters",
            "data": "",
        }
        assert server.close_stdin_and_wait() == 0


def test_piped_input_gets_every_reply_before_exit() -> None:
    calls = [
        tool_call(i, "get_customer_record", {"customer_id": "CUST-00042"}) for i in range(1, 21)
    ]
    payload = b"".join(json.dumps(m).encode() + b"\n" for m in [INITIALIZE, INITIALIZED, *calls])

    with ServerProcess() as server:
        assert server.proc.stdin is not None
        server.proc.stdin.write(payload)
        exit_code = server.close_stdin_and_wait()  # stdin closes right after the last request
        replies = server.remaining_messages()

    assert exit_code == 0
    assert sorted(reply["id"] for reply in replies) == list(range(21))
    assert all("result" in reply for reply in replies)


def test_stray_output_during_serving_never_reaches_stdout() -> None:
    with ServerProcess("tests.mcp_server.noisy_server") as server:
        server.request(INITIALIZE)
        server.send(INITIALIZED)
        reply = server.request(tool_call(1, "get_customer_record", {"customer_id": "CUST-00042"}))
        assert reply["result"]["structuredContent"]["customer_id"] == "CUST-00042"
        assert server.close_stdin_and_wait() == 0

    server.assert_stdout_is_pure_jsonrpc()
    assert "STRAY-WRITE-DURING-SERVING" in server.stderr
    assert "STRAY-WRITE-VIA-CAPTURED-REFERENCE" in server.stderr


@pytest.mark.parametrize(
    ("mode", "protocol_version"), [("legacy", "2025-11-25"), ("auto", "2026-07-28")]
)
async def test_official_client_over_real_stdio(mode: str, protocol_version: str) -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", SERVER_MODULE], cwd=str(PROJECT_ROOT)
    )
    async with Client(params, mode=mode) as client:
        assert client.protocol_version == protocol_version

        result = await client.call_tool("get_customer_record", {"customer_id": "CUST-00042"})
        assert result.structured_content["name"] == "Ada Lovelace"

        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("trigger_refund", {**VALID_REFUND, "amount": 0})
        assert excinfo.value.code == INVALID_PARAMS
