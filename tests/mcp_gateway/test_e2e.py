"""Task 2 end to end: official MCP client -> real gateway -> real SDK downstream, all over TCP.

The gateway and the downstream run as subprocesses started from their entry points. The downstream
runs once in JSON mode and once in SSE mode, and the client speaks both protocol eras: ``legacy``
(2025-11-25 handshake) and ``auto`` (stateless 2026-07-28).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import httpx2
import pytest
from mcp import Client, MCPError
from mcp.client.streamable_http import streamable_http_client

from ai_gateway.mcp_gateway.auth import Role, TokenSettings, issue_token

pytestmark = pytest.mark.e2e

SECRET = "e2e-gateway-secret-0123456789abcdefgh"
TOKENS = TokenSettings(secret=SECRET)
VIEWER = issue_token(TOKENS, "alice", Role.VIEWER)
ADMIN = issue_token(TOKENS, "root", Role.ADMIN)
ERAS = {"legacy": "2025-11-25", "auto": "2026-07-28"}
ALL_TOOLS = ["admin_delete_user", "admin_reset_key", "echo", "get_status", "slow_echo"]
MODERN_META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until_listening(port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"nothing is listening on port {port}")


@dataclass
class Stack:
    mode: str
    url: str
    calls_url: str
    gateway_log: Path

    def downstream_events(self) -> list[dict[str, str]]:
        return httpx2.get(self.calls_url).json()

    def gateway_audit(self) -> str:
        return self.gateway_log.read_text()


@pytest.fixture(scope="module", params=["json", "sse"])
def stack(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Stack]:
    mode: str = request.param
    downstream_port, gateway_port = free_port(), free_port()
    gateway_log = tmp_path_factory.mktemp(f"gateway-{mode}") / "gateway.log"
    downstream_args = ["--port", str(downstream_port), "--expose-calls"]
    if mode == "sse":
        downstream_args.append("--sse")
    gateway_env = {
        **os.environ,
        "GATEWAY_JWT_SECRET": SECRET,
        "GATEWAY_PORT": str(gateway_port),
        "GATEWAY_DOWNSTREAM_URL": f"http://127.0.0.1:{downstream_port}/mcp",
    }
    with gateway_log.open("wb") as log_file:
        processes = [
            subprocess.Popen(
                [sys.executable, "-m", "ai_gateway.mcp_gateway.mock_downstream", *downstream_args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ),
            subprocess.Popen(
                [sys.executable, "-m", "ai_gateway.mcp_gateway"],
                env=gateway_env,
                stdout=subprocess.DEVNULL,
                stderr=log_file,
            ),
        ]
    try:
        wait_until_listening(downstream_port)
        wait_until_listening(gateway_port)
        yield Stack(
            mode=mode,
            url=f"http://127.0.0.1:{gateway_port}/mcp",
            calls_url=f"http://127.0.0.1:{downstream_port}/_calls",
            gateway_log=gateway_log,
        )
    finally:
        for process in processes:
            process.terminate()
            process.wait(timeout=10)


@asynccontextmanager
async def connect(stack: Stack, token: str, mode: str) -> AsyncIterator[Client]:
    """The official MCP client over Streamable HTTP, carrying a bearer token."""
    headers = {"authorization": f"Bearer {token}"}
    async with httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(30, read=60)) as http:
        async with Client(streamable_http_client(stack.url, http_client=http), mode=mode) as client:
            assert client.protocol_version == ERAS[mode]
            yield client


def tool_events(stack: Stack, tool: str) -> list[str]:
    return [event["event"] for event in stack.downstream_events() if event["tool"] == tool]


ERA = pytest.mark.parametrize("mode", list(ERAS))


@ERA
async def test_tools_list_passes_through_unfiltered(stack: Stack, mode: str) -> None:
    async with connect(stack, VIEWER, mode) as client:
        tools = sorted(tool.name for tool in (await client.list_tools()).tools)
    assert tools == ALL_TOOLS


@ERA
async def test_viewer_can_call_ordinary_tools(stack: Stack, mode: str) -> None:
    async with connect(stack, VIEWER, mode) as client:
        result = await client.call_tool("echo", {"text": "hello"})
    assert result.content[0].text == "hello"


@ERA
async def test_viewer_admin_call_is_denied_and_never_reaches_the_downstream(
    stack: Stack, mode: str
) -> None:
    before = tool_events(stack, "admin_reset_key")
    async with connect(stack, VIEWER, mode) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("admin_reset_key", {"key_id": "k-1"})

    assert excinfo.value.code == -32001
    assert excinfo.value.message == "Unauthorized Tool Call"
    assert excinfo.value.data == {
        "reason": "forbidden_tool",
        "tool": "admin_reset_key",
        "required_role": "admin",
    }
    assert tool_events(stack, "admin_reset_key") == before


@ERA
async def test_admin_can_call_admin_tools(stack: Stack, mode: str) -> None:
    before = len(tool_events(stack, "admin_reset_key"))
    async with connect(stack, ADMIN, mode) as client:
        result = await client.call_tool("admin_reset_key", {"key_id": "k-1"})
    assert result.content[0].text == "Key k-1 rotated"
    assert len(tool_events(stack, "admin_reset_key")) == before + 1


async def test_requests_without_a_valid_token_are_stopped_at_the_gateway(stack: Stack) -> None:
    before = stack.downstream_events()
    forged = issue_token(
        TokenSettings(secret="not-the-gateway-secret-0123456789ab"), "eve", Role.ADMIN
    )
    call = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "admin_reset_key", "arguments": {"key_id": "k-1"}},
    }
    async with httpx2.AsyncClient() as http:
        missing = await http.post(stack.url, json=call)
        invalid = await http.post(
            stack.url, json=call, headers={"authorization": f"Bearer {forged}"}
        )

    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == 'Bearer realm="mcp-gateway"'
    assert invalid.status_code == 401
    assert 'error="invalid_token"' in invalid.headers["www-authenticate"]
    assert stack.downstream_events() == before


@pytest.mark.slow
async def test_tools_slower_than_five_seconds_are_not_cut_off(stack: Stack) -> None:
    # A default HTTP client would time out after 5 s of silence; the gateway waits for the tool.
    async with connect(stack, VIEWER, "auto") as client:
        result = await client.call_tool("slow_echo", {"text": "done", "seconds": 5.5})
    assert result.content[0].text == "done"


def raw_slow_call(era: str) -> tuple[dict[str, Any], dict[str, str]]:
    params: dict[str, Any] = {"name": "slow_echo", "arguments": {"text": "x", "seconds": 8}}
    headers = {
        "authorization": f"Bearer {VIEWER}",
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
    }
    if era == "modern":
        params["_meta"] = MODERN_META
        headers |= {
            "mcp-protocol-version": "2026-07-28",
            "mcp-method": "tools/call",
            "mcp-name": "slow_echo",
        }
    return {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params}, headers


@pytest.mark.slow
@pytest.mark.parametrize("era", ["legacy", "modern"])
async def test_client_disconnect_is_passed_on_to_the_downstream(stack: Stack, era: str) -> None:
    body, headers = raw_slow_call(era)
    cancelled_before = tool_events(stack, "slow_echo").count("cancelled")
    gave_up_before = stack.gateway_audit().count("decision=client_disconnected")

    async with httpx2.AsyncClient() as http:
        with anyio.move_on_after(1.0):  # the client hangs up while the tool is still running
            await http.post(stack.url, json=body, headers=headers)

    with anyio.fail_after(10):
        if stack.mode == "sse":
            # The SDK cancels a tool when its event stream closes, so the hang-up reaches the tool.
            while tool_events(stack, "slow_echo").count("cancelled") == cancelled_before:
                await anyio.sleep(0.1)
        else:
            # In JSON mode the SDK does not watch for disconnects, so the tool runs to completion.
            # The gateway's part, closing its own downstream request, shows up in its audit log.
            while stack.gateway_audit().count("decision=client_disconnected") == gave_up_before:
                await anyio.sleep(0.1)
