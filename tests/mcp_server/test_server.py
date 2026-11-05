"""Task 1: the MCP server, exercised through the official MCP client in memory.

``legacy`` mode uses real JSON-RPC framing and the initialize handshake (what stdio hosts such as
Claude Desktop or MCP Inspector do). ``auto`` mode takes the stateless 2026-07-28 path. Raw-wire
behavior (malformed lines, invalid ids, stdout purity) is covered by the stdio tests in Step 3.

Every error test runs in both modes on purpose. Without the server's error wrapper, ``legacy`` mode
leaks unexpected exceptions as ``{"code": 0, "message": str(exc)}``, while ``auto`` mode hides them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from mcp import Client, MCPError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS

from ai_gateway.mcp_server.schemas import (
    CustomerRecord,
    GetCustomerRecordInput,
    RefundReceipt,
    TriggerRefundInput,
)
from ai_gateway.mcp_server.server import build_server
from ai_gateway.mcp_server.store import CustomerStore
from tests.mcp_server.edge_cases import (
    INVALID_AMOUNTS,
    INVALID_CUSTOMER_IDS,
    INVALID_REASONS,
    Case,
)

MODES = ["legacy", "auto"]

VALID_REFUND: dict[str, Any] = {
    "customer_id": "CUST-00042",
    "amount": 25.0,
    "reason": "Duplicate charge on invoice INV-2041",
}


@asynccontextmanager
async def connect(mode: str, store: CustomerStore | None = None) -> AsyncIterator[Client]:
    async with Client(build_server(store), mode=mode) as client:
        yield client


def cases_by_mode(cases: list[Case]) -> list[Any]:
    return [
        pytest.param(mode, json.loads(case.json), id=f"{mode}-{case.id}")
        for mode in MODES
        for case in cases
    ]


def assert_invalid_params(error: MCPError, tool: str, fields: list[str]) -> None:
    assert error.code == INVALID_PARAMS
    assert error.message == "Invalid params"
    assert error.data["tool"] == tool
    assert [detail["field"] for detail in error.data["errors"]] == fields
    json.dumps(error.data)  # error data must always be JSON-serializable


# --- tools/list --------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
async def test_tools_list_publishes_the_enforced_schemas(mode: str) -> None:
    async with connect(mode) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {"get_customer_record", "trigger_refund"}
    assert tools["get_customer_record"].input_schema == GetCustomerRecordInput.model_json_schema()
    assert tools["trigger_refund"].input_schema == TriggerRefundInput.model_json_schema()
    assert tools["get_customer_record"].output_schema == CustomerRecord.model_json_schema(
        mode="serialization"
    )
    assert tools["trigger_refund"].output_schema == RefundReceipt.model_json_schema(
        mode="serialization"
    )
    assert tools["get_customer_record"].annotations.read_only_hint is True
    assert tools["trigger_refund"].annotations.read_only_hint is False


# --- successful calls --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
async def test_get_customer_record_returns_a_structured_record(mode: str) -> None:
    async with connect(mode) as client:
        result = await client.call_tool("get_customer_record", {"customer_id": "CUST-00042"})

    assert result.is_error is False
    assert result.structured_content == {
        "customer_id": "CUST-00042",
        "name": "Ada Lovelace",
        "email": "ada.lovelace@example.com",
        "tier": "premium",
        "status": "active",
    }
    assert json.loads(result.content[0].text) == result.structured_content


@pytest.mark.parametrize("mode", MODES)
async def test_trigger_refund_returns_a_receipt_and_records_the_refund(mode: str) -> None:
    store = CustomerStore.with_demo_data()
    async with connect(mode, store) as client:
        result = await client.call_tool("trigger_refund", VALID_REFUND)

    receipt = result.structured_content
    assert result.is_error is False
    assert receipt["refund_id"].startswith("rf_")
    assert receipt["customer_id"] == "CUST-00042"
    assert receipt["amount"] == 25.0
    assert receipt["status"] == "accepted"
    assert [refund.refund_id for refund in store.refunds] == [receipt["refund_id"]]


# --- business errors are tool execution errors, not protocol errors -----------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tool", ["get_customer_record", "trigger_refund"])
async def test_unknown_customer_is_a_tool_execution_error(mode: str, tool: str) -> None:
    arguments = {**VALID_REFUND, "customer_id": "CUST-99999"}
    if tool == "get_customer_record":
        arguments = {"customer_id": "CUST-99999"}

    async with connect(mode) as client:
        result = await client.call_tool(tool, arguments)

    assert result.is_error is True
    assert result.content[0].text == "No customer with id CUST-99999."


# --- invalid arguments are JSON-RPC -32602 errors ------------------------------------------------


@pytest.mark.parametrize(("mode", "customer_id"), cases_by_mode(INVALID_CUSTOMER_IDS))
async def test_invalid_customer_id_is_invalid_params(mode: str, customer_id: Any) -> None:
    async with connect(mode) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("get_customer_record", {"customer_id": customer_id})
    assert_invalid_params(excinfo.value, "get_customer_record", ["customer_id"])


@pytest.mark.parametrize(("mode", "amount"), cases_by_mode(INVALID_AMOUNTS))
async def test_invalid_amount_is_invalid_params(mode: str, amount: Any) -> None:
    async with connect(mode) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("trigger_refund", {**VALID_REFUND, "amount": amount})
    assert_invalid_params(excinfo.value, "trigger_refund", ["amount"])


@pytest.mark.parametrize(("mode", "reason"), cases_by_mode(INVALID_REASONS))
async def test_invalid_reason_is_invalid_params(mode: str, reason: Any) -> None:
    async with connect(mode) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("trigger_refund", {**VALID_REFUND, "reason": reason})
    assert_invalid_params(excinfo.value, "trigger_refund", ["reason"])


@pytest.mark.parametrize("mode", MODES)
async def test_unknown_argument_is_invalid_params(mode: str) -> None:
    async with connect(mode) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("trigger_refund", {**VALID_REFUND, "approved_by": "me"})
    assert_invalid_params(excinfo.value, "trigger_refund", ["approved_by"])


@pytest.mark.parametrize("mode", MODES)
async def test_missing_arguments_are_invalid_params(mode: str) -> None:
    async with connect(mode) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("trigger_refund")
    assert_invalid_params(excinfo.value, "trigger_refund", ["customer_id", "amount", "reason"])


@pytest.mark.parametrize("mode", MODES)
async def test_unknown_tool_is_invalid_params(mode: str) -> None:
    async with connect(mode) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("delete_all_customers", {})
    assert excinfo.value.code == INVALID_PARAMS
    assert excinfo.value.message == "Unknown tool: delete_all_customers"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ["x" * 65, "bad name", "evil\nname", "<script>"])
async def test_unsafe_unknown_tool_names_are_not_echoed(mode: str, name: str) -> None:
    async with connect(mode) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool(name, {})
    assert excinfo.value.code == INVALID_PARAMS
    assert excinfo.value.message == "Unknown tool"


# --- unexpected failures never leak details -------------------------------------------------


class BrokenStore(CustomerStore):
    """A store whose backend fails with a message that must never reach the client."""

    def get(self, customer_id: str) -> CustomerRecord:
        raise RuntimeError("connection to db://admin:hunter2@10.0.0.5 refused")


@pytest.mark.parametrize("mode", MODES)
async def test_unexpected_errors_become_a_generic_internal_error(
    mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger="ai_gateway.mcp_server.server"):
        async with connect(mode, BrokenStore([])) as client:
            with pytest.raises(MCPError) as excinfo:
                await client.call_tool("get_customer_record", {"customer_id": "CUST-00042"})

    assert excinfo.value.code == INTERNAL_ERROR
    assert excinfo.value.message == "Internal error"
    assert excinfo.value.data is None
    assert "hunter2" not in str(excinfo.value)
    # The details stay in the server log (stderr in production), not on the wire.
    assert "hunter2" in caplog.text
