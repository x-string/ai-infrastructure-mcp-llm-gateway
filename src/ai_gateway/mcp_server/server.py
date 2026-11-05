"""Task 1 MCP server: a tool registry with strict validation and standard JSON-RPC errors.

The low-level ``mcp.server.Server`` is used on purpose. The high-level ``MCPServer`` reports
invalid arguments and unknown tools as ``isError`` tool results, ignores unknown keys, and echoes
input values. The brief requires standard JSON-RPC error codes instead.

Error mapping:

* arguments that fail the strict model: ``-32602 "Invalid params"`` with field-level details
* unknown tool: ``-32602 "Unknown tool: <name>"``
* valid arguments but no such customer: a tool execution error (``isError: true``)
* anything unexpected: ``-32603 "Internal error"``; the traceback goes to the log only
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, Final

from mcp import MCPError
from mcp.server import Server, ServerRequestContext
from mcp.types import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
    ToolAnnotations,
)
from pydantic import BaseModel, ValidationError

from ai_gateway.mcp_server.schemas import (
    CustomerRecord,
    GetCustomerRecordInput,
    RefundReceipt,
    TriggerRefundInput,
    validation_error_details,
)
from ai_gateway.mcp_server.store import CustomerNotFoundError, CustomerStore

logger = logging.getLogger(__name__)

SERVER_NAME: Final = "customer-records"
SERVER_VERSION: Final = "0.1.0"
SERVER_INSTRUCTIONS: Final = (
    "Customer support tools. Use get_customer_record to look up a customer by id "
    "(CUST- followed by five digits) and trigger_refund to issue a refund."
)

# Tool names that are safe to echo in an error message (MCP tool-name characters, short).
_ECHOABLE_TOOL_NAME = re.compile(r"[A-Za-z0-9_.\-]{1,64}")


@dataclass(frozen=True)
class ToolSpec:
    """A published tool definition, its strict input model and its business logic."""

    tool: Tool
    input_model: type[BaseModel]
    run: Callable[[Any], BaseModel]


def _hide_internal_errors[**P, R](
    handler: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """Turn unexpected exceptions into a generic ``-32603 Internal error``.

    Without this, the SDK answers stdio (handshake-era) clients with
    ``{"code": 0, "message": str(exc)}``, which uses a non-standard code and leaks the exception
    text. ``MCPError`` passes through untouched.
    """

    @wraps(handler)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await handler(*args, **kwargs)
        except MCPError:
            raise
        except Exception:
            logger.exception("Unhandled error in %s", handler.__name__)
            raise MCPError(INTERNAL_ERROR, "Internal error") from None

    return wrapper


def _unknown_tool_message(name: str) -> str:
    if _ECHOABLE_TOOL_NAME.fullmatch(name):
        return f"Unknown tool: {name}"
    return "Unknown tool"


def build_server(store: CustomerStore | None = None) -> Server:
    """Create the MCP server. Tests inject their own ``store``."""
    customers = store if store is not None else CustomerStore.with_demo_data()

    def get_customer_record(arguments: GetCustomerRecordInput) -> CustomerRecord:
        return customers.get(arguments.customer_id)

    def trigger_refund(arguments: TriggerRefundInput) -> RefundReceipt:
        return customers.record_refund(arguments.customer_id, arguments.amount, arguments.reason)

    specs = (
        ToolSpec(
            tool=Tool(
                name="get_customer_record",
                description=(
                    "Look up a customer's profile by customer id: 'CUST-' followed by five digits."
                ),
                input_schema=GetCustomerRecordInput.model_json_schema(),
                output_schema=CustomerRecord.model_json_schema(mode="serialization"),
                annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
            ),
            input_model=GetCustomerRecordInput,
            run=get_customer_record,
        ),
        ToolSpec(
            tool=Tool(
                name="trigger_refund",
                description=(
                    "Issue a refund to a customer. Requires the customer id, a positive amount "
                    "and a reason of 10 to 1,000 characters."
                ),
                input_schema=TriggerRefundInput.model_json_schema(),
                output_schema=RefundReceipt.model_json_schema(mode="serialization"),
                annotations=ToolAnnotations(
                    read_only_hint=False,
                    destructive_hint=False,
                    idempotent_hint=False,
                    open_world_hint=False,
                ),
            ),
            input_model=TriggerRefundInput,
            run=trigger_refund,
        ),
    )
    registry = {spec.tool.name: spec for spec in specs}

    @_hide_internal_errors
    async def list_tools(
        ctx: ServerRequestContext, params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        return ListToolsResult(tools=[spec.tool for spec in specs])

    @_hide_internal_errors
    async def call_tool(ctx: ServerRequestContext, params: CallToolRequestParams) -> CallToolResult:
        spec = registry.get(params.name)
        if spec is None:
            raise MCPError(INVALID_PARAMS, _unknown_tool_message(params.name))

        try:
            arguments = spec.input_model.model_validate(params.arguments or {})
        except ValidationError as exc:
            raise MCPError(
                INVALID_PARAMS,
                "Invalid params",
                data={"tool": spec.tool.name, "errors": validation_error_details(exc)},
            ) from None

        try:
            result = spec.run(arguments)
        except CustomerNotFoundError as exc:
            return CallToolResult(content=[TextContent(type="text", text=str(exc))], is_error=True)

        payload = result.model_dump(mode="json")
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload))],
            structured_content=payload,
        )

    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions=SERVER_INSTRUCTIONS,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )
