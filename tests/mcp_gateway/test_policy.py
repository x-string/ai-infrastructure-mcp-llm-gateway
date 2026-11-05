"""Task 2: method allowlist and tool-level authorization."""

from __future__ import annotations

from typing import Any

import pytest
from mcp.types import INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND

from ai_gateway.mcp_gateway.auth import Principal, Role
from ai_gateway.mcp_gateway.jsonrpc import Notification, Request
from ai_gateway.mcp_gateway.policy import ALLOW, UNAUTHORIZED_TOOL_CALL, Deny, authorize

ADMIN = Principal(subject="root", role=Role.ADMIN)
VIEWER = Principal(subject="alice", role=Role.VIEWER)
BOTH_ROLES = pytest.mark.parametrize("principal", [ADMIN, VIEWER], ids=["admin", "viewer"])


def call(name: object) -> Request:
    return Request(id=1, method="tools/call", params={"name": name, "arguments": {}})


def denied_code(principal: Principal, message: Request | Notification) -> int:
    decision = authorize(principal, message)
    assert isinstance(decision, Deny), decision
    return decision.code


# --- open to every authenticated role -------------------------------------------------------------


@BOTH_ROLES
@pytest.mark.parametrize("method", ["initialize", "ping", "server/discover", "tools/list"])
def test_lifecycle_and_tool_listing_are_open_to_every_role(
    principal: Principal, method: str
) -> None:
    assert authorize(principal, Request(id=1, method=method, params=None)) is ALLOW


NON_ADMIN_TOOLS = [
    "get_customer_record",
    "echo",
    "administrator",
    "get_admin_report",
    "admin",
    "adminreset",
    "x.admin_y",
    "tool-1.v2",
]


@BOTH_ROLES
@pytest.mark.parametrize("name", NON_ADMIN_TOOLS)
def test_non_admin_tools_are_open_to_every_role(principal: Principal, name: str) -> None:
    assert authorize(principal, call(name)) is ALLOW


# --- admin_* tools --------------------------------------------------------------------------------

ADMIN_TOOLS = [
    "admin_reset_key",
    "admin_delete_user",
    "ADMIN_reset_key",
    "Admin_Reset_Key",
    "aDmIn_x",
    "admin_",
]


@pytest.mark.parametrize("name", ADMIN_TOOLS)
def test_admin_tools_are_denied_to_viewers_with_unauthorized_tool_call(name: str) -> None:
    assert authorize(VIEWER, call(name)) == Deny(
        code=UNAUTHORIZED_TOOL_CALL,
        message="Unauthorized Tool Call",
        data={"reason": "forbidden_tool", "tool": name, "required_role": "admin"},
    )


@pytest.mark.parametrize("name", ADMIN_TOOLS)
def test_admin_tools_are_allowed_for_admins(name: str) -> None:
    assert authorize(ADMIN, call(name)) is ALLOW


def test_unauthorized_tool_call_code_is_the_one_the_brief_requires() -> None:
    assert UNAUTHORIZED_TOOL_CALL == -32001


# --- tool names outside the MCP charset are rejected before the admin check -----------------------

MALFORMED_TOOL_PARAMS: list[Any] = [
    {"name": " admin_reset_key"},
    {"name": "admin_reset_key "},
    {"name": "admin_reset_key\n"},
    {"name": "admin reset"},
    {"name": "admin_reset_key​"},
    {"name": "ａｄｍｉｎ_reset_key"},  # fullwidth "admin"
    {"name": "x" * 129},
    {"name": ""},
    {"name": 42},
    {"name": None},
    {"name": ["admin_reset_key"]},
    {"name": {"value": "admin_reset_key"}},
    {},
    None,
]


@BOTH_ROLES
@pytest.mark.parametrize("params", MALFORMED_TOOL_PARAMS)
def test_malformed_tool_names_are_invalid_params(principal: Principal, params: Any) -> None:
    request = Request(id=1, method="tools/call", params=params)
    assert denied_code(principal, request) == INVALID_PARAMS


# --- everything else is denied --------------------------------------------------------------------


@BOTH_ROLES
@pytest.mark.parametrize(
    "method",
    [
        "resources/list",
        "resources/read",
        "prompts/get",
        "logging/setLevel",
        "completion/complete",
        "tools/call ",
        "TOOLS/CALL",
        "Tools/List",
        "tools/list/",
        "rpc.discover",
    ],
)
def test_methods_outside_the_allowlist_are_not_found(principal: Principal, method: str) -> None:
    assert denied_code(principal, Request(id=1, method=method, params=None)) == METHOD_NOT_FOUND


# --- notifications --------------------------------------------------------------------------------


@BOTH_ROLES
@pytest.mark.parametrize("method", ["notifications/initialized", "notifications/cancelled"])
def test_lifecycle_notifications_pass(principal: Principal, method: str) -> None:
    assert authorize(principal, Notification(method=method, params=None)) is ALLOW


@BOTH_ROLES
@pytest.mark.parametrize("method", ["tools/call", "tools/list", "initialize", "ping"])
def test_requests_sent_as_notifications_are_invalid(principal: Principal, method: str) -> None:
    # A tools/call notification would run with no reply, so it is never forwarded.
    notification = Notification(method=method, params={"name": "admin_reset_key"})
    assert denied_code(principal, notification) == INVALID_REQUEST


@pytest.mark.parametrize("method", ["notifications/progress", "notifications/roots/list_changed"])
def test_other_notifications_are_not_forwarded(method: str) -> None:
    assert denied_code(VIEWER, Notification(method=method, params=None)) == METHOD_NOT_FOUND
