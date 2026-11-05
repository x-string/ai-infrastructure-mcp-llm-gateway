"""Authorization for the MCP security gateway: which methods and tools each role may use.

The policy is a pure function of the authenticated principal and the parsed message, so every
decision can be tested without HTTP.

* Allowlisted methods only: the lifecycle requests (``initialize``, ``ping``, ``server/discover``),
  ``tools/list`` (forwarded unfiltered, as the brief requires) and ``tools/call``. Everything else
  is ``-32601`` and never reaches the downstream server.
* ``tools/call`` needs a tool name made of MCP tool-name characters (``[A-Za-z0-9_.-]{1,128}``),
  which rules out whitespace and Unicode look-alike tricks. A name that starts with ``admin_``,
  compared case-insensitively so ``ADMIN_x`` fails closed, requires the ``admin`` role. Otherwise
  the call is answered with ``-32001 Unauthorized Tool Call``.
* Only lifecycle notifications pass. A ``tools/call`` sent without an ``id`` would run with no
  reply, so it is rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from mcp.types import INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND

from ai_gateway.mcp_gateway.auth import Principal, Role
from ai_gateway.mcp_gateway.jsonrpc import Message, Notification

UNAUTHORIZED_TOOL_CALL: Final = -32001  # the code the brief requires
ADMIN_TOOL_PREFIX: Final = "admin_"
TOOL_NAME: Final = re.compile(r"[A-Za-z0-9_.\-]{1,128}")

OPEN_REQUESTS: Final = frozenset({"initialize", "ping", "server/discover", "tools/list"})
FORWARDED_REQUESTS: Final = OPEN_REQUESTS | {"tools/call"}
FORWARDED_NOTIFICATIONS: Final = frozenset({"notifications/initialized", "notifications/cancelled"})


@dataclass(frozen=True)
class Allow:
    pass


ALLOW: Final = Allow()


@dataclass(frozen=True)
class Deny:
    code: int
    message: str
    data: dict[str, Any] | None = None


def authorize(principal: Principal, message: Message) -> Allow | Deny:
    """Decide whether ``principal`` may send ``message`` to the downstream server."""
    if isinstance(message, Notification):
        if message.method in FORWARDED_NOTIFICATIONS:
            return ALLOW
        if message.method in FORWARDED_REQUESTS:
            return Deny(
                INVALID_REQUEST,
                "Invalid Request",
                {"reason": f"{message.method} must be sent as a request with an id"},
            )
        return Deny(METHOD_NOT_FOUND, "Method not found", {"reason": "notification not allowed"})

    if message.method in OPEN_REQUESTS:
        return ALLOW
    if message.method == "tools/call":
        return _authorize_tool_call(principal, message.params)
    return Deny(METHOD_NOT_FOUND, "Method not found", {"reason": "method not allowed"})


def _authorize_tool_call(principal: Principal, params: dict[str, Any] | None) -> Allow | Deny:
    name = (params or {}).get("name")
    if not isinstance(name, str) or not TOOL_NAME.fullmatch(name):
        return Deny(
            INVALID_PARAMS,
            "Invalid params",
            {"reason": "tool name must match [A-Za-z0-9_.-]{1,128}"},
        )
    if name.lower().startswith(ADMIN_TOOL_PREFIX) and principal.role is not Role.ADMIN:
        return Deny(
            UNAUTHORIZED_TOOL_CALL,
            "Unauthorized Tool Call",
            {"reason": "forbidden_tool", "tool": name, "required_role": Role.ADMIN.value},
        )
    return ALLOW
