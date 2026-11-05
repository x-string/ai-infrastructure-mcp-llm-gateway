"""Strict JSON-RPC handling for the MCP security gateway.

The gateway authorizes what it parses and forwards exactly what it authorized. Parsing is
therefore stricter than ordinary JSON:

* Duplicate keys are rejected. Python keeps the last value and other parsers may keep the first,
  so ``{"name": "get_status", "name": "admin_reset_key"}`` could be authorized as one tool and
  executed as another.
* ``NaN``, ``Infinity``, numbers that overflow, lone surrogates, a byte order mark, invalid UTF-8,
  huge integers and deep nesting are all rejected.
* The body must be exactly one request or one notification. Batches are rejected, and so are
  client "responses": the SDK downstream ignores unknown keys, so
  ``{"result": {}, "method": "tools/call", ...}`` would be executed as a request.

Forwarded bodies are rebuilt from the parsed fields, never copied from the raw bytes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from mcp.shared.inbound import (
    ERROR_CODE_HTTP_STATUS,
    MCP_PROTOCOL_VERSION_HEADER,
    InboundLadderRejection,
    classify_inbound_request,
    find_duplicated_routing_header,
)
from mcp.types import HEADER_MISMATCH, INVALID_PARAMS, INVALID_REQUEST, PARSE_ERROR
from mcp.types.version import HANDSHAKE_PROTOCOL_VERSIONS

type RequestId = str | int

_ENVELOPE_KEYS: Final = frozenset({"jsonrpc", "id", "method", "params"})


@dataclass(frozen=True)
class Request:
    id: RequestId
    method: str
    params: dict[str, Any] | None


@dataclass(frozen=True)
class Notification:
    method: str
    params: dict[str, Any] | None


type Message = Request | Notification


class JsonRpcError(Exception):
    """A message the gateway rejects before authorization, with everything needed to answer it."""

    def __init__(
        self,
        code: int,
        message: str,
        reason: str,
        *,
        request_id: RequestId | None = None,
        data: Any = None,
    ) -> None:
        super().__init__(reason)
        self.code = code
        self.message = message
        self.reason = reason
        self.request_id = request_id
        self.data = data if data is not None else {"reason": reason}


class _Rejected(ValueError):
    """Raised by the strict JSON hooks."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise _Rejected("duplicate key in object")
        obj[key] = value
    return obj


def _reject_constant(name: str) -> Any:
    raise _Rejected(f"non-standard JSON constant {name}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _Rejected("number out of range")
    return value


def _parse_error(reason: str) -> JsonRpcError:
    return JsonRpcError(PARSE_ERROR, "Parse error", reason)


def _invalid_request(reason: str, request_id: RequestId | None = None) -> JsonRpcError:
    return JsonRpcError(INVALID_REQUEST, "Invalid Request", reason, request_id=request_id)


def _valid_id(value: object) -> RequestId | None:
    """MCP request ids are strings or integers, never null, booleans or floats."""
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _decode_json(body: bytes) -> Any:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        raise _parse_error("body is not valid UTF-8") from None
    if text.startswith("﻿"):
        raise _parse_error("body starts with a byte order mark")
    try:
        obj = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except _Rejected as exc:
        raise _parse_error(str(exc)) from None
    except (ValueError, RecursionError):
        raise _parse_error("body is not valid JSON") from None
    try:
        json.dumps(obj, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        raise _parse_error("body contains an unpaired surrogate escape") from None
    return obj


def parse_message(body: bytes) -> Message:
    """Parse an HTTP body into exactly one request or notification, or raise ``JsonRpcError``."""
    obj = _decode_json(body)
    if isinstance(obj, list):
        raise _invalid_request("batch requests are not supported")
    if not isinstance(obj, dict):
        raise _invalid_request("body must be a JSON object")

    request_id = _valid_id(obj.get("id"))
    if "id" in obj and request_id is None:
        raise _invalid_request("id must be a string or an integer")
    if not obj.keys() <= _ENVELOPE_KEYS:
        raise _invalid_request("body must be a single request or notification", request_id)
    if obj.get("jsonrpc") != "2.0":
        raise _invalid_request('jsonrpc must be "2.0"', request_id)
    method = obj.get("method")
    if not isinstance(method, str) or not method:
        raise _invalid_request("method must be a non-empty string", request_id)

    params = obj.get("params")
    if "params" in obj and not isinstance(params, dict):
        if request_id is None:
            raise _invalid_request("params must be an object")
        raise JsonRpcError(
            INVALID_PARAMS, "Invalid params", "params must be an object", request_id=request_id
        )

    if request_id is None:
        return Notification(method=method, params=params)
    return Request(id=request_id, method=method, params=params)


def _dumps(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def encode_message(message: Message) -> bytes:
    """The bytes to forward: rebuilt from the parsed fields, ASCII-only, nothing else."""
    body: dict[str, Any] = {"jsonrpc": "2.0"}
    if isinstance(message, Request):
        body["id"] = message.id
    body["method"] = message.method
    if message.params is not None:
        body["params"] = message.params
    return _dumps(body)


def error_body(
    code: int,
    message: str,
    *,
    request_id: RequestId | None = None,
    data: Any = None,
    include_id: bool = True,
) -> bytes:
    """A JSON-RPC error response. ``include_id=False`` is for rejected notifications."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    body: dict[str, Any] = {"jsonrpc": "2.0"}
    if include_id:
        body["id"] = request_id
    body["error"] = error
    return _dumps(body)


def http_status_for(code: int, *, modern: bool) -> int:
    """HTTP status for a gateway-generated JSON-RPC error.

    Malformed messages are always 400. Otherwise legacy clients (2025-11-25 and earlier) get
    JSON-RPC errors in-band with 200, and modern clients get the SDK's own status table (for
    example ``-32601`` is 404). ``-32001`` stays 200 in both eras, so the official client surfaces
    it as an ``MCPError`` instead of starting an OAuth step-up.
    """
    if code in (PARSE_ERROR, INVALID_REQUEST, HEADER_MISMATCH):
        return 400
    if modern:
        return ERROR_CODE_HTTP_STATUS.get(code, 200)
    return 200


def is_modern(raw_headers: Sequence[tuple[str, str]]) -> bool:
    """True for 2026-07-28+ clients: an ``MCP-Protocol-Version`` that is not a handshake version.

    The first copy of the header decides, as it does in the SDK downstream.
    """
    for name, value in raw_headers:
        if name.lower() == MCP_PROTOCOL_VERSION_HEADER:
            return value not in HANDSHAKE_PROTOCOL_VERSIONS
    return False


def check_routing_headers(raw_headers: Sequence[tuple[str, str]], message: Message) -> None:
    """Modern clients mirror the method and tool name in headers. They must agree with the body.

    The spec requires this of any intermediary that reads the body. The SDK's own checks are
    reused, so the gateway enforces exactly what the SDK downstream enforces.
    """
    if not is_modern(raw_headers):
        return
    request_id = message.id if isinstance(message, Request) else None
    duplicated = find_duplicated_routing_header(raw_headers)
    if duplicated is not None:
        raise JsonRpcError(
            HEADER_MISMATCH,
            "Header mismatch",
            f"the {duplicated} header was sent more than once",
            request_id=request_id,
        )
    if not isinstance(message, Request):
        return

    headers: dict[str, str] = {}
    for name, value in raw_headers:
        headers.setdefault(name.lower(), value)
    verdict = classify_inbound_request(
        {"method": message.method, "params": message.params}, headers=headers
    )
    if isinstance(verdict, InboundLadderRejection):
        raise JsonRpcError(
            verdict.code,
            verdict.message,
            verdict.message,
            request_id=request_id,
            data=verdict.data,
        )
