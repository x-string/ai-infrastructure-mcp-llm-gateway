"""Validating stdio transport for the Task 1 MCP server.

The SDK's ``stdio_server()`` keeps stdout clean: it writes the protocol through a private duplicate
of fd 1 and points fd 1 at stderr while serving. Its stdin reader is too lenient for this server:

* lines that are not JSON, or not JSON-RPC, are logged at debug level and dropped, so the client
  waits forever for a reply;
* requests whose ``id`` is ``null``, a boolean or a float are parsed as notifications and dropped;
* when stdin closes, in-flight handlers are cancelled and replies still queued for stdout are lost,
  which bites any host that pipes a batch of requests and then closes stdin.

This module keeps the SDK's stdout handling and single writer, but reads stdin itself:

1. Each raw line is validated strictly. Unusable lines are answered with ``-32700``, ``-32600`` or
   ``-32602``, echoing the ``id`` when it is valid and using ``null`` otherwise.
2. Valid messages are handed to the server unchanged.
3. At end of input, pending requests get up to ``DRAIN_TIMEOUT_SECONDS`` to be answered before the
   server sees EOF.

While serving, ``sys.stdout`` is redirected to stderr. The original stream is flushed before fd 1 is
restored, so even buffered stray output cannot reach the protocol stream.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any, BinaryIO, Final, Self, cast

import anyio
from anyio.abc import ObjectSendStream
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage
from mcp.types import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    PARSE_ERROR,
    ErrorData,
    JSONRPCError,
    JSONRPCMessage,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    jsonrpc_message_adapter,
)
from pydantic import ValidationError

if sys.platform != "win32":
    import fcntl

logger = logging.getLogger(__name__)

MAX_LINE_BYTES: Final = 1024 * 1024
DRAIN_TIMEOUT_SECONDS: Final = 1.5  # below the official client's 2-second shutdown grace period

type RequestId = str | int


@dataclass(frozen=True)
class Forward:
    """A valid message for the server."""

    message: JSONRPCMessage


@dataclass(frozen=True)
class Reply:
    """An error the transport answers itself. The server never sees the line."""

    error: JSONRPCError


class _LineTooLong:
    def __repr__(self) -> str:
        return "LINE_TOO_LONG"


LINE_TOO_LONG: Final = _LineTooLong()


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


def _reply(code: int, message: str, reason: str, request_id: RequestId | None = None) -> Reply:
    error = ErrorData(code=code, message=message, data={"reason": reason})
    return Reply(JSONRPCError(jsonrpc="2.0", id=request_id, error=error))


def _parse_error(reason: str) -> Reply:
    return _reply(PARSE_ERROR, "Parse error", reason)


def _invalid_request(reason: str, request_id: RequestId | None = None) -> Reply:
    return _reply(INVALID_REQUEST, "Invalid Request", reason, request_id)


def _valid_id(value: object) -> RequestId | None:
    """MCP request ids are strings or integers (never null, booleans or floats)."""
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def parse_line(line: bytes) -> Forward | Reply | None:
    """Classify one raw stdin line: forward it, answer it with an error, or ignore it (``None``).

    Error replies never echo the input; they carry only a short reason.
    """
    if not line.strip():
        return None
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError:
        return _parse_error("message is not valid UTF-8")
    try:
        obj = json.loads(
            text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
        )
    except _Rejected as exc:
        return _parse_error(str(exc))
    except (ValueError, RecursionError):
        return _parse_error("message is not valid JSON")
    try:
        json.dumps(obj, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        return _parse_error("message contains an unpaired surrogate escape")

    if not isinstance(obj, dict):
        if isinstance(obj, list):
            return _invalid_request("batch requests are not supported")
        return _invalid_request("message must be a JSON object")

    request_id = _valid_id(obj.get("id"))
    if "id" in obj and request_id is None:
        return _invalid_request("id must be a string or an integer")
    if obj.get("jsonrpc") != "2.0":
        return _invalid_request('jsonrpc must be "2.0"', request_id)

    is_response = "result" in obj or "error" in obj
    if "method" in obj:
        if is_response:
            return _invalid_request("a message cannot be both a request and a response", request_id)
        if not isinstance(obj["method"], str):
            return _invalid_request("method must be a string", request_id)
        params = obj.get("params")
        if params is not None and not isinstance(params, dict):
            if request_id is None:
                logger.warning("Dropped a notification whose params are not an object")
                return None
            return _reply(INVALID_PARAMS, "Invalid params", "params must be an object", request_id)
    elif not is_response:
        return _invalid_request("message is not a request, notification or response", request_id)

    try:
        message = jsonrpc_message_adapter.validate_python(obj, by_name=False)
    except ValidationError:
        if request_id is None or is_response:
            logger.warning("Dropped a message that does not match the JSON-RPC schema")
            return None
        return _invalid_request("message does not match the JSON-RPC schema", request_id)
    return Forward(message)


async def read_lines(
    stream: BinaryIO, max_bytes: int = MAX_LINE_BYTES
) -> AsyncIterator[bytes | _LineTooLong]:
    """Yield raw lines from ``stream``. Lines longer than ``max_bytes`` are skipped and reported."""
    while True:
        line = await anyio.to_thread.run_sync(
            stream.readline, max_bytes + 1, abandon_on_cancel=True
        )
        if not line:
            return
        if len(line) > max_bytes and not line.endswith(b"\n"):
            while True:  # discard the rest of the oversized line
                rest = await anyio.to_thread.run_sync(
                    stream.readline, 64 * 1024, abandon_on_cancel=True
                )
                if not rest or rest.endswith(b"\n"):
                    break
            yield LINE_TOO_LONG
            continue
        yield line


class PendingRequests:
    """Ids of forwarded requests that have not been answered yet."""

    def __init__(self) -> None:
        self._ids: set[RequestId] = set()
        self._empty = anyio.Event()
        self._empty.set()

    def add(self, request_id: RequestId) -> None:
        self._ids.add(request_id)
        if self._empty.is_set():
            self._empty = anyio.Event()

    def discard(self, request_id: RequestId) -> None:
        self._ids.discard(request_id)
        if not self._ids:
            self._empty.set()

    async def wait_until_empty(self) -> None:
        await self._empty.wait()


class _AnswerTrackingWriter:
    """The server's write stream. Marks a request answered once its reply is handed to stdout."""

    def __init__(self, inner: ObjectSendStream[SessionMessage], pending: PendingRequests) -> None:
        self._inner = inner
        self._pending = pending

    async def send(self, item: SessionMessage, /) -> None:
        await self._inner.send(item)
        message = item.message
        if isinstance(message, JSONRPCResponse | JSONRPCError) and message.id is not None:
            self._pending.discard(message.id)

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


class _IdleInput:
    """Stands in for stdin inside ``stdio_server()`` and ends when ``done`` is set."""

    def __init__(self, done: anyio.Event) -> None:
        self._done = done

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> str:
        await self._done.wait()
        raise StopAsyncIteration


@contextlib.contextmanager
def _claim_stdin() -> Iterator[BinaryIO]:
    """Read the protocol from a private duplicate of fd 0, and point fd 0 at the null device.

    This mirrors what ``stdio_server()`` does when it owns stdin: handlers and child processes that
    read stdin get EOF instead of consuming protocol bytes.
    """
    private_fd: int | None = None
    if sys.platform != "win32":
        with contextlib.suppress(OSError):
            private_fd = fcntl.fcntl(0, fcntl.F_DUPFD_CLOEXEC, 3)
    if private_fd is None:
        yield sys.stdin.buffer
        return

    null_fd = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(null_fd, 0)
    finally:
        os.close(null_fd)
    try:
        # closefd=False: a reader thread may still be blocked on this descriptor during shutdown.
        yield os.fdopen(private_fd, "rb", closefd=False)
    finally:
        os.dup2(private_fd, 0)


async def _pump_stdin(
    stdin: BinaryIO,
    to_server: ObjectSendStream[SessionMessage | Exception],
    writer: _AnswerTrackingWriter,
    pending: PendingRequests,
    drain_timeout: float,
) -> None:
    async with to_server:
        async for line in read_lines(stdin):
            if isinstance(line, _LineTooLong):
                outcome: Forward | Reply | None = _invalid_request(
                    f"message exceeds {MAX_LINE_BYTES} bytes"
                )
            else:
                outcome = parse_line(line)

            if outcome is None:
                continue
            if isinstance(outcome, Reply):
                logger.warning("Rejected stdin message: %s", outcome.error.error.data["reason"])
                await writer.send(SessionMessage(outcome.error))
                continue

            message = outcome.message
            if isinstance(message, JSONRPCRequest):
                pending.add(message.id)
            elif (
                isinstance(message, JSONRPCNotification)
                and message.method == "notifications/cancelled"
            ):
                cancelled = _valid_id((message.params or {}).get("requestId"))
                if cancelled is not None:
                    pending.discard(cancelled)  # the SDK never answers a cancelled request
            await to_server.send(SessionMessage(message))

        # End of input: let in-flight requests finish before the server sees EOF and cancels them.
        with anyio.move_on_after(drain_timeout):
            await pending.wait_until_empty()


async def serve_stdio(server: Server, *, drain_timeout: float = DRAIN_TIMEOUT_SECONDS) -> None:
    """Serve ``server`` over this process's stdin and stdout until stdin closes."""
    input_done = anyio.Event()
    with _claim_stdin() as stdin:
        async with stdio_server(stdin=cast(Any, _IdleInput(input_done))) as (sdk_read, sdk_write):
            await sdk_read.aclose()  # the SDK's reader stays idle; ours feeds the server
            pending = PendingRequests()
            writer = _AnswerTrackingWriter(sdk_write, pending)
            to_server, from_stdin = anyio.create_memory_object_stream[SessionMessage | Exception](0)
            original_stdout = sys.stdout
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    async with anyio.create_task_group() as tg:
                        tg.start_soon(_pump_stdin, stdin, to_server, writer, pending, drain_timeout)
                        await server.run(from_stdin, writer, server.create_initialization_options())
                        tg.cancel_scope.cancel()
            finally:
                original_stdout.flush()  # fd 1 still points at stderr here
                input_done.set()
