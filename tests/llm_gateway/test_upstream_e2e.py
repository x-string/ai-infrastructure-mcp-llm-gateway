"""Task 3: the upstream client over real sockets, against a server that misbehaves on purpose.

MockTransport cannot fake what the network does: refused connections, silent servers, truncated
framing, connection resets and a full connection pool. A raw TCP server can.
"""

from __future__ import annotations

import json
import re
import socket
import struct
import time
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager

import anyio
import pytest
from anyio.abc import SocketAttribute, SocketStream

from ai_gateway.llm_gateway.events import Channel, StreamEnd, TextDelta
from ai_gateway.llm_gateway.upstream import (
    USER_AGENT,
    UpstreamConfig,
    UpstreamError,
    create_client,
    open_stream,
)
from tests.llm_gateway.provider_streams import DONE, ch, chunk, d

pytestmark = pytest.mark.e2e

SECRET = "sk-gateway-0123456789abcdef"
PAYLOAD = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
HANG, RESET = object(), object()

CHUNKED = (
    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
)
UNFRAMED = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n"
LAST_CHUNK = b"0\r\n\r\n"
ROLE = d(chunk([ch({"role": "assistant", "content": ""})])).encode()
HELLO = d(chunk([ch({"content": "Hello"})])).encode()
STOP = d(chunk([ch({}, "stop")])).encode()


def framed(data: bytes) -> bytes:
    """One HTTP/1.1 chunk."""
    return b"%x\r\n" % len(data) + data + b"\r\n"


async def read_request(client: SocketStream) -> tuple[bytes, bytes]:
    data = b""
    while b"\r\n\r\n" not in data:
        data += await client.receive()
    head, _, body = data.partition(b"\r\n\r\n")
    match = re.search(rb"(?im)^content-length:\s*(\d+)", head)
    while match and len(body) < int(match.group(1)):
        body += await client.receive()
    return head, body


@asynccontextmanager
async def raw_server(*script: object) -> AsyncIterator[tuple[str, list[tuple[bytes, bytes]]]]:
    """Answer every request by playing ``script``: bytes are sent, floats are pauses, HANG never
    finishes, RESET aborts the connection; at the end the connection is closed."""
    requests: list[tuple[bytes, bytes]] = []
    listener = await anyio.create_tcp_listener(local_host="127.0.0.1")
    port = listener.extra(SocketAttribute.local_port)

    async def handle(client: SocketStream) -> None:
        async with client:
            requests.append(await read_request(client))
            for step in script:
                if isinstance(step, bytes):
                    await client.send(step)
                elif isinstance(step, float):
                    await anyio.sleep(step)
                elif step is HANG:
                    await anyio.sleep_forever()
                elif step is RESET:
                    raw = client.extra(SocketAttribute.raw_socket)
                    raw.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    return

    async with anyio.create_task_group() as tg:
        tg.start_soon(listener.serve, handle)
        try:
            yield f"http://127.0.0.1:{port}/v1/chat/completions", requests
        finally:
            tg.cancel_scope.cancel()


def config(url: str, **overrides: float) -> UpstreamConfig:
    timeouts = {"connect_timeout": 1.0, "read_timeout": 0.4, "first_event_timeout": 0.6}
    return UpstreamConfig(url, api_key=SECRET, **(timeouts | overrides))


async def relay(url: str, **overrides: float) -> list:
    settings = config(url, **overrides)
    async with create_client(settings) as client:
        stream = await open_stream(client, settings, PAYLOAD)
        async with aclosing(stream.items()) as items:
            return [item async for item in items]


async def failure(url: str, **overrides: float) -> UpstreamError:
    with pytest.raises(UpstreamError) as excinfo:
        await relay(url, **overrides)
    return excinfo.value


async def test_a_real_stream_is_relayed_and_the_request_is_what_the_provider_expects() -> None:
    script = (CHUNKED, framed(ROLE), framed(HELLO), framed(STOP + DONE.encode()), LAST_CHUNK)
    async with raw_server(*script) as (url, requests):
        assert await relay(url) == [TextDelta(Channel.CONTENT, "Hello"), StreamEnd("stop", None)]
    ((head, body),) = requests
    text = head.decode().lower()
    assert text.startswith("post /v1/chat/completions http/1.1")
    for header in [
        f"authorization: bearer {SECRET.lower()}",
        "accept: text/event-stream",
        "accept-encoding: identity",
        "content-type: application/json",
        f"user-agent: {USER_AGENT}",
    ]:
        assert header in text
    assert json.loads(body) == PAYLOAD


async def test_a_refused_connection_is_unavailable() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    error = await failure(f"http://127.0.0.1:{port}/v1/chat/completions")
    assert (error.status, error.code, error.retryable) == (502, "upstream_unavailable", True)


async def test_a_server_that_never_answers_times_out_at_the_first_event_deadline() -> None:
    async with raw_server(HANG) as (url, _):
        started = time.monotonic()
        error = await failure(url, read_timeout=30.0)
        elapsed = time.monotonic() - started
    assert (error.status, error.code) == (504, "upstream_timeout")
    assert 0.5 < elapsed < 2


async def test_silence_mid_stream_is_a_read_timeout() -> None:
    async with raw_server(CHUNKED, framed(ROLE), framed(HELLO), HANG) as (url, _):
        error = await failure(url)
    assert (error.code, error.detail) == ("upstream_timeout", "ReadTimeout")


async def test_a_slow_stream_within_the_read_timeout_is_fine() -> None:
    script = [CHUNKED, framed(ROLE)]
    for _ in range(6):  # 1.2 s in total, longer than the 0.4 s read timeout
        script += [0.2, framed(b": still here\n\n")]
    script += [framed(HELLO), framed(STOP), LAST_CHUNK]
    async with raw_server(*script) as (url, _):
        assert (await relay(url))[-1] == StreamEnd("stop", None)


async def test_a_truncated_chunked_body_is_unavailable() -> None:
    async with raw_server(CHUNKED, framed(ROLE), framed(HELLO), b"40\r\ndata: {") as (url, _):
        error = await failure(url)
    assert error.code == "upstream_unavailable"


async def test_an_unframed_body_cut_mid_event_is_caught_by_the_missing_finish_reason() -> None:
    # Without Content-Length or chunking, HTTP cannot tell a cut from the real end.
    async with raw_server(UNFRAMED, ROLE, HELLO, b'data: {"choices": [{"del') as (url, _):
        error = await failure(url)
    assert (error.code, error.detail) == (
        "upstream_protocol_error",
        "stream ended before a finish reason",
    )


async def test_a_connection_reset_mid_stream_is_unavailable() -> None:
    async with raw_server(CHUNKED, framed(ROLE), framed(HELLO), 0.05, RESET) as (url, _):
        error = await failure(url)
    assert error.code == "upstream_unavailable"


async def test_a_full_connection_pool_is_gateway_overload() -> None:
    async with raw_server(CHUNKED, framed(ROLE), HANG) as (url, _):
        settings = UpstreamConfig(url, max_connections=1, pool_timeout=0.2, first_event_timeout=2)
        async with create_client(settings) as client:
            async with await open_stream(client, settings, PAYLOAD):
                with pytest.raises(UpstreamError) as excinfo:
                    await open_stream(client, settings, PAYLOAD)
    assert (excinfo.value.status, excinfo.value.code) == (503, "gateway_overloaded")
