"""Task 3: the upstream client, against a scripted upstream on httpx2's MockTransport.

Every test checks that the upstream body was closed, whatever happened: a leaked body is a leaked
connection.
"""

from __future__ import annotations

import gzip
import json
import time
from contextlib import aclosing
from datetime import UTC, datetime
from email.utils import format_datetime

import anyio
import httpx2
import pytest

from ai_gateway.llm_gateway.events import KEEP_ALIVE, Channel, StreamEnd, TextDelta, Usage
from ai_gateway.llm_gateway.upstream import (
    USER_AGENT,
    StreamItem,
    UpstreamConfig,
    UpstreamError,
    UpstreamStream,
    create_client,
    error_for_status,
    open_stream,
    retry_after_seconds,
)
from tests.llm_gateway.provider_streams import DONE, STREAMS, ProviderStream, ch, chunk, d
from tests.llm_gateway.scripted import HANG, HELLO, ROLE, STOP, USAGE_CHUNK, WORLD, Upstream

URL = "http://upstream.test/v1/chat/completions"
SECRET = "sk-gateway-0123456789abcdef"
CONFIG = UpstreamConfig(URL, api_key=SECRET, first_event_timeout=0.3)
PAYLOAD = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}


def client_for(upstream: Upstream, config: UpstreamConfig = CONFIG) -> httpx2.AsyncClient:
    return create_client(config, transport=httpx2.MockTransport(upstream.handle))


GUARD = 10  # MockTransport has no network timeouts: a regression must fail, not hang the suite


async def read_all(stream: UpstreamStream) -> list[StreamItem]:
    with anyio.fail_after(GUARD):
        async with aclosing(stream.items()) as items:
            return [item async for item in items]


async def run(upstream: Upstream, config: UpstreamConfig = CONFIG) -> list[StreamItem]:
    async with client_for(upstream, config) as client:
        return await read_all(await open_stream(client, config, PAYLOAD))


async def failure(upstream: Upstream, config: UpstreamConfig = CONFIG) -> UpstreamError:
    with pytest.raises(UpstreamError) as excinfo:
        await run(upstream, config)
    return excinfo.value


# --- the request ----------------------------------------------------------------------------------


async def test_the_request_carries_the_gateway_key_and_asks_for_an_uncompressed_stream() -> None:
    upstream = Upstream(ROLE, STOP)
    await run(upstream)
    (request,) = upstream.requests
    assert (request.method, str(request.url)) == ("POST", URL)
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["accept-encoding"] == "identity"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["user-agent"] == USER_AGENT
    assert json.loads(request.content) == PAYLOAD


async def test_without_an_api_key_no_authorization_header_is_sent() -> None:
    upstream = Upstream(ROLE, STOP)
    await run(upstream, UpstreamConfig(URL, first_event_timeout=0.3))
    assert "authorization" not in upstream.requests[0].headers


# --- a successful stream --------------------------------------------------------------------------


async def test_a_stream_is_relayed_as_text_deltas_then_its_end() -> None:
    upstream = Upstream(ROLE, HELLO, WORLD, STOP, USAGE_CHUNK, DONE.encode())
    assert await run(upstream) == [
        TextDelta(Channel.CONTENT, "Hello"),
        TextDelta(Channel.CONTENT, " world"),
        StreamEnd("stop", Usage(5, 3, 8)),
    ]
    assert upstream.closed == 1


async def test_a_stream_complete_in_its_first_chunk_is_relayed_in_full() -> None:
    upstream = Upstream(ROLE + HELLO + STOP + DONE.encode())
    assert await run(upstream) == [TextDelta(Channel.CONTENT, "Hello"), StreamEnd("stop", None)]


async def test_the_end_of_the_body_after_a_finish_reason_is_success() -> None:
    assert (await run(Upstream(ROLE, HELLO, STOP)))[-1] == StreamEnd("stop", None)


async def test_reading_stops_at_done() -> None:
    upstream = Upstream(ROLE, HELLO, STOP, DONE.encode(), RuntimeError("must not be read"))
    assert (await run(upstream))[-1] == StreamEnd("stop", None)
    assert upstream.pulled == 4
    assert upstream.closed == 1


async def test_gzip_bodies_are_decoded() -> None:
    body = gzip.compress(ROLE + HELLO + STOP)
    upstream = Upstream(
        body, headers={"content-type": "text/event-stream", "content-encoding": "gzip"}
    )
    assert (await run(upstream))[0] == TextDelta(Channel.CONTENT, "Hello")


async def test_keep_alives_are_merged_and_none_is_relayed_before_the_response_starts() -> None:
    upstream = Upstream(
        b": ping\n\n" + ROLE + b": ping\n\n", HELLO, b": a\n\n: b\n\ndata:\n\n", STOP
    )
    assert await run(upstream) == [
        TextDelta(Channel.CONTENT, "Hello"),
        KEEP_ALIVE,
        StreamEnd("stop", None),
    ]


# --- before the response starts: HTTP errors ------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "client_status", "code", "retryable"),
    [
        (429, 429, "upstream_rate_limited", True),
        (503, 503, "upstream_overloaded", True),
        (529, 503, "upstream_overloaded", True),
        (408, 504, "upstream_timeout", True),
        (504, 504, "upstream_timeout", True),
        (400, 400, "upstream_rejected_request", False),
        (404, 400, "upstream_rejected_request", False),
        (413, 400, "upstream_rejected_request", False),
        (422, 400, "upstream_rejected_request", False),
        (401, 502, "upstream_auth_failed", False),
        (403, 502, "upstream_auth_failed", False),
        (500, 502, "upstream_unavailable", True),
        (502, 502, "upstream_unavailable", True),
        (409, 502, "upstream_unavailable", True),
        (402, 502, "upstream_unavailable", False),
        (410, 502, "upstream_unavailable", False),
        (201, 502, "upstream_protocol_error", False),
        (204, 502, "upstream_protocol_error", False),
        (302, 502, "upstream_protocol_error", False),
    ],
)
async def test_upstream_statuses_map_to_client_errors(
    status: int, client_status: int, code: str, retryable: bool
) -> None:
    upstream = Upstream(b'{"error": {"message": "nope"}}', status=status, headers={})
    error = await failure(upstream)
    assert (error.status, error.code, error.retryable) == (client_status, code, retryable)
    assert upstream.closed == 1


async def test_retry_after_is_passed_on_for_rate_limits_and_overload() -> None:
    error = await failure(Upstream(status=429, headers={"retry-after": "7"}))
    assert error.retry_after == 7
    error = await failure(Upstream(status=503, headers={"retry-after-ms": "2500"}))
    assert error.retry_after == 3
    assert (await failure(Upstream(status=429, headers={}))).retry_after is None


async def test_error_bodies_give_a_safe_hint_and_never_their_message() -> None:
    body = {"error": {"message": "secret prompt text", "type": "invalid_request_error",
                      "code": "context_length_exceeded"}}  # fmt: skip
    error = await failure(Upstream(json.dumps(body).encode(), status=400, headers={}))
    assert error.detail == "HTTP 400 (type=invalid_request_error, code=context_length_exceeded)"
    assert str(error) == error.message == "The model provider rejected the request."
    unsafe = {"error": {"type": "john@example.com", "code": "x" * 65}}
    error = await failure(Upstream(json.dumps(unsafe).encode(), status=400, headers={}))
    assert error.detail == "HTTP 400"


async def test_a_huge_error_body_is_read_only_up_to_the_limit() -> None:
    upstream = Upstream(*[b"x" * 4096] * 1000, status=500, headers={})
    await failure(upstream)
    assert upstream.pulled <= 5
    assert upstream.closed == 1


@pytest.mark.parametrize(
    "headers",
    [
        {"content-type": "application/json"},
        {},
        {"content-type": "text/event-streamx"},
        {"content-type": "text/plain; charset=utf-8"},
        {"content-type": "text/event-stream", "content-encoding": "br"},
        {"content-type": "text/event-stream", "content-encoding": "gzip, zstd"},
    ],
    ids=["json", "missing", "lookalike", "plain", "brotli", "zstd"],
)
async def test_responses_that_are_not_plain_event_streams_are_refused(headers: dict) -> None:
    upstream = Upstream(ROLE, STOP, headers=headers)
    error = await failure(upstream)
    assert (error.status, error.code, error.retryable) == (502, "upstream_protocol_error", False)


@pytest.mark.parametrize(
    "content_type", ["text/event-stream; charset=utf-8", "TEXT/Event-Stream", " text/event-stream "]
)
async def test_event_stream_media_type_case_and_parameters_are_accepted(content_type: str) -> None:
    upstream = Upstream(ROLE, HELLO, STOP, headers={"content-type": content_type})
    assert (await run(upstream))[0] == TextDelta(Channel.CONTENT, "Hello")


# --- before the response starts: transport failures and the first-event deadline ------------------


@pytest.mark.parametrize(
    ("exc", "status", "code", "retryable"),
    [
        (httpx2.ConnectError("refused"), 502, "upstream_unavailable", True),
        (httpx2.RemoteProtocolError("bad status line"), 502, "upstream_unavailable", True),
        (httpx2.UnsupportedProtocol("ftp"), 502, "upstream_unavailable", True),
        (httpx2.ConnectTimeout("slow"), 504, "upstream_timeout", True),
        (httpx2.ReadTimeout("slow"), 504, "upstream_timeout", True),
        (httpx2.WriteTimeout("slow"), 504, "upstream_timeout", True),
        (httpx2.PoolTimeout("full"), 503, "gateway_overloaded", True),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, Exception) else None,
)
async def test_transport_failures_before_the_response(
    exc: Exception, status: int, code: str, retryable: bool
) -> None:
    upstream = Upstream()
    upstream.before_headers = exc
    error = await failure(upstream)
    assert (error.status, error.code, error.retryable) == (status, code, retryable)


@pytest.mark.parametrize(
    "parts",
    [(HANG,), (b": ping\n\n", HANG), (b": ping\n\ndata:\n\n", HANG)],
    ids=["body", "comment-only", "empty-event-only"],
)
async def test_no_first_event_in_time_is_a_timeout(parts: tuple) -> None:
    upstream = Upstream(*parts)
    started = time.monotonic()
    error = await failure(upstream)
    assert (error.status, error.code, error.detail) == (
        504,
        "upstream_timeout",
        "no first event in time",
    )
    assert time.monotonic() - started < 2
    assert upstream.closed == 1


async def test_no_response_headers_in_time_is_a_timeout() -> None:
    upstream = Upstream()
    upstream.before_headers = HANG
    error = await failure(upstream)
    assert (error.status, error.code) == (504, "upstream_timeout")


@pytest.mark.parametrize(
    ("parts", "code", "detail"),
    [
        ((d({"error": {"message": "boom"}}).encode(),), "upstream_error", "error object"),
        ((b"data: {not json\n\n",), "upstream_protocol_error", "invalid JSON"),
        ((), "upstream_protocol_error", "stream ended before the first event"),
        ((b": ping\n\n",), "upstream_protocol_error", "stream ended before the first event"),
        (
            (ROLE + HELLO + DONE.encode(),),
            "upstream_protocol_error",
            "stream ended before a finish reason",
        ),
        ((b"data: \xff\n\n",), "upstream_protocol_error", "invalid UTF-8"),
    ],
    ids=["error-event", "invalid-json", "empty-body", "comments-only", "truncated", "bad-utf8"],
)
async def test_the_first_chunk_is_checked_before_the_response_starts(
    parts: tuple, code: str, detail: str
) -> None:
    upstream = Upstream(*parts)
    async with client_for(upstream) as client:
        with pytest.raises(UpstreamError) as excinfo:
            await open_stream(client, CONFIG, PAYLOAD)
    assert (excinfo.value.status, excinfo.value.code, excinfo.value.detail) == (502, code, detail)
    assert upstream.closed == 1


# --- after the response starts --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("part", "code"),
    [
        (httpx2.ReadError("reset"), "upstream_unavailable"),
        (httpx2.RemoteProtocolError("incomplete chunked read"), "upstream_unavailable"),
        (httpx2.ReadTimeout("silence"), "upstream_timeout"),
        (httpx2.DecodingError("bad gzip"), "upstream_protocol_error"),
        (d({"error": {"message": "overloaded"}}).encode(), "upstream_error"),
        (b"event: error\ndata: {}\n\n", "upstream_error"),
        (b"data: caf\xc3\n\n", "upstream_protocol_error"),
        (d(chunk([ch({"tool_calls": [{"index": 0}]})])).encode(), "upstream_protocol_error"),
        (b"data: " + b"x" * (1 << 20) + b"\n\n", "upstream_protocol_error"),
    ],
    ids=["read-error", "truncated-body", "read-timeout", "decoding", "error-object", "error-event",
         "bad-utf8", "tool-calls", "oversized-event"],
)  # fmt: skip
async def test_failures_mid_stream_end_the_stream_with_an_error(part: object, code: str) -> None:
    upstream = Upstream(ROLE, HELLO, part, STOP)
    async with client_for(upstream) as client:
        stream = await open_stream(client, CONFIG, PAYLOAD)
        received = []
        with pytest.raises(UpstreamError) as excinfo, anyio.fail_after(GUARD):
            async with aclosing(stream.items()) as items:
                async for item in items:
                    received.append(item)
    assert received == [TextDelta(Channel.CONTENT, "Hello")]
    assert excinfo.value.code == code
    assert upstream.closed == 1


async def test_a_body_that_ends_without_a_finish_reason_is_an_error() -> None:
    error = await failure(Upstream(ROLE, HELLO))
    assert (error.code, error.detail) == (
        "upstream_protocol_error",
        "stream ended before a finish reason",
    )
    error = await failure(Upstream(ROLE, HELLO, STOP.removesuffix(b"\n")))  # unfinished last event
    assert error.detail == "stream ended before a finish reason"


# --- closing and cancellation ---------------------------------------------------------------------


async def test_closing_is_idempotent_and_releases_the_body() -> None:
    upstream = Upstream(ROLE, HELLO, STOP)
    async with client_for(upstream) as client:
        stream = await open_stream(client, CONFIG, PAYLOAD)
        await stream.aclose()
        await stream.aclose()
    assert upstream.closed == 1


async def test_the_stream_is_an_async_context_manager() -> None:
    upstream = Upstream(ROLE, HELLO, STOP)
    async with client_for(upstream) as client:
        async with await open_stream(client, CONFIG, PAYLOAD):
            pass
    assert upstream.closed == 1


async def test_stopping_early_releases_the_body() -> None:
    upstream = Upstream(ROLE, HELLO, WORLD, HANG)
    async with client_for(upstream) as client:
        stream = await open_stream(client, CONFIG, PAYLOAD)
        async with aclosing(stream.items()) as items:
            async for _ in items:
                break
    assert upstream.closed == 1


async def test_cancellation_while_waiting_for_the_first_event_releases_the_body() -> None:
    upstream = Upstream(HANG)
    config = UpstreamConfig(URL, first_event_timeout=30)
    async with client_for(upstream, config) as client:
        with anyio.move_on_after(0.1) as scope:
            await open_stream(client, config, PAYLOAD)
    assert scope.cancelled_caught
    assert upstream.closed == 1


async def test_cancellation_while_reading_releases_the_body() -> None:
    upstream = Upstream(ROLE, HELLO, HANG)
    async with client_for(upstream) as client:
        stream = await open_stream(client, CONFIG, PAYLOAD)
        with anyio.move_on_after(0.1) as scope:
            await read_all(stream)
    assert scope.cancelled_caught
    assert upstream.closed == 1


# --- nothing secret leaks into errors -------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 429, 500])
async def test_errors_never_carry_the_api_key_or_the_upstream_message(status: int) -> None:
    body = json.dumps({"error": {"message": f"key {SECRET} rejected: my ssn is 123-45-6789"}})
    error = await failure(Upstream(body.encode(), status=status, headers={}))
    for text in (str(error), error.message, error.detail, repr(error.args)):
        assert SECRET not in text and "123-45-6789" not in text


def test_the_config_never_shows_the_api_key() -> None:
    assert SECRET not in repr(CONFIG)


# --- configuration --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "ftp://upstream.test/v1/chat/completions"},
        {"url": "/v1/chat/completions"},
        {"url": "http://"},
        {"url": "http://[::1"},
        {"read_timeout": 0},
        {"first_event_timeout": -1},
        {"connect_timeout": float("inf")},
        {"pool_timeout": float("nan")},
        {"max_connections": 0},
    ],
)
def test_invalid_configuration_is_refused_at_startup(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        UpstreamConfig(**({"url": URL} | kwargs))


async def test_the_client_has_explicit_timeouts_and_ignores_the_environment() -> None:
    config = UpstreamConfig(URL, connect_timeout=1, read_timeout=2, write_timeout=3, pool_timeout=4)
    async with create_client(config) as client:
        assert client.timeout == httpx2.Timeout(connect=1, read=2, write=3, pool=4)
        assert client.trust_env is False
        assert client.follow_redirects is False


# --- Retry-After ----------------------------------------------------------------------------------

NOW = 1_800_000_000.0


def http_date(offset: float) -> str:
    return format_datetime(datetime.fromtimestamp(NOW + offset, UTC), usegmt=True)


@pytest.mark.parametrize(
    ("headers", "seconds"),
    [
        ({"retry-after": "7"}, 7),
        ({"retry-after": " 7 "}, 7),
        ({"retry-after": "1.5"}, 2),
        ({"retry-after": "0"}, None),
        ({"retry-after": "-5"}, None),
        ({"retry-after": "+5"}, None),
        ({"retry-after": "1e3"}, None),
        ({"retry-after": "٥"}, None),
        ({"retry-after": "soon"}, None),
        ({"retry-after": "99999999"}, 3600),
        ({"retry-after": http_date(30)}, 30),
        ({"retry-after": http_date(-30)}, None),
        ({"retry-after": "Wed, 21 Oct 2015 07:28:00"}, None),
        ({"retry-after-ms": "36"}, 1),
        ({"retry-after-ms": "2500"}, 3),
        ({"retry-after-ms": "2500", "retry-after": "60"}, 3),
        ({"retry-after-ms": "soon", "retry-after": "5"}, 5),
        ({"retry-after-ms": "0", "retry-after": "5"}, 5),
        ({}, None),
    ],
)
def test_retry_after_seconds(headers: dict, seconds: int | None) -> None:
    assert retry_after_seconds(headers, now=NOW) == seconds


def test_error_for_status_takes_retry_after_from_the_headers() -> None:
    assert error_for_status(429, {"retry-after": "12"}).retry_after == 12
    assert error_for_status(500, {"retry-after": "12"}).retry_after is None


# --- real provider streams through the client -----------------------------------------------------


@pytest.mark.parametrize("stream", STREAMS, ids=[s.name for s in STREAMS])
async def test_provider_streams_through_the_client(stream: ProviderStream) -> None:
    events = [part + b"\n\n" for part in stream.body.split(b"\n\n") if part]
    if not stream.body.endswith(b"\n\n"):  # keep an unfinished last event unfinished
        events[-1] = events[-1].removesuffix(b"\n\n")
    upstream = Upstream(*events)
    if stream.outcome == "success":
        items = await run(upstream)
        deltas = [item for item in items if isinstance(item, TextDelta)]
        text = {c: "".join(d.text for d in deltas if d.channel is c) for c in Channel}
        assert (text[Channel.CONTENT], text[Channel.REFUSAL]) == (stream.text, stream.refusal)
        assert isinstance(items[-1], StreamEnd)
        assert items[-1].finish_reason == stream.finish
        assert (items[-1].usage is not None) is stream.usage
    else:
        error = await failure(upstream)
        expected = (
            "upstream_error" if stream.outcome == "reported_error" else "upstream_protocol_error"
        )
        assert error.code == expected
    assert upstream.closed == 1
