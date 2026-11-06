"""Task 3: the gateway's HTTP layer, driven as a raw ASGI app.

httpx2's ASGITransport buffers whole responses, so these tests call the app directly and watch
each body message as it is sent: that is how they show the first chunk leaving before the
upstream has finished, and what happens when the client disconnects.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
import pytest
from fastapi import FastAPI

from ai_gateway.llm_gateway.app import create_app
from ai_gateway.llm_gateway.redaction import REDACTED
from ai_gateway.llm_gateway.settings import GatewaySettings
from ai_gateway.llm_gateway.sse import Comment, Decoder
from ai_gateway.llm_gateway.upstream import UpstreamConfig
from tests.llm_gateway.scripted import DONE, HANG, HELLO, ROLE, STOP, Upstream, content

R = REDACTED
SECRET = "sk-gateway-0123456789abcdef"
REQUEST = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}
GUARD = 10


def settings(**overrides: object) -> GatewaySettings:
    upstream = UpstreamConfig(
        "http://upstream.test/v1/chat/completions", api_key=SECRET, first_event_timeout=0.5
    )
    return GatewaySettings(upstream=upstream, **overrides)  # type: ignore[arg-type]


class Exchange:
    """One request through the app: what was sent back, and a switch to disconnect."""

    def __init__(self, body: bytes, headers: list[tuple[bytes, bytes]]) -> None:
        self.body, self.headers = body, headers
        self.messages: list[dict] = []
        self.disconnect = anyio.Event()
        self.sent = anyio.Event()  # set on every body message
        self._body_sent = False

    async def receive(self) -> dict:
        if not self._body_sent:
            self._body_sent = True
            return {"type": "http.request", "body": self.body, "more_body": False}
        await self.disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(self, message: dict) -> None:
        self.messages.append(message)
        if message["type"] == "http.response.body":
            self.sent.set()
            self.sent = anyio.Event()

    @property
    def status(self) -> int:
        return self.messages[0]["status"]

    @property
    def response_headers(self) -> dict[str, str]:
        return {k.decode(): v.decode() for k, v in self.messages[0]["headers"]}

    @property
    def raw(self) -> bytes:
        return b"".join(
            m.get("body", b"") for m in self.messages if m["type"] == "http.response.body"
        )

    def json(self) -> dict:
        return json.loads(self.raw)

    def frames(self) -> list:
        decoder, out = Decoder(), []
        for item in decoder.feed(self.raw):
            if isinstance(item, Comment):
                continue
            out.append(item.data if item.data == "[DONE]" else json.loads(item.data))
        return out

    def text(self) -> str:
        return "".join(
            f["choices"][0]["delta"].get("content", "")
            for f in self.frames()
            if isinstance(f, dict) and f.get("choices")
        )


def exchange_for(
    payload: object = REQUEST, *, content_type: str = "application/json", extra=()
) -> Exchange:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    headers = [(b"content-type", content_type.encode()), *extra] if content_type else [*extra]
    return Exchange(body, headers)


async def call(
    app: FastAPI, exchange: Exchange, path: str = "/v1/chat/completions", method: str = "POST"
) -> Exchange:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": exchange.headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8300),
    }
    with anyio.fail_after(GUARD):
        await app(scope, exchange.receive, exchange.send)
    return exchange


@asynccontextmanager
async def gateway(upstream: Upstream, **overrides: object) -> AsyncIterator[FastAPI]:
    app = create_app(settings(**overrides), transport=upstream.transport)
    async with app.router.lifespan_context(app):
        yield app


async def run(upstream: Upstream, payload: object = REQUEST, **kwargs: object) -> Exchange:
    async with gateway(upstream) as app:
        return await call(app, exchange_for(payload, **kwargs))


# --- a stream through the gateway -----------------------------------------------------------------


async def test_the_reply_streams_redacted_with_streaming_headers() -> None:
    upstream = Upstream(ROLE, content("Mail john.do"), content("e@example.com now"), STOP, DONE)
    exchange = await run(upstream)
    assert exchange.status == 200
    headers = exchange.response_headers
    assert headers["content-type"].startswith("text/event-stream")
    assert headers["cache-control"] == "no-cache"
    assert headers["x-accel-buffering"] == "no"
    assert len(headers["x-request-id"]) == 32
    assert exchange.text() == f"Mail {R} now"
    assert exchange.frames()[-1] == "[DONE]"
    assert b"john" not in exchange.raw
    assert upstream.closed == 1


async def test_the_provider_gets_a_rebuilt_request_with_the_gateways_own_key() -> None:
    upstream = Upstream(ROLE, STOP)
    payload = {**REQUEST, "temperature": 0.5, "logprobs": False}
    await run(upstream, payload, extra=[(b"authorization", b"Bearer client-token")])
    (request,) = upstream.requests
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert json.loads(request.content) == {**REQUEST, "temperature": 0.5, "stream": True}


async def test_the_first_chunk_leaves_before_the_upstream_finishes() -> None:
    gate = anyio.Event()
    upstream = Upstream(ROLE, HELLO, content(" there,"), gate, content(" bye"), STOP, DONE)
    async with gateway(upstream) as app:
        exchange = exchange_for()
        async with anyio.create_task_group() as tg:
            tg.start_soon(call, app, exchange)
            with anyio.fail_after(GUARD):
                while b"Hello" not in exchange.raw:
                    await exchange.sent.wait()
            assert not gate.is_set() and b"[DONE]" not in exchange.raw
            gate.set()
    assert exchange.text() == "Hello there, bye"


async def test_an_error_mid_stream_ends_with_an_error_frame_and_no_pii() -> None:
    error = b'data: {"error": {"message": "secret prompt"}}\n\n'
    exchange = await run(Upstream(ROLE, content("Mail john.do"), error))
    assert exchange.status == 200
    assert exchange.frames()[-1]["error"]["code"] == "upstream_error"
    assert b"john" not in exchange.raw and b"secret prompt" not in exchange.raw
    assert b"[DONE]" not in exchange.raw


# --- errors before the stream starts --------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "headers", "client_status", "code", "should_retry", "retry_after"),
    [
        (429, {"retry-after": "7"}, 429, "upstream_rate_limited", "true", "7"),
        (503, {}, 503, "upstream_overloaded", "true", None),
        (401, {}, 502, "upstream_auth_failed", "false", None),
        (400, {}, 400, "upstream_rejected_request", "false", None),
        (500, {}, 502, "upstream_unavailable", "true", None),
    ],
)
async def test_upstream_errors_become_openai_style_http_errors(
    status: int,
    headers: dict,
    client_status: int,
    code: str,
    should_retry: str,
    retry_after: str | None,
) -> None:
    upstream = Upstream(
        b'{"error": {"message": "secret prompt text"}}', status=status, headers=headers
    )
    exchange = await run(upstream)
    assert exchange.status == client_status
    assert exchange.json()["error"]["code"] == code
    assert exchange.response_headers["x-should-retry"] == should_retry
    assert exchange.response_headers.get("retry-after") == retry_after
    assert b"secret prompt" not in exchange.raw


async def test_no_first_event_in_time_is_a_504() -> None:
    exchange = await run(Upstream(HANG))
    assert (exchange.status, exchange.json()["error"]["code"]) == (504, "upstream_timeout")


@pytest.mark.parametrize(
    ("payload", "content_type", "status", "code"),
    [
        (REQUEST, "text/plain", 415, "unsupported_media_type"),
        (REQUEST, "", 415, "unsupported_media_type"),
        (b"{not json", "application/json", 400, "invalid_request"),
        (b'{"model": NaN}', "application/json", 400, "invalid_request"),
        (b"\xff\xfe", "application/json", 400, "invalid_request"),
        ({**REQUEST, "tools": [{"type": "function"}]}, "application/json", 400, "invalid_request"),
        ({**REQUEST, "stream": False}, "application/json", 400, "invalid_request"),
        (b"x" * (2 << 20), "application/json", 413, "request_too_large"),
    ],
    ids=[
        "text-plain",
        "no-content-type",
        "bad-json",
        "nan",
        "bad-utf8",
        "tools",
        "no-stream",
        "huge",
    ],
)
async def test_bad_requests_are_answered_before_the_provider_is_called(
    payload: object, content_type: str, status: int, code: str
) -> None:
    upstream = Upstream(ROLE, STOP)
    exchange = await run(upstream, payload, content_type=content_type)
    assert exchange.status == status
    assert exchange.json()["error"]["code"] == code
    assert len(exchange.response_headers["x-request-id"]) == 32
    assert upstream.requests == []


async def test_a_refused_parameter_names_itself() -> None:
    exchange = await run(Upstream(), {**REQUEST, "tools": [{"type": "function"}]})
    error = exchange.json()["error"]
    assert (error["param"], error["type"]) == ("tools", "invalid_request_error")
    assert error["message"].startswith("tools: tool calls are not supported")


@pytest.mark.parametrize(
    ("path", "method", "status"), [("/nope", "GET", 404), ("/v1/chat/completions", "GET", 405)]
)
async def test_unknown_routes_get_json_errors(path: str, method: str, status: int) -> None:
    async with gateway(Upstream()) as app:
        exchange = await call(app, exchange_for(), path=path, method=method)
    assert exchange.status == status
    assert exchange.json()["error"]["code"] in ("not_found", "method_not_allowed")


# --- stream slots and disconnects -----------------------------------------------------------------


async def test_streams_beyond_the_limit_get_503_until_a_slot_frees_up() -> None:
    gate = anyio.Event()
    # "Hello" alone would be held back: it could still become an email address.
    upstream = Upstream(ROLE, content("Hello, "), gate, STOP, DONE)
    async with gateway(upstream, max_streams=1) as app:
        first = exchange_for()
        async with anyio.create_task_group() as tg:
            tg.start_soon(call, app, first)
            with anyio.fail_after(GUARD):
                while b"Hello" not in first.raw:
                    await first.sent.wait()
            second = await call(app, exchange_for())
            assert second.status == 503
            assert second.json()["error"]["code"] == "gateway_overloaded"
            assert second.response_headers["retry-after"] == "1"
            gate.set()
        third = await call(app, exchange_for())
    assert third.status == 200 and third.frames()[-1] == "[DONE]"


async def test_a_disconnect_before_the_first_event_cancels_the_upstream_and_frees_the_slot() -> (
    None
):
    upstream = Upstream(HANG)
    async with gateway(upstream, max_streams=1) as app:
        exchange = exchange_for()
        async with anyio.create_task_group() as tg:
            tg.start_soon(call, app, exchange)
            with anyio.fail_after(GUARD):
                while not upstream.requests:
                    await anyio.sleep(0.01)
            exchange.disconnect.set()
        assert upstream.closed == 1
        upstream.parts = [ROLE, STOP]
        assert (await call(app, exchange_for())).status == 200


async def test_a_disconnect_mid_stream_closes_the_upstream_and_frees_the_slot() -> None:
    upstream = Upstream(ROLE, content("Hello, "), HANG)
    async with gateway(upstream, max_streams=1) as app:
        exchange = exchange_for()
        async with anyio.create_task_group() as tg:
            tg.start_soon(call, app, exchange)
            with anyio.fail_after(GUARD):
                while b"Hello" not in exchange.raw:
                    await exchange.sent.wait()
            exchange.disconnect.set()
        assert upstream.closed == 1
        upstream.parts = [ROLE, STOP]
        assert (await call(app, exchange_for())).status == 200


# --- logging --------------------------------------------------------------------------------------


async def test_one_log_line_per_stream_and_no_text_from_the_prompt_or_reply(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="ai_gateway.llm_gateway")
    upstream = Upstream(ROLE, content("Reach john.doe@example.com"), STOP, DONE)
    exchange = await run(upstream, {**REQUEST, "messages": [{"role": "user", "content": "my ssn"}]})
    request_id = exchange.response_headers["x-request-id"]
    lines = [r.getMessage() for r in caplog.records if request_id in r.getMessage()]
    assert len(lines) == 1
    assert "outcome=completed" in lines[0] and "redactions=1" in lines[0]
    assert "john" not in caplog.text and "my ssn" not in caplog.text


async def test_health() -> None:
    async with gateway(Upstream()) as app:
        exchange = await call(app, exchange_for(b""), path="/healthz", method="GET")
    assert (exchange.status, exchange.json()) == (200, {"status": "ok"})
