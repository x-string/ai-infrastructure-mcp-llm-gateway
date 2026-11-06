"""The upstream provider: opening its stream and reading it, failing closed (Task 3).

``open_stream`` sends the request and waits for the first event before returning, so the endpoint
can still answer with a proper HTTP error until then. Everything that can go wrong becomes an
``UpstreamError``, which says what the client should see: an HTTP status before the response has
started, and an error code either way. Its message is generic; upstream bodies, which can echo the
prompt, are never passed on. ``detail`` is written by the gateway and is only for logs.

Status mapping before the response starts (``retryable`` becomes the ``x-should-retry`` header,
which the OpenAI SDKs honour; they retry 408, 409, 429 and 5xx otherwise):

=====================================  =======  ===========================  =========
Upstream                               Client   Code                         Retryable
=====================================  =======  ===========================  =========
429                                    429      upstream_rate_limited        yes
503, 529                               503      upstream_overloaded          yes
408, 504, or any timeout               504      upstream_timeout             yes
400, 404, 413, 422                     400      upstream_rejected_request    no
401, 403 (the gateway's own key)       502      upstream_auth_failed         no
other 5xx, 409, connection failures    502      upstream_unavailable         yes
other 4xx                              502      upstream_unavailable         no
not a 200 event stream                 502      upstream_protocol_error      no
an error reported in the stream        502      upstream_error               no
invalid or unexpected stream data      502      upstream_protocol_error      no
connection pool exhausted              503      gateway_overloaded           yes
=====================================  =======  ===========================  =========

``Retry-After`` comes from the upstream's ``retry-after-ms`` or ``retry-after`` (seconds or an
HTTP date), as whole seconds between 1 and 3600.
"""

from __future__ import annotations

import email.utils
import json
import math
import re
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field
from types import TracebackType

import anyio
import httpx2

from ai_gateway.llm_gateway import sse
from ai_gateway.llm_gateway.events import (
    KEEP_ALIVE,
    KeepAlive,
    ProtocolViolation,
    ReportedError,
    StreamEnd,
    StreamNormalizer,
    TextDelta,
)

StreamItem = TextDelta | KeepAlive | StreamEnd

USER_AGENT = "ai-gateway-llm-gateway"
_ERROR_BODY_LIMIT = 16 * 1024
_HINT = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_SECONDS = re.compile(r"[0-9]{1,10}(?:\.[0-9]{1,6})?")
_DECODABLE_ENCODINGS = frozenset({"", "identity", "gzip", "x-gzip", "deflate"})

_MESSAGES = {
    "upstream_rate_limited": "The model provider is rate limiting requests. Try again later.",
    "upstream_overloaded": "The model provider is overloaded. Try again later.",
    "upstream_timeout": "The model provider did not respond in time.",
    "upstream_rejected_request": "The model provider rejected the request.",
    "upstream_auth_failed": "The gateway could not authenticate with the model provider.",
    "upstream_unavailable": "The model provider could not be reached.",
    "upstream_protocol_error": "The model provider sent a response the gateway cannot relay.",
    "upstream_error": "The model provider reported an error.",
    "gateway_overloaded": "The gateway is handling too many streams. Try again later.",
}


@dataclass(frozen=True, slots=True)
class UpstreamConfig:
    url: str  # the full chat completions URL, e.g. https://api.openai.com/v1/chat/completions
    api_key: str | None = field(default=None, repr=False)
    connect_timeout: float = 5.0
    read_timeout: float = 120.0  # the longest silence allowed between two pieces of the stream
    write_timeout: float = 10.0
    pool_timeout: float = 1.0
    first_event_timeout: float = 60.0  # from sending the request to the first event
    max_connections: int = 100

    def __post_init__(self) -> None:
        try:
            url = httpx2.URL(self.url)
        except httpx2.InvalidURL:
            raise ValueError("the upstream URL is not a valid URL") from None
        if url.scheme not in ("http", "https") or not url.host:
            raise ValueError("the upstream URL must be an absolute http or https URL")
        for name in (
            "connect_timeout",
            "read_timeout",
            "write_timeout",
            "pool_timeout",
            "first_event_timeout",
        ):
            value = getattr(self, name)
            if not (isinstance(value, int | float) and 0 < value < math.inf):
                raise ValueError(f"{name} must be a positive number of seconds")
        if self.max_connections < 1:
            raise ValueError("max_connections must be at least 1")


class UpstreamError(Exception):
    """Why the upstream request failed, as the client should see it."""

    def __init__(
        self,
        status: int,
        code: str,
        *,
        retryable: bool,
        retry_after: int | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(_MESSAGES[code])
        self.status = status
        self.code = code
        self.message = _MESSAGES[code]
        self.retryable = retryable
        self.retry_after = retry_after
        self.detail = detail or code


def create_client(
    config: UpstreamConfig, *, transport: httpx2.AsyncBaseTransport | None = None
) -> httpx2.AsyncClient:
    """The HTTP client for the upstream: explicit timeouts, no proxies from the environment, and
    no redirects (a redirect would send the API key to another URL)."""
    return httpx2.AsyncClient(
        transport=transport,
        timeout=httpx2.Timeout(
            connect=config.connect_timeout,
            read=config.read_timeout,
            write=config.write_timeout,
            pool=config.pool_timeout,
        ),
        limits=httpx2.Limits(
            max_connections=config.max_connections,
            max_keepalive_connections=config.max_connections,
        ),
        trust_env=False,
        follow_redirects=False,
    )


async def open_stream(
    client: httpx2.AsyncClient, config: UpstreamConfig, payload: Mapping[str, object]
) -> UpstreamStream:
    """Send the request, and return once the first event has arrived and been checked."""
    request = client.build_request(
        "POST", config.url, content=_encode(payload), headers=_request_headers(config)
    )
    response: httpx2.Response | None = None
    stream: UpstreamStream | None = None
    try:
        with anyio.fail_after(config.first_event_timeout):
            response = await client.send(request, stream=True)
            if response.status_code != 200:
                raise await _status_error(response)
            _check_stream_headers(response.headers)
            stream = UpstreamStream(response)
            await stream._prime()
        return stream
    except TimeoutError:
        await _close(stream, response)
        raise UpstreamError(
            504, "upstream_timeout", retryable=True, detail="no first event in time"
        ) from None
    except httpx2.RequestError as exc:
        await _close(stream, response)
        raise _transport_error(exc) from None
    except BaseException:
        await _close(stream, response)
        raise


class UpstreamStream:
    """An upstream stream whose first event has arrived. Iterate ``items()`` once, then close."""

    def __init__(self, response: httpx2.Response) -> None:
        self._response = response
        self._chunks = response.aiter_bytes()
        self._decoder = sse.Decoder()
        self._normalizer = StreamNormalizer()
        self._pending: list[StreamItem] = []
        self._ended = False
        self._closed = False

    async def __aenter__(self) -> UpstreamStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def items(self) -> AsyncIterator[StreamItem]:
        """Text deltas and keep-alives in order, then one ``StreamEnd``; or an UpstreamError."""
        try:
            pending, self._pending = self._pending, []
            for item in pending:
                yield item
            while not self._ended:
                for item in self._process(await self._read()):
                    yield item
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """Release the connection. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        with anyio.CancelScope(shield=True):
            try:
                # Closing the response alone leaves the body iterator suspended.
                await self._chunks.aclose()
            except RuntimeError:  # it is being iterated in another task
                pass
            finally:
                await self._response.aclose()

    async def _prime(self) -> None:
        """Read until the first event with data, checking everything that arrives with it.

        Keep-alives (comments, empty events) neither count nor need relaying before the response
        starts. A stream that ends here, or that is already complete, is checked in full."""
        while True:
            items = await self._read()
            if items is None:
                raise UpstreamError(
                    502,
                    "upstream_protocol_error",
                    retryable=False,
                    detail="stream ended before the first event",
                )
            first = any(isinstance(i, sse.ServerSentEvent) and i.data.strip() for i in items)
            self._pending += [i for i in self._process(items) if not isinstance(i, KeepAlive)]
            if first or self._ended:
                return

    async def _read(self) -> list[sse.ServerSentEvent | sse.Comment] | None:
        """The events and comments in the next chunk of the body, or None at its end."""
        try:
            chunk = await anext(self._chunks)
        except StopAsyncIteration:
            self._decoder.close()  # an unfinished event is discarded
            return None
        except httpx2.RequestError as exc:
            raise _transport_error(exc) from None
        except httpx2.StreamError:  # the stream was closed under us
            raise UpstreamError(
                502, "upstream_unavailable", retryable=False, detail="stream closed"
            ) from None
        try:
            return self._decoder.feed(chunk)
        except sse.DecodeError as exc:
            raise UpstreamError(
                502, "upstream_protocol_error", retryable=False, detail=str(exc)
            ) from None

    def _process(self, items: list[sse.ServerSentEvent | sse.Comment] | None) -> list[StreamItem]:
        """Normalize decoded items. None means the body ended. Keep-alives are merged into one."""
        if self._ended:
            return []
        out: list[StreamItem] = []
        alive = False
        try:
            for item in items or []:
                if isinstance(item, sse.Comment):
                    alive = True
                    continue
                for result in self._normalizer.accept(item):
                    if isinstance(result, KeepAlive):
                        alive = True
                    else:
                        out.append(result)
                if self._normalizer.done:
                    break
            if items is None or self._normalizer.done:
                out.append(self._normalizer.end())
                self._ended = True
        except ReportedError as exc:
            raise UpstreamError(502, "upstream_error", retryable=False, detail=exc.detail) from None
        except ProtocolViolation as exc:
            raise UpstreamError(
                502, "upstream_protocol_error", retryable=False, detail=exc.detail
            ) from None
        if alive and not self._ended:
            out.append(KEEP_ALIVE)
        return out


def error_for_status(
    status: int, headers: httpx2.Headers | Mapping[str, str], detail: str = ""
) -> UpstreamError:
    """The client-facing error for an upstream response that is not a 200."""
    detail = detail or f"HTTP {status}"
    if status == 429:
        return UpstreamError(
            429,
            "upstream_rate_limited",
            retryable=True,
            retry_after=retry_after_seconds(headers),
            detail=detail,
        )
    if status in (503, 529):
        return UpstreamError(
            503,
            "upstream_overloaded",
            retryable=True,
            retry_after=retry_after_seconds(headers),
            detail=detail,
        )
    if status in (408, 504):
        return UpstreamError(504, "upstream_timeout", retryable=True, detail=detail)
    if status in (400, 404, 413, 422):
        return UpstreamError(400, "upstream_rejected_request", retryable=False, detail=detail)
    if status in (401, 403):
        return UpstreamError(502, "upstream_auth_failed", retryable=False, detail=detail)
    if status >= 500 or status == 409:
        return UpstreamError(502, "upstream_unavailable", retryable=True, detail=detail)
    if status >= 400:
        return UpstreamError(502, "upstream_unavailable", retryable=False, detail=detail)
    return UpstreamError(502, "upstream_protocol_error", retryable=False, detail=detail)


def retry_after_seconds(
    headers: httpx2.Headers | Mapping[str, str], now: float | None = None
) -> int | None:
    """Whole seconds to wait, from ``retry-after-ms`` or ``retry-after``; None when unusable."""
    milliseconds = headers.get("retry-after-ms")
    if milliseconds is not None and _SECONDS.fullmatch(milliseconds.strip()):
        seconds = float(milliseconds) / 1000
        if seconds > 0:
            return _whole_seconds(seconds)
    value = headers.get("retry-after")
    if value is None:
        return None
    value = value.strip()
    if _SECONDS.fullmatch(value):
        seconds = float(value)
    else:
        try:
            moment = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if moment.tzinfo is None:
            return None
        seconds = moment.timestamp() - (time.time() if now is None else now)
    return _whole_seconds(seconds) if seconds > 0 else None


def _whole_seconds(seconds: float) -> int:
    return min(max(math.ceil(seconds), 1), 3600)


def _request_headers(config: UpstreamConfig) -> dict[str, str]:
    headers = {
        "accept": "text/event-stream",
        "accept-encoding": "identity",  # compression would only delay the stream
        "content-type": "application/json",
        "user-agent": USER_AGENT,
    }
    if config.api_key:
        headers["authorization"] = f"Bearer {config.api_key}"
    return headers


def _encode(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _check_stream_headers(headers: httpx2.Headers) -> None:
    media_type = headers.get("content-type", "").partition(";")[0].strip().lower()
    if media_type != "text/event-stream":
        raise UpstreamError(
            502, "upstream_protocol_error", retryable=False, detail="not an event stream"
        )
    encodings = {part.strip().lower() for part in headers.get("content-encoding", "").split(",")}
    if not encodings <= _DECODABLE_ENCODINGS:
        raise UpstreamError(
            502, "upstream_protocol_error", retryable=False, detail="unsupported content encoding"
        )


async def _status_error(response: httpx2.Response) -> UpstreamError:
    """Map an error response, reading a little of its body for a safe hint in the logs."""
    hint = ""
    try:
        body = bytearray()
        async with aclosing(response.aiter_bytes()) as chunks:
            async for chunk in chunks:
                body += chunk
                if len(body) >= _ERROR_BODY_LIMIT:
                    break
        hint = _error_hint(bytes(body[:_ERROR_BODY_LIMIT]))
    except (httpx2.RequestError, httpx2.StreamError):
        pass
    return error_for_status(
        response.status_code, response.headers, f"HTTP {response.status_code}{hint}"
    )


def _error_hint(body: bytes) -> str:
    """``type`` and ``code`` from an OpenAI-style error body, when they look like identifiers."""
    try:
        error = json.loads(body).get("error")
    except (ValueError, RecursionError, AttributeError):
        return ""
    if not isinstance(error, dict):
        return ""
    parts = [
        f"{key}={error[key]}"
        for key in ("type", "code")
        if isinstance(error.get(key), str) and _HINT.fullmatch(error[key])
    ]
    return f" ({', '.join(parts)})" if parts else ""


def _transport_error(exc: httpx2.RequestError) -> UpstreamError:
    if isinstance(exc, httpx2.PoolTimeout):
        return UpstreamError(
            503, "gateway_overloaded", retryable=True, detail="connection pool full"
        )
    if isinstance(exc, httpx2.TimeoutException):
        return UpstreamError(504, "upstream_timeout", retryable=True, detail=type(exc).__name__)
    if isinstance(exc, httpx2.DecodingError):
        return UpstreamError(
            502, "upstream_protocol_error", retryable=False, detail="content decoding failed"
        )
    return UpstreamError(502, "upstream_unavailable", retryable=True, detail=type(exc).__name__)


async def _close(stream: UpstreamStream | None, response: httpx2.Response | None) -> None:
    if stream is not None:
        await stream.aclose()
    elif response is not None:
        with anyio.CancelScope(shield=True):
            await response.aclose()
