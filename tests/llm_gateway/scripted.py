"""A scripted upstream provider for httpx2's MockTransport, shared by the Task 3 tests."""

from __future__ import annotations

import anyio
import httpx2

from tests.llm_gateway.provider_streams import USAGE, ch, chunk, d

HANG = object()

ROLE = d(chunk([ch({"role": "assistant", "content": ""})])).encode()
HELLO = d(chunk([ch({"content": "Hello"})])).encode()
WORLD = d(chunk([ch({"content": " world"})])).encode()
STOP = d(chunk([ch({}, "stop")])).encode()
USAGE_CHUNK = d(chunk([], usage=USAGE)).encode()
DONE = b"data: [DONE]\n\n"


def content(text: str) -> bytes:
    return d(chunk([ch({"content": text})])).encode()


class Upstream:
    """A scripted upstream: records each request, and how much of each body was read.

    Body parts are bytes to send, HANG to wait forever, an ``anyio.Event`` to wait for, or an
    exception to raise.
    """

    def __init__(self, *parts: object, status: int = 200, headers: dict | None = None) -> None:
        self.parts = list(parts)
        self.status = status
        self.headers = {"content-type": "text/event-stream"} if headers is None else headers
        self.before_headers: object = None  # HANG, or an exception to raise instead of answering
        self.requests: list[httpx2.Request] = []
        self.pulled = 0
        self.closed = 0

    @property
    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handle)

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.before_headers is HANG:
            await anyio.sleep_forever()
        if isinstance(self.before_headers, BaseException):
            raise self.before_headers
        return httpx2.Response(self.status, headers=self.headers, content=self._body())

    async def _body(self):
        try:
            for part in self.parts:
                if part is HANG:
                    await anyio.sleep_forever()
                if isinstance(part, anyio.Event):
                    await part.wait()
                    continue
                if isinstance(part, BaseException):
                    raise part
                self.pulled += 1
                yield part
        finally:
            self.closed += 1
