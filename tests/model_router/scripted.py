"""Scripted OpenAI-compatible providers for the Task 4 tests, on httpx2's MockTransport."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import anyio
import httpx2

USAGE = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}


def completion(content: str = "Hello!", usage: object = USAGE, **overrides: object) -> dict:
    body: dict = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1_800_000_000,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],  # fmt: skip
        "usage": usage,
    }
    if usage is None:
        del body["usage"]
    body.update(overrides)
    return body


Behaviour = Callable[[httpx2.Request], object]


def named(behaviour: Behaviour, name: str) -> Behaviour:
    """Give a behaviour a readable name, which pytest uses as its test id."""
    behaviour.__name__ = behaviour.__qualname__ = name
    return behaviour


def ok(**kwargs: object) -> Behaviour:
    return named(lambda _request: httpx2.Response(200, json=completion(**kwargs)), "ok")


def status(code: int, body: object = None) -> Behaviour:
    content = json.dumps(body or {"error": {"message": "no", "type": "x"}}).encode()
    return named(lambda _request: httpx2.Response(code, content=content), f"http{code}")


def raw(code: int, content: bytes) -> Behaviour:
    return named(lambda _request: httpx2.Response(code, content=content), f"raw{code}")


def fail(exc: Exception) -> Behaviour:
    def behave(_request: httpx2.Request) -> object:
        raise exc

    return named(behave, type(exc).__name__)


def hang() -> Behaviour:
    async def behave(_request: httpx2.Request) -> object:
        await anyio.sleep_forever()

    return named(behave, "hang")


def delayed(seconds: float, then: Behaviour) -> Behaviour:
    async def behave(request: httpx2.Request) -> object:
        await anyio.sleep(seconds)
        return then(request)

    return named(behave, f"after{seconds:g}s")


def drip(interval: float, pieces: int = 20) -> Behaviour:
    """Headers at once, then the body a few bytes at a time: each read is quick, the whole slow."""
    body = json.dumps(completion()).encode()
    size = -(-len(body) // pieces)

    async def chunks() -> AsyncIterator[bytes]:
        for start in range(0, len(body), size):
            await anyio.sleep(interval)
            yield body[start : start + size]

    return named(lambda _request: httpx2.Response(200, content=chunks()), "drip")


@dataclass
class Providers:
    """Answers requests to ``http://<name>.test`` with that provider's next behaviour."""

    behaviours: dict[str, list[Behaviour]] = field(default_factory=dict)
    requests: list[httpx2.Request] = field(default_factory=list)

    def script(self, name: str, *behaviours: Behaviour) -> None:
        self.behaviours[name] = list(behaviours)

    def calls(self, name: str) -> int:
        return sum(1 for request in self.requests if request.url.host == f"{name}.test")

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        queue = self.behaviours[request.url.host.removesuffix(".test")]
        behaviour = queue.pop(0) if len(queue) > 1 else queue[0]
        result = behaviour(request)
        if hasattr(result, "__await__"):
            result = await result
        return result

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handle)
