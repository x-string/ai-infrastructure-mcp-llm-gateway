"""An OpenAI-compatible provider for the Task 4 router's demos and end-to-end tests.

``t4-mock-provider`` answers ``POST /v1/chat/completions`` with a completion that echoes the last
user message, and token usage. ``--mode`` picks a behaviour:

* ``ok``: answer at once
* ``rate_limited``: 429 with ``Retry-After: 1``
* ``slow``: answer after ``--delay`` seconds
* ``slow_drip``: send headers at once, then the body a little at a time over ``--delay`` seconds
* ``error``: 500, with a body the router must never pass on
* ``malformed``: 200 with a body that is not JSON
* ``no_usage``: answer without usage

Options: ``--port`` (default 9201), ``--host`` (default 127.0.0.1), ``--name`` (shown in the
answer), ``--control`` to add ``GET /_calls`` (requests so far) and ``PUT /_mode`` (change mode),
which the end-to-end tests use. It has no authentication: bind it to localhost.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import AsyncIterator, Sequence
from typing import Final

import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

MODES: Final = ("ok", "rate_limited", "slow", "slow_drip", "error", "malformed", "no_usage")
LEAKY_ERROR: Final = "internal stack: provider-secret-0xDEADBEEF"


class MockState:
    def __init__(self, mode: str, delay: float, name: str) -> None:
        self.mode, self.delay, self.name, self.calls = mode, delay, name, 0


def completion(body: dict, name: str, *, usage: bool = True) -> dict:
    messages = body.get("messages") or [{}]
    last = next((m for m in reversed(messages) if m.get("role") == "user"), messages[-1])
    content = f"[{name}] {str(last.get('content', ''))[:200]}"
    result: dict = {
        "id": f"chatcmpl-{name}-{time.time_ns()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(body.get("model", "mock")),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],  # fmt: skip
    }
    if usage:
        prompt = max(1, len(json.dumps(messages)) // 4)
        completion_tokens = max(1, len(content) // 4)
        result["usage"] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt + completion_tokens,
        }
    return result


def create_app(mode: str = "ok", delay: float = 0.0, name: str = "mock", *, control: bool = False):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    state = MockState(mode, delay, name)

    async def completions(request: Request) -> Response:
        state.calls += 1
        try:
            body = json.loads(await request.body())
        except ValueError:
            return JSONResponse({"error": {"message": "invalid JSON"}}, status_code=400)
        if state.mode == "rate_limited":
            return JSONResponse(
                {"error": {"message": "Rate limit reached", "type": "requests"}},
                status_code=429,
                headers={"retry-after": "1"},
            )
        if state.mode == "error":
            return JSONResponse({"error": {"message": LEAKY_ERROR}}, status_code=500)
        if state.mode == "malformed":
            return Response(b'{"choices": [', media_type="application/json")
        if state.mode == "slow":
            await anyio.sleep(state.delay)
        answer = completion(body, state.name, usage=state.mode != "no_usage")
        if state.mode == "slow_drip":
            return StreamingResponse(_drip(json.dumps(answer).encode(), state.delay),
                                     media_type="application/json")  # fmt: skip
        return JSONResponse(answer)

    async def calls(_request: Request) -> Response:
        return JSONResponse({"calls": state.calls, "mode": state.mode})

    async def set_mode(request: Request) -> Response:
        update = json.loads(await request.body())
        if update.get("mode") not in MODES:
            return JSONResponse({"error": "unknown mode"}, status_code=400)
        state.mode = update["mode"]
        state.delay = float(update.get("delay", state.delay))
        return JSONResponse({"mode": state.mode, "delay": state.delay})

    async def healthz(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    routes = [
        Route("/v1/chat/completions", completions, methods=["POST"]),
        Route("/healthz", healthz),
    ]
    if control:
        routes += [Route("/_calls", calls), Route("/_mode", set_mode, methods=["PUT"])]
    return Starlette(routes=routes)


async def _drip(body: bytes, seconds: float) -> AsyncIterator[bytes]:
    pieces = 20
    size = -(-len(body) // pieces)
    for start in range(0, len(body), size):
        yield body[start : start + size]
        await anyio.sleep(seconds / pieces)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="t4-mock-provider", description="An OpenAI-compatible provider for the model router."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9201)
    parser.add_argument("--mode", choices=MODES, default="ok")
    parser.add_argument("--delay", type=float, default=5.0, help="seconds, for slow modes")
    parser.add_argument("--name", default="mock", help="shown in the answers")
    parser.add_argument("--control", action="store_true", help="add GET /_calls and PUT /_mode")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    uvicorn.run(
        create_app(args.mode, args.delay, args.name, control=args.control),
        host=args.host,
        port=args.port,
        server_header=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
