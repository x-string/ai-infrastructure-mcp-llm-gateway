"""A mock OpenAI-compatible provider for the Task 3 gateway: ``t3-mock-provider``.

It streams chat completions in small pieces, so PII in the reply arrives split across deltas. The
request's ``model`` picks the behaviour, and any other model name gets the ``--mode`` default:

| Model | Behaviour |
|---|---|
| ``echo`` | streams the last user message back: put PII in your prompt and watch it disappear |
| ``pii`` | streams a fixed reply full of PII |
| ``status-<code>`` | answers with that HTTP error, e.g. ``status-429`` (with ``retry-after: 2``) |
| ``midstream-error`` | streams ``Mail john.do``, then an error event |
| ``hang`` | streams one chunk, then nothing |
| ``slow-headers`` | waits 30 s before answering at all |
| ``garbage`` | sends an event that is not JSON |
| ``tool-calls`` | streams a tool call, which the gateway refuses |

The events are written by hand, independently of the gateway's own encoder. With
``--expose-stats``, ``GET /_stats`` reports how many streams completed or were cancelled.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

PII_REPLY = (
    "Sure. Contact John at john.doe@example.com or call about SSN 123-45-6789. "
    "The card on file is 4111 1111 1111 1111."
)


@dataclass
class Stats:
    started: int = 0
    completed: int = 0  # sent [DONE]
    ended: int = 0  # stopped on purpose (error scenarios)
    cancelled: int = 0  # closed by the server because the gateway went away
    models: list[str] = field(default_factory=list)


def event(payload: object) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def chunk(
    model: str, delta: dict | None = None, finish: str | None = None, **extra: object
) -> bytes:
    choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]
    return event(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            **extra,
        }
    )


def last_user_text(body: dict) -> str:
    for message in reversed(body.get("messages") or []):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "Hello!"


def create_app(
    *, mode: str = "echo", delay: float = 0.02, piece: int = 3, expose_stats: bool = False
) -> Starlette:
    stats = Stats()

    async def stream(model: str, scenario: str, body: dict) -> AsyncIterator[bytes]:
        stats.started += 1
        outcome = "ended"
        try:
            yield chunk(model, {"role": "assistant", "content": ""})
            if scenario == "hang":
                await anyio.sleep_forever()
            if scenario == "garbage":
                yield b"data: {this is not json\n\n"
                return
            if scenario == "tool-calls":
                call = {"index": 0, "id": "c1", "type": "function",
                        "function": {"name": "f", "arguments": '{"email": "a@b.co"}'}}  # fmt: skip
                yield chunk(model, {"tool_calls": [call]})
                yield chunk(model, {}, "tool_calls")
                yield b"data: [DONE]\n\n"
                return
            text = {"echo": last_user_text(body), "pii": PII_REPLY}.get(scenario, "Mail john.do")
            limit = body.get("max_tokens") or body.get("max_completion_tokens")
            pieces = [text[i : i + piece] for i in range(0, len(text), piece)]
            truncated = isinstance(limit, int) and limit < len(pieces)
            if truncated:
                pieces = pieces[:limit]
            for part in pieces:
                await anyio.sleep(delay)
                yield chunk(model, {"content": part})
            if scenario == "midstream-error":
                yield event({"error": {"message": "The model crashed.", "type": "server_error"}})
                return
            yield chunk(model, {}, "length" if truncated else "stop")
            if (body.get("stream_options") or {}).get("include_usage"):
                count = len(pieces)
                usage = {"prompt_tokens": 7, "completion_tokens": count, "total_tokens": 7 + count}
                yield chunk(model, None, usage=usage)
            yield b"data: [DONE]\n\n"
            outcome = "completed"
        except BaseException:  # GeneratorExit or cancellation: the connection went away
            outcome = "cancelled"
            raise
        finally:
            setattr(stats, outcome, getattr(stats, outcome) + 1)

    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        model = str(body.get("model", "mock"))
        stats.models.append(model)
        scenario = model if model in _SCENARIOS or model.startswith("status-") else mode
        if scenario.startswith("status-"):
            status = int(scenario.removeprefix("status-"))
            headers = {"retry-after": "2"} if status in (429, 503) else {}
            error = {"error": {"message": "mock failure", "type": "mock_error", "code": "mock"}}
            return JSONResponse(error, status_code=status, headers=headers)
        if scenario == "slow-headers":  # until 30 s pass or the caller gives up
            for _ in range(300):
                if await request.is_disconnected():
                    return Response(status_code=499)
                await anyio.sleep(0.1)
        return StreamingResponse(stream(model, scenario, body), media_type="text/event-stream")

    async def get_stats(request: Request) -> Response:
        return JSONResponse(
            {
                "started": stats.started,
                "completed": stats.completed,
                "ended": stats.ended,
                "cancelled": stats.cancelled,
                "models": stats.models,
            }
        )

    routes = [Route("/v1/chat/completions", chat_completions, methods=["POST"])]
    if expose_stats:
        routes.append(Route("/_stats", get_stats))
    return Starlette(routes=routes)


_SCENARIOS = frozenset(
    {"echo", "pii", "midstream-error", "hang", "slow-headers", "garbage", "tool-calls"}
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock OpenAI-compatible provider for Task 3.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--mode", default="echo", choices=sorted(_SCENARIOS))
    parser.add_argument("--delay", type=float, default=0.02, help="seconds between deltas")
    parser.add_argument("--piece", type=int, default=3, help="characters per delta")
    parser.add_argument("--expose-stats", action="store_true", help="serve GET /_stats")
    args = parser.parse_args()
    app = create_app(
        mode=args.mode, delay=args.delay, piece=args.piece, expose_stats=args.expose_stats
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        server_header=False,
        timeout_graceful_shutdown=1,
    )


if __name__ == "__main__":
    main()
