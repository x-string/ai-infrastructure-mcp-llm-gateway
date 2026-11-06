"""Task 3 end to end: the official OpenAI SDK -> the real gateway -> the mock provider, over TCP.

Both services run as subprocesses from their entry points. The mock streams its reply three
characters at a time, so every piece of PII arrives split across deltas.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx2
import openai
import pytest

from ai_gateway.llm_gateway.redaction import REDACTED

pytestmark = pytest.mark.e2e

R = REDACTED
PII = "Mail john.doe@example.com, SSN 123-45-6789, card 4111 1111 1111 1111 now."


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until_listening(port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"nothing is listening on port {port}")


@dataclass
class Stack:
    base_url: str
    stats_url: str

    def client(self) -> openai.OpenAI:
        return openai.OpenAI(base_url=self.base_url, api_key="unused", max_retries=0, timeout=20)

    def stats(self) -> dict:
        return httpx2.get(self.stats_url).json()


@pytest.fixture(scope="module")
def stack() -> Iterator[Stack]:
    provider_port, gateway_port = free_port(), free_port()
    env = {
        **os.environ,
        "T3_PORT": str(gateway_port),
        "T3_UPSTREAM_URL": f"http://127.0.0.1:{provider_port}/v1/chat/completions",
        "T3_FIRST_EVENT_TIMEOUT": "1",
        "T3_READ_TIMEOUT": "1",
    }
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "ai_gateway.llm_gateway.mock_provider", "--port",
             str(provider_port), "--delay", "0.02", "--expose-stats"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),
        subprocess.Popen(
            [sys.executable, "-m", "ai_gateway.llm_gateway"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),
    ]  # fmt: skip
    try:
        wait_until_listening(provider_port)
        wait_until_listening(gateway_port)
        yield Stack(
            base_url=f"http://127.0.0.1:{gateway_port}/v1",
            stats_url=f"http://127.0.0.1:{provider_port}/_stats",
        )
    finally:
        for process in processes:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def ask(stack: Stack, content: str = PII, model: str = "echo", **options: object):
    return stack.client().chat.completions.create(
        model=model, stream=True, messages=[{"role": "user", "content": content}], **options
    )


def test_the_sdk_streams_a_redacted_reply_incrementally(stack: Stack) -> None:
    started, arrivals, parts, usage = time.monotonic(), [], [], None
    for chunk in ask(stack, stream_options={"include_usage": True}):
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            arrivals.append(time.monotonic() - started)
            parts.append(chunk.choices[0].delta.content)
    assert "".join(parts) == f"Mail {R}, SSN {R}, card {R} now."
    assert len(parts) >= 5
    assert arrivals[0] < arrivals[-1] - 0.2  # the first text arrived well before the end
    assert usage is not None and usage.total_tokens > 0


def test_the_sdk_stream_helper_accepts_the_rebuilt_chunks(stack: Stack) -> None:
    with stack.client().chat.completions.stream(
        model="pii", messages=[{"role": "user", "content": "x"}]
    ) as stream:
        final = stream.get_final_completion()
    message = final.choices[0].message.content
    assert message == f"Sure. Contact John at {R} or call about SSN {R}. The card on file is {R}."
    assert final.choices[0].finish_reason == "stop"


def test_the_raw_stream_is_server_sent_events_ending_with_done(stack: Stack) -> None:
    body = {"model": "echo", "messages": [{"role": "user", "content": "hi there"}]}
    with httpx2.stream("POST", f"{stack.base_url}/chat/completions", json=body) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-request-id"]
        lines = [line for line in response.iter_lines() if line]
    assert lines[-1] == "data: [DONE]"
    for line in lines[:-1]:
        assert json.loads(line.removeprefix("data: "))["object"] == "chat.completion.chunk"


def test_a_provider_rate_limit_reaches_the_sdk_as_a_429(stack: Stack) -> None:
    with pytest.raises(openai.RateLimitError) as excinfo:
        ask(stack, model="status-429")
    assert excinfo.value.response.headers["retry-after"] == "2"
    assert excinfo.value.code == "upstream_rate_limited"


def test_a_bad_gateway_key_is_a_502_that_must_not_be_retried(stack: Stack) -> None:
    with pytest.raises(openai.APIStatusError) as excinfo:
        ask(stack, model="status-401")
    assert excinfo.value.status_code == 502
    assert excinfo.value.response.headers["x-should-retry"] == "false"


def test_an_error_mid_stream_raises_in_the_sdk_without_leaking_held_back_pii(stack: Stack) -> None:
    received = []
    with pytest.raises(openai.APIError) as excinfo:
        for chunk in ask(stack, model="midstream-error"):
            if chunk.choices and chunk.choices[0].delta.content:
                received.append(chunk.choices[0].delta.content)
    assert excinfo.value.code == "upstream_error"
    assert "".join(received) == "Mail "  # "john.do" was held back, then discarded


@pytest.mark.parametrize(
    ("model", "code"),
    [("tool-calls", "upstream_protocol_error"), ("garbage", "upstream_protocol_error"),
     ("hang", "upstream_timeout")],
)  # fmt: skip
def test_unrelayable_or_silent_streams_end_with_an_error(
    stack: Stack, model: str, code: str
) -> None:
    with pytest.raises(openai.APIError) as excinfo:
        list(ask(stack, model=model))
    assert excinfo.value.code == code


def test_a_provider_that_never_answers_is_a_504_at_the_first_event_deadline(stack: Stack) -> None:
    started = time.monotonic()
    with pytest.raises(openai.APIStatusError) as excinfo:
        ask(stack, model="slow-headers")
    assert excinfo.value.status_code == 504
    assert time.monotonic() - started < 5


def test_refused_parameters_are_a_400_naming_them(stack: Stack) -> None:
    with pytest.raises(openai.BadRequestError) as excinfo:
        ask(stack, tools=[{"type": "function", "function": {"name": "f"}}])
    assert "tools: tool calls are not supported" in excinfo.value.message


def test_a_client_that_hangs_up_mid_stream_stops_the_provider(stack: Stack) -> None:
    before = stack.stats()["cancelled"]
    stream = ask(stack, content="x" * 600)  # 200 deltas, 4 s at 20 ms each
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            break
    stream.close()
    deadline = time.monotonic() + 5
    while stack.stats()["cancelled"] == before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert stack.stats()["cancelled"] == before + 1
