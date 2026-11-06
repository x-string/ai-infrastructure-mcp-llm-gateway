"""Task 4 end to end: the router and two mock providers as real processes, over TCP.

The tenant is created with the ``t4-tenants`` command, the providers run as ``t4-mock-provider``
and the router as ``t4-router``, all from their entry points with their own environment.
"""

from __future__ import annotations

import concurrent.futures
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx2
import pytest

pytestmark = pytest.mark.e2e

BODY = {"model": "gpt-test", "messages": [{"role": "user", "content": "Hello router"}]}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until_healthy(url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx2.get(url, timeout=1).status_code == 200:
                return
        except httpx2.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"{url} did not become healthy")


@dataclass
class Stack:
    router_url: str
    primary_url: str
    secondary_url: str
    db_path: Path
    env: dict[str, str]
    router: subprocess.Popen

    def tenant(self, name: str, tokens_per_minute: int) -> str:
        added = subprocess.run(
            [sys.executable, "-m", "ai_gateway.model_router.tenants", "--db", str(self.db_path),
             "add", name, "--tokens-per-minute", str(tokens_per_minute)],
            capture_output=True, text=True, timeout=60, check=True,
        )  # fmt: skip
        return added.stdout.strip()

    def set_mode(self, provider_url: str, mode: str, delay: float = 0.0) -> None:
        httpx2.put(f"{provider_url}/_mode", json={"mode": mode, "delay": delay}, timeout=5)

    def calls(self, provider_url: str) -> int:
        return httpx2.get(f"{provider_url}/_calls", timeout=5).json()["calls"]

    def ask(self, key: str, body: dict = BODY) -> httpx2.Response:
        return httpx2.post(
            f"{self.router_url}/v1/chat/completions",
            json=body,
            headers={"authorization": f"Bearer {key}"},
            timeout=30,
        )

    def restart_router(self) -> None:
        self.router.terminate()
        self.router.wait(timeout=10)
        self.router = start_router(self.env)
        wait_until_healthy(f"{self.router_url}/healthz")


def start_router(env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "ai_gateway.model_router"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    db_path = tmp_path_factory.mktemp("t4") / "router.db"
    primary_port, secondary_port, router_port = free_port(), free_port(), free_port()
    providers = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ai_gateway.model_router.mock_provider",
                "--port",
                str(port),
                "--name",
                name,
                "--control",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )  # fmt: skip
        for name, port in (("primary", primary_port), ("secondary", secondary_port))
    ]
    env = {
        **os.environ,
        "T4_DB_PATH": str(db_path),
        "T4_PORT": str(router_port),
        "T4_PRIMARY_URL": f"http://127.0.0.1:{primary_port}/v1/chat/completions",
        "T4_SECONDARY_URL": f"http://127.0.0.1:{secondary_port}/v1/chat/completions",
        "T4_PRIMARY_DEADLINE_MS": "500",
    }
    stack = Stack(
        f"http://127.0.0.1:{router_port}",
        f"http://127.0.0.1:{primary_port}",
        f"http://127.0.0.1:{secondary_port}",
        db_path,
        env,
        router=None,  # type: ignore[arg-type]
    )
    subprocess.run(  # creates the database before the router starts
        [sys.executable, "-m", "ai_gateway.model_router.tenants", "--db", str(db_path), "list"],
        check=True, capture_output=True, timeout=60,
    )  # fmt: skip
    stack.router = start_router(env)
    try:
        for url in (stack.primary_url, stack.secondary_url, stack.router_url):
            wait_until_healthy(f"{url}/healthz")
        yield stack
    finally:
        for process in [*providers, stack.router]:
            process.terminate()
            process.wait(timeout=10)


@pytest.fixture(autouse=True)
def healthy_providers(stack: Stack) -> None:
    stack.set_mode(stack.primary_url, "ok")
    stack.set_mode(stack.secondary_url, "ok")


def test_the_primary_answers(stack: Stack) -> None:
    key = stack.tenant("acme", 50_000)
    response = stack.ask(key)
    assert response.status_code == 200
    assert response.headers["x-gateway-provider"] == "primary"
    assert response.json()["choices"][0]["message"]["content"] == "[primary] Hello router"
    assert response.headers["x-ratelimit-limit-tokens"] == "50000"


@pytest.mark.parametrize("mode", ["rate_limited", "slow", "slow_drip", "error", "malformed"])
def test_failover_to_the_secondary(stack: Stack, mode: str) -> None:
    key = stack.tenant(f"failover-{mode}", 50_000)
    stack.set_mode(stack.primary_url, mode, delay=3.0)
    before = stack.calls(stack.secondary_url)
    started = time.monotonic()
    response = stack.ask(key)
    assert response.status_code == 200
    assert response.headers["x-gateway-provider"] == "secondary"
    assert response.json()["choices"][0]["message"]["content"] == "[secondary] Hello router"
    assert stack.calls(stack.secondary_url) == before + 1
    if mode in ("slow", "slow_drip"):  # the 500 ms deadline, not the 3 s the primary would take
        assert time.monotonic() - started < 2.5


def test_both_providers_failing_gives_a_clean_error(stack: Stack) -> None:
    key = stack.tenant("both-down", 50_000)
    stack.set_mode(stack.primary_url, "error")
    stack.set_mode(stack.secondary_url, "error")
    response = stack.ask(key)
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "upstream_error"
    assert error["request_id"] == response.headers["x-request-id"]
    assert "provider-secret" not in response.text


def test_a_burst_gets_429_with_retry_after_and_the_window_survives_a_restart(stack: Stack) -> None:
    key = stack.tenant("bursty", 2_500)
    stack.set_mode(stack.primary_url, "slow", delay=0.3)  # keep the reservations in flight
    body = {**BODY, "max_tokens": 1_000}
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        responses = list(pool.map(lambda _: stack.ask(key, body), range(5)))
    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200, 200, 429, 429, 429]
    denied = next(r for r in responses if r.status_code == 429)
    assert 1 <= int(denied.headers["retry-after"]) <= 60
    assert denied.json()["error"]["code"] == "tenant_token_limit_exceeded"

    # Fill the window with real usage, restart the router, and the limit still holds.
    stack.set_mode(stack.primary_url, "ok")
    long = {"model": "gpt-test", "messages": [{"role": "user", "content": "x" * 9_000}],
            "max_tokens": 100}  # fmt: skip
    heavy = stack.tenant("heavy", 5_000)
    statuses = [stack.ask(heavy, long).status_code for _ in range(3)]
    assert statuses[0] == 200 and 429 in statuses
    stack.restart_router()
    after = stack.ask(heavy, long)
    assert after.status_code == 429
    assert int(after.headers["x-ratelimit-remaining-tokens"]) < 5_000


def test_unknown_keys_are_refused_before_any_provider_is_called(stack: Stack) -> None:
    before = stack.calls(stack.primary_url)
    response = stack.ask("sk-t4-not-a-real-key")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
    assert stack.calls(stack.primary_url) == before
