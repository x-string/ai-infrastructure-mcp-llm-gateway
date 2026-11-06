"""Task 4: failover between providers, and what each outcome charges the tenant."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import httpx2
import pytest

from ai_gateway.model_router import errors
from ai_gateway.model_router.limiter import TokenLimiter
from ai_gateway.model_router.providers import Provider
from ai_gateway.model_router.router import Router
from ai_gateway.model_router.storage import Database
from ai_gateway.model_router.tenants import Tenant
from ai_gateway.model_router.tokens import request_cost
from tests.model_router.scripted import (
    Providers,
    completion,
    delayed,
    drip,
    fail,
    hang,
    ok,
    raw,
    status,
)

DEADLINE = 0.05
TENANT = Tenant("acme", 50_000)
PAYLOAD = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}
COST = request_cost(PAYLOAD, 64)
PROMPT = COST - 64
USED = 17  # the scripted providers' usage


class Harness:
    def __init__(self, database: Database, providers: Providers, client: httpx2.AsyncClient):
        self.database, self.providers = database, providers
        self.router = Router(
            TokenLimiter(database),
            client,
            [
                Provider("primary", "http://primary.test/v1/chat/completions", DEADLINE),
                Provider("secondary", "http://secondary.test/v1/chat/completions", 1.0),
            ],
        )
        self.count = 0

    async def complete(self, tenant: Tenant = TENANT):
        self.count += 1
        return await self.router.complete(
            tenant=tenant, request_id=f"req-{self.count}", payload=PAYLOAD, max_tokens=64
        )

    async def rows(self) -> list[tuple]:
        await asyncio.wrap_future(self.database.submit(lambda _c: None))  # settlement is queued
        return self.database.submit(
            lambda c: c.execute(
                "SELECT request_id, tokens, status FROM token_usage ORDER BY request_id"
            ).fetchall()
        ).result()


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[Harness]:
    database = Database(tmp_path / "t4.db")
    database.open()
    providers = Providers()
    async with httpx2.AsyncClient(transport=providers.transport()) as client:
        yield Harness(database, providers, client)
    database.close()


async def test_the_primary_answers_and_actual_usage_is_charged(harness: Harness) -> None:
    harness.providers.script("primary", ok())
    harness.providers.script("secondary", ok())
    routed = await harness.complete()
    assert routed.provider == "primary"
    assert routed.body["choices"][0]["message"]["content"] == "Hello!"
    assert routed.remaining == TENANT.tokens_per_minute - USED
    assert harness.providers.calls("secondary") == 0
    assert await harness.rows() == [("req-1", USED, "committed")]


FAILOVERS = [
    ("rate-limited", status(429), 0),
    ("unreachable", fail(httpx2.ConnectError("refused")), 0),
    ("pool-full", fail(httpx2.PoolTimeout("full")), 0),
    ("timeout", hang(), PROMPT),
    ("slow-drip", drip(DEADLINE / 4), PROMPT),
    ("server-error", status(500), PROMPT),
    ("broken", fail(httpx2.RemoteProtocolError("reset")), PROMPT),
    ("malformed", raw(200, b"{not json"), PROMPT),
]


@pytest.mark.parametrize(("behaviour", "primary_charge"), [f[1:] for f in FAILOVERS],
                         ids=[f[0] for f in FAILOVERS])  # fmt: skip
async def test_failover_to_the_secondary_and_what_the_primary_costs(
    harness: Harness, behaviour, primary_charge: int
) -> None:
    harness.providers.script("primary", behaviour)
    harness.providers.script("secondary", ok())
    routed = await harness.complete()
    assert routed.provider == "secondary"
    assert harness.providers.calls("primary") == harness.providers.calls("secondary") == 1
    assert await harness.rows() == [("req-1", primary_charge + USED, "committed")]


@pytest.mark.parametrize(
    ("behaviour", "status_code", "code"),
    [
        (status(400), 400, "upstream_rejected_request"),
        (status(413), 400, "upstream_rejected_request"),
        (status(422), 400, "upstream_rejected_request"),
        (status(401), 502, "upstream_configuration_error"),
        (status(403), 502, "upstream_configuration_error"),
        (status(404), 502, "upstream_configuration_error"),
    ],
)
async def test_no_failover_when_the_request_or_the_setup_is_wrong(
    harness: Harness, behaviour, status_code: int, code: str
) -> None:
    harness.providers.script("primary", behaviour)
    harness.providers.script("secondary", ok())
    with pytest.raises(errors.GatewayError) as excinfo:
        await harness.complete()
    assert (excinfo.value.status, excinfo.value.code) == (status_code, code)
    assert harness.providers.calls("secondary") == 0
    assert await harness.rows() == []  # nothing was generated: the reservation is released


@pytest.mark.parametrize(
    ("primary", "secondary", "status_code", "code", "charge"),
    [
        (status(429), status(429), 503, "upstream_unavailable", 0),
        (hang(), hang(), 504, "upstream_timeout", 2 * PROMPT),
        (status(429), status(500), 502, "upstream_error", PROMPT),
        (status(500), fail(httpx2.ConnectError("refused")), 502, "upstream_error", PROMPT),
        (hang(), raw(200, b"[]"), 502, "upstream_error", 2 * PROMPT),
    ],
    ids=["both-429", "both-timeout", "429-then-500", "500-then-unreachable", "timeout-then-junk"],
)
async def test_when_both_fail_the_last_attempt_decides_the_error(
    harness: Harness, primary, secondary, status_code: int, code: str, charge: int
) -> None:
    harness.router._providers = (  # the secondary gets the short deadline too
        harness.router._providers[0],
        Provider("secondary", "http://secondary.test/v1/chat/completions", DEADLINE),
    )
    harness.providers.script("primary", primary)
    harness.providers.script("secondary", secondary)
    with pytest.raises(errors.GatewayError) as excinfo:
        await harness.complete()
    assert (excinfo.value.status, excinfo.value.code) == (status_code, code)
    expected = [("req-1", charge, "committed")] if charge else []
    assert await harness.rows() == expected


async def test_missing_usage_keeps_the_whole_reservation(harness: Harness) -> None:
    harness.providers.script("primary", ok(usage=None))
    await harness.complete()
    assert await harness.rows() == [("req-1", COST, "committed")]


@pytest.mark.parametrize("delay", [DEADLINE * 0.9, DEADLINE, DEADLINE * 1.1])
async def test_an_answer_at_the_deadline_gives_one_response_and_one_settlement(
    harness: Harness, delay: float
) -> None:
    harness.providers.script("primary", delayed(delay, ok()))
    harness.providers.script("secondary", ok())
    routed = await harness.complete()
    assert routed.provider in ("primary", "secondary")
    primary_charge = 0 if routed.provider == "primary" else PROMPT
    assert await harness.rows() == [("req-1", primary_charge + USED, "committed")]


async def test_a_request_bigger_than_the_whole_budget_is_refused_before_anything(
    harness: Harness,
) -> None:
    harness.providers.script("primary", ok())
    with pytest.raises(errors.GatewayError) as excinfo:
        await harness.complete(Tenant("tiny", COST - 1))
    assert (excinfo.value.status, excinfo.value.code) == (400, "request_too_large")
    assert harness.providers.requests == []
    assert await harness.rows() == []


async def test_over_the_limit_no_provider_is_called(harness: Harness) -> None:
    harness.providers.script("primary", ok(usage={"prompt_tokens": 1, "completion_tokens": 1,
                                                  "total_tokens": COST}))  # fmt: skip
    tenant = Tenant("small", COST + COST // 2)
    await harness.complete(tenant)
    with pytest.raises(errors.GatewayError) as excinfo:
        await harness.complete(tenant)
    assert excinfo.value.status == 429
    assert harness.providers.calls("primary") == 1


async def test_cancellation_mid_attempt_charges_the_prompt_and_leaves_no_reservation(
    harness: Harness,
) -> None:
    harness.providers.script("primary", hang())
    harness.router._providers = (Provider("primary", "http://primary.test/v1", 30),)
    with anyio.move_on_after(0.1) as scope:
        await harness.complete()
    assert scope.cancelled_caught
    assert await harness.rows() == [("req-1", PROMPT, "committed")]


async def test_a_settlement_failure_never_changes_the_response(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    async def break_storage(request: httpx2.Request) -> httpx2.Response:
        harness.database.submit(lambda c: c.execute("DROP TABLE token_usage"))
        return httpx2.Response(200, json=completion())

    harness.providers.script("primary", break_storage)
    routed = await harness.complete()
    assert routed.provider == "primary"
    await asyncio.wrap_future(harness.database.submit(lambda _c: None))
    assert "settlement failed for request req-1" in caplog.text
    with pytest.raises(sqlite3.OperationalError):
        harness.database.submit(lambda c: c.execute("SELECT * FROM token_usage")).result()
