"""Task 4: the sliding-window limiter, against exact cases and a brute-force reference model."""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ai_gateway.model_router import errors
from ai_gateway.model_router.limiter import (
    WINDOW_MS,
    Admitted,
    Denied,
    TokenLimiter,
    admit,
    purge_expired,
    settle,
    used_tokens,
)
from ai_gateway.model_router.storage import Database, connect

NOW = 1_800_000_000_000
LIMIT = 1_000


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = connect(tmp_path / "t4.db")
    yield connection
    connection.close()


def take(connection: sqlite3.Connection, request_id: str, cost: int, now: int = NOW):
    return admit(
        connection, tenant_id="acme", request_id=request_id, cost=cost, limit=LIMIT, now_ms=now
    )


def rows(connection: sqlite3.Connection) -> list[tuple]:
    return connection.execute(
        "SELECT request_id, ts_ms, tokens, status FROM token_usage ORDER BY ts_ms, request_id"
    ).fetchall()


# --- admission ------------------------------------------------------------------------------------


def test_requests_are_admitted_up_to_exactly_the_limit(db: sqlite3.Connection) -> None:
    assert take(db, "a", 600) == Admitted(LIMIT, 400)
    assert take(db, "b", 400) == Admitted(LIMIT, 0)
    assert isinstance(take(db, "c", 1), Denied)


def test_tenants_have_separate_windows(db: sqlite3.Connection) -> None:
    take(db, "a", LIMIT)
    other = admit(db, tenant_id="other", request_id="b", cost=LIMIT, limit=LIMIT, now_ms=NOW)
    assert other == Admitted(LIMIT, 0)


def test_a_row_stops_counting_exactly_sixty_seconds_later(db: sqlite3.Connection) -> None:
    take(db, "a", LIMIT)
    assert isinstance(take(db, "b", 1, NOW + WINDOW_MS - 1), Denied)  # 59.999 s: still counted
    assert take(db, "c", LIMIT, NOW + WINDOW_MS) == Admitted(LIMIT, 0)  # 60 s: gone
    assert [r[0] for r in rows(db)] == ["c"]  # the expired row was evicted


def test_retry_after_is_the_whole_seconds_until_enough_has_expired(db: sqlite3.Connection) -> None:
    take(db, "a", 300, NOW)
    take(db, "b", 300, NOW + 10_000)
    take(db, "c", 400, NOW + 20_500)
    denied = take(db, "d", 500, NOW + 30_000)  # needs a and b gone: b leaves at NOW + 70 s
    assert denied == Denied(LIMIT, 0, 40)
    assert isinstance(take(db, "e", 500, NOW + 69_999), Denied)
    assert isinstance(take(db, "f", 500, NOW + 70_000), Admitted)


def test_retry_after_rounds_up_and_stays_within_one_to_sixty(db: sqlite3.Connection) -> None:
    take(db, "a", LIMIT, NOW)
    assert take(db, "b", 1, NOW + 59_999).retry_after == 1  # 1 ms left: still 1 s
    assert take(db, "c", 1, NOW + 1).retry_after == 60
    assert take(db, "d", 1, NOW + 500).retry_after == 60  # 59.5 s rounds up


def test_rows_dated_in_the_future_are_brought_back_to_now(db: sqlite3.Connection) -> None:
    take(db, "a", LIMIT, NOW + 3_600_000)  # recorded while the clock was an hour fast
    denied = take(db, "b", 1, NOW)  # the clock stepped back
    assert denied == Denied(LIMIT, 0, 60)
    assert rows(db)[0][1] == NOW


def test_used_tokens_reads_the_current_window(db: sqlite3.Connection) -> None:
    take(db, "a", 100, NOW)
    take(db, "b", 200, NOW + 30_000)
    assert used_tokens(db, tenant_id="acme", now_ms=NOW + 30_000) == 300
    assert used_tokens(db, tenant_id="acme", now_ms=NOW + 60_000) == 200


def test_purge_removes_expired_rows_of_every_tenant(db: sqlite3.Connection) -> None:
    take(db, "a", 100, NOW)
    admit(db, tenant_id="other", request_id="b", cost=100, limit=LIMIT, now_ms=NOW)
    take(db, "c", 100, NOW + 30_000)
    assert purge_expired(db, now_ms=NOW + 60_000) == 2
    assert [r[0] for r in rows(db)] == ["c"]


# --- settlement -----------------------------------------------------------------------------------


def settle_now(db: sqlite3.Connection, request_id: str, tokens: int, now: int = NOW) -> None:
    settle(db, tenant_id="acme", request_id=request_id, tokens=tokens, now_ms=now)


def test_settlement_replaces_the_estimate_with_actual_usage(db: sqlite3.Connection) -> None:
    take(db, "a", 900)
    settle_now(db, "a", 120)
    assert rows(db) == [("a", NOW, 120, "committed")]
    assert take(db, "b", 880) == Admitted(LIMIT, 0)


def test_settling_zero_releases_the_reservation(db: sqlite3.Connection) -> None:
    take(db, "a", 900)
    settle_now(db, "a", 0)
    assert rows(db) == []


def test_settlement_happens_once_only(db: sqlite3.Connection) -> None:
    take(db, "a", 900)
    settle_now(db, "a", 120)
    settle_now(db, "a", 999)
    settle_now(db, "a", 0)
    assert rows(db) == [("a", NOW, 120, "committed")]


def test_usage_of_a_reservation_that_expired_is_recorded_at_settlement(
    db: sqlite3.Connection,
) -> None:
    take(db, "a", 500)
    take(db, "b", 10, NOW + WINDOW_MS)  # evicts "a" while its request is still running
    settle_now(db, "a", 300, NOW + WINDOW_MS + 5_000)
    assert ("a", NOW + WINDOW_MS + 5_000, 300, "committed") in rows(db)


def test_settlement_is_scoped_to_the_tenant(db: sqlite3.Connection) -> None:
    take(db, "a", 500)
    settle(db, tenant_id="other", request_id="a", tokens=0, now_ms=NOW)
    assert rows(db) == [("a", NOW, 500, "reserved")]


# --- the brute-force reference model --------------------------------------------------------------


class Reference:
    """The window rules restated as plain Python over a list of rows."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.rows: dict[str, list] = {}  # request_id -> [ts, tokens, status]

    def _expire(self, now: int) -> None:
        for row in self.rows.values():
            row[0] = min(row[0], now)
        self.rows = {k: v for k, v in self.rows.items() if v[0] > now - WINDOW_MS}

    def _used_at(self, moment: int) -> int:
        return sum(t for ts, t, _ in self.rows.values() if ts > moment - WINDOW_MS)

    def admit(self, request_id: str, cost: int, now: int) -> Admitted | Denied:
        self._expire(now)
        used = self._used_at(now)
        if used + cost <= self.limit:
            self.rows[request_id] = [now, cost, "reserved"]
            return Admitted(self.limit, self.limit - used - cost)
        wait = next(s for s in range(1, 61) if self._used_at(now + s * 1000) + cost <= self.limit)
        return Denied(self.limit, self.limit - used, wait)

    def settle(self, request_id: str, tokens: int, now: int) -> None:
        row = self.rows.get(request_id)
        if row is None:
            if tokens > 0:
                self.rows[request_id] = [now, tokens, "committed"]
        elif row[2] == "reserved":
            if tokens <= 0:
                del self.rows[request_id]
            else:
                row[1:] = [tokens, "committed"]


STEPS = st.one_of(
    st.sampled_from([0, 1, 999, 1_000, 59_999, 60_000, 60_001]),
    st.integers(0, 70_000),
    st.integers(-120_000, -1),  # the clock stepping back
)
OPERATIONS = st.lists(
    st.tuples(st.sampled_from(["admit", "admit", "settle"]), STEPS, st.integers(1, 400)),
    min_size=1,
    max_size=40,
)
_DB_DIR = tempfile.mkdtemp(prefix="t4-limiter-")
_DB = connect(Path(_DB_DIR) / "reference.db", busy_timeout=10)
_LOCK = threading.Lock()
_tenants = iter(range(10**9))


@settings(max_examples=400, deadline=None)
@given(OPERATIONS, st.sampled_from([400, 1_000, 2_500]))
def test_the_limiter_matches_a_brute_force_reference(operations: list, limit: int) -> None:
    with _LOCK:
        tenant = f"t{next(_tenants)}"
        reference = Reference(limit)
        now, unsettled = NOW, []
        for index, (kind, step, amount) in enumerate(operations):
            now += step
            if kind == "admit" or not unsettled:
                request_id = f"{tenant}-{index}"
                got = admit(
                    _DB, tenant_id=tenant, request_id=request_id, cost=amount, limit=limit,
                    now_ms=now,
                )  # fmt: skip
                assert got == reference.admit(request_id, amount, now)
                if isinstance(got, Admitted):
                    unsettled.append(request_id)
            else:
                request_id = unsettled.pop(amount % len(unsettled))
                tokens = amount % 3 * amount  # sometimes 0 (a release)
                settle(_DB, tenant_id=tenant, request_id=request_id, tokens=tokens, now_ms=now)
                reference.settle(request_id, tokens, now)
            assert used_tokens(_DB, tenant_id=tenant, now_ms=now) == reference._used_at(now)


# --- the async facade -----------------------------------------------------------------------------


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    database = Database(tmp_path / "t4.db")
    database.open()
    yield database
    database.close()


def all_rows(database: Database) -> list[tuple]:
    return database.submit(rows).result()


async def test_reserve_admits_or_raises_a_429_with_rate_limit_headers(database: Database) -> None:
    limiter = TokenLimiter(database, clock=lambda: NOW / 1000)
    assert await limiter.reserve("acme", 1_000, "a", 800) == Admitted(1_000, 200)
    with pytest.raises(errors.GatewayError) as excinfo:
        await limiter.reserve("acme", 1_000, "b", 300)
    error = excinfo.value
    assert (error.status, error.code) == (429, "tenant_token_limit_exceeded")
    assert error.headers == {
        "Retry-After": "60",
        "X-RateLimit-Limit-Tokens": "1000",
        "X-RateLimit-Remaining-Tokens": "200",
    }
    assert "retry after 60 s" in error.message


async def test_a_storage_failure_fails_closed_with_503(database: Database) -> None:
    limiter = TokenLimiter(database)
    database.submit(lambda c: c.execute("DROP TABLE token_usage")).result()
    with pytest.raises(errors.GatewayError) as excinfo:
        await limiter.reserve("acme", 1_000, "a", 10)
    assert (excinfo.value.status, excinfo.value.code) == (503, "limiter_unavailable")


async def test_cancelling_a_reservation_never_leaves_a_reserved_row(database: Database) -> None:
    limiter = TokenLimiter(database)
    for delay in (0.0, 0.001, 0.005, 0.02):
        gate = threading.Event()
        database.submit(lambda _c, gate=gate: gate.wait(5))  # queue the admission behind this
        task = asyncio.ensure_future(limiter.reserve("acme", 1_000, f"r{delay}", 10))
        await asyncio.sleep(delay)
        task.cancel()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    await asyncio.wrap_future(database.submit(lambda _c: None))  # drain
    assert all_rows(database) == []


async def test_settlement_is_queued_and_its_failure_never_raises(
    database: Database, caplog: pytest.LogCaptureFixture
) -> None:
    limiter = TokenLimiter(database)
    await limiter.reserve("acme", 1_000, "a", 500)
    future = limiter.settle("acme", "a", 42)
    assert future is not None
    await asyncio.wrap_future(future)
    assert all_rows(database)[0][2:] == (42, "committed")
    database.submit(lambda c: c.execute("DROP TABLE token_usage")).result()
    failing = limiter.settle("acme", "a", 7)
    assert failing is not None
    with pytest.raises(sqlite3.OperationalError):
        failing.result()
    database.submit(lambda _c: None).result()  # the worker runs the callback before its next job
    assert "settlement failed for request a" in caplog.text
