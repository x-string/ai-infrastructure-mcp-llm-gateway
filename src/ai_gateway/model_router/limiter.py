"""Token-aware sliding-window rate limiting, kept in SQLite (Task 4).

Each tenant may use ``tokens_per_minute`` tokens within any 60-second window: a usage row counts
while ``now - 60 s < ts_ms <= now``, so a row exactly 60 seconds old no longer counts.

Admission is a single ``BEGIN IMMEDIATE`` transaction, so it is atomic across threads and
processes:

1. rows dated in the future (the clock stepped back) are moved to ``now``;
2. the tenant's rows older than the window are deleted;
3. the remaining tokens are summed;
4. if ``used + cost`` exceeds the limit, the request is denied with a ``Retry-After`` that is both
   sufficient and tight: the whole seconds until enough of the oldest rows have expired.
   Otherwise a ``reserved`` row for the full estimated cost is inserted.

So reservations never exceed the limit at admission time. Settlement later replaces the estimate
with what the request really used, exactly once: every statement is filtered on
``status = 'reserved'``, so a second settlement changes nothing. If the reservation had already
expired out of the window during a long request, the usage is recorded as a new row instead.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from typing import Final

from ai_gateway.model_router import errors
from ai_gateway.model_router.storage import Database, StorageError, write_transaction

WINDOW_MS: Final = 60_000
MAX_RETRY_AFTER_SECONDS: Final = 60

logger = logging.getLogger("ai_gateway.model_router.limiter")


@dataclass(frozen=True, slots=True)
class Admitted:
    limit: int
    remaining: int


@dataclass(frozen=True, slots=True)
class Denied:
    limit: int
    remaining: int
    retry_after: int  # whole seconds


def admit(
    connection: sqlite3.Connection,
    *,
    tenant_id: str,
    request_id: str,
    cost: int,
    limit: int,
    now_ms: int,
) -> Admitted | Denied:
    """Reserve ``cost`` tokens for ``request_id`` if the tenant's window has room."""
    with write_transaction(connection):
        connection.execute(
            "UPDATE token_usage SET ts_ms = ? WHERE tenant_id = ? AND ts_ms > ?",
            (now_ms, tenant_id, now_ms),
        )
        connection.execute(
            "DELETE FROM token_usage WHERE tenant_id = ? AND ts_ms <= ?",
            (tenant_id, now_ms - WINDOW_MS),
        )
        used = _used(connection, tenant_id)
        if used + cost > limit:
            retry_after = _retry_after(connection, tenant_id, used + cost - limit, now_ms)
            return Denied(limit, limit - used, retry_after)
        connection.execute(
            "INSERT INTO token_usage (request_id, tenant_id, ts_ms, tokens, status) "
            "VALUES (?, ?, ?, ?, 'reserved')",
            (request_id, tenant_id, now_ms, cost),
        )
        return Admitted(limit, limit - used - cost)


def settle(
    connection: sqlite3.Connection,
    *,
    tenant_id: str,
    request_id: str,
    tokens: int,
    now_ms: int,
) -> None:
    """Replace the reservation with ``tokens`` actually used (0 releases it). Runs once only."""
    with write_transaction(connection):
        if tokens <= 0:
            connection.execute(
                "DELETE FROM token_usage "
                "WHERE tenant_id = ? AND request_id = ? AND status = 'reserved'",
                (tenant_id, request_id),
            )
            return
        updated = connection.execute(
            "UPDATE token_usage SET tokens = ?, status = 'committed' "
            "WHERE tenant_id = ? AND request_id = ? AND status = 'reserved'",
            (tokens, tenant_id, request_id),
        ).rowcount
        if not updated:  # the reservation expired out of the window: record the usage now
            connection.execute(
                "INSERT OR IGNORE INTO token_usage (request_id, tenant_id, ts_ms, tokens, status) "
                "VALUES (?, ?, ?, ?, 'committed')",
                (request_id, tenant_id, now_ms, tokens),
            )


def used_tokens(connection: sqlite3.Connection, *, tenant_id: str, now_ms: int) -> int:
    """Tokens the tenant has used or reserved in the window ending at ``now_ms``."""
    row = connection.execute(
        "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant_id = ? AND ts_ms > ?",
        (tenant_id, now_ms - WINDOW_MS),
    ).fetchone()
    return int(row[0])


def purge_expired(connection: sqlite3.Connection, *, now_ms: int) -> int:
    """Delete every tenant's rows that are out of the window; returns how many."""
    with write_transaction(connection):
        return connection.execute(
            "DELETE FROM token_usage WHERE ts_ms <= ?", (now_ms - WINDOW_MS,)
        ).rowcount


def _used(connection: sqlite3.Connection, tenant_id: str) -> int:
    row = connection.execute(
        "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant_id = ?", (tenant_id,)
    ).fetchone()
    return int(row[0])


def _retry_after(connection: sqlite3.Connection, tenant_id: str, excess: int, now_ms: int) -> int:
    """Whole seconds until rows holding at least ``excess`` tokens have left the window."""
    freed = 0
    rows = connection.execute(
        "SELECT ts_ms, tokens FROM token_usage WHERE tenant_id = ? ORDER BY ts_ms, request_id",
        (tenant_id,),
    )
    for ts_ms, tokens in rows:
        freed += tokens
        if freed >= excess:
            wait_ms = ts_ms + WINDOW_MS - now_ms  # a row leaves the window at ts + 60 s
            return min(max(math.ceil(wait_ms / 1000), 1), MAX_RETRY_AFTER_SECONDS)
    return MAX_RETRY_AFTER_SECONDS  # unreachable while cost <= limit


class TokenLimiter:
    """The limiter as the router uses it: awaited admission, settlement queued without waiting."""

    def __init__(self, database: Database, *, clock: Callable[[], float] = time.time) -> None:
        self._database = database
        self._clock = clock

    def now_ms(self) -> int:
        return int(self._clock() * 1000)

    async def reserve(self, tenant_id: str, limit: int, request_id: str, cost: int) -> Admitted:
        """Admit the request or raise: 429 when over the limit, 503 when SQLite fails.

        If the caller is cancelled while waiting, a release is queued right away: it runs after
        the admission (same FIFO worker), so a reservation can never be left behind.
        """
        job = partial(
            admit,
            tenant_id=tenant_id,
            request_id=request_id,
            cost=cost,
            limit=limit,
            now_ms=self.now_ms(),
        )
        try:
            result = await self._database.run(job)
        except (sqlite3.Error, StorageError):
            logger.exception("admission failed for tenant %s, request %s", tenant_id, request_id)
            raise errors.limiter_unavailable() from None
        except BaseException:
            self.settle(tenant_id, request_id, 0)
            raise
        if isinstance(result, Denied):
            raise errors.rate_limited(result.retry_after, result.limit, result.remaining)
        return result

    def settle(self, tenant_id: str, request_id: str, tokens: int) -> Future[None] | None:
        """Queue the settlement; a failure is logged and never changes the response."""
        job = partial(
            settle, tenant_id=tenant_id, request_id=request_id, tokens=tokens, now_ms=self.now_ms()
        )
        try:
            future = self._database.submit(job)
        except StorageError:  # the worker has shut down
            logger.error("settlement dropped for request %s: storage is closed", request_id)
            return None
        future.add_done_callback(partial(_log_failure, request_id))
        return future

    async def used(self, tenant_id: str) -> int:
        return await self._database.run(
            partial(used_tokens, tenant_id=tenant_id, now_ms=self.now_ms())
        )

    async def purge(self) -> int:
        return await self._database.run(partial(purge_expired, now_ms=self.now_ms()))


def _log_failure(request_id: str, future: Future[None]) -> None:
    if not future.cancelled() and (exc := future.exception()) is not None:
        logger.error(
            "settlement failed for request %s: %s", request_id, type(exc).__name__, exc_info=exc
        )
