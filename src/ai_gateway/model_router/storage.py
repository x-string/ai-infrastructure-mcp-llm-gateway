"""SQLite storage for Task 4: one connection, one worker thread, explicit write transactions.

Each process opens one long-lived connection, owned by a single worker thread. Every query runs on
that thread, in the order it was submitted. That keeps blocking I/O off the event loop, and it
makes ordering deterministic: a job submitted after an admission for the same request always runs
after it.

Across threads and processes, SQLite's locking serializes writers. Every write transaction starts
with ``BEGIN IMMEDIATE``, which takes the write lock before the first read, waiting up to the busy
timeout. So a read-then-write such as "sum the window, then insert a reservation" can never race
another writer. WAL journaling lets readers proceed while one writer works. It must be confirmed at
startup, because ``PRAGMA journal_mode=WAL`` quietly returns the old mode when it cannot switch
(for example on an in-memory database).
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Final, TypeVar

T = TypeVar("T")

DEFAULT_DB_PATH: Final = "t4-router.db"
BUSY_TIMEOUT_SECONDS: Final = 2.0

SCHEMA: Final = (
    """CREATE TABLE IF NOT EXISTS tenants (
        tenant_id TEXT PRIMARY KEY,
        tokens_per_minute INTEGER NOT NULL CHECK (tokens_per_minute > 0),
        created_ms INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS api_keys (
        key_sha256 TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL REFERENCES tenants (tenant_id) ON DELETE CASCADE,
        created_ms INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS token_usage (
        request_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        ts_ms INTEGER NOT NULL,
        tokens INTEGER NOT NULL CHECK (tokens >= 0),
        status TEXT NOT NULL CHECK (status IN ('reserved', 'committed'))
    )""",
    "CREATE INDEX IF NOT EXISTS token_usage_by_tenant ON token_usage (tenant_id, ts_ms)",
)


class StorageError(Exception):
    """The database cannot be used safely, so the service must not start."""


@contextmanager
def write_transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` ... ``COMMIT``, rolled back on any exception, including cancellation."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def connect(path: str | Path, *, busy_timeout: float = BUSY_TIMEOUT_SECONDS) -> sqlite3.Connection:
    """Open the database in WAL mode and create the schema. Raises StorageError otherwise."""
    connection = sqlite3.connect(path, timeout=busy_timeout, isolation_level=None)
    try:
        mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        if mode != "wal":
            raise StorageError(f"the database must use WAL journaling, but it uses {mode!r}")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        with write_transaction(connection):
            for statement in SCHEMA:
                connection.execute(statement)
    except BaseException:
        connection.close()
        raise
    return connection


class Database:
    """The process's connection, used only from its own worker thread, in FIFO order."""

    def __init__(self, path: str | Path, *, busy_timeout: float = BUSY_TIMEOUT_SECONDS) -> None:
        self.path = Path(path)
        self._busy_timeout = busy_timeout
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="t4-sqlite")
        self._connection: sqlite3.Connection | None = None

    def open(self) -> None:
        """Connect on the worker thread (a connection is tied to the thread that made it)."""
        self._executor.submit(self._open).result()

    def submit(self, job: Callable[[sqlite3.Connection], T]) -> Future[T]:
        """Queue ``job`` to run with the connection, after every job queued before it."""
        try:
            return self._executor.submit(self._call, job)
        except RuntimeError:  # the worker has shut down
            raise StorageError("the database is closed") from None

    async def run(self, job: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``job`` and wait for it. If the wait is cancelled before the job starts, it is
        withdrawn; once it has started, it runs to completion."""
        return await asyncio.wrap_future(self.submit(job))

    def close(self) -> None:
        """Finish every queued job, then close the connection and stop the worker."""
        try:
            self._executor.submit(self._close).result()
        finally:
            self._executor.shutdown(wait=True)

    def _open(self) -> None:
        self._connection = connect(self.path, busy_timeout=self._busy_timeout)

    def _call(self, job: Callable[[sqlite3.Connection], T]) -> T:
        if self._connection is None:
            raise StorageError("the database is not open")
        return job(self._connection)

    def _close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
