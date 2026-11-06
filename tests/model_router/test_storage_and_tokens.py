"""Task 4: the SQLite layer (WAL, write transactions, the FIFO worker) and token estimates."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from ai_gateway.model_router.storage import Database, StorageError, connect, write_transaction
from ai_gateway.model_router.tokens import prompt_tokens, request_cost, text_tokens

# --- connections ----------------------------------------------------------------------------------


def test_a_database_opens_in_wal_mode_with_the_schema(tmp_path: Path) -> None:
    connection = connect(tmp_path / "t4.db")
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
        assert {"tenants", "api_keys", "token_usage", "token_usage_by_tenant"} <= tables
        assert connection.isolation_level is None  # transactions are explicit
    finally:
        connection.close()


def test_a_database_that_cannot_use_wal_is_refused() -> None:
    with pytest.raises(StorageError, match="WAL"):
        connect(":memory:")


def test_write_transactions_roll_back_on_any_exception(tmp_path: Path) -> None:
    connection = connect(tmp_path / "t4.db")
    insert = "INSERT INTO tenants VALUES ('acme', 10, 0)"
    for exc in (ValueError("boom"), KeyboardInterrupt()):
        with pytest.raises(type(exc)), write_transaction(connection):
            connection.execute(insert)
            raise exc
        assert not connection.in_transaction
        assert connection.execute("SELECT COUNT(*) FROM tenants").fetchone()[0] == 0
    with write_transaction(connection):
        connection.execute(insert)
    assert connection.execute("SELECT COUNT(*) FROM tenants").fetchone()[0] == 1
    connection.close()


def test_a_writer_waits_for_the_lock_then_fails_with_operational_error(tmp_path: Path) -> None:
    holder = connect(tmp_path / "t4.db")
    waiter = connect(tmp_path / "t4.db", busy_timeout=0.2)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"), write_transaction(waiter):
            pass
    finally:
        holder.execute("ROLLBACK")
        holder.close()
        waiter.close()


# --- the worker thread ----------------------------------------------------------------------------


def test_jobs_run_on_one_thread_in_submission_order(tmp_path: Path) -> None:
    database = Database(tmp_path / "t4.db")
    database.open()
    seen: list[tuple[int, str]] = []
    futures = [
        database.submit(lambda _c, i=i: seen.append((i, threading.current_thread().name)))
        for i in range(50)
    ]
    for future in futures:
        future.result()
    database.close()
    assert [i for i, _ in seen] == list(range(50))
    assert len({name for _, name in seen}) == 1


def test_close_finishes_queued_jobs_first(tmp_path: Path) -> None:
    database = Database(tmp_path / "t4.db")
    database.open()
    started = threading.Event()
    release = threading.Event()
    done: list[int] = []
    database.submit(lambda _c: (started.set(), release.wait(5)))
    for i in range(5):
        database.submit(lambda _c, i=i: done.append(i))
    started.wait(5)
    closer = threading.Thread(target=database.close)
    closer.start()
    release.set()
    closer.join(5)
    assert done == [0, 1, 2, 3, 4]
    with pytest.raises(StorageError, match="closed"):
        database.submit(lambda _c: None)


async def test_a_cancelled_wait_withdraws_a_job_that_has_not_started(tmp_path: Path) -> None:
    database = Database(tmp_path / "t4.db")
    database.open()
    gate = threading.Event()
    ran: list[str] = []
    database.submit(lambda _c: gate.wait(5))  # keep the worker busy
    waiting = asyncio.ensure_future(database.run(lambda _c: ran.append("withdrawn job")))
    await asyncio.sleep(0.05)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting  # the cancellation reaches the queue on the loop, before the job can start
    gate.set()
    await asyncio.wrap_future(database.submit(lambda _c: None))
    database.close()
    assert ran == []


def test_jobs_fail_cleanly_when_the_database_is_not_open(tmp_path: Path) -> None:
    database = Database(tmp_path / "t4.db")
    with pytest.raises(StorageError, match="not open"):
        database.submit(lambda c: c.execute("SELECT 1")).result()
    database.close()


# --- token estimates ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "tokens"),
    [("", 0), ("abc", 1), ("abcd", 2), ("hello world", 4), ("héllo", 2 + 1), ("日本語", 3)],
)
def test_text_tokens_count_ascii_in_threes_and_anything_else_as_one(text: str, tokens: int) -> None:
    assert text_tokens(text) == tokens


def test_the_prompt_estimate_covers_the_whole_payload_and_overheads() -> None:
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    canonical = json.dumps(payload, separators=(",", ":"))
    assert prompt_tokens(payload) == text_tokens(canonical) + 4 + 3
    assert request_cost(payload, 100) == prompt_tokens(payload) + 100


def test_the_estimate_errs_high_for_english() -> None:
    sentence = "The quick brown fox jumps over the lazy dog. " * 20  # about 200 real tokens
    payload = {"model": "m", "messages": [{"role": "user", "content": sentence}]}
    assert prompt_tokens(payload) > 250
