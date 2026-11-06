"""Task 4: admission stays exact under contention, from threads and from separate processes.

Every worker opens its own connection to the same file, waits at a barrier, then races to admit.
With a shared clock, exactly ``limit // cost`` requests may succeed, whatever the interleaving,
and no worker may see "database is locked".
"""

from __future__ import annotations

import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ai_gateway.model_router.limiter import Admitted, Denied, admit, used_tokens
from ai_gateway.model_router.storage import connect

NOW = 1_800_000_000_000
LIMIT, COST = 10_000, 7


def race(path: str, worker: int, attempts: int, barrier) -> tuple[int, int]:
    """Admit ``attempts`` requests as fast as possible; return (admitted, denied)."""
    connection = connect(path, busy_timeout=60)
    try:
        barrier.wait(30)
        results = [
            admit(
                connection,
                tenant_id="acme",
                request_id=f"w{worker}-{i}",
                cost=COST,
                limit=LIMIT,
                now_ms=NOW,
            )
            for i in range(attempts)
        ]
    finally:
        connection.close()
    admitted = sum(isinstance(r, Admitted) for r in results)
    assert admitted + sum(isinstance(r, Denied) for r in results) == attempts
    return admitted, attempts - admitted


def check_total(path: Path, admitted: int) -> None:
    assert admitted == LIMIT // COST
    connection = connect(path)
    try:
        assert used_tokens(connection, tenant_id="acme", now_ms=NOW) == admitted * COST
    finally:
        connection.close()


def test_threads_with_their_own_connections_admit_exactly_the_limit(tmp_path: Path) -> None:
    path = tmp_path / "t4.db"
    connect(path).close()
    workers, attempts = 8, 300  # 2,400 attempts for 1,428 slots
    barrier = threading.Barrier(workers)
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda w: race(str(path), w, attempts, barrier), range(workers)))
    check_total(path, sum(admitted for admitted, _ in results))


def _process(path: str, worker: int, attempts: int, barrier, results) -> None:
    results.put(race(path, worker, attempts, barrier))


@pytest.mark.slow
def test_processes_admit_exactly_the_limit(tmp_path: Path) -> None:
    path = tmp_path / "t4.db"
    connect(path).close()
    context = multiprocessing.get_context("spawn")
    workers, attempts = 4, 600
    barrier, results = context.Barrier(workers), context.Queue()
    processes = [
        context.Process(target=_process, args=(str(path), w, attempts, barrier, results))
        for w in range(workers)
    ]
    for process in processes:
        process.start()
    collected = [results.get(timeout=120) for _ in processes]
    for process in processes:
        process.join(60)
        assert process.exitcode == 0
    check_total(path, sum(admitted for admitted, _ in collected))
