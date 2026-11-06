"""Task 4: tenants and API keys, through the functions and through the t4-tenants command."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from ai_gateway.model_router.storage import connect
from ai_gateway.model_router.tenants import (
    DEFAULT_TOKENS_PER_MINUTE,
    KEY_PREFIX,
    Tenant,
    add_tenant,
    find_tenant,
    key_digest,
    list_tenants,
    remove_tenant,
)


@pytest.fixture
def db(tmp_path: Path):
    connection = connect(tmp_path / "t4.db")
    yield connection
    connection.close()


def test_a_new_tenant_gets_a_key_and_the_default_limit(db: sqlite3.Connection) -> None:
    key = add_tenant(db, "acme")
    assert key.startswith(KEY_PREFIX) and len(key) > 40
    assert find_tenant(db, key) == Tenant("acme", DEFAULT_TOKENS_PER_MINUTE)
    assert find_tenant(db, key + "x") is None
    assert find_tenant(db, "") is None


def test_adding_again_rotates_in_a_second_key_and_can_change_the_limit(
    db: sqlite3.Connection,
) -> None:
    first = add_tenant(db, "acme", 1_000)
    second = add_tenant(db, "acme")
    assert first != second
    assert find_tenant(db, first) == find_tenant(db, second) == Tenant("acme", 1_000)
    add_tenant(db, "acme", 2_000)
    assert list_tenants(db) == [("acme", 2_000, 3)]


def test_keys_are_stored_only_as_digests(db: sqlite3.Connection, tmp_path: Path) -> None:
    key = add_tenant(db, "acme")
    stored = [row[0] for row in db.execute("SELECT key_sha256 FROM api_keys")]
    assert stored == [key_digest(key)]
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert key.encode() not in (tmp_path / "t4.db").read_bytes()


def test_removing_a_tenant_removes_its_keys_and_usage(db: sqlite3.Connection) -> None:
    key = add_tenant(db, "acme")
    db.execute("INSERT INTO token_usage VALUES ('r1', 'acme', 1, 5, 'committed')")
    assert remove_tenant(db, "acme") is True
    assert find_tenant(db, key) is None
    assert db.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0] == 0
    assert remove_tenant(db, "acme") is False


@pytest.mark.parametrize("tenant_id", ["", "-acme", "a" * 65, "acme corp", "acme/1", "acéme"])
def test_tenant_ids_are_restricted(db: sqlite3.Connection, tenant_id: str) -> None:
    with pytest.raises(ValueError, match="tenant id"):
        add_tenant(db, tenant_id)


def test_limits_must_be_positive(db: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="positive"):
        add_tenant(db, "acme", 0)


def run_cli(db_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ai_gateway.model_router.tenants", "--db", str(db_path), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_the_command_adds_lists_and_removes_tenants(tmp_path: Path) -> None:
    db_path = tmp_path / "t4.db"
    added = run_cli(db_path, "add", "acme", "--tokens-per-minute", "1000")
    assert added.returncode == 0, added.stderr
    key = added.stdout.strip()
    assert key.startswith(KEY_PREFIX) and "\n" not in key
    assert "not shown again" in added.stderr
    listed = run_cli(db_path, "list")
    assert listed.stdout == "acme\t1000 tokens/min\t1 key(s)\n"
    assert key not in listed.stdout
    connection = connect(db_path)
    assert find_tenant(connection, key) == Tenant("acme", 1_000)
    connection.close()
    assert run_cli(db_path, "remove", "acme").returncode == 0
    missing = run_cli(db_path, "remove", "acme")
    assert missing.returncode == 1 and "no tenant acme" in missing.stderr
    invalid = run_cli(db_path, "add", "bad name")
    assert invalid.returncode == 2 and "tenant id" in invalid.stderr
