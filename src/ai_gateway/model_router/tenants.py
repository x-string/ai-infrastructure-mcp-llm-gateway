"""Tenants and their API keys (Task 4), and the ``t4-tenants`` command that manages them.

A tenant has a token budget per minute and one or more API keys. Keys are random, shown once when
created, and stored only as SHA-256 digests (a plain digest is enough for 256-bit random keys).
Usage is counted per tenant, not per key, so rotating keys does not reset the window.

Usage::

    t4-tenants add acme --tokens-per-minute 50000   # prints a new key for acme, once
    t4-tenants list
    t4-tenants remove acme

The database is ``--db`` or ``T4_DB_PATH`` (default ``t4-router.db``), the router's own file.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import secrets
import sqlite3
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from ai_gateway.model_router.storage import DEFAULT_DB_PATH, StorageError, connect
from ai_gateway.model_router.storage import write_transaction as transaction

KEY_PREFIX: Final = "sk-t4-"
DEFAULT_TOKENS_PER_MINUTE: Final = 50_000
TENANT_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


@dataclass(frozen=True, slots=True)
class Tenant:
    tenant_id: str
    tokens_per_minute: int


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def key_digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def add_tenant(
    connection: sqlite3.Connection, tenant_id: str, tokens_per_minute: int | None = None
) -> str:
    """Create the tenant (or update its limit) and issue a new key, which is returned."""
    if not TENANT_ID_PATTERN.fullmatch(tenant_id):
        raise ValueError("a tenant id is 1-64 letters, digits, '.', '_' or '-'")
    if tokens_per_minute is not None and tokens_per_minute < 1:
        raise ValueError("tokens per minute must be positive")
    key, now_ms = generate_key(), int(time.time() * 1000)
    with transaction(connection):
        connection.execute(
            "INSERT INTO tenants (tenant_id, tokens_per_minute, created_ms) VALUES (?, ?, ?) "
            "ON CONFLICT (tenant_id) DO UPDATE SET tokens_per_minute = "
            "COALESCE(?, tokens_per_minute)",
            (tenant_id, tokens_per_minute or DEFAULT_TOKENS_PER_MINUTE, now_ms, tokens_per_minute),
        )
        connection.execute(
            "INSERT INTO api_keys (key_sha256, tenant_id, created_ms) VALUES (?, ?, ?)",
            (key_digest(key), tenant_id, now_ms),
        )
    return key


def remove_tenant(connection: sqlite3.Connection, tenant_id: str) -> bool:
    """Delete the tenant, its keys and its usage. Returns whether it existed."""
    with transaction(connection):
        connection.execute("DELETE FROM api_keys WHERE tenant_id = ?", (tenant_id,))
        connection.execute("DELETE FROM token_usage WHERE tenant_id = ?", (tenant_id,))
        return bool(
            connection.execute("DELETE FROM tenants WHERE tenant_id = ?", (tenant_id,)).rowcount
        )


def list_tenants(connection: sqlite3.Connection) -> list[tuple[str, int, int]]:
    """``(tenant_id, tokens_per_minute, number of keys)`` for every tenant, by id."""
    return connection.execute(
        "SELECT t.tenant_id, t.tokens_per_minute, COUNT(k.key_sha256) FROM tenants t "
        "LEFT JOIN api_keys k ON k.tenant_id = t.tenant_id GROUP BY t.tenant_id ORDER BY 1"
    ).fetchall()


def find_tenant(connection: sqlite3.Connection, key: str) -> Tenant | None:
    row = connection.execute(
        "SELECT t.tenant_id, t.tokens_per_minute FROM api_keys k "
        "JOIN tenants t ON t.tenant_id = k.tenant_id WHERE k.key_sha256 = ?",
        (key_digest(key),),
    ).fetchone()
    return None if row is None else Tenant(row[0], row[1])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="t4-tenants", description="Manage the model router's tenants and their API keys."
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("T4_DB_PATH") or DEFAULT_DB_PATH,
        help="the router's SQLite file (default: $T4_DB_PATH or t4-router.db)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="create a tenant or give it a new key; prints the key")
    add.add_argument("tenant")
    add.add_argument("--tokens-per-minute", type=int, help=f"default {DEFAULT_TOKENS_PER_MINUTE}")
    commands.add_parser("list", help="list tenants, their limits and how many keys they have")
    remove = commands.add_parser("remove", help="delete a tenant, its keys and its usage")
    remove.add_argument("tenant")
    args = parser.parse_args(argv)

    try:
        connection = connect(args.db)
    except (StorageError, sqlite3.Error) as exc:
        parser.exit(2, f"t4-tenants: cannot use {args.db}: {exc}\n")
    try:
        if args.command == "add":
            try:
                key = add_tenant(connection, args.tenant, args.tokens_per_minute)
            except ValueError as exc:
                parser.error(str(exc))
            sys.stdout.write(key + "\n")
            sys.stderr.write(f"t4-tenants: new key for {args.tenant}; it is not shown again\n")
        elif args.command == "list":
            for tenant_id, limit, keys in list_tenants(connection):
                sys.stdout.write(f"{tenant_id}\t{limit} tokens/min\t{keys} key(s)\n")
        elif not remove_tenant(connection, args.tenant):
            parser.exit(1, f"t4-tenants: no tenant {args.tenant}\n")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
