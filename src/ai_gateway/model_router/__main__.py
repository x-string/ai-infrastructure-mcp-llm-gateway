"""Run the Task 4 model router: ``t4-router`` (configured by ``T4_*`` variables)."""

from __future__ import annotations

import logging
import sqlite3
import sys

import uvicorn

from ai_gateway.model_router.app import create_app
from ai_gateway.model_router.settings import RouterSettings
from ai_gateway.model_router.storage import StorageError, connect


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = RouterSettings.from_env()
        connect(settings.db_path).close()  # refuse to start without a usable WAL database
    except (ValueError, StorageError, sqlite3.Error) as exc:
        sys.stderr.write(f"t4-router: {exc}\n")
        raise SystemExit(2) from None
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        server_header=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
