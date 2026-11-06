"""Run the streaming PII-redaction gateway: ``t3-gateway`` (configured by ``T3_*`` variables)."""

from __future__ import annotations

import logging
import sys

import uvicorn

from ai_gateway.llm_gateway.app import create_app
from ai_gateway.llm_gateway.settings import GatewaySettings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = GatewaySettings.from_env()
    except ValueError as exc:
        sys.stderr.write(f"t3-gateway: {exc}\n")
        raise SystemExit(2) from None
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        server_header=False,
        log_config=None,
        timeout_graceful_shutdown=10,  # streams still running then are cut on shutdown
    )


if __name__ == "__main__":
    main()
