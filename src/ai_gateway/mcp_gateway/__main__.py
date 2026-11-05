"""Run the MCP security gateway: ``t2-gateway`` (configured by ``GATEWAY_*`` variables)."""

from __future__ import annotations

import logging
import sys

import uvicorn

from ai_gateway.mcp_gateway.app import create_app
from ai_gateway.mcp_gateway.settings import GatewaySettings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = GatewaySettings.from_env()
    except ValueError as exc:
        sys.stderr.write(f"t2-gateway: {exc}\n")
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
