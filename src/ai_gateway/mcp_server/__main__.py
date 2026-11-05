"""Run the Task 1 MCP server over stdio: ``t1-mcp-server`` or ``python -m ai_gateway.mcp_server``.

stdout carries JSON-RPC messages only, and every log record goes to stderr. Set
``MCP_SERVER_LOG_LEVEL`` (default ``INFO``) to change the log level.
"""

from __future__ import annotations

import logging
import os
import sys

import anyio

from ai_gateway.mcp_server.server import build_server
from ai_gateway.mcp_server.stdio_transport import serve_stdio

LOG_LEVEL_ENV = "MCP_SERVER_LOG_LEVEL"


def configure_logging() -> None:
    """Send every log record to stderr, before anything else runs."""
    level_name = os.environ.get(LOG_LEVEL_ENV, "INFO").upper()
    level = logging.getLevelNamesMapping().get(level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.captureWarnings(True)


def main() -> None:
    configure_logging()
    try:
        anyio.run(serve_stdio, build_server())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
