"""Test helper: the real stdio server, with a store that writes stray output to stdout.

Run with ``python -m tests.mcp_server.noisy_server`` from the repository root. A
``get_customer_record`` call writes through ``sys.stdout`` and through a reference to the original
stdout captured at import time, the way a library's logging handler might. Neither may reach the
protocol stream.
"""

from __future__ import annotations

import sys

import anyio

from ai_gateway.mcp_server.__main__ import configure_logging
from ai_gateway.mcp_server.schemas import CustomerRecord
from ai_gateway.mcp_server.server import build_server
from ai_gateway.mcp_server.stdio_transport import serve_stdio
from ai_gateway.mcp_server.store import DEMO_CUSTOMERS, CustomerStore

CAPTURED_STDOUT = sys.stdout


class NoisyStore(CustomerStore):
    def get(self, customer_id: str) -> CustomerRecord:
        sys.stdout.write("STRAY-WRITE-DURING-SERVING\n")
        CAPTURED_STDOUT.write("STRAY-WRITE-VIA-CAPTURED-REFERENCE\n")
        return super().get(customer_id)


if __name__ == "__main__":
    configure_logging()
    anyio.run(serve_stdio, build_server(NoisyStore(DEMO_CUSTOMERS)))
