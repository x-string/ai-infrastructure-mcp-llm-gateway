"""Mint a development bearer token for the MCP gateway: ``t2-mint-jwt --role viewer``.

Uses the same ``GATEWAY_JWT_*`` variables as the gateway. In production, tokens come from an
identity provider instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from ai_gateway.mcp_gateway.auth import Role, issue_token
from ai_gateway.mcp_gateway.settings import token_settings_from_env


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="t2-mint-jwt", description="Mint a development bearer token for the MCP gateway."
    )
    parser.add_argument("--role", required=True, choices=[role.value for role in Role])
    parser.add_argument("--subject", default="dev-user", help="the token's sub claim")
    parser.add_argument("--ttl", type=int, default=3600, help="lifetime in seconds")
    args = parser.parse_args(argv)
    try:
        settings = token_settings_from_env(os.environ)
    except ValueError as exc:
        parser.error(str(exc))
    sys.stdout.write(
        issue_token(settings, args.subject, Role(args.role), ttl_seconds=args.ttl) + "\n"
    )


if __name__ == "__main__":
    main()
