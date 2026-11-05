"""Shared pytest configuration."""

from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on asyncio only. The services use asyncio-specific APIs."""
    return "asyncio"
