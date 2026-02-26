"""Shared fixtures for Tallgrass tests.

Provides pre-built KSSession instances covering the three session types:
current biennium, historical biennium, and special session.
"""

import pytest

from tallgrass.session import KSSession

# ── Session fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def current_session() -> KSSession:
    """Current biennium (2025-2026)."""
    return KSSession(start_year=2025)


@pytest.fixture
def historical_session() -> KSSession:
    """Historical biennium (2023-2024)."""
    return KSSession(start_year=2023)


@pytest.fixture
def special_session() -> KSSession:
    """Special session (2024)."""
    return KSSession(start_year=2024, special=True)
