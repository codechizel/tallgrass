"""Shared fixtures for Tallgrass tests.

Provides pre-built KSSession instances covering the three session types:
current biennium, historical biennium, and special session.

Shared data factory functions live in ``tests/factories.py`` — import as
``from factories import make_legislators, make_votes, make_rollcalls``.
"""

import sys
from pathlib import Path

import pytest

from tallgrass.session import KSSession

# Make tests/ directory importable so test modules can use shared factories
sys.path.insert(0, str(Path(__file__).parent))

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
