"""StateAdapter protocol — contract for state-specific bill text discovery."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from tallgrass.text.models import BillDocumentRef


@runtime_checkable
class StateAdapter(Protocol):
    """Contract for state-specific bill text discovery.

    Each state implements this protocol to discover bill documents
    and provide directory paths. Kansas is the first implementation;
    adding a state means writing one new file.
    """

    state_name: str  # "kansas", "nebraska", etc.

    def discover_bills(self, session_id: str) -> list[BillDocumentRef]:
        """Given a session identifier, return all fetchable bill documents.

        The ``session_id`` is a string like ``"2025-26"`` or ``"2024s"`` —
        the adapter interprets it for its state.
        """
        ...

    def data_dir(self, session_id: str) -> Path:
        """Return the data directory for this state + session."""
        ...

    def cache_dir(self, session_id: str) -> Path:
        """Return the cache directory for downloaded documents."""
        ...
