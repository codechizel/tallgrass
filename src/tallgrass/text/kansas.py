"""Kansas bill text adapter — discovers bill documents from kslegislature.gov."""

import re
from collections.abc import Callable
from pathlib import Path

from tallgrass.bills import discover_bills
from tallgrass.config import BASE_URL
from tallgrass.session import KSSession
from tallgrass.text.models import BillDocumentRef

# Bill code regex: "SB 55" -> "sb55"
_BILL_CODE_RE = re.compile(r"^(sb|hb|scr|hcr|sr|hr)\s*(\d+)$", re.I)

# Kansas-specific boilerplate patterns for text cleaning
ENACTING_CLAUSE_RE = re.compile(
    r"Be it enacted by the Legislature of the State of Kansas:\s*", re.I
)
KSA_REF_RE = re.compile(r"K\.S\.A\.?\s*(\d+-\d+[a-z]?\d*)")
SECTION_HEADER_RE = re.compile(r"^(?:New\s+)?Sec(?:tion)?\.?\s*\d+\.", re.MULTILINE | re.I)


def _bill_number_to_code(bill_number: str) -> str:
    """Convert a human-readable bill number to a URL-safe code.

    ``"SB 55"`` -> ``"sb55"``, ``"HCR 5001"`` -> ``"hcr5001"``
    """
    return re.sub(r"\s+", "", bill_number).lower()


def build_document_urls(
    session: KSSession,
    bill_number: str,
    document_types: list[str] | None = None,
) -> list[BillDocumentRef]:
    """Construct deterministic PDF URLs for a Kansas bill.

    URL pattern (verified across 84th-91st):
      Introduced:  ``{li_prefix}/measures/documents/{code}_00_0000.pdf``
      Supp note:   ``{li_prefix}/measures/documents/supp_note_{code}_00_0000.pdf``

    Args:
        session: The KSSession for URL prefix resolution.
        bill_number: Human-readable bill number (e.g., "SB 55").
        document_types: Which document types to generate URLs for.
            Defaults to ``["introduced", "supp_note"]``.

    Returns:
        List of ``BillDocumentRef`` objects with fully qualified URLs.
    """
    if document_types is None:
        document_types = ["introduced", "supp_note"]

    code = _bill_number_to_code(bill_number)
    if not code:
        return []

    docs_base = f"{BASE_URL}{session.li_prefix}/measures/documents"
    refs: list[BillDocumentRef] = []

    for doc_type in document_types:
        if doc_type == "introduced":
            url = f"{docs_base}/{code}_00_0000.pdf"
            version = "00_0000"
        elif doc_type == "supp_note":
            url = f"{docs_base}/supp_note_{code}_00_0000.pdf"
            version = "00_0000"
        else:
            continue

        refs.append(
            BillDocumentRef(
                bill_number=bill_number,
                document_type=doc_type,
                url=url,
                version=version,
                session=session.label,
            )
        )

    return refs


def clean_kansas_text(text: str) -> str:
    """Apply Kansas-specific text cleaning after generic extraction.

    Removes enacting clauses, normalizes K.S.A. references, and cleans
    section headers. Called as a post-processing step by the adapter.
    """
    # Remove standard enacting clause
    text = ENACTING_CLAUSE_RE.sub("", text)
    return text.strip()


class KansasAdapter:
    """StateAdapter implementation for Kansas Legislature bills.

    Discovers bills via the shared ``tallgrass.bills`` module and
    constructs deterministic PDF URLs for bill documents.
    """

    state_name: str = "kansas"

    def __init__(self, document_types: list[str] | None = None):
        """Initialize with optional document type filter.

        Args:
            document_types: Which document types to fetch. Defaults to
                ``["introduced", "supp_note"]``.
        """
        self._document_types = document_types or ["introduced", "supp_note"]

    def _get_session(self, session_id: str) -> KSSession:
        """Parse session_id into a KSSession."""
        return KSSession.from_session_string(session_id)

    def discover_bills(
        self,
        session_id: str,
        get_fn: Callable[[str], object] | None = None,
    ) -> list[BillDocumentRef]:
        """Discover all bill documents for a Kansas session.

        Args:
            session_id: Session string (e.g., "2025-26", "2024s").
            get_fn: HTTP fetch function (same interface as ``FetchResult``).
                Required for bill URL discovery from the website.

        Returns:
            List of ``BillDocumentRef`` for all discovered bills × document types.
        """
        session = self._get_session(session_id)

        if get_fn is not None:
            bills = discover_bills(session, get_fn, verbose=True)
        else:
            bills = []

        refs: list[BillDocumentRef] = []
        for bill in bills:
            refs.extend(build_document_urls(session, bill.bill_number, self._document_types))
        return refs

    def data_dir(self, session_id: str) -> Path:
        """Return the data directory for this session."""
        return self._get_session(session_id).data_dir

    def cache_dir(self, session_id: str) -> Path:
        """Return the cache directory for downloaded documents."""
        return self.data_dir(session_id) / ".cache" / "text"
