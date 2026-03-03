"""Data models for bill text retrieval (state-agnostic)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BillDocumentRef:
    """A fetchable bill document.

    Produced by a StateAdapter — describes what to fetch.
    Joins to existing vote data on ``bill_number``.
    """

    bill_number: str  # "SB 55", "HB 2084"
    document_type: str  # "introduced", "enrolled", "supp_note", "committee_amended", "ccrb"
    url: str  # full URL to the document
    version: str = ""  # version identifier (e.g., "00_0000", "03_0000")
    session: str = ""  # session label for provenance


@dataclass(frozen=True)
class BillText:
    """Extracted text from a bill document.

    Produced by the extraction pipeline — the result of downloading
    and parsing a ``BillDocumentRef``.
    """

    bill_number: str
    document_type: str
    version: str
    session: str
    text: str  # extracted full text
    page_count: int  # number of pages in the source document
    source_url: str  # provenance
    extraction_method: str  # "pdfplumber", "html", etc.
