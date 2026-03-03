"""PDF text extraction and legislative text cleaning (pure functions, no I/O).

Follows the same pattern as ``odt_parser.py`` — all functions take bytes or
strings and return processed data. No file system access.
"""

import re

import pdfplumber


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract text from PDF bytes.

    Uses pdfplumber for structured text extraction. Handles multi-page
    documents by concatenating pages with newlines.

    Args:
        pdf_bytes: Raw PDF file content.

    Returns:
        Tuple of (extracted_text, page_count). Returns ("", 0) on failure.
    """
    try:
        import io

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages: list[str] = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages), len(pdf.pages)
    except Exception:
        return "", 0


def clean_legislative_text(raw_text: str) -> str:
    """Remove common PDF artifacts and normalize whitespace.

    Handles universal legislative PDF issues (not state-specific):
    - Page numbers and headers/footers
    - Ligature artifacts
    - Excessive whitespace
    - Line-break hyphenation
    """
    text = raw_text

    # Fix common ligature artifacts from PDF extraction
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl")
    text = text.replace("\ufb03", "ffi")
    text = text.replace("\ufb04", "ffl")

    # Remove page numbers (standalone numbers on their own line)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)

    # Rejoin hyphenated words split across lines
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize whitespace: collapse multiple spaces to one
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def strip_line_numbers(text: str) -> str:
    """Remove line numbers from legislative bill text.

    Kansas bills (and most state legislatures) print line numbers in the
    left margin: ``1``, ``2``, ..., ``43`` per page. These appear at the
    start of lines after PDF extraction.
    """
    return re.sub(r"(?m)^\s{0,4}\d{1,2}\s{1,4}", "", text)
