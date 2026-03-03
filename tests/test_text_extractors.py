"""Tests for PDF text extraction and cleaning functions."""

import pytest
from fpdf import FPDF

from tallgrass.text.extractors import (
    clean_legislative_text,
    extract_pdf_text,
    strip_line_numbers,
)


def _make_pdf(pages: list[str]) -> bytes:
    """Create a test PDF with the given page texts using fpdf2."""
    pdf = FPDF()
    for text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        for line in text.split("\n"):
            pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    return pdf.output()


# ── extract_pdf_text() ───────────────────────────────────────────────────


class TestExtractPdfText:
    def test_single_page(self):
        pdf_bytes = _make_pdf(["AN ACT concerning taxation."])
        text, page_count = extract_pdf_text(pdf_bytes)
        assert "AN ACT concerning taxation" in text
        assert page_count == 1

    def test_multi_page(self):
        pdf_bytes = _make_pdf(["Page one content.", "Page two content."])
        text, page_count = extract_pdf_text(pdf_bytes)
        assert "Page one content" in text
        assert "Page two content" in text
        assert page_count == 2

    def test_invalid_bytes(self):
        text, page_count = extract_pdf_text(b"not a pdf")
        assert text == ""
        assert page_count == 0

    def test_empty_bytes(self):
        text, page_count = extract_pdf_text(b"")
        assert text == ""
        assert page_count == 0


# ── clean_legislative_text() ─────────────────────────────────────────────


class TestCleanLegislativeText:
    def test_ligature_fi(self):
        assert "fi" in clean_legislative_text("of\ufb01cial")

    def test_ligature_fl(self):
        assert "fl" in clean_legislative_text("in\ufb02ation")

    def test_page_numbers_removed(self):
        text = "Some text.\n\n42\n\nMore text."
        result = clean_legislative_text(text)
        assert "\n42\n" not in result
        assert "Some text." in result
        assert "More text." in result

    def test_hyphenation_rejoined(self):
        text = "legis-\nlation"
        result = clean_legislative_text(text)
        assert "legislation" in result

    def test_whitespace_normalized(self):
        text = "too   many    spaces"
        result = clean_legislative_text(text)
        assert "too many spaces" == result

    def test_excessive_newlines(self):
        text = "line1\n\n\n\n\nline2"
        result = clean_legislative_text(text)
        assert "line1\n\nline2" == result

    def test_strips_leading_trailing(self):
        text = "\n\n  content here  \n\n"
        result = clean_legislative_text(text)
        assert result == "content here"

    def test_preserves_meaningful_newlines(self):
        text = "Section 1.\n\nSection 2."
        result = clean_legislative_text(text)
        assert "Section 1.\n\nSection 2." == result


# ── strip_line_numbers() ─────────────────────────────────────────────────


class TestStripLineNumbers:
    def test_removes_line_numbers(self):
        text = "  1  AN ACT concerning taxation;\n  2  amending K.S.A. 79-3220."
        result = strip_line_numbers(text)
        assert "AN ACT concerning taxation;" in result
        assert "amending K.S.A." in result
        # Line numbers should be gone
        assert not result.startswith("1")

    def test_two_digit_numbers(self):
        text = " 42  Section forty-two."
        result = strip_line_numbers(text)
        assert "Section forty-two." in result

    def test_preserves_inline_numbers(self):
        """Numbers within text (not at line start) are preserved."""
        text = "there are 42 senators"
        result = strip_line_numbers(text)
        assert "42" in result

    def test_no_line_numbers(self):
        """Text without line numbers passes through unchanged."""
        text = "No line numbers here."
        result = strip_line_numbers(text)
        assert result == text
