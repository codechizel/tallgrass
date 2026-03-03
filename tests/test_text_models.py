"""Tests for bill text data models."""

import pytest

from tallgrass.text.models import BillDocumentRef, BillText


class TestBillDocumentRef:
    """Tests for BillDocumentRef frozen dataclass."""

    def test_required_fields(self):
        ref = BillDocumentRef(
            bill_number="SB 55",
            document_type="introduced",
            url="https://example.com/sb55.pdf",
        )
        assert ref.bill_number == "SB 55"
        assert ref.document_type == "introduced"
        assert ref.url == "https://example.com/sb55.pdf"

    def test_default_fields(self):
        ref = BillDocumentRef(
            bill_number="HB 2084",
            document_type="supp_note",
            url="https://example.com/hb2084_supp.pdf",
        )
        assert ref.version == ""
        assert ref.session == ""

    def test_all_fields(self):
        ref = BillDocumentRef(
            bill_number="SB 55",
            document_type="introduced",
            url="https://example.com/sb55.pdf",
            version="00_0000",
            session="91st (2025-2026)",
        )
        assert ref.version == "00_0000"
        assert ref.session == "91st (2025-2026)"

    def test_frozen(self):
        ref = BillDocumentRef(
            bill_number="SB 55",
            document_type="introduced",
            url="https://example.com/sb55.pdf",
        )
        with pytest.raises(AttributeError):
            ref.bill_number = "SB 99"  # type: ignore[misc]

    def test_equality(self):
        ref1 = BillDocumentRef("SB 55", "introduced", "https://example.com/sb55.pdf")
        ref2 = BillDocumentRef("SB 55", "introduced", "https://example.com/sb55.pdf")
        assert ref1 == ref2

    def test_inequality_different_type(self):
        ref1 = BillDocumentRef("SB 55", "introduced", "https://example.com/sb55.pdf")
        ref2 = BillDocumentRef("SB 55", "supp_note", "https://example.com/sb55_supp.pdf")
        assert ref1 != ref2

    def test_hashable(self):
        ref = BillDocumentRef("SB 55", "introduced", "https://example.com/sb55.pdf")
        s = {ref}
        assert len(s) == 1

    def test_bill_types(self):
        """All Kansas bill types produce valid refs."""
        for bill_num in ["SB 55", "HB 2084", "SCR 1601", "HCR 5001", "SR 1801", "HR 6001"]:
            ref = BillDocumentRef(bill_num, "introduced", f"https://example.com/{bill_num}.pdf")
            assert ref.bill_number == bill_num

    def test_document_types(self):
        """All document types produce valid refs."""
        for doc_type in ["introduced", "enrolled", "supp_note", "committee_amended", "ccrb"]:
            ref = BillDocumentRef("SB 1", doc_type, "https://example.com/sb1.pdf")
            assert ref.document_type == doc_type


class TestBillText:
    """Tests for BillText frozen dataclass."""

    def test_all_fields(self):
        bt = BillText(
            bill_number="SB 55",
            document_type="introduced",
            version="00_0000",
            session="91st (2025-2026)",
            text="AN ACT concerning taxation.",
            page_count=3,
            source_url="https://example.com/sb55.pdf",
            extraction_method="pdfplumber",
        )
        assert bt.bill_number == "SB 55"
        assert bt.document_type == "introduced"
        assert bt.version == "00_0000"
        assert bt.session == "91st (2025-2026)"
        assert bt.text == "AN ACT concerning taxation."
        assert bt.page_count == 3
        assert bt.source_url == "https://example.com/sb55.pdf"
        assert bt.extraction_method == "pdfplumber"

    def test_frozen(self):
        bt = BillText(
            bill_number="SB 55",
            document_type="introduced",
            version="00_0000",
            session="91st (2025-2026)",
            text="AN ACT concerning taxation.",
            page_count=3,
            source_url="https://example.com/sb55.pdf",
            extraction_method="pdfplumber",
        )
        with pytest.raises(AttributeError):
            bt.text = "modified"  # type: ignore[misc]

    def test_equality(self):
        kwargs = dict(
            bill_number="SB 55",
            document_type="introduced",
            version="00_0000",
            session="91st (2025-2026)",
            text="AN ACT concerning taxation.",
            page_count=3,
            source_url="https://example.com/sb55.pdf",
            extraction_method="pdfplumber",
        )
        assert BillText(**kwargs) == BillText(**kwargs)

    def test_hashable(self):
        bt = BillText(
            bill_number="SB 55",
            document_type="introduced",
            version="00_0000",
            session="91st (2025-2026)",
            text="AN ACT concerning taxation.",
            page_count=3,
            source_url="https://example.com/sb55.pdf",
            extraction_method="pdfplumber",
        )
        s = {bt}
        assert len(s) == 1

    def test_empty_text(self):
        """Bills with empty extracted text are valid (e.g., scanned PDFs)."""
        bt = BillText(
            bill_number="SB 55",
            document_type="introduced",
            version="00_0000",
            session="91st (2025-2026)",
            text="",
            page_count=1,
            source_url="https://example.com/sb55.pdf",
            extraction_method="pdfplumber",
        )
        assert bt.text == ""
        assert bt.page_count == 1
