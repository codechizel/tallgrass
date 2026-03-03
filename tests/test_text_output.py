"""Tests for bill text CSV output."""

import csv

from tallgrass.text.models import BillText
from tallgrass.text.output import FIELDNAMES, save_bill_texts


def _make_bill_text(**kwargs) -> BillText:
    defaults = dict(
        bill_number="SB 55",
        document_type="introduced",
        version="00_0000",
        session="91st (2025-2026)",
        text="AN ACT concerning taxation.",
        page_count=3,
        source_url="https://example.com/sb55.pdf",
        extraction_method="pdfplumber",
    )
    defaults.update(kwargs)
    return BillText(**defaults)


class TestSaveBillTexts:
    def test_creates_csv(self, tmp_path):
        texts = [_make_bill_text()]
        path = save_bill_texts(tmp_path, "91st_2025-2026", texts)
        assert path.exists()
        assert path.name == "91st_2025-2026_bill_texts.csv"

    def test_csv_headers(self, tmp_path):
        texts = [_make_bill_text()]
        path = save_bill_texts(tmp_path, "test", texts)

        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert headers == FIELDNAMES

    def test_csv_row_count(self, tmp_path):
        texts = [
            _make_bill_text(bill_number="SB 1"),
            _make_bill_text(bill_number="SB 2"),
            _make_bill_text(bill_number="HB 2001"),
        ]
        path = save_bill_texts(tmp_path, "test", texts)

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3

    def test_csv_content(self, tmp_path):
        texts = [_make_bill_text(bill_number="SB 55", text="Full bill text here.")]
        path = save_bill_texts(tmp_path, "test", texts)

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["bill_number"] == "SB 55"
        assert row["document_type"] == "introduced"
        assert row["text"] == "Full bill text here."
        assert row["page_count"] == "3"
        assert row["session"] == "91st (2025-2026)"
        assert row["source_url"] == "https://example.com/sb55.pdf"

    def test_creates_output_dir(self, tmp_path):
        output_dir = tmp_path / "nested" / "dir"
        texts = [_make_bill_text()]
        save_bill_texts(output_dir, "test", texts)
        assert output_dir.exists()

    def test_empty_list(self, tmp_path):
        path = save_bill_texts(tmp_path, "test", [])

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 0

    def test_multiline_text(self, tmp_path):
        """Text with newlines is preserved in CSV."""
        texts = [_make_bill_text(text="Line 1.\nLine 2.\nLine 3.")]
        path = save_bill_texts(tmp_path, "test", texts)

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert "Line 1.\nLine 2.\nLine 3." == row["text"]

    def test_roundtrip(self, tmp_path):
        """Write + read roundtrip preserves all fields."""
        original = _make_bill_text(
            bill_number="HB 2084",
            document_type="supp_note",
            version="01_0000",
            session="90th (2023-2024)",
            text="Supplemental note text with 'quotes' and, commas.",
            page_count=5,
            source_url="https://example.com/hb2084_supp.pdf",
        )
        path = save_bill_texts(tmp_path, "test", [original])

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["bill_number"] == original.bill_number
        assert row["document_type"] == original.document_type
        assert row["version"] == original.version
        assert row["session"] == original.session
        assert row["text"] == original.text
        assert int(row["page_count"]) == original.page_count
        assert row["source_url"] == original.source_url
