"""Tests for BillTextFetcher concurrent download + extraction."""

from unittest.mock import MagicMock, patch

import pytest
import requests
from fpdf import FPDF

from tallgrass.text.fetcher import BillTextFetcher
from tallgrass.text.models import BillDocumentRef, BillText


def _make_pdf_bytes(text: str = "AN ACT concerning taxation.") -> bytes:
    """Create a simple test PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")
    return pdf.output()


def _make_ref(bill_number: str = "SB 55", doc_type: str = "introduced") -> BillDocumentRef:
    return BillDocumentRef(
        bill_number=bill_number,
        document_type=doc_type,
        url=f"https://example.com/{bill_number.lower().replace(' ', '')}.pdf",
        version="00_0000",
        session="91st (2025-2026)",
    )


class TestBillTextFetcher:
    def test_init_creates_cache_dir(self, tmp_path):
        cache_dir = tmp_path / "cache"
        fetcher = BillTextFetcher(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_process_ref_with_cached_pdf(self, tmp_path):
        """Extracts text from a cached PDF file."""
        cache_dir = tmp_path / "cache"
        fetcher = BillTextFetcher(cache_dir=cache_dir)
        ref = _make_ref()

        # Pre-populate cache
        import hashlib

        cache_hash = hashlib.sha256(ref.url.encode()).hexdigest()[:16]
        cache_file = cache_dir / f"{cache_hash}.pdf"
        cache_file.write_bytes(_make_pdf_bytes())

        result = fetcher._process_ref(ref)
        assert result is not None
        assert isinstance(result, BillText)
        assert result.bill_number == "SB 55"
        assert result.document_type == "introduced"
        assert "AN ACT concerning taxation" in result.text
        assert result.page_count == 1
        assert result.extraction_method == "pdfplumber"

    def test_process_ref_returns_none_for_missing(self, tmp_path):
        """Returns None when download fails."""
        cache_dir = tmp_path / "cache"
        fetcher = BillTextFetcher(cache_dir=cache_dir, delay=0)
        ref = _make_ref()

        # Mock HTTP to fail with requests-specific exception
        with patch.object(
            fetcher.http, "get", side_effect=requests.ConnectionError("Connection refused")
        ):
            result = fetcher._process_ref(ref)
        assert result is None

    def test_process_ref_returns_none_for_html_error(self, tmp_path):
        """Returns None when server returns HTML error page."""
        cache_dir = tmp_path / "cache"
        fetcher = BillTextFetcher(cache_dir=cache_dir, delay=0)
        ref = _make_ref()

        mock_resp = MagicMock()
        mock_resp.content = b"<html><body>Not Found</body></html>"
        mock_resp.raise_for_status = MagicMock()

        with patch.object(fetcher.http, "get", return_value=mock_resp):
            result = fetcher._process_ref(ref)
        assert result is None

    def test_fetch_all_empty(self, tmp_path):
        """Empty input returns empty output."""
        fetcher = BillTextFetcher(cache_dir=tmp_path / "cache")
        assert fetcher.fetch_all([]) == []

    def test_fetch_all_with_cached_pdfs(self, tmp_path):
        """Fetches multiple documents from cache."""
        cache_dir = tmp_path / "cache"
        fetcher = BillTextFetcher(cache_dir=cache_dir)

        refs = [_make_ref("SB 1"), _make_ref("HB 2001")]

        # Pre-populate cache for both
        import hashlib

        for ref in refs:
            cache_hash = hashlib.sha256(ref.url.encode()).hexdigest()[:16]
            cache_file = cache_dir / f"{cache_hash}.pdf"
            cache_file.write_bytes(_make_pdf_bytes(f"Bill {ref.bill_number}"))

        results = fetcher.fetch_all(refs)
        assert len(results) == 2
        # Sorted by bill_number
        assert results[0].bill_number == "HB 2001"
        assert results[1].bill_number == "SB 1"

    def test_clear_cache(self, tmp_path):
        cache_dir = tmp_path / "cache"
        fetcher = BillTextFetcher(cache_dir=cache_dir)

        # Create some cache files
        (cache_dir / "test1.pdf").write_bytes(b"test")
        (cache_dir / "test2.pdf").write_bytes(b"test")
        assert len(list(cache_dir.iterdir())) == 2

        fetcher.clear_cache()
        assert len(list(cache_dir.iterdir())) == 0

    def test_download_caches_result(self, tmp_path):
        """Successful downloads are cached to disk."""
        cache_dir = tmp_path / "cache"
        fetcher = BillTextFetcher(cache_dir=cache_dir, delay=0)
        url = "https://example.com/sb1.pdf"
        pdf_bytes = _make_pdf_bytes()

        mock_resp = MagicMock()
        mock_resp.content = pdf_bytes
        mock_resp.raise_for_status = MagicMock()

        with patch.object(fetcher.http, "get", return_value=mock_resp):
            result = fetcher._download(url)

        assert result == pdf_bytes
        # Verify cached
        import hashlib

        cache_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        assert (cache_dir / f"{cache_hash}.pdf").exists()

    def test_download_uses_cache(self, tmp_path):
        """Cached files are returned without HTTP request."""
        cache_dir = tmp_path / "cache"
        fetcher = BillTextFetcher(cache_dir=cache_dir)
        url = "https://example.com/sb1.pdf"
        pdf_bytes = _make_pdf_bytes()

        import hashlib

        cache_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        (cache_dir / f"{cache_hash}.pdf").write_bytes(pdf_bytes)

        # Should not hit HTTP
        with patch.object(fetcher.http, "get", side_effect=AssertionError("Should not be called")):
            result = fetcher._download(url)

        assert result == pdf_bytes
