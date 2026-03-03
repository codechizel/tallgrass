"""Tests for Kansas bill text adapter."""

from dataclasses import dataclass

import pytest

from tallgrass.config import BASE_URL
from tallgrass.session import KSSession
from tallgrass.text.kansas import (
    KansasAdapter,
    _bill_number_to_code,
    build_document_urls,
    clean_kansas_text,
)
from tallgrass.text.models import BillDocumentRef


# ── _bill_number_to_code() ───────────────────────────────────────────────


class TestBillNumberToCode:
    def test_senate_bill(self):
        assert _bill_number_to_code("SB 55") == "sb55"

    def test_house_bill(self):
        assert _bill_number_to_code("HB 2084") == "hb2084"

    def test_senate_resolution(self):
        assert _bill_number_to_code("SCR 1601") == "scr1601"

    def test_house_resolution(self):
        assert _bill_number_to_code("HR 6001") == "hr6001"

    def test_no_space(self):
        assert _bill_number_to_code("SB55") == "sb55"

    def test_multiple_spaces(self):
        assert _bill_number_to_code("SB  55") == "sb55"


# ── build_document_urls() ────────────────────────────────────────────────


class TestBuildDocumentUrls:
    def test_current_session_introduced(self):
        """91st (current) uses /li/b2025_26/ prefix."""
        session = KSSession.from_year(2025)
        refs = build_document_urls(session, "SB 55", ["introduced"])
        assert len(refs) == 1
        assert refs[0].bill_number == "SB 55"
        assert refs[0].document_type == "introduced"
        assert refs[0].url == f"{BASE_URL}/li/b2025_26/measures/documents/sb55_00_0000.pdf"
        assert refs[0].version == "00_0000"
        assert refs[0].session == "91st (2025-2026)"

    def test_current_session_supp_note(self):
        session = KSSession.from_year(2025)
        refs = build_document_urls(session, "SB 55", ["supp_note"])
        assert len(refs) == 1
        assert refs[0].url == f"{BASE_URL}/li/b2025_26/measures/documents/supp_note_sb55_00_0000.pdf"

    def test_historical_session(self):
        """90th (historical) uses /li_2024/b2023_24/ prefix."""
        session = KSSession.from_year(2023)
        refs = build_document_urls(session, "HB 2001", ["introduced"])
        assert refs[0].url == (
            f"{BASE_URL}/li_2024/b2023_24/measures/documents/hb2001_00_0000.pdf"
        )
        assert refs[0].session == "90th (2023-2024)"

    def test_89th_session(self):
        session = KSSession.from_year(2021)
        refs = build_document_urls(session, "SB 1", ["introduced"])
        assert refs[0].url == (
            f"{BASE_URL}/li_2022/b2021_22/measures/documents/sb1_00_0000.pdf"
        )

    def test_88th_session(self):
        session = KSSession.from_year(2019)
        refs = build_document_urls(session, "HB 2001", ["introduced"])
        assert refs[0].url == (
            f"{BASE_URL}/li_2020/b2019_20/measures/documents/hb2001_00_0000.pdf"
        )

    def test_86th_session(self):
        session = KSSession.from_year(2015)
        refs = build_document_urls(session, "SB 1", ["introduced"])
        assert refs[0].url == (
            f"{BASE_URL}/li_2016/b2015_16/measures/documents/sb1_00_0000.pdf"
        )

    def test_84th_session(self):
        """Oldest supported biennium."""
        session = KSSession.from_year(2011)
        refs = build_document_urls(session, "HB 2001", ["introduced"])
        assert refs[0].url == (
            f"{BASE_URL}/li_2012/b2011_12/measures/documents/hb2001_00_0000.pdf"
        )

    def test_special_session(self):
        session = KSSession.from_year(2024, special=True)
        refs = build_document_urls(session, "SB 1", ["introduced"])
        assert "/li_2024s/" in refs[0].url
        assert refs[0].session == "2024 Special"

    def test_default_document_types(self):
        """Default types are introduced + supp_note."""
        session = KSSession.from_year(2025)
        refs = build_document_urls(session, "SB 55")
        assert len(refs) == 2
        types = {r.document_type for r in refs}
        assert types == {"introduced", "supp_note"}

    def test_unknown_document_type_skipped(self):
        session = KSSession.from_year(2025)
        refs = build_document_urls(session, "SB 55", ["unknown_type"])
        assert len(refs) == 0

    def test_resolution_url(self):
        session = KSSession.from_year(2025)
        refs = build_document_urls(session, "SCR 1601", ["introduced"])
        assert "scr1601_00_0000.pdf" in refs[0].url

    def test_all_bill_types(self):
        """All Kansas bill types produce valid URLs."""
        session = KSSession.from_year(2025)
        for bill_num, expected_code in [
            ("SB 1", "sb1"),
            ("HB 2001", "hb2001"),
            ("SCR 1601", "scr1601"),
            ("HCR 5001", "hcr5001"),
            ("SR 1801", "sr1801"),
            ("HR 6001", "hr6001"),
        ]:
            refs = build_document_urls(session, bill_num, ["introduced"])
            assert len(refs) == 1
            assert f"{expected_code}_00_0000.pdf" in refs[0].url


# ── clean_kansas_text() ──────────────────────────────────────────────────


class TestCleanKansasText:
    def test_removes_enacting_clause(self):
        text = "Be it enacted by the Legislature of the State of Kansas: Section 1."
        result = clean_kansas_text(text)
        assert "Be it enacted" not in result
        assert "Section 1." in result

    def test_strips_whitespace(self):
        text = "  Some bill text.  "
        result = clean_kansas_text(text)
        assert result == "Some bill text."

    def test_preserves_content(self):
        text = "AN ACT concerning taxation; relating to property tax exemptions."
        result = clean_kansas_text(text)
        assert result == text


# ── KansasAdapter ────────────────────────────────────────────────────────


@dataclass
class MockFetchResult:
    ok: bool
    html: str | None = None


class TestKansasAdapter:
    def test_state_name(self):
        adapter = KansasAdapter()
        assert adapter.state_name == "kansas"

    def test_data_dir(self):
        adapter = KansasAdapter()
        data_dir = adapter.data_dir("2025-26")
        assert "91st_2025-2026" in str(data_dir)

    def test_cache_dir(self):
        adapter = KansasAdapter()
        cache_dir = adapter.cache_dir("2025-26")
        assert ".cache/text" in str(cache_dir)

    def test_discover_bills_with_mock(self):
        """Discovers bills and constructs document URLs."""
        adapter = KansasAdapter(document_types=["introduced"])
        session = KSSession.from_year(2025)

        def mock_get(url: str) -> MockFetchResult:
            if "bills/" in url:
                return MockFetchResult(
                    ok=True,
                    html=(
                        f'<a href="{session.li_prefix}/measures/sb1/">SB 1</a>'
                        f'<a href="{session.li_prefix}/measures/hb2001/">HB 2001</a>'
                    ),
                )
            return MockFetchResult(ok=False)

        refs = adapter.discover_bills("2025-26", get_fn=mock_get)
        assert len(refs) == 2
        assert all(isinstance(r, BillDocumentRef) for r in refs)
        assert refs[0].bill_number == "SB 1"
        assert refs[0].document_type == "introduced"
        assert refs[1].bill_number == "HB 2001"

    def test_discover_bills_multiple_types(self):
        """With multiple doc types, produces N × types refs."""
        adapter = KansasAdapter(document_types=["introduced", "supp_note"])
        session = KSSession.from_year(2025)

        def mock_get(url: str) -> MockFetchResult:
            if "bills/" in url:
                return MockFetchResult(
                    ok=True,
                    html=f'<a href="{session.li_prefix}/measures/sb1/">SB 1</a>',
                )
            return MockFetchResult(ok=False)

        refs = adapter.discover_bills("2025-26", get_fn=mock_get)
        assert len(refs) == 2
        types = {r.document_type for r in refs}
        assert types == {"introduced", "supp_note"}

    def test_discover_bills_no_get_fn(self):
        """Without get_fn, returns empty list (no HTTP)."""
        adapter = KansasAdapter()
        refs = adapter.discover_bills("2025-26")
        assert refs == []

    def test_historical_session_id(self):
        adapter = KansasAdapter(document_types=["introduced"])
        session = KSSession.from_year(2023)

        def mock_get(url: str) -> MockFetchResult:
            if "bills/" in url:
                return MockFetchResult(
                    ok=True,
                    html=f'<a href="{session.li_prefix}/measures/sb1/">SB 1</a>',
                )
            return MockFetchResult(ok=False)

        refs = adapter.discover_bills("2023-24", get_fn=mock_get)
        assert len(refs) == 1
        assert "li_2024" in refs[0].url

    def test_special_session_id(self):
        adapter = KansasAdapter(document_types=["introduced"])
        session = KSSession.from_year(2024, special=True)

        def mock_get(url: str) -> MockFetchResult:
            if "bills/" in url or "li_2024s" in url:
                return MockFetchResult(
                    ok=True,
                    html=f'<a href="{session.li_prefix}/measures/sb1/">SB 1</a>',
                )
            return MockFetchResult(ok=False)

        refs = adapter.discover_bills("2024s", get_fn=mock_get)
        assert len(refs) == 1
        assert "li_2024s" in refs[0].url
        assert refs[0].session == "2024 Special"
