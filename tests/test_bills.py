"""Tests for shared bill discovery module."""

from dataclasses import dataclass

from tallgrass.bills import (
    BillInfo,
    bill_sort_key,
    discover_bill_urls,
    discover_bills,
    parse_js_array,
    parse_js_bill_data,
    url_to_bill_number,
)
from tallgrass.session import KSSession

# ── parse_js_array() ───────────────────────────────────────────────────────


class TestParseJsArray:
    def test_quoted_keys(self):
        js = 'var x = [{"measures_url": "/li/measures/sb1/"}]'
        result = parse_js_array(js)
        assert len(result) == 1
        assert result[0]["measures_url"] == "/li/measures/sb1/"

    def test_unquoted_keys(self):
        js = """var measures_data = [
    {
        measures_url: "/li_2018/b2017_18/measures/sb1/"
    }
]"""
        result = parse_js_array(js)
        assert len(result) == 1
        assert result[0]["measures_url"] == "/li_2018/b2017_18/measures/sb1/"

    def test_no_array(self):
        assert parse_js_array("var x = 42;") == []

    def test_empty_array(self):
        assert parse_js_array("var x = []") == []

    def test_invalid_json(self):
        assert parse_js_array("var x = [{invalid}]") == []


# ── parse_js_bill_data() ──────────────────────────────────────────────────


class TestParseJsBillData:
    def test_extracts_urls(self):
        js = """var measures_data = [
    {"measures_url": "/li_2020/b2019_20/measures/sb1/"},
    {"measures_url": "/li_2020/b2019_20/measures/hb2001/"}
]"""
        urls = parse_js_bill_data(js)
        assert len(urls) == 2
        assert "/li_2020/b2019_20/measures/sb1/" in urls

    def test_empty_url(self):
        js = 'var x = [{"measures_url": ""}]'
        assert parse_js_bill_data(js) == []

    def test_no_measures_url_key(self):
        js = 'var x = [{"other_key": "value"}]'
        assert parse_js_bill_data(js) == []

    def test_non_dict_entries(self):
        assert parse_js_bill_data("var x = [undefined];") == []


# ── bill_sort_key() ───────────────────────────────────────────────────────


class TestBillSortKey:
    def test_sb_before_hb(self):
        assert bill_sort_key("/sb1/") < bill_sort_key("/hb1/")

    def test_numerical_order(self):
        assert bill_sort_key("/sb2/") < bill_sort_key("/sb10/")

    def test_unknown_prefix(self):
        assert bill_sort_key("/unknown/") == (99, 0)

    def test_resolution_order(self):
        sr = bill_sort_key("/sr1/")
        scr = bill_sort_key("/scr1/")
        hb = bill_sort_key("/hb1/")
        assert sr < scr < hb


# ── url_to_bill_number() ─────────────────────────────────────────────────


class TestUrlToBillNumber:
    def test_senate_bill(self):
        assert url_to_bill_number("/li/b2025_26/measures/sb55/") == "SB 55"

    def test_house_bill(self):
        assert url_to_bill_number("/li_2024/b2023_24/measures/hb2084/") == "HB 2084"

    def test_resolution(self):
        assert url_to_bill_number("/li/b2025_26/measures/scr1601/") == "SCR 1601"

    def test_no_match(self):
        assert url_to_bill_number("/some/random/path/") == ""


# ── discover_bill_urls() ─────────────────────────────────────────────────


@dataclass
class MockFetchResult:
    """Minimal FetchResult-compatible object for testing."""

    ok: bool
    html: str | None = None


class TestDiscoverBillUrls:
    def test_html_discovery(self):
        """Finds bills from HTML listing pages."""
        session = KSSession.from_year(2025)

        def mock_get(url: str) -> MockFetchResult:
            if "bills/" in url:
                return MockFetchResult(
                    ok=True,
                    html=f'<a href="{session.li_prefix}/measures/sb1/">SB 1</a>'
                    f'<a href="{session.li_prefix}/measures/hb2001/">HB 2001</a>',
                )
            return MockFetchResult(ok=False)

        urls = discover_bill_urls(session, mock_get, verbose=False)
        assert len(urls) == 2
        assert any("sb1" in u for u in urls)
        assert any("hb2001" in u for u in urls)

    def test_js_fallback(self):
        """Falls back to JS data for pre-2021 sessions."""
        session = KSSession.from_year(2019)

        def mock_get(url: str) -> MockFetchResult:
            if url.endswith(".js"):
                return MockFetchResult(
                    ok=True,
                    html=f"""var measures_data = [
                        {{"measures_url": "{session.li_prefix}/measures/sb1/"}},
                        {{"measures_url": "{session.li_prefix}/measures/hb2001/"}}
                    ]""",
                )
            # HTML pages return no bill links (JS-rendered sessions)
            return MockFetchResult(ok=True, html="<html><body>No links</body></html>")

        urls = discover_bill_urls(session, mock_get, verbose=False)
        assert len(urls) == 2

    def test_empty_when_no_bills(self):
        """Returns empty list when no bills found."""
        session = KSSession.from_year(2025)

        def mock_get(url: str) -> MockFetchResult:
            return MockFetchResult(ok=True, html="<html><body></body></html>")

        urls = discover_bill_urls(session, mock_get, verbose=False)
        assert urls == []

    def test_sorted_output(self):
        """Results are sorted (SB before HB, numerically)."""
        session = KSSession.from_year(2025)

        def mock_get(url: str) -> MockFetchResult:
            return MockFetchResult(
                ok=True,
                html=(
                    f'<a href="{session.li_prefix}/measures/hb1/">HB 1</a>'
                    f'<a href="{session.li_prefix}/measures/sb99/">SB 99</a>'
                    f'<a href="{session.li_prefix}/measures/sb1/">SB 1</a>'
                ),
            )

        urls = discover_bill_urls(session, mock_get, verbose=False)
        assert len(urls) == 3
        # SB 1 < SB 99 < HB 1
        assert "sb1" in urls[0]
        assert "sb99" in urls[1]
        assert "hb1" in urls[2]


# ── discover_bills() ─────────────────────────────────────────────────────


class TestDiscoverBills:
    def test_returns_bill_info(self):
        """Returns BillInfo objects with correct bill numbers."""
        session = KSSession.from_year(2025)

        def mock_get(url: str) -> MockFetchResult:
            return MockFetchResult(
                ok=True,
                html=(
                    f'<a href="{session.li_prefix}/measures/sb55/">SB 55</a>'
                    f'<a href="{session.li_prefix}/measures/hb2084/">HB 2084</a>'
                ),
            )

        bills = discover_bills(session, mock_get, verbose=False)
        assert len(bills) == 2
        assert isinstance(bills[0], BillInfo)
        assert bills[0].bill_number == "SB 55"
        assert bills[1].bill_number == "HB 2084"

    def test_url_paths_preserved(self):
        """BillInfo.url_path contains the full path."""
        session = KSSession.from_year(2025)

        def mock_get(url: str) -> MockFetchResult:
            return MockFetchResult(
                ok=True,
                html=f'<a href="{session.li_prefix}/measures/sb1/">SB 1</a>',
            )

        bills = discover_bills(session, mock_get, verbose=False)
        assert len(bills) == 1
        assert session.li_prefix in bills[0].url_path
