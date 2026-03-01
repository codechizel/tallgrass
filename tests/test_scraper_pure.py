"""
Tests for pure functions and staticmethods in scraper.py.

Covers bill code normalization, vote datetime parsing, motion classification,
result derivation (Bug #4 regression), elapsed time formatting, bill sort keys,
FetchResult dataclass behavior (including binary mode), and JS bill data parsing.

Run: uv run pytest tests/test_scraper_pure.py -v
"""

import dataclasses
import json

import pytest

from tallgrass.scraper import FetchResult, KSVoteScraper, _normalize_bill_code

pytestmark = pytest.mark.scraper

# ── _normalize_bill_code() ───────────────────────────────────────────────────


class TestNormalizeBillCode:
    """Compact lowercase bill identifiers: 'SB 1' → 'sb1'."""

    def test_standard(self):
        assert _normalize_bill_code("SB 1") == "sb1"

    def test_house_bill(self):
        assert _normalize_bill_code("HB 2124") == "hb2124"

    def test_extra_whitespace(self):
        assert _normalize_bill_code("  SB   1  ") == "sb1"

    def test_already_lowercase(self):
        assert _normalize_bill_code("sb1") == "sb1"

    def test_resolution(self):
        assert _normalize_bill_code("SCR 1601") == "scr1601"


# ── _parse_vote_datetime() ───────────────────────────────────────────────────


class TestParseVoteDatetime:
    """Extract ISO 8601 datetime from vote_id timestamps."""

    def test_standard_vote_id(self):
        result = KSVoteScraper._parse_vote_datetime("je_20250320203513")
        assert result == "2025-03-20T20:35:13"

    def test_with_suffix(self):
        """Some vote IDs have extra characters after the timestamp."""
        result = KSVoteScraper._parse_vote_datetime("je_20250320203513_abc")
        assert result == "2025-03-20T20:35:13"

    def test_no_match_returns_empty(self):
        assert KSVoteScraper._parse_vote_datetime("no_timestamp_here") == ""

    def test_malformed_short(self):
        """Too few digits — no 14-digit sequence to match."""
        assert KSVoteScraper._parse_vote_datetime("je_2025032020") == ""


# ── _parse_vote_type_and_result() ────────────────────────────────────────────


class TestParseVoteTypeAndResult:
    """Classify motion text into (vote_type, result) tuples."""

    def test_empty_motion(self):
        assert KSVoteScraper._parse_vote_type_and_result("") == ("", "")

    def test_emergency_final_action(self):
        vtype, result = KSVoteScraper._parse_vote_type_and_result(
            "Emergency Final Action - Passed as amended"
        )
        assert vtype == "Emergency Final Action"
        assert result == "Passed as amended"

    def test_final_action(self):
        vtype, result = KSVoteScraper._parse_vote_type_and_result("Final Action - Passed")
        assert vtype == "Final Action"
        assert result == "Passed"

    def test_committee_of_the_whole(self):
        vtype, result = KSVoteScraper._parse_vote_type_and_result(
            "Committee of the Whole - Be passed as amended"
        )
        assert vtype == "Committee of the Whole"
        assert result == "Be passed as amended"

    def test_consent_calendar(self):
        vtype, result = KSVoteScraper._parse_vote_type_and_result("Consent Calendar - Passed")
        assert vtype == "Consent Calendar"
        assert result == "Passed"

    def test_veto_override(self):
        vtype, result = KSVoteScraper._parse_vote_type_and_result(
            "Motion to override the Governor's veto"
        )
        assert vtype == "Veto Override"

    def test_conference_committee(self):
        vtype, _ = KSVoteScraper._parse_vote_type_and_result(
            "Adopt conference committee report on SB 1"
        )
        assert vtype == "Conference Committee"

    def test_concurrence(self):
        vtype, _ = KSVoteScraper._parse_vote_type_and_result("Concur in House amendments to SB 25")
        assert vtype == "Concurrence"

    def test_procedural_motion(self):
        vtype, _ = KSVoteScraper._parse_vote_type_and_result("Motion to table")
        assert vtype == "Procedural Motion"

    def test_unrecognized_returns_empty_type(self):
        vtype, result = KSVoteScraper._parse_vote_type_and_result("Something unusual happened")
        assert vtype == ""
        assert result == "Something unusual happened"


# ── _derive_passed() ────────────────────────────────────────────────────────
# Bug #4 regression: "Not passed" must return False, not True.


class TestDerivePassed:
    """Derive passed boolean from result text. Regression tests for Bug #4."""

    def test_not_passed(self):
        """'Not passed' contains 'passed' but must be False."""
        assert KSVoteScraper._derive_passed("Not passed") is False

    def test_passed(self):
        assert KSVoteScraper._derive_passed("Passed") is True

    def test_passed_as_amended(self):
        assert KSVoteScraper._derive_passed("Passed as amended") is True

    def test_failed(self):
        assert KSVoteScraper._derive_passed("Failed") is False

    def test_adopted(self):
        assert KSVoteScraper._derive_passed("Adopted") is True

    def test_rejected(self):
        assert KSVoteScraper._derive_passed("Rejected") is False

    def test_sustained(self):
        """'Veto sustained' means the bill failed."""
        assert KSVoteScraper._derive_passed("Sustained") is False

    def test_concurred(self):
        assert KSVoteScraper._derive_passed("Concurred") is True

    def test_prevailed(self):
        assert KSVoteScraper._derive_passed("Prevailed") is True

    def test_empty_returns_none(self):
        assert KSVoteScraper._derive_passed("") is None

    def test_unrecognized_returns_none(self):
        assert KSVoteScraper._derive_passed("Some other text") is None


# ── _fmt_elapsed() ───────────────────────────────────────────────────────────


class TestFmtElapsed:
    """Format elapsed time for display."""

    def test_under_60s(self):
        assert KSVoteScraper._fmt_elapsed(42.3) == "42.3s"

    def test_exactly_60s(self):
        assert KSVoteScraper._fmt_elapsed(60.0) == "1m 0s"

    def test_over_60s(self):
        assert KSVoteScraper._fmt_elapsed(125.7) == "2m 5s"

    def test_zero(self):
        assert KSVoteScraper._fmt_elapsed(0.0) == "0.0s"


# ── _bill_sort_key() ────────────────────────────────────────────────────────


class TestBillSortKey:
    """Sort bills: SB before HB, then numerically."""

    def test_sb_before_hb(self):
        assert KSVoteScraper._bill_sort_key("/sb1/") < KSVoteScraper._bill_sort_key("/hb1/")

    def test_numerical_order(self):
        assert KSVoteScraper._bill_sort_key("/sb2/") < KSVoteScraper._bill_sort_key("/sb10/")

    def test_unknown_prefix_sorts_last(self):
        """URLs that don't match get sort key (99, 0)."""
        assert KSVoteScraper._bill_sort_key("/unknown/") == (99, 0)

    def test_resolution_types(self):
        """SCR sorts between SR and HB."""
        sr = KSVoteScraper._bill_sort_key("/sr1/")
        scr = KSVoteScraper._bill_sort_key("/scr1/")
        hb = KSVoteScraper._bill_sort_key("/hb1/")
        assert sr < scr < hb


# ── FetchResult ──────────────────────────────────────────────────────────────


class TestFetchResult:
    """FetchResult dataclass: .ok property and immutability."""

    def test_ok_when_html_set(self):
        r = FetchResult(url="http://example.com", html="<html></html>")
        assert r.ok is True

    def test_not_ok_when_html_none(self):
        r = FetchResult(url="http://example.com", html=None, error_type="permanent")
        assert r.ok is False

    def test_frozen(self):
        r = FetchResult(url="http://example.com", html="<html></html>")
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.url = "other"  # type: ignore[misc]


# ── FetchResult binary mode ─────────────────────────────────────────────────


class TestFetchResultBinary:
    """FetchResult with content_bytes for binary downloads (ODT files)."""

    def test_ok_when_content_bytes_set(self):
        r = FetchResult(url="http://example.com", html=None, content_bytes=b"\x50\x4b")
        assert r.ok is True

    def test_not_ok_when_both_none(self):
        r = FetchResult(url="http://example.com", html=None, content_bytes=None)
        assert r.ok is False

    def test_ok_with_html_and_no_bytes(self):
        """Backwards-compatible: html-only FetchResult still works."""
        r = FetchResult(url="http://example.com", html="<html></html>")
        assert r.ok is True
        assert r.content_bytes is None

    def test_content_bytes_default_none(self):
        r = FetchResult(url="http://example.com", html="<html></html>")
        assert r.content_bytes is None


# ── _parse_js_bill_data() ───────────────────────────────────────────────────


class TestParseJsBillData:
    """Parse bill URLs from JavaScript data files (pre-2021 sessions)."""

    def test_standard_js_file(self):
        """JS file with measures_data containing measures_url fields."""
        data = [
            {"measures_url": "/li_2020/b2019_20/measures/hb2001/", "title": "Tax bill"},
            {"measures_url": "/li_2020/b2019_20/measures/sb100/", "title": "Budget"},
        ]
        js_content = f"var measures_data = {json.dumps(data)};"
        urls = KSVoteScraper._parse_js_bill_data(js_content)
        assert len(urls) == 2
        assert "/li_2020/b2019_20/measures/hb2001/" in urls

    def test_empty_array(self):
        js_content = "var measures_data = [];"
        assert KSVoteScraper._parse_js_bill_data(js_content) == []

    def test_no_brackets(self):
        """Malformed JS content with no array brackets."""
        assert KSVoteScraper._parse_js_bill_data("var x = 42;") == []

    def test_invalid_json(self):
        """Brackets present but content isn't valid JSON."""
        assert KSVoteScraper._parse_js_bill_data("var x = [undefined];") == []

    def test_missing_measures_url(self):
        """Entries without measures_url are skipped."""
        data = [{"title": "no url"}, {"measures_url": "/sb1/"}]
        js_content = f"var measures_data = {json.dumps(data)};"
        urls = KSVoteScraper._parse_js_bill_data(js_content)
        assert len(urls) == 1
        assert urls[0] == "/sb1/"

    def test_unquoted_js_keys(self):
        """Sessions 2017-18 and earlier use unquoted JS object literal keys.

        Run: uv run pytest tests/test_scraper_pure.py -k test_unquoted_js_keys
        """
        js_content = (
            "let measures_data = [\n"
            "    {\n"
            '        measures_url: "/li_2018/b2017_18/measures/sb1/",\n'
            '        billno: "sb1",\n'
            '        display_text: "SB1 - Some bill title"\n'
            "    },\n"
            "    {\n"
            '        measures_url: "/li_2018/b2017_18/measures/hb2001/",\n'
            '        billno: "hb2001",\n'
            '        display_text: "HB2001 - Another bill"\n'
            "    }\n"
            "];"
        )
        urls = KSVoteScraper._parse_js_bill_data(js_content)
        assert len(urls) == 2
        assert "/li_2018/b2017_18/measures/sb1/" in urls
        assert "/li_2018/b2017_18/measures/hb2001/" in urls
