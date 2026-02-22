"""
Tests for pure functions and staticmethods in scraper.py.

Covers bill code normalization, vote datetime parsing, motion classification,
result derivation (Bug #4 regression), elapsed time formatting, bill sort keys,
and FetchResult dataclass behavior. No BeautifulSoup or HTTP required.

Run: uv run pytest tests/test_scraper_pure.py -v
"""

import dataclasses

import pytest

from ks_vote_scraper.scraper import FetchResult, KSVoteScraper, _normalize_bill_code

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
        vtype, result = KSVoteScraper._parse_vote_type_and_result(
            "Consent Calendar - Passed"
        )
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
