"""
Tests for external validation against DIME/CFscores.

All tests use synthetic data — no downloads or disk I/O required.
Covers name normalization, DIME parsing/filtering, biennium filtering,
legislator matching, min-givers filtering, and overlap detection.

Run: uv run pytest tests/test_external_validation_dime.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.external_validation_dime_data import (
    DIME_OVERLAPPING_BIENNIUMS,
    DIME_PARTY_MAP,
    MIN_GIVERS,
    compute_correlations,
    filter_dime_to_biennium,
    has_dime_overlap,
    match_dime_legislators,
    normalize_dime_name,
)

# ── Name Normalization ───────────────────────────────────────────────────────


class TestNormalizeDimeName:
    """Normalize DIME separate lname/fname fields to "first last" lowercase."""

    def test_basic(self):
        """Standard separate fields: "timothy" + "hodge" -> "timothy hodge"."""
        assert normalize_dime_name("hodge", "timothy") == "timothy hodge"

    def test_already_lowercase(self):
        """DIME names are typically already lowercase."""
        assert normalize_dime_name("masterson", "ty") == "ty masterson"

    def test_mixed_case(self):
        """Mixed case is lowercased."""
        assert normalize_dime_name("Alcala", "John") == "john alcala"

    def test_strips_periods(self):
        """Periods are removed."""
        assert normalize_dime_name("smith jr.", "john") == "john smith jr"

    def test_drops_middle_name(self):
        """Only first token of fname is kept (middle names dropped)."""
        assert normalize_dime_name("hodge", "timothy charles") == "timothy hodge"

    def test_empty_both(self):
        """Empty fields return empty string."""
        assert normalize_dime_name("", "") == ""

    def test_empty_fname(self):
        """Empty first name returns just last name."""
        assert normalize_dime_name("hodge", "") == "hodge"

    def test_empty_lname(self):
        """Empty last name returns just first name."""
        assert normalize_dime_name("", "timothy") == "timothy"

    def test_none_handling(self):
        """None fields are handled gracefully."""
        assert normalize_dime_name(None, None) == ""
        assert normalize_dime_name("hodge", None) == "hodge"
        assert normalize_dime_name(None, "timothy") == "timothy"

    def test_extra_whitespace(self):
        """Extra whitespace is stripped."""
        assert normalize_dime_name("  hodge  ", "  timothy  ") == "timothy hodge"

    def test_suffix_in_fname(self):
        """Suffix in fname is treated as middle name and dropped."""
        assert normalize_dime_name("barker", "john jr") == "john barker"


# ── DIME Parsing (Synthetic) ────────────────────────────────────────────────


def _make_dime_df(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal DIME-like DataFrame for testing."""
    defaults = {
        "name": "",
        "lname": "",
        "fname": "",
        "party": "200",
        "party_name": "Republican",
        "state": "KS",
        "seat": "state:lower",
        "district": "KS-1",
        "cycle": 2020,
        "ico_status": "I",
        "recipient_cfscore": 0.5,
        "recipient_cfscore_dyn": 0.5,
        "num_givers": 50,
        "bonica_rid": "cand1",
        "normalized_name": "",
    }
    filled = []
    for row in rows:
        r = defaults.copy()
        r.update(row)
        filled.append(r)
    return pl.DataFrame(filled)


class TestParseDimeKansas:
    """Test DIME parsing and filtering logic using synthetic DataFrames."""

    def test_party_mapping(self):
        """Party code mapping: 100->D, 200->R, 328->Libertarian, 500->Ind."""
        assert DIME_PARTY_MAP["100"] == "Democrat"
        assert DIME_PARTY_MAP["200"] == "Republican"
        assert DIME_PARTY_MAP["328"] == "Libertarian"
        assert DIME_PARTY_MAP["500"] == "Independent"

    def test_normalized_name_present(self):
        """DIME DataFrames should include normalized_name column."""
        df = _make_dime_df(
            [
                {"lname": "hodge", "fname": "timothy", "normalized_name": "timothy hodge"},
            ]
        )
        assert "normalized_name" in df.columns
        assert df["normalized_name"][0] == "timothy hodge"

    def test_cfscore_types(self):
        """CFscore columns are Float64."""
        df = _make_dime_df(
            [
                {"recipient_cfscore": 1.23, "recipient_cfscore_dyn": -0.45},
            ]
        )
        assert df["recipient_cfscore"].dtype == pl.Float64
        assert df["recipient_cfscore_dyn"].dtype == pl.Float64

    def test_cycle_type(self):
        """Cycle column is Int64."""
        df = _make_dime_df([{"cycle": 2020}])
        assert df["cycle"].dtype == pl.Int64


# ── Biennium Filtering ───────────────────────────────────────────────────────


class TestFilterDimeToBiennium:
    """Filter DIME data by election cycle, incumbent status, and donor threshold."""

    def test_filters_by_cycle(self):
        """Only matching cycles are kept."""
        df = _make_dime_df(
            [
                {"normalized_name": "a legislator", "cycle": 2020, "seat": "state:lower"},
                {"normalized_name": "b legislator", "cycle": 2018, "seat": "state:lower"},
                {"normalized_name": "c legislator", "cycle": 2016, "seat": "state:lower"},
            ]
        )
        house, senate = filter_dime_to_biennium(df, [2020, 2022])
        assert house.height == 1
        assert house["normalized_name"][0] == "a legislator"

    def test_splits_house_senate(self):
        """state:lower -> House, state:upper -> Senate."""
        df = _make_dime_df(
            [
                {"normalized_name": "house rep", "cycle": 2020, "seat": "state:lower"},
                {"normalized_name": "senate sen", "cycle": 2020, "seat": "state:upper"},
            ]
        )
        house, senate = filter_dime_to_biennium(df, [2020])
        assert house.height == 1
        assert senate.height == 1
        assert house["normalized_name"][0] == "house rep"
        assert senate["normalized_name"][0] == "senate sen"

    def test_incumbent_only(self):
        """Only incumbent records (ico_status == "I") are kept."""
        df = _make_dime_df(
            [
                {"normalized_name": "incumbent", "cycle": 2020, "ico_status": "I"},
                {"normalized_name": "challenger", "cycle": 2020, "ico_status": "C"},
                {"normalized_name": "open seat", "cycle": 2020, "ico_status": "O"},
            ]
        )
        house, _ = filter_dime_to_biennium(df, [2020])
        assert house.height == 1
        assert house["normalized_name"][0] == "incumbent"

    def test_deduplicates_by_name(self):
        """Duplicate names across cycles are deduplicated (keep most recent)."""
        df = _make_dime_df(
            [
                {"normalized_name": "alice legislator", "cycle": 2018, "recipient_cfscore": 0.3},
                {"normalized_name": "alice legislator", "cycle": 2020, "recipient_cfscore": 0.5},
            ]
        )
        house, _ = filter_dime_to_biennium(df, [2018, 2020])
        assert house.height == 1
        # Most recent cycle (2020) should be kept
        assert house["recipient_cfscore"][0] == pytest.approx(0.5)

    def test_null_cfscore_excluded(self):
        """Null CFscores are excluded."""
        df = _make_dime_df(
            [
                {"normalized_name": "valid", "cycle": 2020, "recipient_cfscore": 0.5},
                {"normalized_name": "missing", "cycle": 2020, "recipient_cfscore": None},
            ]
        )
        house, _ = filter_dime_to_biennium(df, [2020])
        assert house.height == 1
        assert house["normalized_name"][0] == "valid"

    def test_empty_input(self):
        """Empty input returns empty DataFrames."""
        df = pl.DataFrame(
            schema={
                "name": pl.Utf8,
                "normalized_name": pl.Utf8,
                "recipient_cfscore": pl.Float64,
                "cycle": pl.Int64,
                "seat": pl.Utf8,
                "ico_status": pl.Utf8,
                "num_givers": pl.Int64,
            }
        )
        house, senate = filter_dime_to_biennium(df, [2020])
        assert house.height == 0
        assert senate.height == 0


# ── Min Givers Filter ────────────────────────────────────────────────────────


class TestMinGiversFilter:
    """Test minimum donor threshold filtering."""

    def test_below_threshold_excluded(self):
        """Candidates with fewer than min_givers donors are excluded."""
        df = _make_dime_df(
            [
                {"normalized_name": "enough donors", "cycle": 2020, "num_givers": 10},
                {"normalized_name": "too few", "cycle": 2020, "num_givers": 3},
            ]
        )
        house, _ = filter_dime_to_biennium(df, [2020], min_givers=5)
        assert house.height == 1
        assert house["normalized_name"][0] == "enough donors"

    def test_exact_threshold_included(self):
        """Candidate at exactly min_givers is included."""
        df = _make_dime_df(
            [
                {"normalized_name": "exact threshold", "cycle": 2020, "num_givers": 5},
            ]
        )
        house, _ = filter_dime_to_biennium(df, [2020], min_givers=5)
        assert house.height == 1

    def test_custom_threshold(self):
        """Custom min_givers parameter is respected."""
        df = _make_dime_df(
            [
                {"normalized_name": "leg a", "cycle": 2020, "num_givers": 15},
                {"normalized_name": "leg b", "cycle": 2020, "num_givers": 8},
            ]
        )
        house, _ = filter_dime_to_biennium(df, [2020], min_givers=10)
        assert house.height == 1
        assert house["normalized_name"][0] == "leg a"

    def test_default_threshold(self):
        """Default MIN_GIVERS constant is 5."""
        assert MIN_GIVERS == 5


# ── Legislator Matching ──────────────────────────────────────────────────────


def _make_our_df(legislators: list[dict]) -> pl.DataFrame:
    """Build a minimal IRT-style DataFrame for matching tests."""
    return pl.DataFrame(legislators)


class TestMatchDimeLegislators:
    """Match our IRT legislators to DIME candidates by normalized name."""

    def test_exact_match(self):
        """Exact normalized name match works."""
        our = _make_our_df(
            [
                {
                    "full_name": "John Alcala",
                    "xi_mean": 0.5,
                    "district": 1,
                    "party": "Democrat",
                    "legislator_slug": "rep_alcala",
                },
            ]
        )
        dime = _make_dime_df(
            [
                {
                    "name": "alcala, john",
                    "normalized_name": "john alcala",
                    "recipient_cfscore": -0.45,
                },
            ]
        )
        matched, unmatched = match_dime_legislators(our, dime, "House")
        assert matched.height == 1
        assert matched["recipient_cfscore"][0] == pytest.approx(-0.45)

    def test_phase2_last_name_match(self):
        """Phase 2: last-name match catches middle-name mismatch."""
        our = _make_our_df(
            [
                {
                    "full_name": "Robert James Wilson",
                    "xi_mean": 0.5,
                    "district": 10,
                    "party": "Republican",
                    "legislator_slug": "rep_wilson",
                },
            ]
        )
        dime = _make_dime_df(
            [
                {
                    "name": "wilson, bob",
                    "normalized_name": "bob wilson",
                    "recipient_cfscore": 0.6,
                },
            ]
        )
        matched, _ = match_dime_legislators(our, dime, "House")
        # Phase 1 won't match (robert wilson != bob wilson),
        # Phase 2 matches on last name "wilson"
        assert matched.height == 1

    def test_unmatched_report(self):
        """Unmatched legislators from both sides appear in report."""
        our = _make_our_df(
            [
                {
                    "full_name": "John Unknown",
                    "xi_mean": 0.5,
                    "district": 1,
                    "party": "Republican",
                    "legislator_slug": "rep_unknown",
                },
            ]
        )
        dime = _make_dime_df(
            [
                {
                    "name": "mystery, jane",
                    "normalized_name": "jane mystery",
                    "recipient_cfscore": -0.45,
                },
            ]
        )
        _, unmatched = match_dime_legislators(our, dime, "House")
        assert unmatched.height == 2
        sources = set(unmatched["source"].to_list())
        assert sources == {"our_data", "dime"}

    def test_empty_our_df(self):
        """Empty our_df returns empty matched."""
        our = pl.DataFrame(
            schema={
                "full_name": pl.Utf8,
                "xi_mean": pl.Float64,
                "district": pl.Int64,
                "party": pl.Utf8,
                "legislator_slug": pl.Utf8,
            }
        )
        dime = _make_dime_df(
            [
                {
                    "name": "alcala, john",
                    "normalized_name": "john alcala",
                    "recipient_cfscore": -0.45,
                },
            ]
        )
        matched, _ = match_dime_legislators(our, dime, "House")
        assert matched.height == 0

    def test_empty_dime_df(self):
        """Empty dime_df returns empty matched."""
        our = _make_our_df(
            [
                {
                    "full_name": "John Alcala",
                    "xi_mean": 0.5,
                    "district": 1,
                    "party": "Democrat",
                    "legislator_slug": "rep_alcala",
                },
            ]
        )
        dime = pl.DataFrame(
            schema={
                "name": pl.Utf8,
                "normalized_name": pl.Utf8,
                "recipient_cfscore": pl.Float64,
            }
        )
        matched, _ = match_dime_legislators(our, dime, "House")
        assert matched.height == 0

    def test_no_cross_match_different_last_names(self):
        """Different last names don't match in either phase."""
        our = _make_our_df(
            [
                {
                    "full_name": "John Alcala",
                    "xi_mean": 0.5,
                    "district": 1,
                    "party": "Democrat",
                    "legislator_slug": "rep_alcala",
                },
            ]
        )
        dime = _make_dime_df(
            [
                {
                    "name": "masterson, ty",
                    "normalized_name": "ty masterson",
                    "recipient_cfscore": 0.6,
                },
            ]
        )
        matched, _ = match_dime_legislators(our, dime, "House")
        assert matched.height == 0

    def test_multiple_matches(self):
        """Multiple legislators can match in a single call."""
        our = _make_our_df(
            [
                {
                    "full_name": "John Alcala",
                    "xi_mean": -0.5,
                    "district": 1,
                    "party": "Democrat",
                    "legislator_slug": "rep_alcala",
                },
                {
                    "full_name": "Ty Masterson",
                    "xi_mean": 1.2,
                    "district": 16,
                    "party": "Republican",
                    "legislator_slug": "sen_masterson",
                },
            ]
        )
        dime = _make_dime_df(
            [
                {
                    "name": "alcala, john",
                    "normalized_name": "john alcala",
                    "recipient_cfscore": -0.45,
                },
                {
                    "name": "masterson, ty",
                    "normalized_name": "ty masterson",
                    "recipient_cfscore": 1.1,
                },
            ]
        )
        matched, _ = match_dime_legislators(our, dime, "House")
        assert matched.height == 2

    def test_leadership_suffix_match(self):
        """Our "Ty Masterson - President" matches DIME "masterson, ty"."""
        our = _make_our_df(
            [
                {
                    "full_name": "Ty Masterson - President of the Senate",
                    "xi_mean": 1.2,
                    "district": 16,
                    "party": "Republican",
                    "legislator_slug": "sen_masterson",
                },
            ]
        )
        dime = _make_dime_df(
            [
                {
                    "name": "masterson, ty",
                    "normalized_name": "ty masterson",
                    "recipient_cfscore": 1.1,
                },
            ]
        )
        matched, _ = match_dime_legislators(our, dime, "Senate")
        assert matched.height == 1


# ── Overlap Detection ────────────────────────────────────────────────────────


class TestHasDimeOverlap:
    """Check whether a session name overlaps with DIME coverage."""

    def test_84th_yes(self):
        assert has_dime_overlap("84th_2011-2012") is True

    def test_88th_yes(self):
        assert has_dime_overlap("88th_2019-2020") is True

    def test_89th_yes(self):
        """89th is the key gain — DIME extends one biennium beyond SM."""
        assert has_dime_overlap("89th_2021-2022") is True

    def test_90th_no(self):
        """90th is too stale for meaningful validation."""
        assert has_dime_overlap("90th_2023-2024") is False

    def test_91st_no(self):
        assert has_dime_overlap("91st_2025-2026") is False

    def test_special_session_no(self):
        assert has_dime_overlap("2024s") is False

    def test_all_overlapping_bienniums(self):
        """All 6 overlapping bienniums are recognized."""
        for name in DIME_OVERLAPPING_BIENNIUMS:
            assert has_dime_overlap(name) is True

    def test_six_bienniums(self):
        """DIME covers 6 bienniums (one more than SM's 5)."""
        assert len(DIME_OVERLAPPING_BIENNIUMS) == 6


# ── Correlation Reuse ────────────────────────────────────────────────────────


class TestCorrelationReuse:
    """Verify that correlation functions work with CFscore column names."""

    def _make_matched(self, n: int = 30, seed: int = 42) -> pl.DataFrame:
        """Generate synthetic matched data with CFscore column."""
        rng = np.random.default_rng(seed)
        xi = rng.normal(0, 1, n)
        cf = 0.85 * xi + rng.normal(0, 0.3, n)
        return pl.DataFrame(
            {
                "xi_mean": xi,
                "recipient_cfscore": cf,
                "full_name": [f"Leg_{i}" for i in range(n)],
                "party": ["Republican"] * (n // 2) + ["Democrat"] * (n - n // 2),
            }
        )

    def test_correlations_with_cfscore_col(self):
        """compute_correlations works with np_col='recipient_cfscore'."""
        df = self._make_matched()
        result = compute_correlations(df, xi_col="xi_mean", np_col="recipient_cfscore")
        assert math.isfinite(result["pearson_r"])
        assert result["n"] == 30
        assert result["quality"] != "insufficient_data"

    def test_positive_correlation(self):
        """Correlated data produces positive r."""
        df = self._make_matched(n=50)
        result = compute_correlations(df, xi_col="xi_mean", np_col="recipient_cfscore")
        assert result["pearson_r"] > 0.5
