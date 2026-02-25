"""
Tests for external validation against Shor-McCarty ideology scores.

All tests use synthetic data — no downloads or network access required.
Covers name normalization, SM parsing, biennium filtering, legislator
matching, correlation computation, outlier detection, and overlap logic.

Run: uv run pytest tests/test_external_validation.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.external_validation_data import (
    OVERLAPPING_BIENNIUMS,
    compute_correlations,
    compute_intra_party_correlations,
    filter_to_biennium,
    has_shor_mccarty_overlap,
    identify_outliers,
    match_legislators,
    normalize_our_name,
    normalize_sm_name,
    parse_shor_mccarty,
)

# ── Name Normalization: SM Names ─────────────────────────────────────────────


class TestNormalizeSMName:
    """Normalize Shor-McCarty "Last, First" names to "first last" lowercase."""

    def test_basic(self):
        """Standard "Last, First" -> "first last"."""
        assert normalize_sm_name("Alcala, John") == "john alcala"

    def test_strips_middle_name(self):
        """Middle names are dropped: "Arnberger, Tory Marie" -> "tory arnberger"."""
        assert normalize_sm_name("Arnberger, Tory Marie") == "tory arnberger"

    def test_strips_middle_initial(self):
        """Middle initials dropped: "Smith, John Q." -> "john smith"."""
        assert normalize_sm_name("Smith, John Q.") == "john smith"

    def test_strips_suffix_jr(self):
        """Jr suffix in first part: "Barker, John Jr." -> "john barker"."""
        # Jr in first-name portion gets dropped as middle
        assert normalize_sm_name("Barker, John Jr.") == "john barker"

    def test_lowercase(self):
        """All output is lowercase."""
        assert normalize_sm_name("MASTERSON, TY") == "ty masterson"

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert normalize_sm_name("") == ""

    def test_whitespace_only(self):
        """Whitespace-only returns empty string."""
        assert normalize_sm_name("   ") == ""

    def test_single_name(self):
        """Name without comma: just lowercases."""
        assert normalize_sm_name("Onename") == "onename"

    def test_extra_whitespace(self):
        """Extra whitespace is stripped."""
        assert normalize_sm_name("  Alcala ,  John  ") == "john alcala"


class TestNormalizeOurName:
    """Normalize our "First Last" names to "first last" lowercase."""

    def test_basic(self):
        """Standard "First Last" -> "first last"."""
        assert normalize_our_name("John Alcala") == "john alcala"

    def test_strips_leadership_suffix(self):
        """Leadership suffix removed: "Ty Masterson - President" -> "ty masterson"."""
        assert normalize_our_name("Ty Masterson - President of the Senate") == "ty masterson"

    def test_strips_middle_name(self):
        """Middle names dropped: "Tory Marie Arnberger" -> "tory arnberger"."""
        assert normalize_our_name("Tory Marie Arnberger") == "tory arnberger"

    def test_strips_sr_suffix(self):
        """Sr. suffix removed: "John Barker Sr." -> "john barker"."""
        assert normalize_our_name("John Barker Sr.") == "john barker"

    def test_strips_jr_suffix(self):
        """Jr suffix removed: "Bob Smith Jr" -> "bob smith"."""
        assert normalize_our_name("Bob Smith Jr") == "bob smith"

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert normalize_our_name("") == ""

    def test_two_word_name(self):
        """Two-word name passes through: "Tom Holland" -> "tom holland"."""
        assert normalize_our_name("Tom Holland") == "tom holland"


class TestNameMatchingEndToEnd:
    """Both normalizers produce the same canonical form for known pairs."""

    def test_alcala(self):
        assert normalize_sm_name("Alcala, John") == normalize_our_name("John Alcala")

    def test_arnberger_middle(self):
        assert normalize_sm_name("Arnberger, Tory") == normalize_our_name("Tory Marie Arnberger")

    def test_masterson_leadership(self):
        assert normalize_sm_name("Masterson, Ty") == normalize_our_name("Ty Masterson - President")

    def test_barker_suffix(self):
        assert normalize_sm_name("Barker, John") == normalize_our_name("John Barker Sr.")


# ── SM Parsing ───────────────────────────────────────────────────────────────


def _make_tab_data(rows: list[dict], header: list[str] | None = None) -> str:
    """Build tab-separated text from a list of row dicts."""
    if not rows:
        return ""
    cols = header or list(rows[0].keys())
    lines = ["\t".join(cols)]
    for row in rows:
        lines.append("\t".join(str(row.get(c, "")) for c in cols))
    return "\n".join(lines)


class TestParseShorMcCarty:
    """Parse tab data and filter to Kansas."""

    def test_filters_to_ks(self):
        """Only KS rows are kept."""
        raw = _make_tab_data(
            [
                {"name": "Smith, John", "st": "KS", "np_score": "0.5", "house2019": "1"},
                {"name": "Jones, Bob", "st": "MO", "np_score": "0.3", "house2019": "1"},
            ]
        )
        df = parse_shor_mccarty(raw)
        assert df.height == 1
        assert df["name"][0] == "Smith, John"

    def test_casts_np_score(self):
        """np_score is cast to Float64."""
        raw = _make_tab_data(
            [
                {"name": "Smith, John", "st": "KS", "np_score": "1.234", "house2019": "1"},
            ]
        )
        df = parse_shor_mccarty(raw)
        assert df["np_score"].dtype == pl.Float64
        assert df["np_score"][0] == pytest.approx(1.234)

    def test_drops_null_np_score(self):
        """Rows with empty np_score are dropped."""
        raw = _make_tab_data(
            [
                {"name": "Smith, John", "st": "KS", "np_score": "", "house2019": "1"},
                {"name": "Jones, Bob", "st": "KS", "np_score": "0.5", "house2019": "1"},
            ]
        )
        df = parse_shor_mccarty(raw)
        assert df.height == 1
        assert df["name"][0] == "Jones, Bob"

    def test_adds_normalized_name(self):
        """Parsed data includes normalized_name column."""
        raw = _make_tab_data(
            [
                {"name": "Alcala, John", "st": "KS", "np_score": "0.5", "house2019": "1"},
            ]
        )
        df = parse_shor_mccarty(raw)
        assert "normalized_name" in df.columns
        assert df["normalized_name"][0] == "john alcala"

    def test_casts_year_columns(self):
        """Year indicator columns are cast to Int64."""
        raw = _make_tab_data(
            [
                {
                    "name": "Smith, John",
                    "st": "KS",
                    "np_score": "0.5",
                    "house2019": "1",
                    "senate2019": "0",
                },
            ]
        )
        df = parse_shor_mccarty(raw)
        assert df["house2019"].dtype == pl.Int64
        assert df["house2019"][0] == 1

    def test_empty_input(self):
        """Empty input returns empty DataFrame."""
        df = parse_shor_mccarty("")
        assert df.height == 0

    def test_header_only(self):
        """Header-only input returns empty DataFrame."""
        df = parse_shor_mccarty("name\tst\tnp_score")
        assert df.height == 0

    def test_no_ks_rows(self):
        """No KS rows returns empty DataFrame."""
        raw = _make_tab_data(
            [
                {"name": "Jones, Bob", "st": "MO", "np_score": "0.3"},
            ]
        )
        df = parse_shor_mccarty(raw)
        assert df.height == 0

    def test_quoted_values(self):
        """Real SM data has quoted values: "KS" not KS. Quotes are stripped."""
        raw = (
            "name\tst\tnp_score\thouse2019\n"
            '"Alcala, John"\t"KS"\t0.5\t1\n'
            '"Jones, Bob"\t"MO"\t0.3\t1'
        )
        df = parse_shor_mccarty(raw)
        assert df.height == 1
        assert df["name"][0] == "Alcala, John"
        assert df["normalized_name"][0] == "john alcala"


# ── Biennium Filtering ───────────────────────────────────────────────────────


def _make_sm_df(legislators: list[dict]) -> pl.DataFrame:
    """Build a minimal SM-style DataFrame for filtering tests."""
    return pl.DataFrame(legislators)


class TestFilterToBiennium:
    """Filter SM data by biennium year indicator columns."""

    def test_house_filter(self):
        """House legislator with house2019=1 appears in 2019-2020 house filter."""
        df = _make_sm_df(
            [
                {
                    "name": "A",
                    "normalized_name": "a",
                    "np_score": 0.5,
                    "house2019": 1,
                    "house2020": 1,
                    "senate2019": 0,
                    "senate2020": 0,
                },
                {
                    "name": "B",
                    "normalized_name": "b",
                    "np_score": 0.3,
                    "house2019": 0,
                    "house2020": 0,
                    "senate2019": 1,
                    "senate2020": 1,
                },
            ]
        )
        house, senate = filter_to_biennium(df, 2019, 2020)
        assert house.height == 1
        assert house["name"][0] == "A"
        assert senate.height == 1
        assert senate["name"][0] == "B"

    def test_no_matching_years(self):
        """No matching year columns returns empty DataFrames."""
        df = _make_sm_df(
            [
                {
                    "name": "A",
                    "normalized_name": "a",
                    "np_score": 0.5,
                    "house2019": 1,
                    "senate2019": 0,
                },
            ]
        )
        house, senate = filter_to_biennium(df, 2011, 2012)
        assert house.height == 0
        assert senate.height == 0

    def test_deduplicates_by_name(self):
        """Duplicate names within a biennium are deduplicated."""
        df = _make_sm_df(
            [
                {
                    "name": "A",
                    "normalized_name": "a",
                    "np_score": 0.5,
                    "house2019": 1,
                    "house2020": 1,
                },
                {
                    "name": "A",
                    "normalized_name": "a",
                    "np_score": 0.5,
                    "house2019": 1,
                    "house2020": 1,
                },
            ]
        )
        house, _ = filter_to_biennium(df, 2019, 2020)
        assert house.height == 1

    def test_partial_year_match(self):
        """Active in only one year of biennium still matches."""
        df = _make_sm_df(
            [
                {
                    "name": "A",
                    "normalized_name": "a",
                    "np_score": 0.5,
                    "house2019": 1,
                    "house2020": 0,
                },
            ]
        )
        house, _ = filter_to_biennium(df, 2019, 2020)
        assert house.height == 1

    def test_zero_means_inactive(self):
        """house2019=0 and house2020=0 means not active in house."""
        df = _make_sm_df(
            [
                {
                    "name": "A",
                    "normalized_name": "a",
                    "np_score": 0.5,
                    "house2019": 0,
                    "house2020": 0,
                    "senate2019": 1,
                    "senate2020": 0,
                },
            ]
        )
        house, senate = filter_to_biennium(df, 2019, 2020)
        assert house.height == 0
        assert senate.height == 1

    def test_both_chambers(self):
        """Legislator active in both chambers appears in both."""
        df = _make_sm_df(
            [
                {
                    "name": "A",
                    "normalized_name": "a",
                    "np_score": 0.5,
                    "house2019": 1,
                    "house2020": 0,
                    "senate2019": 0,
                    "senate2020": 1,
                },
            ]
        )
        house, senate = filter_to_biennium(df, 2019, 2020)
        assert house.height == 1
        assert senate.height == 1

    def test_empty_input(self):
        """Empty input returns empty DataFrames."""
        df = pl.DataFrame(
            schema={"name": pl.Utf8, "normalized_name": pl.Utf8, "np_score": pl.Float64}
        )
        house, senate = filter_to_biennium(df, 2019, 2020)
        assert house.height == 0


# ── Legislator Matching ──────────────────────────────────────────────────────


def _make_our_df(legislators: list[dict]) -> pl.DataFrame:
    """Build a minimal IRT-style DataFrame for matching tests."""
    return pl.DataFrame(legislators)


class TestMatchLegislators:
    """Match our IRT legislators to SM legislators by normalized name."""

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
        sm = pl.DataFrame(
            [
                {"name": "Alcala, John", "normalized_name": "john alcala", "np_score": 0.45},
            ]
        )
        matched, unmatched = match_legislators(our, sm, "House")
        assert matched.height == 1
        assert matched["np_score"][0] == pytest.approx(0.45)

    def test_middle_name_tolerance(self):
        """Our "Tory Marie Arnberger" matches SM "Arnberger, Tory"."""
        our = _make_our_df(
            [
                {
                    "full_name": "Tory Marie Arnberger",
                    "xi_mean": 0.3,
                    "district": 5,
                    "party": "Republican",
                    "legislator_slug": "rep_arnberger",
                },
            ]
        )
        sm = pl.DataFrame(
            [
                {"name": "Arnberger, Tory", "normalized_name": "tory arnberger", "np_score": 0.35},
            ]
        )
        matched, _ = match_legislators(our, sm, "House")
        assert matched.height == 1

    def test_leadership_suffix(self):
        """Our "Ty Masterson - President" matches SM "Masterson, Ty"."""
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
        sm = pl.DataFrame(
            [
                {"name": "Masterson, Ty", "normalized_name": "ty masterson", "np_score": 1.1},
            ]
        )
        matched, _ = match_legislators(our, sm, "Senate")
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
        sm = pl.DataFrame(
            [
                {"name": "Mystery, Jane", "normalized_name": "jane mystery", "np_score": 0.45},
            ]
        )
        _, unmatched = match_legislators(our, sm, "House")
        assert unmatched.height == 2
        sources = set(unmatched["source"].to_list())
        assert sources == {"our_data", "shor_mccarty"}

    def test_empty_our_df(self):
        """Empty our_df returns empty matched and SM in unmatched."""
        our = pl.DataFrame(
            schema={
                "full_name": pl.Utf8,
                "xi_mean": pl.Float64,
                "district": pl.Int64,
                "party": pl.Utf8,
                "legislator_slug": pl.Utf8,
            }
        )
        sm = pl.DataFrame(
            [
                {"name": "Alcala, John", "normalized_name": "john alcala", "np_score": 0.45},
            ]
        )
        matched, unmatched = match_legislators(our, sm, "House")
        assert matched.height == 0

    def test_empty_sm_df(self):
        """Empty sm_df returns empty matched and our data in unmatched."""
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
        sm = pl.DataFrame(
            schema={"name": pl.Utf8, "normalized_name": pl.Utf8, "np_score": pl.Float64}
        )
        matched, unmatched = match_legislators(our, sm, "House")
        assert matched.height == 0

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
        sm = pl.DataFrame(
            [
                {"name": "Wilson, Bob", "normalized_name": "bob wilson", "np_score": 0.6},
            ]
        )
        matched, _ = match_legislators(our, sm, "House")
        # Phase 1 won't match (robert wilson != bob wilson),
        # Phase 2 matches on last name "wilson"
        assert matched.height == 1

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
        sm = pl.DataFrame(
            [
                {"name": "Masterson, Ty", "normalized_name": "ty masterson", "np_score": 0.6},
            ]
        )
        matched, _ = match_legislators(our, sm, "House")
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
        sm = pl.DataFrame(
            [
                {"name": "Alcala, John", "normalized_name": "john alcala", "np_score": -0.45},
                {"name": "Masterson, Ty", "normalized_name": "ty masterson", "np_score": 1.1},
            ]
        )
        matched, _ = match_legislators(our, sm, "Senate")
        assert matched.height == 2


# ── Correlations ─────────────────────────────────────────────────────────────


def _make_matched(n: int, r: float = 0.95, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic matched data with known correlation."""
    rng = np.random.default_rng(seed)
    xi = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    np_scores = r * xi + math.sqrt(1 - r**2) * noise
    return pl.DataFrame(
        {
            "xi_mean": xi,
            "np_score": np_scores,
            "full_name": [f"Leg_{i}" for i in range(n)],
            "party": ["Republican"] * (n // 2) + ["Democrat"] * (n - n // 2),
        }
    )


class TestComputeCorrelations:
    """Correlation computation with Pearson r, Spearman rho, Fisher z CI."""

    def test_perfect_correlation(self):
        """Perfectly correlated data produces r ≈ 1.0."""
        df = pl.DataFrame(
            {
                "xi_mean": list(range(20)),
                "np_score": list(range(20)),
            }
        )
        result = compute_correlations(df)
        assert result["pearson_r"] == pytest.approx(1.0)
        assert result["spearman_rho"] == pytest.approx(1.0)

    def test_negative_correlation(self):
        """Perfectly anti-correlated data produces r ≈ -1.0."""
        df = pl.DataFrame(
            {
                "xi_mean": list(range(20)),
                "np_score": list(range(19, -1, -1)),
            }
        )
        result = compute_correlations(df)
        assert result["pearson_r"] == pytest.approx(-1.0)

    def test_ci_contains_true_r(self):
        """Fisher z CI should contain the true correlation (approximately)."""
        df = _make_matched(100, r=0.9, seed=42)
        result = compute_correlations(df)
        # CI should be reasonable — not testing statistical property, just validity
        assert result["ci_lower"] < result["pearson_r"]
        assert result["ci_upper"] > result["pearson_r"]

    def test_quality_strong(self):
        """r >= 0.90 gets 'strong' label."""
        df = _make_matched(100, r=0.95, seed=42)
        result = compute_correlations(df)
        # With n=100 and r=0.95, observed r should be close to 0.95
        assert result["quality"] in ("strong", "good")  # allow slight randomness

    def test_quality_concern(self):
        """r < 0.70 gets 'concern' label."""
        df = _make_matched(100, r=0.3, seed=42)
        result = compute_correlations(df)
        assert result["quality"] == "concern"

    def test_insufficient_data(self):
        """Fewer than MIN_MATCHED legislators returns insufficient_data."""
        df = pl.DataFrame(
            {
                "xi_mean": [1.0, 2.0],
                "np_score": [1.0, 2.0],
            }
        )
        result = compute_correlations(df)
        assert result["quality"] == "insufficient_data"
        assert math.isnan(result["pearson_r"])

    def test_n_count(self):
        """Returned n matches actual matched count."""
        df = _make_matched(50)
        result = compute_correlations(df)
        assert result["n"] == 50

    def test_p_values_present(self):
        """Both p-values are returned and finite."""
        df = _make_matched(50, r=0.9)
        result = compute_correlations(df)
        assert math.isfinite(result["pearson_p"])
        assert math.isfinite(result["spearman_p"])


class TestIntraPartyCorrelations:
    """Within-party correlation computation."""

    def test_returns_both_parties(self):
        """Returns results for both Republican and Democrat."""
        df = _make_matched(40, r=0.9)
        result = compute_intra_party_correlations(df)
        assert "Republican" in result
        assert "Democrat" in result

    def test_each_party_has_correlation(self):
        """Each party's result has pearson_r."""
        df = _make_matched(40, r=0.9)
        result = compute_intra_party_correlations(df)
        for party in ["Republican", "Democrat"]:
            assert "pearson_r" in result[party]


# ── Outlier Detection ────────────────────────────────────────────────────────


class TestIdentifyOutliers:
    """Z-score discrepancy outlier detection."""

    def test_top_n_limit(self):
        """Returns at most top_n outliers."""
        df = _make_matched(50, r=0.8)
        outliers = identify_outliers(df, top_n=3)
        assert outliers.height <= 3

    def test_highest_discrepancy_first(self):
        """Outliers are sorted by discrepancy descending."""
        df = _make_matched(50, r=0.8, seed=123)
        outliers = identify_outliers(df, top_n=5)
        if outliers.height >= 2:
            discs = outliers["discrepancy_z"].to_list()
            assert discs[0] >= discs[1]

    def test_has_z_scores(self):
        """Output includes xi_z, np_z, discrepancy_z columns."""
        df = _make_matched(30, r=0.8)
        outliers = identify_outliers(df)
        assert "xi_z" in outliers.columns
        assert "np_z" in outliers.columns
        assert "discrepancy_z" in outliers.columns

    def test_empty_with_few_rows(self):
        """Fewer than 3 rows returns empty DataFrame."""
        df = pl.DataFrame({"xi_mean": [1.0], "np_score": [2.0]})
        outliers = identify_outliers(df)
        assert outliers.height == 0

    def test_zero_std_handled(self):
        """All-same values don't crash (zero std handled)."""
        df = pl.DataFrame(
            {
                "xi_mean": [1.0] * 10,
                "np_score": [1.0] * 10,
                "full_name": [f"Leg_{i}" for i in range(10)],
            }
        )
        outliers = identify_outliers(df, top_n=3)
        assert outliers.height <= 3


# ── Overlap Detection ────────────────────────────────────────────────────────


class TestHasShorMcCartyOverlap:
    """Check whether a session name overlaps with SM coverage."""

    def test_84th_yes(self):
        assert has_shor_mccarty_overlap("84th_2011-2012") is True

    def test_88th_yes(self):
        assert has_shor_mccarty_overlap("88th_2019-2020") is True

    def test_89th_no(self):
        assert has_shor_mccarty_overlap("89th_2021-2022") is False

    def test_91st_no(self):
        assert has_shor_mccarty_overlap("91st_2025-2026") is False

    def test_special_session_no(self):
        assert has_shor_mccarty_overlap("2024s") is False

    def test_all_overlapping_bienniums(self):
        """All 5 overlapping bienniums are recognized."""
        for name in OVERLAPPING_BIENNIUMS:
            assert has_shor_mccarty_overlap(name) is True
