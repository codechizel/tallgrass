"""Tests for cross-session validation data logic.

Run:
    uv run pytest tests/test_cross_session.py -v
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from analysis.cross_session_data import (
    align_irt_scales,
    classify_turnover,
    compute_ideology_shift,
    compute_metric_stability,
    compute_turnover_impact,
    match_legislators,
    normalize_name,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_legislators(
    names: list[str],
    *,
    prefix: str = "rep",
    party: str = "Republican",
    chamber: str = "House",
    start_district: int = 1,
) -> pl.DataFrame:
    """Build a legislators DataFrame matching the CSV schema."""
    return pl.DataFrame(
        {
            "slug": [f"{prefix}_{n.split()[-1].lower()}" for n in names],
            "full_name": names,
            "party": [party] * len(names),
            "chamber": [chamber] * len(names),
            "district": list(range(start_district, start_district + len(names))),
        }
    )


def _make_ideal_points(
    slugs: list[str],
    xi_means: list[float],
    *,
    names: list[str] | None = None,
) -> pl.DataFrame:
    """Build an IRT ideal points DataFrame matching the parquet schema."""
    if names is None:
        names = [s.replace("rep_", "").replace("sen_", "").title() for s in slugs]
    return pl.DataFrame(
        {
            "legislator_slug": slugs,
            "xi_mean": xi_means,
            "xi_sd": [0.1] * len(slugs),
            "full_name": names,
            "party": ["Republican"] * len(slugs),
            "district": list(range(1, len(slugs) + 1)),
            "chamber": ["House"] * len(slugs),
        }
    )


def _make_large_matched(n: int = 25) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create two sessions with n overlapping legislators for alignment tests."""
    names_shared = [f"Member {i}" for i in range(n)]
    names_only_a = [f"Departing D{i}" for i in range(5)]
    names_only_b = [f"Newcomer N{i}" for i in range(5)]

    leg_a = _make_legislators(names_shared + names_only_a)
    leg_b = _make_legislators(names_shared + names_only_b, prefix="rep2")

    matched = match_legislators(leg_a, leg_b)
    return leg_a, leg_b, matched


# ── TestNormalizeName ────────────────────────────────────────────────────────


class TestNormalizeName:
    """Tests for legislator name normalization."""

    def test_lowercase(self) -> None:
        """Names should be lowercased."""
        assert normalize_name("John Smith") == "john smith"

    def test_strip_whitespace(self) -> None:
        """Leading/trailing whitespace should be removed."""
        assert normalize_name("  Jane Doe  ") == "jane doe"

    def test_strip_leadership_suffix(self) -> None:
        """Leadership titles after ' - ' should be removed."""
        assert normalize_name("Bob Jones - House Minority Caucus Chair") == "bob jones"

    def test_strip_complex_suffix(self) -> None:
        """Multiple-word suffixes should be fully removed."""
        assert normalize_name("Alice Brown - Speaker Pro Tem") == "alice brown"

    def test_no_suffix_unchanged(self) -> None:
        """Names without suffixes should pass through normally."""
        assert normalize_name("Simple Name") == "simple name"

    def test_hyphenated_name_preserved(self) -> None:
        """Hyphenated surnames should not be stripped (no space before hyphen)."""
        assert normalize_name("Mary Smith-Jones") == "mary smith-jones"

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert normalize_name("") == ""


# ── TestMatchLegislators ─────────────────────────────────────────────────────


class TestMatchLegislators:
    """Tests for cross-session legislator matching."""

    def test_exact_match(self) -> None:
        """Legislators with the same name should match."""
        names = [f"Member {i}" for i in range(25)]
        leg_a = _make_legislators(names)
        leg_b = _make_legislators(names, prefix="rep2")
        matched = match_legislators(leg_a, leg_b)
        assert matched.height == 25

    def test_case_insensitive_match(self) -> None:
        """Matching should be case-insensitive."""
        leg_a = _make_legislators(["JOHN SMITH"] + [f"M {i}" for i in range(24)])
        leg_b = _make_legislators(["john smith"] + [f"M {i}" for i in range(24)], prefix="x")
        matched = match_legislators(leg_a, leg_b)
        assert matched.height == 25

    def test_suffix_stripped_match(self) -> None:
        """Leadership suffixes should not prevent matching."""
        names_a = ["Bob Jones - Speaker"] + [f"M {i}" for i in range(24)]
        names_b = ["Bob Jones"] + [f"M {i}" for i in range(24)]
        leg_a = _make_legislators(names_a)
        leg_b = _make_legislators(names_b, prefix="x")
        matched = match_legislators(leg_a, leg_b)
        assert matched.height == 25

    def test_partial_overlap(self) -> None:
        """Only shared legislators should appear in output."""
        names_shared = [f"Shared {i}" for i in range(22)]
        leg_a = _make_legislators(names_shared + ["Only In A", "Also Only A", "Third Only A"])
        leg_b = _make_legislators(names_shared + ["Only In B", "Also Only B"], prefix="x")
        matched = match_legislators(leg_a, leg_b)
        assert matched.height == 22

    def test_too_few_matches_raises(self) -> None:
        """Should raise ValueError if overlap < MIN_OVERLAP."""
        leg_a = _make_legislators([f"A {i}" for i in range(10)])
        leg_b = _make_legislators([f"B {i}" for i in range(10)])
        with pytest.raises(ValueError, match="Only 0 legislators matched"):
            match_legislators(leg_a, leg_b)

    def test_chamber_switch_flagged(self) -> None:
        """A legislator who changed chambers should be flagged."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names, chamber="House")
        leg_b = _make_legislators(names, chamber="House", prefix="x")
        # Change one legislator's chamber in session B
        leg_b = leg_b.with_columns(
            pl.when(pl.col("slug") == "x_0")
            .then(pl.lit("Senate"))
            .otherwise("chamber")
            .alias("chamber")
        )
        matched = match_legislators(leg_a, leg_b)
        switches = matched.filter(pl.col("is_chamber_switch"))
        assert switches.height == 1

    def test_party_switch_flagged(self) -> None:
        """A legislator who changed parties should be flagged."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names, party="Republican")
        leg_b = _make_legislators(names, party="Republican", prefix="x")
        leg_b = leg_b.with_columns(
            pl.when(pl.col("slug") == "x_0")
            .then(pl.lit("Democrat"))
            .otherwise("party")
            .alias("party")
        )
        matched = match_legislators(leg_a, leg_b)
        switches = matched.filter(pl.col("is_party_switch"))
        assert switches.height == 1

    def test_slug_column_name_flexibility(self) -> None:
        """Should handle both 'slug' and 'legislator_slug' column names."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names)  # has 'slug' column
        leg_b = _make_legislators(names, prefix="x").rename({"slug": "legislator_slug"})
        matched = match_legislators(leg_a, leg_b)
        assert matched.height == 25

    def test_output_columns(self) -> None:
        """Output should have all expected columns."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names)
        leg_b = _make_legislators(names, prefix="x")
        matched = match_legislators(leg_a, leg_b)
        expected = {
            "name_norm",
            "full_name_a",
            "full_name_b",
            "slug_a",
            "slug_b",
            "party_a",
            "party_b",
            "chamber_a",
            "chamber_b",
            "district_a",
            "district_b",
            "is_chamber_switch",
            "is_party_switch",
        }
        assert set(matched.columns) == expected

    def test_sorted_by_name(self) -> None:
        """Output should be sorted by name_norm."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names)
        leg_b = _make_legislators(names, prefix="x")
        matched = match_legislators(leg_a, leg_b)
        norms = matched["name_norm"].to_list()
        assert norms == sorted(norms)


# ── TestClassifyTurnover ─────────────────────────────────────────────────────


class TestClassifyTurnover:
    """Tests for turnover classification."""

    def test_counts(self) -> None:
        """Returning + departing should equal session A; returning + new should equal session B."""
        leg_a, leg_b, matched = _make_large_matched(25)

        cohorts = classify_turnover(leg_a, leg_b, matched)
        assert cohorts["returning"].height == 25
        assert cohorts["departing"].height == 5
        assert cohorts["new"].height == 5

    def test_no_departing(self) -> None:
        """If all A legislators are in B, departing should be empty."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names)
        leg_b = _make_legislators(names + ["Extra"], prefix="x")
        matched = match_legislators(leg_a, leg_b)
        cohorts = classify_turnover(leg_a, leg_b, matched)
        assert cohorts["departing"].height == 0
        assert cohorts["new"].height == 1

    def test_no_new(self) -> None:
        """If all B legislators were in A, new should be empty."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names + ["Old Timer"])
        leg_b = _make_legislators(names, prefix="x")
        matched = match_legislators(leg_a, leg_b)
        cohorts = classify_turnover(leg_a, leg_b, matched)
        assert cohorts["departing"].height == 1
        assert cohorts["new"].height == 0


# ── TestAlignIrtScales ───────────────────────────────────────────────────────


class TestAlignIrtScales:
    """Tests for IRT scale alignment."""

    def test_identity_transform(self) -> None:
        """If both sessions have identical xi values, A~1 and B~0."""
        _, _, matched = _make_large_matched(25)
        xi_vals = [float(i) / 10 for i in range(25)]
        xi_a = _make_ideal_points(matched["slug_a"].to_list(), xi_vals)
        xi_b = _make_ideal_points(
            matched["slug_b"].to_list(),
            xi_vals,
            names=matched["full_name_b"].to_list(),
        )
        a, b, aligned = align_irt_scales(xi_a, xi_b, matched)
        assert abs(a - 1.0) < 0.05, f"A should be ~1, got {a}"
        assert abs(b) < 0.05, f"B should be ~0, got {b}"

    def test_known_affine_transform(self) -> None:
        """If session B = 2*A + 1, alignment should recover A~2, B~1."""
        _, _, matched = _make_large_matched(25)
        xi_vals_a = [float(i) / 10 for i in range(25)]
        xi_vals_b = [2 * v + 1 for v in xi_vals_a]
        xi_a = _make_ideal_points(matched["slug_a"].to_list(), xi_vals_a)
        xi_b = _make_ideal_points(
            matched["slug_b"].to_list(),
            xi_vals_b,
            names=matched["full_name_b"].to_list(),
        )
        a, b, aligned = align_irt_scales(xi_a, xi_b, matched)
        assert abs(a - 2.0) < 0.1, f"A should be ~2, got {a}"
        assert abs(b - 1.0) < 0.1, f"B should be ~1, got {b}"

    def test_aligned_delta_near_zero(self) -> None:
        """After alignment with no real movers, delta_xi should be near zero."""
        _, _, matched = _make_large_matched(25)
        xi_vals = [float(i) / 10 for i in range(25)]
        xi_a = _make_ideal_points(matched["slug_a"].to_list(), xi_vals)
        xi_b = _make_ideal_points(
            matched["slug_b"].to_list(),
            xi_vals,
            names=matched["full_name_b"].to_list(),
        )
        _, _, aligned = align_irt_scales(xi_a, xi_b, matched)
        max_delta = aligned["abs_delta_xi"].max()
        assert max_delta < 0.1, f"Max delta should be near 0, got {max_delta}"

    def test_trimming_resists_outliers(self) -> None:
        """A few genuine movers should not distort the alignment."""
        _, _, matched = _make_large_matched(30)
        xi_vals_a = [float(i) / 10 for i in range(30)]
        # Session B matches A except two extreme movers
        xi_vals_b = list(xi_vals_a)
        xi_vals_b[0] += 5.0  # Huge positive shift
        xi_vals_b[1] -= 5.0  # Huge negative shift
        xi_a = _make_ideal_points(matched["slug_a"].to_list(), xi_vals_a)
        xi_b = _make_ideal_points(
            matched["slug_b"].to_list(),
            xi_vals_b,
            names=matched["full_name_b"].to_list(),
        )
        a, b, _ = align_irt_scales(xi_a, xi_b, matched)
        # Should still recover ~identity despite outliers
        assert abs(a - 1.0) < 0.3, f"A should be ~1 despite outliers, got {a}"
        assert abs(b) < 0.3, f"B should be ~0 despite outliers, got {b}"

    def test_too_few_irt_scores_raises(self) -> None:
        """Should raise ValueError if too few legislators have IRT scores."""
        _, _, matched = _make_large_matched(25)
        # Only give IRT scores to 5 legislators
        xi_a = _make_ideal_points(matched["slug_a"].to_list()[:5], [0.1, 0.2, 0.3, 0.4, 0.5])
        xi_b = _make_ideal_points(
            matched["slug_b"].to_list()[:5],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            names=matched["full_name_b"].to_list()[:5],
        )
        with pytest.raises(ValueError, match="Only 5 legislators"):
            align_irt_scales(xi_a, xi_b, matched)

    def test_output_columns(self) -> None:
        """Aligned DataFrame should have all expected columns."""
        _, _, matched = _make_large_matched(25)
        xi_vals = [float(i) / 10 for i in range(25)]
        xi_a = _make_ideal_points(matched["slug_a"].to_list(), xi_vals)
        xi_b = _make_ideal_points(
            matched["slug_b"].to_list(),
            xi_vals,
            names=matched["full_name_b"].to_list(),
        )
        _, _, aligned = align_irt_scales(xi_a, xi_b, matched)
        expected = {
            "name_norm",
            "slug_a",
            "slug_b",
            "party",
            "chamber",
            "xi_a",
            "xi_b",
            "xi_a_aligned",
            "delta_xi",
            "abs_delta_xi",
            "full_name",
        }
        assert set(aligned.columns) == expected


# ── TestComputeIdeologyShift ─────────────────────────────────────────────────


class TestComputeIdeologyShift:
    """Tests for ideology shift classification."""

    def _make_aligned(self, deltas: list[float]) -> pl.DataFrame:
        """Build an aligned DataFrame with known delta values."""
        n = len(deltas)
        xi_a = [float(i) for i in range(n)]
        xi_b = [xi_a[i] + deltas[i] for i in range(n)]
        return pl.DataFrame(
            {
                "name_norm": [f"member {i}" for i in range(n)],
                "slug_a": [f"rep_a_{i}" for i in range(n)],
                "slug_b": [f"rep_b_{i}" for i in range(n)],
                "full_name": [f"Member {i}" for i in range(n)],
                "party": ["Republican"] * n,
                "chamber": ["House"] * n,
                "xi_a": xi_a,
                "xi_b": xi_b,
                "xi_a_aligned": xi_a,
                "delta_xi": deltas,
                "abs_delta_xi": [abs(d) for d in deltas],
            }
        )

    def test_no_movers_all_stable(self) -> None:
        """If all deltas are zero, everyone should be 'stable'."""
        aligned = self._make_aligned([0.0] * 25)
        result = compute_ideology_shift(aligned)
        assert result["is_significant_mover"].sum() == 0
        assert set(result["shift_direction"].to_list()) == {"stable"}

    def test_large_positive_shift_is_rightward(self) -> None:
        """A large positive delta should be classified as rightward."""
        deltas = [0.0] * 24 + [5.0]
        aligned = self._make_aligned(deltas)
        result = compute_ideology_shift(aligned)
        movers = result.filter(pl.col("is_significant_mover"))
        assert movers.height >= 1
        # The legislator with delta=5.0 should be rightward
        big_mover = result.filter(pl.col("delta_xi") == 5.0)
        assert big_mover["shift_direction"][0] == "rightward"

    def test_large_negative_shift_is_leftward(self) -> None:
        """A large negative delta should be classified as leftward."""
        deltas = [0.0] * 24 + [-5.0]
        aligned = self._make_aligned(deltas)
        result = compute_ideology_shift(aligned)
        big_mover = result.filter(pl.col("delta_xi") == -5.0)
        assert big_mover["shift_direction"][0] == "leftward"

    def test_rank_columns_present(self) -> None:
        """Should add rank_a, rank_b, rank_shift columns."""
        aligned = self._make_aligned([0.0] * 25)
        result = compute_ideology_shift(aligned)
        assert "rank_a" in result.columns
        assert "rank_b" in result.columns
        assert "rank_shift" in result.columns

    def test_threshold_sensitivity(self) -> None:
        """With SHIFT_THRESHOLD_SD=1.0, deltas within 1 SD should be stable."""
        # All deltas small and uniform → std is tiny → even small deltas are significant
        # Use a spread of deltas to get a meaningful threshold
        rng = np.random.default_rng(42)
        deltas = rng.normal(0, 0.1, 24).tolist() + [2.0]
        aligned = self._make_aligned(deltas)
        result = compute_ideology_shift(aligned)
        big_mover = result.filter(pl.col("delta_xi") > 1.5)
        assert big_mover["is_significant_mover"].all()


# ── TestComputeMetricStability ───────────────────────────────────────────────


class TestComputeMetricStability:
    """Tests for cross-session metric correlation."""

    def _make_leg_df(
        self, slugs: list[str], metric_vals: list[float], metric_name: str = "unity_score"
    ) -> pl.DataFrame:
        """Build a minimal legislator DataFrame with one metric."""
        return pl.DataFrame({"legislator_slug": slugs, metric_name: metric_vals})

    def test_perfect_correlation(self) -> None:
        """Identical values should give r=1.0."""
        _, _, matched = _make_large_matched(25)
        vals = [float(i) / 25 for i in range(25)]
        df_a = self._make_leg_df(matched["slug_a"].to_list(), vals)
        df_b = self._make_leg_df(matched["slug_b"].to_list(), vals)
        result = compute_metric_stability(df_a, df_b, matched, ["unity_score"])
        assert result.height == 1
        assert result["pearson_r"][0] == pytest.approx(1.0, abs=0.001)

    def test_negative_correlation(self) -> None:
        """Reversed values should give r=-1.0."""
        _, _, matched = _make_large_matched(25)
        vals = [float(i) / 25 for i in range(25)]
        df_a = self._make_leg_df(matched["slug_a"].to_list(), vals)
        df_b = self._make_leg_df(matched["slug_b"].to_list(), list(reversed(vals)))
        result = compute_metric_stability(df_a, df_b, matched, ["unity_score"])
        assert result["pearson_r"][0] == pytest.approx(-1.0, abs=0.001)

    def test_missing_metric_skipped(self) -> None:
        """Metrics not present in one DataFrame should be silently skipped."""
        _, _, matched = _make_large_matched(25)
        vals = [0.5] * 25
        df_a = self._make_leg_df(matched["slug_a"].to_list(), vals)
        df_b = pl.DataFrame({"legislator_slug": matched["slug_b"].to_list()})
        result = compute_metric_stability(df_a, df_b, matched, ["unity_score"])
        assert result.height == 0

    def test_multiple_metrics(self) -> None:
        """Should compute correlations for multiple metrics."""
        _, _, matched = _make_large_matched(25)
        slugs_a = matched["slug_a"].to_list()
        slugs_b = matched["slug_b"].to_list()
        vals = [float(i) / 25 for i in range(25)]
        df_a = pl.DataFrame(
            {"legislator_slug": slugs_a, "unity_score": vals, "maverick_rate": vals}
        )
        df_b = pl.DataFrame(
            {"legislator_slug": slugs_b, "unity_score": vals, "maverick_rate": vals}
        )
        result = compute_metric_stability(df_a, df_b, matched, ["unity_score", "maverick_rate"])
        assert result.height == 2

    def test_default_metrics(self) -> None:
        """Should use STABILITY_METRICS when no metrics specified."""
        _, _, matched = _make_large_matched(25)
        slugs_a = matched["slug_a"].to_list()
        slugs_b = matched["slug_b"].to_list()
        vals = [0.5] * 25
        df_a = pl.DataFrame({"legislator_slug": slugs_a, "unity_score": vals})
        df_b = pl.DataFrame({"legislator_slug": slugs_b, "unity_score": vals})
        result = compute_metric_stability(df_a, df_b, matched)
        # Only unity_score is present in both → 1 row
        assert result.height == 1
        assert result["metric"][0] == "unity_score"

    def test_empty_result_schema(self) -> None:
        """Empty result should have the correct schema."""
        _, _, matched = _make_large_matched(25)
        df_a = pl.DataFrame({"legislator_slug": matched["slug_a"].to_list()})
        df_b = pl.DataFrame({"legislator_slug": matched["slug_b"].to_list()})
        result = compute_metric_stability(df_a, df_b, matched, ["nonexistent"])
        assert result.height == 0
        assert set(result.columns) == {"metric", "pearson_r", "spearman_rho", "n_legislators"}


# ── TestComputeTurnoverImpact ────────────────────────────────────────────────


class TestComputeTurnoverImpact:
    """Tests for turnover impact analysis."""

    def test_basic_stats(self) -> None:
        """Should compute mean, std, n for each cohort."""
        ret = np.array([1.0, 2.0, 3.0])
        dep = np.array([0.5, 1.5])
        new = np.array([2.5, 3.5])
        result = compute_turnover_impact(ret, dep, new)
        assert result["returning_n"] == 3
        assert result["departing_n"] == 2
        assert result["new_n"] == 2
        assert result["returning_mean"] == pytest.approx(2.0)

    def test_ks_tests_present(self) -> None:
        """KS test results should be present when cohorts are large enough."""
        ret = np.array([1.0, 2.0, 3.0, 4.0])
        dep = np.array([0.5, 1.5, 2.5])
        new = np.array([2.5, 3.5, 4.5])
        result = compute_turnover_impact(ret, dep, new)
        assert "ks_departing_vs_returning" in result
        assert "p_departing_vs_returning" in result
        assert "ks_new_vs_returning" in result
        assert "p_new_vs_returning" in result

    def test_empty_cohort(self) -> None:
        """Empty arrays should produce None for mean/std."""
        ret = np.array([1.0, 2.0])
        dep = np.array([])
        new = np.array([3.0])
        result = compute_turnover_impact(ret, dep, new)
        assert result["departing_mean"] is None
        assert result["departing_std"] is None
        assert result["departing_n"] == 0
        assert "ks_departing_vs_returning" not in result

    def test_single_element_cohort(self) -> None:
        """A single-element cohort should have mean but no std."""
        ret = np.array([1.0, 2.0])
        dep = np.array([1.5])
        new = np.array([2.5])
        result = compute_turnover_impact(ret, dep, new)
        assert result["departing_mean"] == pytest.approx(1.5)
        assert result["departing_std"] is None
        # KS test needs at least 2 elements
        assert "ks_departing_vs_returning" not in result

    def test_identical_distributions(self) -> None:
        """Identical distributions should have high p-value."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_turnover_impact(arr, arr, arr)
        assert result["p_departing_vs_returning"] > 0.9
        assert result["p_new_vs_returning"] > 0.9
