"""Tests for Phase 19: Issue-Specific Ideal Points.

Synthetic data tests for issue_irt_data.py — pure functions only, no MCMC.
Follows the class-based pattern from test_tbip.py.

Run:
    uv run pytest tests/test_issue_irt.py -v
"""

import math

import numpy as np
import polars as pl
import pytest
from analysis.issue_irt_data import (
    ESS_THRESHOLD,
    GOOD_CORRELATION,
    MIN_BILLS_PER_TOPIC,
    MIN_LEGISLATORS_PER_TOPIC,
    MIN_VOTES_IN_TOPIC,
    MODERATE_CORRELATION,
    OUTLIER_TOP_N,
    PARTY_COLORS,
    RHAT_THRESHOLD,
    STRONG_CORRELATION,
    _fisher_z_ci,
    _quality_label,
    _sanitize_label,
    align_topic_ideal_points,
    build_cross_topic_matrix,
    check_anchor_stability,
    compute_cross_topic_correlations,
    compute_topic_irt_correlation,
    compute_topic_pca_scores,
    filter_legislators_in_topic,
    get_eligible_topics,
    identify_topic_outliers,
    subset_vote_matrix_for_topic,
)

# ── Factories ────────────────────────────────────────────────────────────────


def _make_topics(rows: list[dict]) -> pl.DataFrame:
    """Build a topic assignments DataFrame."""
    defaults = {"bill_number": "HB 1", "topic_id": 0, "topic_label": "Topic 0: test"}
    filled = [{**defaults, **r} for r in rows]
    return pl.DataFrame(filled)


def _make_rollcalls(rows: list[dict]) -> pl.DataFrame:
    """Build a rollcalls DataFrame."""
    defaults = {"vote_id": "v1", "bill_number": "HB 1", "chamber": "House"}
    filled = [{**defaults, **r} for r in rows]
    return pl.DataFrame(filled)


def _make_vote_matrix(n_legislators: int = 10, n_votes: int = 5, seed: int = 42) -> pl.DataFrame:
    """Build a synthetic wide vote matrix."""
    rng = np.random.default_rng(seed)
    slugs = [f"rep_{chr(97 + i)}_1" for i in range(n_legislators)]
    vote_ids = [f"v{i + 1}" for i in range(n_votes)]

    data: dict[str, list] = {"legislator_slug": slugs}
    for vid in vote_ids:
        # Mix of 0, 1, and None
        col: list = []
        for _ in range(n_legislators):
            r = rng.random()
            if r < 0.15:
                col.append(None)
            elif r < 0.55:
                col.append(1)
            else:
                col.append(0)
        data[vid] = col

    return pl.DataFrame(data)


def _make_irt(rows: list[dict]) -> pl.DataFrame:
    """Build an IRT ideal points DataFrame."""
    defaults = {
        "legislator_slug": "rep_a_1",
        "xi_mean": 0.5,
        "xi_sd": 0.1,
        "xi_hdi_2.5": 0.3,
        "xi_hdi_97.5": 0.7,
        "full_name": "John Doe",
        "party": "Republican",
        "district": 1,
        "chamber": "house",
    }
    filled = [{**defaults, **r} for r in rows]
    return pl.DataFrame(filled)


def _make_full_irt(n: int = 10, seed: int = 42) -> pl.DataFrame:
    """Build synthetic full-model IRT ideal points."""
    rng = np.random.default_rng(seed)
    slugs = [f"rep_{chr(97 + i)}_1" for i in range(n)]
    xi = np.linspace(-2, 2, n) + rng.normal(0, 0.1, n)
    parties = ["Democrat"] * (n // 3) + ["Republican"] * (n - n // 3)
    names = [f"Person {chr(65 + i)}" for i in range(n)]

    return pl.DataFrame(
        {
            "legislator_slug": slugs,
            "xi_mean": xi.tolist(),
            "xi_sd": [0.1] * n,
            "xi_hdi_2.5": (xi - 0.2).tolist(),
            "xi_hdi_97.5": (xi + 0.2).tolist(),
            "full_name": names,
            "party": parties,
            "district": list(range(1, n + 1)),
        }
    )


# ── Constants ────────────────────────────────────────────────────────────────


class TestConstants:
    """Verify constants are internally consistent."""

    def test_thresholds_ordered(self):
        assert STRONG_CORRELATION > GOOD_CORRELATION > MODERATE_CORRELATION > 0

    def test_min_bills_positive(self):
        assert MIN_BILLS_PER_TOPIC > 0

    def test_min_legislators_positive(self):
        assert MIN_LEGISLATORS_PER_TOPIC > 0

    def test_min_votes_positive(self):
        assert MIN_VOTES_IN_TOPIC > 0

    def test_rhat_threshold_reasonable(self):
        assert 1.0 < RHAT_THRESHOLD <= 1.1

    def test_ess_threshold_positive(self):
        assert ESS_THRESHOLD > 0

    def test_party_colors_complete(self):
        assert "Republican" in PARTY_COLORS
        assert "Democrat" in PARTY_COLORS
        assert "Independent" in PARTY_COLORS

    def test_outlier_top_n_positive(self):
        assert OUTLIER_TOP_N > 0


# ── get_eligible_topics ──────────────────────────────────────────────────────


class TestGetEligibleTopics:
    """Test topic eligibility filtering."""

    def test_basic_eligibility(self):
        topics = _make_topics(
            [{"bill_number": f"HB {i}", "topic_id": 0, "topic_label": "T0"} for i in range(1, 15)]
            + [{"bill_number": f"HB {i}", "topic_id": 1, "topic_label": "T1"} for i in range(1, 4)]
        )
        rollcalls = _make_rollcalls(
            [{"vote_id": f"v{i}", "bill_number": f"HB {i}"} for i in range(1, 15)]
        )

        eligible, report = get_eligible_topics(topics, rollcalls, min_bills=10)
        assert eligible.height == 1
        assert eligible["topic_id"][0] == 0
        assert report.height == 2

    def test_no_eligible_topics(self):
        topics = _make_topics(
            [{"bill_number": f"HB {i}", "topic_id": 0, "topic_label": "T0"} for i in range(1, 4)]
        )
        rollcalls = _make_rollcalls(
            [{"vote_id": f"v{i}", "bill_number": f"HB {i}"} for i in range(1, 4)]
        )

        eligible, report = get_eligible_topics(topics, rollcalls, min_bills=10)
        assert eligible.height == 0

    def test_bills_without_rollcalls_not_counted(self):
        topics = _make_topics(
            [{"bill_number": f"HB {i}", "topic_id": 0, "topic_label": "T0"} for i in range(1, 20)]
        )
        # Only 5 bills have rollcalls
        rollcalls = _make_rollcalls(
            [{"vote_id": f"v{i}", "bill_number": f"HB {i}"} for i in range(1, 6)]
        )

        eligible, report = get_eligible_topics(topics, rollcalls, min_bills=10)
        assert eligible.height == 0
        assert report["n_rollcall_bills"][0] == 5


# ── subset_vote_matrix_for_topic ─────────────────────────────────────────────


class TestSubsetVoteMatrix:
    """Test vote matrix subsetting by topic."""

    def test_basic_subset(self):
        matrix = _make_vote_matrix(n_legislators=5, n_votes=6)
        topics = _make_topics(
            [
                {"bill_number": "HB 1", "topic_id": 0},
                {"bill_number": "HB 2", "topic_id": 0},
                {"bill_number": "HB 3", "topic_id": 1},
            ]
        )
        rollcalls = _make_rollcalls(
            [
                {"vote_id": "v1", "bill_number": "HB 1"},
                {"vote_id": "v2", "bill_number": "HB 2"},
                {"vote_id": "v3", "bill_number": "HB 3"},
                {"vote_id": "v4", "bill_number": "HB 4"},
            ]
        )

        subset = subset_vote_matrix_for_topic(matrix, topics, rollcalls, topic_id=0)
        vote_cols = [c for c in subset.columns if c != "legislator_slug"]
        assert set(vote_cols) == {"v1", "v2"}

    def test_no_matching_votes(self):
        matrix = _make_vote_matrix(n_legislators=5, n_votes=3)
        topics = _make_topics([{"bill_number": "HB 99", "topic_id": 0}])
        rollcalls = _make_rollcalls([{"vote_id": "v99", "bill_number": "HB 99"}])

        subset = subset_vote_matrix_for_topic(matrix, topics, rollcalls, topic_id=0)
        vote_cols = [c for c in subset.columns if c != "legislator_slug"]
        assert len(vote_cols) == 0

    def test_preserves_all_legislators(self):
        matrix = _make_vote_matrix(n_legislators=8, n_votes=5)
        topics = _make_topics([{"bill_number": "HB 1", "topic_id": 0}])
        rollcalls = _make_rollcalls([{"vote_id": "v1", "bill_number": "HB 1"}])

        subset = subset_vote_matrix_for_topic(matrix, topics, rollcalls, topic_id=0)
        assert subset.height == 8


# ── filter_legislators_in_topic ──────────────────────────────────────────────


class TestFilterLegislatorsInTopic:
    """Test legislator filtering by vote count."""

    def test_filters_low_vote_legislators(self):
        # Create matrix where first legislator has only 1 non-null vote
        data = {
            "legislator_slug": ["rep_a_1", "rep_b_1", "rep_c_1"],
            "v1": [1, 1, 0],
            "v2": [None, 0, 1],
            "v3": [None, 1, 0],
            "v4": [None, 0, 1],
            "v5": [None, 1, 1],
        }
        matrix = pl.DataFrame(data)

        filtered = filter_legislators_in_topic(matrix, min_votes=3)
        # rep_a_1 has 1 non-null, rep_b_1 and rep_c_1 have 5
        assert filtered.height == 2
        assert "rep_a_1" not in filtered["legislator_slug"].to_list()

    def test_empty_matrix(self):
        matrix = pl.DataFrame({"legislator_slug": []})
        filtered = filter_legislators_in_topic(matrix, min_votes=5)
        assert filtered.height == 0

    def test_keeps_all_when_sufficient(self):
        matrix = _make_vote_matrix(n_legislators=5, n_votes=10)
        # Most legislators should have >= 5 non-null votes with 10 columns
        filtered = filter_legislators_in_topic(matrix, min_votes=5)
        assert filtered.height >= 3


# ── compute_topic_pca_scores ─────────────────────────────────────────────────


class TestComputeTopicPcaScores:
    """Test per-topic PCA for fallback anchor selection."""

    def test_basic_pca(self):
        matrix = _make_vote_matrix(n_legislators=10, n_votes=8)
        result = compute_topic_pca_scores(matrix)
        assert result is not None
        assert "legislator_slug" in result.columns
        assert "PC1" in result.columns
        assert result.height == 10

    def test_too_few_legislators(self):
        matrix = _make_vote_matrix(n_legislators=2, n_votes=5)
        result = compute_topic_pca_scores(matrix)
        # 2 legislators is minimum for PCA but may still work
        assert result is None or result.height <= 2

    def test_too_few_vote_columns(self):
        data = {"legislator_slug": ["a", "b", "c"], "v1": [1, 0, 1]}
        matrix = pl.DataFrame(data)
        result = compute_topic_pca_scores(matrix)
        assert result is None


# ── align_topic_ideal_points ─────────────────────────────────────────────────


class TestAlignTopicIdealPoints:
    """Test sign alignment of per-topic ideal points."""

    def test_no_flip_when_positive_correlation(self):
        topic_xi = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c", "d"],
                "xi_mean": [1.0, 0.5, -0.5, -1.0],
            }
        )
        full_irt = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c", "d"],
                "xi_mean": [1.2, 0.6, -0.4, -0.9],
            }
        )

        aligned = align_topic_ideal_points(topic_xi, full_irt)
        # Should not flip — already positively correlated
        assert aligned["xi_mean"][0] > 0

    def test_flips_when_negative_correlation(self):
        topic_xi = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c", "d"],
                "xi_mean": [-1.0, -0.5, 0.5, 1.0],
            }
        )
        full_irt = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c", "d"],
                "xi_mean": [1.0, 0.5, -0.5, -1.0],
            }
        )

        aligned = align_topic_ideal_points(topic_xi, full_irt)
        # Should flip — negatively correlated
        assert aligned["xi_mean"][0] > 0

    def test_flips_hdi_columns_too(self):
        topic_xi = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c", "d"],
                "xi_mean": [-1.0, -0.5, 0.5, 1.0],
                "xi_sd": [0.1, 0.1, 0.1, 0.1],
                "xi_hdi_2.5": [-1.2, -0.7, 0.3, 0.8],
                "xi_hdi_97.5": [-0.8, -0.3, 0.7, 1.2],
            }
        )
        full_irt = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c", "d"],
                "xi_mean": [1.0, 0.5, -0.5, -1.0],
            }
        )

        aligned = align_topic_ideal_points(topic_xi, full_irt)
        assert aligned["xi_hdi_2.5"][0] > 0  # was negative, now positive

    def test_too_few_matches(self):
        topic_xi = pl.DataFrame({"legislator_slug": ["a", "b"], "xi_mean": [-1.0, 1.0]})
        full_irt = pl.DataFrame({"legislator_slug": ["a", "x"], "xi_mean": [1.0, -1.0]})

        aligned = align_topic_ideal_points(topic_xi, full_irt)
        # Only 1 match — should return unchanged
        assert aligned["xi_mean"][0] == topic_xi["xi_mean"][0]


# ── build_cross_topic_matrix ─────────────────────────────────────────────────


class TestBuildCrossTopicMatrix:
    """Test cross-topic matrix assembly."""

    def test_basic_assembly(self):
        results = {
            0: {
                "label": "education",
                "ideal_points": pl.DataFrame(
                    {"legislator_slug": ["a", "b", "c"], "xi_mean": [1.0, 0.5, -1.0]}
                ),
            },
            1: {
                "label": "taxes",
                "ideal_points": pl.DataFrame(
                    {"legislator_slug": ["a", "b", "d"], "xi_mean": [0.8, -0.3, 0.2]}
                ),
            },
        }

        matrix = build_cross_topic_matrix(results)
        assert "legislator_slug" in matrix.columns
        assert matrix.height == 4  # a, b, c, d
        topic_cols = [c for c in matrix.columns if c != "legislator_slug"]
        assert len(topic_cols) == 2

    def test_empty_results(self):
        matrix = build_cross_topic_matrix({})
        assert "legislator_slug" in matrix.columns
        assert matrix.height == 0

    def test_null_for_missing_legislators(self):
        results = {
            0: {
                "label": "topic_a",
                "ideal_points": pl.DataFrame({"legislator_slug": ["a"], "xi_mean": [1.0]}),
            },
            1: {
                "label": "topic_b",
                "ideal_points": pl.DataFrame({"legislator_slug": ["b"], "xi_mean": [-1.0]}),
            },
        }

        matrix = build_cross_topic_matrix(results)
        # legislator "a" should have null for topic_b
        a_row = matrix.filter(pl.col("legislator_slug") == "a")
        topic_cols = [c for c in matrix.columns if c != "legislator_slug"]
        # One topic column should be null for legislator "a"
        null_count = sum(a_row[c][0] is None for c in topic_cols)
        assert null_count == 1


# ── compute_cross_topic_correlations ─────────────────────────────────────────


class TestComputeCrossTopicCorrelations:
    """Test pairwise cross-topic correlations."""

    def test_basic_correlations(self):
        matrix = pl.DataFrame(
            {
                "legislator_slug": [f"l{i}" for i in range(10)],
                "topic_a": list(range(10)),
                "topic_b": list(range(10)),
            }
        )
        corr = compute_cross_topic_correlations(matrix)
        assert corr.height == 1
        assert corr["pearson_r"][0] == pytest.approx(1.0, abs=0.001)

    def test_negative_correlation(self):
        matrix = pl.DataFrame(
            {
                "legislator_slug": [f"l{i}" for i in range(10)],
                "topic_a": list(range(10)),
                "topic_b": list(range(9, -1, -1)),
            }
        )
        corr = compute_cross_topic_correlations(matrix)
        assert corr["pearson_r"][0] == pytest.approx(-1.0, abs=0.001)

    def test_single_topic(self):
        matrix = pl.DataFrame(
            {
                "legislator_slug": ["a", "b"],
                "topic_a": [1.0, 2.0],
            }
        )
        corr = compute_cross_topic_correlations(matrix)
        assert corr.height == 0

    def test_handles_nans(self):
        matrix = pl.DataFrame(
            {
                "legislator_slug": [f"l{i}" for i in range(5)],
                "topic_a": [1.0, 2.0, None, 4.0, 5.0],
                "topic_b": [1.0, 2.0, 3.0, None, 5.0],
            }
        )
        corr = compute_cross_topic_correlations(matrix)
        assert corr.height == 1
        assert corr["n"][0] == 3  # only 3 valid pairs


# ── identify_topic_outliers ──────────────────────────────────────────────────


class TestIdentifyTopicOutliers:
    """Test outlier detection."""

    def test_basic_outlier_detection(self):
        # Most legislators are aligned, one deviates strongly
        n = 20
        topic_xi = pl.DataFrame(
            {
                "legislator_slug": [f"l{i}" for i in range(n)],
                "xi_mean": [float(i) / n for i in range(n)],
            }
        )
        # Full IRT is mostly the same but one legislator is flipped
        full_vals = [float(i) / n for i in range(n)]
        full_vals[0] = 1.0  # was 0.0, now 1.0 — big deviation
        full_irt = pl.DataFrame(
            {
                "legislator_slug": [f"l{i}" for i in range(n)],
                "xi_mean": full_vals,
                "full_name": [f"Person {i}" for i in range(n)],
                "party": ["Republican"] * n,
            }
        )

        outliers = identify_topic_outliers(topic_xi, full_irt, top_n=5)
        assert outliers.height == 5
        assert "discrepancy_z" in outliers.columns
        # The first row should have the largest discrepancy
        assert outliers["discrepancy_z"][0] >= outliers["discrepancy_z"][1]

    def test_too_few_legislators(self):
        topic_xi = pl.DataFrame({"legislator_slug": ["a", "b"], "xi_mean": [1.0, -1.0]})
        full_irt = pl.DataFrame(
            {
                "legislator_slug": ["a", "b"],
                "xi_mean": [1.0, -1.0],
                "full_name": ["A", "B"],
                "party": ["R", "D"],
            }
        )
        outliers = identify_topic_outliers(topic_xi, full_irt, top_n=5)
        assert outliers.height == 0


# ── check_anchor_stability ───────────────────────────────────────────────────


class TestCheckAnchorStability:
    """Test anchor stability checking."""

    def test_stable_anchors(self):
        n = 20
        full_irt = _make_full_irt(n)

        # Create results where anchors stay at extremes
        results = {
            0: {
                "label": "topic_0",
                "ideal_points": pl.DataFrame(
                    {
                        "legislator_slug": full_irt["legislator_slug"].to_list(),
                        "xi_mean": full_irt["xi_mean"].to_list(),
                    }
                ),
            },
        }

        cons_slug = full_irt.sort("xi_mean", descending=True)["legislator_slug"][0]
        lib_slug = full_irt.sort("xi_mean")["legislator_slug"][0]

        stability = check_anchor_stability(results, full_irt, (cons_slug, lib_slug))
        assert stability.height == 1
        assert stability["cons_rank"][0] == 1  # most conservative stays #1
        assert stability["lib_rank"][0] == n  # most liberal stays last

    def test_missing_anchor_in_topic(self):
        results = {
            0: {
                "label": "topic_0",
                "ideal_points": pl.DataFrame(
                    {
                        "legislator_slug": ["a", "b", "c"],
                        "xi_mean": [1.0, 0.0, -1.0],
                    }
                ),
            },
        }
        full_irt = _make_full_irt(10)

        stability = check_anchor_stability(results, full_irt, ("missing_cons", "missing_lib"))
        assert stability.height == 1
        assert stability["cons_rank"][0] is None


# ── compute_topic_irt_correlation ────────────────────────────────────────────


class TestComputeTopicIrtCorrelation:
    """Test per-topic correlation computation."""

    def test_perfect_correlation(self):
        n = 20
        topic_xi = pl.DataFrame(
            {"legislator_slug": [f"l{i}" for i in range(n)], "xi_mean": list(range(n))}
        )
        full_irt = pl.DataFrame(
            {"legislator_slug": [f"l{i}" for i in range(n)], "xi_mean": list(range(n))}
        )

        corr = compute_topic_irt_correlation(topic_xi, full_irt)
        assert corr["pearson_r"] == pytest.approx(1.0, abs=0.001)
        assert corr["quality"] == "strong"

    def test_insufficient_data(self):
        topic_xi = pl.DataFrame({"legislator_slug": ["a", "b"], "xi_mean": [1.0, -1.0]})
        full_irt = pl.DataFrame({"legislator_slug": ["a", "b"], "xi_mean": [1.0, -1.0]})

        corr = compute_topic_irt_correlation(topic_xi, full_irt)
        assert corr["quality"] == "insufficient_data"

    def test_moderate_correlation(self):
        rng = np.random.default_rng(42)
        n = 30
        base = np.linspace(-2, 2, n)
        topic_xi = pl.DataFrame(
            {
                "legislator_slug": [f"l{i}" for i in range(n)],
                "xi_mean": (base + rng.normal(0, 1.0, n)).tolist(),
            }
        )
        full_irt = pl.DataFrame(
            {
                "legislator_slug": [f"l{i}" for i in range(n)],
                "xi_mean": base.tolist(),
            }
        )

        corr = compute_topic_irt_correlation(topic_xi, full_irt)
        assert 0.3 < corr["pearson_r"] < 1.0
        assert corr["n"] == n
        assert not math.isnan(corr["ci_lower"])
        assert not math.isnan(corr["ci_upper"])

    def test_fisher_z_ci_bounds(self):
        # Use near-perfect but not identical data so r < 1.0 and Fisher z CI is computable
        vals = list(range(20))
        noisy = [v + 0.01 * ((-1) ** v) for v in vals]
        corr = compute_topic_irt_correlation(
            pl.DataFrame({"legislator_slug": [f"l{i}" for i in range(20)], "xi_mean": vals}),
            pl.DataFrame({"legislator_slug": [f"l{i}" for i in range(20)], "xi_mean": noisy}),
        )
        assert corr["ci_lower"] <= corr["pearson_r"] <= corr["ci_upper"]


# ── Quality Label ────────────────────────────────────────────────────────────


class TestQualityLabel:
    """Test quality label assignment."""

    def test_strong(self):
        assert _quality_label(0.85) == "strong"

    def test_good(self):
        assert _quality_label(0.65) == "good"

    def test_moderate(self):
        assert _quality_label(0.45) == "moderate"

    def test_weak(self):
        assert _quality_label(0.20) == "weak"

    def test_boundary_strong(self):
        assert _quality_label(STRONG_CORRELATION) == "strong"

    def test_boundary_good(self):
        assert _quality_label(GOOD_CORRELATION) == "good"

    def test_boundary_moderate(self):
        assert _quality_label(MODERATE_CORRELATION) == "moderate"


# ── Fisher Z CI ──────────────────────────────────────────────────────────────


class TestFisherZCI:
    """Test Fisher z confidence interval."""

    def test_basic_ci(self):
        low, high = _fisher_z_ci(0.5, 50)
        assert low < 0.5 < high

    def test_small_sample(self):
        low, high = _fisher_z_ci(0.5, 3)
        assert math.isnan(low) and math.isnan(high)

    def test_perfect_r(self):
        low, high = _fisher_z_ci(1.0, 50)
        assert math.isnan(low) and math.isnan(high)


# ── Sanitize Label ───────────────────────────────────────────────────────────


class TestSanitizeLabel:
    """Test label sanitization for column names."""

    def test_basic(self):
        assert _sanitize_label("Topic 0: education, school") == "topic_0_education_school"

    def test_truncation(self):
        long_label = "A" * 100
        result = _sanitize_label(long_label)
        assert len(result) <= 35  # 30 chars + underscores

    def test_empty(self):
        assert _sanitize_label("") == "unknown"

    def test_special_chars(self):
        result = _sanitize_label("Topic #3: K-12 & Higher Ed!")
        assert "#" not in result
        assert "&" not in result
