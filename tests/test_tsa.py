"""Tests for Phase 15: Time Series Analysis.

Covers rolling-window PCA, sign alignment, party trajectories, early/late comparison,
top movers, Rice timeseries, weekly aggregation, PELT changepoint detection,
joint detection, and penalty sensitivity.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.tsa import (
    WINDOW_SIZE,
    aggregate_weekly,
    align_pc_signs,
    build_rice_timeseries,
    build_vote_matrix,
    compute_early_vs_late,
    compute_party_trajectories,
    cross_reference_veto_overrides,
    detect_changepoints_joint,
    detect_changepoints_pelt,
    find_top_movers,
    rolling_window_pca,
    run_penalty_sensitivity,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_legislators(n_rep: int = 60, n_dem: int = 25, chamber: str = "House") -> pl.DataFrame:
    """Create a synthetic legislators DataFrame."""
    prefix = "rep_" if chamber == "House" else "sen_"
    rows = []
    for i in range(n_rep):
        rows.append(
            {
                "legislator_slug": f"{prefix}r{i:03d}",
                "full_name": f"Rep{i}",
                "party": "Republican",
                "district": str(i + 1),
            }
        )
    for i in range(n_dem):
        rows.append(
            {
                "legislator_slug": f"{prefix}d{i:03d}",
                "full_name": f"Dem{i}",
                "party": "Democrat",
                "district": str(n_rep + i + 1),
            }
        )
    return pl.DataFrame(rows)


def _make_votes(
    n_rollcalls: int = 200,
    n_rep: int = 60,
    n_dem: int = 25,
    chamber: str = "House",
    party_line_rate: float = 0.85,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create synthetic votes and rollcalls DataFrames.

    Generates party-line voting with some noise.
    """
    prefix = "rep_" if chamber == "House" else "sen_"
    rng = np.random.default_rng(42)

    vote_rows = []
    rc_rows = []

    for v in range(n_rollcalls):
        vote_id = f"je_2025{v:06d}"
        # Random date spread across a session
        month = 1 + v * 5 // n_rollcalls
        day = 1 + (v * 28 // n_rollcalls) % 28
        vote_date = f"2025-{month:02d}-{day:02d}"
        vote_datetime = f"{vote_date}T12:00:00"
        bill_number = f"HB{v + 1}"
        motion = "Final Action"
        passed = "True"

        rc_rows.append(
            {
                "vote_id": vote_id,
                "bill_number": bill_number,
                "motion": motion,
                "passed": passed,
                "vote_date": vote_date,
                "vote_datetime": vote_datetime,
            }
        )

        # Republicans mostly vote Yea, Democrats mostly vote Nay
        for i in range(n_rep):
            slug = f"{prefix}r{i:03d}"
            cat = "Yea" if rng.random() < party_line_rate else "Nay"
            vote_rows.append(
                {
                    "legislator_slug": slug,
                    "vote_id": vote_id,
                    "vote_category": cat,
                    "party": "Republican",
                }
            )

        for i in range(n_dem):
            slug = f"{prefix}d{i:03d}"
            cat = "Nay" if rng.random() < party_line_rate else "Yea"
            vote_rows.append(
                {
                    "legislator_slug": slug,
                    "vote_id": vote_id,
                    "vote_category": cat,
                    "party": "Democrat",
                }
            )

    votes = pl.DataFrame(vote_rows)
    rollcalls = pl.DataFrame(rc_rows)
    return votes, rollcalls


# ── TestRollingWindowPCA ─────────────────────────────────────────────────────


class TestRollingWindowPCA:
    """Tests for rolling_window_pca()."""

    def test_basic_shape(self):
        """Rolling PCA produces expected columns."""
        n_legs, n_votes = 30, 100
        rng = np.random.default_rng(42)
        matrix = rng.binomial(1, 0.5, (n_legs, n_votes)).astype(float)
        slugs = [f"rep_r{i:03d}" for i in range(n_legs)]
        vote_ids = [f"v{i}" for i in range(n_votes)]
        datetimes = [f"2025-01-{(i % 28) + 1:02d}" for i in range(n_votes)]

        result = rolling_window_pca(
            matrix,
            slugs,
            vote_ids,
            datetimes,
            window_size=30,
            step_size=10,
            min_votes=5,
            min_legislators=10,
        )

        assert result.height > 0
        assert set(result.columns) == {
            "slug",
            "window_start",
            "window_end",
            "window_midpoint",
            "window_idx",
            "pc1_score",
        }

    def test_window_count(self):
        """Number of windows matches expected formula."""
        n_legs, n_votes = 30, 100
        window_size, step_size = 30, 10
        rng = np.random.default_rng(42)
        matrix = rng.binomial(1, 0.5, (n_legs, n_votes)).astype(float)
        slugs = [f"rep_r{i:03d}" for i in range(n_legs)]
        vote_ids = [f"v{i}" for i in range(n_votes)]
        datetimes = ["2025-01-01" for _ in range(n_votes)]

        result = rolling_window_pca(
            matrix,
            slugs,
            vote_ids,
            datetimes,
            window_size=window_size,
            step_size=step_size,
            min_votes=5,
            min_legislators=10,
        )

        expected_windows = (n_votes - window_size) // step_size + 1
        actual_windows = result["window_idx"].n_unique()
        assert actual_windows == expected_windows

    def test_too_few_legislators(self):
        """Returns empty DataFrame if below min_legislators."""
        n_legs, n_votes = 5, 100
        rng = np.random.default_rng(42)
        matrix = rng.binomial(1, 0.5, (n_legs, n_votes)).astype(float)
        slugs = [f"rep_r{i:03d}" for i in range(n_legs)]
        vote_ids = [f"v{i}" for i in range(n_votes)]
        datetimes = ["2025-01-01"] * n_votes

        result = rolling_window_pca(
            matrix,
            slugs,
            vote_ids,
            datetimes,
            window_size=30,
            step_size=10,
            min_votes=5,
            min_legislators=20,
        )

        assert result.height == 0

    def test_too_short_for_window(self):
        """Returns empty DataFrame if fewer votes than window_size."""
        n_legs, n_votes = 30, 20
        rng = np.random.default_rng(42)
        matrix = rng.binomial(1, 0.5, (n_legs, n_votes)).astype(float)
        slugs = [f"rep_r{i:03d}" for i in range(n_legs)]
        vote_ids = [f"v{i}" for i in range(n_votes)]
        datetimes = ["2025-01-01"] * n_votes

        result = rolling_window_pca(
            matrix,
            slugs,
            vote_ids,
            datetimes,
            window_size=WINDOW_SIZE,
            step_size=15,
        )

        assert result.height == 0

    def test_scores_bounded(self):
        """PC1 scores should not be extreme."""
        n_legs, n_votes = 40, 150
        rng = np.random.default_rng(42)
        matrix = rng.binomial(1, 0.5, (n_legs, n_votes)).astype(float)
        slugs = [f"rep_r{i:03d}" for i in range(n_legs)]
        vote_ids = [f"v{i}" for i in range(n_votes)]
        datetimes = [f"2025-01-{(i % 28) + 1:02d}" for i in range(n_votes)]

        result = rolling_window_pca(
            matrix,
            slugs,
            vote_ids,
            datetimes,
            window_size=50,
            step_size=10,
            min_votes=5,
            min_legislators=10,
        )

        scores = result["pc1_score"].to_numpy()
        assert np.all(np.isfinite(scores))

    def test_nan_handling(self):
        """PCA handles NaN (absent) votes via imputation."""
        n_legs, n_votes = 30, 100
        rng = np.random.default_rng(42)
        matrix = rng.binomial(1, 0.5, (n_legs, n_votes)).astype(float)
        # Sprinkle NaNs
        nan_mask = rng.random((n_legs, n_votes)) < 0.1
        matrix[nan_mask] = np.nan
        slugs = [f"rep_r{i:03d}" for i in range(n_legs)]
        vote_ids = [f"v{i}" for i in range(n_votes)]
        datetimes = ["2025-01-01"] * n_votes

        result = rolling_window_pca(
            matrix,
            slugs,
            vote_ids,
            datetimes,
            window_size=30,
            step_size=10,
            min_votes=5,
            min_legislators=10,
        )

        assert result.height > 0
        assert result["pc1_score"].null_count() == 0

    def test_constant_votes_skipped(self):
        """Windows with all-constant columns should be skipped."""
        n_legs, n_votes = 30, 100
        matrix = np.ones((n_legs, n_votes))  # Everyone votes the same
        slugs = [f"rep_r{i:03d}" for i in range(n_legs)]
        vote_ids = [f"v{i}" for i in range(n_votes)]
        datetimes = ["2025-01-01"] * n_votes

        result = rolling_window_pca(
            matrix,
            slugs,
            vote_ids,
            datetimes,
            window_size=30,
            step_size=10,
            min_votes=5,
            min_legislators=10,
        )

        assert result.height == 0

    def test_pc1_per_legislator_per_window(self):
        """Each legislator gets exactly one PC1 score per window."""
        n_legs, n_votes = 30, 100
        rng = np.random.default_rng(42)
        matrix = rng.binomial(1, 0.5, (n_legs, n_votes)).astype(float)
        slugs = [f"rep_r{i:03d}" for i in range(n_legs)]
        vote_ids = [f"v{i}" for i in range(n_votes)]
        datetimes = ["2025-01-01"] * n_votes

        result = rolling_window_pca(
            matrix,
            slugs,
            vote_ids,
            datetimes,
            window_size=30,
            step_size=10,
            min_votes=5,
            min_legislators=10,
        )

        counts = result.group_by("window_idx", "slug").agg(pl.len().alias("n"))
        assert counts["n"].max() == 1


# ── TestSignAlignment ────────────────────────────────────────────────────────


class TestSignAlignment:
    """Tests for align_pc_signs()."""

    def test_republicans_positive(self):
        """After alignment, Republican mean should be positive."""
        df = pl.DataFrame(
            {
                "slug": ["rep_r001", "rep_r002", "rep_d001", "rep_d002"] * 2,
                "window_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                "pc1_score": [-1.0, -0.8, 0.5, 0.6, -0.9, -0.7, 0.4, 0.5],
                "window_start": [""] * 8,
                "window_end": [""] * 8,
                "window_midpoint": [""] * 8,
            }
        )
        meta = pl.DataFrame(
            {
                "legislator_slug": ["rep_r001", "rep_r002", "rep_d001", "rep_d002"],
                "party": ["Republican", "Republican", "Democrat", "Democrat"],
            }
        )

        result = align_pc_signs(df, meta)

        for widx in [0, 1]:
            window = result.filter(pl.col("window_idx") == widx)
            rep_scores = window.filter(pl.col("slug").is_in(["rep_r001", "rep_r002"]))["pc1_score"]
            assert rep_scores.mean() > 0

    def test_already_aligned(self):
        """If Republicans are already positive, no flip."""
        df = pl.DataFrame(
            {
                "slug": ["rep_r001", "rep_d001"],
                "window_idx": [0, 0],
                "pc1_score": [1.0, -0.5],
                "window_start": ["", ""],
                "window_end": ["", ""],
                "window_midpoint": ["", ""],
            }
        )
        meta = pl.DataFrame(
            {
                "legislator_slug": ["rep_r001", "rep_d001"],
                "party": ["Republican", "Democrat"],
            }
        )

        result = align_pc_signs(df, meta)
        rep_score = result.filter(pl.col("slug") == "rep_r001")["pc1_score"][0]
        assert rep_score == 1.0

    def test_empty_input(self):
        """Empty DataFrame returns empty."""
        df = pl.DataFrame(
            schema={
                "slug": pl.Utf8,
                "window_idx": pl.Int64,
                "pc1_score": pl.Float64,
                "window_start": pl.Utf8,
                "window_end": pl.Utf8,
                "window_midpoint": pl.Utf8,
            }
        )
        meta = pl.DataFrame(
            schema={
                "legislator_slug": pl.Utf8,
                "party": pl.Utf8,
            }
        )
        result = align_pc_signs(df, meta)
        assert result.height == 0

    def test_single_party_no_crash(self):
        """Works with only one party present."""
        df = pl.DataFrame(
            {
                "slug": ["rep_r001", "rep_r002"],
                "window_idx": [0, 0],
                "pc1_score": [-1.0, -0.5],
                "window_start": [""] * 2,
                "window_end": [""] * 2,
                "window_midpoint": [""] * 2,
            }
        )
        meta = pl.DataFrame(
            {
                "legislator_slug": ["rep_r001", "rep_r002"],
                "party": ["Republican", "Republican"],
            }
        )

        result = align_pc_signs(df, meta)
        assert result.height == 2
        # Republicans should be positive
        assert result["pc1_score"].mean() > 0

    def test_flips_all_scores(self):
        """When flipping, all scores in the window are flipped (not just Republicans)."""
        df = pl.DataFrame(
            {
                "slug": ["rep_r001", "rep_d001"],
                "window_idx": [0, 0],
                "pc1_score": [-2.0, 1.0],
                "window_start": [""] * 2,
                "window_end": [""] * 2,
                "window_midpoint": [""] * 2,
            }
        )
        meta = pl.DataFrame(
            {
                "legislator_slug": ["rep_r001", "rep_d001"],
                "party": ["Republican", "Democrat"],
            }
        )

        result = align_pc_signs(df, meta)
        rep = result.filter(pl.col("slug") == "rep_r001")["pc1_score"][0]
        dem = result.filter(pl.col("slug") == "rep_d001")["pc1_score"][0]
        assert rep == 2.0  # flipped
        assert dem == -1.0  # also flipped


# ── TestPartyTrajectories ────────────────────────────────────────────────────


class TestPartyTrajectories:
    """Tests for compute_party_trajectories()."""

    def test_correct_columns(self):
        """Output has expected columns."""
        df = pl.DataFrame(
            {
                "slug": ["rep_r001", "rep_d001"] * 3,
                "window_idx": [0, 0, 1, 1, 2, 2],
                "pc1_score": [1.0, -0.5, 1.2, -0.6, 0.8, -0.4],
                "window_midpoint": ["2025-01-15"] * 6,
                "window_start": [""] * 6,
                "window_end": [""] * 6,
            }
        )
        meta = pl.DataFrame(
            {
                "legislator_slug": ["rep_r001", "rep_d001"],
                "party": ["Republican", "Democrat"],
            }
        )

        result = compute_party_trajectories(df, meta)
        assert "window_idx" in result.columns
        assert "party" in result.columns
        assert "mean_pc1" in result.columns
        assert "polarization_gap" in result.columns

    def test_gap_computed(self):
        """Polarization gap = |Rep mean - Dem mean|."""
        df = pl.DataFrame(
            {
                "slug": ["rep_r001", "rep_d001"],
                "window_idx": [0, 0],
                "pc1_score": [2.0, -1.0],
                "window_midpoint": ["2025-01-15"] * 2,
                "window_start": [""] * 2,
                "window_end": [""] * 2,
            }
        )
        meta = pl.DataFrame(
            {
                "legislator_slug": ["rep_r001", "rep_d001"],
                "party": ["Republican", "Democrat"],
            }
        )

        result = compute_party_trajectories(df, meta)
        gap = result["polarization_gap"].to_list()
        assert all(g == pytest.approx(3.0) for g in gap if g is not None)

    def test_empty_input(self):
        """Empty input returns empty with correct schema."""
        df = pl.DataFrame(
            schema={
                "slug": pl.Utf8,
                "window_idx": pl.Int64,
                "pc1_score": pl.Float64,
                "window_midpoint": pl.Utf8,
                "window_start": pl.Utf8,
                "window_end": pl.Utf8,
            }
        )
        meta = pl.DataFrame(
            schema={
                "legislator_slug": pl.Utf8,
                "party": pl.Utf8,
            }
        )
        result = compute_party_trajectories(df, meta)
        assert result.height == 0

    def test_independents_excluded(self):
        """Independent legislators should not appear in party trajectories."""
        df = pl.DataFrame(
            {
                "slug": ["rep_r001", "rep_d001", "rep_i001"],
                "window_idx": [0, 0, 0],
                "pc1_score": [1.0, -0.5, 0.0],
                "window_midpoint": ["2025-01-15"] * 3,
                "window_start": [""] * 3,
                "window_end": [""] * 3,
            }
        )
        meta = pl.DataFrame(
            {
                "legislator_slug": ["rep_r001", "rep_d001", "rep_i001"],
                "party": ["Republican", "Democrat", "Independent"],
            }
        )

        result = compute_party_trajectories(df, meta)
        parties = result["party"].unique().to_list()
        assert "Independent" not in parties


# ── TestEarlyVsLate ──────────────────────────────────────────────────────────


class TestEarlyVsLate:
    """Tests for compute_early_vs_late()."""

    def test_correct_columns(self):
        """Output has expected columns."""
        n_legs, n_votes = 30, 100
        rng = np.random.default_rng(42)
        matrix = rng.binomial(1, 0.5, (n_legs, n_votes)).astype(float)
        slugs = [f"rep_r{i:03d}" for i in range(n_legs)]
        vote_ids = [f"v{i}" for i in range(n_votes)]
        meta = _make_legislators(n_rep=30, n_dem=0)

        result = compute_early_vs_late(matrix, slugs, vote_ids, meta)
        assert "slug" in result.columns
        assert "early_pc1" in result.columns
        assert "late_pc1" in result.columns
        assert "drift" in result.columns

    def test_drift_is_difference(self):
        """Drift = late_pc1 - early_pc1."""
        n_legs, n_votes = 30, 100
        rng = np.random.default_rng(42)
        matrix = rng.binomial(1, 0.5, (n_legs, n_votes)).astype(float)
        slugs = [f"rep_r{i:03d}" for i in range(n_legs)]
        vote_ids = [f"v{i}" for i in range(n_votes)]
        meta = _make_legislators(n_rep=30, n_dem=0)

        result = compute_early_vs_late(matrix, slugs, vote_ids, meta)
        if result.height > 0:
            computed = (result["late_pc1"] - result["early_pc1"]).to_numpy()
            actual = result["drift"].to_numpy()
            np.testing.assert_allclose(computed, actual, atol=1e-10)

    def test_identical_halves_zero_drift(self):
        """If first and second halves are identical, drift should be near zero."""
        n_legs, n_votes = 30, 100
        rng = np.random.default_rng(42)
        half = rng.binomial(1, 0.5, (n_legs, n_votes // 2)).astype(float)
        matrix = np.hstack([half, half])
        slugs = [f"rep_r{i:03d}" for i in range(n_legs)]
        vote_ids = [f"v{i}" for i in range(n_votes)]
        meta = _make_legislators(n_rep=30, n_dem=0)

        result = compute_early_vs_late(matrix, slugs, vote_ids, meta)
        if result.height > 0:
            max_drift = result["drift"].abs().max()
            # With identical data, drift should be very small
            assert max_drift < 1.0

    def test_too_few_legislators(self):
        """Returns empty if insufficient legislators."""
        n_legs, n_votes = 5, 100
        matrix = np.ones((n_legs, n_votes))
        slugs = [f"rep_r{i:03d}" for i in range(n_legs)]
        vote_ids = [f"v{i}" for i in range(n_votes)]
        meta = _make_legislators(n_rep=5, n_dem=0)

        result = compute_early_vs_late(matrix, slugs, vote_ids, meta)
        assert result.height == 0

    def test_sign_alignment(self):
        """Republicans should be positive in both halves."""
        n_rep, n_dem = 40, 20
        n_votes = 200
        rng = np.random.default_rng(42)
        # Strong party line: Rs vote 1, Ds vote 0
        matrix = np.zeros((n_rep + n_dem, n_votes))
        matrix[:n_rep, :] = rng.binomial(1, 0.9, (n_rep, n_votes))
        matrix[n_rep:, :] = rng.binomial(1, 0.1, (n_dem, n_votes))

        slugs = [f"rep_r{i:03d}" for i in range(n_rep)] + [f"rep_d{i:03d}" for i in range(n_dem)]
        vote_ids = [f"v{i}" for i in range(n_votes)]
        meta = _make_legislators(n_rep=n_rep, n_dem=n_dem)

        result = compute_early_vs_late(matrix, slugs, vote_ids, meta)
        if result.height > 0:
            rep_early = result.filter(pl.col("party") == "Republican")["early_pc1"].mean()
            assert rep_early > 0


# ── TestFindTopMovers ────────────────────────────────────────────────────────


class TestFindTopMovers:
    """Tests for find_top_movers()."""

    def test_count(self):
        """Returns at most N movers."""
        df = pl.DataFrame(
            {
                "slug": [f"rep_r{i:03d}" for i in range(20)],
                "full_name": [f"Rep{i}" for i in range(20)],
                "party": ["Republican"] * 20,
                "early_pc1": [float(i) for i in range(20)],
                "late_pc1": [float(20 - i) for i in range(20)],
                "drift": [float(20 - 2 * i) for i in range(20)],
            }
        )
        result = find_top_movers(df, 5)
        assert result.height == 5

    def test_ordering(self):
        """Top movers are sorted by |drift| descending."""
        df = pl.DataFrame(
            {
                "slug": ["a", "b", "c"],
                "full_name": ["A", "B", "C"],
                "party": ["Republican"] * 3,
                "early_pc1": [0.0, 0.0, 0.0],
                "late_pc1": [0.1, 0.5, -0.3],
                "drift": [0.1, 0.5, -0.3],
            }
        )
        result = find_top_movers(df, 3)
        drifts = result["drift"].abs().to_list()
        assert drifts == sorted(drifts, reverse=True)

    def test_fewer_than_n(self):
        """If fewer legislators than N, returns all."""
        df = pl.DataFrame(
            {
                "slug": ["a", "b"],
                "full_name": ["A", "B"],
                "party": ["Republican"] * 2,
                "early_pc1": [0.0, 0.0],
                "late_pc1": [0.5, -0.3],
                "drift": [0.5, -0.3],
            }
        )
        result = find_top_movers(df, 10)
        assert result.height == 2


# ── TestBuildRiceTimeseries ──────────────────────────────────────────────────


class TestBuildRiceTimeseries:
    """Tests for build_rice_timeseries()."""

    def test_rice_formula(self):
        """Rice = |Yea - Nay| / (Yea + Nay)."""
        votes = pl.DataFrame(
            {
                "legislator_slug": ["rep_r001", "rep_r002", "rep_r003", "rep_r004"],
                "vote_id": ["v1"] * 4,
                "vote_category": ["Yea", "Yea", "Yea", "Nay"],
                "party": ["Republican"] * 4,
            }
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["v1"],
                "vote_date": ["2025-01-15"],
                "vote_datetime": ["2025-01-15T12:00:00"],
            }
        )
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_r001", "rep_r002", "rep_r003", "rep_r004"],
                "party": ["Republican"] * 4,
                "full_name": ["A", "B", "C", "D"],
            }
        )

        result = build_rice_timeseries(votes, rollcalls, legislators, "House")
        rice = result.filter(pl.col("party") == "Republican")["rice"][0]
        # 3 Yea, 1 Nay: |3-1|/(3+1) = 0.5
        assert rice == pytest.approx(0.5)

    def test_unanimous_rice_one(self):
        """Unanimous vote has Rice = 1.0."""
        votes = pl.DataFrame(
            {
                "legislator_slug": [f"rep_r{i:03d}" for i in range(5)],
                "vote_id": ["v1"] * 5,
                "vote_category": ["Yea"] * 5,
                "party": ["Republican"] * 5,
            }
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["v1"],
                "vote_date": ["2025-01-15"],
            }
        )
        legislators = pl.DataFrame(
            {
                "legislator_slug": [f"rep_r{i:03d}" for i in range(5)],
                "party": ["Republican"] * 5,
                "full_name": [f"R{i}" for i in range(5)],
            }
        )

        result = build_rice_timeseries(votes, rollcalls, legislators, "House")
        rice = result.filter(pl.col("party") == "Republican")["rice"][0]
        assert rice == pytest.approx(1.0)

    def test_perfect_split_rice_zero(self):
        """50-50 split has Rice = 0.0."""
        votes = pl.DataFrame(
            {
                "legislator_slug": [f"rep_r{i:03d}" for i in range(4)],
                "vote_id": ["v1"] * 4,
                "vote_category": ["Yea", "Yea", "Nay", "Nay"],
                "party": ["Republican"] * 4,
            }
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["v1"],
                "vote_date": ["2025-01-15"],
            }
        )
        legislators = pl.DataFrame(
            {
                "legislator_slug": [f"rep_r{i:03d}" for i in range(4)],
                "party": ["Republican"] * 4,
                "full_name": [f"R{i}" for i in range(4)],
            }
        )

        result = build_rice_timeseries(votes, rollcalls, legislators, "House")
        rice = result.filter(pl.col("party") == "Republican")["rice"][0]
        assert rice == pytest.approx(0.0)

    def test_chronological_ordering(self):
        """Output is sorted chronologically."""
        votes = pl.DataFrame(
            {
                "legislator_slug": ["rep_r001"] * 2,
                "vote_id": ["v2", "v1"],
                "vote_category": ["Yea", "Yea"],
                "party": ["Republican"] * 2,
            }
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["v1", "v2"],
                "vote_date": ["2025-01-10", "2025-01-20"],
                "vote_datetime": ["2025-01-10T12:00:00", "2025-01-20T12:00:00"],
            }
        )
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_r001"],
                "party": ["Republican"],
                "full_name": ["R1"],
            }
        )

        result = build_rice_timeseries(votes, rollcalls, legislators, "House")
        dates = result["vote_datetime"].to_list()
        assert dates == sorted(dates)

    def test_filters_to_chamber(self):
        """Only includes votes from the specified chamber."""
        votes = pl.DataFrame(
            {
                "legislator_slug": ["rep_r001", "sen_r001"],
                "vote_id": ["v1", "v2"],
                "vote_category": ["Yea", "Yea"],
                "party": ["Republican", "Republican"],
            }
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["v1", "v2"],
                "vote_date": ["2025-01-15", "2025-01-15"],
            }
        )
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_r001", "sen_r001"],
                "party": ["Republican", "Republican"],
                "full_name": ["HR1", "SR1"],
            }
        )

        house = build_rice_timeseries(votes, rollcalls, legislators, "House")
        senate = build_rice_timeseries(votes, rollcalls, legislators, "Senate")
        assert house.height > 0
        assert senate.height > 0
        # They should have different vote_ids
        h_ids = set(house["vote_id"].to_list())
        s_ids = set(senate["vote_id"].to_list())
        assert h_ids != s_ids


# ── TestWeeklyAggregation ────────────────────────────────────────────────────


class TestWeeklyAggregation:
    """Tests for aggregate_weekly()."""

    def test_reduction(self):
        """Weekly aggregation reduces row count."""
        # 30 daily observations -> should produce fewer weekly rows
        rice_ts = pl.DataFrame(
            {
                "vote_id": [f"v{i}" for i in range(30)],
                "party": ["Republican"] * 30,
                "rice": [0.8 + 0.01 * i for i in range(30)],
                "vote_datetime": [f"2025-01-{i + 1:02d}T12:00:00" for i in range(30)],
                "yea_count": [50] * 30,
                "nay_count": [10] * 30,
            }
        )

        result = aggregate_weekly(rice_ts)
        assert result.height < 30
        assert result.height > 0

    def test_mean_computation(self):
        """Weekly mean should be the average of daily values."""
        # 7 days, all same value -> weekly mean should be that value
        rice_ts = pl.DataFrame(
            {
                "vote_id": [f"v{i}" for i in range(7)],
                "party": ["Republican"] * 7,
                "rice": [0.75] * 7,
                "vote_datetime": [f"2025-01-{i + 1:02d}T12:00:00" for i in range(7)],
                "yea_count": [50] * 7,
                "nay_count": [10] * 7,
            }
        )

        result = aggregate_weekly(rice_ts)
        assert result["mean_rice"][0] == pytest.approx(0.75)

    def test_single_week(self):
        """Single day produces one weekly observation."""
        rice_ts = pl.DataFrame(
            {
                "vote_id": ["v1"],
                "party": ["Republican"],
                "rice": [0.9],
                "vote_datetime": ["2025-01-15T12:00:00"],
                "yea_count": [50],
                "nay_count": [5],
            }
        )

        result = aggregate_weekly(rice_ts)
        assert result.height == 1

    def test_empty_input(self):
        """Empty input returns empty."""
        rice_ts = pl.DataFrame(
            schema={
                "vote_id": pl.Utf8,
                "party": pl.Utf8,
                "rice": pl.Float64,
                "vote_datetime": pl.Utf8,
                "yea_count": pl.Int64,
                "nay_count": pl.Int64,
            }
        )
        result = aggregate_weekly(rice_ts)
        assert result.height == 0


# ── TestChangepoints ─────────────────────────────────────────────────────────


class TestChangepoints:
    """Tests for detect_changepoints_pelt() and related functions."""

    def test_step_function(self):
        """Detects a single changepoint in a step function."""
        signal = np.concatenate([np.ones(30) * 0.9, np.ones(30) * 0.3])
        cps = detect_changepoints_pelt(signal, penalty=5.0, min_size=3)
        # Should find one changepoint near index 30 (plus the terminal)
        assert len(cps) >= 2  # at least one CP + terminal
        interior_cps = [c for c in cps if c < len(signal)]
        assert len(interior_cps) >= 1
        assert any(25 <= c <= 35 for c in interior_cps)

    def test_constant_signal(self):
        """Constant signal should have no interior changepoints."""
        signal = np.ones(50) * 0.8
        cps = detect_changepoints_pelt(signal, penalty=10.0, min_size=3)
        # Only terminal point
        assert cps == [len(signal)]

    def test_two_steps(self):
        """Detects two changepoints in a signal with three segments."""
        signal = np.concatenate(
            [
                np.ones(20) * 0.9,
                np.ones(20) * 0.3,
                np.ones(20) * 0.7,
            ]
        )
        cps = detect_changepoints_pelt(signal, penalty=3.0, min_size=3)
        interior_cps = [c for c in cps if c < len(signal)]
        assert len(interior_cps) >= 2

    def test_high_penalty_fewer_cps(self):
        """Higher penalty should produce fewer or equal changepoints."""
        rng = np.random.default_rng(42)
        signal = np.concatenate(
            [
                rng.normal(0.8, 0.05, 30),
                rng.normal(0.3, 0.05, 30),
            ]
        )
        cps_low = detect_changepoints_pelt(signal, penalty=1.0, min_size=3)
        cps_high = detect_changepoints_pelt(signal, penalty=50.0, min_size=3)
        assert len(cps_high) <= len(cps_low)

    def test_short_signal(self):
        """Short signal returns only terminal."""
        signal = np.array([0.5, 0.5, 0.5])
        cps = detect_changepoints_pelt(signal, penalty=5.0, min_size=5)
        assert cps == [len(signal)]

    def test_joint_detection(self):
        """Joint detection finds changepoints in 2D signal."""
        rep = np.concatenate([np.ones(30) * 0.9, np.ones(30) * 0.5])
        dem = np.concatenate([np.ones(30) * 0.8, np.ones(30) * 0.4])
        cps = detect_changepoints_joint(rep, dem, penalty=5.0, min_size=3)
        # Should detect the joint break
        interior_cps = [c for c in cps if c < len(rep)]
        assert len(interior_cps) >= 1

    def test_joint_unequal_lengths(self):
        """Joint detection handles unequal signal lengths."""
        rep = np.ones(40) * 0.8
        dem = np.ones(35) * 0.7
        cps = detect_changepoints_joint(rep, dem, penalty=10.0, min_size=3)
        assert len(cps) >= 1  # at least terminal

    def test_min_size_respected(self):
        """Changepoints respect minimum segment size."""
        signal = np.concatenate([np.ones(10) * 0.9, np.ones(10) * 0.3])
        cps = detect_changepoints_pelt(signal, penalty=1.0, min_size=5)
        # All segments should be at least min_size
        boundaries = [0] + cps
        for i in range(len(boundaries) - 1):
            assert boundaries[i + 1] - boundaries[i] >= 5


# ── TestPenaltySensitivity ───────────────────────────────────────────────────


class TestPenaltySensitivity:
    """Tests for run_penalty_sensitivity()."""

    def test_monotone_nonincreasing(self):
        """Changepoint count should be non-increasing with penalty."""
        rng = np.random.default_rng(42)
        signal = np.concatenate(
            [
                rng.normal(0.8, 0.05, 30),
                rng.normal(0.3, 0.05, 30),
                rng.normal(0.6, 0.05, 30),
            ]
        )
        results = run_penalty_sensitivity(signal, penalties=[1, 5, 10, 50, 100])
        counts = [r["n_changepoints"] for r in results]
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i + 1]

    def test_all_penalties_returned(self):
        """Returns one result per penalty value."""
        signal = np.ones(50) * 0.8
        penalties = [3, 5, 10, 15]
        results = run_penalty_sensitivity(signal, penalties=penalties)
        assert len(results) == len(penalties)

    def test_extreme_penalty(self):
        """Very high penalty produces zero changepoints."""
        rng = np.random.default_rng(42)
        signal = rng.normal(0.5, 0.1, 50)
        results = run_penalty_sensitivity(signal, penalties=[10000])
        assert results[0]["n_changepoints"] == 0


# ── TestBuildVoteMatrix ──────────────────────────────────────────────────────


class TestBuildVoteMatrix:
    """Tests for build_vote_matrix()."""

    def test_correct_dimensions(self):
        """Matrix dimensions match filtered legislators and roll calls."""
        votes, rollcalls = _make_votes(n_rollcalls=50, n_rep=30, n_dem=10)

        matrix, slugs, vote_ids, datetimes = build_vote_matrix(votes, rollcalls, "House")

        assert matrix.shape[0] == len(slugs)
        assert matrix.shape[1] == len(vote_ids)
        assert len(datetimes) == len(vote_ids)

    def test_binary_values(self):
        """Matrix values are 0, 1, or NaN."""
        votes, rollcalls = _make_votes(n_rollcalls=50, n_rep=30, n_dem=10)

        matrix, _, _, _ = build_vote_matrix(votes, rollcalls, "House")

        valid = matrix[~np.isnan(matrix)]
        assert set(valid.astype(int)) <= {0, 1}

    def test_chamber_filtering(self):
        """House matrix only includes rep_ slugs."""
        votes, rollcalls = _make_votes(n_rollcalls=50, n_rep=30, n_dem=10)

        matrix, slugs, _, _ = build_vote_matrix(votes, rollcalls, "House")
        assert all(s.startswith("rep_") for s in slugs)


# ── TestCrossReferenceVetoOverrides ──────────────────────────────────────────


class TestCrossReferenceVetoOverrides:
    """Tests for cross_reference_veto_overrides()."""

    def test_finds_nearby(self):
        """Finds override within 14 days of changepoint."""
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["v1"],
                "bill_number": ["HB100"],
                "motion": ["Override Veto"],
                "vote_date": ["2025-03-15"],
                "vote_datetime": ["2025-03-15T12:00:00"],
            }
        )

        result = cross_reference_veto_overrides(["2025-03-10"], rollcalls)
        assert result.height == 1
        assert result["days_apart"][0] == 5

    def test_ignores_distant(self):
        """Ignores overrides more than 14 days from changepoint."""
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["v1"],
                "bill_number": ["HB100"],
                "motion": ["Override Veto"],
                "vote_date": ["2025-06-01"],
                "vote_datetime": ["2025-06-01T12:00:00"],
            }
        )

        result = cross_reference_veto_overrides(["2025-03-10"], rollcalls)
        assert result.height == 0

    def test_empty_changepoints(self):
        """No changepoints returns empty."""
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["v1"],
                "bill_number": ["HB100"],
                "motion": ["Override Veto"],
                "vote_date": ["2025-03-15"],
                "vote_datetime": ["2025-03-15T12:00:00"],
            }
        )
        result = cross_reference_veto_overrides([], rollcalls)
        assert result.height == 0
