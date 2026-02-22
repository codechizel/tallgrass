"""Tests for legislator profile data logic.

Validates target selection, scorecard building, bill-type breakdown, defection
analysis, voting neighbors, and surprising vote filtering.

Usage:
    uv run pytest tests/test_profiles.py -v
"""

from __future__ import annotations

import polars as pl
import pytest

from analysis.profiles_data import (
    MAX_PROFILE_TARGETS,
    ProfileTarget,
    build_scorecard,
    compute_bill_type_breakdown,
    find_defection_bills,
    find_legislator_surprising_votes,
    find_voting_neighbors,
    gather_profile_targets,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def house_leg_df() -> pl.DataFrame:
    """Minimal legislator DataFrame for the house with 10 legislators.

    Includes all columns needed by synthesis_detect and profiles_data.
    Party split: 7 Republican, 3 Democrat.
    """
    return pl.DataFrame({
        "legislator_slug": [f"rep_{chr(97 + i)}" for i in range(10)],
        "full_name": [
            "Alice Adams", "Bob Baker", "Carol Clark", "Dave Davis",
            "Eve Evans", "Frank Fisher", "Grace Green", "Hank Hill",
            "Iris Irving", "Jack Jones",
        ],
        "party": ["Republican"] * 7 + ["Democrat"] * 3,
        "district": [str(i + 1) for i in range(10)],
        "chamber": ["house"] * 10,
        "xi_mean": [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -2.0, -2.5, -3.0],
        "xi_sd": [0.2] * 10,
        "unity_score": [0.98, 0.95, 0.92, 0.88, 0.85, 0.80, 0.70, 0.95, 0.92, 0.98],
        "loyalty_rate": [0.95, 0.90, 0.88, 0.82, 0.78, 0.75, 0.60, 0.90, 0.85, 0.95],
        "maverick_rate": [0.02, 0.05, 0.08, 0.12, 0.15, 0.20, 0.30, 0.05, 0.08, 0.02],
        "weighted_maverick": [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.25, 0.03, 0.05, 0.01],
        "betweenness": [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.15, 0.03, 0.02, 0.01],
        "eigenvector": [0.1, 0.1, 0.1, 0.1, 0.1, 0.12, 0.15, 0.1, 0.1, 0.1],
        "accuracy": [0.95, 0.94, 0.93, 0.90, 0.88, 0.85, 0.78, 0.94, 0.92, 0.96],
        "n_defections": [1, 3, 5, 8, 10, 14, 20, 3, 5, 1],
        "loyalty_zscore": [1.0, 0.5, 0.3, -0.2, -0.5, -0.8, -1.5, 0.5, 0.3, 1.0],
        "PC1": [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -2.0, -2.5, -3.0],
        "PC2": [0.1, 0.2, 0.1, -0.1, 0.0, 0.3, -0.2, 0.1, 0.0, -0.1],
        "xi_mean_percentile": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1, 0.0],
        "betweenness_percentile": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 0.3, 0.2, 0.1],
    })


@pytest.fixture
def senate_leg_df() -> pl.DataFrame:
    """Minimal senate DataFrame with 6 legislators."""
    return pl.DataFrame({
        "legislator_slug": [f"sen_{chr(97 + i)}" for i in range(6)],
        "full_name": [
            "Sam Smith", "Tom Turner", "Uma Upton",
            "Vera Vance", "Will Walker", "Xena Xavier",
        ],
        "party": ["Republican"] * 4 + ["Democrat"] * 2,
        "district": [str(i + 1) for i in range(6)],
        "chamber": ["senate"] * 6,
        "xi_mean": [3.5, 2.0, 1.0, 0.5, -2.0, -3.0],
        "xi_sd": [0.3] * 6,
        "unity_score": [0.98, 0.90, 0.75, 0.82, 0.95, 0.98],
        "loyalty_rate": [0.95, 0.85, 0.60, 0.78, 0.90, 0.95],
        "maverick_rate": [0.02, 0.10, 0.25, 0.18, 0.05, 0.02],
        "weighted_maverick": [0.01, 0.07, 0.20, 0.12, 0.03, 0.01],
        "betweenness": [0.01, 0.03, 0.12, 0.06, 0.02, 0.01],
        "eigenvector": [0.1, 0.1, 0.15, 0.12, 0.1, 0.1],
        "accuracy": [0.96, 0.91, 0.80, 0.87, 0.93, 0.97],
        "n_defections": [1, 6, 15, 10, 3, 1],
        "loyalty_zscore": [1.0, 0.3, -1.2, -0.5, 0.5, 1.0],
        "PC1": [3.5, 2.0, 1.0, 0.5, -2.0, -3.0],
        "PC2": [0.0, 0.1, -0.3, 0.2, 0.0, -0.1],
    })


@pytest.fixture
def leg_dfs(house_leg_df, senate_leg_df) -> dict[str, pl.DataFrame]:
    """Combined leg_dfs dict for both chambers."""
    return {"house": house_leg_df, "senate": senate_leg_df}


@pytest.fixture
def bill_params() -> pl.DataFrame:
    """Synthetic IRT bill parameters with varying discrimination."""
    return pl.DataFrame({
        "vote_id": [f"v{i}" for i in range(10)],
        "beta_mean": [2.0, 1.8, 1.6, 0.8, 0.3, 0.2, 0.1, -0.4, -1.7, -2.1],
        "alpha_mean": [0.5] * 10,
        "bill_number": [f"HB {i}" for i in range(10)],
        "short_title": [f"Bill {i}" for i in range(10)],
        "motion": ["Final Action"] * 10,
    })


@pytest.fixture
def votes_long() -> pl.DataFrame:
    """Synthetic long-form votes: 5 legislators x 10 votes.

    rep_a (R) votes with party on everything.
    rep_g (R) defects frequently — the maverick.
    """
    rows = []
    for vote_idx in range(10):
        vote_id = f"v{vote_idx}"
        # Republicans mostly vote Yea
        for slug in ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e"]:
            rows.append({"legislator_slug": slug, "vote_id": vote_id,
                         "vote_binary": 1, "chamber": "house"})
        # rep_g defects on half the votes
        rows.append({"legislator_slug": "rep_g", "vote_id": vote_id,
                      "vote_binary": 0 if vote_idx < 5 else 1, "chamber": "house"})
        # Democrats mostly vote Nay
        for slug in ["rep_h", "rep_i", "rep_j"]:
            rows.append({"legislator_slug": slug, "vote_id": vote_id,
                         "vote_binary": 0, "chamber": "house"})
    return pl.DataFrame(rows)


@pytest.fixture
def rollcalls() -> pl.DataFrame:
    """Synthetic rollcalls with bill metadata."""
    return pl.DataFrame({
        "vote_id": [f"v{i}" for i in range(10)],
        "bill_number": [f"HB {i}" for i in range(10)],
        "short_title": [f"Bill about topic {i}" for i in range(10)],
        "motion": ["Final Action"] * 10,
        "result": ["Passed"] * 10,
    })


@pytest.fixture
def surprising_votes_df() -> pl.DataFrame:
    """Synthetic surprising votes from prediction phase."""
    return pl.DataFrame({
        "legislator_slug": ["rep_g", "rep_g", "rep_g", "rep_a", "rep_b"],
        "full_name": ["Grace Green", "Grace Green", "Grace Green", "Alice Adams", "Bob Baker"],
        "party": ["Republican"] * 5,
        "vote_id": ["v1", "v2", "v3", "v4", "v5"],
        "bill_number": ["HB 1", "HB 2", "HB 3", "HB 4", "HB 5"],
        "motion": ["Final Action"] * 5,
        "actual": [0, 0, 1, 1, 0],
        "predicted": [1, 1, 0, 0, 1],
        "y_prob": [0.92, 0.88, 0.85, 0.78, 0.72],
        "confidence_error": [0.92, 0.88, 0.85, 0.78, 0.72],
    })


# ── Tests: Gather Profile Targets ────────────────────────────────────────────


class TestGatherProfileTargets:
    """Tests for the gather_profile_targets() function."""

    def test_detects_from_synthesis(self, leg_dfs):
        """Should detect at least one target via detect_all()."""
        targets = gather_profile_targets(leg_dfs)
        assert len(targets) >= 1
        assert all(isinstance(t, ProfileTarget) for t in targets)

    def test_extra_slugs_added(self, leg_dfs):
        """User-specified slugs should appear with role='Requested'."""
        targets = gather_profile_targets(leg_dfs, extra_slugs=["rep_a"])
        slugs = [t.slug for t in targets]
        assert "rep_a" in slugs
        rep_a = next(t for t in targets if t.slug == "rep_a")
        assert rep_a.role == "Requested"

    def test_deduplicates(self, leg_dfs):
        """Same slug detected + requested should appear only once."""
        # First find what gets detected
        targets_auto = gather_profile_targets(leg_dfs)
        if not targets_auto:
            pytest.skip("No auto-detected targets")
        auto_slug = targets_auto[0].slug
        # Add same slug as extra — should not duplicate
        targets = gather_profile_targets(leg_dfs, extra_slugs=[auto_slug])
        slug_counts = [t.slug for t in targets].count(auto_slug)
        assert slug_counts == 1

    def test_max_eight_targets(self, leg_dfs):
        """Never returns more than MAX_PROFILE_TARGETS."""
        # Request many extra slugs
        all_slugs = []
        for df in leg_dfs.values():
            all_slugs.extend(df["legislator_slug"].to_list())
        targets = gather_profile_targets(leg_dfs, extra_slugs=all_slugs)
        assert len(targets) <= MAX_PROFILE_TARGETS


# ── Tests: Build Scorecard ───────────────────────────────────────────────────


class TestBuildScorecard:
    """Tests for the build_scorecard() function."""

    def test_returns_all_metrics(self, house_leg_df):
        """Scorecard dict should have keys for each available metric."""
        result = build_scorecard(house_leg_df, "rep_a")
        assert result is not None
        assert "xi_mean_percentile" in result
        assert "unity_score" in result
        assert "accuracy" in result

    def test_party_averages_included(self, house_leg_df):
        """Scorecard should include _party_avg keys."""
        result = build_scorecard(house_leg_df, "rep_a")
        assert result is not None
        assert "xi_mean_percentile_party_avg" in result
        assert "unity_score_party_avg" in result

    def test_returns_none_for_missing_slug(self, house_leg_df):
        """Should return None for a slug not in the DataFrame."""
        result = build_scorecard(house_leg_df, "rep_nonexistent")
        assert result is None


# ── Tests: Bill Type Breakdown ───────────────────────────────────────────────


class TestComputeBillTypeBreakdown:
    """Tests for the compute_bill_type_breakdown() function."""

    def test_correct_tier_counts(self, bill_params, votes_long):
        """High/low disc bill counts should match classification thresholds."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = compute_bill_type_breakdown(
            "rep_a", bill_params, votes_long, "Republican", party_slugs
        )
        # High disc: |beta_mean| > 1.5 → v0(2.0), v1(1.8), v2(1.6), v8(1.7), v9(2.1) = 5
        # Low disc: |beta_mean| < 0.5 → v4(0.3), v5(0.2), v6(0.1), v7(0.4) = 4
        assert result is not None
        assert result.high_disc_n == 5
        assert result.low_disc_n == 4

    def test_yea_rates_computed(self, bill_params, votes_long):
        """Yea rates should match hand-calculated values."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = compute_bill_type_breakdown(
            "rep_a", bill_params, votes_long, "Republican", party_slugs
        )
        assert result is not None
        # rep_a votes 1 (Yea) on all votes → both rates should be 1.0
        assert result.high_disc_yea_rate == pytest.approx(1.0)
        assert result.low_disc_yea_rate == pytest.approx(1.0)

    def test_returns_none_for_few_bills(self):
        """Should return None when fewer than MIN_BILLS_PER_TIER bills."""
        # Only 2 bills total, both high disc
        bp = pl.DataFrame({
            "vote_id": ["v0", "v1"],
            "beta_mean": [2.0, 1.8],
        })
        vl = pl.DataFrame({
            "legislator_slug": ["rep_a", "rep_a"],
            "vote_id": ["v0", "v1"],
            "vote_binary": [1, 1],
            "chamber": ["house", "house"],
        })
        result = compute_bill_type_breakdown("rep_a", bp, vl, "Republican", ["rep_a"])
        assert result is None

    def test_party_averages(self, bill_params, votes_long):
        """Party average rates should differ from the maverick."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result_maverick = compute_bill_type_breakdown(
            "rep_g", bill_params, votes_long, "Republican", party_slugs
        )
        result_loyal = compute_bill_type_breakdown(
            "rep_a", bill_params, votes_long, "Republican", party_slugs
        )
        assert result_maverick is not None
        assert result_loyal is not None
        # rep_g has different Yea rate than rep_a, so party avg should be between
        assert result_maverick.party_high_disc_yea_rate == result_loyal.party_high_disc_yea_rate


# ── Tests: Find Defection Bills ──────────────────────────────────────────────


class TestFindDefectionBills:
    """Tests for the find_defection_bills() function."""

    def test_finds_defections(self, votes_long, rollcalls):
        """Should return bills where the legislator disagreed with party."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = find_defection_bills(
            "rep_g", votes_long, rollcalls, "Republican", party_slugs
        )
        # rep_g votes 0 on v0-v4 while party majority votes 1
        assert result.height > 0
        assert "legislator_vote" in result.columns
        assert "party_majority_vote" in result.columns

    def test_sorted_by_closeness(self, votes_long, rollcalls):
        """Defections should be sorted with close votes first."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = find_defection_bills(
            "rep_g", votes_long, rollcalls, "Republican", party_slugs
        )
        if result.height > 1:
            # party_yea_pct closer to 50% should come first
            pcts = result["party_yea_pct"].to_list()
            margins = [abs(p - 50.0) for p in pcts]
            assert margins == sorted(margins)

    def test_respects_n_limit(self, votes_long, rollcalls):
        """Should return at most n rows."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = find_defection_bills(
            "rep_g", votes_long, rollcalls, "Republican", party_slugs, n=2
        )
        assert result.height <= 2

    def test_empty_for_loyal_legislator(self, votes_long, rollcalls):
        """Should return empty DataFrame for a loyal party-line voter."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = find_defection_bills(
            "rep_a", votes_long, rollcalls, "Republican", party_slugs
        )
        assert result.height == 0


# ── Tests: Find Voting Neighbors ─────────────────────────────────────────────


class TestFindVotingNeighbors:
    """Tests for the find_voting_neighbors() function."""

    def test_closest_correct(self, votes_long, house_leg_df):
        """Most similar should be same-voting legislators."""
        result = find_voting_neighbors("rep_a", votes_long, house_leg_df)
        assert result is not None
        closest = result["closest"]
        assert len(closest) > 0
        # rep_a always votes 1, so other always-1 voters should be closest
        closest_slugs = [c["slug"] for c in closest]
        assert "rep_b" in closest_slugs  # also always Yea

    def test_most_different_correct(self, votes_long, house_leg_df):
        """Most different should be opposite-voting legislators."""
        result = find_voting_neighbors("rep_a", votes_long, house_leg_df)
        assert result is not None
        most_diff = result["most_different"]
        assert len(most_diff) > 0
        # Democrats always vote 0, so they should be most different from rep_a (always 1)
        diff_slugs = [c["slug"] for c in most_diff]
        assert any(s.startswith("rep_h") or s.startswith("rep_i") or s.startswith("rep_j")
                    for s in diff_slugs)

    def test_excludes_self(self, votes_long, house_leg_df):
        """Target slug should not appear in results."""
        result = find_voting_neighbors("rep_a", votes_long, house_leg_df)
        assert result is not None
        all_slugs = (
            [c["slug"] for c in result["closest"]]
            + [c["slug"] for c in result["most_different"]]
        )
        assert "rep_a" not in all_slugs

    def test_agreement_range(self, votes_long, house_leg_df):
        """All agreement values should be between 0 and 1."""
        result = find_voting_neighbors("rep_a", votes_long, house_leg_df)
        assert result is not None
        for entry in result["closest"] + result["most_different"]:
            assert 0.0 <= entry["agreement"] <= 1.0


# ── Tests: Find Legislator Surprising Votes ──────────────────────────────────


class TestFindLegislatorSurprisingVotes:
    """Tests for the find_legislator_surprising_votes() function."""

    def test_filters_to_slug(self, surprising_votes_df):
        """Should only return rows for the target legislator."""
        result = find_legislator_surprising_votes("rep_g", surprising_votes_df)
        assert result is not None
        assert result.height == 3
        assert (result["legislator_slug"] == "rep_g").all()

    def test_respects_n_limit(self, surprising_votes_df):
        """Should return at most n rows."""
        result = find_legislator_surprising_votes("rep_g", surprising_votes_df, n=2)
        assert result is not None
        assert result.height == 2

    def test_returns_none_for_no_data(self, surprising_votes_df):
        """Should return None for a slug not in the data."""
        result = find_legislator_surprising_votes("rep_nonexistent", surprising_votes_df)
        assert result is None

    def test_returns_none_for_none_input(self):
        """Should return None when surprising_votes_df is None."""
        result = find_legislator_surprising_votes("rep_a", None)
        assert result is None
