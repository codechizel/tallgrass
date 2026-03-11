"""Tests for Phase 06: 2D Bayesian IRT (experimental).

Covers sign-flip logic, PCA correlation, and convergence constants.
Model building and MCMC sampling are not unit-tested.

Run: uv run pytest tests/test_irt_2d.py -v
"""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from analysis.irt_2d import (  # noqa: E402
    ANNOTATE_SLUGS,
    ESS_THRESHOLD,
    MAX_DIVERGENCES,
    N_CHAINS,
    N_SAMPLES,
    N_TUNE,
    N_TUNE_SUPERMAJORITY,
    RHAT_THRESHOLD,
    SUPERMAJORITY_THRESHOLD,
    THOMPSON_SLUGS,
    TYSON_SLUGS,
    apply_dim1_sign_check,
    compute_beta_init_from_pca,
    correlate_with_pca,
    parse_args,
    plot_dim1_forest,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def ideal_2d_positive() -> pl.DataFrame:
    """Ideal points where Republican mean Dim 1 is positive (no flip)."""
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_a", "rep_b", "rep_c", "rep_d"],
            "full_name": ["Alice A", "Bob B", "Carol C", "Dave D"],
            "party": ["Republican", "Republican", "Democrat", "Democrat"],
            "xi_dim1_mean": [1.5, 0.8, -1.2, -0.5],
            "xi_dim1_hdi_3%": [1.0, 0.3, -1.7, -1.0],
            "xi_dim1_hdi_97%": [2.0, 1.3, -0.7, 0.0],
            "xi_dim2_mean": [0.1, -0.2, 0.3, -0.4],
            "xi_dim2_hdi_3%": [-0.4, -0.7, -0.2, -0.9],
            "xi_dim2_hdi_97%": [0.6, 0.3, 0.8, 0.1],
        }
    )


@pytest.fixture
def ideal_2d_negative() -> pl.DataFrame:
    """Ideal points where Republican mean Dim 1 is negative (needs flip)."""
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_a", "rep_b", "rep_c", "rep_d"],
            "full_name": ["Alice A", "Bob B", "Carol C", "Dave D"],
            "party": ["Republican", "Republican", "Democrat", "Democrat"],
            "xi_dim1_mean": [-1.5, -0.8, 1.2, 0.5],
            "xi_dim1_hdi_3%": [-2.0, -1.3, 0.7, 0.0],
            "xi_dim1_hdi_97%": [-1.0, -0.3, 1.7, 1.0],
            "xi_dim2_mean": [0.1, -0.2, 0.3, -0.4],
            "xi_dim2_hdi_3%": [-0.4, -0.7, -0.2, -0.9],
            "xi_dim2_hdi_97%": [0.6, 0.3, 0.8, 0.1],
        }
    )


@pytest.fixture
def pca_scores() -> pl.DataFrame:
    """PCA scores aligned with ideal_2d fixtures."""
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_a", "rep_b", "rep_c", "rep_d"],
            "PC1": [2.0, 1.0, -1.5, -0.5],
            "PC2": [0.1, -0.3, 0.4, -0.5],
        }
    )


# ── apply_dim1_sign_check ────────────────────────────────────────────────────


class TestApplyDim1SignCheck:
    """Sign-flip for Dim 1 when Republican mean is negative."""

    def test_no_flip_when_positive(self, ideal_2d_positive):
        result = apply_dim1_sign_check(ideal_2d_positive)
        # Means should be unchanged
        assert result["xi_dim1_mean"].to_list() == ideal_2d_positive["xi_dim1_mean"].to_list()

    def test_flip_when_negative(self, ideal_2d_negative):
        result = apply_dim1_sign_check(ideal_2d_negative)
        # Republican means should now be positive
        r_mean = result.filter(pl.col("party") == "Republican")["xi_dim1_mean"].mean()
        assert r_mean > 0

    def test_flip_negates_means(self, ideal_2d_negative):
        result = apply_dim1_sign_check(ideal_2d_negative)
        original_means = ideal_2d_negative["xi_dim1_mean"].to_list()
        flipped_means = result["xi_dim1_mean"].to_list()
        for orig, flipped in zip(original_means, flipped_means):
            assert flipped == pytest.approx(-orig)

    def test_flip_swaps_hdi_bounds(self, ideal_2d_negative):
        result = apply_dim1_sign_check(ideal_2d_negative)
        # After flip: new hdi_3% = -old hdi_97%, new hdi_97% = -old hdi_3%
        for i in range(ideal_2d_negative.height):
            old_lo = ideal_2d_negative["xi_dim1_hdi_3%"][i]
            old_hi = ideal_2d_negative["xi_dim1_hdi_97%"][i]
            new_lo = result["xi_dim1_hdi_3%"][i]
            new_hi = result["xi_dim1_hdi_97%"][i]
            assert new_lo == pytest.approx(-old_hi)
            assert new_hi == pytest.approx(-old_lo)

    def test_dim2_unchanged(self, ideal_2d_negative):
        result = apply_dim1_sign_check(ideal_2d_negative)
        assert result["xi_dim2_mean"].to_list() == ideal_2d_negative["xi_dim2_mean"].to_list()

    def test_preserves_columns(self, ideal_2d_positive):
        result = apply_dim1_sign_check(ideal_2d_positive)
        assert set(result.columns) == set(ideal_2d_positive.columns)

    def test_preserves_row_count(self, ideal_2d_negative):
        result = apply_dim1_sign_check(ideal_2d_negative)
        assert result.height == ideal_2d_negative.height


# ── correlate_with_pca ────────────────────────────────────────────────────────


class TestCorrelateWithPca:
    """2D IRT vs PCA correlation checks."""

    def test_returns_four_keys(self, ideal_2d_positive, pca_scores):
        data = {"leg_slugs": ["rep_a", "rep_b", "rep_c", "rep_d"]}
        result = correlate_with_pca(ideal_2d_positive, pca_scores, data)
        expected_keys = {
            "dim1_vs_pc1_pearson",
            "dim1_vs_pc1_spearman",
            "dim2_vs_pc2_pearson",
            "dim2_vs_pc2_spearman",
        }
        assert set(result.keys()) == expected_keys

    def test_correlations_in_range(self, ideal_2d_positive, pca_scores):
        data = {"leg_slugs": ["rep_a", "rep_b", "rep_c", "rep_d"]}
        result = correlate_with_pca(ideal_2d_positive, pca_scores, data)
        for key, val in result.items():
            assert -1.0 <= val <= 1.0, f"{key} = {val} out of range"

    def test_high_dim1_pc1_correlation(self, ideal_2d_positive, pca_scores):
        """Dim 1 and PC1 are constructed to be positively correlated."""
        data = {"leg_slugs": ["rep_a", "rep_b", "rep_c", "rep_d"]}
        result = correlate_with_pca(ideal_2d_positive, pca_scores, data)
        assert result["dim1_vs_pc1_pearson"] > 0.9

    def test_subset_of_legislators(self, ideal_2d_positive, pca_scores):
        """Only shared legislators are used."""
        data = {"leg_slugs": ["rep_a", "rep_c"]}
        result = correlate_with_pca(ideal_2d_positive, pca_scores, data)
        # Should still return valid correlations
        for val in result.values():
            assert isinstance(val, float)


# ── Constants ─────────────────────────────────────────────────────────────────


class TestConstants:
    """Verify experimental constants are relaxed vs production."""

    def test_rhat_threshold_relaxed(self):
        assert RHAT_THRESHOLD > 1.01  # Production is 1.01

    def test_ess_threshold_relaxed(self):
        assert ESS_THRESHOLD < 400  # Production is 400

    def test_max_divergences_nonzero(self):
        assert MAX_DIVERGENCES > 0  # Production is 0

    def test_sampling_params(self):
        assert N_SAMPLES >= 1000
        assert N_TUNE >= 1000
        assert N_CHAINS >= 2

    def test_annotate_slugs_union(self):
        assert ANNOTATE_SLUGS == TYSON_SLUGS | THOMPSON_SLUGS


# ── Init Strategy Default ────────────────────────────────────────────────────


class TestInitStrategyDefault:
    """Phase 06 must default to pca-informed to avoid horseshoe contamination."""

    def test_default_is_pca_informed(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["irt_2d.py"])
        args = parse_args()
        assert args.init_strategy == "pca-informed"


# ── Dim 1 Forest Plot ───────────────────────────────────────────────────────


class TestPlotDim1Forest:
    """Smoke tests for the Dim 1 forest plot."""

    def test_creates_png(self, ideal_2d_positive, tmp_path):
        plot_dim1_forest(ideal_2d_positive, "Senate", tmp_path)
        assert (tmp_path / "dim1_forest_senate.png").exists()

    def test_creates_png_house(self, ideal_2d_positive, tmp_path):
        plot_dim1_forest(ideal_2d_positive, "House", tmp_path)
        assert (tmp_path / "dim1_forest_house.png").exists()


# ── Supermajority Tuning Constants (ADR-0112) ────────────────────────────────


class TestSupermajorityConstants:
    """ADR-0112: constants for adaptive N_TUNE and supermajority detection."""

    def test_n_tune_supermajority_doubled(self):
        assert N_TUNE_SUPERMAJORITY == 2 * N_TUNE

    def test_supermajority_threshold_fraction(self):
        assert 0.5 < SUPERMAJORITY_THRESHOLD < 1.0
        assert SUPERMAJORITY_THRESHOLD == 0.70


# ── Contested-Only Flag ──────────────────────────────────────────────────────


class TestContestedOnlyFlag:
    """ADR-0112: --contested-only CLI flag."""

    def test_contested_only_default_false(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["irt_2d.py"])
        args = parse_args()
        assert args.contested_only is False

    def test_contested_only_flag(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["irt_2d.py", "--contested-only"])
        args = parse_args()
        assert args.contested_only is True

    def test_csv_flag_default_false(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["irt_2d.py"])
        args = parse_args()
        assert args.csv is False


# ── Beta Init from PCA Loadings (ADR-0112) ──────────────────────────────────


@pytest.fixture
def vote_matrix() -> pl.DataFrame:
    """Small vote matrix for testing beta init from PCA."""
    rng = np.random.default_rng(42)
    n_legs, n_votes = 20, 10
    # Create a matrix with clear structure: first 10 legs vote one way, rest opposite
    data = {"legislator_slug": [f"leg_{i}" for i in range(n_legs)]}
    for j in range(n_votes):
        votes = np.zeros(n_legs)
        # First group votes Yea, second group votes Nay (with noise)
        votes[:10] = rng.binomial(1, 0.8, 10)
        votes[10:] = rng.binomial(1, 0.2, 10)
        data[f"vote_{j}"] = votes.tolist()
    return pl.DataFrame(data)


@pytest.fixture
def irt_data_dict() -> dict:
    """Minimal IRT data dict matching vote_matrix fixture."""
    return {
        "vote_ids": [f"vote_{j}" for j in range(10)],
        "leg_slugs": [f"leg_{i}" for i in range(20)],
        "n_votes": 10,
        "n_legislators": 20,
    }


class TestComputeBetaInitFromPca:
    """ADR-0112: beta init from PCA loadings."""

    def test_returns_tuple(self, vote_matrix, irt_data_dict):
        result = compute_beta_init_from_pca(vote_matrix, irt_data_dict)
        assert result is not None
        beta_col0, anchor_pos = result
        assert isinstance(beta_col0, np.ndarray)
        assert isinstance(anchor_pos, float)

    def test_beta_col0_shape(self, vote_matrix, irt_data_dict):
        beta_col0, _ = compute_beta_init_from_pca(vote_matrix, irt_data_dict)
        assert beta_col0.shape == (irt_data_dict["n_votes"],)

    def test_anchor_positive_is_positive(self, vote_matrix, irt_data_dict):
        _, anchor_pos = compute_beta_init_from_pca(vote_matrix, irt_data_dict)
        assert anchor_pos > 0

    def test_beta_col0_nonzero(self, vote_matrix, irt_data_dict):
        """PCA loadings should have nonzero values for a structured matrix."""
        beta_col0, _ = compute_beta_init_from_pca(vote_matrix, irt_data_dict)
        assert np.any(beta_col0 != 0)

    def test_handles_missing_votes(self, irt_data_dict):
        """Handles NaN in vote matrix (imputes with column mean)."""
        data = {
            "legislator_slug": [f"leg_{i}" for i in range(5)],
            "vote_0": [1.0, 0.0, None, 1.0, 0.0],
            "vote_1": [0.0, 1.0, 1.0, None, 0.0],
            "vote_2": [1.0, 1.0, 0.0, 0.0, 1.0],
        }
        matrix = pl.DataFrame(data)
        small_data = {
            "vote_ids": ["vote_0", "vote_1", "vote_2"],
            "n_votes": 3,
            "n_legislators": 5,
        }
        result = compute_beta_init_from_pca(matrix, small_data)
        assert result is not None
