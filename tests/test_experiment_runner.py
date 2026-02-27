"""Tests for experiment runner infrastructure.

Verifies ExperimentConfig defaults and immutability, compute_pca_initvals()
with synthetic data, and _fmt_elapsed() formatting. The run_experiment()
orchestrator is not unit-tested here (it requires MCMC sampling).

Run: uv run pytest tests/test_experiment_runner.py -v
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.model_spec import PRODUCTION_BETA, BetaPriorSpec

from analysis.experiment_runner import (
    ExperimentConfig,
    _fmt_elapsed,
    compute_pca_initvals,
)

# ── ExperimentConfig ────────────────────────────────────────────────────────


class TestExperimentConfig:
    """ExperimentConfig frozen dataclass with production defaults."""

    def test_frozen(self):
        config = ExperimentConfig(name="test", description="test desc")
        with pytest.raises(AttributeError):
            config.name = "changed"  # type: ignore[misc]

    def test_required_fields(self):
        config = ExperimentConfig(name="test", description="test desc")
        assert config.name == "test"
        assert config.description == "test desc"

    def test_default_session(self):
        config = ExperimentConfig(name="test", description="test desc")
        assert config.session == "2025-26"

    def test_default_beta_prior(self):
        config = ExperimentConfig(name="test", description="test desc")
        assert config.beta_prior == PRODUCTION_BETA
        assert config.beta_prior.distribution == "normal"

    def test_default_sampling_params(self):
        config = ExperimentConfig(name="test", description="test desc")
        assert config.n_samples == 2000
        assert config.n_tune == 1500
        assert config.n_chains == 4
        assert config.target_accept == 0.95

    def test_default_include_joint(self):
        config = ExperimentConfig(name="test", description="test desc")
        assert config.include_joint is False

    def test_default_chambers(self):
        config = ExperimentConfig(name="test", description="test desc")
        assert config.chambers == ("House", "Senate")

    def test_custom_beta_prior(self):
        spec = BetaPriorSpec("lognormal", {"mu": 0, "sigma": 0.5})
        config = ExperimentConfig(name="test", description="test desc", beta_prior=spec)
        assert config.beta_prior.distribution == "lognormal"
        assert config.beta_prior.params == {"mu": 0, "sigma": 0.5}

    def test_custom_sampling_params(self):
        config = ExperimentConfig(
            name="test",
            description="test desc",
            n_samples=4000,
            n_tune=3000,
            n_chains=2,
            target_accept=0.99,
        )
        assert config.n_samples == 4000
        assert config.n_tune == 3000
        assert config.n_chains == 2
        assert config.target_accept == 0.99

    def test_include_joint(self):
        config = ExperimentConfig(name="test", description="test desc", include_joint=True)
        assert config.include_joint is True

    def test_single_chamber(self):
        config = ExperimentConfig(name="test", description="test desc", chambers=("House",))
        assert config.chambers == ("House",)

    def test_equality(self):
        a = ExperimentConfig(name="test", description="desc")
        b = ExperimentConfig(name="test", description="desc")
        assert a == b

    def test_inequality(self):
        a = ExperimentConfig(name="test", description="desc")
        b = ExperimentConfig(name="other", description="desc")
        assert a != b


# ── compute_pca_initvals ────────────────────────────────────────────────────


class TestComputePcaInitvals:
    """PCA-informed initialization for xi_offset."""

    @pytest.fixture
    def pca_data(self) -> tuple[pl.DataFrame, dict]:
        """Synthetic PCA scores and hierarchical data dict."""
        slugs = ["rep_a", "rep_b", "rep_c", "rep_d"]
        pca_scores = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "PC1": [2.0, -1.0, 0.0, 1.0],
                "PC2": [0.1, 0.2, 0.3, 0.4],
            }
        )
        data = {"leg_slugs": slugs}
        return pca_scores, data

    def test_output_shape(self, pca_data):
        pca_scores, data = pca_data
        result = compute_pca_initvals(pca_scores, data)
        assert result.shape == (4,)

    def test_output_dtype(self, pca_data):
        pca_scores, data = pca_data
        result = compute_pca_initvals(pca_scores, data)
        assert result.dtype == np.float64

    def test_standardized(self, pca_data):
        """Output should be approximately standardized (mean~0, std~1)."""
        pca_scores, data = pca_data
        result = compute_pca_initvals(pca_scores, data)
        assert abs(result.mean()) < 1e-6
        assert abs(result.std() - 1.0) < 0.01

    def test_preserves_ordering(self, pca_data):
        """Legislators with higher PC1 should have higher initvals."""
        pca_scores, data = pca_data
        result = compute_pca_initvals(pca_scores, data)
        # rep_a has PC1=2.0 (highest), rep_b has PC1=-1.0 (lowest)
        slug_to_idx = {s: i for i, s in enumerate(data["leg_slugs"])}
        assert result[slug_to_idx["rep_a"]] > result[slug_to_idx["rep_b"]]

    def test_subset_of_legislators(self):
        """Only legislators in data['leg_slugs'] are used."""
        pca_scores = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e"],
                "PC1": [2.0, -1.0, 0.0, 1.0, 5.0],
            }
        )
        data = {"leg_slugs": ["rep_a", "rep_c"]}
        result = compute_pca_initvals(pca_scores, data)
        assert result.shape == (2,)


# ── _fmt_elapsed ────────────────────────────────────────────────────────────


class TestFmtElapsed:
    """Elapsed time formatting."""

    def test_seconds(self):
        assert _fmt_elapsed(30.5) == "30.5s"

    def test_minutes(self):
        assert _fmt_elapsed(125) == "2m 5s"

    def test_hours(self):
        assert _fmt_elapsed(3725) == "1h 2m 5s"

    def test_zero(self):
        assert _fmt_elapsed(0) == "0.0s"

    def test_under_minute(self):
        assert _fmt_elapsed(59.9) == "59.9s"

    def test_exactly_one_minute(self):
        assert _fmt_elapsed(60) == "1m 0s"

    def test_exactly_one_hour(self):
        assert _fmt_elapsed(3600) == "1h 0m 0s"
