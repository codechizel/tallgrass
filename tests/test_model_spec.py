"""Tests for BetaPriorSpec model specification dataclass.

Verifies frozen immutability, describe() output, PRODUCTION_BETA values,
build() dispatch with mocked PyMC, unknown distribution handling, and equality.

Run: uv run pytest tests/test_model_spec.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.model_spec import PRODUCTION_BETA, BetaPriorSpec

# ── Frozen Immutability ──────────────────────────────────────────────────────


class TestFrozenImmutability:
    """BetaPriorSpec is a frozen dataclass — no mutation allowed."""

    def test_cannot_set_distribution(self):
        spec = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})
        with pytest.raises(AttributeError):
            spec.distribution = "lognormal"  # type: ignore[misc]

    def test_cannot_set_params(self):
        spec = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})
        with pytest.raises(AttributeError):
            spec.params = {"mu": 1, "sigma": 2}  # type: ignore[misc]


# ── describe() ───────────────────────────────────────────────────────────────


class TestDescribe:
    """Human-readable description strings."""

    def test_normal(self):
        spec = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})
        assert spec.describe() == "Normal(mu=0, sigma=1)"

    def test_lognormal(self):
        spec = BetaPriorSpec("lognormal", {"mu": 0, "sigma": 0.5})
        assert spec.describe() == "LogNormal(mu=0, sigma=0.5)"

    def test_halfnormal(self):
        spec = BetaPriorSpec("halfnormal", {"sigma": 1})
        assert spec.describe() == "HalfNormal(sigma=1)"

    def test_normal_nonzero_mu(self):
        spec = BetaPriorSpec("normal", {"mu": 1.5, "sigma": 2.0})
        assert spec.describe() == "Normal(mu=1.5, sigma=2.0)"


# ── PRODUCTION_BETA ──────────────────────────────────────────────────────────


class TestProductionBeta:
    """PRODUCTION_BETA must match the hardcoded pm.Normal('beta', mu=0, sigma=1)."""

    def test_distribution(self):
        assert PRODUCTION_BETA.distribution == "normal"

    def test_params(self):
        assert PRODUCTION_BETA.params == {"mu": 0, "sigma": 1}

    def test_describe(self):
        assert PRODUCTION_BETA.describe() == "Normal(mu=0, sigma=1)"


# ── build() dispatch ─────────────────────────────────────────────────────────


class TestBuild:
    """build() dispatches to the correct PyMC distribution."""

    def test_build_normal(self):
        spec = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})
        mock_pm = MagicMock()
        mock_pm.Normal.return_value = "beta_var"
        with patch.dict("sys.modules", {"pymc": mock_pm}):
            result = spec.build(100)
        mock_pm.Normal.assert_called_once_with("beta", shape=100, dims="vote", mu=0, sigma=1)
        assert result == "beta_var"

    def test_build_lognormal(self):
        spec = BetaPriorSpec("lognormal", {"mu": 0, "sigma": 0.5})
        mock_pm = MagicMock()
        mock_pm.LogNormal.return_value = "beta_var"
        with patch.dict("sys.modules", {"pymc": mock_pm}):
            result = spec.build(200)
        mock_pm.LogNormal.assert_called_once_with("beta", shape=200, dims="vote", mu=0, sigma=0.5)
        assert result == "beta_var"

    def test_build_halfnormal(self):
        spec = BetaPriorSpec("halfnormal", {"sigma": 1})
        mock_pm = MagicMock()
        mock_pm.HalfNormal.return_value = "beta_var"
        with patch.dict("sys.modules", {"pymc": mock_pm}):
            result = spec.build(50)
        mock_pm.HalfNormal.assert_called_once_with("beta", shape=50, dims="vote", sigma=1)
        assert result == "beta_var"

    def test_build_custom_dims(self):
        spec = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})
        mock_pm = MagicMock()
        mock_pm.Normal.return_value = "beta_var"
        with patch.dict("sys.modules", {"pymc": mock_pm}):
            spec.build(100, dims="bill")
        mock_pm.Normal.assert_called_once_with("beta", shape=100, dims="bill", mu=0, sigma=1)

    def test_build_unknown_distribution(self):
        spec = BetaPriorSpec("cauchy", {"alpha": 0, "beta": 1})
        with pytest.raises(ValueError, match="Unknown beta prior distribution: 'cauchy'"):
            spec.build(100)


# ── Equality and Hashing ────────────────────────────────────────────────────


class TestEquality:
    """Frozen dataclasses support equality and hashing."""

    def test_equal_specs(self):
        a = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})
        b = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})
        assert a == b

    def test_unequal_distribution(self):
        a = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})
        b = BetaPriorSpec("lognormal", {"mu": 0, "sigma": 1})
        assert a != b

    def test_unequal_params(self):
        a = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})
        b = BetaPriorSpec("normal", {"mu": 0, "sigma": 2})
        assert a != b

    def test_production_beta_equals_manual(self):
        manual = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})
        assert PRODUCTION_BETA == manual
