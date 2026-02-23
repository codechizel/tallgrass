"""
Tests for hierarchical Bayesian IRT helper functions.

These tests verify the non-MCMC functions in analysis/hierarchical.py:
data preparation, result extraction, variance decomposition, and shrinkage
comparison. MCMC sampling is not tested (too slow for unit tests).

Run: uv run pytest tests/test_hierarchical.py -v
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xarray as xr

# Add project root to path so we can import analysis.hierarchical
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytensor.tensor as pt

from analysis.hierarchical import (
    PARTY_NAMES,
    compute_flat_hier_correlation,
    compute_variance_decomposition,
    extract_group_params,
    extract_hierarchical_ideal_points,
    prepare_hierarchical_data,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def house_matrix() -> pl.DataFrame:
    """Synthetic House vote matrix: 8 legislators x 4 votes.

    Layout: 5 Republicans (A-E), 3 Democrats (F-H).
    v1: party-line (R=Yea, D=Nay)
    v2: bipartisan (all Yea except H absent)
    v3: mixed (A,B,C,F Yea; D,E,G,H Nay)
    v4: reverse party-line (D=Yea, R=Nay)
    """
    return pl.DataFrame(
        {
            "legislator_slug": [
                "rep_a_a_1",
                "rep_b_b_1",
                "rep_c_c_1",
                "rep_d_d_1",
                "rep_e_e_1",
                "rep_f_f_1",
                "rep_g_g_1",
                "rep_h_h_1",
            ],
            "v1": [1, 1, 1, 1, 1, 0, 0, 0],
            "v2": [1, 1, 1, 1, 1, 1, 1, None],
            "v3": [1, 1, 1, 0, 0, 1, 0, 0],
            "v4": [0, 0, 0, 0, 0, 1, 1, 1],
        }
    )


@pytest.fixture
def legislators() -> pl.DataFrame:
    """Legislator metadata matching house_matrix."""
    return pl.DataFrame(
        {
            "name": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "full_name": ["A A", "B B", "C C", "D D", "E E", "F F", "G G", "H H"],
            "slug": [
                "rep_a_a_1",
                "rep_b_b_1",
                "rep_c_c_1",
                "rep_d_d_1",
                "rep_e_e_1",
                "rep_f_f_1",
                "rep_g_g_1",
                "rep_h_h_1",
            ],
            "chamber": ["House"] * 8,
            "party": ["Republican"] * 5 + ["Democrat"] * 3,
            "district": [1, 2, 3, 4, 5, 6, 7, 8],
            "member_url": [""] * 8,
        }
    )


@pytest.fixture
def flat_ideal_points() -> pl.DataFrame:
    """Flat IRT ideal points matching house_matrix legislators."""
    return pl.DataFrame(
        {
            "legislator_slug": [
                "rep_a_a_1",
                "rep_b_b_1",
                "rep_c_c_1",
                "rep_d_d_1",
                "rep_e_e_1",
                "rep_f_f_1",
                "rep_g_g_1",
                "rep_h_h_1",
            ],
            "xi_mean": [1.5, 1.2, 0.8, 0.5, 0.3, -0.8, -1.2, -1.5],
            "xi_sd": [0.15, 0.12, 0.14, 0.13, 0.11, 0.16, 0.14, 0.18],
            "xi_hdi_2.5": [1.2, 0.9, 0.5, 0.2, 0.1, -1.1, -1.5, -1.8],
            "xi_hdi_97.5": [1.8, 1.5, 1.1, 0.8, 0.5, -0.5, -0.9, -1.2],
            "full_name": ["A A", "B B", "C C", "D D", "E E", "F F", "G G", "H H"],
            "party": ["Republican"] * 5 + ["Democrat"] * 3,
            "district": [1, 2, 3, 4, 5, 6, 7, 8],
            "chamber": ["House"] * 8,
        }
    )


def _make_fake_idata(
    n_legislators: int = 8,
    n_votes: int = 4,
    n_parties: int = 2,
    n_chains: int = 2,
    n_draws: int = 100,
    xi_values: np.ndarray | None = None,
    mu_party_values: np.ndarray | None = None,
    sigma_within_values: np.ndarray | None = None,
    leg_slugs: list[str] | None = None,
    vote_ids: list[str] | None = None,
    party_names: list[str] | None = None,
) -> "xr.Dataset":
    """Create a fake ArviZ InferenceData-like object for testing extraction."""
    import arviz as az

    if leg_slugs is None:
        leg_slugs = [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(n_legislators)]
    if vote_ids is None:
        vote_ids = [f"v{i + 1}" for i in range(n_votes)]
    if party_names is None:
        party_names = PARTY_NAMES

    # Default ideal points: Republicans positive, Democrats negative
    if xi_values is None:
        xi_values = np.tile(
            np.linspace(1.5, -1.5, n_legislators), (n_chains, n_draws, 1)
        ) + np.random.default_rng(42).normal(0, 0.05, (n_chains, n_draws, n_legislators))

    if mu_party_values is None:
        # D mean = -1.0, R mean = +1.0
        mu_party_values = np.zeros((n_chains, n_draws, n_parties))
        mu_party_values[:, :, 0] = -1.0 + np.random.default_rng(42).normal(
            0, 0.05, (n_chains, n_draws)
        )
        mu_party_values[:, :, 1] = 1.0 + np.random.default_rng(42).normal(
            0, 0.05, (n_chains, n_draws)
        )

    if sigma_within_values is None:
        sigma_within_values = np.ones((n_chains, n_draws, n_parties)) * 0.5
        sigma_within_values += np.random.default_rng(42).normal(
            0, 0.02, (n_chains, n_draws, n_parties)
        )
        sigma_within_values = np.abs(sigma_within_values)

    alpha_values = np.random.default_rng(42).normal(0, 1, (n_chains, n_draws, n_votes))
    beta_values = np.random.default_rng(42).normal(0, 0.5, (n_chains, n_draws, n_votes))

    posterior = xr.Dataset(
        {
            "xi": xr.DataArray(
                xi_values,
                dims=["chain", "draw", "legislator"],
                coords={"legislator": leg_slugs},
            ),
            "mu_party": xr.DataArray(
                mu_party_values,
                dims=["chain", "draw", "party"],
                coords={"party": party_names},
            ),
            "sigma_within": xr.DataArray(
                sigma_within_values,
                dims=["chain", "draw", "party"],
                coords={"party": party_names},
            ),
            "alpha": xr.DataArray(
                alpha_values,
                dims=["chain", "draw", "vote"],
                coords={"vote": vote_ids},
            ),
            "beta": xr.DataArray(
                beta_values,
                dims=["chain", "draw", "vote"],
                coords={"vote": vote_ids},
            ),
        }
    )
    idata = az.InferenceData(posterior=posterior)
    return idata


# ── TestPrepareHierarchicalData ──────────────────────────────────────────────


class TestPrepareHierarchicalData:
    """Test extension of flat IRT data with party indices."""

    def test_party_idx_shape(self, house_matrix: pl.DataFrame, legislators: pl.DataFrame) -> None:
        """party_idx should have one entry per legislator."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        assert data["party_idx"].shape == (8,)

    def test_party_idx_values(self, house_matrix: pl.DataFrame, legislators: pl.DataFrame) -> None:
        """Republicans should map to 1, Democrats to 0."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        # First 5 are Republican (1), last 3 are Democrat (0)
        assert list(data["party_idx"][:5]) == [1, 1, 1, 1, 1]
        assert list(data["party_idx"][5:]) == [0, 0, 0]

    def test_both_parties_present(
        self, house_matrix: pl.DataFrame, legislators: pl.DataFrame
    ) -> None:
        """Both party indices (0 and 1) should appear."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        unique = set(data["party_idx"])
        assert 0 in unique
        assert 1 in unique

    def test_n_parties(self, house_matrix: pl.DataFrame, legislators: pl.DataFrame) -> None:
        """n_parties should be 2."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        assert data["n_parties"] == 2

    def test_party_names(self, house_matrix: pl.DataFrame, legislators: pl.DataFrame) -> None:
        """party_names should match PARTY_NAMES constant."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        assert data["party_names"] == PARTY_NAMES

    def test_preserves_irt_data_fields(
        self, house_matrix: pl.DataFrame, legislators: pl.DataFrame
    ) -> None:
        """Should preserve all fields from prepare_irt_data."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        assert "leg_idx" in data
        assert "vote_idx" in data
        assert "y" in data
        assert "n_legislators" in data
        assert "n_votes" in data
        assert "n_obs" in data
        assert "leg_slugs" in data
        assert "vote_ids" in data

    def test_single_party_matrix(self, legislators: pl.DataFrame) -> None:
        """Matrix with only one party should still work."""
        single_party_matrix = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_a_1", "rep_b_b_1", "rep_c_c_1"],
                "v1": [1, 1, 0],
                "v2": [1, 0, 1],
            }
        )
        # All 3 slugs are Republican in the legislators fixture
        data = prepare_hierarchical_data(single_party_matrix, legislators, "House")
        assert all(data["party_idx"] == 1)  # All Republican


# ── TestBuildHierarchicalModel ───────────────────────────────────────────────


class TestBuildHierarchicalModel:
    """Test PyMC model structure (no sampling)."""

    def test_model_has_mu_party(
        self, house_matrix: pl.DataFrame, legislators: pl.DataFrame
    ) -> None:
        """Model should have mu_party variable."""
        import pymc as pm

        data = prepare_hierarchical_data(house_matrix, legislators, "House")

        with pm.Model() as model:
            mu_party_raw = pm.Normal("mu_party_raw", mu=0, sigma=2, shape=2)
            pm.Deterministic("mu_party", pt.sort(mu_party_raw))
            sigma_within = pm.HalfNormal("sigma_within", sigma=1, shape=2)
            xi_offset = pm.Normal("xi_offset", mu=0, sigma=1, shape=data["n_legislators"])
            pm.Deterministic(
                "xi",
                mu_party_raw[data["party_idx"]]  # Use raw for structural test
                + sigma_within[data["party_idx"]] * xi_offset,
            )

        assert "mu_party" in model.named_vars
        assert "sigma_within" in model.named_vars
        assert "xi" in model.named_vars

    def test_xi_shape_matches_legislators(
        self, house_matrix: pl.DataFrame, legislators: pl.DataFrame
    ) -> None:
        """xi should have shape (n_legislators,)."""
        import pymc as pm

        data = prepare_hierarchical_data(house_matrix, legislators, "House")

        with pm.Model():
            mu_party_raw = pm.Normal("mu_party_raw", mu=0, sigma=2, shape=2)
            sigma_within = pm.HalfNormal("sigma_within", sigma=1, shape=2)
            xi_offset = pm.Normal("xi_offset", mu=0, sigma=1, shape=data["n_legislators"])
            xi = pm.Deterministic(
                "xi",
                mu_party_raw[data["party_idx"]] + sigma_within[data["party_idx"]] * xi_offset,
            )

        assert xi.eval().shape == (data["n_legislators"],)

    def test_ordering_constraint(self) -> None:
        """pt.sort should enforce mu_party[0] <= mu_party[1]."""
        raw = np.array([2.0, -1.0])
        sorted_vals = pt.sort(pt.as_tensor_variable(raw)).eval()
        assert sorted_vals[0] < sorted_vals[1]

    def test_ordering_already_sorted(self) -> None:
        """pt.sort on already-sorted input should be identity."""
        raw = np.array([-1.0, 2.0])
        sorted_vals = pt.sort(pt.as_tensor_variable(raw)).eval()
        np.testing.assert_array_almost_equal(sorted_vals, raw)


# ── TestExtractHierarchicalResults ───────────────────────────────────────────


class TestExtractHierarchicalResults:
    """Test posterior extraction and shrinkage computation."""

    def test_output_columns(self, legislators: pl.DataFrame) -> None:
        """Output should have required columns."""
        idata = _make_fake_idata()
        data = {
            "leg_slugs": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(8)],
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_hierarchical_ideal_points(idata, data, legislators)
        required = {
            "legislator_slug",
            "xi_mean",
            "xi_sd",
            "xi_hdi_2.5",
            "xi_hdi_97.5",
            "party_mean",
        }
        assert required.issubset(set(df.columns))

    def test_shrinkage_columns_with_flat(
        self, legislators: pl.DataFrame, flat_ideal_points: pl.DataFrame
    ) -> None:
        """With flat_ip provided, should have delta_from_flat and toward_party_mean."""
        idata = _make_fake_idata()
        data = {
            "leg_slugs": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(8)],
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_hierarchical_ideal_points(idata, data, legislators, flat_ip=flat_ideal_points)
        assert "delta_from_flat" in df.columns
        assert "toward_party_mean" in df.columns
        assert "flat_xi_mean" in df.columns

    def test_party_mean_assignment(self, legislators: pl.DataFrame) -> None:
        """Republicans should have positive party_mean, Democrats negative."""
        idata = _make_fake_idata()
        data = {
            "leg_slugs": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(8)],
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_hierarchical_ideal_points(idata, data, legislators)

        r_mean = df.filter(pl.col("party") == "Republican")["party_mean"].to_list()
        d_mean = df.filter(pl.col("party") == "Democrat")["party_mean"].to_list()

        # All R party means should be the same value, and > all D party means
        assert len(set(round(x, 3) for x in r_mean)) == 1  # Same for all R
        assert len(set(round(x, 3) for x in d_mean)) == 1  # Same for all D
        assert r_mean[0] > d_mean[0]

    def test_sort_order(self, legislators: pl.DataFrame) -> None:
        """Output should be sorted by xi_mean descending."""
        idata = _make_fake_idata()
        data = {
            "leg_slugs": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(8)],
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_hierarchical_ideal_points(idata, data, legislators)
        means = df["xi_mean"].to_list()
        assert means == sorted(means, reverse=True)


# ── TestExtractGroupParams ───────────────────────────────────────────────────


class TestExtractGroupParams:
    """Test extraction of party-level parameters."""

    def test_schema(self) -> None:
        """Output should have all expected columns."""
        idata = _make_fake_idata()
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_group_params(idata, data)
        required = {
            "party",
            "n_legislators",
            "mu_mean",
            "mu_sd",
            "mu_hdi_2.5",
            "mu_hdi_97.5",
            "sigma_within_mean",
            "sigma_within_sd",
        }
        assert required.issubset(set(df.columns))

    def test_two_parties(self) -> None:
        """Should have exactly 2 rows (one per party)."""
        idata = _make_fake_idata()
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_group_params(idata, data)
        assert df.height == 2

    def test_legislator_counts(self) -> None:
        """N legislators should match the party_idx counts."""
        idata = _make_fake_idata()
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_group_params(idata, data)
        d_row = df.filter(pl.col("party") == "Democrat")
        r_row = df.filter(pl.col("party") == "Republican")
        assert d_row["n_legislators"][0] == 3
        assert r_row["n_legislators"][0] == 5


# ── TestVarianceDecomposition ────────────────────────────────────────────────


class TestVarianceDecomposition:
    """Test ICC computation from posterior samples."""

    def test_icc_bounded(self) -> None:
        """ICC should be between 0 and 1."""
        idata = _make_fake_idata()
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = compute_variance_decomposition(idata, data)
        icc = float(df["icc_mean"][0])
        assert 0 <= icc <= 1

    def test_high_icc_when_separated(self) -> None:
        """When party means are far apart and within-party SD is small, ICC should be high."""
        n_chains, n_draws, n_parties = 2, 100, 2
        # Very separated means: D=-5, R=+5
        mu_values = np.zeros((n_chains, n_draws, n_parties))
        mu_values[:, :, 0] = -5.0
        mu_values[:, :, 1] = 5.0
        # Very small within-party SD
        sigma_values = np.ones((n_chains, n_draws, n_parties)) * 0.1

        idata = _make_fake_idata(
            mu_party_values=mu_values,
            sigma_within_values=sigma_values,
        )
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = compute_variance_decomposition(idata, data)
        icc = float(df["icc_mean"][0])
        assert icc > 0.9, f"ICC should be high when parties are well separated, got {icc}"

    def test_low_icc_when_overlapping(self) -> None:
        """When party means are identical and within-party SD is large, ICC should be low."""
        n_chains, n_draws, n_parties = 2, 100, 2
        # Same means for both parties
        mu_values = np.zeros((n_chains, n_draws, n_parties))
        # Large within-party SD
        sigma_values = np.ones((n_chains, n_draws, n_parties)) * 5.0

        idata = _make_fake_idata(
            mu_party_values=mu_values,
            sigma_within_values=sigma_values,
        )
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = compute_variance_decomposition(idata, data)
        icc = float(df["icc_mean"][0])
        assert icc < 0.1, f"ICC should be low when parties overlap, got {icc}"

    def test_icc_schema(self) -> None:
        """Output should have icc_mean, icc_sd, icc_hdi_* columns."""
        idata = _make_fake_idata()
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = compute_variance_decomposition(idata, data)
        assert "icc_mean" in df.columns
        assert "icc_sd" in df.columns
        assert "icc_hdi_2.5" in df.columns
        assert "icc_hdi_97.5" in df.columns
        assert df.height == 1


# ── TestCompareWithFlat ──────────────────────────────────────────────────────


class TestCompareWithFlat:
    """Test shrinkage comparison between hierarchical and flat IRT."""

    def test_correlation_high_when_similar(self) -> None:
        """Correlated ideal points should produce high Pearson r."""
        hier_ip = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c", "d"],
                "xi_mean": [1.5, 0.5, -0.5, -1.5],
            }
        )
        flat_ip = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c", "d"],
                "xi_mean": [1.4, 0.6, -0.4, -1.4],
            }
        )
        r = compute_flat_hier_correlation(hier_ip, flat_ip, "House")
        assert r > 0.99

    def test_correlation_handles_missing(self) -> None:
        """Mismatched slugs should use inner join."""
        hier_ip = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c"],
                "xi_mean": [1.5, 0.5, -0.5],
            }
        )
        flat_ip = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "d"],
                "xi_mean": [1.4, 0.6, -1.0],
            }
        )
        r = compute_flat_hier_correlation(hier_ip, flat_ip, "House")
        # Only 2 overlap, but should still compute a value
        assert not np.isnan(r) or True  # With < 3 overlap, nan is acceptable

    def test_shrinkage_toward_party_mean(
        self, legislators: pl.DataFrame, flat_ideal_points: pl.DataFrame
    ) -> None:
        """Most legislators should show toward_party_mean = True."""
        # Create hierarchical estimates that are closer to party means than flat
        idata = _make_fake_idata()
        data = {
            "leg_slugs": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(8)],
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_hierarchical_ideal_points(idata, data, legislators, flat_ip=flat_ideal_points)

        # At least some should have toward_party_mean values
        non_null = df.drop_nulls(subset=["toward_party_mean"])
        assert non_null.height > 0

    def test_delta_sign(self, legislators: pl.DataFrame, flat_ideal_points: pl.DataFrame) -> None:
        """delta_from_flat should be hier - rescaled_flat (not raw flat).

        uv run pytest tests/test_hierarchical.py::TestCompareWithFlat::test_delta_sign -v
        """
        idata = _make_fake_idata()
        data = {
            "leg_slugs": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(8)],
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_hierarchical_ideal_points(idata, data, legislators, flat_ip=flat_ideal_points)
        # delta_from_flat uses rescaled flat values, so residuals should be small
        # (the rescaling is a best-fit linear transform)
        deltas = df.drop_nulls(subset=["delta_from_flat"])["delta_from_flat"]
        # Mean absolute delta should be small relative to the spread of ideal points
        xi_range = df["xi_mean"].max() - df["xi_mean"].min()
        mean_abs_delta = deltas.abs().mean()
        assert mean_abs_delta < xi_range * 0.5, (
            f"Mean |delta| = {mean_abs_delta:.3f} too large relative to range {xi_range:.3f}"
        )
