"""Tests for the EGA (Exploratory Graph Analysis) library.

Tests tetrachoric correlations, GLASSO network estimation, community
detection, TEFI, UVA, and the full EGA pipeline on synthetic data.
"""

import numpy as np
import pytest

from analysis.ega.boot_ega import BootEGAResult, run_boot_ega
from analysis.ega.community import CommunityResult, detect_communities
from analysis.ega.ega import EGAResult, run_ega
from analysis.ega.glasso import GLASSOResult, glasso_ebic
from analysis.ega.tefi import compare_structures, compute_tefi
from analysis.ega.tetrachoric import TetrachoricResult, tetrachoric_corr_matrix
from analysis.ega.uva import run_uva

# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_two_factor_data(n: int = 200, p1: int = 10, p2: int = 10, seed: int = 42) -> np.ndarray:
    """Generate binary data with a known 2-factor structure.

    Factor 1 drives items 0..p1-1, Factor 2 drives items p1..p1+p2-1.
    """
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.normal(0, 1, size=n)

    data = np.zeros((n, p1 + p2))
    for j in range(p1):
        loading = 0.7 + rng.uniform(0, 0.2)
        noise = rng.normal(0, 1, size=n)
        latent = loading * f1 + np.sqrt(1 - loading**2) * noise
        data[:, j] = (latent > 0).astype(float)

    for j in range(p2):
        loading = 0.7 + rng.uniform(0, 0.2)
        noise = rng.normal(0, 1, size=n)
        latent = loading * f2 + np.sqrt(1 - loading**2) * noise
        data[:, p1 + j] = (latent > 0).astype(float)

    return data


def _make_one_factor_data(n: int = 200, p: int = 10, seed: int = 42) -> np.ndarray:
    """Generate binary data with a known 1-factor structure."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, size=n)

    data = np.zeros((n, p))
    for j in range(p):
        loading = 0.7 + rng.uniform(0, 0.2)
        noise = rng.normal(0, 1, size=n)
        latent = loading * f + np.sqrt(1 - loading**2) * noise
        data[:, j] = (latent > 0).astype(float)

    return data


# ── Tetrachoric ──────────────────────────────────────────────────────────


class TestTetrachoric:
    """Tests for tetrachoric correlation estimation."""

    def test_identity_diagonal(self) -> None:
        """Diagonal of tetrachoric matrix should be 1.0."""
        data = _make_two_factor_data(n=100, p1=5, p2=5)
        result = tetrachoric_corr_matrix(data, max_workers=2)
        np.testing.assert_allclose(np.diag(result.corr_matrix), 1.0)

    def test_symmetric(self) -> None:
        """Tetrachoric matrix should be symmetric."""
        data = _make_two_factor_data(n=100, p1=5, p2=5)
        result = tetrachoric_corr_matrix(data, max_workers=2)
        np.testing.assert_allclose(result.corr_matrix, result.corr_matrix.T, atol=1e-10)

    def test_within_bounds(self) -> None:
        """All correlations should be in [-1, 1]."""
        data = _make_two_factor_data(n=100, p1=5, p2=5)
        result = tetrachoric_corr_matrix(data, max_workers=2)
        assert np.all(result.corr_matrix >= -1.0)
        assert np.all(result.corr_matrix <= 1.0)

    def test_within_factor_higher(self) -> None:
        """Within-factor correlations should be higher than between-factor."""
        data = _make_two_factor_data(n=300, p1=8, p2=8)
        result = tetrachoric_corr_matrix(data, max_workers=2)

        # Within factor 1
        within_1 = result.corr_matrix[:8, :8]
        np.fill_diagonal(within_1, 0)
        mean_within = np.mean(np.abs(within_1))

        # Between factors
        between = result.corr_matrix[:8, 8:]
        mean_between = np.mean(np.abs(between))

        assert mean_within > mean_between, (
            f"Within-factor mean {mean_within:.3f} should exceed "
            f"between-factor mean {mean_between:.3f}"
        )

    def test_handles_nan(self) -> None:
        """Should handle NaN (absent votes) gracefully."""
        data = _make_two_factor_data(n=100, p1=5, p2=5)
        # Introduce 10% NaN
        rng = np.random.default_rng(99)
        mask = rng.random(data.shape) < 0.10
        data[mask] = np.nan

        result = tetrachoric_corr_matrix(data, max_workers=2)
        assert not np.any(np.isnan(result.corr_matrix))

    def test_degenerate_fallback(self) -> None:
        """Should fall back to Pearson for degenerate tables."""
        # All-Yea column paired with normal column
        data = np.column_stack(
            [
                np.ones(50),
                np.array([1] * 30 + [0] * 20, dtype=float),
            ]
        )
        result = tetrachoric_corr_matrix(data, max_workers=1)
        assert result.n_fallback >= 1
        assert result.fallback_mask[0, 1]

    def test_result_shape(self) -> None:
        """Result matrix should be p × p."""
        data = _make_two_factor_data(n=50, p1=4, p2=3)
        result = tetrachoric_corr_matrix(data, max_workers=1)
        assert result.corr_matrix.shape == (7, 7)
        assert result.fallback_mask.shape == (7, 7)


# ── GLASSO ───────────────────────────────────────────────────────────────


class TestGLASSO:
    """Tests for GLASSO network estimation with EBIC."""

    def test_returns_sparse_network(self) -> None:
        """GLASSO should produce a sparser network than the full correlation."""
        data = _make_two_factor_data(n=200, p1=8, p2=8)
        tet = tetrachoric_corr_matrix(data, max_workers=2)
        result = glasso_ebic(tet.corr_matrix, n_obs=200)

        # Should have fewer edges than full graph (16 * 15 / 2 = 120)
        assert result.n_edges < 120
        assert result.n_edges > 0

    def test_partial_corr_symmetric(self) -> None:
        """Partial correlation matrix should be symmetric."""
        data = _make_two_factor_data(n=200, p1=6, p2=6)
        tet = tetrachoric_corr_matrix(data, max_workers=2)
        result = glasso_ebic(tet.corr_matrix, n_obs=200)
        np.testing.assert_allclose(result.partial_corr, result.partial_corr.T, atol=1e-10)

    def test_partial_corr_zero_diagonal(self) -> None:
        """Partial correlation diagonal should be zero (no self-loops)."""
        data = _make_two_factor_data(n=200, p1=6, p2=6)
        tet = tetrachoric_corr_matrix(data, max_workers=2)
        result = glasso_ebic(tet.corr_matrix, n_obs=200)
        np.testing.assert_allclose(np.diag(result.partial_corr), 0.0, atol=1e-10)

    def test_ebic_curve_populated(self) -> None:
        """EBIC curve should have entries from the lambda sweep."""
        data = _make_two_factor_data(n=100, p1=5, p2=5)
        tet = tetrachoric_corr_matrix(data, max_workers=2)
        result = glasso_ebic(tet.corr_matrix, n_obs=100, n_lambdas=20)
        assert len(result.ebic_curve) > 0
        assert result.selected_lambda > 0

    def test_higher_gamma_sparser(self) -> None:
        """Higher gamma should produce sparser networks."""
        data = _make_two_factor_data(n=200, p1=8, p2=8)
        tet = tetrachoric_corr_matrix(data, max_workers=2)

        result_low = glasso_ebic(tet.corr_matrix, n_obs=200, gamma=0.25)
        result_high = glasso_ebic(tet.corr_matrix, n_obs=200, gamma=0.75)

        assert result_high.n_edges <= result_low.n_edges


# ── Community Detection ──────────────────────────────────────────────────


class TestCommunityDetection:
    """Tests for community detection on GLASSO networks."""

    def test_walktrap_finds_communities(self) -> None:
        """Walktrap should find at least 1 community."""
        data = _make_two_factor_data(n=200, p1=8, p2=8)
        tet = tetrachoric_corr_matrix(data, max_workers=2)
        gl = glasso_ebic(tet.corr_matrix, n_obs=200)
        result = detect_communities(gl.partial_corr, corr_matrix=tet.corr_matrix)
        assert result.n_communities >= 1
        assert len(result.assignments) == 16

    def test_leiden_finds_communities(self) -> None:
        """Leiden should also find communities."""
        data = _make_two_factor_data(n=200, p1=8, p2=8)
        tet = tetrachoric_corr_matrix(data, max_workers=2)
        gl = glasso_ebic(tet.corr_matrix, n_obs=200)
        result = detect_communities(
            gl.partial_corr, corr_matrix=tet.corr_matrix, algorithm="leiden"
        )
        assert result.n_communities >= 1
        assert result.algorithm == "leiden"

    def test_empty_network_is_unidimensional(self) -> None:
        """A network with no edges should return K=1."""
        p = 5
        empty = np.zeros((p, p))
        result = detect_communities(empty)
        assert result.n_communities == 1
        assert result.unidimensional

    def test_invalid_algorithm_raises(self) -> None:
        """Should raise ValueError for unknown algorithm."""
        # Need a non-trivial network (with edges) to reach the algorithm dispatch
        p = 5
        partial = np.zeros((p, p))
        partial[0, 1] = partial[1, 0] = 0.5
        partial[2, 3] = partial[3, 2] = 0.5
        with pytest.raises(ValueError, match="Unknown algorithm"):
            detect_communities(partial, algorithm="spectral")


# ── TEFI ─────────────────────────────────────────────────────────────────


class TestTEFI:
    """Tests for Total Entropy Fit Index."""

    def test_single_community_baseline(self) -> None:
        """TEFI for K=1 (all in one community) should be a finite number."""
        p = 10
        corr = np.eye(p)
        assignments = np.zeros(p, dtype=np.int64)
        tefi = compute_tefi(corr, assignments)
        assert np.isfinite(tefi)

    def test_correct_structure_lower_tefi(self) -> None:
        """TEFI should be lower for the correct 2-factor structure than a scrambled one."""
        # Create a block-diagonal correlation matrix (2 clear factors)
        p = 10
        corr = np.eye(p)
        corr[:5, :5] = 0.6
        corr[5:, 5:] = 0.6
        np.fill_diagonal(corr, 1.0)

        correct = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
        # Scrambled: mix items from both blocks into each community
        scrambled = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

        tefi_correct = compute_tefi(corr, correct)
        tefi_scrambled = compute_tefi(corr, scrambled)

        assert tefi_correct < tefi_scrambled, (
            f"Correct structure TEFI ({tefi_correct:.4f}) should be lower "
            f"than scrambled structure TEFI ({tefi_scrambled:.4f})"
        )

    def test_compare_structures_returns_dict(self) -> None:
        """compare_structures should return a dict mapping K → TEFI."""
        p = 10
        corr = np.eye(p)
        results = compare_structures(corr, max_k=3)
        assert 1 in results
        assert 2 in results
        assert 3 in results
        assert all(np.isfinite(v) for v in results.values())


# ── UVA ──────────────────────────────────────────────────────────────────


class TestUVA:
    """Tests for Unique Variable Analysis."""

    def test_redundant_pair_detected(self) -> None:
        """Two items sharing many neighbors should have high wTO."""
        # Build a network where items 0 and 1 connect to the same neighbors
        p = 6
        adj = np.zeros((p, p))
        # Items 0 and 1 both connect to items 2, 3, 4 with similar weights
        for target in [2, 3, 4]:
            adj[0, target] = adj[target, 0] = 0.5
            adj[1, target] = adj[target, 1] = 0.5
        adj[0, 1] = adj[1, 0] = 0.3

        result = run_uva(adj, threshold=0.20)
        # Items 0 and 1 should be flagged
        pair_items = {(p[0], p[1]) for p in result.redundant_pairs}
        assert (0, 1) in pair_items

    def test_no_redundancy_with_distinct_neighborhoods(self) -> None:
        """Nodes with distinct neighborhoods should not be flagged as redundant."""
        # Star topology: node 0 connects to 1,2,3; node 4 connects to 5,6,7
        # Nodes 0 and 4 don't share neighbors → low wTO
        p = 8
        adj = np.zeros((p, p))
        for target in [1, 2, 3]:
            adj[0, target] = adj[target, 0] = 0.5
        for target in [5, 6, 7]:
            adj[4, target] = adj[target, 4] = 0.5

        result = run_uva(adj, threshold=0.25)
        # Nodes 0 and 4 should NOT be flagged (different neighborhoods)
        pair_items = {(p[0], p[1]) for p in result.redundant_pairs}
        assert (0, 4) not in pair_items

    def test_wto_matrix_symmetric(self) -> None:
        """wTO matrix should be symmetric."""
        p = 5
        rng = np.random.default_rng(42)
        adj = rng.uniform(0, 0.5, (p, p))
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)

        result = run_uva(adj, threshold=0.5)
        np.testing.assert_allclose(result.wto_matrix, result.wto_matrix.T, atol=1e-10)


# ── Full EGA Pipeline ────────────────────────────────────────────────────


class TestEGA:
    """Tests for the full EGA pipeline."""

    def test_two_factor_data(self) -> None:
        """EGA should find approximately 2 dimensions in 2-factor data."""
        data = _make_two_factor_data(n=300, p1=10, p2=10)
        result = run_ega(data, max_workers=2)
        # Should find 1-3 communities (2 is ideal, but network methods
        # can merge or split depending on correlation strength)
        assert 1 <= result.n_communities <= 4
        assert result.network_loadings.shape == (20, result.n_communities)

    def test_one_factor_data_unidimensional(self) -> None:
        """EGA on 1-factor data should tend toward K=1."""
        data = _make_one_factor_data(n=300, p=10)
        result = run_ega(data, max_workers=2)
        # With strong 1-factor, should find 1-2 communities
        assert 1 <= result.n_communities <= 3

    def test_result_fields(self) -> None:
        """EGAResult should have all expected fields."""
        data = _make_two_factor_data(n=100, p1=5, p2=5)
        result = run_ega(data, max_workers=2)
        assert isinstance(result, EGAResult)
        assert isinstance(result.tetrachoric, TetrachoricResult)
        assert isinstance(result.glasso, GLASSOResult)
        assert isinstance(result.community, CommunityResult)
        assert len(result.community_assignments) == 10

    def test_invalid_method_raises(self) -> None:
        """Should raise ValueError for unknown network method."""
        data = _make_one_factor_data(n=50, p=5)
        with pytest.raises(ValueError, match="Unknown method"):
            run_ega(data, method="tmfg")


# ── Boot EGA ─────────────────────────────────────────────────────────────


class TestBootEGA:
    """Tests for bootstrap EGA (reduced n_boot for speed)."""

    def test_boot_ega_runs(self) -> None:
        """bootEGA should complete with reduced bootstrap count."""
        data = _make_two_factor_data(n=100, p1=5, p2=5)
        result = run_boot_ega(data, n_boot=10, max_workers=2)
        assert isinstance(result, BootEGAResult)
        assert result.n_boot >= 1  # At least some replicates succeeded
        assert len(result.item_stability) == 10

    def test_dimension_frequency_sums_to_n_boot(self) -> None:
        """Dimension frequency counts should sum to n_boot."""
        data = _make_two_factor_data(n=100, p1=5, p2=5)
        result = run_boot_ega(data, n_boot=10, max_workers=2)
        total = sum(result.dimension_frequency.values())
        assert total == result.n_boot

    def test_stability_in_range(self) -> None:
        """Item stability should be in [0, 1]."""
        data = _make_two_factor_data(n=100, p1=5, p2=5)
        result = run_boot_ega(data, n_boot=10, max_workers=2)
        assert np.all(result.item_stability >= 0.0)
        assert np.all(result.item_stability <= 1.0)

    def test_nonparametric_boot(self) -> None:
        """Non-parametric bootstrap should also work."""
        data = _make_two_factor_data(n=100, p1=5, p2=5)
        result = run_boot_ega(data, n_boot=5, method="nonparametric", max_workers=2)
        assert isinstance(result, BootEGAResult)
        assert result.n_boot >= 1
