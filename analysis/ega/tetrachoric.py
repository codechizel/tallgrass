"""Tetrachoric correlation estimation for binary vote matrices.

Tetrachoric correlation estimates the Pearson correlation between two
latent continuous variables that underlie observed binary outcomes.
This is the correct correlation type for Yea/Nay vote data — using
Pearson on raw binary data underestimates true associations.

Algorithm: For each item pair, build a 2×2 contingency table and
maximize the bivariate normal log-likelihood over rho ∈ (-1, 1).
Falls back to Pearson for degenerate tables (any cell count = 0).
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal, norm


@dataclass(frozen=True)
class TetrachoricResult:
    """Result of tetrachoric correlation estimation.

    Attributes:
        corr_matrix: p × p symmetric tetrachoric correlation matrix.
        fallback_mask: p × p boolean matrix; True where Pearson was used
            instead of tetrachoric (due to degenerate 2×2 tables).
        n_pairs: Total number of item pairs computed.
        n_fallback: Number of pairs that fell back to Pearson.
    """

    corr_matrix: NDArray[np.float64]
    fallback_mask: NDArray[np.bool_]
    n_pairs: int
    n_fallback: int


def _bivariate_normal_prob(
    thresh_a: float, thresh_b: float, rho: float
) -> tuple[float, float, float, float]:
    """Compute 2×2 cell probabilities from bivariate normal with correlation rho.

    Returns (p00, p01, p10, p11) where pij = P(X <= thresh_a if i=0, X > thresh_a if i=1,
    Y <= thresh_b if j=0, Y > thresh_b if j=1).
    """
    # P(X <= a, Y <= b) via scipy multivariate normal CDF
    cov_mat = np.array([[1.0, rho], [rho, 1.0]])
    rv = multivariate_normal(mean=[0.0, 0.0], cov=cov_mat, allow_singular=True)
    p00 = float(rv.cdf([thresh_a, thresh_b]))

    # Marginals
    pa = norm.cdf(thresh_a)
    pb = norm.cdf(thresh_b)

    p01 = pa - p00
    p10 = pb - p00
    p11 = 1.0 - pa - pb + p00

    return float(p00), float(p01), float(p10), float(p11)


def _tetrachoric_pair(n00: int, n01: int, n10: int, n11: int) -> tuple[float, bool]:
    """Estimate tetrachoric correlation for a single 2×2 table.

    Returns (rho, fallback) where fallback=True if Pearson was used.
    """
    total = n00 + n01 + n10 + n11
    if total == 0:
        return 0.0, True

    # Degenerate: any cell is zero → Pearson fallback
    if n00 == 0 or n01 == 0 or n10 == 0 or n11 == 0:
        # Phi coefficient (Pearson on binary)
        denom = np.sqrt(float((n00 + n01) * (n10 + n11) * (n00 + n10) * (n01 + n11)))
        if denom == 0:
            return 0.0, True
        phi = (n00 * n11 - n01 * n10) / denom
        return float(np.clip(phi, -0.999, 0.999)), True

    # Thresholds from marginals
    pa = (n00 + n01) / total  # P(X=0)
    pb = (n00 + n10) / total  # P(Y=0)
    thresh_a = norm.ppf(pa)
    thresh_b = norm.ppf(pb)

    # Observed proportions
    props = np.array([n00, n01, n10, n11], dtype=np.float64) / total

    def neg_loglik(rho: float) -> float:
        p00, p01, p10, p11 = _bivariate_normal_prob(thresh_a, thresh_b, rho)
        # Clip to avoid log(0)
        probs = np.array([p00, p01, p10, p11])
        probs = np.clip(probs, 1e-12, 1.0)
        return -float(np.sum(props * np.log(probs)))

    result = minimize_scalar(neg_loglik, bounds=(-0.999, 0.999), method="bounded")
    return float(np.clip(result.x, -0.999, 0.999)), False


def tetrachoric_corr_matrix(
    vote_matrix: NDArray[np.float64],
    max_workers: int = 6,
) -> TetrachoricResult:
    """Compute tetrachoric correlation matrix for a binary vote matrix.

    Parameters:
        vote_matrix: n_legislators × n_bills matrix. 1=Yea, 0=Nay, NaN=absent.
        max_workers: Thread pool size for parallel pair computation.

    Returns:
        TetrachoricResult with the correlation matrix and fallback info.
    """
    n_items = vote_matrix.shape[1]
    corr = np.eye(n_items, dtype=np.float64)
    fallback = np.zeros((n_items, n_items), dtype=bool)

    # Precompute 2×2 tables for all pairs
    pairs: list[tuple[int, int]] = []
    for i in range(n_items):
        for j in range(i + 1, n_items):
            pairs.append((i, j))

    def compute_pair(pair: tuple[int, int]) -> tuple[int, int, float, bool]:
        i, j = pair
        col_i = vote_matrix[:, i]
        col_j = vote_matrix[:, j]
        # Drop rows where either is NaN
        valid = ~(np.isnan(col_i) | np.isnan(col_j))
        ci = col_i[valid]
        cj = col_j[valid]
        if len(ci) < 5:
            return i, j, 0.0, True
        # Build 2×2 table
        n11 = int(np.sum((ci == 1) & (cj == 1)))
        n10 = int(np.sum((ci == 1) & (cj == 0)))
        n01 = int(np.sum((ci == 0) & (cj == 1)))
        n00 = int(np.sum((ci == 0) & (cj == 0)))
        rho, fb = _tetrachoric_pair(n00, n01, n10, n11)
        return i, j, rho, fb

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(compute_pair, pairs)

    n_fallback = 0
    for i, j, rho, fb in results:
        corr[i, j] = rho
        corr[j, i] = rho
        fallback[i, j] = fb
        fallback[j, i] = fb
        if fb:
            n_fallback += 1

    return TetrachoricResult(
        corr_matrix=corr,
        fallback_mask=fallback,
        n_pairs=len(pairs),
        n_fallback=n_fallback,
    )
