"""Unique Variable Analysis (UVA) for detecting redundant items.

UVA identifies locally dependent (redundant) variable pairs by computing
weighted topological overlap (wTO) — a measure of how much two items
share the same network neighbors with similar edge weights. High wTO
indicates the items are structurally equivalent (redundant).

For legislative voting data, this detects:
- Procedural vote sequences (2nd reading, 3rd reading, final action)
- Amendment cascades (votes on amendments to the same bill)
- Near-duplicate party-line votes

References:
    Christensen, A. P., Garrido, L. E., & Golino, H. (2023). Unique
    variable analysis: A network psychometrics method to detect local
    dependence. Multivariate Behavioral Research, 58(6).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class UVAResult:
    """Result of Unique Variable Analysis.

    Attributes:
        wto_matrix: p × p weighted topological overlap matrix.
        redundant_pairs: List of (item_i, item_j, wTO) for pairs exceeding threshold.
        suggested_removals: Set of item indices recommended for removal
            (from each redundant pair, keep the one with lower max wTO to others).
        threshold: The wTO threshold used.
    """

    wto_matrix: NDArray[np.float64]
    redundant_pairs: list[tuple[int, int, float]]
    suggested_removals: set[int]
    threshold: float


def _weighted_topological_overlap(
    adj_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute weighted topological overlap (wTO) matrix.

    wTO(i,j) measures how much nodes i and j share common neighbors
    with similar edge weights:

        wTO(i,j) = (sum_k |a_ik * a_jk| + |a_ij|) /
                   (min(s_i, s_j) - |a_ij| + 1)

    where a_ij = adjacency weight, s_i = sum of |a_ik| for k != i
    (node strength excluding the pair edge).

    Returns p × p symmetric wTO matrix with diagonal = 0.
    """
    p = adj_matrix.shape[0]
    abs_adj = np.abs(adj_matrix)
    np.fill_diagonal(abs_adj, 0.0)

    # Node strengths
    strengths = np.sum(abs_adj, axis=1)

    wto = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(i + 1, p):
            # Shared neighbor contribution
            numerator = np.sum(abs_adj[i, :] * abs_adj[j, :]) + abs_adj[i, j]
            # Denominator: min strength (excluding the pair edge) + 1
            si = strengths[i] - abs_adj[i, j]
            sj = strengths[j] - abs_adj[i, j]
            denom = min(si, sj) + 1.0
            if denom > 1e-10:
                wto[i, j] = numerator / denom
                wto[j, i] = wto[i, j]

    return wto


def run_uva(
    partial_corr: NDArray[np.float64],
    threshold: float = 0.25,
) -> UVAResult:
    """Run Unique Variable Analysis on a partial correlation network.

    Parameters:
        partial_corr: p × p sparse partial correlation matrix (from GLASSO).
        threshold: wTO threshold for flagging redundant pairs.
            0.20 = small-to-moderate, 0.25 = moderate-to-large (default),
            0.30 = large-to-very-large.

    Returns:
        UVAResult with redundancy analysis.
    """
    wto = _weighted_topological_overlap(partial_corr)
    p = partial_corr.shape[0]

    # Find redundant pairs
    redundant_pairs: list[tuple[int, int, float]] = []
    for i in range(p):
        for j in range(i + 1, p):
            if wto[i, j] > threshold:
                redundant_pairs.append((i, j, float(wto[i, j])))

    # Sort by wTO descending
    redundant_pairs.sort(key=lambda x: x[2], reverse=True)

    # Suggest removals: for each pair, remove the item with higher max wTO
    # to other items (it's more redundant overall)
    max_wto = np.max(wto, axis=1)
    suggested_removals: set[int] = set()
    for i, j, _ in redundant_pairs:
        if i in suggested_removals or j in suggested_removals:
            continue  # Already removing one of the pair
        if max_wto[i] >= max_wto[j]:
            suggested_removals.add(i)
        else:
            suggested_removals.add(j)

    return UVAResult(
        wto_matrix=wto,
        redundant_pairs=redundant_pairs,
        suggested_removals=suggested_removals,
        threshold=threshold,
    )
