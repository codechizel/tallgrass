"""Core Exploratory Graph Analysis (EGA) orchestrator.

Chains tetrachoric correlation → GLASSO network estimation →
community detection to estimate dimensionality from binary vote data.

Usage:
    from analysis.ega import run_ega

    result = run_ega(vote_matrix)
    print(f"Estimated dimensions: {result.n_communities}")
    print(f"Unidimensional: {result.unidimensional}")

References:
    Golino, H., & Epskamp, S. (2017). Exploratory graph analysis: A new
    approach for estimating the number of dimensions in psychological
    research. PLoS ONE, 12(6), e0174035.

    Golino, H., Shi, D., Christensen, A. P., et al. (2020). Investigating
    the performance of EGA. Psychological Methods, 25(3), 292-320.
"""

from dataclasses import dataclass

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from analysis.ega.community import CommunityResult, detect_communities
from analysis.ega.glasso import GLASSOResult, glasso_ebic
from analysis.ega.tetrachoric import TetrachoricResult, tetrachoric_corr_matrix


@dataclass(frozen=True)
class EGAResult:
    """Complete result of Exploratory Graph Analysis.

    Attributes:
        n_communities: Estimated number of dimensions (K).
        community_assignments: Length-p array of dimension labels (0-indexed).
        unidimensional: True if the unidimensional check concluded K=1.
        network: igraph Graph of the GLASSO partial correlation network.
        tetrachoric: TetrachoricResult (correlation matrix + fallback info).
        glasso: GLASSOResult (partial correlations + EBIC curve).
        community: CommunityResult (assignments + modularity).
        network_loadings: p × K matrix of network loadings per community.
    """

    n_communities: int
    community_assignments: NDArray[np.int64]
    unidimensional: bool
    network: ig.Graph
    tetrachoric: TetrachoricResult
    glasso: GLASSOResult
    community: CommunityResult
    network_loadings: NDArray[np.float64]


def _compute_network_loadings(
    partial_corr: NDArray[np.float64],
    assignments: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute network loadings (Christensen & Golino, 2021).

    For each item i and community c, the network loading is the sum of
    absolute partial correlations between item i and all other items
    assigned to community c, normalized by the total absolute partial
    correlation of item i.

    Returns p × K matrix of loadings (rows sum to ~1 for well-fitting items).
    """
    p = partial_corr.shape[0]
    communities = sorted(set(assignments))
    k = len(communities)
    loadings = np.zeros((p, k), dtype=np.float64)

    for i in range(p):
        total = np.sum(np.abs(partial_corr[i, :])) - abs(partial_corr[i, i])
        if total < 1e-10:
            continue
        for c_idx, c in enumerate(communities):
            members = np.where(assignments == c)[0]
            # Exclude self
            members = members[members != i]
            if len(members) == 0:
                continue
            loadings[i, c_idx] = np.sum(np.abs(partial_corr[i, members]))

    return loadings


def _build_graph(partial_corr: NDArray[np.float64]) -> ig.Graph:
    """Build igraph Graph from partial correlation matrix."""
    p = partial_corr.shape[0]
    g = ig.Graph(n=p)
    edges = []
    weights = []
    for i in range(p):
        for j in range(i + 1, p):
            w = abs(partial_corr[i, j])
            if w > 1e-10:
                edges.append((i, j))
                weights.append(w)
    g.add_edges(edges)
    if weights:
        g.es["weight"] = weights
    return g


def run_ega(
    vote_matrix: NDArray[np.float64],
    method: str = "glasso",
    algorithm: str = "walktrap",
    gamma: float = 0.5,
    max_workers: int = 6,
) -> EGAResult:
    """Run Exploratory Graph Analysis on a binary vote matrix.

    Parameters:
        vote_matrix: n_legislators × n_bills matrix. 1=Yea, 0=Nay, NaN=absent.
        method: Network estimation method. Currently only "glasso" supported.
        algorithm: Community detection algorithm: "walktrap" (default) or "leiden".
        gamma: EBIC hyperparameter for GLASSO sparsity. Default 0.5.
        max_workers: Thread pool size for tetrachoric computation.

    Returns:
        EGAResult with dimensionality estimate, network, and loadings.
    """
    n_obs = vote_matrix.shape[0]

    # Step 1: Tetrachoric correlations
    tet_result = tetrachoric_corr_matrix(vote_matrix, max_workers=max_workers)

    # Ensure correlation matrix is positive semi-definite
    corr = tet_result.corr_matrix.copy()
    eigvals, eigvecs = np.linalg.eigh(corr)
    if np.any(eigvals < 0):
        eigvals = np.maximum(eigvals, 1e-8)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Re-normalize to unit diagonal
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)

    # Step 2: GLASSO network estimation
    if method != "glasso":
        msg = f"Unknown method: {method!r}. Currently only 'glasso' is supported."
        raise ValueError(msg)
    glasso_result = glasso_ebic(corr, n_obs=n_obs, gamma=gamma)

    # Step 3: Community detection
    comm_result = detect_communities(
        glasso_result.partial_corr,
        corr_matrix=corr,
        algorithm=algorithm,
    )

    # Step 4: Network loadings
    loadings = _compute_network_loadings(glasso_result.partial_corr, comm_result.assignments)

    # Build graph for visualization
    graph = _build_graph(glasso_result.partial_corr)

    return EGAResult(
        n_communities=comm_result.n_communities,
        community_assignments=comm_result.assignments,
        unidimensional=comm_result.unidimensional,
        network=graph,
        tetrachoric=TetrachoricResult(
            corr_matrix=corr,
            fallback_mask=tet_result.fallback_mask,
            n_pairs=tet_result.n_pairs,
            n_fallback=tet_result.n_fallback,
        ),
        glasso=glasso_result,
        community=comm_result,
        network_loadings=loadings,
    )
