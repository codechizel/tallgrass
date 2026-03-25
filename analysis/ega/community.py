"""Community detection for EGA networks.

Wraps igraph's Walktrap and Leiden algorithms for identifying
communities (dimensions) in the GLASSO partial correlation network.

Includes Golino's unidimensional check: after community detection
finds K >= 2, test whether Louvain on the zero-order correlation
matrix finds K=1. If so, the data may be unidimensional despite
apparent multidimensionality in the partial correlation network.

References:
    Pons, P., & Latapy, M. (2006). Computing communities in large
    networks using random walks. JGAA, 10(2), 191-218.

    Golino et al. (2020). Investigating the performance of EGA and
    traditional techniques. Psychological Methods, 25(3), 292-320.
"""

from dataclasses import dataclass

import igraph as ig
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CommunityResult:
    """Result of community detection on an EGA network.

    Attributes:
        assignments: Length-p array of community labels (0-indexed).
        n_communities: Number of detected communities (K).
        modularity: Modularity of the partition.
        unidimensional: True if the unidimensional check overrode
            the community detection result.
        algorithm: Algorithm used ("walktrap" or "leiden").
    """

    assignments: NDArray[np.int64]
    n_communities: int
    modularity: float
    unidimensional: bool
    algorithm: str


def _partial_corr_to_graph(partial_corr: NDArray[np.float64]) -> ig.Graph:
    """Build an igraph Graph from a partial correlation matrix.

    Only non-zero edges (|partial_corr| > 1e-10) are included.
    Edge weights are absolute partial correlations.
    """
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
    g.es["weight"] = weights
    return g


def _walktrap(graph: ig.Graph, steps: int = 4) -> ig.VertexClustering:
    """Run Walktrap community detection (Golino's default)."""
    dendro = graph.community_walktrap(weights="weight", steps=steps)
    return dendro.as_clustering()


def _leiden(graph: ig.Graph) -> ig.VertexClustering:
    """Run Leiden community detection (modularity optimization)."""
    import leidenalg

    return leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        seed=42,
    )


def _unidimensional_check(
    corr_matrix: NDArray[np.float64],
) -> bool:
    """Golino's unidimensional check: Louvain on the zero-order correlation matrix.

    If Louvain finds K=1 on the full (non-regularized) correlation matrix,
    the data is likely unidimensional even if the GLASSO network has K >= 2.

    Returns True if the data appears unidimensional.
    """
    p = corr_matrix.shape[0]
    g = ig.Graph(n=p)

    edges = []
    weights = []
    for i in range(p):
        for j in range(i + 1, p):
            w = abs(corr_matrix[i, j])
            if w > 0.01:  # Small threshold to exclude near-zero correlations
                edges.append((i, j))
                weights.append(w)

    if not edges:
        return True  # No edges → unidimensional

    g.add_edges(edges)
    g.es["weight"] = weights

    # Louvain community detection
    clustering = g.community_multilevel(weights="weight")
    return len(clustering) == 1


def detect_communities(
    partial_corr: NDArray[np.float64],
    corr_matrix: NDArray[np.float64] | None = None,
    algorithm: str = "walktrap",
    check_unidimensional: bool = True,
) -> CommunityResult:
    """Detect communities (dimensions) in a GLASSO partial correlation network.

    Parameters:
        partial_corr: p × p sparse partial correlation matrix from GLASSO.
        corr_matrix: p × p zero-order correlation matrix for unidimensional check.
            Required if check_unidimensional=True.
        algorithm: "walktrap" (default, Golino's recommendation) or "leiden".
        check_unidimensional: If True and K >= 2, run the unidimensional check.

    Returns:
        CommunityResult with community assignments and metadata.
    """
    graph = _partial_corr_to_graph(partial_corr)
    p = partial_corr.shape[0]

    # Handle disconnected graph (isolated nodes with no edges)
    if graph.ecount() == 0:
        return CommunityResult(
            assignments=np.zeros(p, dtype=np.int64),
            n_communities=1,
            modularity=0.0,
            unidimensional=True,
            algorithm=algorithm,
        )

    if algorithm == "walktrap":
        clustering = _walktrap(graph)
    elif algorithm == "leiden":
        clustering = _leiden(graph)
    else:
        msg = f"Unknown algorithm: {algorithm!r}. Use 'walktrap' or 'leiden'."
        raise ValueError(msg)

    assignments = np.array(clustering.membership, dtype=np.int64)
    k = len(set(clustering.membership))
    modularity = float(clustering.modularity)

    # Unidimensional check
    uni = False
    if check_unidimensional and k >= 2 and corr_matrix is not None:
        uni = _unidimensional_check(corr_matrix)
        if uni:
            assignments = np.zeros(p, dtype=np.int64)
            k = 1

    return CommunityResult(
        assignments=assignments,
        n_communities=k,
        modularity=modularity,
        unidimensional=uni,
        algorithm=algorithm,
    )
