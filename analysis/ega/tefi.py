"""Total Entropy Fit Index (TEFI) for dimensionality assessment.

TEFI applies Von Neumann entropy from quantum information theory to
the correlation matrix, evaluated relative to a proposed dimensional
structure. Lower TEFI = better fit.

Key property: TEFI properly penalizes over-extraction — when more
dimensions are specified than truly exist, TEFI increases. Traditional
fit measures (RMSEA, CFI) often fail to detect over-extraction.

References:
    Golino, H., Moulder, R., Shi, D., et al. (2021). Entropy fit indices:
    New fit measures for assessing the structure and dimensionality of
    multiple latent variables. Multivariate Behavioral Research, 56(6).
"""

import numpy as np
from numpy.typing import NDArray


def _von_neumann_entropy(matrix: NDArray[np.float64]) -> float:
    """Compute Von Neumann entropy of a density matrix.

    S(rho) = -Tr(rho * log2(rho)) = -sum(lambda_i * log2(lambda_i))
    where lambda_i are eigenvalues of the density matrix.

    The input matrix is normalized to unit trace (density matrix convention).
    """
    # Normalize to unit trace
    tr = np.trace(matrix)
    if tr < 1e-10:
        return 0.0
    rho = matrix / tr

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(rho)
    # Filter out near-zero and negative eigenvalues
    eigvals = eigvals[eigvals > 1e-15]

    if len(eigvals) == 0:
        return 0.0

    return -float(np.sum(eigvals * np.log2(eigvals)))


def compute_tefi(
    corr_matrix: NDArray[np.float64],
    assignments: NDArray[np.int64],
) -> float:
    """Compute the Total Entropy Fit Index (TEFI.vn).

    TEFI sums the Von Neumann entropy of each within-community
    correlation sub-matrix, then subtracts the entropy of the
    full matrix. Lower TEFI = better structural fit.

    TEFI = sum_c(VN_entropy(R_c)) - VN_entropy(R)

    where R_c is the correlation sub-matrix for community c
    and R is the full correlation matrix.

    Parameters:
        corr_matrix: p × p symmetric correlation matrix.
        assignments: Length-p array of community labels (integer).

    Returns:
        TEFI value. Lower is better. Can be negative (good fit).
    """
    # Full matrix entropy
    full_entropy = _von_neumann_entropy(corr_matrix)

    # Per-community entropy
    communities = sorted(set(assignments))
    community_entropy_sum = 0.0

    for c in communities:
        members = np.where(assignments == c)[0]
        if len(members) < 2:
            # Single-item community contributes zero entropy
            continue
        sub_matrix = corr_matrix[np.ix_(members, members)]
        community_entropy_sum += _von_neumann_entropy(sub_matrix)

    return community_entropy_sum - full_entropy


def compare_structures(
    corr_matrix: NDArray[np.float64],
    max_k: int = 5,
    assignments_list: list[NDArray[np.int64]] | None = None,
) -> dict[int, float]:
    """Compare TEFI across multiple K values.

    If assignments_list is not provided, generates naive assignments by
    splitting items into K equal groups (for comparison purposes only;
    real assignments should come from EGA or PCA).

    Parameters:
        corr_matrix: p × p symmetric correlation matrix.
        max_k: Maximum K to evaluate.
        assignments_list: Optional list of assignment arrays, one per K.
            If provided, len(assignments_list) determines max_k.

    Returns:
        Dict mapping K → TEFI value. Lowest TEFI indicates best K.
    """
    results: dict[int, float] = {}
    p = corr_matrix.shape[0]

    if assignments_list is not None:
        for k_idx, assignments in enumerate(assignments_list):
            k = len(set(assignments))
            results[k] = compute_tefi(corr_matrix, assignments)
    else:
        for k in range(1, max_k + 1):
            # Naive equal-split assignments
            assignments = np.array([i % k for i in range(p)], dtype=np.int64)
            results[k] = compute_tefi(corr_matrix, assignments)

    return results
