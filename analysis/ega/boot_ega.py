"""Bootstrap Exploratory Graph Analysis (bootEGA).

Assesses the stability of EGA's dimensionality estimate by running
EGA on B bootstrap replicates and computing:
- Dimension frequency: how often each K was found.
- Item stability: proportion of replicates assigning each item to
  its empirical community.
- Structural consistency: how often each community is exactly replicated.

References:
    Christensen, A. P., & Golino, H. (2021). Estimating the stability
    of psychological dimensions via bootstrap exploratory graph analysis.
    Psych, 3(3), 479-500.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis.ega.ega import EGAResult, run_ega


@dataclass(frozen=True)
class BootEGAResult:
    """Result of bootstrap EGA stability assessment.

    Attributes:
        empirical: The EGA result on the original data.
        n_boot: Number of bootstrap replicates completed.
        dimension_frequency: Dict mapping K → count of replicates finding K dimensions.
        modal_k: Most frequent K across replicates.
        median_k: Median K across replicates.
        item_stability: Length-p array; proportion of replicates assigning
            each item to its empirical community.
        structural_consistency: Dict mapping community label → proportion of
            replicates where that community is exactly replicated.
        boot_assignments: n_boot × p matrix of community assignments per replicate.
    """

    empirical: EGAResult
    n_boot: int
    dimension_frequency: dict[int, int]
    modal_k: int
    median_k: float
    item_stability: NDArray[np.float64]
    structural_consistency: dict[int, float]
    boot_assignments: NDArray[np.int64]


def _parametric_resample(
    corr_matrix: NDArray[np.float64],
    n_obs: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate a parametric bootstrap sample from a correlation matrix.

    Draws from multivariate normal with the given correlation structure,
    then thresholds to binary (>0 → 1, ≤0 → 0).
    """
    p = corr_matrix.shape[0]
    # Ensure PSD
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    eigvals = np.maximum(eigvals, 1e-8)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    z = rng.multivariate_normal(np.zeros(p), cov, size=n_obs)
    return (z > 0).astype(np.float64)


def _nonparametric_resample(
    vote_matrix: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate a non-parametric bootstrap sample by resampling rows."""
    n = vote_matrix.shape[0]
    idx = rng.integers(0, n, size=n)
    return vote_matrix[idx, :]


def _match_communities(
    empirical: NDArray[np.int64],
    bootstrap: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Relabel bootstrap communities to best match empirical communities.

    Uses a greedy maximum-overlap matching.
    """
    emp_labels = sorted(set(empirical))
    boot_labels = sorted(set(bootstrap))

    # Build overlap matrix
    overlap = np.zeros((len(emp_labels), len(boot_labels)), dtype=int)
    for ei, el in enumerate(emp_labels):
        for bi, bl in enumerate(boot_labels):
            overlap[ei, bi] = int(np.sum((empirical == el) & (bootstrap == bl)))

    # Greedy matching: assign each bootstrap label to its best empirical match
    relabeled = bootstrap.copy()
    used_emp: set[int] = set()
    mapping: dict[int, int] = {}

    for _ in range(min(len(emp_labels), len(boot_labels))):
        best = np.unravel_index(np.argmax(overlap), overlap.shape)
        ei, bi = int(best[0]), int(best[1])
        mapping[boot_labels[bi]] = emp_labels[ei]
        used_emp.add(emp_labels[ei])
        overlap[ei, :] = -1
        overlap[:, bi] = -1

    # Unmapped bootstrap labels get next available integer
    next_label = max(emp_labels) + 1 if emp_labels else 0
    for bl in boot_labels:
        if bl not in mapping:
            mapping[bl] = next_label
            next_label += 1

    for bl, el in mapping.items():
        relabeled[bootstrap == bl] = el

    return relabeled


def run_boot_ega(
    vote_matrix: NDArray[np.float64],
    n_boot: int = 500,
    method: str = "parametric",
    algorithm: str = "walktrap",
    gamma: float = 0.5,
    max_workers: int = 6,
    seed: int = 42,
) -> BootEGAResult:
    """Run bootstrap EGA for stability assessment.

    Parameters:
        vote_matrix: n_legislators × n_bills matrix. 1=Yea, 0=Nay, NaN=absent.
        n_boot: Number of bootstrap replicates. Default 500 (Golino's recommendation).
        method: "parametric" (generate from correlation matrix) or "nonparametric"
            (resample rows with replacement).
        algorithm: Community detection algorithm for each replicate.
        gamma: EBIC hyperparameter for GLASSO.
        max_workers: Thread pool size for tetrachoric computation within each replicate.
        seed: Random seed for reproducibility.

    Returns:
        BootEGAResult with stability metrics.
    """
    rng = np.random.default_rng(seed)

    # Run empirical EGA
    empirical = run_ega(vote_matrix, algorithm=algorithm, gamma=gamma, max_workers=max_workers)
    n_obs = vote_matrix.shape[0]
    p = vote_matrix.shape[1]

    # Bootstrap
    k_counts: dict[int, int] = {}
    all_assignments = np.zeros((n_boot, p), dtype=np.int64)
    completed = 0

    for b in range(n_boot):
        try:
            if method == "parametric":
                sample = _parametric_resample(empirical.tetrachoric.corr_matrix, n_obs, rng)
            elif method == "nonparametric":
                sample = _nonparametric_resample(vote_matrix, rng)
            else:
                msg = f"Unknown method: {method!r}. Use 'parametric' or 'nonparametric'."
                raise ValueError(msg)

            # Check for degenerate columns (all same value)
            col_var = np.nanvar(sample, axis=0)
            good_cols = col_var > 1e-10
            if np.sum(good_cols) < 3:
                continue

            boot_result = run_ega(
                sample[:, good_cols],
                algorithm=algorithm,
                gamma=gamma,
                max_workers=max_workers,
            )

            k = boot_result.n_communities
            k_counts[k] = k_counts.get(k, 0) + 1

            # Map assignments back to full item set
            full_assignments = np.full(p, -1, dtype=np.int64)
            good_idx = np.where(good_cols)[0]
            for idx, assign in zip(good_idx, boot_result.community_assignments):
                full_assignments[idx] = assign

            # Match communities to empirical labeling
            valid = full_assignments >= 0
            if np.any(valid):
                matched = full_assignments.copy()
                matched[valid] = _match_communities(
                    empirical.community_assignments[valid],
                    full_assignments[valid],
                )
                all_assignments[completed] = matched
            else:
                all_assignments[completed] = full_assignments

            completed += 1

        except Exception:
            # Skip failed replicates (degenerate GLASSO, etc.)
            continue

    all_assignments = all_assignments[:completed]

    # Item stability: proportion of replicates matching empirical assignment
    item_stability = np.zeros(p, dtype=np.float64)
    if completed > 0:
        for i in range(p):
            emp_label = empirical.community_assignments[i]
            valid_boots = all_assignments[:, i] >= 0
            if np.sum(valid_boots) > 0:
                item_stability[i] = float(np.mean(all_assignments[valid_boots, i] == emp_label))

    # Structural consistency: per-community exact replication rate
    structural_consistency: dict[int, float] = {}
    emp_labels = sorted(set(empirical.community_assignments))
    for c in emp_labels:
        emp_members = set(np.where(empirical.community_assignments == c)[0])
        n_exact = 0
        for b in range(completed):
            boot_members = set(np.where(all_assignments[b] == c)[0])
            if boot_members == emp_members:
                n_exact += 1
        structural_consistency[int(c)] = n_exact / max(completed, 1)

    # Modal and median K
    modal_k = max(k_counts, key=k_counts.get) if k_counts else empirical.n_communities
    all_ks = []
    for k, count in k_counts.items():
        all_ks.extend([k] * count)
    median_k = float(np.median(all_ks)) if all_ks else float(empirical.n_communities)

    return BootEGAResult(
        empirical=empirical,
        n_boot=completed,
        dimension_frequency=k_counts,
        modal_k=modal_k,
        median_k=median_k,
        item_stability=item_stability,
        structural_consistency=structural_consistency,
        boot_assignments=all_assignments,
    )
