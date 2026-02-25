# ADR-0028: Clustering Deep Dive Improvements

**Date:** 2026-02-24
**Status:** Accepted

## Context

A literature review and code audit of the clustering implementation (ADR-0007, ADR-0014) identified several correctness issues, code quality improvements, and opportunities to add new clustering methods. The deep dive surveyed Python clustering libraries (scikit-learn, scipy, hdbscan), reviewed clustering methodology in the political science literature, and audited every function in `analysis/clustering.py`.

Full analysis: `docs/clustering-deep-dive.md` and `docs/ward-linkage-non-euclidean.md`.

## Decision

### 1. Ward → average linkage (correctness)

Ward linkage requires Euclidean distance. Our Kappa-derived distance matrix (1 - Kappa) is not guaranteed to be Euclidean — it may violate the triangle inequality for legislator pairs with sufficiently different marginal distributions. scipy's documentation explicitly warns about this. Switched `LINKAGE_METHOD` from `"ward"` to `"average"`, which is valid for any distance metric and standard in the political science literature for agreement-based hierarchical clustering.

### 2. Extracted `_kappa_to_distance()` helper (deduplication)

The Kappa-to-distance conversion (symmetrize, fill NaN, clip negatives) was duplicated between `run_hierarchical()` and `run_sensitivity_clustering()`. Extracted to a single helper function. `find_optimal_k_hierarchical()` now accepts a pre-computed distance array instead of re-computing from scratch.

### 3. Renamed `DEFAULT_K` → `COMPARISON_K`

The constant was named `DEFAULT_K` (suggesting it's the default cluster count) but actually represents a forced k=3 cut for downstream comparison after k=2 was confirmed optimal. Renamed to `COMPARISON_K` to reflect its actual purpose.

### 4. Added spectral clustering

SpectralClustering on the precomputed Kappa affinity matrix (agreement = 1 - distance) using `assign_labels="cluster_qr"` for deterministic results. Spectral clustering captures non-convex cluster structure by embedding the affinity graph's Laplacian eigenvectors, complementing the other methods.

### 5. Added HDBSCAN on PCA embeddings

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) on standardized PCA scores. Unlike k-based methods, HDBSCAN does not require specifying k and designates noise points (label=-1) for outlier legislators. Constants: `HDBSCAN_MIN_CLUSTER_SIZE=5`, `HDBSCAN_MIN_SAMPLES=3`.

### 6. Added `n_other` to cluster characterization

`characterize_clusters()` now counts Independent (non-R/non-D) legislators per cluster, addressing ADR-0021 (Independent party handling).

### 7. Extracted named constants for magic numbers

Replaced magic numbers in plot functions with named constants: `DENDROGRAM_HEIGHT_PER_LEGISLATOR`, `DENDROGRAM_TRUNCATED_HEIGHT`, `VOTING_BLOCS_HEIGHT_PER_LEGISLATOR`, `POLAR_SIZE_PER_LEGISLATOR`, `NOTABLE_LOYALTY_THRESHOLD`, `NOTABLE_XI_THRESHOLD`, `EXTREME_PERCENTILE`, `LABEL_STAGGER_RATIO`.

### 8. Updated clustering_report.py

Updated the HTML report builder to render HDBSCAN summary, updated cross-method interpretation for spectral clustering, fixed `DEFAULT_K` → `COMPARISON_K` import, added HDBSCAN constants to the parameters table.

### 9. Expanded test coverage

Added 47 new tests (23 → 70 total) covering: `_kappa_to_distance`, `_standardize_2d`, `_build_display_labels`, `run_hierarchical`, `find_optimal_k_hierarchical`, `run_spectral_clustering`, `run_hdbscan_pca`, `characterize_clusters`, `run_kmeans_irt`, and new constants.

## Consequences

- **Correctness:** Average linkage is methodologically pure for non-Euclidean distances. Does not change the k=2 finding or cross-method ARI scores.
- **Robustness:** Five independent clustering methods (hierarchical, k-means, GMM, spectral, HDBSCAN) provide stronger cross-method validation than three.
- **Outlier detection:** HDBSCAN noise labels complement synthesis-phase maverick detection.
- **Breaking changes:** `DEFAULT_K` renamed to `COMPARISON_K` — any downstream code importing `DEFAULT_K` will need updating. The `clustering_report.py` import was fixed in this change.
- **LCA not added:** Latent Class Analysis (StepMix) was evaluated but not implemented. It's the statistically principled method for binary data clustering but requires a separate dependency and is medium-high effort. Recommended for future evaluation.
