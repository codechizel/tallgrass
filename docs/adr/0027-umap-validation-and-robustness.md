# ADR-0027: UMAP Validation and Robustness Improvements

**Date:** 2026-02-24
**Status:** Accepted

## Context

A literature review and code audit of the UMAP implementation (ADR-0011) identified several areas where the validation and robustness methodology could be strengthened. The 2024-2025 UMAP criticism debate (Simply Statistics, arXiv:2506.08725) specifically challenged the stochastic nature of UMAP embeddings and the lack of quantitative validation metrics in many implementations. Our original implementation used Spearman correlation against PCA/IRT and Procrustes sensitivity across n_neighbors values, but lacked (a) a direct neighborhood preservation metric, (b) multi-seed stability analysis, and (c) imputation-aware cross-party outlier detection.

Additionally, the sensitivity sweep silently allowed n_neighbors values exceeding n_samples for the Senate (n=42), the plot legend did not include Independent legislators (89th biennium), and the gradient plot functions contained ~50 lines of duplication.

Full analysis: `docs/umap-deep-dive.md`.

## Decision

### 1. Trustworthiness score (sklearn)

Add `sklearn.manifold.trustworthiness()` as a quantitative validation metric. This directly measures what fraction of a point's k-nearest neighbors in the 2D embedding were also k-nearest neighbors in the original vote matrix. Clamped to `n_samples // 2 - 1` per sklearn requirements; returns NaN for datasets too small (n < 4).

### 2. Multi-seed stability analysis

Run UMAP with 5 random seeds (`STABILITY_SEEDS = [42, 123, 456, 789, 1337]`), compute pairwise Procrustes similarity. Reports mean and min similarity. This directly addresses the criticism that UMAP results depend on random initialization — high mean similarity (> 0.7) indicates the structure is real, not an artifact of a particular seed.

### 3. Imputation-aware cross-party annotation

Added `compute_imputation_pct()` to track per-legislator imputation rates. Cross-party outliers with >= 50% imputed votes (`HIGH_IMPUTATION_PCT`) are labeled "imputation artifact"; those with mostly real votes are labeled "cross-party voter" (genuine maverick). Previously, all cross-party outliers were unconditionally labeled as imputation artifacts.

### 4. Dynamic party legend

The landscape plot legend is now built from parties present in the data, not hardcoded to Republican/Democrat. This correctly handles sessions with Independent legislators (e.g., 89th biennium).

### 5. Sensitivity sweep clamping

`n_neighbors` values >= `n_samples` are skipped with a log message rather than being silently clamped by umap-learn. For the Senate (n~42), this means n_neighbors=50 is now explicitly skipped.

### 6. Refactoring

- Extracted `find_irt_column()` helper (eliminates duplicated column discovery)
- Unified `plot_umap_gradient()` function (eliminates duplicated gradient plot code)
- Simplified redundant slug column conditionals
- Fixed exception syntax: `except OSError:` (was `except FileNotFoundError, OSError:` — redundant since `FileNotFoundError` is a subclass of `OSError`)

## Consequences

**Benefits:**
- Trustworthiness provides a concrete, comparable number for "how well did UMAP preserve neighborhoods" — useful for cross-session comparisons and report tables.
- Multi-seed stability directly rebuts the methodological concern that UMAP is stochastic. If all seed pairs show > 0.7 similarity, the structure is robust.
- Imputation-aware annotation prevents mislabeling genuine mavericks as artifacts, improving report accuracy.
- Dynamic legend is correct for all sessions without manual adjustment.
- Sensitivity clamping makes the log output honest about what was tested.
- Test coverage increased from 21 to 40 tests.

**Trade-offs:**
- Multi-seed stability adds ~5x UMAP compute time per chamber to the sensitivity phase. For the 91st (House ~130 legislators), this is a few seconds per seed. Acceptable since it runs only when `--skip-sensitivity` is not set.
- `sklearn` is already a dependency (via umap-learn), so `trustworthiness` adds no new dependencies.
- New constants (`STABILITY_SEEDS`, `HIGH_IMPUTATION_PCT`) add to the parameter space that must be documented and justified.
