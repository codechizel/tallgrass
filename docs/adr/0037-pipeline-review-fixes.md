# ADR-0037: Pipeline Review Fixes

**Date:** 2026-02-25
**Status:** Accepted

## Context

A full pipeline code review traced data flow through all 14 analysis phases after recent refactoring and deep-dive work (ADR-0030 through ADR-0036). The review found no cross-phase schema mismatches — column names, join keys, and parquet schemas are consistent throughout — but identified 11 issues ranging from latent crashes to misleading report output.

## Decision

Fix all issues in a single commit. The fixes fall into four categories:

### 1. Exception syntax clarity (5 locations)

PEP 758 (Python 3.14) allows `except ValueError, ZeroDivisionError:` without parentheses. However, this syntax is visually identical to Python 2's `except X, Y:` (which meant `except X as Y:`). Parenthesized form is universally understood.

**Changed:** `run_context.py:119`, `network.py:304,451,524` — added parentheses.

### 2. RunContext failure safety (2 fixes)

- `_append_missing_votes` crashed with `ValueError` when `total_vote_pages` was missing from the failure manifest (the `"?"` string default can't be formatted with `,`). Fixed with `isinstance(total, int)` guard.
- `finalize()` updated the `latest` symlink unconditionally, including on failed runs. Downstream phases following `latest` would see partial/broken results. Fixed by accepting a `failed` parameter from `__exit__` and skipping the symlink update when `True`.

### 3. Data correctness (4 fixes)

- **EDA `vote_date` sort:** The "Present and Passing" detail table sorted on `vote_date` (MM/DD/YYYY string), which orders lexicographically, not chronologically. Changed to sort on `vote_datetime` (ISO 8601).
- **Cross-session PC1 stability:** `compute_metric_stability()` correlated raw PC1 values across sessions. PCA sign is conventional (orient_pc1 normalizes Republicans-positive), but if the convention fails on edge-case data, the correlation flips to -1.0, masquerading as instability. Added `SIGN_ARBITRARY_METRICS = {"PC1"}` and `abs()` on correlations for those metrics.
- **Clustering label alignment:** `plot_irt_loyalty_clusters()` used a positional array slice as fallback when mapping cluster labels from ideal_points row order to the IRT-loyalty merged DataFrame. If the merge dropped mid-array rows, labels would be assigned to wrong data points. Replaced with slug-based dict lookup.
- **Synthesis hardcoded AUC:** The pipeline summary infographic fell back to `"~0.98"` when prediction holdout results were unavailable. A fabricated number in a report read by nontechnical users. Changed to `"N/A"`.

### 4. Dead code and doc accuracy (2 fixes)

- **UMAP dead branch:** `if not validation:` at line 546 was unreachable because a key was always added at line 545. Removed.
- **External validation docstring:** `_phase2_last_name_match()` docstring claimed "using district as tiebreaker" but the implementation calls `unique()`. Fixed docstring to match reality.

### 5. Pre-existing lint (4 fixes)

E501 line-too-long in `irt.py`, `hierarchical.py`, `profiles.py`, `cross_session.py`. Reformatted.

## Consequences

- `RunContext.finalize()` signature changed: now accepts `failed: bool = False`. All callers go through `__exit__`, which passes the parameter automatically. Direct callers of `finalize()` (none currently) get the old behavior by default.
- The `latest` symlink is no longer updated on failed runs. A failed run's output is still saved in its dated directory but `latest` continues pointing to the last successful run.
- Cross-session PC1 stability correlations will now always be non-negative (absolute value). This is the correct interpretation for a sign-arbitrary metric.
- All 1,096 tests pass. No behavioral changes to any analysis output for successful runs.
