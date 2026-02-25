# ADR-0026: EDA Literature-Backed Diagnostics

**Date:** 2026-02-24
**Status:** Accepted

## Context

A literature review (`docs/eda-deep-dive.md`) compared our EDA implementation against the political science canon (Poole-Rosenthal, Clinton-Jackman-Rivers, Desposato, Carey, Rosas-Shomer). The review confirmed all existing implementations are correct but identified five diagnostics that the literature recommends and we lacked.

The key question was scope: which diagnostics belong in EDA (Phase 1) vs downstream phases (indices, IRT, PCA)?

## Decision

Add five purely descriptive diagnostics to EDA. All are additive — they produce new outputs but do not modify the vote matrix, filtering, or any existing data that downstream phases consume.

| Function | Reference | Rationale for EDA Placement |
|----------|-----------|---------------------------|
| `compute_party_unity_scores()` | Carey Legislative Voting Data Project | Descriptive baseline; indices phase computes formal scores but EDA should provide the reference |
| `compute_eigenvalue_preview()` | Standard PCA pre-check | Early dimensionality signal before PCA (Phase 2); would have caught IRT convergence issues earlier |
| `compute_strategic_absence()` | Rosas & Shomer 2008 | Tests EDA design doc Assumption #3 ("absences are uninformative"); purely informational |
| `compute_desposato_rice_correction()` | Desposato 2005 | Corrects known bias in cross-party Rice comparison; bootstrap resampling (100 iterations) |
| `compute_item_total_correlations()` | Classical psychometrics | Flags non-discriminating roll calls; informational only (does not filter them) |

Scope boundary: these diagnostics **describe and flag** but do not **filter or transform**. Item-total correlations flag low-correlation votes but do not remove them — that decision belongs to the IRT phase if desired.

Three constants were added: `RICE_BOOTSTRAP_ITERATIONS = 100`, `STRATEGIC_ABSENCE_RATIO = 2.0`, `ITEM_TOTAL_CORRELATION_THRESHOLD = 0.1`.

## Consequences

**Benefits:**
- EDA report is now a complete pre-modeling diagnostic suite matching published state-legislature analyses
- Eigenvalue preview provides early warning for dimensionality issues before investing in MCMC
- Strategic absence diagnostic directly tests a documented assumption
- Desposato correction enables unbiased cross-party Rice comparison (Kansas Democrats at ~28% seats)
- 11 new tests (777 → 788) cover all new functions plus previously untested integrity checks

**Costs:**
- EDA runtime increases slightly (~2-3s for eigenvalue + bootstrap on M3 Pro)
- `eda.py` grows from ~1,710 to ~2,120 lines; `eda_report.py` from ~590 to ~730 lines
- Five new parquet outputs per chamber in the results directory

**Bug found during testing:**
- `_check_near_duplicate_rollcalls()` assumed unique `(legislator_slug, vote_id)` pairs for Polars pivot. Duplicate records caused a crash. Fixed by adding `.unique()` before pivot — a latent production bug.

**No downstream impact:** All five diagnostics are read-only. Existing vote matrices, filtering manifests, and agreement matrices are unchanged. No downstream phase needs modification.
