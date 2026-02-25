# ADR-0019: Cross-Session Validation

**Date:** 2026-02-22
**Status:** Accepted

## Context

The analytics pipeline has been validated within individual sessions (2023-24 and 2025-26), but never *across* sessions. This is the single biggest gap in current results:

1. **Prediction honesty.** Within-session holdout AUC (0.98) is optimistic — the model sees the same legislators and session dynamics. Cross-session tests whether learned patterns generalize.
2. **Temporal comparison.** "Who moved ideologically?" is the most newsworthy output for the nontechnical audience. It requires placing two sessions' IRT ideal points on a common scale.
3. **Detection robustness.** The synthesis detection thresholds (maverick, bridge, paradox) were calibrated on 2025-26. If they fail on 2023-24, they're overfit.

Both bienniums are fully scraped and have complete pipeline results through 10 phases (EDA, PCA, IRT, clustering, network, prediction, indices, UMAP, beta-binomial, synthesis). There are 131 overlapping legislators (78% of each session), providing strong anchor points.

### Open Source Landscape

An exhaustive search found no Python library for cross-session ideal point bridging. The R ecosystem has tools (ipbridging, emIRT, idealstan, DW-NOMINATE common space), but none have Python ports. The closest Python IRT libraries (py-irt, GIRTH) don't support linking or equating.

Everything needed is already in the stack: PyMC for posteriors, ArviZ for diagnostics, SciPy for distribution comparison (Wasserstein distance, KS test), polars for data wrangling, scikit-learn/XGBoost for prediction.

## Decision

Build a cross-session validation phase from scratch using existing dependencies. No new packages.

**Four sub-analyses:**

### 1. Ideology Stability (Anchor-Based Scale Alignment)

Use the same affine transformation approach already proven for cross-chamber equating (see `analysis/design/irt.md`, "Cross-Chamber Equating"). With 131 overlapping legislators instead of 3 bridging legislators, the transformation is far more robust.

- Fit IRT separately per session (already done).
- Match returning legislators by normalized `full_name`.
- Compute affine transform: `xi_aligned = A * xi_session_a + B` using robust estimation (median-based, trimming outliers).
- Report Pearson/Spearman correlation, identify significant movers (>1 SD shift).
- Use SciPy `wasserstein_distance` on posterior samples for Bayesian shift quantification.
- Classify turnover cohorts: returning, departing, new — compare ideology distributions.

**Why affine transformation over concurrent calibration (stacking vote matrices):** Concurrent calibration would require fitting a single IRT model on both sessions simultaneously — doubling the model size and requiring careful parameter sharing for overlapping legislators. With 131 anchors, the affine approach gives equivalent accuracy at a fraction of the complexity. The cross-chamber equating section of the IRT design doc already validates this methodology. Concurrent calibration remains available as a future upgrade if the affine approach proves insufficient.

### 2. Metric Stability

- Compute Pearson r and Spearman rho for 8 legislative metrics (party unity, maverick rate, weighted maverick, betweenness, eigenvector, pagerank, clustering loyalty, PC1) across sessions for returning legislators.
- High correlations indicate these measures capture stable traits rather than session-specific noise.
- Warn when any metric falls below r = 0.70.

### 3. Out-of-Sample Prediction

- Train XGBoost on session A's vote features, test on session B's returning legislators (and vice versa).
- Standardize features (z-score within session) before cross-session application, since IRT scales differ.
- Compare cross-session AUC to within-session AUC (0.98).
- Compare feature importance rankings (SHAP) across sessions — stable rankings indicate generalizable patterns.

### 4. Detection Threshold Validation

- Run synthesis detection (`detect_chamber_maverick`, `detect_bridge_builder`, `detect_metric_paradox`) on both sessions' legislator DataFrames.
- Compare: same roles flagged? Same threshold behavior?
- If thresholds fail on 2023-24, propose adaptive alternatives.

**Output location:** `results/kansas/cross-session/<pair>/validation/YYYY-MM-DD/` (e.g., `cross-session/90th-vs-91st/validation/2026-02-22/`). The comparison pair is encoded in the path so that multiple pairwise comparisons (89th-vs-90th, 90th-vs-91st, 89th-vs-91st) each get their own `latest` symlink. RunContext's `_normalize_session` passes the `cross-session/90th-vs-91st` session string through unchanged (it only transforms "YYYY-YY" patterns).

## Consequences

**Benefits:**
- Out-of-sample prediction provides honest AUC estimates (expected to drop from 0.98 to ~0.85-0.92)
- Ideology shift analysis produces the most compelling output for journalists ("Senator X moved 1.2 points rightward")
- Detection validation builds confidence in the synthesis report's generalizability
- Reuses the cross-chamber equating methodology already validated in this codebase
- No new dependencies

**Trade-offs:**
- Affine alignment assumes a linear relationship between session scales — nonlinear distortions aren't captured (same limitation as cross-chamber equating)
- Legislator name matching is fragile to name changes, typos, or suffix differences between sessions — needs robust normalization
- Cross-session prediction comparison is imperfect: different bills, different political context, different base rates may confound AUC differences
- This phase reads from both sessions' results directories, breaking the single-session assumption of RunContext — requires manual path construction for the second session

**Post-implementation fixes (ADR-0035):** A deep dive audit found and fixed three bugs: (1) turnover impact compared IRT values on different scales — departing cohort now affine-transformed, (2) ddof mismatch between classification and visualization thresholds, (3) prediction tested all legislators instead of returning-only as specified above. See `docs/cross-session-deep-dive.md` for the full audit.
