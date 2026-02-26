# ADR-0039: Cross-Session Validation Enhancements

**Date:** 2026-02-25
**Status:** Accepted

## Context

A deep dive research audit of Phase 13 (cross-session validation) identified 7 recommendations from the academic literature to improve metric stability assessment, detection robustness, and matching resilience. Two were already done (XGBoost param extraction, tau asymmetry docs — ADR-0035). The remaining five are standard practices from the political science and psychometric measurement literature:

1. **PSI (Population Stability Index)** — standard concept drift metric from credit risk modeling, applicable to cross-session feature distribution comparison.
2. **ICC (Intraclass Correlation Coefficient)** — the standard psychometric measure for test-retest reliability (Koo & Li 2016), directly applicable to our "are metrics stable across sessions?" question.
3. **Percentile-based detection thresholds** — the synthesis detection functions used hardcoded magic numbers. Extracting constants and offering percentile-based alternatives improves transparency and adaptability.
4. **Fuzzy matching** — exact name matching works with 78% overlap, but name changes or transcription errors could silently reduce the match rate.
5. **Stability interpretation** — Spearman rho was computed but not explicitly interpreted for nontechnical readers.

## Decision

### 1. PSI in Metric Stability

Added `compute_psi(a, b, n_bins=10)` to `cross_session_data.py`. Quantile-bins distribution A, histograms both distributions, computes `Σ (p_b - p_a) * ln(p_b / p_a)` with epsilon floor to avoid log(0). `interpret_psi()` maps to "stable" (< 0.10), "investigate" (0.10–0.25), "significant drift" (> 0.25).

`compute_metric_stability()` output gains `psi` and `psi_interpretation` columns.

### 2. ICC in Metric Stability

Added `compute_icc(a, b)` implementing ICC(3,1) — two-way mixed, single measures, consistency. Formula: `(MS_row - MS_error) / (MS_row + (k-1) * MS_error)` where k=2. `interpret_icc()` maps to Koo & Li 2016 thresholds: poor (< 0.50), moderate (0.50–0.75), good (0.75–0.90), excellent (> 0.90).

`compute_metric_stability()` output gains `icc` and `icc_interpretation` columns.

### 3. Percentile-Based Detection

Extracted four hardcoded thresholds from `synthesis_detect.py` to module-level constants:
- `UNITY_SKIP_THRESHOLD = 0.95`
- `BRIDGE_SD_TOLERANCE = 1.0`
- `PARADOX_RANK_GAP = 0.5`
- `PARADOX_MIN_PARTY_SIZE = 5`

Added optional parameters:
- `detect_chamber_maverick(..., percentile=0.10)` — select bottom 10% by unity score instead of threshold-based skip.
- `detect_metric_paradox(..., rank_gap_percentile=0.50)` — use data-driven gap threshold instead of fixed 0.5.

Both default to `None` (existing behavior). Additive — no callers need changes.

### 4. Fuzzy Matching Fallback

Added `fuzzy_match_legislators(unmatched_a, unmatched_b, threshold=0.85)` using `difflib.SequenceMatcher` (stdlib). Returns a DataFrame of suggested matches with similarity scores.

`match_legislators()` gains `fuzzy_threshold: float | None = None`. When set, unmatched names from the exact-match pass go through a second fuzzy pass. No new dependency — stdlib only.

### 5. Stability Interpretation

Added `interpret_stability(rho)` mapping abs(Spearman rho) to Koo & Li 2016 thresholds. `compute_metric_stability()` output gains `stability_interpretation` column. The report table now shows ICC and Reliability columns.

### Tests (29 new, 1096 → 1125)

- `TestComputePSI` (5): identical, shifted, empty, non-negative, interpretation
- `TestComputeICC` (5): perfect, no agreement, known value, too few, interpretation
- `TestFuzzyMatchLegislators` (5): typo, threshold, empty, no false positives, best match
- `TestMatchLegislatorsFuzzy` (2): extra match, None unchanged
- `TestStabilityInterpretation` (4): thresholds, negative rho, output column, empty schema
- `TestThresholdConstants` (4): all constants match hardcoded values
- `TestPercentileMaverick` (2): percentile mode, default unchanged
- `TestPercentileParadox` (2): percentile mode, low percentile permissive

Updated 1 existing test (`test_empty_result_schema`) for new columns.
Updated report test stability DataFrame to include new columns.

## Consequences

**Benefits:**
- PSI provides an intuitive single-number drift metric for each stability metric — complements correlation with distributional information
- ICC is the field-standard reliability measure, making our stability table directly comparable to published psychometric research
- Extracted constants make detection thresholds transparent and documentable
- Percentile-based alternatives enable adaptive detection that doesn't depend on absolute scale
- Fuzzy matching adds resilience for edge cases without changing the default exact-match behavior
- Stability interpretation helps nontechnical readers interpret the stability table

**Trade-offs:**
- Five new columns in the stability DataFrame increase table width. The report selectively shows ICC and Reliability; PSI is available in the raw parquet for analysts.
- Fuzzy matching is O(n*m) but only runs on the (small) unmatched subset. No performance concern.
- `numpy` import added to `synthesis_detect.py` (previously pure polars). Acceptable — numpy is already a transitive dependency.
