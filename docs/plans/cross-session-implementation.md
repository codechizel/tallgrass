# Cross-Session Validation — Implementation Plan

**ADR:** `docs/adr/0019-cross-session-validation.md`
**Design doc:** `analysis/design/cross_session.md`
**Target:** 4 new files, 3 updated files

---

## Files Created

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `analysis/cross_session_data.py` | Pure data logic: matching, alignment, comparison metrics | ~415 | Done |
| `analysis/cross_session.py` | CLI, orchestration, plot functions | ~600 | Done |
| `analysis/cross_session_report.py` | HTML report builder (~15 sections) | ~380 | Done |
| `tests/test_cross_session.py` | Tests for data logic (42 tests) | ~400 | Done |

## Files Updated

| File | Change | Status |
|------|--------|--------|
| `Justfile` | Add `cross-session` recipe | Done |
| `docs/roadmap.md` | Move cross-session to "In Progress" | Done |
| `CLAUDE.md` | Add cross-session to analysis architecture section | Done |

---

## Step 1: Data Layer (`analysis/cross_session_data.py`)

Pure data logic, no I/O, no plots. All functions take DataFrames in, return DataFrames out.

### Functions

```python
# ── Constants ──
MIN_OVERLAP: int = 20
SHIFT_THRESHOLD_SD: float = 1.0
ALIGNMENT_TRIM_PCT: int = 10
CORRELATION_WARN: float = 0.70

# ── Legislator Matching ──
def normalize_name(name: str) -> str:
    """Lowercase, strip whitespace, remove leadership suffixes."""

def match_legislators(
    leg_a: pl.DataFrame,  # legislators CSV from session A
    leg_b: pl.DataFrame,  # legislators CSV from session B
) -> pl.DataFrame:
    """Match legislators across sessions by normalized full_name.

    Returns DataFrame with columns:
      full_name, slug_a, slug_b, party_a, party_b,
      chamber_a, chamber_b, district_a, district_b
    Flags chamber switches and party switches.
    """

def classify_turnover(
    leg_a: pl.DataFrame,
    leg_b: pl.DataFrame,
    matched: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Classify into returning/departing/new cohorts.

    Returns {"returning": df, "departing": df, "new": df}
    """

# ── IRT Scale Alignment ──
def align_irt_scales(
    xi_a: pl.DataFrame,  # ideal_points from session A (xi_mean, legislator_slug, full_name)
    xi_b: pl.DataFrame,  # ideal_points from session B
    matched: pl.DataFrame,
) -> tuple[float, float, pl.DataFrame]:
    """Robust affine alignment: xi_a_aligned = A * xi_a + B.

    1. Inner-join on full_name to get paired xi values.
    2. OLS fit: xi_b = A * xi_a + B.
    3. Trim top/bottom ALIGNMENT_TRIM_PCT% residuals.
    4. Re-fit on trimmed set.

    Returns (A, B, aligned_df) where aligned_df has
    xi_a_aligned, xi_b, delta_xi, abs_delta_xi per matched legislator.
    """

# ── Shift Analysis ──
def compute_ideology_shift(
    aligned: pl.DataFrame,
) -> pl.DataFrame:
    """Add shift metrics: delta_xi, rank_shift, is_significant_mover.

    Significant = |delta_xi| > SHIFT_THRESHOLD_SD * std(delta_xi).
    """

def compute_metric_stability(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    matched: pl.DataFrame,
    metrics: list[str],  # e.g., ["unity_score", "maverick_rate", "betweenness"]
) -> pl.DataFrame:
    """Compute Pearson r and Spearman rho for each metric across sessions.

    Returns DataFrame: metric, pearson_r, spearman_rho, n_legislators.
    """

def compute_turnover_impact(
    xi_returning: pl.DataFrame,
    xi_departing: pl.DataFrame,
    xi_new: pl.DataFrame,
) -> dict:
    """Compare ideology distributions across turnover cohorts.

    Returns dict with means, SDs, KS test results.
    """
```

### Tests (Step 1)

```
class TestNormalizeName         # suffix stripping, whitespace, case
class TestMatchLegislators      # exact matches, no matches, chamber/party switches
class TestClassifyTurnover      # returning/departing/new counts, edge cases
class TestAlignIrtScales        # known affine transform, trimming, min overlap guard
class TestComputeIdeologyShift  # significant movers, threshold behavior
class TestComputeMetricStability # known correlations, missing columns graceful
```

---

## Step 2: Cross-Session Prediction (`analysis/cross_session.py`, prediction section)

Uses existing `build_vote_features()` from `analysis/prediction.py`.

### Functions

```python
def prepare_cross_session_features(
    features_a: pl.DataFrame,
    features_b: pl.DataFrame,
    matched: pl.DataFrame,
    numeric_cols: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Z-score normalize numeric features within each session.

    Filter session B to returning legislators only.
    Returns (standardized_a, standardized_b).
    """

def cross_session_predict(
    features_train: pl.DataFrame,
    features_test: pl.DataFrame,
    feature_cols: list[str],
    target_col: str = "vote_binary",
) -> dict:
    """Train XGBoost on session A, predict session B.

    Returns dict with accuracy, auc_roc, precision, recall, f1,
    plus the fitted model for SHAP extraction.
    """

def compare_feature_importance(
    shap_a: np.ndarray,
    shap_b: np.ndarray,
    feature_names: list[str],
    top_k: int = FEATURE_IMPORTANCE_TOP_K,
) -> pl.DataFrame:
    """Compare SHAP importance rankings across sessions.

    Returns DataFrame: feature, rank_a, rank_b, importance_a, importance_b.
    Computes Kendall's tau on top-K rankings.
    """
```

---

## Step 3: Detection Validation (`analysis/cross_session.py`, detection section)

Reuses `synthesis_detect.py` functions directly.

```python
def validate_detection_thresholds(
    leg_df_a: pl.DataFrame,  # build_legislator_df() output for session A
    leg_df_b: pl.DataFrame,  # build_legislator_df() output for session B
    chamber: str,
) -> dict:
    """Run synthesis detection on both sessions, compare results.

    Returns dict with:
      maverick_a, maverick_b: NotableLegislator | None
      bridge_a, bridge_b: NotableLegislator | None
      paradox_a, paradox_b: ParadoxCase | None
      same_maverick: bool
      same_bridge: bool
    """
```

---

## Step 4: Plots (`analysis/cross_session.py`)

~9 plot functions, all following existing project conventions (PARTY_COLORS, annotated findings, plain-English titles).

```python
# ── Ideology ──
def plot_ideology_shift_scatter(aligned, chamber, plots_dir)
    # THE signature plot. 2023 xi on x, 2025 xi on y.
    # Diagonal = no change. Annotate top 5 movers. Color by party.

def plot_biggest_movers(aligned, chamber, plots_dir)
    # Horizontal bar chart: top 15 by |delta_xi|.
    # Color by direction (moved left = blue, moved right = red).

def plot_shift_distribution(aligned, chamber, plots_dir)
    # Histogram of delta_xi. Annotate mean, SD, significant mover threshold.

def plot_turnover_impact(xi_returning, xi_departing, xi_new, chamber, plots_dir)
    # Ridge/violin plot: ideology distributions by cohort.

# ── Metric Stability ──
def plot_metric_scatter(df_a, df_b, matched, metric, chamber, plots_dir)
    # Generic scatter for any metric (unity, maverick, betweenness).
    # Diagonal = no change. Color by party.

# ── Prediction ──
def plot_prediction_comparison(within_auc, cross_auc, chamber, plots_dir)
    # Grouped bar chart: within-session vs cross-session AUC.

def plot_feature_importance_comparison(shap_df, chamber, plots_dir)
    # Side-by-side horizontal bar charts of top-K SHAP features.
```

---

## Step 5: Report Builder (`analysis/cross_session_report.py`)

Follows the same pattern as `synthesis_report.py`: function that takes RunContext + data, adds sections.

```python
def build_cross_session_report(
    ctx: RunContext,
    results: dict,  # all computed data from the analysis
) -> None:
    """Add ~15-20 sections to the HTML report."""
```

### Section order

1. Overview & data summary
2. Legislator matching summary (table: matched, switches, overlap %)
3. Ideology shift scatter — House
4. Ideology shift scatter — Senate
5. Biggest movers — House (bar chart + table)
6. Biggest movers — Senate (bar chart + table)
7. Shift distribution (histogram)
8. Turnover impact — House (violin/ridge)
9. Turnover impact — Senate (violin/ridge)
10. Metric stability summary (table: metric, pearson_r, spearman_rho)
11. Party loyalty stability scatter (per chamber)
12. Network influence stability scatter (per chamber)
13. Cross-session prediction AUC (bar chart)
14. Feature importance comparison (side-by-side SHAP)
15. Detection validation (table: role, 2023-24 legislator, 2025-26 legislator)
16. Methodology notes

---

## Step 6: CLI & Orchestration (`analysis/cross_session.py` main block)

```python
# CLI args
--session-a   # First session (default: "2023-24")
--session-b   # Second session (default: "2025-26")
--skip-prediction  # Skip the prediction cross-validation (faster)
--chambers    # "house", "senate", or "both" (default: "both")
```

### Orchestration flow

```
1. Parse CLI args
2. Load raw legislator CSVs from both sessions (for matching)
3. Match legislators by name
4. Classify turnover cohorts
5. For each chamber:
   a. Load IRT ideal points from both sessions
   b. Align IRT scales (affine transform)
   c. Compute ideology shift metrics
   d. Load synthesis legislator DFs from both sessions
   e. Compute metric stability (unity, maverick, betweenness)
   f. Plot ideology scatter, movers, shift distribution, turnover
   g. Plot metric stability scatters
   h. (If not --skip-prediction) Load vote features, run cross-session prediction
   i. Run detection validation
6. Build HTML report
7. Save parquets: ideology_shift_{chamber}, metric_stability, prediction_results
```

---

## Step 7: Justfile & Doc Updates

```just
# Run cross-session validation
cross-session *args:
    uv run python analysis/cross_session.py {{args}}
```

Update CLAUDE.md analysis architecture section to include:
- `analysis/cross_session.py` + `cross_session_data.py` + `cross_session_report.py`
- Output path: `results/kansas/cross-session/validation/`

Update roadmap: move cross-session from "Next Up" to "In Progress".

---

## Implementation Order

| Order | What | Depends On | Status |
|-------|------|-----------|--------|
| 1 | `cross_session_data.py` + tests | Nothing (pure logic) | Done (v2026.02.22.27) |
| 2 | Plot functions in `cross_session.py` | Step 1 | Done (v2026.02.22.28) |
| 3 | `cross_session_report.py` | Steps 1-2 | Done (v2026.02.22.28) |
| 4 | CLI orchestration in `cross_session.py` | Steps 1-3 | Done (v2026.02.22.28) |
| 5 | Cross-session prediction logic | Steps 1, 4 | Done (v2026.02.22.30) |
| 6 | Detection validation | Steps 1, 4 | Done (v2026.02.22.28) |
| 7 | Justfile + doc updates | Steps 1-6 | Done (v2026.02.22.28) |

All 7 steps complete. `--skip-prediction` flag available for faster runs when prediction transfer is not needed.

**Code review (v2026.02.22.31):** Fixed 6 bugs — broken ternary precedence in methodology section, `np.argsort` vs rank inversion in SHAP comparison, dead `n_dropped` code, truthy float check on AUC, RNG re-creation inside loop, "three questions" → "four questions" text.

---

## Open Questions (to resolve during implementation)

1. **Should we save MCMC traces for posterior comparison?** The IRT phase saves parquets (point estimates + HDI), but Wasserstein distance on posteriors requires the full trace. Option: load the ArviZ `InferenceData` NetCDF from the IRT run directory if available, otherwise fall back to point-estimate-only comparison.

2. **How to handle the hierarchical IRT?** 2023-24 doesn't have hierarchical results. Options: (a) only compare flat IRT (both available), (b) run hierarchical on 2023-24 first. Recommendation: (a) for now, add hierarchical comparison later.

3. **Should cross-session results feed back into synthesis?** The ideology shift data could enrich the synthesis report. Defer to a follow-up PR — keep this phase self-contained first.
