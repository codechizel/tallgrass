# Cross-Session Validation Deep Dive

A code audit, ecosystem survey, and fresh-eyes evaluation of the Cross-Session Validation phase (Phase 13).

**Date:** 2026-02-25

---

## Executive Summary

The cross-session validation phase is architecturally sound — a clean three-file separation (data, orchestration, report), four well-motivated sub-analyses, and 55 tests covering the pure data layer. The affine alignment approach with trimmed OLS is the standard method for two-session comparison, and with 131 overlapping legislators (78% overlap), the anchor set is more than adequate.

An ecosystem survey confirms the **Python gap is real**: there is no Python package for cross-session ideal point bridging. The R ecosystem dominates (DW-NOMINATE, Nokken-Poole, `emIRT::dynIRT`, `idealstan`), but the "build from scratch using existing stack" decision (ADR-0019) was the right call.

This deep dive identifies **three bugs** (one high-severity, two medium), **one design gap**, **eleven test gaps**, and **two refactoring opportunities**. It also surveys the academic and open-source landscape for cross-session legislative analysis — a survey that validates the implementation's methodological choices while identifying where future multi-session work would diverge from the current pairwise approach.

---

## 1. Field Survey: How Do Others Compare Legislators Across Sessions?

### 1.1 The Core Problem: Scale Identification

The fundamental challenge in comparing IRT ideal points across sessions is the **identification problem**. Each independent IRT fit produces scores on an arbitrary latent scale that can be shifted, stretched, or reflected without changing the model likelihood. Raw scores from two sessions are not comparable — this is a mathematical property of the model class, not a software limitation.

Key reference: Herron (2004), "Studying Dynamics in Legislator Ideal Points: Scale Matters," *Political Analysis* 12(2), 182–190. His central argument: research designs that compare ideal points across time "must allow for changes in underlying policy spaces."

### 1.2 Academic Methods for Cross-Session Estimation

| Method | How It Works | Cross-Session? | Python Available? |
|--------|-------------|----------------|-------------------|
| **DW-NOMINATE** | Joint estimation across all congresses; linear movement trajectory | By construction | pynominate (minimal) |
| **Nokken-Poole** | Fix bill parameters from common-space model; re-estimate legislators per session | By construction | No |
| **Martin-Quinn** | Dynamic Bayesian IRT; random walk prior on ideal points across terms | By construction | No (Stan/PyMC possible) |
| **Affine alignment** | Post-hoc linear transform using bridge legislators | Post-hoc | Our implementation |
| **Shor-McCarty** | External survey data as bridge across state chambers and Congress | External bridge | No |
| **Bailey Bridge** | Bridge *issues* (not people) across institutions and time | Issue bridge | No |
| **Bateman-Clinton-Lapinski** | Match identical bills across sessions as anchors | Content bridge | No |

**Our approach (affine alignment)** is Method B in this taxonomy. It is the standard post-hoc approach when sessions are estimated independently and overlap is high. The literature validates it for two-session pairwise comparison:

- Shor, McCarty, and Berry (2011) showed that "only a few such bridges are necessary" — our 131 anchors far exceed the minimum.
- The trimmed OLS variant (removing genuine movers before fitting) is standard robust regression practice.
- Session B as reference is the natural choice ("where did legislators come from?").

**Limitations relative to joint estimation:** Affine alignment assumes a *linear* relationship between scales. If the policy space rotated (e.g., the primary cleavage shifted from economic to social issues), an affine transform won't capture that. For adjacent bienniums in the same state, this assumption is safe.

### 1.3 Python Implementations

| Package | URL | Framework | Cross-Session? |
|---------|-----|-----------|----------------|
| **pynominate** | github.com/voteview/pynominate | sklearn, matplotlib | Yes (by design) |
| **py-irt** | github.com/nd-ball/py-irt | Pyro/PyTorch | No (static only) |
| **tbip** | github.com/keyonvafa/tbip | TF / NumPyro | Partial (multi-session speech) |
| **idealstan** (R) | github.com/saudiwin/idealstan | Stan | Yes (time-varying) |
| **emIRT** (R) | github.com/kosukeimai/emIRT | EM | Yes (dynIRT) |
| **MCMCpack** (R) | CRAN | C++ | Yes (MCMCdynamicIRT1d) |

**The Python ecosystem gap is real.** There is no Python equivalent of `MCMCpack::MCMCdynamicIRT1d` or `emIRT::dynIRT`. For dynamic/longitudinal IRT in Python, you'd need custom Stan code via CmdStanPy or a custom PyMC model (the random walk prior is straightforward to implement). This is relevant if/when the project moves to 3+ sessions.

### 1.4 R Ecosystem (Dominant for This Problem)

The R ecosystem has mature tools that our Python implementation doesn't replicate:

- **wnominate**: W-NOMINATE (single session, analogous to our flat IRT)
- **pscl**: Bayesian ideal points (Clinton-Jackman-Rivers), single session
- **Rvoteview**: Data access package — downloads NOMINATE/Nokken-Poole scores from Voteview
- **emIRT**: Fast EM algorithms including `dynIRT` for dynamic estimation
- **idealstan**: Generalized IRT with Stan — time-varying, missing data, multiple response types

The project's decision to build cross-session from scratch rather than bridging to R (via rpy2) is consistent with the "Python over R" technology preference (ADR-0030, analysis-framework.md).

### 1.5 Data Access Projects

| Project | What It Does |
|---------|-------------|
| **unitedstates/congress** | Python scrapers for US Congress bills, votes, amendments |
| **Voteview** | Downloadable NOMINATE/Nokken-Poole scores for all US Congresses |
| **Open States** | API and data for all 50 state legislatures |
| **OpenStatesParser** | Roll call matrices + pairwise similarity from Open States data |
| **DIME** | Campaign finance-based ideology scores (Bonica CFscores) — cross-session by design |

### 1.6 What the Literature Suggests for Future Work

Two capabilities appear in the literature but are out of scope for our current two-session comparison:

1. **Dynamic IRT (random walk).** When three or more sessions are available, the Martin-Quinn random walk prior (`xi[t] ~ Normal(xi[t-1], tau^2)`) is the standard Bayesian approach. It provides full posterior uncertainty on movements and allows nonlinear trajectories. The design doc already notes this as a future upgrade (cross_session.md, line 152).

2. **Content-based anchoring (Bateman-Clinton-Lapinski).** Instead of matching legislators, match bills that appear in both sessions. This sidesteps the assumption that bridge legislators are stable — an assumption that Gray (2020) found is "technically, but immaterially, false." Kansas has recurring legislation (e.g., annual budget bills) that could serve as content anchors.

Neither is a deficiency in the current implementation — they require either more data (3+ sessions) or upstream scraper changes (bill text). The pairwise affine approach is the right tool for the current problem.

---

## 2. Code Audit

### 2.1 Architecture

The three-file structure follows the project standard:

```
cross_session_data.py   (549 lines) — Pure data logic: matching, alignment, shift, stability, prediction helpers
cross_session.py        (982 lines) — Orchestrator: CLI, plots, detection validation, prediction transfer, main loop
cross_session_report.py (511 lines) — Report builder: ~15 section functions
```

**Separation of concerns is clean.** `cross_session_data.py` has no I/O, no plotting, and no report logic. Every public function takes DataFrames in and returns structured data out. The orchestrator handles all I/O, plotting, and report assembly.

**Dependency chain is clean:**
- `cross_session_data.py` → scipy.stats, numpy, polars (no project imports)
- `cross_session.py` → `cross_session_data`, `cross_session_report`, `synthesis_data`, `synthesis_detect`, `run_context`
- `cross_session_report.py` → `cross_session_data` (constants only), `report`

No circular dependencies. The data module is independently testable.

### 2.2 Issues Found

#### Issue 1 (Bug — High): Turnover impact compares IRT values on different scales

**Location:** `cross_session.py:874-878`

```python
# Get xi values for each cohort from the raw IRT DataFrames
ret_slugs = set(chamber_matched["slug_b"].to_list())
xi_ret = irt_b.filter(...)["xi_mean"].to_numpy()    # Session B scale
xi_dep = irt_a.filter(...)["xi_mean"].to_numpy()     # Session A scale ← WRONG
xi_new = irt_b.filter(...)["xi_mean"].to_numpy()     # Session B scale
```

The code extracts departing legislators' ideology from `irt_a` (Session A's raw IRT output) but compares it against returning and new legislators from `irt_b` (Session B's raw IRT output). These are on **different latent scales**. The affine alignment coefficients (`a_coef`, `b_coef`) are computed earlier (line 834) but never applied to the departing cohort.

**Impact:** `compute_turnover_impact()` runs KS tests and computes means on mismatched scales. If Session A's IRT scale is shifted or stretched relative to Session B's, the departing cohort's mean/distribution will appear systematically different from returning/new — a statistical artifact, not a real finding. The turnover impact strip plot and all downstream interpretations are affected.

**Why departing legislators aren't in `aligned`:** The `aligned` DataFrame only contains legislators in *both* sessions (the intersection). Departing legislators are by definition absent from Session B, so they can't appear in the alignment output.

**Fix:** Apply the affine transform to departing legislators' xi values:

```python
xi_dep_raw = irt_a.filter(...)["xi_mean"].to_numpy()
xi_dep = xi_dep_raw * a_coef + b_coef  # Transform to Session B scale
```

This puts all three cohorts on Session B's scale, making the comparison valid. The affine coefficients are already available in scope (`a_coef`, `b_coef` from line 834).

**Severity:** High. Produces incorrect statistical comparisons and misleading visualizations.

#### Issue 2 (Bug — Medium): ddof mismatch between classification and visualization thresholds

**Location:** `cross_session_data.py:277` vs `cross_session.py:320`

```python
# Classification (cross_session_data.py:277) — Polars default ddof=1
delta_std = aligned["delta_xi"].std()
threshold = SHIFT_THRESHOLD_SD * delta_std

# Visualization (cross_session.py:320) — NumPy default ddof=0
std = float(np.std(deltas))
threshold = SHIFT_THRESHOLD_SD * std
```

Polars `.std()` uses `ddof=1` (sample standard deviation). NumPy `np.std()` uses `ddof=0` (population standard deviation). The dashed threshold lines on the shift distribution histogram don't match the actual threshold used to classify significant movers.

**Impact:** For n=131, the difference is ~0.4% (`sqrt(131/130)` ≈ 1.0038). Visually negligible, but a legislator right at the boundary could appear inside the threshold lines on the plot while actually being classified as a significant mover (or vice versa). More importantly, it's a correctness issue — the visualization should match the computation exactly.

**Fix:** Use `np.std(deltas, ddof=1)` in `plot_shift_distribution` to match Polars behavior.

**Severity:** Medium. Cosmetically minor for n=131, but a correctness violation that would become visible with smaller samples.

#### Issue 3 (Bug — Medium): Cross-session prediction tests all legislators, not returning only

**Location:** `cross_session.py:567-738`

The design doc (cross_session.md, line 96) specifies:

> "Train XGBoost on session A features → predict session B votes **(returning legislators only)**."

But `_run_cross_prediction()` loads ALL votes from both sessions and never filters to returning legislators:

```python
vf_a = _load_vote_features(ks_a, chamber)  # All session A votes
vf_b = _load_vote_features(ks_b, chamber)  # All session B votes — includes new legislators
# ...
X_b = vf_b_std.select(feature_cols).to_numpy()  # Unfiltered
y_pred_ab = model_ab.predict(X_b)                # Predicting on ALL legislators
```

The `matched` DataFrame is never passed to `_run_cross_prediction()`.

**Impact:** Cross-session AUC includes predictions on new legislators who weren't in the training session. This conflates two questions: (1) does the model generalize to the same legislators in a new context? and (2) does it generalize to entirely new legislators? These have different expected performance characteristics and should be measured separately (or at least limited to returning only, as the design doc states).

**Counterargument:** One could argue that testing on all legislators is a *stronger* generalization test — the model should work on anyone, not just returners. But this contradicts the stated design, and the AUC comparison against within-session holdout becomes less meaningful (within-session includes all legislators by definition).

**Fix:** Pass `matched` to `_run_cross_prediction()` and filter both `vf_a` and `vf_b` to returning legislators' slugs:

```python
ret_slugs_a = set(matched["slug_a"].to_list())
ret_slugs_b = set(matched["slug_b"].to_list())
vf_a = vf_a.filter(pl.col("legislator_slug").is_in(ret_slugs_a))
vf_b = vf_b.filter(pl.col("legislator_slug").is_in(ret_slugs_b))
```

**Severity:** Medium. The current behavior is defensible as a design choice, but it contradicts the documented methodology.

#### Issue 4 (Design Gap): Feature importance tau asymmetry is undocumented in code

**Location:** `cross_session_data.py:536-544`

```python
df = ... .sort("rank_a")  # Sort by Session A's ranking
top_features = df.head(min(top_k, len(feature_names)))
tau, _ = stats.kendalltau(
    top_features["rank_a"].to_numpy(),
    top_features["rank_b"].to_numpy(),
)
```

Kendall's tau is computed on Session A's top-K features vs where those features rank in Session B. This is inherently asymmetric — tau computed on Session B's top-K would likely differ. The design doc notes this correctly (cross_session.md, line 101: "top-K by session A"), but the function docstring doesn't call out the asymmetry, and there's no test verifying it's intentional.

**Severity:** Low. The asymmetry is reasonable (Session A is the "training" session in the A→B direction), but should be documented in the function docstring.

### 2.3 No Dead Code Found

Every function in all three files is called:
- All 10 public functions in `cross_session_data.py` are called from `cross_session.py`
- All 6 plot functions are called from `main()`
- All ~15 report section functions are called from `build_cross_session_report()`
- All constants are used
- Both private helpers (`_normalize_slug_col`, `_add_name_norm`) are used by `match_legislators`

No dead code. No unused imports. Clean.

### 2.4 Things Done Well

- **Pure data layer is exemplary.** `cross_session_data.py` has zero I/O, zero side effects, and every function is independently testable. This is the cleanest data module in the analysis pipeline.
- **Robust alignment with trimming.** The two-pass OLS with 10% residual trimming is textbook robust regression. The fallback to untrimmed fit when trimming leaves too few points is a good safety net.
- **Bidirectional prediction transfer.** Testing both A→B and B→A catches asymmetric generalization failures.
- **SHAP-based feature comparison.** Comparing feature importance rankings (not just raw AUC) is a genuinely insightful validation — stable rankings mean the model captures structural patterns rather than session-specific noise. This goes beyond what most political science cross-validation approaches do.
- **Graceful degradation.** Missing IRT data, missing prediction results, and missing metrics are all handled with warnings rather than crashes.
- **The primer is excellent.** Clear, structured, plain-English. Follows the project's audience-first principle.

### 2.5 Design Doc Discrepancy: Wasserstein Distance

The design doc (cross_session.md, lines 72–73) lists three complementary shift metrics:

1. Point estimate shift (`delta_xi`) — **implemented**
2. Posterior overlap (Wasserstein distance on MCMC posterior samples) — **not implemented**
3. Rank shift — **implemented**

The Wasserstein distance metric is mentioned in both the design doc and ADR-0019 but was never implemented. This is a missed opportunity — posterior overlap would account for estimation uncertainty, distinguishing genuine movements from noise. A legislator with wide credible intervals in both sessions might show `delta_xi = 0.5` but overlapping posteriors (not a meaningful shift), while a precisely-estimated legislator with the same delta is a genuine mover.

**Recommendation:** Either implement it or explicitly note in the design doc that it was deferred. The data is available (MCMC traces in ArviZ InferenceData objects).

---

## 3. Test Gaps

The existing 55 tests provide strong coverage of `cross_session_data.py`. The main gaps are in the orchestration and report layers, and in edge cases that would have caught the bugs above.

### 3.1 Missing Coverage for `cross_session.py` Functions

| Function | Tests | Status |
|----------|-------|--------|
| `validate_detection()` | 0 | **Missing** — no test verifies detection runs on both sessions |
| `_majority_party()` | 0 | **Missing** — no test for empty DataFrame or ties |
| `_extract_name()` | 0 | **Missing** — no test for suffix stripping, single-word names |
| `plot_ideology_shift_scatter()` | 0 | **Missing** — no smoke test |
| `plot_biggest_movers()` | 0 | **Missing** — no smoke test |
| `plot_shift_distribution()` | 0 | **Missing** — no smoke test |
| `plot_turnover_impact()` | 0 | **Missing** — no smoke test |
| `plot_prediction_comparison()` | 0 | **Missing** — no smoke test |
| `plot_feature_importance_comparison()` | 0 | **Missing** — no smoke test |
| `_run_cross_prediction()` | 0 | **Missing** — integration test (acceptable to skip) |

### 3.2 Missing Coverage for `cross_session_report.py`

| Function | Tests | Status |
|----------|-------|--------|
| `build_cross_session_report()` | 0 | **Missing** |
| All `_add_*` section builders | 0 | **Missing** |

### 3.3 Missing Edge Case and Bug-Catching Tests

1. **`test_turnover_impact_same_scale`** — Verify that all three cohorts are compared on the same scale. Pass cohorts with a known affine offset and confirm the results reflect the transformed values. (Would have caught Issue 1.)

2. **`test_shift_threshold_matches_plot`** — Verify that the classification threshold in `compute_ideology_shift` matches what `plot_shift_distribution` would display. Compute both with the same data and assert equality. (Would have caught Issue 2.)

3. **`test_validate_detection_basic`** — Pass two synthetic legislator DataFrames with known maverick/bridge/paradox patterns, verify the function returns the expected names.

4. **`test_majority_party_returns_largest`** — Pass a DataFrame with 3 Republicans and 2 Democrats, verify "Republican" is returned.

5. **`test_majority_party_empty_df`** — Pass an empty DataFrame, verify `None` is returned.

6. **`test_extract_name_strips_suffix`** — Verify `_extract_name("Bob Jones - Speaker")` returns `"Jones"`.

7. **`test_extract_name_single_word`** — Verify `_extract_name("Cher")` returns `"Cher"`.

8. **`test_feature_importance_asymmetry`** — Verify that swapping session A/B SHAP values produces a different tau (documenting the intentional asymmetry).

9. **`test_normalize_name_hyphen_dash_distinction`** — Verify that `"Mary-Jane Smith"` (hyphenated name) is preserved but `"Mary Smith - Chair"` (leadership suffix) is stripped. Currently this works, but the regex pattern warrants explicit coverage.

### 3.4 Smoke Tests for Plot Functions

Six plot functions produce PNGs. Smoke tests should:
- Pass synthetic data
- Verify the function doesn't crash
- Verify the output file exists on disk
- Clean up with `tmp_path`

These catch matplotlib API regressions (deprecated kwargs, changed defaults).

### 3.5 Summary

| Category | Current | Recommended | New Tests |
|----------|---------|-------------|-----------|
| `cross_session_data.py` | 55 | 58 | 3 edge cases + bug catchers |
| `cross_session.py` (helpers) | 0 | 5 | detection, majority_party, extract_name |
| `cross_session.py` (plotting) | 0 | 6 | smoke tests |
| `cross_session_report.py` | 0 | 2 | report integration |
| **Total** | **55** | **71** | **+16** |

---

## 4. Refactoring Opportunities

### 4.1 XGBoost Hyperparameters Duplicated

**Location:** `cross_session.py:621-629` and `cross_session.py:640-648`

The XGBoost model is constructed identically in two places (A→B and B→A). Extract to a constant or factory function:

```python
XGBOOST_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
    n_jobs=-1,
)

model_ab = XGBClassifier(**XGBOOST_PARAMS)
model_ba = XGBClassifier(**XGBOOST_PARAMS)
```

This prevents drift if one is updated without the other.

### 4.2 `report: object` Typing (Project-Wide)

**Location:** `cross_session_report.py:45` (`report: ReportBuilder` would be correct)

All report section functions receive `report: ReportBuilder` but the type is elided because of the try/except import pattern. Same pattern exists in profiles_report.py, synthesis_report.py, and others. A project-wide fix (conditional import or Protocol type) would restore type safety. Not cross-session-specific.

### 4.3 `report._sections` Access (Project-Wide)

**Location:** `cross_session_report.py:87`

Accesses private `report._sections` for section count logging. Same pattern in 16+ files. Add a `section_count` property to `ReportBuilder`.

---

## 5. Comparison: Our Implementation vs. the Field

### 5.1 What We Do Better

| Capability | Us | Best Alternative |
|-----------|-----|-----------------|
| Prediction transfer | XGBoost cross-session AUC + SHAP comparison | Not found in any open-source project |
| Feature importance stability | Kendall's tau on SHAP rankings | Novel approach |
| Detection threshold validation | Same thresholds tested on independent data | Not found |
| Turnover cohort analysis | KS tests on departing/returning/new distributions | Voteview (descriptive only, no statistical tests) |
| Pure Python, no R bridge | Affine alignment + scipy + polars | Most projects require R (wnominate, pscl) |

### 5.2 What the Field Does That We Don't

| Capability | Who Does It | Gap Assessment |
|-----------|------------|----------------|
| Joint estimation (DW-NOMINATE) | Voteview (R, Fortran) | Requires pooled model; 2-session pairwise doesn't warrant it |
| Session-specific scores (Nokken-Poole) | Voteview | Requires fixing bill parameters from a common-space model |
| Dynamic random walk (Martin-Quinn) | MCMCpack (R), idealstan (R) | Only valuable with 3+ sessions; no Python implementation |
| Content-based anchoring | Bateman-Clinton-Lapinski | Requires bill text matching; future work (see `docs/future-bill-text-analysis.md`) |
| Posterior overlap (Wasserstein) | Standard in Bayesian literature | Mentioned in design doc but not implemented |
| External bridging (surveys/donations) | Shor-McCarty, DIME/CFscores | Requires external data not currently scraped |

### 5.3 Methodological Validation from the Literature

Three of our design choices are directly supported by academic work:

1. **Affine alignment with trimmed OLS.** Shor, McCarty, and Berry (2011) validated that "only a few bridges are necessary" for reliable alignment. Our 131 anchors far exceed the minimum. The trimming approach is standard robust regression.

2. **`MIN_OVERLAP = 20` threshold.** Consistent with simulation results from the sparse-bridging literature. Below ~20 bridge legislators, affine fits become unreliable.

3. **Session B as reference scale.** The most recent session is the standard reference in cross-session political science work — DW-NOMINATE uses the most recent Congress as the endpoint of the linear trajectory.

One choice diverges from the literature:

4. **Z-score standardization for cross-session prediction.** The standard approach in the prediction transfer literature is to use domain adaptation techniques (transfer learning, covariate shift correction). Z-score standardization is a simple baseline. For this application (same institution, adjacent sessions, high overlap), it's likely sufficient — but domain adaptation could improve cross-session AUC if the current ~0.85-0.92 performance needs improvement.

---

## 6. Recommendations Summary

### Priority 1 (Bug Fixes)

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| 1 | Turnover impact scale mismatch | `cross_session.py:877` | Apply `a_coef * xi_dep + b_coef` to departing cohort |
| 2 | ddof mismatch (plot vs classification) | `cross_session.py:320` | Use `np.std(deltas, ddof=1)` |
| 3 | Prediction tests all legislators | `cross_session.py:567-738` | Filter to returning legislators or update design doc |

### Priority 2 (New Tests)

| # | Test | Covers |
|---|------|--------|
| 4 | `test_turnover_impact_same_scale` | Scale mismatch prevention |
| 5 | `test_shift_threshold_matches_plot` | ddof consistency |
| 6 | `test_validate_detection_basic` | Detection function coverage |
| 7 | `test_majority_party_returns_largest` | Helper coverage |
| 8 | `test_majority_party_empty_df` | Edge case |
| 9 | `test_extract_name_strips_suffix` | Helper coverage |
| 10 | `test_extract_name_single_word` | Edge case |
| 11 | `test_feature_importance_asymmetry` | Documenting intentional behavior |
| 12 | `test_normalize_name_hyphen_dash_distinction` | Regex edge case |
| 13-18 | Six plot function smoke tests | Matplotlib regression protection |
| 19 | `test_build_cross_session_report` | Report integration |

### Priority 3 (Refactoring & Design)

| # | Issue | Scope |
|---|-------|-------|
| 20 | Extract XGBoost hyperparameters to constant | `cross_session.py` |
| 21 | Implement or defer Wasserstein distance | Design doc vs code gap |
| 22 | Document tau asymmetry in function docstring | `cross_session_data.py:495` |
| 23 | `report: object` → `ReportBuilder` typing | Project-wide |
| 24 | `report._sections` → `report.section_count` | Project-wide |

---

## 7. References

### Foundational

- Poole, K.T. and H. Rosenthal. 1985. "A Spatial Model for Legislative Roll Call Analysis." *AJPS* 29(2), 357–384.
- Clinton, J., S. Jackman, and D. Rivers. 2004. "The Statistical Analysis of Roll Call Data." *APSR* 98(2), 355–370.
- Herron, M.C. 2004. "Studying Dynamics in Legislator Ideal Points: Scale Matters." *Political Analysis* 12(2), 182–190.
- Martin, A.D. and K.M. Quinn. 2002. "Dynamic Ideal Point Estimation via Markov Chain Monte Carlo for the U.S. Supreme Court, 1953–1999." *Political Analysis* 10(2), 134–153.

### Cross-Session / Cross-Institutional

- Nokken, T.P. and K.T. Poole. 2004. "Congressional Party Defection in American History." *Legislative Studies Quarterly* 29, 545–568.
- Shor, B. and N. McCarty. 2011. "The Ideological Mapping of American Legislatures." *APSR* 105(3), 530–551.
- Shor, B., N. McCarty, and C. Berry. 2011. "Methodological Issues in Bridging Ideal Points in Disparate Institutions in a Data Sparse Environment." SSRN.
- Bailey, M.A. 2007. "Comparable Preference Estimates across Time and Institutions for the Court, Congress, and Presidency." *AJPS* 51(3), 433–448.
- Bateman, D.A., J.D. Clinton, and J.S. Lapinski. 2017. "Anchors Away: A New Approach for Estimating Ideal Points Comparable across Time and Chambers." *Political Analysis* 25(2), 172–191.
- Gray, T. 2020. "A Bridge Too Far? Examining Bridging Assumptions in Common-Space Estimations." *Legislative Studies Quarterly* 45(2).
- Groseclose, T., S.D. Levitt, and J.M. Snyder. 1999. "Comparing Interest Group Scores across Time and Chambers: Adjusted ADA Scores for the U.S. Congress." *APSR* 93(1), 33–50.

### Validation and Methods

- Remmel, M.L. and J.J. Mondak. 2020. "Three Validation Tests of the Shor-McCarty State Legislator Ideology Data." *American Politics Research*.
- Imai, K., J. Lo, and J. Olmsted. 2016. "Fast Estimation of Ideal Points with Massive Data." *APSR* 110(4), 631–656.
- Lewis, J.B. and L. Sonnet. Working paper. "Estimating NOMINATE scores over time using penalized splines."
- Shin, S. 2024. "Measuring Issue Specific Ideal Points from Roll Call Votes." Working paper.
- Shin, S., J. Lim, and J.H. Park. 2025. "L1-based Bayesian Ideal Point Model for Roll Call Data." *JASA* 120(550), 631–644.

### Open-Source Projects

- Voteview data: https://voteview.com/data
- pynominate: https://github.com/voteview/pynominate
- py-irt: https://github.com/nd-ball/py-irt
- emIRT: https://github.com/kosukeimai/emIRT
- idealstan: https://github.com/saudiwin/idealstan
- Rvoteview: https://github.com/voteview/Rvoteview
- Open States: https://docs.openstates.org/api-v3/
- DIME (CFscores): https://data.stanford.edu/dime
- Martin-Quinn Scores: https://mqscores.wustl.edu/
- Stan ideal point examples: https://jrnold.github.io/bugs-examples-in-stan/legislators.html
- Bailey Bridge Ideal Points: https://michaelbailey.georgetown.domains/bridge-ideal-points-2020/
- Shor-McCarty data: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GZJOT3
