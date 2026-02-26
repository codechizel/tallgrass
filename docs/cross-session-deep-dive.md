# Cross-Session Validation Deep Dive

An ecosystem survey, code audit, and integration assessment for the Cross-Session Validation phase (Phase 13).

**Date:** 2026-02-26 (v2, expanded from 2026-02-25 audit)

---

## Executive Summary

Cross-session validation is the problem of making IRT ideal points comparable across independently estimated legislative sessions, then using that comparability to answer questions about who moved, which metrics are stable traits, and whether predictive models generalize. It is the single biggest gap in most state-level legislative analysis projects — and the Python ecosystem offers essentially nothing off the shelf.

Our Phase 13 implementation handles the two-session case with a clean three-layer architecture (data, orchestration, report), four well-motivated sub-analyses, and 73 tests. The affine alignment approach with trimmed OLS is the academic standard for pairwise comparison, validated by Shor, McCarty & Berry (2011). With 131 overlapping legislators (78% overlap), the anchor set far exceeds the minimum.

This deep dive surveys the academic literature and open-source ecosystem, audits the existing code, evaluates integration readiness, and identifies where future multi-session work would diverge from the current pairwise approach.

**Key findings:**

- The **Python ecosystem gap is real and persistent**: no Python package handles cross-session ideal point bridging. R dominates (DW-NOMINATE, Nokken-Poole, `emIRT::dynIRT`, `idealstan`). Building from scratch was the right call (ADR-0019).
- Three bugs found in the original audit (all fixed in ADR-0035): turnover scale mismatch, ddof inconsistency, prediction filtering.
- Two **novel capabilities** not found in any open-source project: SHAP feature importance stability via Kendall's tau, and synthesis detection threshold validation across sessions.
- The implementation is **production-ready** for its first real run once both sessions' upstream phases are complete.
- **Future 3+ session work** would require a fundamentally different approach (dynamic IRT with random walk priors), implementable in PyMC but not yet built.

---

## 1. The Core Problem: Scale Identification

The fundamental challenge in comparing IRT ideal points across sessions is the **identification problem**. Each independent IRT fit produces scores on an arbitrary latent scale — the model likelihood is invariant to shifting, stretching, or reflecting all ideal points simultaneously. Raw scores from two sessions are not comparable. This is a mathematical property of the model class, not a software limitation.

Concretely: if Session A places a moderate Republican at xi = 0.5 and Session B places the same person at xi = 2.1, that tells you nothing about whether they moved. The scales are different.

Three nonidentifiabilities must be resolved:

1. **Translation** — the zero point is arbitrary (xi → xi + c)
2. **Scale** — the unit is arbitrary (xi → a·xi)
3. **Reflection** — the sign is arbitrary (xi → −xi)

Key reference: Herron (2004), "Studying Dynamics in Legislator Ideal Points: Scale Matters," *Political Analysis* 12(2), 182–190. His central argument: research designs that compare ideal points across time "must allow for changes in underlying policy spaces."

---

## 2. Ecosystem Survey: How Do Others Solve This?

### 2.1 Academic Methods

| Method | Authors | How It Works | Cross-Session? | Python? |
|--------|---------|-------------|----------------|---------|
| **DW-NOMINATE** | Poole & Rosenthal (1985, 2007) | Joint estimation across all congresses; legislators follow a linear movement trajectory in the policy space | By construction | pynominate (minimal) |
| **Nokken-Poole** | Nokken & Poole (2004) | Fix bill parameters from a common-space DW-NOMINATE model; re-estimate each legislator per session independently | By construction | No |
| **Martin-Quinn** | Martin & Quinn (2002) | Dynamic Bayesian IRT with random walk prior: `xi[t] ~ Normal(xi[t-1], tau^2)` | By construction | No (Stan/PyMC possible) |
| **Affine alignment** | Standard post-hoc | Linear transform using bridge legislators: `xi_b = A·xi_a + B` | Post-hoc | Our implementation |
| **Shor-McCarty** | Shor & McCarty (2011) | External survey data as bridge across state chambers and Congress | External bridge | No |
| **Bailey Bridge** | Bailey (2007) | Bridge *issues* (not people) across institutions and time | Issue bridge | No |
| **Bateman-Clinton-Lapinski** | Bateman, Clinton & Lapinski (2017) | Match identical/similar bills across sessions as content anchors, impute votes on bills predating a legislator's tenure | Content bridge | No |
| **Groseclose-Levitt-Snyder** | Groseclose, Levitt & Snyder (1999) | Adjusted ADA scores: use overlapping members to normalize interest group ratings across congresses | Post-hoc | No |
| **Lewis-Sonnet splines** | Lewis & Sonnet (working paper) | Penalized B-splines on NOMINATE scores over time, allowing smooth nonlinear trajectories | By construction | No |

**Our approach (affine alignment)** is the standard post-hoc method when sessions are estimated independently and overlap is high. The literature validates it for pairwise comparison:

- Shor, McCarty & Berry (2011) showed that "only a few such bridges are necessary" — our 131 anchors far exceed the minimum.
- The trimmed OLS variant (removing genuine movers before fitting) is standard robust regression practice.
- Gray (2020) found the bridge legislator stability assumption is "technically, but immaterially, false" — bridge legislators do move slightly, but the bias is negligible with many anchors.

**Limitations relative to joint estimation:** Affine alignment assumes a *linear* relationship between scales. If the policy space rotated (e.g., the primary cleavage shifted from economic to social issues), an affine transform won't capture that. For adjacent bienniums in the same state, this assumption is safe. For longer time horizons or cross-institutional comparison, joint estimation or content-based anchoring would be necessary.

### 2.2 Python Implementations

| Package | URL | Framework | What It Does | Cross-Session? |
|---------|-----|-----------|-------------|----------------|
| **pynominate** | github.com/voteview/pynominate | sklearn, matplotlib | Python port of W-NOMINATE; estimates ideal points and bill parameters jointly | Yes (by design) |
| **py-irt** | github.com/nd-ball/py-irt | Pyro/PyTorch | 1PL/2PL/4PL IRT models for educational testing; no legislative focus | No (static only) |
| **tbip** | github.com/keyonvafa/tbip | TF / NumPyro | Text-Based Ideal Points — speech-based ideology from floor debates | Partial (multi-session speech) |
| **pylegiscan** | github.com/poliquin/pylegiscan | requests | API wrapper for LegiScan; data access, not analysis | N/A (data only) |
| **legcop** | PyPI | requests | API wrapper for legislative data from Congress and 50 states | N/A (data only) |
| **openstates** | github.com/openstates | Django, API | Comprehensive state legislature data; bills, votes, legislators | N/A (data only) |
| **unitedstates/congress** | github.com/unitedstates/congress | Python scrapers | US Congress bills, votes, amendments scraping | N/A (data only) |

**The Python ecosystem gap is real and persistent.** There is no Python equivalent of `MCMCpack::MCMCdynamicIRT1d` or `emIRT::dynIRT`. The data access packages (pylegiscan, legcop, openstates) provide raw legislative data but no analytical tools for cross-session comparison.

For dynamic/longitudinal IRT in Python, you would need:

- **Custom PyMC model** — the Martin-Quinn random walk prior is straightforward: add `xi[t] ~ Normal(xi[t-1], tau^2)` to the IRT model. Our existing PyMC IRT infrastructure could support this.
- **CmdStanPy** — write Stan code directly; idealstan's Stan models could be ported.
- **NumPyro/Pyro** — JAX-based or PyTorch-based; py-irt already demonstrates the pattern.

None of these exist as packages. Building cross-session from scratch was the right call.

### 2.3 R Ecosystem (Dominant for This Problem)

The R ecosystem has mature tools:

| Package | What It Does | Cross-Session? |
|---------|-------------|----------------|
| **wnominate** | W-NOMINATE (single session, 2D) | No — but DW-NOMINATE does |
| **pscl** | Bayesian ideal points (Clinton-Jackman-Rivers) | No (static) |
| **Rvoteview** | Downloads NOMINATE/Nokken-Poole scores from Voteview | Pre-computed cross-session |
| **emIRT** | Fast EM algorithms including `dynIRT` for dynamic estimation | Yes |
| **idealstan** | Generalized IRT with Stan: time-varying, missing data, multiple response types | Yes |
| **MCMCpack** | C++ MCMC: `MCMCdynamicIRT1d` and `MCMCdynamicIRT` (1D and KD) | Yes |
| **basicspace** | Aldrich-McKelvey scaling, Blackbox, etc. | Partial |

The project's decision to stay Python-only (ADR-0002, ADR-0019) is consistent with the technology preference. rpy2 bridges are fragile, add deployment complexity, and create a maintenance burden for a single analysis phase.

### 2.4 Data Access and Pre-Computed Scores

| Project | What It Provides | URL |
|---------|-----------------|-----|
| **Voteview** | DW-NOMINATE and Nokken-Poole scores for every US Congress (1st–118th) | voteview.com/data |
| **Shor-McCarty** | Common-space ideal points for all 50 state legislatures (1993–2018) | Harvard Dataverse |
| **DIME/CFscores** | Campaign-finance ideology for candidates, donors, committees | data.stanford.edu/dime |
| **Martin-Quinn** | Dynamic ideal points for US Supreme Court justices | mqscores.wustl.edu |
| **Open States** | Bills, votes, legislators for all 50 states (API v3) | docs.openstates.org/api-v3 |
| **Bailey Bridge** | Bridge ideal points linking Congress, President, and Supreme Court | Georgetown |

Shor-McCarty is already integrated (Phase 14, external validation). DIME/CFscores is deferred (roadmap). Voteview is Congress-only, not directly applicable to Kansas state data.

### 2.5 Key Academic Insights for Our Implementation

**Nokken-Poole "freed" ideal points.** Nokken & Poole (2004) showed that when you allow each Congress to estimate independently (rather than constraining a linear trajectory as DW-NOMINATE does), "a sizable portion of members of both major political parties" show meaningful ideological variation across terms. This validates our core premise: cross-session comparison reveals real movements, not just noise.

**Bridge legislator stability (Gray 2020).** The assumption that bridge legislators are ideologically stable is technically violated — everyone moves a little. Gray found this violation is immaterial with sufficient anchors. Our 131 bridges (78% overlap) provide robust alignment even if some bridges moved.

**Groseclose-Levitt-Snyder (1999) adjusted scores.** Their approach to normalizing interest group ratings across congresses using overlapping members is conceptually identical to our affine alignment. They showed that unadjusted cross-congress comparisons are "severely misleading" — the same result Herron (2004) formalized for IRT. This validates the necessity of alignment.

**Lewis & Sonnet penalized splines.** Their working paper proposes smooth nonlinear trajectories for NOMINATE scores using B-splines. This is more flexible than Martin-Quinn's random walk (which is Markov — no long-range smoothness) and DW-NOMINATE (which is linear). Relevant for future work if we move beyond pairwise comparison.

**Shor-McCarty common-space methodology.** Shor & McCarty (2011) use survey responses from state legislative candidates (Project Vote Smart / National Political Awareness Test) as bridges to place state legislators on the same scale as US Congress. Their key insight: you don't need many bridges — "only a few are necessary" for reliable alignment. Their approach is inherently cross-session because the survey data provides a fixed external reference.

**Test-retest reliability.** Political science treats metric stability across sessions as a **test-retest reliability** problem. The standard psychometric measure is the Intraclass Correlation Coefficient (ICC), with interpretation thresholds: ICC < 0.50 = poor, 0.50–0.75 = moderate, 0.75–0.90 = good, > 0.90 = excellent (Koo & Li 2016). Our `compute_metric_stability()` reports Pearson/Spearman correlations for 8 metrics — functionally equivalent for the two-session case. A caveat from Groseclose, Levitt & Snyder: party unity scores are sensitive to **agenda composition** (the vote set changes across sessions), so raw level comparisons can be misleading even after alignment. Rank-based metrics are more robust.

**Entity resolution for matching.** Python offers several tools for fuzzy legislator matching beyond our current exact-normalized approach: Splink (probabilistic record linkage at scale), dedupe (ML-powered entity resolution), and polars-fuzzy-match (Rust-powered fzf-style matching). Blasingame et al. (2024) showed zero-shot LLM matching outperforms existing fuzzy matching by up to 39%. Our exact matching is sufficient with 78% overlap, but fuzzy matching would become necessary if name changes or transcription errors reduce the match rate.

**Concept drift monitoring.** Cross-session prediction is a natural case of **concept drift** — the relationship between features and votes changes as the agenda shifts. The Population Stability Index (PSI) provides a single number: PSI < 0.10 = stable, 0.10–0.25 = investigate, > 0.25 = significant drift. This could augment our z-score standardization approach by flagging when feature distributions have shifted enough to warrant caution.

---

## 3. Code Audit

### 3.1 Architecture

Three-layer separation follows the project standard:

```
cross_session_data.py   (556 lines) — Pure data logic: matching, alignment, shift, stability, prediction
cross_session.py        (993 lines) — Orchestrator: CLI, plots, detection validation, prediction, main()
cross_session_report.py (510 lines) — Report builder: ~15 section functions
```

**Separation of concerns is clean.** `cross_session_data.py` has no I/O, no plotting, and no report logic. Every public function takes DataFrames in and returns structured data out. The orchestrator handles all I/O, plotting, and report assembly.

**Dependency chain:**
- `cross_session_data.py` → scipy.stats, numpy, polars (no project imports)
- `cross_session.py` → `cross_session_data`, `cross_session_report`, `synthesis_data`, `synthesis_detect`, `run_context`
- `cross_session_report.py` → `cross_session_data` (constants only), `report`

No circular dependencies. The data module is independently testable.

### 3.2 Data Flow

```
Session A (e.g., 90th 2023-24)       Session B (e.g., 91st 2025-26)
     │                                       │
     └──── Load legislator CSVs ─────────────┘
                    │
           match_legislators()
                    │
             ~131 matched pairs (78% overlap)
                    │
           align_irt_scales()
          xi_a_aligned = A·xi_a + B
                    │
     ┌──────┬──────┼──────┬────────┐
     │      │      │      │        │
  Ideology  Metric  Turn- Detec-  Cross-
  Shift     Stab.  over   tion   Prediction
     │      │      │      │        │
     └──────┴──────┴──────┴────────┘
                    │
          Save parquets + plots + HTML report
```

### 3.3 Function Inventory

**Data layer (`cross_session_data.py`):**

| Function | Purpose | Lines |
|----------|---------|-------|
| `normalize_name()` | Lowercase, strip whitespace, remove leadership suffixes | ~10 |
| `match_legislators()` | Name-based cross-session matching; flags chamber/party switches | ~70 |
| `classify_turnover()` | Split into returning/departing/new cohorts | ~30 |
| `align_irt_scales()` | Robust affine transform with trimmed OLS | ~80 |
| `compute_ideology_shift()` | Point estimate + rank shift + significant mover classification | ~60 |
| `compute_metric_stability()` | Pearson/Spearman for 8 metrics across sessions | ~50 |
| `compute_turnover_impact()` | Cohort distribution comparison + KS tests | ~40 |
| `align_feature_columns()` | Find shared feature columns for prediction transfer | ~20 |
| `standardize_features()` | Z-score numeric features within session | ~30 |
| `compare_feature_importance()` | SHAP ranking comparison via Kendall's tau | ~40 |

All functions are independently testable with synthetic DataFrames. No function exceeds ~80 lines. Clean.

**Orchestration layer (`cross_session.py`):**

6 plot functions, `validate_detection()`, `_run_cross_prediction()`, `main()`, plus helpers (`_majority_party`, `_extract_name`, `_load_vote_features`).

**Report layer (`cross_session_report.py`):**

`build_cross_session_report()` with ~15 section builders. Standard `ReportBuilder` pattern.

### 3.4 Constants and Parameters

| Constant | Value | Justification |
|----------|-------|---------------|
| `MIN_OVERLAP` | 20 | Below ~20 bridge legislators, affine fits become unreliable (simulation literature) |
| `SHIFT_THRESHOLD_SD` | 1.0 | Flag movers > 1 SD; standard outlier threshold |
| `ALIGNMENT_TRIM_PCT` | 10 | Trim 10% extreme residuals; standard robust regression |
| `CORRELATION_WARN` | 0.70 | Below r=0.70, alignment may be unreliable |
| `FEATURE_IMPORTANCE_TOP_K` | 10 | Compare top 10 SHAP features |
| `STABILITY_METRICS` | 8 metrics | unity, maverick, weighted_maverick, betweenness, eigenvector, pagerank, loyalty, PC1 |
| `SIGN_ARBITRARY_METRICS` | `{"PC1"}` | Metrics whose sign is conventional (orient_pc1 normalizes, but edge cases can flip). Correlations use `abs()` so a sign flip doesn't masquerade as instability. |
| XGBoost params | n=200, depth=6, lr=0.1 | Fixed for both A→B and B→A |

All constants are documented with docstrings.

### 3.5 Bugs Found and Fixed (ADR-0035)

Three bugs were identified in the original audit (2026-02-25) and fixed in the same commit:

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | **Turnover impact scale mismatch** — departing legislators' xi from Session A compared on different scale against returning/new from Session B | HIGH | Apply `xi_dep = xi_dep_raw * a_coef + b_coef` to transform departing cohort |
| 2 | **ddof mismatch** — classification uses Polars `std()` (ddof=1), visualization uses `np.std()` (ddof=0) | MEDIUM | Use `np.std(deltas, ddof=1)` in plot to match Polars |
| 3 | **Prediction tests all legislators** — design doc says "returning only" but code tested on all including new | MEDIUM | Filter vote features to returning legislators' slugs |

All three are now fixed. 18 regression tests were added.

### 3.6 Strengths

- **Pure data layer is exemplary.** Zero I/O, zero side effects, fully independently testable.
- **Robust alignment with trimming.** Two-pass OLS with fallback is textbook.
- **Bidirectional prediction.** Testing both A→B and B→A catches asymmetric generalization failures.
- **SHAP-based feature comparison.** Comparing feature importance rankings via Kendall's tau is genuinely novel — most political science projects don't do this.
- **Graceful degradation.** Missing upstream results produce warnings, not crashes.
- **Session-pair agnostic.** Works with any two bienniums, not just 90th/91st.

---

## 4. Test Coverage

### 4.1 Current State: 73 Tests

| Category | Tests | Coverage |
|----------|-------|----------|
| `normalize_name()` | 7 | Complete |
| `match_legislators()` | 10 | Complete |
| `classify_turnover()` | 3 | Complete |
| `align_irt_scales()` | 6 | Complete |
| `compute_ideology_shift()` | 5 | Complete |
| `compute_metric_stability()` | 6 | Complete |
| `compute_turnover_impact()` | 5 | Complete |
| `align_feature_columns()` | 4 | Complete |
| `standardize_features()` | 4 | Complete |
| `compare_feature_importance()` | 5 | Complete |
| Bug-fix regression tests (ADR-0035) | 18 | Scale, ddof, detection, naming, plots, report |
| **Total** | **73** | **Data layer: strong. Orchestration: smoke tests only.** |

### 4.2 Coverage by Layer

```
Data layer   (cross_session_data.py):    55 unit tests + 18 regression   ✓ Strong
Orchestration (cross_session.py):        6 plot smoke tests + helpers    ◑ Adequate
Report       (cross_session_report.py):  1 integration smoke test        ◑ Minimal
```

### 4.3 Remaining Test Gaps

The orchestration and report layers are lightly tested. This is acceptable for a phase that hasn't had its first real run yet — integration issues will surface during the run. The data layer, where correctness matters most, has thorough coverage.

---

## 5. Comparison: Our Implementation vs. the Field

### 5.1 What We Do That Nobody Else Does

| Capability | Our Approach | Closest Alternative |
|-----------|-------------|-------------------|
| **SHAP feature stability** | Kendall's tau on SHAP importance rankings across sessions | Not found in any open-source project |
| **Synthesis detection validation** | Same thresholds tested on independent session | Not found |
| **Turnover cohort analysis** | KS tests on departing/returning/new ideology distributions | Voteview (descriptive only, no tests) |
| **Bidirectional prediction transfer** | Train A→test B and train B→test A with AUC comparison | Not found as integrated analysis |
| **Pure Python, no R bridge** | Affine alignment + scipy + polars | Most projects require R |

The SHAP comparison is particularly valuable: stable feature importance rankings mean the prediction model captures structural patterns in legislative voting, not session-specific noise. If `xi_mean` is the top feature in both sessions, it confirms that ideology drives votes regardless of the specific bills considered.

### 5.2 What the Field Does That We Don't

| Capability | Who Does It | Our Status | Gap Assessment |
|-----------|------------|-----------|----------------|
| Joint estimation (DW-NOMINATE) | Voteview | Not needed | Requires pooled model; 2-session pairwise doesn't warrant it |
| Session-specific scores (Nokken-Poole) | Voteview | Not needed | Requires fixing bill parameters from common-space model |
| Dynamic random walk (Martin-Quinn) | MCMCpack, idealstan (R) | **Deferred** | Valuable with 3+ sessions; implementable in PyMC |
| Content-based anchoring (Bateman-Clinton-Lapinski) | Custom R code | **Deferred** | Requires bill text matching (see `docs/future-bill-text-analysis.md`) |
| Posterior overlap (Wasserstein) | Standard Bayesian practice | **Deferred** | Design doc mentions it; requires loading two sessions' MCMC traces |
| Penalized spline trajectories (Lewis-Sonnet) | Custom R code | Not applicable | Only relevant for 5+ time points with smooth trajectories |
| External bridging (surveys) | Shor-McCarty | Already in Phase 14 | SM external validation covers this separately |

The "deferred" items are genuine future work opportunities, not deficiencies. None is necessary for the two-session case. Dynamic IRT becomes the priority when the 89th (2021-22) or 87th-88th sessions are integrated for multi-session trajectories.

### 5.3 Methodological Validation

Three of our design choices are directly supported by the literature:

1. **Affine alignment with trimmed OLS.** Shor, McCarty & Berry (2011) validated that "only a few bridges are necessary." Groseclose, Levitt & Snyder (1999) demonstrated the same approach for interest group rating normalization. Our 131 anchors far exceed the minimum.

2. **`MIN_OVERLAP = 20` threshold.** Consistent with simulation results from the sparse-bridging literature. Below ~20 bridge legislators, affine fits become unreliable due to outlier sensitivity.

3. **Session B as reference scale.** The most recent session is the standard reference — DW-NOMINATE uses the most recent Congress as the endpoint.

One choice is simpler than the literature norm:

4. **Z-score standardization for cross-session prediction.** The transfer learning literature recommends domain adaptation techniques (covariate shift correction, TrAdaBoost). For same-institution adjacent sessions with 78% overlap, z-scores are likely sufficient. Would revisit if cross-session AUC drops more than 0.10 below within-session AUC.

---

## 6. Pipeline Integration Assessment

### 6.1 Upstream Dependencies

Cross-session reads completed results from both sessions via the `latest` symlinks:

| Upstream Phase | What It Reads | Required? |
|---------------|--------------|-----------|
| **IRT (Phase 4)** | `irt_{chamber}.parquet` → ideal points (xi_mean, xi_sd) | **Yes** — alignment requires ideal points |
| **Prediction (Phase 7)** | `vote_features_{chamber}.parquet`, `holdout_results_{chamber}.parquet` | Optional (behind `--skip-prediction`) |
| **Synthesis (Phase 11)** | `legislator_df_{chamber}.parquet` → joined metrics | **Yes** — metric stability and detection validation |

If IRT or Synthesis is missing for a chamber, that chamber is skipped with a warning. Prediction data being missing only skips the prediction transfer analysis.

### 6.2 Downstream Consumers (None Currently)

Cross-session results are currently standalone — not consumed by other phases. The design doc identifies two integration points:

1. **Synthesis** could add a "Cross-Session" section when multiple sessions are available, joining ideology shift and stability metrics into the legislator DataFrame.
2. **Profiles** could include a "Historical Comparison" section showing ideology trajectory.

Neither is implemented. Both would add value but are not blockers for the first run.

### 6.3 Invocation

```bash
just cross-session                                         # Default: 2023-24 vs 2025-26
just cross-session --session-a 2023-24 --session-b 2025-26 # Explicit
just cross-session --chambers house                        # Single chamber
just cross-session --skip-prediction                       # Faster (skip XGBoost)
```

### 6.4 Output Structure

```
results/kansas/cross-session/90th-vs-91st/validation/YYYY-MM-DD/
├── data/
│   ├── ideology_shift_{chamber}.parquet
│   ├── metric_stability_{chamber}.parquet
│   ├── prediction_transfer_{chamber}.parquet
│   ├── feature_importance_{chamber}.parquet
│   ├── turnover_impact_{chamber}.json
│   └── detection_validation.json
├── plots/
│   ├── ideology_shift_scatter_{chamber}.png
│   ├── biggest_movers_{chamber}.png
│   ├── shift_distribution_{chamber}.png
│   ├── turnover_impact_{chamber}.png
│   ├── prediction_comparison_{chamber}.png
│   └── feature_importance_comparison_{chamber}.png
├── validation_report.html
├── run_info.json
└── run_log.txt
```

### 6.5 Readiness for First Run

**Prerequisites:**
- Both 90th (2023-24) and 91st (2025-26) sessions must have completed at minimum: EDA, PCA, IRT, Clustering, Network, Prediction, Indices, Synthesis.
- The 91st session data exists and has been run through the full pipeline (confirmed in roadmap).
- The 90th session was run through 11 phases (confirmed in roadmap, "90th Biennium Pipeline Run" on 2026-02-22).

**Status:** Ready for first run. Both sessions have completed upstream phases.

---

## 7. Future Work: The 3+ Session Horizon

### 7.1 When Pairwise Alignment Breaks Down

Affine alignment is pairwise: it maps Session A onto Session B's scale. With three sessions (A, B, C), you can chain: align A→B, then B→C. But chaining accumulates error — alignment noise in A→B propagates into the A→C comparison.

With four or more sessions, the number of pairwise comparisons grows quadratically. Chain alignment becomes unwieldy. The standard solution is **joint dynamic estimation**.

### 7.2 Martin-Quinn Dynamic IRT in PyMC

The random walk prior on ideal points is straightforward to implement in PyMC:

```python
# Pseudocode — not production
with pm.Model():
    # Innovation standard deviation
    tau = pm.HalfNormal("tau", sigma=0.5)

    # Initial ideal points (session 1)
    xi_0 = pm.Normal("xi_0", mu=0, sigma=1, shape=n_legislators)

    # Random walk across sessions
    xi = [xi_0]
    for t in range(1, n_sessions):
        xi_t = pm.Normal(f"xi_{t}", mu=xi[t-1], sigma=tau, shape=n_legislators)
        xi.append(xi_t)

    # Standard 2PL IRT likelihood per session
    for t in range(n_sessions):
        alpha_t = pm.Normal(f"alpha_{t}", mu=0, sigma=1, shape=n_bills[t])
        beta_t = pm.Normal(f"beta_{t}", mu=0, sigma=5, shape=n_bills[t])
        logit_p = alpha_t[bill_idx[t]] * (xi[t][leg_idx[t]] - beta_t[bill_idx[t]])
        pm.Bernoulli(f"votes_{t}", logit_p=logit_p, observed=votes[t])
```

This produces full posterior distributions for each legislator in each session, with movements quantified as posterior differences. No post-hoc alignment needed — the model handles identification through the random walk constraint.

**Practical considerations:**
- Requires legislator-session mapping (who served when) — our `match_legislators()` provides this.
- Computational cost scales linearly with sessions but quadratically with MCMC dimensions.
- Identification: constrain tau or fix one legislator per session.
- Our PCA-informed initialization (ADR-0023) could extend to multi-session.

### 7.3 Content-Based Anchoring

Bateman, Clinton & Lapinski (2017) match identical or similar bills across sessions, using shared bill content as anchors rather than shared legislators. This sidesteps the bridge legislator stability assumption entirely.

Kansas has recurring legislation (budget bills, reauthorizations) that could serve as content anchors. This requires:
- Bill text scraping (see `docs/future-bill-text-analysis.md`)
- Text similarity matching (TF-IDF or embedding-based)
- Joint IRT with bill parameters constrained to be equal for matched bills

This is a substantial project but becomes more valuable as the number of sessions grows and legislator overlap decreases (turnover).

### 7.4 Wasserstein Distance for Posterior Overlap

The deferred Wasserstein distance metric (design doc § 3) would account for estimation uncertainty in shift quantification. A legislator with wide credible intervals in both sessions might show delta_xi = 0.5 but overlapping posteriors — not a meaningful shift. A precisely-estimated legislator with the same delta is a genuine mover.

Implementation requires loading ArviZ InferenceData from both sessions and aligning full posterior chains (not just point estimates). The scipy `wasserstein_distance` function handles the computation. The main challenge is I/O: MCMC traces are large (hundreds of MB per session).

---

## 8. Recommendations

### 8.1 Immediate (First Run)

1. **Run cross-session validation.** Both sessions' upstream phases are complete. `just cross-session` should produce the first real results.
2. **Review alignment quality.** Check Pearson/Spearman r between aligned ideal points (expect r > 0.85). If r < 0.70, investigate whether the policy space shifted.
3. **Spot-check movers.** Do the flagged "biggest movers" make political sense? Kansas political knowledge provides ground truth.

### 8.2 Near-Term Enhancements

4. **Extract XGBoost hyperparameters** to a module-level constant (duplicated in two places).
5. **Document tau asymmetry** in `compare_feature_importance()` docstring.
6. **Add `report.section_count` property** (project-wide, not cross-session-specific).

### 8.3 Future Sessions (When 3+ Available)

7. **Implement Martin-Quinn dynamic IRT** as an alternative to chained affine alignment.
8. **Explore content-based anchoring** once bill text scraping is available.
9. **Add Wasserstein distance** for posterior-aware shift quantification.

---

## 9. References

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
- Bateman, D.A., J.D. Clinton, and J.S. Lapinski. 2017. "A House Divided? Roll Calls, Polarization, and Policy Differences in the U.S. House, 1877–2011." *AJPS* 61(3), 698–714.
- Gray, T. 2020. "A Bridge Too Far? Examining Bridging Assumptions in Common-Space Estimations." *Legislative Studies Quarterly* 45(2).
- Groseclose, T., S.D. Levitt, and J.M. Snyder. 1999. "Comparing Interest Group Scores across Time and Chambers: Adjusted ADA Scores for the U.S. Congress." *APSR* 93(1), 33–50.
- Lewis, J.B. and L. Sonnet. Working paper. "Estimating NOMINATE scores over time using penalized splines."
- McCarty, N. 2011. "Measuring Legislative Preferences." In *The Oxford Handbook of the American Congress*, edited by E. Schickler and F.E. Lee. Oxford University Press.

### Validation, Stability, and Drift

- Remmel, M.L. and J.J. Mondak. 2020. "Three Validation Tests of the Shor-McCarty State Legislator Ideology Data." *American Politics Research*.
- Imai, K., J. Lo, and J. Olmsted. 2016. "Fast Estimation of Ideal Points with Massive Data." *APSR* 110(4), 631–656.
- Shin, S., J. Lim, and J.H. Park. 2025. "L1-based Bayesian Ideal Point Model for Roll Call Data." *JASA* 120(550), 631–644.
- Bonica, A. 2014. "Mapping the Ideological Marketplace." *AJPS* 58(2), 367–386. (DIME/CFscores)
- Caughey, D. and C. Warshaw. 2015. "Dynamic Estimation of Latent Opinion Using a Hierarchical Group-Level IRT Model." *Political Analysis* 23(2), 197–211.
- Minozzi, W. and C. Volden. 2021. "Measuring Party Loyalty." *Political Science Research and Methods* 9(2), 351–367.
- Koo, T.K. and M.Y. Li. 2016. "A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability Research." *Journal of Chiropractic Medicine* 15(2), 155–163.
- Ornstein, J., E. Blasingame, and J. Truscott. 2024. "How to Train Your Stochastic Parrot: Large Language Models for Political Texts." Working paper.

### Open-Source Projects

- Voteview data: https://voteview.com/data
- pynominate: https://github.com/voteview/pynominate
- py-irt: https://github.com/nd-ball/py-irt
- tbip (Text-Based Ideal Points): https://github.com/keyonvafa/tbip
- emIRT: https://github.com/kosukeimai/emIRT
- idealstan: https://github.com/saudiwin/idealstan
- MCMCpack: https://CRAN.R-project.org/package=MCMCpack
- Rvoteview: https://github.com/voteview/Rvoteview
- pylegiscan: https://github.com/poliquin/pylegiscan
- legcop: https://pypi.org/project/legcop/
- Open States: https://docs.openstates.org/api-v3/
- DIME (CFscores): https://data.stanford.edu/dime
- Martin-Quinn Scores: https://mqscores.wustl.edu/
- Bailey Bridge Ideal Points: https://michaelbailey.georgetown.domains/bridge-ideal-points-2020/
- Shor-McCarty data: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GZJOT3
- Stan ideal point examples: https://jrnold.github.io/bugs-examples-in-stan/legislators.html
- Bateman congressional data: http://www.davidalexbateman.net/congressional-data.html
- Splink (entity resolution): https://github.com/moj-analytical-services/splink
- dedupe (ML entity resolution): https://github.com/dedupeio/dedupe
- polars-fuzzy-match: https://pypi.org/project/polars-fuzzy-match/
- NumPyro TBIP tutorial: https://num.pyro.ai/en/stable/tutorials/tbip.html
