# Roadmap

What's been done, what's next, and what's on the horizon for the KS Vote Scraper analytics pipeline.

**Last updated:** 2026-02-23 (after PCA init default, parallelism experiment, --cores flag)

---

## Completed Phases

| # | Phase | Date | Key Finding |
|---|-------|------|-------------|
| 1 | EDA | 2026-02-19 | 82% Yea base rate, 72% R supermajority, 34 veto overrides |
| 2 | PCA | 2026-02-19 | PC1 = party (57% variance), PC2 = contrarianism (Tyson/Thompson) |
| 3 | Bayesian IRT | 2026-02-20 | 1D ideal points converge cleanly; Tyson paradox identified |
| 4 | Clustering | 2026-02-20 | k=2 optimal (party split); intra-R variation is continuous, not factional |
| 5 | Network | 2026-02-20 | Zero cross-party edges at kappa=0.40; Schreiber sole bipartisan bridge |
| 6 | Prediction | 2026-02-20 | Vote AUC=0.98; IRT features do all the work; XGBoost adds nothing over logistic |
| 7 | Classical Indices | 2026-02-21 | Rice, CQ unity, ENP, weighted maverick; Schreiber/Dietrich top mavericks |
| 2b | UMAP | 2026-02-22 | Nonlinear ideological landscape; validates PCA/IRT; most accessible visualization |
| 6+ | NLP Bill Text Features | 2026-02-22 | NMF topics on short_title; House temporal AUC 0.90→0.96, Senate 0.86→0.96 |
| — | Synthesis Report | 2026-02-22 | 32-section narrative HTML; joins all 8 phases into one deliverable |
| — | Legislator Profiles | 2026-02-22 | Per-legislator deep-dives: scorecard, bill-type breakdown, defections, neighbors, surprising votes |
| 7b | Beta-Binomial Party Loyalty | 2026-02-22 | Bayesian shrinkage on CQ unity; empirical Bayes, closed-form posteriors, 4 plots per chamber |
| — | Cross-Biennium Portability | 2026-02-22 | Removed all hardcoded legislator names from general phases; full pipeline re-run validated |
| — | Visualization Improvement Pass | 2026-02-22 | All 6 phases (Network, IRT, Prediction, PCA, Clustering, EDA) retrofitted for nontechnical audience; plain-English titles, annotated findings, data-driven highlights |
| — | Missing Votes Visibility | 2026-02-22 | Auto-injected "Missing Votes" section in every HTML report + standalone `missing_votes.md` in data directory; close votes bolded |
| 8 | Hierarchical Bayesian IRT | 2026-02-22 | 2-level partial pooling by party, non-centered parameterization, ICC variance decomposition, shrinkage vs flat IRT |
| — | 90th Biennium Pipeline Run | 2026-02-22 | Full 11-phase pipeline on 2023-24 data (94K votes, 168 legislators); cross-biennium analysis now possible |
| — | ty Type Checker | 2026-02-22 | Two-tier policy: scraper strict (0 errors), analysis warnings-only; caught 2 real type bugs on first run (ADR-0018) |
| — | Historical Session Support | 2026-02-22 | 2011-2026 coverage: JS bill discovery fallback, ODT vote parser, pre-2015 party detection (ADR-0020) |
| — | Independent Party Handling | 2026-02-22 | Pipeline-wide null-party fill, PARTY_COLORS in all 12 modules, dynamic plots, Independent exclusion from party-specific models (ADR-0021) |
| — | 89th Biennium Pipeline Run | 2026-02-22 | Full 12-phase pipeline on 2021-22 data; Dennis Pyle (Independent) handled correctly across all phases |
| — | PCA-Informed IRT Init (Default) | 2026-02-23 | Fixes 5/16 convergence failures; literature-backed (Jackman pscl::ideal); `--no-pca-init` to disable (ADR-0023) |
| — | Parallelism Performance Experiment | 2026-02-23 | `cores=n_chains` was already PyMC default; batch-job CPU contention was the real cause; sequential chains 1.8x slower due to thermal throttling (ADR-0022 addendum) |

---

## Next Up

### 1. MCMC Performance Experiment: Sequential vs Parallel Chains (Complete Runs)

**Priority:** High — needed to establish reliable batch-run practices.
**Status:** Partial results from 91st Legislature. Needs re-run on a smaller biennium to get complete joint model timings.

The initial experiment (`results/experiments/2026-02-23_parallel-chains-performance/`) produced valuable but incomplete results. The 91st Legislature's joint cross-chamber model (172 legislators, 491 votes, 43,612 observations) was killed after 2+ hours without completing, making it impractical as an experiment subject.

**What we learned so far:**
- `cores=n_chains` is equivalent to PyMC's default (not a new change)
- Sequential chains (`cores=1`) are ~1.8x slower than parallel due to thermal throttling on M3 Pro
- Batch-running multiple bienniums simultaneously caused the original slowdown (CPU contention)

**What remains:**
- Re-run the full 4-run experiment on a smaller biennium to get joint model completion times
- Validate that the thermal throttling pattern holds across different model sizes
- Establish baseline timings for joint models to set expectations for batch jobs

**Recommended experiment bienniums** (sorted by size, smallest first):

| Biennium | Votes CSV rows | House model | Senate model | Hier. House time |
|----------|---------------|-------------|--------------|-----------------|
| **88th (2019-20)** | 45K | 127 × 141 | 41 × 108 | ~176s |
| 84th (2011-12) | 68K | 113 × 260 | 37 × 254 | ~600s |
| 86th (2015-16) | 79K | 129 × 236 | 40 × 225 | ~442s |

**Use the 88th (2019-20) for experiments.** It is the smallest biennium by a wide margin (~45K votes vs 68K+ for all others), completes per-chamber models in under 3 minutes, and its joint model completed in 23 minutes with `cores=2`. This makes it feasible to run all 4 experiment configurations (sequential/parallel × with/without joint) in under 2 hours total.

Avoid the 91st (2025-26) for experiments — it has the most votes per legislator and its joint model takes 2+ hours. The 90th (2023-24) is even larger at 95K votes.

The experiment template and infrastructure are in `results/experiments/TEMPLATE.md`. New experiments go in `results/experiments/YYYY-MM-DD_short-description/experiment.md`.

### 2. Cross-Session Validation (Feature Complete)

**Priority:** High — the single biggest gap in current results.
**Status:** All 7 implementation steps complete (data layer, plots, report builder, CLI, prediction transfer, detection validation, docs). 55 tests. Ready for first real run once both sessions' upstream phases are complete.

Four distinct analyses become possible now that both bienniums are scraped:

- **Temporal comparison (who moved?):** Compare IRT ideal points for returning legislators across bienniums. Who shifted ideology? Are the 2025-26 mavericks (Schreiber, Dietrich) the same people who were mavericks in 2023-24? This is the most newsworthy output for the nontechnical audience — "Senator X moved 1.2 points rightward since last session" is a concrete, actionable finding.
- **Metric stability:** Cross-session correlations for party unity, maverick rates, network centrality, and other legislative metrics. High Pearson/Spearman r for returning legislators indicates these measures capture stable traits, not session-specific noise.
- **Prediction honesty (out-of-sample):** Train vote prediction on 2023-24, test on 2025-26 (and vice versa). This is the gold standard for prediction validation — within-session holdout (AUC=0.98) is optimistic because the model sees the same legislators and session dynamics. Cross-session tests whether the learned patterns generalize. SHAP feature importance rankings compared via Kendall's tau.
- **Detection threshold validation:** The synthesis detection thresholds (unity > 0.95 skip, rank gap > 0.5 for paradox, betweenness within 1 SD for bridge) were calibrated on 2025-26. Running synthesis on 2023-24 tests whether they produce sensible results on a different session with potentially different partisan dynamics. If they don't, the thresholds need to become adaptive or session-parameterized.

### 3. MCA (Multiple Correspondence Analysis)

**Priority:** Medium — alternative view on the vote matrix.

Method documented in `Analytic_Methods/10_DIM_correspondence_analysis.md`. MCA treats each vote as a categorical variable rather than numeric, which is technically more appropriate for Yea/Nay data:

- Complementary to PCA: PCA assumes continuous, MCA assumes categorical
- May reveal structure PCA misses, especially in voting patterns where abstention is meaningful
- `prince` library already in `pyproject.toml`
- Compare MCA dimensions to PCA PC1/PC2 — if they agree, PCA's linear assumption is validated

### 4. Time Series Analysis

**Priority:** Medium — adds temporal depth to static snapshots.

Two methods documented but not yet implemented:

- **Ideological drift** (`Analytic_Methods/26_TSA_ideological_drift.md`): Rolling IRT or rolling party unity within a session. Did anyone change position mid-session? Track 15-vote rolling windows.
- **Changepoint detection** (`Analytic_Methods/27_TSA_changepoint_detection.md`): Structural breaks in voting patterns. When did the session's character shift? (e.g., pre- vs post-veto override period)

Requires the `ruptures` library (already in `pyproject.toml`). Becomes much more powerful once 2023-24 data is available for cross-session comparison.

### 5. 2D Bayesian IRT Model

**Priority:** Medium — solves the Tyson paradox properly.

The 1D model compresses Tyson's two-dimensional behavior (ideology + contrarianism) into one axis. A 2D model would:
- Place Tyson as (very conservative on Dim 1, extreme outlier on Dim 2)
- Improve predictions for legislators with unusual PC2 patterns
- Validate whether the PCA PC2 "contrarianism" dimension has a Bayesian counterpart

This is computationally expensive (doubles MCMC time) and requires careful identification constraints for rotation.

---

## Deferred / Low Priority

### Joint Cross-Chamber IRT

A full joint MCMC model was attempted and failed: 71 shared bills for 169 legislators is too sparse. Currently using classical test equating (A=1.136, B=-0.305). Revisit only if a future session has significantly more shared bills, or if the 2023-24 session provides additional bridging data.

### Latent Class Mixture Models

Documented in `Analytic_Methods/28_CLU_latent_class_mixture_models.md`. Probabilistic alternative to k-means for discrete faction discovery. Low priority because clustering already showed within-party variation is continuous, not factional — there aren't discrete factions to discover.

### Dynamic Ideal Points (Martin-Quinn)

Track legislator positions over time within a session using a state-space model. Deferred until cross-session data is available, where it becomes much more powerful (track a legislator across bienniums).

### Bipartite Bill-Legislator Network

Documented in `Analytic_Methods/21_NET_bipartite_bill_legislator.md`. Two-mode network connecting legislators to bills they voted on. Intentionally deferred — the Kappa-based co-voting network already captures the same information more efficiently.

### Posterior Predictive Checks (Standalone)

Documented in `Analytic_Methods/17_BAY_posterior_predictive_checks.md`. Already partially integrated into the IRT phase (PPC plots in the IRT report). A standalone, cross-model PPC comparison could be useful once the hierarchical model and 2D IRT are implemented.

### Analysis Phase Primers

Each results directory should have a `README.md` explaining the analysis for non-code readers. Low priority — the HTML reports serve this role for now, but standalone primers would be useful for the `results/` directory.

### Test Suite Expansion

712 tests exist across scraper (219) and analysis (493) modules. All passing. Coverage could be expanded:
- Integration tests that run a mini end-to-end pipeline on fixture data
- Cross-session tests (once 2023-24 is scraped) to verify scripts handle multiple sessions
- Snapshot tests for HTML report output stability

---

## Explicitly Rejected

| Method | Why |
|--------|-----|
| W-NOMINATE (`Analytic_Methods/12_DIM_w_nominate.md`) | R-only; project policy is Python-only (no rpy2) |
| Optimal Classification (`Analytic_Methods/13_DIM_optimal_classification.md`) | R-only; same as above |

---

## Scraper Maintenance

| Item | When | Details |
|------|------|---------|
| Update `CURRENT_BIENNIUM_START` | 2027 | Change from 2025 to 2027 in `session.py` |
| Add special sessions | As needed | Add year to `SPECIAL_SESSION_YEARS` in `session.py` |
| ~~Fix Shallenburger suffix~~ | ~~Done~~ | Fixed in analysis via `strip_leadership_suffix()` in `run_context.py` — applied at every CSV load point across all phases (ADR-0014). Scraper stores the raw name; analysis handles display. |

---

## All 29 Analytic Methods — Status

| # | Method | Category | Status |
|---|--------|----------|--------|
| 01 | Data Loading & Cleaning | DATA | Completed (EDA) |
| 02 | Vote Matrix Construction | DATA | Completed (EDA) |
| 03 | Descriptive Statistics | EDA | Completed (EDA) |
| 04 | Missing Data Analysis | EDA | Completed (EDA) |
| 05 | Rice Index | IDX | Completed (Indices) |
| 06 | Party Unity | IDX | Completed (Indices) |
| 07 | ENP | IDX | Completed (Indices) |
| 08 | Maverick Scores | IDX | Completed (Indices) |
| 09 | PCA | DIM | Completed (PCA) |
| 10 | MCA / Correspondence Analysis | DIM | **Planned** — item #4 above |
| 11 | UMAP / t-SNE | DIM | Completed (UMAP, Phase 2b) |
| 12 | W-NOMINATE | DIM | Rejected (R-only) |
| 13 | Optimal Classification | DIM | Rejected (R-only) |
| 14 | Beta-Binomial Party Loyalty | BAY | Completed (Beta-Binomial, Phase 7b) |
| 15 | Bayesian IRT (1D) | BAY | Completed (IRT) |
| 16 | Hierarchical Bayesian Model | BAY | Completed (Hierarchical IRT, Phase 8) |
| 17 | Posterior Predictive Checks | BAY | Partial (embedded in IRT) |
| 18 | Hierarchical Clustering | CLU | Completed (Clustering) |
| 19 | K-Means / GMM Clustering | CLU | Completed (Clustering) |
| 20 | Co-Voting Network | NET | Completed (Network) |
| 21 | Bipartite Bill-Legislator Network | NET | Deferred (redundant) |
| 22 | Centrality Measures | NET | Completed (Network) |
| 23 | Community Detection | NET | Completed (Network) |
| 24 | Vote Prediction | PRD | Completed (Prediction) |
| 25 | SHAP Analysis | PRD | Completed (Prediction) |
| 26 | Ideological Drift | TSA | **Planned** — item #5 above |
| 27 | Changepoint Detection | TSA | **Planned** — item #5 above |
| 28 | Latent Class Mixture Models | CLU | Deferred (no discrete factions found) |

**Score: 20 completed, 2 rejected, 3 planned, 2 deferred, 1 partial = 29 total** (was 19+1 after hierarchical IRT)

---

## Key Architectural Decisions Still Standing

- **Polars over pandas** everywhere
- **Python over R** — no rpy2, no W-NOMINATE
- **Ruff + ty + uv** — all-Astral toolchain (lint, type check, package management)
- **IRT ideal points are the primary feature** — prediction confirmed this; everything else is marginal
- **Chambers analyzed separately** unless explicitly doing cross-chamber comparison
- **Tables never truncated** in analysis reports — full data for initial analysis
- **Nontechnical audience** — all visualizations self-explanatory, plain-English labels, annotated findings
