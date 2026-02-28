# Roadmap

What's been done, what's next, and what's on the horizon for the Tallgrass analytics pipeline.

**Last updated:** 2026-02-28 (after 2D IRT pipeline integration as phase 04b)

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
| — | Synthesis Report | 2026-02-22 | 29-32-section narrative HTML; joins all 10 phases into one deliverable |
| — | Synthesis Deep Dive | 2026-02-25 | Code audit, field survey, 9 fixes (dynamic AUC, minority mavericks, data extraction, 47 tests). ADR-0034. |
| — | Legislator Profiles | 2026-02-22 | Per-legislator deep-dives: scorecard, bill-type breakdown, defections, neighbors, surprising votes. Name-based lookup (`--names`) added 2026-02-25. |
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
| — | PCA-Informed IRT Init (Default) | 2026-02-23 | Prevents reflection mode-splitting; literature-backed (Jackman pscl::ideal); `--no-pca-init` to disable (ADR-0023). Combined with nutpie (ADR-0053), all 16/16 flat IRT sessions now converge. |
| — | Parallelism Performance Experiment | 2026-02-23 | `cores=n_chains` was already PyMC default; batch-job CPU contention was the real cause; sequential chains 1.8x slower due to thermal throttling (ADR-0022 addendum) |
| — | Analysis Primer | 2026-02-24 | `docs/analysis-primer.md`: plain-English guide to the 13-step pipeline for general audiences (journalists, policymakers, citizens) |
| — | Parallelism Experiment (Complete) | 2026-02-24 | 88th Legislature 4-run experiment: parallel 1.83-1.89x faster; convergence bit-identical; OMP_NUM_THREADS=6 cap applied (ADR-0022). Writeup: `docs/apple-silicon-mcmc-tuning.md` |
| — | Full 91st Pipeline with Joint Model | 2026-02-24 | 12/12 phases succeeded including hierarchical joint cross-chamber model; first complete run with joint model. Re-run 2026-02-26 with bill-matching: 39m 36s total (joint 31 min), Senate all checks passed. |
| — | Landscape Survey & Method Evaluation | 2026-02-24 | `docs/landscape-legislative-vote-analysis.md` and `docs/method-evaluation.md`: surveyed the field, evaluated all major methods, identified external validation as the priority gap |
| — | External Validation (Shor-McCarty) | 2026-02-24 | Shor-McCarty external validation phase: name matching, Pearson/Spearman correlations, scatter plots, outlier analysis. 5 overlapping bienniums (84th-88th). ADR-0025. |
| — | External Validation Results Article | 2026-02-24 | `docs/external-validation-results.md`: general-audience article explaining SM validation results (flat House r=0.981, flat Senate r=0.929, hierarchical Senate r=-0.541) |
| — | Hierarchical Shrinkage Deep Dive | 2026-02-24 | `docs/hierarchical-shrinkage-deep-dive.md`: literature-grounded analysis of J=2 over-shrinkage problem (Gelman 2006/2015, James-Stein, Peress 2009), 6 remedies proposed |
| — | IRT Deep Dive & Field Survey | 2026-02-25 | `docs/irt-deep-dive.md` and `docs/irt-field-survey.md`: field survey of IRT implementations, code audit, identification problem, unconstrained β contribution, Python ecosystem gap. Implemented: tail-ESS, shrinkage warning, sign-constraint removal, 28 new tests (853 total). |
| 2c | MCA (Multiple Correspondence Analysis) | 2026-02-25 | Categorical-data analogue of PCA using chi-square distance; Yea/Nay/Absent as 3 categories; prince library; Greenacre correction; PCA validation (Spearman r), horseshoe detection, biplot, absence map. 34 new tests. Deep dive: `docs/mca-deep-dive.md`, design: `analysis/design/mca.md`. |
| — | Hierarchical IRT Fixes (Bill-Matching + Adaptive Priors) | 2026-02-26 | Joint model bill-matching (ADR-0043): 71 shared bills bridge chambers via concurrent calibration. Group-size-adaptive priors fix Senate-D convergence. Joint model runtime 93 min → 31 min. ADR-0042, ADR-0043. |
| 9 | Cross-Session Validation (90th vs 91st) | 2026-02-26 | First post-fix run: ideology r=0.940 (House), 0.975 (Senate). Cross-session prediction AUC 0.967-0.976 (nearly matches within-session 0.975-0.984). 94 tests. IRT ideal points confirmed as stable traits; network centrality metrics confirmed session-specific. Tyson flagged as paradox in both bienniums. |
| — | PCA-Informed Init for Hierarchical IRT | 2026-02-26 | Experiment proved PCA init fixes House R-hat (1.0102→1.0026) with r=0.999996 agreement. Implemented as default in `build_per_chamber_model()`. Per-chain ESS reporting added. ADR-0044. Article: `docs/hierarchical-pca-init-experiment.md`. |
| — | 4-Chain Hierarchical IRT Experiment | 2026-02-26 | 4 chains resolve both ESS warnings (xi: 397→564, mu_party: 356→512) at +4% wall time. Discovered jitter mode-splitting: `jitter+adapt_diag` causes R-hat ~1.53 with 4 chains; fix is `adapt_diag` with PCA init. Run 3 unnecessary. Article: `docs/hierarchical-4-chain-experiment.md`. |
| 4b | 2D Bayesian IRT (Pipeline, Experimental) | 2026-02-26 (experiment), 2026-02-28 (pipeline) | M2PL model with PLT identification to resolve Tyson paradox. Pipeline phase 04b: both chambers, nutpie sampling, RunContext/HTML report, relaxed convergence thresholds. Deep dive: `docs/2d-irt-deep-dive.md`, design: `analysis/design/irt_2d.md`, ADR-0046, ADR-0054. |

---

## Next Up

### 1. Time Series Analysis

**Priority:** Medium — adds temporal depth to static snapshots. Now that cross-session validation confirms IRT ideal points are stable across bienniums, temporal analysis can build on that foundation.

Two methods documented but not yet implemented:

- **Ideological drift** (`Analytic_Methods/26_TSA_ideological_drift.md`): Rolling IRT or rolling party unity within a session. Did anyone change position mid-session? Track 15-vote rolling windows.
- **Changepoint detection** (`Analytic_Methods/27_TSA_changepoint_detection.md`): Structural breaks in voting patterns. When did the session's character shift? (e.g., pre- vs post-veto override period)

Requires the `ruptures` library (not yet in `pyproject.toml`). Now that cross-session data is available, both methods become immediately actionable.

---

## Deferred / Low Priority

### ~~Joint Cross-Chamber IRT~~

~~A full joint MCMC model was attempted and failed.~~ **Completed (2026-02-24), fixed (2026-02-26).** The hierarchical joint cross-chamber model now runs with bill-matching (ADR-0043): shared `alpha`/`beta` parameters for 71 matched bills (91st) provide natural cross-chamber identification. Bill-matching reduced joint model runtime from 93 min to 31 min by shrinking the problem size (420 unified votes vs 491). Group-size-adaptive priors mitigate small-group convergence failures. Senate convergence now passes all checks. See ADR-0043 and `docs/joint-hierarchical-irt-diagnosis.md`.

### DIME/CFscores External Validation (Second Source)

Campaign-finance-based ideology from Bonica's DIME project ([data.stanford.edu/dime](https://data.stanford.edu/dime)) provides a completely independent data source — it captures who donors think you are, not how you vote. However, within-party correlation with Shor-McCarty is only 0.65-0.67. For our Kansas analysis where intra-Republican variation is the primary interest, CFscores may lack resolution. Pursue after Shor-McCarty validation succeeds. See `docs/method-evaluation.md`.

### Latent Class Mixture Models

Documented in `Analytic_Methods/28_CLU_latent_class_mixture_models.md`. Probabilistic alternative to k-means for discrete faction discovery. Low priority because clustering already showed within-party variation is continuous, not factional — there aren't discrete factions to discover.

### Dynamic Ideal Points (Martin-Quinn)

Track legislator positions over time within a session using a state-space model. Deferred until cross-session data is available, where it becomes much more powerful (track a legislator across bienniums).

### Bipartite Bill-Legislator Network

Documented in `Analytic_Methods/21_NET_bipartite_bill_legislator.md`. Two-mode network connecting legislators to bills they voted on. Intentionally deferred — the Kappa-based co-voting network already captures the same information more efficiently.

### Posterior Predictive Checks (Standalone)

Documented in `Analytic_Methods/17_BAY_posterior_predictive_checks.md`. Already partially integrated into the IRT phase (PPC plots in the IRT report). A standalone, cross-model PPC comparison could be useful once the hierarchical model and 2D IRT are implemented.

### Per-Phase Results Primers

Each results directory should have a `README.md` explaining the analysis for non-code readers. Low priority — the HTML reports serve this role for now, and the project-level primer (`docs/analysis-primer.md`) provides the general-audience overview. Standalone per-phase primers would still be useful for the `results/` directory.

### Test Suite Expansion

1172 tests exist across scraper (~219) and analysis (~953, including 94 cross-session) modules. All passing. Coverage could be expanded:
- Integration tests that run a mini end-to-end pipeline on fixture data
- Cross-session tests (once 2023-24 is scraped) to verify scripts handle multiple sessions
- Snapshot tests for HTML report output stability

---

## Explicitly Rejected

| Method | Why |
|--------|-----|
| W-NOMINATE (`Analytic_Methods/12_DIM_w_nominate.md`) | R-only; project policy is Python-only (no rpy2) |
| Optimal Classification (`Analytic_Methods/13_DIM_optimal_classification.md`) | R-only; same as above |
| emIRT (fast EM ideal points) | R-only; PyMC gives full posteriors; speed not a bottleneck for single-state |
| Vote-type-stratified IRT (IssueIRT) | Data doesn't support it: overrides are party-line (98%/1%), other types too few (N < 56) |
| Strategic absence modeling (idealstan) | 2.6% absence rate, 22 "Present and Passing" instances — negligible impact |
| Dynamic IRT within biennium | 2-year window too short; cross-session handles between-biennium |
| GGUM unfolding models | No extreme-alliance voting pattern in Kansas data |
| LLM legislative agents | Too experimental; XGBoost already at 0.98 AUC |
| TBIP text-based ideal points | No full bill text available from scraper — revisit if bill text phase lands (see `docs/future-bill-text-analysis.md`) |

See `docs/method-evaluation.md` for detailed rationale on each rejection.

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
| 10 | MCA / Correspondence Analysis | DIM | Completed (MCA, Phase 2c) |
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

**Score: 22 completed, 2 rejected, 2 planned, 2 deferred, 1 partial = 29 total**

---

## Key Architectural Decisions Still Standing

- **Polars over pandas** everywhere
- **Python over R** — no rpy2, no W-NOMINATE
- **Ruff + ty + uv** — all-Astral toolchain (lint, type check, package management)
- **IRT ideal points are the primary feature** — prediction confirmed this; everything else is marginal
- **Chambers analyzed separately** unless explicitly doing cross-chamber comparison
- **Tables never truncated** in analysis reports — full data for initial analysis
- **Nontechnical audience** — all visualizations self-explanatory, plain-English labels, annotated findings
