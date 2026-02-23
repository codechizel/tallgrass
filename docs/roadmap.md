# Roadmap

What's been done, what's next, and what's on the horizon for the KS Vote Scraper analytics pipeline.

**Last updated:** 2026-02-22 (after ty type checker adoption + 90th biennium pipeline run)

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

---

## Next Up

### 1. Cross-Session Validation (In Progress)

**Priority:** High — the single biggest gap in current results.
**Status:** Data layer + plots + CLI + report builder complete. Prediction transfer remaining (behind `--skip-prediction` flag). Ready for first real run once both sessions' upstream phases are complete.

Three distinct analyses become possible now that both bienniums are scraped:

- **Prediction honesty (out-of-sample):** Train vote prediction on 2023-24, test on 2025-26 (and vice versa). This is the gold standard for prediction validation — within-session holdout (AUC=0.98) is optimistic because the model sees the same legislators and session dynamics. Cross-session tests whether the learned patterns generalize. Also solves the Senate bill passage small-N problem (59 test bills in 2025-26 is too few; stacking sessions doubles the data).
- **Temporal comparison (who moved?):** Compare IRT ideal points for returning legislators across bienniums. Who shifted ideology? Are the 2025-26 mavericks (Schreiber, Dietrich) the same people who were mavericks in 2023-24? This is the most newsworthy output for the nontechnical audience — "Senator X moved 1.2 points rightward since last session" is a concrete, actionable finding.
- **Detection threshold validation:** The synthesis detection thresholds (unity > 0.95 skip, rank gap > 0.5 for paradox, betweenness within 1 SD for bridge) were calibrated on 2025-26. Running synthesis on 2023-24 tests whether they produce sensible results on a different session with potentially different partisan dynamics. If they don't, the thresholds need to become adaptive or session-parameterized.

### 4. MCA (Multiple Correspondence Analysis)

**Priority:** Medium — alternative view on the vote matrix.

Method documented in `Analytic_Methods/10_DIM_correspondence_analysis.md`. MCA treats each vote as a categorical variable rather than numeric, which is technically more appropriate for Yea/Nay data:

- Complementary to PCA: PCA assumes continuous, MCA assumes categorical
- May reveal structure PCA misses, especially in voting patterns where abstention is meaningful
- `prince` library already in `pyproject.toml`
- Compare MCA dimensions to PCA PC1/PC2 — if they agree, PCA's linear assumption is validated

### 5. Time Series Analysis

**Priority:** Medium — adds temporal depth to static snapshots.

Two methods documented but not yet implemented:

- **Ideological drift** (`Analytic_Methods/26_TSA_ideological_drift.md`): Rolling IRT or rolling party unity within a session. Did anyone change position mid-session? Track 15-vote rolling windows.
- **Changepoint detection** (`Analytic_Methods/27_TSA_changepoint_detection.md`): Structural breaks in voting patterns. When did the session's character shift? (e.g., pre- vs post-veto override period)

Requires the `ruptures` library (already in `pyproject.toml`). Becomes much more powerful once 2023-24 data is available for cross-session comparison.

### 6. 2D Bayesian IRT Model

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

596 tests exist across scraper (146) and analysis (450) modules. Coverage could be expanded:
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
| ~~Fix Shallenburger suffix~~ | ~~Done~~ | Fixed in analysis via `_build_display_labels()` — strips " - " suffixes before name extraction (ADR-0014). Scraper stores the raw name; analysis handles display. |

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
