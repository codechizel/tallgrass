# Roadmap

What's been done, what's next, and what's on the horizon for the KS Vote Scraper analytics pipeline.

**Last updated:** 2026-02-22 (after clustering viz overhaul and EDA heatmap sizing fix)

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
| — | Cross-Biennium Portability | 2026-02-22 | Removed all hardcoded legislator names from general phases; full pipeline re-run validated |

---

## Next Up

### 1. Visualization Improvement Pass

**Priority:** Highest — the nontechnical audience rule was added during Phase 7 (Indices). Phases 1-6 predate it and need retrofitting. The Synthesis Report reuses upstream plots directly, so improving them improves the deliverable.

The Indices phase is the gold standard: plain-English titles ("Who Are the Most Independent Legislators?"), annotated key actors, and report text that defines every metric before showing plots. The earlier phases have good HTML report prose but their plots are still analyst-facing. 112+ total plots across 8 phases; roughly half need improvement.

**Guiding principle:** If a finding is explained in the HTML report, it should also appear visually in at least one plot. If a legislator is flagged in `docs/analytic-flags.md`, they should have a visual highlight somewhere.

#### Network Phase — DONE

All five network visualization improvements completed (2026-02-22):
- **Bridge annotations** — red halos + yellow callout labels on high-betweenness nodes (already existed)
- **"What is betweenness?" inset** — 3-node diagram on centrality scatter (already existed)
- **Ranked bar chart** — "Who Holds the Most Influence?" with plain-English annotation (already existed)
- **Cross-party bridge before/after** — data-driven `find_cross_party_bridge()` + `plot_cross_party_bridge()` showing network with and without top cross-party connector at κ=0.30
- **Threshold sweep event markers** — default threshold + party split annotations on all 4 panels (was only on 1 each), plus "further fragmentation" marker, narrative panel titles
- **Community composition labels** — "Mostly Republican (96%, n=87)" instead of "Community 0"
- **Edge weight cross-party gap** — arrow annotation on strongest cross-party κ value, narrative title
- **Network layout narrative titles** — "Who Votes With Whom? (N legislators, N connections)" + subtitle support
- 8 new tests (3 `TestFindCrossPartyBridge`, 5 `TestCommunityLabel`)

#### IRT Phase — DONE

All four IRT visualization improvements completed (2026-02-22):
- **Paradox spotlight subplot** — `find_paradox_legislator()` detects ideologically extreme but contrarian legislators data-driven; `plot_paradox_spotlight()` produces two-panel figure (grouped bar chart of Yea rates by bill type + simplified forest plot with callout)
- **Convergence summary panel** — already existed: "The model ran N chains and they all agree" (done previously)
- **Data-driven forest highlights** — `_detect_forest_highlights()` replaces hardcoded slug annotations; detects most extreme, widest HDI, most moderate, capped at 5
- **Plain-English title** — already existed: "Where Does Each Legislator Fall on the Ideological Spectrum?" (done previously)
- **Bugfix** — fixed Python 2 `except` syntax in `run_context.py` line 100
- 8 new tests (4 `TestDetectForestHighlights`, 4 `TestFindParadoxLegislator`), 458 total passing

#### Prediction Phase — DONE

All three prediction visualization improvements completed (2026-02-22):
- **Conversational SHAP labels** — all 14 `FEATURE_DISPLAY_NAMES` rewritten for nontechnical audiences (e.g., "How conservative the legislator is" instead of "xi_mean", "Legislator–bill ideology match" instead of "Ideology × partisanship interaction"). Flows through SHAP beeswarm, SHAP bar, and XGBoost feature importance plots automatically via `_rename_shap_features()`.
- **"Hardest to Predict" spotlight** — `detect_hardest_legislators()` pure-data function + `HardestLegislator` frozen dataclass. Horizontal dot chart (`plot_hardest_to_predict()`) showing bottom 8 legislators with party-colored dots, data-driven plain-English explanations (moderate, centrist for party, strongly partisan crossover, or doesn't fit 1D model), chamber median reference line, and callout box.
- **Calibration plot annotation** — already done: "When the model says 80% chance of Yea, it's right about 80% of the time" (lightyellow callout box).
- 14 tests in `TestDetectHardestLegislators` (6 original + 8 added in review pass covering null full_name, single-party, custom n, all explanation branches, null xi_mean, field correctness), `HARDEST_N=8` added to design doc parameters table.
- **Bugfixes (review pass):** null `full_name` crash in `detect_hardest_legislators()` (`.get()` returns `None` not fallback when key exists with null value); dead code removal in `plot_hardest_to_predict()`; consistent leadership suffix stripping in `plot_per_legislator_accuracy()` and `plot_surprising_votes()`.

#### PCA Phase — DONE

All three PCA visualization improvements completed (2026-02-22):
- **Scree plot elbow annotation** — "The sharp drop means Kansas is essentially a one-dimensional legislature — party affiliation explains almost everything" (lightyellow callout with arrow)
- **PC2 axis label** — "contrarianism — voting Nay on routine, near-unanimous bills" on ideological map Y axis
- **Data-driven extreme PC2 callout** — `detect_extreme_pc2()` pure-data function + `ExtremePC2Legislator` frozen dataclass; detects >3σ outlier dynamically (Tyson at -24.8 in Senate, Parshall at -22.5 in House)
- **Bugfixes:** null `full_name` crash in outlier labels and extreme PC2 callout (`.get()` returns `None` not fallback when key exists with null value); leadership suffix stripping in outlier labels
- 6 new tests in `TestDetectExtremePC2`

#### Clustering Phase — DONE

All clustering visualization improvements completed (2026-02-22):
- **Three dendrogram alternatives**: voting blocs (sorted dot plot), polar dendrogram (circular tree with radial label staggering), icicle chart (flame chart with majority-party coloring). Original dendrograms kept as supplementary figures. See ADR-0014.
- **Centralized name extraction**: `_build_display_labels()` helper strips leadership suffixes ("Vice President of the Senate" → "Shallenburger") and disambiguates duplicate last names with first-name prefix ("Jo. Claeys" vs "J.R. Claeys").
- **Report integration**: all 3 new plot types added to clustering report with `path.exists()` guards.
- Notable-legislator annotations and report notes are data-driven (2026-02-22 portability refactor).

#### EDA Phase — DONE

All EDA visualization improvements completed (2026-02-22):
- **Heatmap sizing fix**: `size = max(8, n * 0.19)` and `fontsize = max(4, min(7, 500 / n))` — House heatmap now 24.7" with 4pt labels (up from 15.6"/3pt). Senate unchanged.
- Name labels on heatmap axes already exist. Cross-party annotation is now data-driven (2026-02-22 portability refactor).

### 2. Beta-Binomial Party Loyalty (Bayesian)

**Priority:** High — experimental code already exists.

Method documented in `Analytic_Methods/14_BAY_beta_binomial_party_loyalty.md`. Experimental implementation exists at `analysis/irt_beta_experiment.py` (558 lines) but is not integrated into the main pipeline.

- Bayesian alternative to the frequentist CQ party unity from Phase 7
- Produces credible intervals on loyalty — crucial for legislators with few party votes
- Beta(alpha, beta) posterior per legislator; shrinks noisy estimates toward the group mean
- Compare Bayesian loyalty posteriors to CQ unity point estimates
- Especially useful for Miller (30 votes) and other sparse legislators

### 3. Hierarchical Bayesian Legislator Model

**Priority:** High — the "Crown Jewel" from the methods overview.

Method documented in `Analytic_Methods/16_BAY_hierarchical_legislator_model.md`. Legislators nested within party and chamber, with partial pooling:

- Models legislator-level parameters as draws from party-level distributions
- Naturally handles the R supermajority (more data = tighter party estimate)
- Quantifies how much individual legislators deviate from their party's typical behavior
- Partial pooling shrinks extreme estimates (Tyson, Miller) toward party mean — the statistically principled version of what CQ unity does informally
- Uses PyMC (already installed for IRT)

### 4. Cross-Session Scrape (2023-24)

**Priority:** High — unlocks temporal analysis and honest out-of-sample validation.

- Run `uv run ks-vote-scraper 2023` to scrape the prior biennium
- Produces a second set of 3 CSVs in `data/90th_2023-2024/`
- Then run the full 8-phase pipeline: `just scrape 2023 && just eda --session 2023-24 && ...`
- Enables all three cross-session analyses below

### 5. Cross-Session Validation

**Priority:** High — the single biggest gap in current results.

Three distinct analyses become possible once 2023-24 is scraped:

- **Prediction honesty (out-of-sample):** Train vote prediction on 2023-24, test on 2025-26 (and vice versa). This is the gold standard for prediction validation — within-session holdout (AUC=0.98) is optimistic because the model sees the same legislators and session dynamics. Cross-session tests whether the learned patterns generalize. Also solves the Senate bill passage small-N problem (59 test bills in 2025-26 is too few; stacking sessions doubles the data).
- **Temporal comparison (who moved?):** Compare IRT ideal points for returning legislators across bienniums. Who shifted ideology? Are the 2025-26 mavericks (Schreiber, Dietrich) the same people who were mavericks in 2023-24? This is the most newsworthy output for the nontechnical audience — "Senator X moved 1.2 points rightward since last session" is a concrete, actionable finding.
- **Detection threshold validation:** The synthesis detection thresholds (unity > 0.95 skip, rank gap > 0.5 for paradox, betweenness within 1 SD for bridge) were calibrated on 2025-26. Running synthesis on 2023-24 tests whether they produce sensible results on a different session with potentially different partisan dynamics. If they don't, the thresholds need to become adaptive or session-parameterized.

### 6. MCA (Multiple Correspondence Analysis)

**Priority:** Medium — alternative view on the vote matrix.

Method documented in `Analytic_Methods/10_DIM_correspondence_analysis.md`. MCA treats each vote as a categorical variable rather than numeric, which is technically more appropriate for Yea/Nay data:

- Complementary to PCA: PCA assumes continuous, MCA assumes categorical
- May reveal structure PCA misses, especially in voting patterns where abstention is meaningful
- `prince` library already in `pyproject.toml`
- Compare MCA dimensions to PCA PC1/PC2 — if they agree, PCA's linear assumption is validated

### 7. Time Series Analysis

**Priority:** Medium — adds temporal depth to static snapshots.

Two methods documented but not yet implemented:

- **Ideological drift** (`Analytic_Methods/26_TSA_ideological_drift.md`): Rolling IRT or rolling party unity within a session. Did anyone change position mid-session? Track 15-vote rolling windows.
- **Changepoint detection** (`Analytic_Methods/27_TSA_changepoint_detection.md`): Structural breaks in voting patterns. When did the session's character shift? (e.g., pre- vs post-veto override period)

Requires the `ruptures` library (already in `pyproject.toml`). Becomes much more powerful once 2023-24 data is available for cross-session comparison.

### 8. 2D Bayesian IRT Model

**Priority:** Medium — solves the Tyson paradox properly. (Formerly item #9.)

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

502 tests exist across scraper (146) and analysis (356) modules. Coverage could be expanded:
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
| 10 | MCA / Correspondence Analysis | DIM | **Planned** — item #6 above |
| 11 | UMAP / t-SNE | DIM | Completed (UMAP, Phase 2b) |
| 12 | W-NOMINATE | DIM | Rejected (R-only) |
| 13 | Optimal Classification | DIM | Rejected (R-only) |
| 14 | Beta-Binomial Party Loyalty | BAY | **Experimental** — item #2 above |
| 15 | Bayesian IRT (1D) | BAY | Completed (IRT) |
| 16 | Hierarchical Bayesian Model | BAY | **Planned** — item #3 above |
| 17 | Posterior Predictive Checks | BAY | Partial (embedded in IRT) |
| 18 | Hierarchical Clustering | CLU | Completed (Clustering) |
| 19 | K-Means / GMM Clustering | CLU | Completed (Clustering) |
| 20 | Co-Voting Network | NET | Completed (Network) |
| 21 | Bipartite Bill-Legislator Network | NET | Deferred (redundant) |
| 22 | Centrality Measures | NET | Completed (Network) |
| 23 | Community Detection | NET | Completed (Network) |
| 24 | Vote Prediction | PRD | Completed (Prediction) |
| 25 | SHAP Analysis | PRD | Completed (Prediction) |
| 26 | Ideological Drift | TSA | **Planned** — item #7 above |
| 27 | Changepoint Detection | TSA | **Planned** — item #7 above |
| 28 | Latent Class Mixture Models | CLU | Deferred (no discrete factions found) |

**Score: 18 completed, 2 rejected, 5 planned, 1 experimental, 2 deferred, 1 partial = 29 total**

---

## Key Architectural Decisions Still Standing

- **Polars over pandas** everywhere
- **Python over R** — no rpy2, no W-NOMINATE
- **IRT ideal points are the primary feature** — prediction confirmed this; everything else is marginal
- **Chambers analyzed separately** unless explicitly doing cross-chamber comparison
- **Tables never truncated** in analysis reports — full data for initial analysis
- **Nontechnical audience** — all visualizations self-explanatory, plain-English labels, annotated findings
