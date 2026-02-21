# Roadmap

What's been done, what's next, and what's on the horizon for the KS Vote Scraper analytics pipeline.

**Last updated:** 2026-02-21 (after Phase 7: Classical Indices)

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

---

## Next Up

### 1. Synthesis Report — "State of the Kansas Legislature"

**Priority:** Highest — this is the deliverable.

A single, narrative-driven HTML report that pulls the most compelling findings from all 7 phases into a story a journalist or citizen can read. Not a summary of methods — a summary of *what we learned about Kansas politics*.

Sections to include:
- **The party line is everything.** k=2, zero cross-party edges, AUC=0.98 from party-line features alone.
- **Who are the mavericks?** Schreiber (House) and Dietrich (Senate) profiles — CQ unity, weighted maverick, network position, IRT ideal point, all in one place.
- **The Tyson paradox.** Most conservative IRT rank but lowest party loyalty. Why these aren't contradictions.
- **Veto overrides tell you nothing new.** Strictly party-line — no bipartisan override coalitions like Congress.
- **What the model can't predict.** The residuals that point to issue-specific, non-ideological voting.

Reuse existing plots where possible, add 2-3 new synthesis visualizations (e.g., a combined legislator dashboard scatter plot).

### 2. UMAP/t-SNE Ideological Landscape

**Priority:** High — quick win, high visual impact for nontechnical audience.

Method documented in `Analytic_Methods/11_DIM_umap_tsne_visualization.md`. Non-linear dimensionality reduction on the vote matrix produces an "ideological map" where nearby legislators vote alike. Much more intuitive than PCA scatter plots for a general audience:

- UMAP on the binary vote matrix (same input as PCA)
- Color by party, annotate key legislators (Schreiber, Tyson, Dietrich, Thompson)
- Overlay IRT ideal points or cluster labels as secondary encoding
- Compare UMAP structure to PCA — do non-linear methods reveal structure the linear model misses?

Estimated effort: small. `umap-learn` already in `pyproject.toml`.

### 3. Beta-Binomial Party Loyalty (Bayesian)

**Priority:** High — experimental code already exists.

Method documented in `Analytic_Methods/14_BAY_beta_binomial_party_loyalty.md`. Experimental implementation exists at `analysis/irt_beta_experiment.py` (558 lines) but is not integrated into the main pipeline.

- Bayesian alternative to the frequentist CQ party unity from Phase 7
- Produces credible intervals on loyalty — crucial for legislators with few party votes
- Beta(alpha, beta) posterior per legislator; shrinks noisy estimates toward the group mean
- Compare Bayesian loyalty posteriors to CQ unity point estimates
- Especially useful for Miller (30 votes) and other sparse legislators

### 4. Hierarchical Bayesian Legislator Model

**Priority:** High — the "Crown Jewel" from the methods overview.

Method documented in `Analytic_Methods/16_BAY_hierarchical_legislator_model.md`. Legislators nested within party and chamber, with partial pooling:

- Models legislator-level parameters as draws from party-level distributions
- Naturally handles the R supermajority (more data = tighter party estimate)
- Quantifies how much individual legislators deviate from their party's typical behavior
- Partial pooling shrinks extreme estimates (Tyson, Miller) toward party mean — the statistically principled version of what CQ unity does informally
- Uses PyMC (already installed for IRT)

### 5. Cross-Session Scrape (2023-24)

**Priority:** High — unlocks temporal analysis and honest out-of-sample validation.

- Run `uv run ks-vote-scraper 2023` to scrape the prior biennium
- Produces a second set of 3 CSVs in `data/ks_2023/`
- Enables: cross-session prediction validation, ideological drift tracking, multi-session index stacking

### 6. Cross-Session Validation

**Priority:** High — the single biggest gap in current results.

- Train vote prediction on 2023-24, test on 2025-26 (and vice versa)
- Train bill passage on 2023-24, test on 2025-26 (solves the Senate small-N problem: 59 test bills is too few)
- Stack indices across sessions: all parquets already include a `session` column
- Compare IRT ideal points for returning legislators: did anyone shift?

### 7. MCA (Multiple Correspondence Analysis)

**Priority:** Medium — alternative view on the vote matrix.

Method documented in `Analytic_Methods/10_DIM_correspondence_analysis.md`. MCA treats each vote as a categorical variable rather than numeric, which is technically more appropriate for Yea/Nay data:

- Complementary to PCA: PCA assumes continuous, MCA assumes categorical
- May reveal structure PCA misses, especially in voting patterns where abstention is meaningful
- `prince` library already in `pyproject.toml`
- Compare MCA dimensions to PCA PC1/PC2 — if they agree, PCA's linear assumption is validated

### 8. Time Series Analysis

**Priority:** Medium — adds temporal depth to static snapshots.

Two methods documented but not yet implemented:

- **Ideological drift** (`Analytic_Methods/26_TSA_ideological_drift.md`): Rolling IRT or rolling party unity within a session. Did anyone change position mid-session? Track 15-vote rolling windows.
- **Changepoint detection** (`Analytic_Methods/27_TSA_changepoint_detection.md`): Structural breaks in voting patterns. When did the session's character shift? (e.g., pre- vs post-veto override period)

Requires the `ruptures` library (already in `pyproject.toml`). Becomes much more powerful once 2023-24 data is available for cross-session comparison.

### 9. NLP on Bill Text for Passage Prediction

**Priority:** Medium — the obvious missing feature for prediction.

Current bill passage prediction uses only structural features (beta, vote_type, bill_prefix, day_of_session). AUC is 0.90 (House) / 0.84 (Senate). Adding bill text features could improve substantially:

- Bill titles and descriptions are already available from the KLISS API (fetched during scraping)
- Simple TF-IDF or sentence embeddings on `short_title` / `bill_title`
- Topic modeling to classify bills into policy areas
- Would explain the "surprising" bills that structural features miss

### 10. 2D Bayesian IRT Model

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

### Test Suite

No formal tests exist. The pipeline is verified manually via spot-checks and data integrity assertions embedded in the scripts. A proper test suite would:
- Mock the scraper HTTP calls
- Test data transformations with fixture data
- Validate output schema for each analysis phase

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
| Fix Shallenburger suffix | Next scraper touch | "Vice President of the Senate" not in leadership suffix strip pattern |

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
| 10 | MCA / Correspondence Analysis | DIM | **Planned** — item #7 above |
| 11 | UMAP / t-SNE | DIM | **Planned** — item #2 above |
| 12 | W-NOMINATE | DIM | Rejected (R-only) |
| 13 | Optimal Classification | DIM | Rejected (R-only) |
| 14 | Beta-Binomial Party Loyalty | BAY | **Experimental** — item #3 above |
| 15 | Bayesian IRT (1D) | BAY | Completed (IRT) |
| 16 | Hierarchical Bayesian Model | BAY | **Planned** — item #4 above |
| 17 | Posterior Predictive Checks | BAY | Partial (embedded in IRT) |
| 18 | Hierarchical Clustering | CLU | Completed (Clustering) |
| 19 | K-Means / GMM Clustering | CLU | Completed (Clustering) |
| 20 | Co-Voting Network | NET | Completed (Network) |
| 21 | Bipartite Bill-Legislator Network | NET | Deferred (redundant) |
| 22 | Centrality Measures | NET | Completed (Network) |
| 23 | Community Detection | NET | Completed (Network) |
| 24 | Vote Prediction | PRD | Completed (Prediction) |
| 25 | SHAP Analysis | PRD | Completed (Prediction) |
| 26 | Ideological Drift | TSA | **Planned** — item #8 above |
| 27 | Changepoint Detection | TSA | **Planned** — item #8 above |
| 28 | Latent Class Mixture Models | CLU | Deferred (no discrete factions found) |

**Score: 17 completed, 2 rejected, 6 planned, 1 experimental, 2 deferred, 1 partial = 29 total**

---

## Key Architectural Decisions Still Standing

- **Polars over pandas** everywhere
- **Python over R** — no rpy2, no W-NOMINATE
- **IRT ideal points are the primary feature** — prediction confirmed this; everything else is marginal
- **Chambers analyzed separately** unless explicitly doing cross-chamber comparison
- **Tables never truncated** in analysis reports — full data for initial analysis
- **Nontechnical audience** — all visualizations self-explanatory, plain-English labels, annotated findings
