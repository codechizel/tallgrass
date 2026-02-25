---
paths:
  - "analysis/**/*.py"
---

# Analysis Framework

## 12-Phase Pipeline

EDA -> PCA -> UMAP -> IRT -> Clustering -> Network -> Prediction -> Indices -> Beta-Binomial -> Hierarchical IRT -> Synthesis -> Profiles

Cross-session validation compares across bienniums (separate from the per-session pipeline).
External validation compares IRT ideal points against Shor-McCarty scores (84th-88th bienniums only).

## Technology Preferences

- **Polars over pandas** for all data manipulation (pandas only when downstream library requires it)
- **Python over R** — no rpy2 or Rscript. Use PCA/Bayesian IRT instead of W-NOMINATE/OC.
- Tables: great_tables with polars DataFrames (no pandas conversion). Plots: base64-embedded PNGs. See ADR-0004.

## HTML Report System

Each phase produces a self-contained HTML report with SPSS/APA-style tables and embedded plots:

- `analysis/report.py` — Generic: `TableSection`, `FigureSection`, `TextSection`, `ReportBuilder`, `make_gt()`, Jinja2 template + CSS
- `analysis/run_context.py` — `RunContext` context manager: structured output, elapsed timing, auto-primers, `strip_leadership_suffix()` utility
- Phase-specific report builders: `eda_report.py`, `umap_report.py`, `beta_binomial_report.py`, `hierarchical_report.py`, `synthesis_report.py`, `profiles_report.py`, `cross_session_report.py`, `external_validation_report.py`

## Key Data Modules (Pure Logic, No I/O)

- `analysis/nlp_features.py` — TF-IDF + NMF topic modeling on bill `short_title` text
- `analysis/synthesis_detect.py` — Notable legislator detection (mavericks, bridge-builders, paradoxes)
- `analysis/profiles_data.py` — Profile targets, scorecards, bill-type breakdown, defections
- `analysis/cross_session_data.py` — Legislator matching, IRT alignment, shift metrics, prediction transfer
- `analysis/external_validation_data.py` — SM parsing, name normalization, matching, correlations, outlier detection

## Design Documents

Each phase has a design doc in `analysis/design/` — **read before interpreting results or adding a new phase:**

- `eda.md` — Binary encoding, filtering thresholds, agreement metrics
- `pca.md` — Imputation, standardization, sign convention, holdout design
- `irt.md` — Priors, MCMC settings, PCA-informed chain initialization
- `clustering.md` — Three methods for robustness, k=2 finding
- `prediction.md` — XGBoost primary, IRT features dominate, NLP topic features
- `beta_binomial.md` — Empirical Bayes, per-party-per-chamber priors, shrinkage
- `synthesis.md` — Data-driven detection thresholds, graceful degradation
- `cross_session.md` — Affine IRT alignment, name matching, prediction transfer
- `external_validation.md` — SM name matching, correlation methodology, career-fixed vs session-specific

## Key Data Structures

The **vote matrix** (legislators x roll calls, binary) is the foundation. Build from `votes.csv`: pivot `legislator_slug` x `vote_id`, Yea=1, Nay=0, absent=NaN.

**Critical preprocessing:**
- Filter near-unanimous votes (minority < 2.5%)
- Filter legislators with < 20 votes
- Analyze chambers separately

## Independent Party Handling

Scraper outputs empty string for non-R/D. Every analysis phase fills to "Independent" at load time. All modules define `PARTY_COLORS` with `"Independent": "#999999"`. Party-specific models exclude Independents. Plots iterate over parties present in data. See ADR-0021.

## Kansas-Specific Notes

- Republican supermajority (~72%) — intra-party variation more interesting than inter-party
- k=2 optimal (party split); intra-R variation is continuous
- 34 veto override votes are analytically rich (cross-party coalitions)
- Beta-Binomial and Bayesian IRT are the recommended Bayesian starting points

## Analytics Method Docs

`Analytic_Methods/` has 28 documents (one per method). Naming: `NN_CAT_method_name.md`. Categories: DATA, EDA, IDX, DIM, BAY, CLU, NET, PRD, TSA.
