# Analytic Workflow Rules

## Technology Preferences
- **Polars over pandas.** Use polars for all data manipulation. Never use pandas unless a downstream library strictly requires a pandas DataFrame (e.g., seaborn, pymc).
- **Python over R.** All analysis in pure Python. No R, no rpy2, no Rscript calls. When an R-only method exists (W-NOMINATE, OC), use the Python alternative (PCA, Bayesian IRT).

## Method Documentation
- One method per file in `Analytic_Methods/`
- Naming convention: `NN_CAT_method_name.md`
- Each file includes: purpose, assumptions, inputs, outputs, validation steps

## Mandatory Workflow Order
1. **Always run EDA first.** Before any model, record: row counts, missingness rates, close-vote fraction, unanimous-vote fraction, chamber split sizes.
2. **Canonical baseline: 1D Bayesian IRT** on Yea/Nay only. All other models are compared against this.
3. **PCA first, Bayesian second.** PCA is cheap and fast — use it to sanity-check before investing in MCMC.

## Filtering and Reproducibility
- All filtering decisions (unanimous threshold, min participation, chamber separation) must be **explicit constants**, not magic numbers buried in code.
- Save a **filtering manifest** with every analysis output: which votes were dropped, which legislators were excluded, and why.
- Default filters: drop votes where minority < 2.5%, drop legislators with < 20 votes.
- **Sensitivity analyses are mandatory**: run the core model with at least two filter settings (e.g., minority < 2.5% vs < 10%, final passage only vs all motions).

## Validation
- Every fitted model must include at least one validation:
  - Holdout prediction (random 20% of vote observations)
  - Posterior predictive check (for Bayesian models)
  - Classification accuracy against observed votes
- Report both accuracy AND AUC-ROC (accuracy alone is misleading with 82% Yea base rate).

## Report Completeness
- **Never truncate tables in analysis reports.** Show all rows (all legislators, all votes, etc.). The full data is needed for initial analysis. Truncation happens later when preparing results for articles or presentations, not during the analysis phase.

## Separation of Concerns
- **ETL (scraping) is separate from analysis.** Never modify scraper code to accommodate analysis needs. Instead, transform data in analysis scripts.
- Analysis scripts read from `data/ks_{session}/` CSVs. Intermediate analysis artifacts go in a separate output directory.

## Audience: Nontechnical Consumers
- While the analysis pipeline uses advanced statistical methods, the final outputs will be consumed by a **nontechnical audience** (journalists, policymakers, engaged citizens).
- Visualizations are the primary communication tool. Every plot should be **self-explanatory** to someone who has never taken a statistics course: clear titles, plain-English labels, annotations that explain what the viewer is seeing, and color choices that convey meaning intuitively (red=Republican, blue=Democrat).
- Prefer concrete, narrative-friendly visualizations (ranked bar charts, annotated scatter plots, highlighted network diagrams) over abstract statistical plots (dendrograms, eigenvalue scree plots, posterior density traces).
- When a plot shows an interesting finding (e.g., a bridge legislator, a maverick, a party split), **annotate it directly** — label the key actors, add callout boxes, draw attention to the pattern. The viewer should understand the story without reading a caption.
- Tables should use plain-English column headers and include interpretive context (e.g., "higher = more independent from party" rather than just "maverick_score").
- Reports should lead with the most accessible findings and build toward technical detail, not the reverse.

## Runtime Timing as a Sanity Check
- Every analysis run records wall-clock elapsed time in `run_info.json` (`elapsed_seconds`, `elapsed_display`) and displays it in the HTML report header.
- **Always check runtime when reviewing results.** A phase that suddenly runs much faster or slower than expected can indicate:
  - A bug that skips important computation (faster than expected)
  - An accidental optimization that removed a needed step (faster)
  - A regression that added unnecessary work (slower)
  - A convergence problem causing extra MCMC iterations (slower)
  - Data quality issues causing degenerate model fits (faster — model converges trivially)
- Typical runtimes for the 91st session (M3 Pro, 36GB): EDA ~30s, PCA ~15s, IRT ~10-20min per chamber, prediction ~2-5min, synthesis ~30s. Large deviations from these baselines warrant investigation.

## Kansas-Specific Defaults
- Always analyze chambers separately unless explicitly doing cross-chamber comparison.
- Use Cohen's Kappa (not raw agreement) when thresholding similarity for networks or clustering.
- The 34 veto override votes should be analyzed as a separate subgroup in addition to the full dataset.
