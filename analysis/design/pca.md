# PCA Design Choices

**Script:** `analysis/pca.py`
**Constants defined at:** `analysis/pca.py:148-157`
**ADR:** `docs/adr/0005-pca-implementation-choices.md`

## Assumptions

1. **Linear ideology.** PCA assumes that voting behavior is a linear function of latent ideological dimensions. A legislator's vote on any bill is modeled as a linear combination of their position on each principal component. This is adequate for a first pass but misses the nonlinear relationship that IRT captures (the logistic link function).

2. **Complete data required.** PCA cannot handle nulls. Every cell in the input matrix must have a value, which requires imputation. The imputation method is itself a design choice (see below).

3. **Equal weighting of roll calls.** After standardization (StandardScaler), every contested roll call contributes equally to PCA. A bill that barely cleared the 2.5% minority threshold has the same influence as a major party-line vote.

4. **Orthogonal dimensions.** PCA forces principal components to be uncorrelated. If the true ideological structure has correlated dimensions (e.g., fiscal conservatism partially predicts social conservatism), PCA distorts this by forcing orthogonality.

## Parameters & Constants

| Constant | Value | Justification | Code location |
|----------|-------|---------------|---------------|
| `DEFAULT_N_COMPONENTS` | 5 | Extracts 5 PCs per chamber. Only PC1-2 are typically interpretable; PC3-5 retained for IRT comparison. | `pca.py:151` |
| `MINORITY_THRESHOLD` | 0.025 | Inherited from EDA. PCA does not re-filter the default matrices. | `pca.py:152` |
| `SENSITIVITY_THRESHOLD` | 0.10 | Per workflow rules: re-run at 10% for sensitivity analysis. | `pca.py:153` |
| `MIN_VOTES` | 20 | Inherited from EDA. | `pca.py:154` |
| `HOLDOUT_FRACTION` | 0.20 | Random 20% of non-null cells masked for holdout validation. | `pca.py:155` |
| `HOLDOUT_SEED` | 42 | NumPy random seed for reproducible holdout selection. | `pca.py:156` |
| PC2 extreme threshold | 3σ | `detect_extreme_pc2()` flags the min-PC2 legislator only if `|PC2| > 3 × std(PC2)`. | `pca.py:detect_extreme_pc2()` |

## Methodological Choices

### Imputation: row-mean (each legislator's Yea rate)

**Decision:** Missing values (nulls) are filled with each legislator's average Yea rate across their non-missing votes. A legislator who voted Yea 80% of the time gets their absences filled with 0.80.

**Alternatives considered:**
- Column-mean (bill's average Yea rate) — rejected because it erases per-legislator signal, which is exactly what PCA needs
- Zero-fill (treat absences as Nay) — rejected because it falsely asserts strategic opposition on every missed vote
- Iterative imputation (SoftImpute, MICE) — rejected as overkill for a pre-IRT sanity check; these methods are principled but harder to explain and debug
- Drop legislators with any nulls — rejected because it would eliminate ~30% of legislators

**Impact:** Row-mean imputation biases absent legislators toward their own base rate. If a legislator strategically missed contentious votes (avoiding recorded dissent), their imputed values will be biased toward their easy-vote average. This makes their PCA score look more moderate than they truly are. IRT handles this properly by simply not including absent cells in the likelihood.

**Key concern for downstream:** Sen. Miller (30/194 Senate votes) has 85% of his matrix imputed. His PC2 extreme is an imputation artifact, not a real voting pattern. See `docs/analytic-flags.md`.

### Standardization: center and scale (StandardScaler)

**Decision:** Each roll call column is centered (subtract mean) and scaled (divide by standard deviation) before PCA.

**Why:** Without scaling, close votes (high variance in the binary column) dominate PCA while near-unanimous votes contribute little. Centering removes the overall Yea rate; scaling ensures each roll call contributes equally.

**Alternatives:** Center-only (no scaling) — used by some in the literature but gives disproportionate weight to contested votes. We chose center+scale to match Poole & Rosenthal's methodology.

**Impact:** Every contested roll call has equal weight after scaling. This means a procedural vote that happened to be contested has the same influence as a major policy vote. The sensitivity analysis (10% threshold) partially addresses this by removing borderline-contested votes.

### PC1 sign convention: Republicans positive

**Decision:** After fitting PCA, compare mean PC1 scores for Republicans and Democrats. If Republicans are negative, flip the sign of PC1 scores and loadings.

**Why:** PCA components have arbitrary sign. This convention (positive = conservative) matches the NOMINATE literature and makes interpretation consistent across runs and sessions.

**Impact:** All downstream consumers of PC1 scores (IRT anchor selection, visualizations, reports) can assume positive = conservative.

### Holdout validation: mask-and-reconstruct

**Decision:** Randomly mask 20% of non-null cells, re-impute, re-fit PCA on the training set, reconstruct the full matrix, and evaluate predictions on the masked cells.

**Impact:** This tests whether PCA captures enough structure to predict held-out votes better than the ~82% Yea base rate. Current results: 93% accuracy, 0.97 AUC-ROC — PCA clearly captures real structure.

**Caveat:** The holdout cells were imputed in the training matrix, so the training PCA is slightly contaminated by the test data through row-mean imputation. This is a minor concern given the high accuracy.

### Sensitivity: duplicated filter logic

**Decision:** The sensitivity analysis re-filters the full vote matrix at 10% minority threshold using ~40 lines of duplicated filter logic (not imported from `eda.py`).

**Why:** Keeps PCA self-contained. Changes to EDA's filtering won't silently alter PCA's sensitivity analysis.

**Impact:** If a filtering bug is found in EDA, it must be fixed in PCA (and IRT) separately.

## Downstream Implications

### For IRT (Phase 3)
- **PCA scores are used to select IRT anchors.** IRT picks the most-conservative (highest PC1) and most-liberal (lowest PC1) legislators as anchors, constrained to xi=+1 and xi=-1 respectively. If PCA scores are wrong, the IRT model will be anchored to the wrong legislators.
- **PCA-IRT correlation is a validation check.** Pearson r > 0.95 between IRT ideal points and PCA PC1 is expected. Lower correlation suggests IRT is capturing nonlinearities that PCA misses.
- **Row-mean imputation is NOT used by IRT.** IRT handles missing data natively. The imputation artifacts that affect PCA (e.g., Miller's PC2 extreme) will not carry over to IRT.

### For Clustering (Phase 5)
- PCA scores (PC1-2 or more) can be used as clustering input features.
- The sign convention (Republicans positive) is baked into the scores. Clustering methods that use distance are sign-agnostic, but any threshold-based cluster interpretation should account for the convention.

### For interpretation
- **PCA gives point estimates only.** No uncertainty intervals. A legislator at PC1=0.5 might truly be anywhere from 0.3 to 0.7 — PCA cannot tell you. Use IRT credible intervals for uncertainty.
- **PC2 interpretation requires examining loadings.** In the current data, Senate PC2 captures "contrarianism on routine legislation" (driven by Tyson/Thompson), not a traditional second ideological dimension. Do not over-interpret PC2 as a coherent dimension without checking what drives it.
