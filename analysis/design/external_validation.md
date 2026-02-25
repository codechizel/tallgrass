# External Validation Design Choices

## Assumptions

1. **Rank-order equivalence, not scale equivalence.** Shor-McCarty `np_score` is career-fixed (one score per legislator). Our `xi_mean` varies by biennium. We compare rank ordering within a session, not absolute scale values. This is why Spearman ρ is reported alongside Pearson r.

2. **Name matching sufficiency.** Kansas legislator names are distinct enough that normalized "first last" matching catches ~90% of cases. Middle names, nicknames, and generational suffixes are the main failure modes. A two-phase strategy (exact match, then last-name fallback) handles the long tail.

3. **SM coverage is complete for the overlap period.** The April 2023 dataset covers Kansas legislators through 2020. We assume every legislator active in 2011-2020 who cast sufficient votes appears in the SM data.

4. **Independent legislators are rare and handled.** Kansas has very few Independents (1-2 per biennium). They are included in overall correlations but excluded from intra-party analysis.

## Parameters & Constants

| Constant | Value | Justification | Code location |
|----------|-------|---------------|---------------|
| `MIN_MATCHED` | 10 | Minimum sample for reliable correlation; below this, Fisher z CI is too wide to be informative | `external_validation_data.py` |
| `STRONG_CORRELATION` | 0.90 | Field standard: Shor & McCarty report r > 0.90 between their scores and DW-NOMINATE for Congress | `external_validation_data.py` |
| `GOOD_CORRELATION` | 0.85 | Accounts for session-specific vs career-level comparison (lower than pure replication) | `external_validation_data.py` |
| `CONCERN_CORRELATION` | 0.70 | Below this, scores may not measure the same construct | `external_validation_data.py` |
| `OUTLIER_TOP_N` | 5 | Report the 5 most discrepant legislators for manual review | `external_validation_data.py` |

## Methodological Choices

### Name Matching Algorithm

**Decision:** Two-phase matching: exact normalized name, then last-name-only with deduplication.

**Alternatives considered:**
- Fuzzy string matching (Levenshtein): adds complexity, risk of false matches
- District-based matching only: names in different districts may collide
- Manual mapping: does not scale across 5 bienniums

**Impact:** Phase 1 handles the common case (identical first/last). Phase 2 catches middle-name divergences (our data includes full middle names, SM often omits them). The `NICKNAME_MAP` dict handles edge cases discovered during real runs.

### Correlation Methods

**Decision:** Report Pearson r, Spearman ρ, and Fisher z 95% CI.

**Rationale:**
- Pearson r: standard in the field for ideal point comparisons
- Spearman ρ: robust to monotonic nonlinearity between scales
- Fisher z CI: standard confidence interval for correlation coefficients

Both are reported because SM's career-level score may not scale linearly with our session-level score, but rank ordering should be preserved.

### Z-Score Discrepancy for Outliers

**Decision:** Z-standardize both score columns before computing |xi_z - np_z|.

**Rationale:** The two scales have different means and variances. Raw differences are meaningless. Z-scoring makes the discrepancy interpretable as "how many standard deviations apart are they relative to their own distributions?"

### Pooled Analysis (--all-sessions)

**Decision:** When running all 5 bienniums, also compute a pooled correlation using unique legislators (most recent session's scores for duplicates).

**Rationale:** Pooling increases sample size and tests whether the relationship holds across the full decade. Deduplication by most-recent-session avoids double-counting returning legislators.

## Downstream Implications

1. **Credibility assessment.** If r > 0.90 for most sessions, the pipeline's IRT scores can be presented as externally validated. This is a requirement for academic-quality reporting.

2. **Convergence diagnostic.** Sessions with low correlation (especially 84th) may indicate IRT convergence problems. The external validation serves as a supplementary convergence check.

3. **Name matching infrastructure.** The `NICKNAME_MAP` and normalization functions can be reused by any future external dataset integration.

4. **Outlier investigation.** High-discrepancy legislators may warrant individual investigation — are they genuinely different, or is there a data issue? This feeds back into the profiles phase.

5. **Hierarchical model diagnostic.** The 88th biennium validation revealed that the per-chamber hierarchical model over-shrinks Senate ideal points (r = -0.541 vs flat r = +0.929), while the House works excellently (hierarchical r = 0.984). Root cause: only 11 Senate Democrats — too few for reliable partial pooling (J=2 small-group problem). See `docs/hierarchical-shrinkage-deep-dive.md` for full analysis. Implication: per-chamber hierarchical results should carry a minimum group size warning (n < 20).
