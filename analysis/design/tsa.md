# Time Series Analysis Design Choices

**Script:** `analysis/15_tsa/tsa.py`
**Constants defined at:** `analysis/15_tsa/tsa.py` (top of file)

## Assumptions

1. **Roll calls are chronologically ordered** by `vote_datetime`. The vote matrix columns follow this ordering, making rolling windows correspond to temporal progression.

2. **Chambers analyzed separately**, consistent with all upstream phases.

3. **PC1 captures the party dimension** within each window. This is empirically validated by PCA (Phase 2) where PC1 explains ~57% of variance and aligns with party.

4. **Rice Index is self-contained.** Recomputed from raw votes rather than depending on Phase 07 output, ensuring the TSA phase can run independently.

5. **Weekly aggregation is appropriate for Kansas.** With ~2 roll calls per day on average, daily Rice is noisy. Weekly aggregation yields ~14 observations per point, sufficient for PELT.

## Parameters & Constants

| Constant | Value | Justification |
|----------|-------|---------------|
| `WINDOW_SIZE` | 75 | ~2-3 weeks of roll calls. Large enough for stable PCA, small enough to detect within-session shifts. |
| `STEP_SIZE` | 15 | 75% overlap between windows. Balances temporal resolution against computational cost. |
| `MIN_WINDOW_VOTES` | 10 | Per-legislator minimum within a window. Ensures meaningful PCA loadings. |
| `MIN_WINDOW_LEGISLATORS` | 20 | Minimum cross-section per window. PCA needs sufficient rank. |
| `PELT_PENALTY_DEFAULT` | 10.0 | Moderate penalty. Sensitivity analysis explores [3, 50]. |
| `PELT_MIN_SIZE` | 5 | Minimum 5 weeks per segment. Prevents overfitting to noise. |
| `WEEKLY_AGG_DAYS` | 7 | Standard calendar week. |
| `SENSITIVITY_PENALTIES` | `np.linspace(1, 50, 25)` | 25 evenly-spaced penalties from 1 to 50. Smoother elbow plot than previous 6-point grid. |
| `TOP_MOVERS_N` | 10 | Top 10 drifters highlighted per chamber. |
| `MIN_TOTAL_VOTES` | 20 | Session-wide minimum for inclusion (matches EDA convention). |
| `PARTY_COLORS` | R=#E81B23, D=#0015BC | Consistent with all prior phases. |

## Methodological Choices

### Rolling PCA over Dynamic IRT

Rolling-window PCA was chosen over dynamic IRT (Martin-Quinn) for within-session drift because:

- **Speed**: PCA runs in seconds vs hours for IRT. The entire TSA phase completes in under a minute.
- **Empirical agreement**: PC1 correlates r > 0.95 with IRT ideal points in Kansas data (Phase 2 vs Phase 4).
- **Interpretability**: PC1 is the dominant axis of variation; for Kansas, this is party.

Dynamic IRT is planned for *cross-biennium* trajectory analysis (Martin-Quinn), where the longer time horizon and multi-session structure justify the computational cost.

### PELT over BOCPD

PELT (Pruned Exact Linear Time) was chosen over Bayesian Online Changepoint Detection (BOCPD) because:

- **Offline data**: The full session is already scraped. PELT's exact offline solution is appropriate.
- **Well-tested**: The `ruptures` library is mature and well-documented.
- **Simplicity**: PELT has one tuning parameter (penalty). BOCPD requires specifying a prior hazard rate, observation model, and run length distribution.

### RBF Kernel

The RBF (Radial Basis Function) kernel was chosen over simpler models (L1, L2) because:

- Detects changes in both mean and variance of the Rice distribution.
- Appropriate for bounded [0, 1] data where distribution shape matters.
- Non-parametric: doesn't assume normality.

### Weekly Aggregation

Daily Rice would produce ~2 observations per day (one per vote), which is noisy and has irregular spacing. Weekly aggregation:

- Smooths daily variation while preserving session-scale trends.
- Provides ~14 observations per weekly point (sufficient for PELT).
- Aligns with legislative calendar (committees typically meet weekly).

### Sign Convention

PC1 signs are aligned per window so that the Republican mean is positive. This matches the convention established in Phase 2 (PCA) and Phase 4 (IRT). Without per-window alignment, PCA sign indeterminacy would cause artificial "drift" artifacts.

### Desposato Small-Group Rice Correction

Rice Index is systematically inflated for small groups: a caucus of 8 members voting randomly will show higher apparent Rice than a caucus of 32 voting randomly (Desposato 2005, *BJPS*). This matters for Kansas: Senate Democrats have ~8 members vs ~32 Republicans.

`desposato_corrected_rice()` uses Monte Carlo simulation (10K draws, seed=42) to compute expected Rice under random (coin-flip) voting for the given group size, then subtracts it from raw Rice (floored at 0.0). Applied by default via `correct_size_bias=True` on `build_rice_timeseries()`. Raw formula validation tests pass `correct_size_bias=False`.

### Imputation Sensitivity

`compute_imputation_sensitivity()` validates the column-mean imputation used in rolling PCA by running `compute_early_vs_late()` twice — once with column-mean imputation (default) and once with listwise deletion (complete cases only). The Pearson correlation between drift scores validates that the imputation method doesn't materially affect results. Result is stored in the filtering manifest and displayed in the HTML report.

### Self-Contained Rice Computation

Rather than depending on Phase 07 (Indices) output, TSA recomputes Rice from raw votes. This ensures:

- TSA can run independently without prior phases.
- No version-coupling between phases.
- Rice computation is simple enough that duplication is cheaper than dependency management.

## Downstream Implications

- **Changepoint dates** can be cross-referenced with the legislative calendar for contextual interpretation.
- **Top movers** identified here can be investigated further in Phase 12 (Profiles).
- **Penalty sensitivity** provides a robustness check — only changepoints stable across penalties should be reported in narratives.
- **Veto override cross-reference** tests whether override coalitions disrupted normal cohesion patterns.

## See Also

- Deep dive: `docs/tsa-deep-dive.md` (literature survey, ecosystem comparison, code audit, 7 recommendations)
- ADR: `docs/adr/0057-time-series-analysis.md`
- Roadmap gaps: `docs/roadmap.md` item #8 (TSA Hardening)
