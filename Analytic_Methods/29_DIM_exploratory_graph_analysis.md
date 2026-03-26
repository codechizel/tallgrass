# 29. Exploratory Graph Analysis (EGA)

**Category:** DIM (Dimensionality Reduction)
**Pipeline Phase:** 02b
**Script:** `analysis/02b_ega/ega_phase.py`
**Library:** `analysis/ega/`

## What It Does

Estimates the number of latent dimensions in voting data using network psychometrics. Builds a sparse partial correlation network (GLASSO) from tetrachoric correlations and detects communities (dimensions) via Walktrap or Leiden. Unlike PCA (which uses marginal correlations), EGA isolates conditional dependencies — the association between two bills after controlling for all other bills.

## Key Components

1. **Tetrachoric correlations** — correct correlation type for binary (Yea/Nay) data. Estimates the Pearson r between latent continuous variables underlying the binary observations.
2. **GLASSO + EBIC** — L1-penalized precision matrix estimation with Extended BIC model selection. Produces a sparse partial correlation network where edges represent conditional dependencies.
3. **Community detection** — Walktrap (default) or Leiden identifies clusters of densely connected bills = latent dimensions.
4. **bootEGA** — 500 bootstrap replicates assess stability: dimension frequency, per-item stability, structural consistency.
5. **TEFI** — Von Neumann entropy fit index compares K=1..5. Lower = better fit. Properly penalizes over-extraction.
6. **UVA** — Weighted topological overlap detects redundant bill pairs (procedural sequences, amendment cascades).

## When to Use

- Before committing to 2D IRT (Phase 06) — if EGA finds K=1, 2D IRT may be unnecessary.
- To diagnose which bills cause dimensional instability (bootEGA item stability < 0.70).
- To detect redundant bills that inflate the effective item count without adding information.

## Interpretation

- **K=1**: Voting is unidimensional (ideology only).
- **K=2+**: Evidence for multidimensionality; community assignments show which bills load on which dimension.
- **Modal K ≠ empirical K**: Dimensional structure is unstable across bootstrap replicates.
- **TEFI minimum**: The K with lowest TEFI is the best-fitting dimensionality.

## Kansas-Specific Notes

- Senate (N~40) may produce very sparse GLASSO networks due to low sample size.
- High base rate (~82% Yea) creates degenerate 2x2 tables for many bill pairs; tetrachoric falls back to Pearson.
- EGA is advisory — canonical routing (Phase 06) makes the final 1D/2D decision.

## References

- Golino, H., & Epskamp, S. (2017). Exploratory graph analysis. *PLoS ONE*, 12(6).
- Golino, H., et al. (2020). Investigating the performance of EGA. *Psychological Methods*, 25(3).
- Christensen, A. P., & Golino, H. (2021). Estimating stability via bootEGA. *Psych*, 3(3).
