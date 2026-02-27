# PCA-Informed Initialization for Hierarchical IRT: Experiment Results

**Date:** 2026-02-26

## Summary

An experiment on the 91st Kansas Legislature (2025-26) tested whether PCA-informed MCMC chain initialization — already the default for flat IRT (ADR-0023) — improves convergence for the hierarchical Bayesian IRT model. Result: it fixes the House R-hat warning that has persisted since the model's first run, reducing worst-case R-hat from 1.0102 to 1.0026 while producing functionally identical ideal point estimates (r = 0.999996).

## Background

### The flat IRT precedent

In February 2026, we discovered that 5 of 16 flat IRT chamber-sessions suffered catastrophic convergence failure (R-hat ~1.83, ESS ~3). The root cause was **reflection mode-splitting**: the two MCMC chains settled into mirror-image modes of the posterior — one placing Democrats on the left, the other on the right. Random initialization gave each chain a coin-flip chance of landing in either mode, and no amount of additional tuning could fix this (the posterior is symmetrically bimodal by construction).

The solution was PCA-informed chain initialization (ADR-0023): both chains start near the PCA PC1 orientation instead of random starts, placing them in the same mode. This is standard practice in the political science IRT literature — Jackman's `pscl::ideal` R package has used eigendecomposition-based starting values as the default since its inception (Jackman, 2001; Clinton, Jackman & Rivers, 2004).

### The hierarchical gap

The hierarchical IRT model (Phase 10) was implemented after ADR-0023 but the PCA initialization was never carried over. The code loads PCA scores — they're passed into the per-chamber loop — but `build_per_chamber_model()` never uses them. They're dead code in the model-building path.

The hierarchical model avoids *catastrophic* failure because its sorted party means (`mu_party = pt.sort(mu_party_raw)`, enforcing D < R ordering) provide partial mode-breaking. But partial isn't complete. Every hierarchical run since the model's introduction on 2026-02-22 has produced the same House warnings:

- R-hat (xi): 1.0102 — marginally above the 1.01 threshold
- ESS (xi): 370 — below the 400 threshold
- ESS (mu_party): 229 — below the 400 threshold

Because MCMC is deterministic with `random_seed=42`, these exact numbers appear in every run. They're not thermal noise or run-to-run variation — they're a persistent baseline issue baked into the default initialization.

## The initialization challenge

The hierarchical model uses **non-centered parameterization** to avoid the funnel geometry that plagues centered hierarchical models:

```
xi_offset ~ Normal(0, 1)              # standardized offsets
xi = mu_party[party] + sigma_within[party] * xi_offset   # actual ideal points
```

This means PCA init can't target `xi` directly (it's a deterministic, not a free parameter). Instead, we initialize `xi_offset` — the free parameter that the sampler actually explores. Since `xi_offset` has a N(0,1) prior, we standardize the PCA PC1 scores to mean 0 and standard deviation 1, which places them on the same scale.

This is an approximation. The true `xi_offset` for a given legislator depends on the (unknown) party mean and within-party spread, not just the raw PCA score. But the approximation works because it solves the real problem: orienting both chains so they agree on which direction is "liberal" and which is "conservative."

## Experiment design

Two runs on the 91st biennium (2025-26), per-chamber models only (no joint), same MCMC settings (2000 draws, 1500 tune, 2 chains, seed 42, target_accept 0.95):

1. **Baseline:** Default `jitter+adapt_diag` initialization (current production behavior)
2. **PCA init:** `xi_offset` initialized from standardized PCA PC1 scores

## Results

### Convergence diagnostics

| Metric | Baseline House | PCA-init House | Change |
|--------|---------------|----------------|--------|
| R-hat (xi) max | **1.0102** | **1.0026** | Fixed (below 1.01) |
| ESS (xi) min | 370 | 397 | +7% (still below 400) |
| ESS (mu_party) min | 229 | 356 | **+55%** (still below 400) |
| ESS (sigma_within) min | 719 | 585 | -19% (still above 400) |
| Divergences | 0 | 0 | No change |
| Sampling time | 419s | 407s | ~3% faster (within noise) |

Senate passed all checks in both runs with no meaningful difference.

### Point estimates

| Chamber | Baseline vs PCA-init correlation |
|---------|--------------------------------|
| House | r = 0.999996 |
| Senate | r = 0.999996 |

The ideal points are functionally identical. PCA init changes how the sampler explores the posterior, not what it finds.

### ICC and flat IRT correlation

| Metric | Baseline House | PCA-init House | Baseline Senate | PCA-init Senate |
|--------|---------------|----------------|-----------------|-----------------|
| ICC | 0.900 | 0.902 | 0.922 | 0.921 |
| Hier vs flat r | 0.9867 | 0.9868 | 0.9763 | 0.9763 |

No change in substantive results.

## Interpretation

### What PCA init fixes

The headline result is **R-hat**. The Gelman-Rubin diagnostic (Vehtari et al., 2021) is the primary convergence check — it measures whether independent chains have converged to the same distribution. The 1.01 threshold is widely used in the Bayesian literature:

- **Baseline:** R-hat = 1.0102 → technically a warning, suggesting the chains haven't fully mixed
- **PCA init:** R-hat = 1.0026 → clean, well below the threshold

This follows exactly the same pattern as the flat IRT fix (ADR-0023). The sorted party means break the *global* reflection symmetry (you can't flip all Democrats and Republicans), but there's residual local ambiguity in individual ideal point orientations that PCA init resolves.

### What PCA init doesn't fix

ESS (effective sample size) improved but didn't fully cross the 400 threshold. The House's 130-legislator × 297-vote model is simply large enough that 2000 posterior draws produce only ~397 effective samples for the worst-case ideal point. This is a sample-size issue, not an initialization issue — more draws would resolve it.

An interesting tradeoff: `sigma_within` ESS decreased from 719 to 585. In non-centered parameterizations, better `xi_offset` initialization can create mild posterior correlations between `xi_offset` and `sigma_within`, slowing exploration of the scale parameter. This is a known property of non-centered models (Betancourt, 2017) and the value (585) is still well above the 400 threshold.

### Why the hierarchical model is more robust than flat IRT

The flat IRT model without PCA init suffered *catastrophic* failure (R-hat 1.83, ESS 3) because it has no structural mode-breaking at all — the posterior is perfectly bimodal. The hierarchical model's sorted party means break the dominant mode, so without PCA init it only shows *marginal* issues (R-hat 1.01, ESS 370). PCA init takes it from "marginal" to "clean" — a smaller improvement in absolute terms but still worth implementing.

## Recommendation

Add PCA-informed initialization to `build_per_chamber_model()` as the default, matching the flat IRT behavior established in ADR-0023. The fix is:

1. Accept an optional `xi_offset_initvals` parameter
2. Compute standardized PCA PC1 scores in the main loop (already loaded, currently unused)
3. Pass as `initvals={"xi_offset": xi_offset_initvals}` to `pm.sample()`

The residual ESS issue (397 vs 400) could be addressed in a follow-up by increasing `n_samples` from 2000 to 2500. This is a much smaller concern — R-hat is the primary diagnostic, and that's now clean.

## References

- Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. *arXiv:1701.02434*.
- Clinton, J., Jackman, S., & Rivers, D. (2004). The statistical analysis of roll call data. *APSR*, 98(2), 355-370.
- Jackman, S. (2001). Multidimensional analysis of roll call data via Bayesian simulation. *Political Analysis*, 9(3), 227-241.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
- ADR-0023: PCA-informed IRT chain initialization (2026-02-23).
- Experiment data: `results/experiments/2026-02-26_hierarchical-pca-init/`
