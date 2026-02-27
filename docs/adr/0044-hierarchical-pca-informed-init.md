# ADR-0044: PCA-Informed Initialization for Hierarchical IRT

**Date:** 2026-02-26
**Status:** Accepted

## Context

The flat IRT model uses PCA-informed chain initialization (ADR-0023) to prevent reflection mode-splitting: both MCMC chains start near the PCA PC1 orientation. This fixed 5 of 16 catastrophic convergence failures and is now the default.

The hierarchical IRT model loaded PCA scores but never used them for initialization. The `pca_scores` variable was dead code in the per-chamber model path -- `build_per_chamber_model()` had no initvals parameter. The hierarchical model mostly converged because the sorted party means (`mu_party = pt.sort(mu_party_raw)`) provide partial mode-breaking, but the 91st House showed a persistent marginal R-hat warning (1.0102) and ESS warning (370) that appeared in every run since the first hierarchical implementation (2026-02-22).

An experiment (`results/experiments/2026-02-26_hierarchical-pca-init/`) tested PCA-informed initialization of `xi_offset` in the hierarchical model:

| Metric | Baseline House | PCA-init House | Baseline Senate | PCA-init Senate |
|--------|---------------|----------------|-----------------|-----------------|
| R-hat (xi) max | **1.0102** | **1.0026** | 1.0029 | 1.0022 |
| ESS (xi) min | **370** | **397** | 536 | 573 |
| ESS (mu_party) min | **229** | **356** | 605 | 508 |
| Divergences | 0 | 0 | 0 | 0 |
| Cross-run r | -- | 0.999996 | -- | 0.999996 |

The key difference from flat IRT: the hierarchical model uses non-centered parameterization (`xi = mu_party[party] + sigma_within[party] * xi_offset`), so PCA init targets `xi_offset` (standardized PC1 scores, matching the N(0,1) prior) rather than `xi` directly.

Additionally, the ESS threshold (400) imported from flat IRT comes from Vehtari et al. (2021), which recommends 100 per chain with Stan's default 4 chains. We run 2 chains, so the per-chain recommendation is 200 total. Rather than change the constant (which would affect flat IRT too), the convergence checker now reports per-chain ESS alongside the total for transparency.

## Decision

**PCA-informed `xi_offset` initialization is now the default for per-chamber hierarchical models.** `build_per_chamber_model()` accepts an `xi_offset_initvals` parameter, and `main()` computes standardized PC1 scores for each chamber before sampling.

**ESS per-chain reporting is added to `check_hierarchical_convergence()`.** Each ESS line now shows both the total ESS (checked against the 400 threshold for backward compatibility) and the per-chain ESS (checked against the 100/chain recommendation from Vehtari et al. 2021). This provides transparency about what the threshold means without changing existing pass/fail behavior.

## Consequences

**Benefits:**
- House R-hat (xi) drops from 1.0102 (WARNING) to 1.0026 (OK), crossing the 1.01 threshold.
- House ESS (mu_party) improves from 229 to 356 (55% increase).
- Point estimates are identical (cross-run r = 0.999996). PCA init changes reliability, not results.
- The `pca_scores` variable in the per-chamber loop is no longer dead code.
- ESS reporting now explains what the 400 threshold means (4-chain assumption) and shows the per-chain value, helping users evaluate convergence for non-standard chain counts.

**Trade-offs:**
- Requires PCA results to exist (same dependency as flat IRT; PCA is always run upstream).
- Initializing `xi_offset` biases chains toward the PCA solution. In practice this is desirable (PCA and IRT agree to r > 0.95) but prevents discovery of fundamentally different posterior modes.
- ESS (sigma_within) decreased slightly (719 to 585 for House) -- better xi_offset initialization creates mild correlations with the scale parameter in the non-centered parameterization. This is a known tradeoff.
