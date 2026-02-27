# ADR-0045: 4-Chain Hierarchical IRT with adapt_diag Initialization

**Date:** 2026-02-26
**Status:** Accepted

## Context

The hierarchical IRT model ran 2 MCMC chains on an M3 Pro with 6 performance cores, leaving 4 P-cores idle. The House chamber consistently produced marginal ESS warnings (xi: 397, mu_party: 356) against the 400 threshold from Vehtari et al. (2021), which assumes 4 chains with ~100 effective samples each. With 2 chains, each chain had to contribute ~200 effective samples — double the per-chain workload the threshold was calibrated for.

An experiment (`results/experimental_lab/2026-02-26_hierarchical-4-chains/`) tested 4 chains and discovered a critical interaction between PyMC's `jitter+adapt_diag` initialization and PCA-informed starting values:

**Initial 4-chain attempt (with jitter):**

| Metric | House | Senate |
|--------|-------|--------|
| R-hat (xi) max | **1.5348** | **1.5312** |
| ESS (xi) min | **7** | **7** |

R-hat ~1.53 with ESS of 7 indicates reflection mode-splitting — one chain explored the mirror-image posterior. The root cause: `jitter+adapt_diag` adds random perturbation to starting values. With 4 chains receiving independent perturbations to the 130-dimensional `xi_offset` vector, one chain's jitter pushed it past the mode boundary. With 2 chains, the sorted party means constraint (`mu_party = pt.sort(mu_party_raw)`) provided sufficient mode-breaking; with 4 chains, the probability of at least one mode-flip increased significantly.

**Fixed 4-chain run (adapt_diag, no jitter):**

| Metric | 2 chains (baseline) | 4 chains (adapt_diag) | Change |
|--------|---------------------|------------------------|--------|
| ESS (xi) House | 397 | **564** | +42% |
| ESS (mu_party) House | 356 | **512** | +44% |
| ESS (xi) Senate | 573 | **1002** | +75% |
| R-hat (xi) House | 1.0026 | 1.0103 | Marginal |
| Sampling time House | 402s | 420s | +4.5% |
| xi correlation (2ch vs 4ch) | — | 0.9999 / 0.9995 | Identical |

## Decision

1. **Default chain count increased from 2 to 4** for per-chamber hierarchical models (`HIER_N_CHAINS = 4`).

2. **`adapt_diag` initialization (no jitter) is used when PCA initvals are provided.** Applied to both `build_per_chamber_model()` in `hierarchical.py` and `build_and_sample()` in `irt.py`. When PCA scores orient the chains correctly, jitter adds noise without benefit and creates a risk of mode-flipping that grows with chain count.

3. **Joint model unchanged.** The joint cross-chamber model was excluded from this experiment. Its longer runtime (31 min with 2 chains) means 4 chains may produce meaningful thermal overhead. Joint model chain count should be evaluated separately.

## Consequences

**Benefits:**
- Both House ESS warnings resolved: xi 397→564, mu_party 356→512. All thresholds cleared.
- Senate ESS nearly doubles (573→1002), providing substantial margin.
- Wall-time cost is minimal (+4.5% House, +2.7% Senate) — 4 chains on 6 P-cores have headroom.
- Per-chain ESS is well-balanced (House xi: 122-133) and all above the 100/chain Vehtari target.
- R-hat computed from 8 split groups (vs 4) provides more reliable convergence diagnostics.
- Ideal points are functionally unchanged (Spearman r > 0.999).
- `adapt_diag` prevents the jitter mode-splitting discovered during the experiment.

**Trade-offs:**
- House R-hat(xi) marginally higher (1.0026→1.0103). This reflects increased R-hat sensitivity with 4 chains, not worse convergence. All per-chain diagnostics improved.
- 4 chains double memory usage (4 OS processes vs 2). On 36GB RAM this is not a concern.
- First 4-chain run after model code changes may appear to hang during Pytensor compilation in spawned workers (10-20 min). Subsequent runs use the disk cache.
- Per-chain ESS is lower (~130 vs ~200) due to 4 chains sharing 6 P-cores. Total ESS still increases substantially.
- Removing jitter means all chains start from exactly the PCA-informed position (plus `adapt_diag` mass matrix estimation). This is desirable for mode-orientation but slightly reduces initial exploration diversity.
