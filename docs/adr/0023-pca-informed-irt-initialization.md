# ADR-0023: PCA-Informed IRT Chain Initialization (Default)

**Date:** 2026-02-23
**Status:** Accepted

## Context

Five of sixteen chamber-sessions in our historical data (84th House, 85th Senate, 86th House, 87th Senate, 89th House) exhibited catastrophic IRT convergence failure: R-hat ~1.83, ESS ~3, zero divergences. Investigation revealed the root cause is **reflection mode-splitting** — the two MCMC chains settle into mirror-image modes of the posterior (one where Democrats are negative, one where Democrats are positive), and the ordering constraint alone is insufficient to prevent this because it acts on discrimination parameters (beta), not ideal points.

Four experiments were conducted on the 87th Senate (the smallest failing case):

| Experiment | R-hat | ESS | PCA-IRT r | Status |
|-----------|-------|-----|-----------|--------|
| Baseline (default init) | 1.8294 | 3 | -0.38 | FAILURE |
| PCA-informed init | 1.004 | 1263 | +0.98 | SUCCESS |
| Increased tuning (3000) | 1.8294 | 3 | -0.38 | FAILURE |
| Sign constraint on beta | 1.8340 | 3 | -0.46 | FAILURE |

Only PCA-informed initialization resolved the problem. See `results/experiments/2026-02-23_irt-convergence-mode-splitting/irt-convergence-investigation.md` for the full write-up.

## Decision

**PCA-informed chain initialization is now the default for IRT models.** The CLI flag changed from `--pca-init` (opt-in) to `--no-pca-init` (opt-out).

**How it works:** Before MCMC sampling, PCA is run on the binary vote matrix. The standardized PC1 scores are used as initial values for the ideal point parameters (`xi`) in both chains, plus small jitter. This places both chains in the same mode of the posterior, preventing reflection mode-splitting.

**Literature support:** This approach is well-established in the political science IRT literature:

- **Jackman (2001, `pscl::ideal`)** — the reference R implementation has used `startvals="eigen"` (PCA-based) as the default since its inception.
- **Clinton, Jackman, & Rivers (2004)** — the foundational political science IRT paper uses eigendecomposition-based starting values.
- **Bafumi, Gelman, Park, & Kaplan (2005)** — demonstrates that good initialization improves MCMC mixing in IRT models.
- **Betancourt (2017)** — Stan case study recommending data-informed initialization for complex hierarchical models.

See `results/experiments/2026-02-23_irt-convergence-mode-splitting/lit-review-irt-initialization.md` for the full literature review.

**Alternatives considered:**
- **More tuning iterations (3000):** Failed completely — more random exploration cannot escape a symmetric bimodal posterior.
- **Sign constraint on beta[0]:** Failed — constraining one discrimination parameter doesn't break the reflection symmetry in ideal point space.
- **Informative priors:** Would work but introduces subjective assumptions about party locations. PCA init is data-driven and assumption-free.
- **Explicit symmetry-breaking prior on xi[0]:** Would work but is fragile and requires knowing which legislator to anchor.

## Consequences

**Benefits:**
- All 16 chamber-sessions now converge (R-hat < 1.01, ESS > 400) where previously 5 failed catastrophically.
- Zero computational cost: PCA is already computed in an upstream phase and loaded from parquet.
- Matches the established practice in the field's reference implementation (`pscl::ideal`).
- Downstream phases (network, synthesis, prediction, profiles) that depend on IRT ideal points now produce valid results for all sessions.

**Trade-offs:**
- Requires PCA results to exist (the `--no-pca-init` flag provides an escape hatch).
- The initialization biases chains toward the PCA solution. In practice this is desirable (PCA and IRT agree to r > 0.95 when both converge), but it means MCMC cannot discover a fundamentally different posterior mode if one exists.
