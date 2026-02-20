# ADR-0006: IRT Implementation Choices

**Date:** 2026-02-19
**Status:** Accepted

## Context

Bayesian IRT is the canonical baseline analysis per the analytic workflow rules: "1D Bayesian IRT on Yea/Nay only." PCA (Phase 2) confirmed strong 1D ideological structure (PC1 explains 53% House, 46% Senate). Several implementation choices needed to be resolved:

1. **Discrimination prior.** The standard IRT formulation uses Normal priors for discrimination (beta), which allows negative values. Negative beta reverses the ideological interpretation of a vote and creates sign identification problems.

2. **Identification method.** The 2PL IRT model is not identified without constraints (sign, location, and scale invariance). Multiple approaches exist: fix two legislators, order constraints, or soft identification via priors.

3. **Number of chains.** Standard recommendation is 4 chains, but IRT on ~170 legislators x ~400 votes is expensive. Trade-off between diagnostic rigor and runtime.

4. **Holdout validation approach.** True out-of-sample validation requires refitting the model on a subset of data. For IRT this means a second expensive MCMC run.

5. **Per-chamber vs. joint model.** Could estimate ideal points jointly across both chambers (using bridging legislators like Miller) or separately.

6. **Posterior storage format.** Full posterior distributions are large (~50-100MB per chamber). Need a standard, reloadable format.

## Decision

1. **LogNormal(0.5, 0.5) prior for discrimination.** Constrains beta > 0, solving sign identification without post-hoc correction. The LogNormal(0.5, 0.5) has mode ~1.0 and 95% mass in [0.3, 5.5], which covers the realistic range of discrimination values. This is the modern standard in IRT (e.g., Bafumi et al. 2005). The trade-off: bills where the liberal position is Yea will have the direction captured by the alpha (difficulty) parameter rather than a negative beta. This is correct — alpha shifts the probability curve left/right, beta only controls steepness.

2. **PCA-based anchor method.** Fix the most conservative legislator (highest PCA PC1) at xi=+1 and most liberal (lowest PC1) at xi=-1. Anchors must have >= 50% participation to ensure tight estimation. This leverages the PCA results we already have and avoids requiring manual knowledge of Kansas legislators. The alternative (soft identification via priors + post-hoc correction) was rejected because it requires checking sign orientation across chains and can produce label-switching artifacts.

3. **2 chains by default.** Sufficient for R-hat and ESS computation with dramatically lower runtime (~15-20 min total vs. ~40+ min for 4 chains). For a mature, well-identified model with strong data, 2 chains typically converge identically. If convergence diagnostics show problems, `--n-chains 4` is available. This follows the PyMC documentation recommendation for computationally expensive models.

4. **In-sample holdout prediction.** Use posterior means from the full model to predict a random 20% of observed cells. This is documented as in-sample (the model saw all data during fitting). The proper Bayesian validation is the posterior predictive check (PPC), which samples from the full posterior and compares replicated data to observed. A true out-of-sample holdout would require refitting — available as a future enhancement but not worth the doubled runtime for the baseline.

5. **Per-chamber models.** House and Senate are fitted independently. Joint modeling with bridging legislators (e.g., Miller who served in both chambers) is deferred to a future enhancement. The analytic flags document notes Miller as a bridging candidate. Per-chamber is the standard approach (NOMINATE, Clinton et al. 2004) and avoids the complexity of cross-chamber identification.

6. **ArviZ NetCDF for posterior storage.** `idata.to_netcdf()` produces a self-describing file that can be reloaded with `az.from_netcdf()` for downstream analysis (e.g., posterior predictive checks on new data, model comparison via LOO-CV). NetCDF is the ArviZ default and is supported by xarray for advanced slicing.

## Consequences

**Benefits:**
- LogNormal discrimination eliminates the most common IRT debugging issue (sign flips) with no loss of expressiveness.
- PCA-based anchors are automated and reproducible — no manual legislator selection required.
- 2 chains cuts runtime nearly in half while maintaining adequate diagnostic coverage.
- In-sample holdout + PPC together provide comprehensive validation without the cost of refitting.
- Per-chamber models are simple, standard, and sufficient for the 1D baseline.
- NetCDF files enable full posterior reuse in downstream phases (clustering, network analysis).

**Trade-offs:**
- LogNormal beta cannot model bills where higher ideal points predict Nay (the alpha parameter absorbs this). If a future 2D model is needed, this prior would need revisiting.
- PCA anchors assume PCA and IRT agree on who is most extreme. This is validated by the PCA-IRT correlation check (r > 0.95 expected).
- 2 chains provide less multi-modal detection power than 4. If R-hat > 1.01 or ESS < 400, the user should re-run with `--n-chains 4`.
- In-sample holdout overstates predictive performance. This is clearly documented in the output and is supplemented by PPC.
- Per-chamber models cannot leverage bridging legislators. This is deferred, not abandoned.
