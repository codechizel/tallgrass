# ADR-0017: Hierarchical Bayesian IRT

**Date:** 2026-02-22
**Status:** Accepted

## Context

The flat IRT model (Phase 3) treats all legislators as independent draws from `Normal(0, 1)`, ignoring the party structure that explains ~70% of voting variance. The Beta-Binomial phase (7b) demonstrated Bayesian shrinkage for party loyalty using closed-form empirical Bayes but operates on a derived metric (CQ party unity) rather than the full vote matrix.

We need a model that:
1. Provides partial pooling by party on the full IRT ideal point model
2. Quantifies how much party explains (variance decomposition)
3. Shows what changes vs. the flat model (shrinkage comparison)
4. Handles the sparse Senate Democrat group (~10 legislators) gracefully

## Decision

Implement a 2-level hierarchical IRT with ordering constraint and non-centered parameterization, run per-chamber as the primary output. Add an optional 3-level joint cross-chamber model as secondary (skippable via `--skip-joint`).

**Key design choices:**

1. **Ordering constraint (`pt.sort`) instead of hard anchors.** The flat IRT fixes two legislators at xi=±1. The hierarchical model uses `mu_party = sort(mu_party_raw)` to ensure Democrat < Republican without constraining any individual. This allows all legislators to participate in partial pooling.

2. **Non-centered parameterization.** `xi = mu_party[p] + sigma_within[p] * xi_offset` with `xi_offset ~ Normal(0, 1)` avoids the "funnel of hell" geometry that makes centered hierarchical models difficult to sample.

3. **Higher target_accept (0.95 vs flat's 0.9) and more tuning (1500 vs 1000).** Hierarchical geometry requires more careful NUTS adaptation.

4. **Per-chamber primary.** Consistent with the flat IRT and all other analysis phases. The joint model is experimental due to sparse cross-chamber overlap.

## Consequences

**Benefits:**
- Partial pooling shrinks noisy estimates toward party means (especially helpful for low-participation legislators)
- ICC provides a single-number summary of party polarization
- Shrinkage comparison identifies which legislators' estimates are least reliable in the flat model
- Non-centered parameterization avoids the main failure mode of hierarchical models

**Trade-offs:**
- 2-4x slower than flat IRT (more parameters, higher target_accept, more tuning)
- Joint model may not converge (~71 shared bills for ~169 legislators)
- Ordering constraint means the scale is set by the data and priors, not by fixed anchors — less directly comparable to the flat IRT's absolute scale
- Small Senate Democrat group (~10) produces wide credible intervals on the Democratic party mean

**Supersedes:** The hierarchical model provides a more principled version of the Beta-Binomial's shrinkage. Beta-Binomial remains as a fast exploratory baseline (no MCMC, instant results).
