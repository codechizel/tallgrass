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
- Ordering constraint means the scale is set by the data and priors, not by fixed anchors — the hierarchical and flat models produce ideal points on different scales (~3x ratio), requiring linear rescaling (`np.polyfit`) for shrinkage comparison
- Small Senate Democrat group (~10) produces wide credible intervals on the Democratic party mean

**Supersedes:** The hierarchical model provides a more principled version of the Beta-Binomial's shrinkage. Beta-Binomial remains as a fast exploratory baseline (no MCMC, instant results).

## Update: Joint Model Ordering Constraint (2026-02-23)

A cross-biennium results audit discovered that the 3-level joint model lacked an ordering constraint on its 4 group-level means (House-D, House-R, Senate-D, Senate-R). While the per-chamber model correctly used `pt.sort(mu_party_raw)` to enforce D < R, the joint model's `group_offset` was unconstrained `Normal(0, 1)`, allowing the sampler to independently flip the party ordering for each chamber — a label-switching pathology.

**Confirmed in:** 90th biennium (2023-24), where the joint model Senate showed r=-0.9999 vs the per-chamber model (perfect sign inversion for Senate while House was correct).

**Fix:** Apply `pt.sort` to each chamber's pair of group offsets (`group_offset_raw[:2]` for House, `group_offset_raw[2:]` for Senate) before computing `mu_group`. This enforces D < R within each chamber independently, consistent with the per-chamber model's identification strategy.

## Update: Shrinkage Rescaling Fallback Warning (2026-02-25)

The shrinkage comparison requires rescaling flat IRT ideal points to the hierarchical scale via `np.polyfit` on matched legislators. When fewer than 3 legislators match (e.g., due to anchor filtering), the rescaling silently fell back to `slope=1.0` (identity transform), producing misleading shrinkage comparisons. A warning is now emitted when this fallback occurs, making the limitation visible in the output. See `docs/irt-deep-dive.md` for the code audit that identified this issue.
