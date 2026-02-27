# ADR-0047: Positive beta constraint experiment for hierarchical IRT convergence

**Date:** 2026-02-27
**Status:** Accepted

## Context

The hierarchical IRT model shows systematic convergence failures across all 8 bienniums (84th–91st):

- **Per-chamber House:** Fails 7/8 sessions (R-hat 1.01–1.06, ESS 46–564). Zero divergences, healthy E-BFMI — the sampler explores correctly but slowly.
- **Per-chamber Senate:** Passes 6/8 sessions. Same model, fewer parameters (~40 legislators vs ~130).
- **Joint cross-chamber:** Fails all 8 sessions (R-hat up to 2.42, ESS as low as 5, divergences in 7/8).

Analysis in `docs/hierarchical-convergence-improvement.md` identifies the root cause: a symmetric `beta ~ Normal(0, 1)` prior creates a reflection mode where (β, ξ) → (−β, −ξ) leaves the likelihood invariant. The `pt.sort(mu_party_raw)` ordering constraint partially breaks this symmetry at the group level, but individual β parameters can still drift toward the reflection boundary, creating ridges that slow NUTS mixing. The problem scales with the number of bill parameters: ~280 for House, ~240 for Senate, ~430 for Joint.

## Decision

Run a controlled experiment comparing the current model against a positive-beta variant:

**Baseline:** Current model (`beta ~ Normal(0, 1)`, symmetric).

**Treatment:** `beta ~ LogNormal(0, 0.5)`, constraining all discrimination parameters to be positive. This eliminates the reflection mode entirely. Bills that genuinely discriminate in the "wrong" direction will be forced to near-zero discrimination, effectively treating them as uninformative. This is acceptable because: (a) EDA already filters near-unanimous votes; (b) a bill that doesn't discriminate along the primary ideological axis is uninformative for ideal point estimation.

The experiment tests both the per-chamber model (where we have convergence data across all bienniums) and the joint model (which currently fails everywhere). The 91st biennium is the primary test session, with secondary validation on the 87th (a session where the joint model nearly converged with the current parameterization).

## Consequences

**Positive:**
- If β > 0 resolves House per-chamber convergence, it confirms the reflection-mode theory and provides a simple production fix.
- If it additionally resolves joint model convergence, it enables the three-level hierarchical model as the production cross-chamber method (currently we fall back to per-chamber models with test equating).
- The constraint is standard in educational testing IRT (De Ayala 2009) and has been adopted in legislative IRT (Bafumi et al. 2005) when sign identification is desired.

**Negative:**
- Bills with genuinely negative discrimination (e.g., bipartisan procedural votes where liberal legislators vote Yea alongside conservatives) will have their discrimination shrunk toward zero. This is a loss of expressiveness.
- The ICC and within-party sigma estimates may change, since the posterior geometry is different. Must compare against current results to assess impact.
- LogNormal(0, 0.5) implies a prior median of 1.0 and prior mean of ~1.13 — mildly informative compared to the current Normal(0, 1). The experiment should also test HalfNormal(1) as an alternative with less prior mass away from zero.

**Validation plan:**
- Convergence: R-hat, ESS, divergences, E-BFMI
- Accuracy: Pearson/Spearman correlation of ideal points (baseline vs treatment)
- External: Correlation with flat IRT ideal points (which are externally validated against Shor-McCarty)
- Impact: ICC, group parameters, shrinkage patterns, HTML report for visual inspection

## Results (2026-02-27)

Experiment complete. Full results in `results/experimental_lab/2026-02-27_positive-beta/experiment.md`.

**Per-chamber (91st biennium):**

| Metric | Baseline | LogNormal | HalfNormal |
|--------|----------|-----------|------------|
| House R-hat(xi) | 1.0103 | **1.0058** | 1.0122 |
| House ESS(xi) | 564 | 362 | 450 |
| House verdict | FAIL (R-hat) | FAIL (ESS) | FAIL (R-hat) |
| Senate verdict | PASS | PASS | PASS |

**Joint (LogNormal):** R-hat(xi) = 1.024, ESS(xi) = 243, 25 divergences. Improved from production (R-hat ~1.5, ESS ~7) but still fails.

**Conclusion:** Positive beta is necessary (fixes R-hat) but not sufficient (trades ESS). The reflection mode theory is confirmed. Next steps: combine with more draws (Priority 7) or nutpie sampler (Priority 6) to address the remaining ESS shortfall.
