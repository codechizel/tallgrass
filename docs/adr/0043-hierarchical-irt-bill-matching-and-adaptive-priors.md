# ADR-0043: Hierarchical IRT Bill-Matching and Adaptive Priors

**Date:** 2026-02-26
**Status:** Accepted

## Context

The joint cross-chamber hierarchical IRT model (`build_joint_model()` in Phase 10) had two critical problems:

1. **No shared items across chambers.** The model combined House and Senate data by `vote_id`, producing zero shared bill parameters. Without common items bridging the chambers, the two likelihoods were completely separable — a textbook IRT linking failure. Result: Senate sign flips, 34% scale distortion, unreliable cross-chamber placement. Diagnosed in `docs/joint-hierarchical-irt-diagnosis.md`.

2. **J=2 over-shrinkage in Senate.** With ~11 Senate Democrats, `sigma_within ~ HalfNormal(1)` allowed the posterior to explore pathological geometries (funnel/mode-splitting), producing R-hat 1.83 and ESS 3. The James-Stein threshold (J >= 3) was violated. External validation confirmed: hierarchical Senate r = -0.541 (inverted) vs flat Senate r = 0.929 against Shor-McCarty.

The flat IRT + test equating pipeline already solved both problems correctly (r = 0.93-0.98 externally validated) by matching bills by `bill_number` for shared betas and using no hierarchical structure.

## Decision

### Fix 1: Bill-matching in `build_joint_model()`

Added a `rollcalls` parameter and a new `_match_bills_across_chambers()` helper function that:

- Groups vote_ids by `bill_number` from the rollcalls table
- Finds bills appearing in both chambers (71-174 per session)
- Prefers "Final Action" / "Emergency Final Action" motions (most likely to be on identical text)
- Falls back to latest chronologically when no final action exists
- Creates shared `alpha`/`beta` indices for matched bills — both chambers' observations point to the same item parameters

The shared betas are the mathematical bridge: the model must satisfy both chambers' observations simultaneously on the same bill, forcing the xi values onto a common scale and sign convention. This is standard concurrent calibration (Clinton, Jackman & Rivers, 2004).

The bill-matching logic was extracted from the flat IRT's proven `build_joint_vote_matrix()` to ensure consistency.

### Fix 2: Group-size-adaptive priors

Added constants `SMALL_GROUP_THRESHOLD = 20` and `SMALL_GROUP_SIGMA_SCALE = 0.5`. Both `build_per_chamber_model()` and `build_joint_model()` now use:

```
sigma_within ~ HalfNormal(0.5)  for groups with < 20 members
sigma_within ~ HalfNormal(1.0)  for groups with >= 20 members
```

Following Gelman (2015) on informative priors when group counts are small. The tighter prior prevents catastrophic convergence failure while preserving the hierarchical structure and ICC decomposition.

## Consequences

**Positive:**
- Joint model now has shared items bridging chambers (natural sign/scale identification)
- `fix_joint_sign_convention()` should stop triggering for sessions with sufficient shared bills
- Small groups get convergence-safe priors (prevents R-hat > 1.8 pathology)
- `combined_data` dict includes `n_shared_bills` and `matched_bills` for downstream use/reporting
- 9 new tests (48 total for hierarchical module)

**Negative:**
- Tighter priors for small groups are more informative (prior-influenced). This is the correct tradeoff for J=2 groups but means Senate-D estimates are prior-influenced. The existing `MIN_GROUP_SIZE_WARN` warning flags this.
- `build_joint_model()` gains a `rollcalls` parameter (backward-compatible: defaults to `None` for legacy behavior with no bill matching)

**Unchanged:**
- Per-chamber models: same structure, same results (only sigma_within prior scale changes for small groups)
- `fix_joint_sign_convention()`: retained as safety net
- Flat IRT + test equating: remains the gold standard for cross-chamber comparisons

## References

- Clinton, J., Jackman, S., & Rivers, D. (2004). The statistical analysis of roll call data. *APSR*, 98(2), 355-370.
- Gelman, A. (2015). Prior distributions for variance parameters in hierarchical models. *Bayesian Analysis*, 1(3), 515-534.
- Shor, B., McCarty, N., & Berry, C. (2011). Methodological issues in bridging ideal points. SSRN Working Paper.
- `docs/joint-hierarchical-irt-diagnosis.md` — full diagnosis of the broken joint model
- `docs/hierarchical-shrinkage-deep-dive.md` — literature survey on J=2 over-shrinkage
- [ADR-0055](0055-reparameterized-beta-and-irt-linking.md) — Reparameterized beta + IRT linking (builds on shared bills for anchor extraction)
