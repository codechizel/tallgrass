# ADR-0049: nutpie Flat IRT Baseline Experiment

**Date:** 2026-02-27
**Status:** Accepted

## Context

The nutpie deep dive (`docs/nutpie-deep-dive.md`) identified nutpie as a promising replacement for PyMC's default NUTS sampler, particularly for hierarchical IRT models that currently fail convergence (House: 1/8 sessions, joint: 0/8). Before testing on hierarchical models, we need to verify basic compatibility with our model structure.

The flat 2PL IRT model (91st House, 130 legislators × 297 votes, 722 free parameters) serves as the baseline. It uses `pt.set_subtensor` for anchor insertion, `pm.Bernoulli(logit_p=...)` for the likelihood, and `pm.Deterministic` with coordinate dims — all of which must survive Numba compilation.

## Decision

Run nutpie (v0.16.6, Numba backend) on the flat 2PL IRT model and compare with the PyMC baseline. Pass criteria: compilation succeeds, convergence diagnostics pass (R-hat < 1.01, ESS > 400, zero divergences), and ideal points correlate |r| > 0.99 with PyMC.

### Results

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Compilation | 13.6s | Success | PASS |
| R-hat (xi) max | 1.0036 | < 1.01 | PASS |
| ESS (xi) min | 1,950 | > 400 | PASS |
| Divergences | 0 | 0 | PASS |
| \|r\| vs PyMC | 0.9935 | > 0.99 | PASS |
| Sign flip | Yes | Expected | OK |
| Total time | 126.4s | — | — |

The sign flip (r = -0.994) is IRT reflection invariance — nutpie found the mirror-image solution. Correctable post-hoc by checking anchor polarity.

## Consequences

**Positive:**
- Numba backend confirmed compatible with all PyTensor ops in Tallgrass IRT models
- InferenceData fully compatible with ArviZ diagnostic pipeline
- Single-process execution (Rust threads) — no orphan child processes
- Green light for Experiment 2 (hierarchical per-chamber with Numba)

**Negative:**
- Sign flip requires post-hoc correction in production integration
- 13.6s compilation overhead per model (acceptable for our use case)
- No `log_likelihood` group in output (known nutpie limitation, not a blocker)

**Dependency added:** `nutpie>=0.14` (installed as dev dependency for experiments)

## Outcome

This baseline validated nutpie for flat IRT and directly led to production migration: ADR-0051 (per-chamber hierarchical) and ADR-0053 (flat IRT + joint hierarchical — all models now use nutpie).

## Related

- [ADR-0006](0006-irt-implementation-choices.md) — Flat IRT model design
- [ADR-0048](0048-experiment-framework.md) — Experiment framework used here
- [ADR-0051](0051-nutpie-production-hierarchical.md) — Per-chamber production migration
- [ADR-0053](0053-nutpie-all-models.md) — All models migrated to nutpie
- [Nutpie Deep Dive](../nutpie-deep-dive.md) — Full architecture and integration plan
- [Experiment Results](../nutpie-flat-irt-experiment.md) — Detailed write-up
