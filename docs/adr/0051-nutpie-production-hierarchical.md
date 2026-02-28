# ADR-0051: nutpie Rust NUTS for per-chamber hierarchical IRT

**Date:** 2026-02-27
**Status:** Accepted

## Context

Per-chamber hierarchical IRT sampling with PyMC's NUTS uses Python-level multiprocessing (`cores=n_chains`) for parallel chains. nutpie provides a Rust-native NUTS sampler that runs chains in a single process using Rust threads, compiled via Numba.

Experiments 2 and 2b (`results/experimental_lab/2026-02-27_nutpie-hierarchical/`) proved nutpie + PCA initialization produces identical results to PyMC for both chambers:

| Metric | House (nutpie) | Senate (nutpie) |
|--------|---------------|-----------------|
| R-hat(xi) max | 1.0027 | 1.0010 |
| ESS(xi) min | 1204 | 1658 |
| Divergences | 0 | 0 |
| \|r\| vs PyMC | 1.0000 | 1.0000 |

## Decision

Migrate `build_per_chamber_model()` to use nutpie's Rust NUTS sampler unconditionally. The joint cross-chamber model stays on PyMC (untested with nutpie).

Key implementation details:

1. **Split `build_per_chamber_model()` into graph + sampling.** New `build_per_chamber_graph()` returns the PyMC model without sampling. `build_per_chamber_model()` compiles with `nutpie.compile_pymc_model()` and samples with `nutpie.sample()`.

2. **`jitter_rvs` must exclude `xi_offset`.** PCA-informed initialization sets `xi_offset` via `initial_points`. All other RVs (especially `sigma_within`, a HalfNormal) need jitter — without it, HalfNormal initializes at its support point (~0), producing `log(0) = -inf`.

3. **`callback` and `target_accept` accepted but ignored.** Preserves API compatibility with experiment runner. nutpie uses its own adaptive dual averaging and has no callback mechanism.

4. **No fallback to PyMC.** nutpie compilation is proven for this model. Failures are genuine errors.

5. **No CLI flag.** nutpie is the unconditional default for per-chamber.

## Consequences

**Gains:**
- Single-process Rust threads instead of Python multiprocessing (simpler process model)
- Proven identical results — zero quality regression
- `build_per_chamber_graph()` is now importable by experiments (eliminates model code duplication)

**Loses:**
- PyMC `callback` parameter no longer functional for per-chamber (still works for joint model)
- `target_accept` no longer directly controllable for per-chamber
- `cores` parameter no longer meaningful for per-chamber (nutpie manages its own threads)

**Scope limitation:**
- Joint model stays on PyMC until a separate experiment validates nutpie for 3-level models
- `OMP_NUM_THREADS=6` cap remains (still needed for BLAS in gradient computation)
