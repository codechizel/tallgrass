# ADR-0111: Canonical Init Strategy for Hierarchical IRT

**Date:** 2026-03-11
**Status:** Accepted

## Context

The hierarchical IRT model (Phase 07) supports multiple initialization strategies via `--init-strategy` (ADR-0107):

- `auto`: prefer 1D IRT, fall back to PCA
- `irt-informed`: use 1D IRT posterior means
- `pca-informed`: use PCA PC1 scores

For horseshoe-affected chambers (Kansas Senate in supermajority bienniums), the 1D IRT scores are confounded — they conflate ideology with establishment-loyalty. Using these confounded scores as initialization propagates the horseshoe distortion into the hierarchical model.

ADR-0109 introduced canonical ideal point routing, which automatically selects the best ideology score per chamber (2D Dim 1 for horseshoe-affected chambers, 1D for balanced chambers). The hierarchical model should consume this routing output rather than always using raw 1D IRT scores.

The hierarchical model already has `--dim1-prior` (ADR-0108) which adds a soft Bayesian prior from 2D Dim 1. But initialization and priors serve different purposes: initialization places chains near the correct mode (preventing mode-splitting), while priors shape the posterior geometry. Both benefit from horseshoe-corrected scores.

## Decision

### New init strategy: `canonical`

Add `--init-strategy canonical` to the hierarchical model. When selected:

1. Load canonical routing output from `{phase_06_dir}/canonical_irt/canonical_ideal_points_{chamber}.parquet`
2. Use `xi_mean` values as `xi_offset` initialization (after standardization)
3. For horseshoe chambers, this is 2D Dim 1 (ideology separated from establishment-loyalty)
4. For balanced chambers, this is 1D IRT (same as `irt-informed`)

### CLI

```bash
just hierarchical --init-strategy canonical   # use canonical routing output
just hierarchical --init-strategy auto         # default: prefer 1D IRT, fall back to PCA
just hierarchical --init-strategy 2d-dim1      # force 2D Dim 1 (research only)
```

### Auto strategy update

The `auto` strategy resolution order becomes:
1. Canonical routing output (if Phase 06 has run and produced canonical output)
2. 1D IRT scores (Phase 05)
3. PCA PC1 scores (Phase 02)

This means `auto` automatically benefits from canonical routing when available, without requiring the user to specify `--init-strategy canonical` explicitly.

### Implementation in init_strategy.py

```python
class InitStrategy(StrEnum):
    AUTO = "auto"
    IRT_INFORMED = "irt-informed"
    PCA_INFORMED = "pca-informed"
    IRT_2D_DIM1 = "2d-dim1"
    CANONICAL = "canonical"        # NEW
```

The `resolve_init_source()` function gains a `canonical_dir` parameter. When strategy is `canonical` (or `auto` with canonical output available), it loads from the canonical parquet file.

## Consequences

**Benefits:**
- Hierarchical model automatically gets horseshoe-corrected initialization for supermajority chambers
- No manual flag selection needed — `auto` does the right thing when Phase 06 has run
- Consistent with the canonical routing philosophy: downstream phases consume the best available score

**Costs:**
- Phase 07 now depends on Phase 06 for optimal initialization (gracefully degrades to 1D/PCA if Phase 06 hasn't run)
- Adds one more strategy to the init system

**Related:**
- ADR-0107 — Shared init strategy system (base infrastructure)
- ADR-0108 — Dim 1 informative prior (complementary: prior vs. init)
- ADR-0109 — Canonical ideal point routing (source of canonical scores)
- ADR-0110 — Tiered convergence quality gate (determines which scores are canonical)
- `analysis/init_strategy.py` — Shared implementation
- `analysis/07_hierarchical/hierarchical.py` — Consumer
