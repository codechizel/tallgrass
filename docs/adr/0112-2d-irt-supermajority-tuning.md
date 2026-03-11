# ADR-0112: 2D IRT Tuning for Supermajority Chambers

**Date:** 2026-03-11
**Status:** Accepted

## Context

The 2D IRT model (Phase 06) consistently fails to converge for the Kansas Senate in supermajority bienniums. Empirical data across 14 bienniums shows:

- **79th Senate (75% R):** R-hat ~2.0, ESS ~5 across ALL 15 historical runs — fundamental mode-splitting
- **Senate pattern:** 10 of 14 bienniums have Senate R-hat > 1.10
- **House pattern:** 7 of 14 bienniums have House R-hat > 1.10 (fewer failures, less severe)

The current configuration (N_TUNE=2000, N_SAMPLES=2000, 4 chains) was set when the 2D model was experimental (ADR-0054). Three specific bottlenecks contribute to the convergence failures:

1. **Insufficient warmup for hard geometry:** The 2D posterior has 8 equivalent modes (2^D × D! for D=2). PLT constraints should eliminate most, but extreme supermajority compositions create near-degenerate geometry where the sampler needs more adaptation time.

2. **Uninitialized bill parameters:** Only `xi` (ideal points) receives PCA-informed initialization. The `beta` (discrimination) parameters start from random values, contributing to mode-splitting early in warmup.

3. **Near-unanimous votes dilute signal:** Bills with 95%+ agreement carry almost no ideological information but add parameters. For 2D models with their harder geometry, this noise is more damaging than for 1D.

## Decision

### 1. Adaptive N_TUNE based on party composition

Detect the majority party fraction at runtime. When the supermajority exceeds 70%, double the tuning budget:

```python
SUPERMAJORITY_THRESHOLD = 0.70
N_TUNE_BASE = 2000
N_TUNE_SUPERMAJORITY = 4000
```

The detection is automatic — no CLI flag needed. The session composition determines the tuning budget.

### 2. Beta initialization from PCA loadings

Extend the PCA-informed initialization to include bill discrimination parameters. PCA component loadings provide a natural starting point for `beta`:

- `beta_col0` (Dim 1 discrimination): PCA PC1 loadings, scaled to unit variance
- `beta_col1` (Dim 2 discrimination): PCA PC2 loadings, respecting PLT constraints (anchor item 0 = 0, anchor item 1 = positive)

This reduces the number of randomly initialized parameters, giving the sampler a head start on the correct mode.

### 3. Contested-only filtering via `--contested-only` flag

Add a `--contested-only` flag that filters to contested votes (minority > 2.5%) before fitting the 2D model. This is the same filter used by Phase 05's `--contested-only` flag (ADR-0104).

For the 2D model specifically, this reduces the bill count from ~400 to ~200, cutting the parameter space nearly in half while retaining all ideologically informative votes.

**Not the default** — the full vote set is used by default for consistency with 1D. But for supermajority chambers where convergence is the bottleneck, the reduced parameter space meaningfully helps.

### Summary of changes

| Change | Mechanism | When Active | Expected Impact |
|--------|-----------|-------------|-----------------|
| Adaptive N_TUNE | Auto-detect supermajority > 70% | Automatic | More warmup for hard geometry |
| Beta init from PCA | `--init-strategy pca-informed` (default) | Always when PCA available | Fewer random starts, faster mode finding |
| Contested-only | `--contested-only` flag | Opt-in | Halve parameter space for 2D |

## Consequences

**Benefits:**
- Supermajority chambers get tuning resources proportional to their difficulty
- Beta initialization eliminates one source of mode-splitting (randomly initialized discrimination parameters)
- Contested-only filtering provides an escape hatch for the hardest chambers
- All changes are backward-compatible — no existing behavior changes unless flags are used

**Costs:**
- Adaptive N_TUNE increases runtime by ~50% for supermajority sessions (~5-10 extra minutes on M3 Pro)
- Beta initialization requires PCA loadings (always available when pipeline runs in order)
- Contested-only filtering loses information from near-unanimous votes (acceptable for 2D since those votes don't contribute to the second dimension anyway)

**What this won't fix:**
- The 79th Senate's fundamental mode-splitting may persist even with 4000 tuning steps — the 75% supermajority creates genuinely degenerate posterior geometry
- For these extreme cases, ADR-0110's tiered quality gate allows the ecologically valid point estimates to be used despite imperfect convergence

**Related:**
- ADR-0054 — 2D IRT pipeline integration (original constants)
- ADR-0104 — IRT robustness flags (`--contested-only` for 1D)
- ADR-0110 — Tiered convergence quality gate (handles cases where tuning alone isn't enough)
- `analysis/06_irt_2d/irt_2d.py` — Implementation
