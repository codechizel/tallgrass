# ADR-0108: Dimension 1 Informative Prior for Ideology Recovery

**Date:** 2026-03-11
**Status:** Accepted

## Context

The 1D IRT model (Phase 05) produces misleading results in supermajority chambers due to the horseshoe effect — the dominant axis of vote variation in these chambers is establishment-vs-rebel, not left-vs-right ideology. Ultra-conservative legislators like Tim Huelskamp (79th Senate) appear "liberal" because they and Democrats both produce Nay votes on the same bills, for completely opposite reasons.

ADR-0104 introduced `--horseshoe-remediate`, which refits using PC2-filtered votes and a PC2 informative prior. ADR-0107 introduced `--init-strategy 2d-dim1`, which initializes MCMC chains with 2D IRT Dimension 1 scores. Both help, but neither fully solves the problem:

- **PC2 remediation** requires vote filtering (discarding data) and uses PCA (a variance decomposition, not a model-based estimate) as the prior source.
- **2d-dim1 initialization** only sets starting values. If the 1D likelihood's dominant mode is on the contrarian axis, the chains drift away during tuning. The 79th Senate shows r = -0.13 between 1D IRT and 2D Dim 1 — essentially orthogonal — confirming that initialization alone cannot overcome a strong wrong-dimension mode.

The 2D IRT model (Phase 06) correctly separates ideology (Dim 1) from contrarianism (Dim 2) via PLT identification. Its Dimension 1 scores are converged Bayesian estimates of ideology with full posterior uncertainty. This is a stronger prior source than PCA PC2, and using it as an informative prior (not just initialization) constrains the 1D posterior to stay near the ideology dimension.

## Decision

### New robustness flag: `--dim1-prior`

Add a `--dim1-prior` flag to Phase 05 (flat IRT), Phase 07 (hierarchical IRT), and the experimental joint IRT. When enabled:

1. Load 2D IRT Dimension 1 scores from Phase 06 output
2. Standardize to zero-mean, unit-variance
3. Use as **informative priors** on legislator ideal points: `xi ~ Normal(dim1, sigma)`
4. Also use as MCMC initialization (belt and suspenders)

### Companion flag: `--dim1-prior-sigma`

Controls the tightness of the informative prior (default: 1.0).

| Sigma | Effect | Use case |
|-------|--------|----------|
| 0.5 | Strong constraint — posterior ≈ 2D Dim 1 with minor refinement | Severe horseshoe (79th Senate) |
| 1.0 | Moderate — ideology recovered, individual positions can adjust | Default recommendation |
| 2.0 | Weak nudge — may not overcome strong contrarian signal | Mild distortion only |

### Implementation by model type

**Flat IRT (Phase 05, experimental joint):** Reuses the existing `external-prior` identification strategy (ADR-0103). The 2D Dim 1 scores are passed as `external_priors` to `build_irt_graph()`, which sets `xi_free ~ Normal(dim1_standardized, sigma)`. No new model code required — pure wiring.

**Hierarchical IRT (Phase 07, per-chamber and joint):** The hierarchical model decomposes `xi = mu_party + sigma_within * xi_offset`. Rather than modifying the decomposition, add a `pm.Potential` on the composed `xi`:

```python
if dim1_prior is not None:
    pm.Potential(
        "dim1_prior",
        pm.logp(pm.Normal.dist(mu=dim1_values, sigma=dim1_prior_sigma), xi),
    )
```

This adds a soft observation that `xi` should be near the 2D Dim 1 values without changing the hierarchical structure. Party-level pooling, `sigma_within`, and `xi_offset` priors are unaffected.

### Flag behavior

- `--dim1-prior` implies `--promote-2d` (automatic 2D cross-referencing in the report)
- Requires Phase 06 (2D IRT) results to exist; fails with a clear error if not found
- Not auto-triggered — explicit user decision, same pattern as `--horseshoe-remediate`
- Report shows a dedicated section documenting: prior sigma, match count, and correlation with standard model (if both were run)

### Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `DIM1_PRIOR_SIGMA_DEFAULT` | 1.0 | Default width of the Dim 1 informative prior |

### CLI

```bash
just irt 2001-02 --dim1-prior                            # default sigma=1.0
just irt 2001-02 --dim1-prior --dim1-prior-sigma 0.5     # tighter constraint
just hierarchical 2001-02 --dim1-prior                   # hierarchical per-chamber
just hierarchical 2001-02 --dim1-prior --run-joint       # hierarchical joint
```

## Consequences

**Benefits:**
- Principled Bayesian solution to the horseshoe effect — informative priors from a model-based source (2D IRT Dim 1) constrain the 1D posterior to the ideology dimension
- No vote filtering required — the full vote set is used, unlike `--horseshoe-remediate`
- Stronger prior source than PC2 — converged Bayesian estimates with uncertainty vs. variance decomposition
- Reuses existing `external-prior` identification strategy (Phase 05) — no new model code for flat IRT
- Combines with `--init-strategy 2d-dim1` for initialization + prior (belt and suspenders)

**Trade-offs:**
- Requires Phase 06 (2D IRT) to have run and converged — adds a reverse dependency (Phase 05 ← Phase 06)
- Not automatic — researcher must inspect diagnostics and decide to enable
- The prior introduces bias toward the 2D model's solution; if the 2D model is wrong, the 1D model inherits that error (mitigated by sigma > 0 allowing data to push back)
- For balanced chambers where 1D already recovers ideology, the prior is unnecessary overhead

**Risks:**
- If 2D IRT convergence is poor (R-hat > 1.05), using its Dim 1 as a prior could introduce noise. Mitigated by checking 2D convergence before enabling.
- Sensitivity to sigma: too tight (0.25) effectively replaces the 1D model with the 2D; too loose (3.0+) has no effect. Default 1.0 validated against PC2 remediation results.

## Alternatives Considered

1. **Use 2D Dim 1 directly as the canonical score** — skips the 1D model entirely. Rejected: the 2D model has relaxed convergence thresholds (R-hat < 1.05 vs. < 1.01) and is considered experimental. Using it to inform the 1D model is more defensible than replacing 1D.

2. **Contested-only filtering (`--contested-only`)** — already exists. Removes non-contested votes. Complementary but loses data: the 79th Senate has only 117 contested votes (27%) vs. 437 total.

3. **Build a 2D hierarchical model** — most principled but highest complexity. Deferred to future work.

4. **Automatic promotion** — auto-trigger when horseshoe is detected and 2D results exist. Rejected for now: researcher judgment about which dimension to recover should remain explicit.

## Superseded

While `--dim1-prior` works (Huelskamp moves to xi=+3.349), this approach is superseded by **canonical ideal point routing** (`docs/canonical-ideal-points.md`). The conclusion: if the prior must dominate the likelihood to produce correct results, the 1D model is contributing noise, not signal. The field-standard solution (DW-NOMINATE) is to use 2D Dim 1 directly. The `--dim1-prior` flag remains available for research.

Additionally, the `--init-strategy auto` default introduced by ADR-0107 caused a regression in Phase 06: the horseshoe-confounded 1D IRT scores were used to initialize the 2D model, degrading the very Dim 1 scores this ADR depends on. See `docs/canonical-ideal-points.md` for the full narrative.

## Related

- [ADR-0103](0103-irt-identification-strategy-system.md) — IRT identification strategy system (provides `external-prior` mechanism)
- [ADR-0104](0104-irt-robustness-flags.md) — IRT robustness flags (provides `--horseshoe-remediate` pattern)
- [ADR-0107](0107-shared-init-strategy.md) — Shared MCMC init strategy (provides `--init-strategy 2d-dim1`)
- [ADR-0054](0054-2d-irt-pipeline-integration.md) — 2D IRT pipeline integration (upstream data source)
- `docs/canonical-ideal-points.md` — **Superseding article:** from 1D fixes to 2D Dim 1 promotion
- `docs/dim1-informative-prior.md` — Full article with methodology, evidence, and sensitivity analysis
- `docs/dim1-prior-implementation-plan.md` — Step-by-step implementation plan
- `docs/79th-horseshoe-robustness-analysis.md` — Empirical evidence motivating this decision
