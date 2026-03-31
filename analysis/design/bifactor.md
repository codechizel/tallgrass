# Phase 06b: Bifactor IRT Design

## Model Specification

```
P(Yea_ij = 1) = logit^-1(
    a_G[j] * theta_G[i]
    + a_S1[j] * theta_S1[i] * mask_high[j]
    + a_S2[j] * theta_S2[i] * mask_low[j]
    - d[j]
)

theta_G, theta_S1, theta_S2 ~ Normal(0, 1)    per legislator
a_G                         ~ Normal(0, 1)    all bills
a_S1_raw, a_S2_raw          ~ Normal(0, 1)    all bills (masked)
d                           ~ Normal(0, 5)    all bills
```

## Assumptions

1. Phase 05 (1D IRT) has run and produced `bill_params_{chamber}.parquet` with `beta_mean` column.
2. Three-factor structure: general ideology + partisan-specific + bipartisan-specific.
3. Factors are orthogonal (independence of priors enforces this).
4. Bill classification by 1D IRT discrimination is meaningful and stable.

## Bill Classification Strategy

| Group | Criterion | Loads On | Rationale |
|-------|-----------|----------|-----------|
| High-disc | \|beta\| > 1.5 | General + Specific 1 | Partisan votes — ideology + partisan-specific |
| Low-disc | \|beta\| < 0.5 | General + Specific 2 | Bipartisan votes — ideology + contrarian-specific |
| Medium | 0.5 ≤ \|beta\| ≤ 1.5 | General only | Pure ideology anchors — no specific factor loading |

**Not circular:** Uses 1D IRT beta *magnitude* (from Phase 05), not the bifactor's own parameters. Medium-disc bills serve as structural anchors for the general factor.

**Fallback:** EDA `vote_alignment.parquet` (party-line → high-disc proxy, bipartisan → low-disc proxy).

## Identification

| Problem | Solution | Comparison to Phase 06 (2D M2PL) |
|---------|----------|----------------------------------|
| Rotation indeterminacy | None — masks fix factor assignments structurally | PLT constraints on beta matrix |
| Scale | Unit-variance priors on all theta | Unit-variance priors on xi |
| Sign (general) | Post-hoc flip: R mean > 0 on theta_G | Post-hoc flip on Dim 1 |
| Sign (specific) | Not constrained — specific factors have no inherent direction | PLT: beta[1,1] > 0 via HalfNormal |

The bifactor identification is simpler than PLT because the masks eliminate rotation freedom.

## Priors

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| theta_G | Normal(0, 1) | Standard IRT prior, centered |
| theta_S1, theta_S2 | Normal(0, 1) | Orthogonal to general by construction |
| a_G | Normal(0, 1) | Unconstrained general discrimination |
| a_S1_raw, a_S2_raw | Normal(0, 1) | Masked; non-target bills contribute zero |
| d | Normal(0, 5) | Diffuse difficulty (matches Phase 05/06) |

## MCMC Settings

Same as Phase 06 (2D IRT):
- 2000 draws, 2000 tune (4000 for supermajority), 4 chains, seed 42
- nutpie (Rust NUTS sampler)
- Relaxed convergence: R-hat < 1.05, ESS > 200, divergences < 50

## Initialization

| Factor | Source | Rationale |
|--------|--------|-----------|
| theta_G | PCA ideology_score or PC1 (via resolve_init_source) | Best available ideology axis |
| theta_S1 | PCA PC2 (establishment axis) | Captures the secondary dimension |
| theta_S2 | Zeros | No strong prior for bipartisan-specific behavior |

## Diagnostics

### ECV (Explained Common Variance)

```
ECV = sum(a_G^2) / [sum(a_G^2) + sum(a_S1^2) + sum(a_S2^2)]
```

| ECV Range | Interpretation |
|-----------|---------------|
| > 0.70 | Unidimensional model adequate; bifactor adds little |
| 0.60 - 0.70 | Moderate multidimensionality |
| < 0.60 | Meaningful bifactor structure; specific factors carry real signal |

### omega_h (Omega Hierarchical)

```
omega_h = (sum(a_G))^2 / [(sum(a_G))^2 + sum(a_S1^2) + sum(a_S2^2) + sum(unique_var)]
```

Proportion of total score variance due to the general factor. Higher = more reliable as a unidimensional ideology measure.

## Empirical Validation (14 Bienniums)

Phase 06b was run on all 14 Kansas bienniums (78th-91st, 1999-2026). 82nd excluded (missing EDA data).

### House Results

| ECV Band | Sessions | Convergence | theta_G vs 1D r |
|----------|----------|-------------|-----------------|
| > 0.70 (1D adequate) | 78th, 79th, 86th-89th | 9/13 passed | 0.95-0.99 |
| 0.60-0.70 (moderate) | 80th, 81st, 83rd, 90th, 91st | All passed | 0.90-0.97 |
| < 0.60 (strong bifactor) | 84th, 85th | Failed (low omega_h) | 0.41-0.47 |

The 90th and 91st achieved full convergence with ECV ~0.62 — the best bifactor results.

### Senate Results

Only 6/13 converged. Three collapsed to ECV < 0.15 (88th, 90th). Three exhibited sign flips (79th, 86th, 91st). Root cause: small Democrat caucus (8-12 members) starves specific factors.

### Key Findings

1. **Discrimination-based bill classification worked** across all runnable sessions without manual intervention.
2. **General factor is well-identified in House** — theta_G correlates r > 0.90 with 1D IRT in all converged chambers.
3. **Specific factors struggle in supermajority Senate** — insufficient within-party signal for theta_S1/S2 estimation.
4. **ECV diagnostic is critical** — correctly distinguishes sessions where bifactor adds value (ECV < 0.70) from those where 1D suffices.

## Known Limitations

1. **Wasted parameters:** `a_S1_raw` and `a_S2_raw` are sampled for ALL bills, not just their target group. Non-target bills get prior-like posteriors (zero contribution to likelihood). This wastes sampler effort but avoids variable-size tensor complexity.
2. **Specific factor convergence:** Low-disc bills have weak signal by definition. theta_S2 may have wide HDIs for most legislators (same issue as Phase 06 Dim 2). Confirmed empirically: Senate convergence rate 6/13.
3. **Classification boundary sensitivity:** Changing HIGH_DISC_THRESHOLD from 1.5 to 1.0 would shift many bills between groups. Sensitivity analysis recommended but lower priority — classification worked across all runnable sessions.
4. **Not yet in canonical routing:** Phase 06b output does not feed into `canonical_ideal_points.py`. House-only integration is viable for ECV < 0.70 sessions; Senate not recommended at current convergence rates.
5. **Senate small-N problem:** 8-12 Democrats per Senate session is insufficient for stable specific-factor estimation. Three sessions collapsed entirely (ECV < 0.15).

## Comparison with Phase 06 (2D M2PL)

| Feature | Phase 06 (2D M2PL) | Phase 06b (Bifactor) |
|---------|--------------------|-----------------------|
| Dimensions | 2 (free rotation via PLT) | 1 general + 2 specific (structural) |
| Bill constraints | None (all bills load on both dims) | Masks: high/low/medium classification |
| Identification | PLT + post-hoc sign | Orthogonal priors + post-hoc sign |
| Ideology score | Dim 1 (rotation-dependent) | General factor (structurally defined) |
| Parameters | ~2N_leg + 2N_votes + N_votes | ~3N_leg + 4N_votes |
| Key diagnostic | Party-d on dimensions | ECV + omega_h |

## Downstream Implications

- Provides three-dimensional ideal points for each legislator.
- General factor (theta_G) is the candidate "pure ideology" score.
- Specific factors can be used for party loyalty analysis (theta_S1 extremes = unusual partisan behavior) and contrarian detection (theta_S2 extremes = Tyson-like behavior on bipartisan bills).
- If promoted to canonical routing, would replace the current 2D Dim 1 extraction for horseshoe-affected chambers.
