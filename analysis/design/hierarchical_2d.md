# Hierarchical 2D IRT Design Choices

## Assumptions

- The 2D latent structure (ideology + secondary axis) identified by Phase 06 is real, not a model artifact.
- Party membership provides meaningful hierarchical structure for both dimensions.
- Informative priors from Phase 06 (Dim 2 party averages) and Phase 07 (party means) are reliable enough to regularize without biasing.
- Non-centered parameterization avoids funnel geometry for both dimensions independently.

## Parameters & Constants

| Parameter | Value | Justification | Location |
|-----------|-------|---------------|----------|
| `N_SAMPLES` | 2000 | Matches Phase 06/07 | `hierarchical_2d.py` |
| `N_TUNE` | 2000 (4000 supermajority) | ADR-0112 adaptive tuning | `hierarchical_2d.py` |
| `N_CHAINS` | 4 | Standard for convergence diagnostics | `hierarchical_2d.py` |
| `SUPERMAJORITY_THRESHOLD` | 0.70 | ADR-0112 | `hierarchical_2d.py` |
| `H2D_RHAT_THRESHOLD` | 1.05 | Relaxed, matches Phase 06 | `hierarchical_2d.py` |
| `H2D_ESS_THRESHOLD` | 200 | Relaxed, matches Phase 06 | `hierarchical_2d.py` |
| `MAX_DIVERGENCES` | 50 | Relaxed, matches Phase 06 | `hierarchical_2d.py` |
| `SMALL_GROUP_THRESHOLD` | 20 | Gelman 2015, matches Phase 07 | `hierarchical_2d.py` |
| `SMALL_GROUP_SIGMA_SCALE` | 0.5 | Informative prior for small groups | `hierarchical_2d.py` |
| `dim1_sigma_prior` | 1.0 (informative) / 2.0 (diffuse) | Tighter when Phase 07 data available | `hierarchical_2d.py` |
| `dim2_sigma_prior` | 2.0 | Always wider — Dim 2 has weaker signal | `hierarchical_2d.py` |

## Model Specification

### Prior Chain

```
Phase 07 (Hierarchical 1D):
  mu_party[D], mu_party[R]  →  mu_party_dim1_raw ~ Normal(mu_07, 1.0)
  sigma_within              →  informs expected within-party spread

Phase 06 (Flat 2D IRT):
  dim2_party_avg[D,R]       →  mu_party_dim2 ~ Normal(dim2_avg, 2.0)
```

### Identification Strategy

1. **Rotation**: PLT constraint on discrimination matrix (same as Phase 06)
   - beta[0, 1] = 0 (rotation anchor)
   - beta[1, 1] > 0 (HalfNormal — positive diagonal)
2. **Reflection Dim 1**: sort(mu_party_dim1) ensures D < R
3. **Post-hoc sign check**: Flip Dim 1 if Republican mean < 0
4. **Scale**: Normal(0, 1) offset prior provides scale identification

### Comparison to Phase 06 and Phase 07

| Feature | Phase 06 (Flat 2D) | Phase 07 (Hier 1D) | Phase 07b (Hier 2D) |
|---------|-------------------|--------------------|---------------------|
| Dimensions | 2 | 1 | 2 |
| Party pooling | No | Yes | Yes |
| PLT identification | Yes | N/A | Yes |
| Horseshoe resolution | Yes (Dim 1) | No | Yes (Dim 1) |
| Dim 2 convergence | Often poor | N/A | Better (party priors) |
| Sparse legislator handling | No shrinkage | Shrinkage to party | Shrinkage per dimension |

## Convergence Expectations

- **Dim 1**: Should match or exceed Phase 06 convergence (party pooling regularizes)
- **Dim 2**: Expected improvement over Phase 06 due to informative party priors
- **Supermajority**: Doubled tuning (4000 steps) for >70% majority chambers
- **Small groups**: Tighter sigma prior (HalfNormal(0.5)) for parties with <20 members

## Quality Gates (ADR-0118)

- **Minimum party separation (R4):** Soft `pm.Potential` penalty on Dim 1 when `mu_party_dim1[1] - mu_party_dim1[0] < 0.5`. Same guard as Phase 07.
- **Dimension swap detection (R7):** After extraction, checks if Dim 2 separates parties better than Dim 1. Swaps columns if so, then re-runs sign check. Records `dimension_swap_corrected` in convergence summary.

## Downstream Implications

- Canonical ideal point routing prefers H2D Dim 1 over flat 2D Dim 1 when converged
- PPC (Phase 08) includes H2D in 4-way model comparison
- group_params output has per-dimension rows (unlike Phase 07's single-dimension rows)
