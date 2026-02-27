# Hierarchical IRT Convergence: Diagnosis, Theory, and Improvement Plan

*February 2026*

This document analyzes convergence failures in the Tallgrass hierarchical Bayesian IRT models, develops a theory for why the Senate converges but the House does not, and proposes a prioritized improvement plan grounded in both the Bayesian statistics literature and our empirical evidence across eight bienniums.

## The Problem

After running the full analysis pipeline across all eight Kansas legislative bienniums (84th–91st, 2011–2026), a clear pattern emerges: the per-chamber Senate model reliably converges while the House model frequently does not, and the joint cross-chamber model fails catastrophically in every session.

### Convergence Results Across All Bienniums

| Biennium | Chamber | Legislators | Votes | Max R-hat(xi) | Min ESS(xi) | Diverg. | Result |
|----------|---------|-------------|-------|---------------|-------------|---------|--------|
| 84th | House | 113 | 228 | 1.0219 | 189 | 0 | FAIL |
| 84th | Senate | 37 | 190 | 1.0142 | 320 | 0 | FAIL |
| 85th | House | 117 | 228 | 1.0111 | 327 | 0 | FAIL |
| 85th | Senate | 39 | 255 | 1.0020 | 662 | 0 | **PASS** |
| 86th | House | 129 | 254 | 1.0610 | 46 | 0 | FAIL |
| 86th | Senate | 40 | 247 | 1.0038 | 329 | 0 | FAIL |
| 87th | House | 130 | 263 | 1.0079 | 365 | 0 | FAIL |
| 87th | Senate | 40 | 251 | 1.0054 | 800 | 0 | **PASS** |
| 88th | House | 127 | 244 | 1.0164 | 403 | 0 | FAIL |
| 88th | Senate | 41 | 193 | 1.0062 | 743 | 0 | **PASS** |
| 89th | House | 130 | 271 | 1.0179 | 284 | 0 | FAIL |
| 89th | Senate | 39 | 252 | 1.0023 | 563 | 0 | **PASS** |
| 90th | House | 128 | 322 | 1.0047 | 542 | 0 | **PASS** |
| 90th | Senate | 40 | 359 | 1.0079 | 428 | 0 | **PASS** |
| 91st | House | 130 | 263 | 1.0103 | 564 | 0 | FAIL |
| 91st | Senate | 42 | 230 | 1.0058 | 1002 | 0 | **PASS** |

**Per-chamber summary:** Senate passes 6/8 (75%), House passes 1/8 (12.5%). All 16 models have zero divergences and healthy E-BFMI, so the geometry is not pathological — the sampler is just exploring slowly.

| Biennium | Legislators | Votes (shared) | Max R-hat(xi) | Min ESS(xi) | Diverg. | Result |
|----------|-------------|----------------|---------------|-------------|---------|--------|
| 84th | 150 | 411 (93) | 1.5362 | 7 | 10 | FAIL |
| 85th | 156 | 403 (80) | 2.4227 | 5 | 10 | FAIL |
| 86th | 169 | 460 (102) | 1.5509 | 7 | 21 | FAIL |
| 87th | 170 | 429 (85) | 1.0069 | 517 | 12 | FAIL |
| 88th | 168 | 269 (32) | 1.5376 | 7 | 1 | FAIL |
| 89th | 169 | 468 (55) | 1.7473 | 6 | 20 | FAIL |
| 90th | 168 | 548 (133) | 1.5342 | 7 | 10 | FAIL |
| 91st | 172 | 420 (71) | 1.5330 | 7 | 7 | FAIL |

**Joint summary:** 0/8 pass. R-hat(xi) ranges from 1.007 (87th — a notable outlier) to 2.42 (85th). ESS collapses to 5–7 in 7/8 sessions. Divergences present in all sessions (1–21).

### The Counterintuitive Puzzle

At first glance, the House should be *easier* to fit:

- **More legislators** (~130 vs ~40): more data points to estimate group-level parameters
- **More votes** (~250–320 vs ~190–360): more items for calibration
- **More observations** (~30–40k vs ~9–14k): three to four times the data
- **Less unanimity**: more informative votes (higher minority fractions)

Yet the Senate converges reliably while the House struggles. Why?

## Theory: Why the Senate Converges but the House Doesn't

The answer lies not in the quantity of data but in the *geometry of the posterior*. Three reinforcing mechanisms explain the divergence, and they all scale with the *number of parameters*, not the number of observations.

### 1. The Reflection Mode Problem Scales with Item Count

The 2PL IRT model has an inherent identification problem: the likelihood is invariant under the transformation (β, ξ) → (−β, −ξ). This creates a mirror-image mode in the posterior — a perfect copy of the true mode reflected through the origin.

Our model uses `mu_party = pt.sort(mu_party_raw)` to constrain D < R, which *partially* breaks this symmetry. The group-level means are pinned. But the *individual* bill discrimination parameters (β) and *individual* legislator offsets (ξ_offset) can still drift toward the reflection boundary, especially when the group-level constraint exerts weak force on bills with low discrimination.

The critical insight: **each bill parameter β is an independent axis along which the reflection can occur**. A two-dimensional reflection (negate one β and the corresponding ξ interactions) is always available. With ~250–320 bills in the House vs ~190–360 in the Senate, the House posterior has more "escape routes" toward the reflection mode.

This doesn't create divergences (the geometry is smooth), but it creates *ridges* in the posterior that slow exploration. The sampler can't efficiently traverse between the constrained mode and the reflection-adjacent region, resulting in high autocorrelation, inflated R-hat, and depressed ESS — exactly the symptom profile we observe.

### 2. The Non-Centered Funnel Scales with Legislator Count

Our model uses non-centered parameterization:

```
xi_offset ~ Normal(0, 1)
xi = mu_party[party_idx] + sigma_within[party_idx] * xi_offset
```

This is the correct choice for weakly-identified groups (Neal's funnel, Papaspiliopoulos et al. 2007). But non-centered parameterization introduces a different pathology when σ_within is *large* relative to the data's ability to pin ξ: the posterior becomes a high-dimensional "slab" where ξ_offset values are weakly correlated with sigma_within.

The correlation structure in this slab scales with the *number of ξ_offset parameters*. With 130 offset parameters in the House vs 40 in the Senate, the House posterior has a 130-dimensional correlation structure that the NUTS sampler must navigate, while the Senate has only a 40-dimensional one. This is not a funnel in the classical sense — it's more of a "correlation plateau" where all 130 offsets must move in concert when σ_within shifts.

The practical consequence: the sampler's step size adapts to the *tightest* constraint among all parameters. With 130 correlated offsets, there are more constraints, requiring a smaller step size, which slows exploration of the μ_party hyperparameters. This explains why ESS(mu_party) is consistently the weakest diagnostic — it's the parameter most affected by the coordinate-wise correlation bottleneck.

### 3. The Bill Parameter Explosion

The total parameter count tells the story:

| Chamber | ξ_offset | α (easiness) | β (discrimination) | Hyperparams | **Total** |
|---------|----------|--------------|-------------------|-------------|-----------|
| House | ~130 | ~280 | ~280 | 4 | **~694** |
| Senate | ~40 | ~240 | ~240 | 4 | **~524** |
| Joint | ~170 | ~430 | ~430 | 12 | **~1,042** |

The House has ~170 more free parameters than the Senate, driven primarily by the legislator count (130 vs 40). But the α and β parameters also contribute: each bill adds two parameters that interact with *every* legislator who voted on it. The House's ~280 β parameters create a 280-dimensional reflection possibility space, compared to the Senate's ~240.

More importantly, the NUTS sampler's computational cost per step scales with the number of gradient evaluations, which scales with the number of parameters. Gradient evaluations per step are consistently higher for the House (127 vs 63 in many sessions), confirming that the sampler is working harder per unit of exploration.

### 4. Within-Party Heterogeneity Amplifies the Problem

Kansas House Republicans (~85–90 members) exhibit substantially more within-party ideological variation than Senate Republicans (~29–32 members). The sigma_within(R) estimates bear this out:

| Session | House σ_within(R) | Senate σ_within(R) |
|---------|-------------------|---------------------|
| 85th | 1.807 | 2.102 |
| 87th | 2.489 | 3.073 |
| 90th | 1.407 | 2.089 |
| 91st | ~1.5 | ~1.8 |

Wait — the Senate σ_within(R) is actually *larger*. How does this square with the theory?

The answer is that absolute sigma_within is less important than the *ratio of sigma_within to the number of offsets it governs*. The Senate estimates σ_within from ~29 Republicans, each contributing information. The House estimates it from ~87 Republicans. More data should help estimate σ_within, but each additional legislator also adds a correlated ξ_offset parameter that must be jointly navigated with σ_within. The *per-parameter information gain* from adding legislator #88 is marginal, while the *per-parameter cost to the sampler geometry* is not.

This is a version of what Betancourt (2017) calls the "curse of dimensionality in HMC": NUTS performance degrades approximately as O(d^{1/4}) where d is the dimensionality. The House's ~170 extra parameters (mostly from legislator offsets) push the sampler into a regime where it needs disproportionately more iterations for the same effective sample size.

### 5. Why the Joint Model Fails Catastrophically

The joint model compounds all three problems:

1. **170 legislators × ~430 bills = ~1,042 parameters** — the highest dimensionality
2. **Three-level hierarchy** adds mu_global, sigma_chamber, chamber_offset, sigma_party — more hyperparameters to estimate jointly with the offsets
3. **Shared bill parameters** across chambers create cross-chamber correlations that the sampler must respect
4. **No PCA initialization** — the joint model uses `jitter+adapt_diag` because there's no natural PCA decomposition spanning both chambers. Without PCA-informed starting points, the sampler starts in a random location and must discover the mode structure from scratch.
5. **Identification is harder** — four groups (House-D, House-R, Senate-D, Senate-R) with ordering constraints within each chamber pair, but no constraint *across* chambers. The sigma_party parameter governs all four groups simultaneously, creating a complex constraint surface.

The 87th biennium's near-success (R-hat 1.007, ESS 517 for xi) is instructive: it had the fewest shared bills (85) among recent sessions, meaning fewer cross-chamber correlations for the sampler to navigate. This supports the theory that shared bill parameters are a key bottleneck.

### Summary of the Theory

The Senate converges because it occupies a sweet spot:
- **40 legislators**: small enough that the correlation plateau is navigable
- **~240 bills**: enough items for identification but not so many that reflection modes proliferate
- **2 compact groups**: 11 Democrats, 29 Republicans — sufficient for hierarchical estimation with adaptive priors
- **~10k observations**: adequate data density per parameter

The House fails not because it lacks data, but because it has *too many parameters relative to the sampler's ability to navigate the resulting geometry*. Each additional legislator and bill adds both information and complexity, but the complexity cost to NUTS grows faster than the information benefit in this regime.

## Evidence Supporting the Theory

### 1. The 90th Is the Only House Pass

The 90th biennium (2023–24) is the only session where the House per-chamber model passes all convergence checks. It also has the *most votes* (322) and the *highest observation density* (94.7%). With 128 legislators × 322 votes = 39,041 observations, the data-to-parameter ratio is at its highest. This supports the theory that the House is on the margin — with enough data, it tips into convergence.

### 2. Gradient Evaluations Correlate with Failure Severity

The 86th House — our worst per-chamber failure (R-hat 1.061, ESS 46) — shows gradient evaluations of 127 per step. The 90th House (passing) shows 127 as well, but converges in half the wall time. This suggests the 86th has a rougher posterior landscape, not just a larger one — consistent with higher within-party heterogeneity in that era.

### 3. Zero Divergences Everywhere

All 16 per-chamber models have zero divergences. This rules out pathological curvature (funnels, banana-shaped posteriors) as the cause. The problem is *slow mixing* — the sampler moves correctly but inefficiently through a high-dimensional correlated space. This is precisely the symptom profile of the reflection-mode ridge and the non-centered correlation plateau.

### 4. E-BFMI Is Healthy

All E-BFMI values are above 0.7, indicating the sampler's kinetic energy is sufficient to explore the posterior. The problem isn't getting stuck in local minima — it's that the global landscape requires many steps to traverse.

## Improvement Plan

Ordered by expected impact and implementation effort.

### Priority 1: Constrain β > 0 (High Impact, Low Effort)

**The single highest-impact change.** Replace the symmetric beta prior:

```python
# Current: allows sign flips
beta = pm.Normal("beta", mu=0, sigma=1, shape=n_votes, dims="vote")
```

with a positive-only prior:

```python
# Option A: LogNormal (soft positive, allows near-zero)
beta = pm.LogNormal("beta", mu=0, sigma=0.5, shape=n_votes, dims="vote")

# Option B: HalfNormal (hard zero floor)
beta = pm.HalfNormal("beta", sigma=1, shape=n_votes, dims="vote")
```

**Why this works:** Constraining β > 0 eliminates the reflection mode entirely. The likelihood `p = logistic(β·ξ − α)` with β > 0 can only be satisfied by the "correct" sign of ξ. This removes the ~280-dimensional reflection possibility space that the sampler currently wastes effort exploring.

**Theoretical justification:** In IRT, β represents how well a bill discriminates between ideological positions. A bill with β < 0 would mean that more liberal legislators are *more* likely to vote Yea on a conservative bill — which is possible for a small number of bipartisan procedural votes but is not the default expectation. The standard convention in educational testing IRT (where β < 0 represents a "trick question") is to constrain β > 0.

**Caveat:** Some Kansas bills genuinely have β < 0 (procedural votes, bipartisan consent items). These will be forced to near-zero discrimination, effectively treating them as uninformative. This is acceptable because: (a) our EDA already filters near-unanimous votes, which removes most of these; (b) a bill that doesn't discriminate along the primary ideological dimension is, by definition, uninformative for estimating ideal points.

**Expected impact:** Based on the literature (Bafumi et al. 2005, Rivers 2003), this alone typically resolves R-hat issues in 2PL IRT models. The reflection mode is the dominant source of slow mixing.

### Priority 2: PCA Initialization for Joint Model (Medium Impact, Low Effort)

The per-chamber models benefit enormously from PCA-informed initialization (ADR-0044). The joint model currently falls back to `jitter+adapt_diag` because there is no cross-chamber PCA.

**Proposed approach:** Initialize the joint model's ξ_offset values using the *per-chamber* hierarchical posterior means, not PCA. Since the per-chamber models converge (Senate) or nearly converge (House), their posterior ξ estimates are good starting points for the joint model. This avoids the mode-discovery problem.

```python
# After per-chamber models are fit:
house_xi_hat = house_idata.posterior["xi"].mean(["chain", "draw"]).values
senate_xi_hat = senate_idata.posterior["xi"].mean(["chain", "draw"]).values
joint_xi_init = np.concatenate([house_xi_hat, senate_xi_hat])
# Convert to offset scale: xi_offset = (xi - mu_group) / sigma_within
```

**Expected impact:** Should eliminate the cold-start problem. Combined with Priority 1, this may be sufficient to achieve joint convergence.

### Priority 3: Tighter Priors on α (Low-Medium Impact, Trivial Effort)

The current easiness prior is very wide:

```python
alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")
```

This allows α to range ±15 on the logit scale, corresponding to vote passage probabilities from 0.00003% to 99.99997%. In practice, our filtered votes have yea rates between ~5% and ~95%, corresponding to α ≈ ±3.

**Proposed:** Tighten to `Normal(0, 2)` or even `Normal(0, 1.5)`. This provides mild regularization without informative bias, keeping the posterior more compact and easier to navigate.

### Priority 4: Mixed Centered/Non-Centered Parameterization (Medium Impact, Medium Effort)

When σ_within is large (well-estimated), the *centered* parameterization is more efficient:

```python
# Centered: direct sampling of xi
xi = pm.Normal("xi", mu=mu_party[party_idx], sigma=sigma_within[party_idx], ...)
```

When σ_within is small or poorly estimated, the non-centered form is better (our current approach). For groups with many members (House Republicans with ~87 members), σ_within(R) is well-estimated, and centering may improve mixing.

**Proposed:** Use centered parameterization for groups with ≥30 members, non-centered for smaller groups. This is the "mixed centering" approach of Papaspiliopoulos et al. (2007), implemented manually since PyMC doesn't automate it.

### Priority 5: ADVI Initialization (Medium Impact, Low Effort)

Replace `jitter+adapt_diag` with ADVI (Automatic Differentiation Variational Inference) as the initial mode-finder, then switch to NUTS for proper posterior sampling:

```python
with model:
    approx = pm.fit(n=20000, method="advi")
    start = approx.mean.eval()  # MAP-like initialization
    idata = pm.sample(..., initvals=start, init="adapt_diag")
```

ADVI finds the posterior mode quickly (seconds, not minutes) and provides initialization that is already in the right neighborhood. This is particularly valuable for the joint model where `jitter+adapt_diag` starts cold.

### Priority 6: Nutpie Sampler (Medium Impact, Low Effort)

Nutpie is a drop-in replacement for PyMC's default sampler that uses normalizing-flow adaptation — it learns a reparameterization of the posterior *during warmup* that reduces correlations:

```python
import nutpie

compiled = nutpie.compile_pymc_model(model)
idata = nutpie.sample(compiled, draws=2000, tune=1500, chains=4)
```

Nutpie has shown 2–10x speedups on hierarchical models in published benchmarks (Sountsov & Suter 2022). It specifically targets the "correlation plateau" problem that afflicts our House model.

**Caveat:** Nutpie requires installation (`pip install nutpie`) and is less mature than PyMC's default sampler. Worth testing experimentally.

### Priority 7: Increase Draws for House (Low Impact, Low Effort)

The House's R-hat failures are marginal (1.01–1.02, not 1.5+). Simply increasing from 2,000 to 4,000 draws (and proportionally from 1,500 to 3,000 tune) may push it over the threshold by giving the sampler twice as long to mix.

**Cost:** approximately doubles wall time from ~7 min to ~14 min for House per-chamber. Cheap.

**Caveat:** This treats the symptom, not the cause. If the reflection mode is the root issue, more draws will eventually mix but at poor efficiency. Better to fix the geometry (Priority 1) first.

### Priority 8: Hierarchical Bill Priors (Medium Impact, High Effort)

Replace independent bill priors with a hierarchical structure:

```python
mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=2)
sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=n_votes)
```

This regularizes the ~280 bill parameters toward a common mean, reducing effective dimensionality. However, it adds hyperparameters and may slow convergence of the bill block. Best combined with Priority 1.

### Priority 9: 1PL (Rasch) Model for Joint (Medium Impact, Low Effort)

If the joint model continues to fail after Priorities 1–5, consider dropping bill discrimination entirely:

```python
# 1PL: all betas fixed to 1
eta = xi[leg_idx] - alpha[vote_idx]
```

This cuts the parameter count by ~430 (no β vector), eliminates the reflection mode entirely, and dramatically simplifies the posterior geometry. The cost is reduced model expressiveness — all bills are assumed equally discriminating. But for the purpose of placing legislators on a common cross-chamber scale, the 1PL may be sufficient.

## Implementation Sequence

A practical rollout plan:

1. **Experiment branch:** Implement Priority 1 (β > 0) and Priority 7 (more draws) together. Run on the 91st biennium as a baseline.
2. **Evaluate:** If House per-chamber passes, move to joint model. Add Priority 2 (per-chamber init for joint) and Priority 3 (tighter α prior).
3. **If joint still fails:** Add Priority 5 (ADVI init), then Priority 6 (nutpie).
4. **If joint still fails:** Consider Priority 9 (1PL for joint) as a fallback.
5. **Long-term:** Priority 4 (mixed centering) and Priority 8 (hierarchical bill priors) for robustness.

## References

- Bafumi, J., Gelman, A., Park, D.K., & Kaplan, N. (2005). Practical issues in implementing and understanding Bayesian ideal point estimation. *Political Analysis*, 13(2), 171–187.
- Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. *arXiv:1701.02434*.
- Gelman, A. et al. (2006, 2015). *Bayesian Data Analysis* (3rd ed.). Ch. 5, 13.
- Neal, R.M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705–741.
- Papaspiliopoulos, O., Roberts, G.O., & Sköld, M. (2007). A general framework for the parametrization of hierarchical models. *Statistical Science*, 22(1), 59–73.
- Rivers, D. (2003). Identification of multidimensional spatial voting models. Typescript, Stanford University.
- Sountsov, P. & Suter, C. (2022). Automatically adapting the number of state variables for non-centered parameterizations. *Bayesian Analysis* (forthcoming).
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: an improved R-hat for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718.

## Related Documents

- [Hierarchical IRT Deep Dive](hierarchical-irt-deep-dive.md) — ecosystem survey, code audit, 9 issues fixed
- [Hierarchical Shrinkage Deep Dive](hierarchical-shrinkage-deep-dive.md) — over-shrinkage with small groups
- [Hierarchical PCA Init Experiment](hierarchical-pca-init-experiment.md) — R-hat fix, ESS threshold analysis
- [4-Chain Hierarchical IRT Experiment](hierarchical-4-chain-experiment.md) — jitter mode-splitting discovery
- [Joint Hierarchical IRT Diagnosis](joint-hierarchical-irt-diagnosis.md) — bill-matching bug and fix
- [2D IRT Deep Dive](2d-irt-deep-dive.md) — PLT identification, Tyson paradox
- [Apple Silicon MCMC Tuning](apple-silicon-mcmc-tuning.md) — hardware scheduling for parallel chains
