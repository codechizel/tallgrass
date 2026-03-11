# Dimension 1 Informative Priors: Recovering Ideology from Multidimensional Voting

**Date:** 2026-03-11
**Context:** 79th Kansas Legislature (2001–2002), IRT Phases 05–07, joint IRT experiment
**Related:** ADR-0103 (identification strategies), ADR-0104 (robustness flags), ADR-0107 (init strategies), `docs/79th-horseshoe-robustness-analysis.md`, `docs/horseshoe-effect-and-solutions.md`

## The Problem: Tim Huelskamp and the Limits of One Dimension

Tim Huelskamp served in the Kansas Senate during the 79th Legislature (2001–2002) as one of the
chamber's most conservative members. He later went on to represent Kansas's 1st congressional
district in the U.S. House, where he was expelled from the House Agriculture Committee for
defying Republican leadership — a pattern of ideological conviction over party loyalty that
defined his career.

In our 2D Bayesian IRT model, Huelskamp appears exactly where he should: **Dim 1 = +1.4**
(conservative) and **Dim 2 = -2.8** (extreme contrarian). The 2D model correctly separates
his ideology from his willingness to buck the establishment.

In our 1D IRT model — flat, hierarchical, and joint — he appears **liberal**.

This is not a bug in the model. It is a fundamental limitation of projecting two-dimensional
behavior onto a single axis. And it affects far more than one legislator.

## Why One Dimension Fails in Supermajority Chambers

The 79th Kansas Senate had 30 Republicans and 10 Democrats — a 75% supermajority. In this
configuration, the dominant source of vote variation is not left-vs-right ideology but
**establishment-vs-rebel dynamics within the Republican caucus**.

Consider the voting patterns on a typical bill that passes with a strong Republican majority:

| Group | Vote | Reason |
|-------|------|--------|
| Establishment Republicans (20+) | Yea | Party-line support |
| Conservative rebels (Huelskamp, Oleen, etc.) | Nay | Bill is too moderate |
| Democrats (most) | Nay | Bill is too conservative |

The 1D model sees Huelskamp and the Democrats both voting Nay on the same bills. It cannot
distinguish *why* they voted Nay — only that they did. With 73% of votes being non-contested
(only one party splits), the rebel-vs-establishment axis explains more variance than
left-vs-right ideology. The 1D model "correctly" (from a likelihood perspective) recovers
the dimension with the most explanatory power. That dimension is not ideology.

This is the **horseshoe effect**: the ideological spectrum folds back on itself in one
dimension, and the two ends of the horseshoe (far-left Democrats, far-right rebels) overlap.

### Quantitative Evidence

Our robustness analysis of the 79th Senate (documented in
`docs/79th-horseshoe-robustness-analysis.md`) measured the severity:

| Metric | Senate Value | Healthy Baseline |
|--------|-------------|-----------------|
| 1D vs. 2D Dim 1 correlation | **r = -0.13** | r > 0.85 |
| Democrat wrong-side fraction | **30%** | 0% |
| Party overlap fraction | **88%** | < 10% |
| PCA eigenvalue ratio (PC1/PC2) | **1.45** | > 2.0 |
| R–D mean separation | **0.20** | > 1.5 |

The r = -0.13 between 1D IRT and 2D Dim 1 means the two models are measuring **orthogonal
things**. The 1D model is not a noisy version of the ideology dimension — it is a clean
measurement of a completely different dimension (establishment loyalty).

Even the 79th House (70% Republican, only barely supermajority) shows severe distortion:
r = 0.16 between 1D and 2D Dim 1, and r = 0.41 between the full model and a contested-only
refit. When fewer than 40% of votes are cross-party contested, the 1D model is unreliable.

## What the 2D Model Gets Right

The 2D Bayesian IRT model (Phase 06) uses Positive Lower Triangular (PLT) identification to
separate two orthogonal dimensions:

- **Dimension 1 (Ideology):** Liberal ← → Conservative
- **Dimension 2 (Establishment):** Contrarian ← → Establishment

PLT identification constrains the discrimination matrix so that the first vote loads only on
Dim 1 (`beta[0,1] = 0`) and the second vote loads positively on Dim 2 (`beta[1,1] > 0`).
This eliminates rotational ambiguity without over-constraining the posterior.

For the 79th Senate:

| Legislator | Party | Dim 1 (Ideology) | Dim 2 (Establishment) | 1D IRT |
|------------|-------|-------------------|-----------------------|--------|
| Tim Huelskamp | R | **+1.4** (conservative) | **-2.8** (extreme contrarian) | "Liberal" |
| Lana Oleen | R | +0.8 (conservative) | -1.9 (contrarian) | "Most liberal" (xi = -3.84) |
| David Haley | D | -2.1 (liberal) | -0.7 (moderate contrarian) | "Moderate conservative" |
| Anthony Hensley | D | -2.3 (liberal) | -0.6 (moderate) | "Moderate" |

The 2D model correctly places Huelskamp and Oleen as conservatives who happen to rebel against
their party. The 1D model cannot make this distinction.

## Why Initialization Alone Isn't Enough

The existing `--init-strategy 2d-dim1` (ADR-0107) uses the 2D model's Dimension 1 scores as
MCMC chain starting values for the 1D model. The idea is sound: start the chains near the
ideology solution so they converge to the right mode of the 1D posterior.

**It doesn't work for severe horseshoe cases.**

The reason is fundamental. MCMC initialization only determines *where the chains start
exploring*. During the tuning phase, the sampler adapts step sizes and mass matrix to the
local posterior geometry. If the posterior's dominant mode is on the contrarian axis — because
that axis genuinely explains more vote variance — the chains will drift toward that mode
regardless of where they started.

Think of it as placing a marble on a hillside. The initialization is where you place the
marble. The posterior is the landscape. If the ideology valley is shallow and the contrarian
valley is deep, the marble rolls downhill no matter where you set it.

The r = -0.13 between 1D and 2D Dim 1 confirms this: the 1D posterior's dominant mode is
essentially orthogonal to ideology. No amount of initialization can change the shape of the
posterior itself.

## The Solution: Informative Priors from Dimension 1

To constrain the 1D model to recover ideology, we need to change the **posterior landscape**,
not just the starting point. The Bayesian mechanism for this is an **informative prior**.

### Current prior (standard IRT)

```
xi_i ~ Normal(0, 1)    for each legislator i
```

This is a "vague" prior — it says nothing about which dimension the model should recover.
The posterior is dominated by the likelihood, which finds the highest-variance dimension
(contrarianism in supermajority chambers).

### Proposed prior (Dim 1-informed)

```
xi_i ~ Normal(dim1_i, sigma)    for each legislator i
```

Where `dim1_i` is the standardized Dimension 1 score from the 2D model, and `sigma` controls
how tightly constrained the prior is.

This tells the 1D model: "we have strong prior information about the ideological ordering
from a higher-dimensional model. Recover ideology, not contrarianism." The posterior blends
this prior with the 1D likelihood, staying close to the ideology dimension unless the data
very strongly disagree.

### Methodological precedent

This approach has direct precedent in the political science ideal point estimation literature:

1. **Clinton, Jackman & Rivers (2004)** — The IDEAL model uses informative priors on "known"
   legislators (e.g., party leaders) to identify the ideological dimension. Our approach
   extends this to all legislators using estimates from the 2D model.

2. **Shor & McCarty (2011)** — State legislature ideal points use informative priors from
   national-level scores to bridge chambers and sessions. We use 2D Dim 1 as an internal
   bridge between the 2D and 1D models.

3. **Our own `--horseshoe-remediate` flag** — Already uses PC2 scores as informative priors
   with the `external-prior` identification strategy (`xi ~ Normal(PC2, 1.0)`). The
   mechanism is identical; we are substituting a better source (converged 2D Bayesian
   estimates vs. a PCA approximation).

### Why Dim 1 is better than PC2 for this purpose

The `--horseshoe-remediate` flag uses PCA PC2 scores as the prior source. This works but has
limitations:

| Property | PCA PC2 | 2D IRT Dim 1 |
|----------|---------|--------------|
| Source | Variance decomposition (no model) | Converged Bayesian posterior |
| Uncertainty | Point estimate only | Full posterior with credible intervals |
| Identification | Sign ambiguous (requires correction) | PLT-identified (sign stable) |
| Relationship to ideology | Indirect (second component of variance) | Direct (first dimension with ideology interpretation) |
| Vote filtering required | Yes (PC2-dominant votes only) | No (full vote set) |

The 2D IRT Dim 1 is a model-based estimate of ideology that has already been identified via
PLT constraints, converged through MCMC, and validated against party labels. It is
the strongest available prior for the ideology dimension.

### The role of `sigma`

The prior sigma parameter controls the trade-off between the 2D Dim 1 prior and the 1D
likelihood:

| Sigma | Effect | Use case |
|-------|--------|----------|
| 0.5 | Strong constraint — posterior ≈ 2D Dim 1 with minor refinement | Severe horseshoe (79th Senate) |
| 1.0 | Moderate — ideology recovered, individual positions can shift | Default recommendation |
| 2.0 | Weak nudge — may not overcome strong contrarian signal | Mild distortion only |

For the 79th Senate where the 1D model's mode is essentially orthogonal to ideology (r = -0.13),
`sigma = 1.0` should provide sufficient constraint. For borderline cases where r is between
0.3 and 0.7, `sigma = 1.5` to `2.0` allows more data influence.

## Application to Each Model Type

### Flat 1D IRT (Phase 05)

The simplest case. The existing `external-prior` identification strategy already implements
the exact mechanism needed:

```python
# Current code (irt.py line 1899-1906):
elif strategy == IS.EXTERNAL_PRIOR:
    xi_free = pm.Normal(
        "xi_free", mu=external_priors, sigma=external_prior_sigma, shape=n_leg
    )
    xi = pm.Deterministic("xi", xi_free, dims="legislator")
```

No new model code is needed. The `--dim1-prior` flag loads 2D Dim 1 scores, standardizes
them, and passes them as `external_priors` with `strategy=EXTERNAL_PRIOR`. This is exactly
how `--horseshoe-remediate` already works with PC2 scores.

The advantage over `--horseshoe-remediate`: no vote filtering is needed. The informative prior
constrains the dimension directly, so the full vote set provides maximum information for
estimating individual legislator positions within that dimension.

### Hierarchical IRT — Per-Chamber (Phase 07)

The hierarchical model decomposes ideal points as:

```
mu_party ~ Normal(0, 2), sorted (D < R)
sigma_within ~ HalfNormal(adaptive)
xi_offset ~ Normal(0, 1)
xi = mu_party[party_idx] + sigma_within[party_idx] * xi_offset
```

The informative prior cannot simply replace `xi_offset`'s prior because `xi` is a composed
quantity. Instead, add a `pm.Potential` on the composed `xi`:

```python
if dim1_prior is not None:
    pm.Potential(
        "dim1_prior",
        pm.logp(pm.Normal.dist(mu=dim1_values, sigma=dim1_prior_sigma), xi),
    )
```

This adds a soft "observation" that `xi` should be near the 2D Dim 1 values without
modifying the hierarchical decomposition. The party-level structure (`mu_party`, `sigma_within`)
remains intact and continues to provide partial pooling. The Potential just nudges individual
positions toward the ideology axis.

### Hierarchical IRT — Joint Cross-Chamber (Phase 07)

Same mechanism as per-chamber, applied to the combined `xi` vector in the joint model. The
2D Dim 1 scores from both chambers are concatenated (House first, Senate second) to match
`build_joint_graph()`'s legislator ordering.

### Joint Pooled IRT (Experimental)

The experimental joint IRT (`analysis/experimental/joint_irt_experiment.py`) uses the flat
model's `build_and_sample()` with `build_irt_graph()`. It can use the same `external-prior`
strategy as Phase 05. The 2D Dim 1 scores from both chambers are merged into a single
prior array matching the joint matrix's legislator ordering.

## Expected Outcomes

### For Huelskamp specifically

With `sigma = 1.0` and his 2D Dim 1 score of +1.4 (standardized to approximately +1.0), the
prior centers his ideal point firmly on the conservative end. The 1D likelihood may pull toward
the contrarian mode, but the prior provides sufficient resistance. The posterior should settle
near **+1.0 to +1.5** — unambiguously conservative.

### For the 79th Senate generally

| Without Dim 1 prior | With Dim 1 prior (expected) |
|---------------------|---------------------------|
| r = -0.13 (1D vs. 2D Dim 1) | r > 0.85 |
| 30% Democrats on wrong side | 0% |
| 88% party overlap | < 15% |
| Huelskamp, Oleen appear "liberal" | Huelskamp, Oleen appear conservative |
| Dimension = establishment loyalty | Dimension = ideology |

### Sensitivity analysis protocol

Every session run with `--dim1-prior` should also run with the standard model and report:

1. **Prior-standard correlation**: How much does the prior change the ideal points?
   - r > 0.90: Prior has minimal effect (the 1D model was already on the right dimension)
   - r = 0.50–0.90: Prior meaningfully corrected the dimension
   - r < 0.50: Prior substantially restructured the ideal points (expected for horseshoe cases)

2. **Dim 1 prior sigma sensitivity**: Run at sigma = 0.5, 1.0, 2.0 and compare rankings.
   Stable rankings across sigma values indicate robust identification. Sensitive rankings
   indicate the prior is fighting the likelihood — consider whether 2D results are more
   appropriate for that session.

3. **Posterior predictive accuracy**: Ensure the Dim 1-constrained model still predicts votes
   accurately. A large drop in PPC accuracy means the prior is forcing the model too far from
   the data. The ideology dimension should still explain a substantial fraction of votes, even
   if it's not the dominant axis.

## When to Use This

### Recommended triggers

1. **`--horseshoe-diagnostic` detects horseshoe** — Democrat wrong-side fraction > 20% or
   party overlap > 50%. Run with `--dim1-prior` as the primary remediation.

2. **`--promote-2d` shows low 1D-vs-2D correlation** — r < 0.50 between 1D IRT and 2D Dim 1.
   The 1D model is measuring the wrong dimension.

3. **Contested vote fraction < 35%** — Fewer than one in three votes are cross-party
   contested. Intra-party dynamics will dominate the 1D model.

### When NOT to use this

1. **Balanced chambers** (R < 65%) where the 1D model naturally recovers ideology.
   Adding a Dim 1 prior is unnecessary and reduces the model's flexibility.

2. **When 2D IRT hasn't converged** — The prior source must be reliable. Check 2D
   convergence diagnostics (R-hat < 1.05, ESS > 200) before using Dim 1 as a prior.

3. **When you want the contrarian dimension** — Sometimes the establishment-vs-rebel
   axis is analytically interesting. The standard 1D model captures it faithfully.
   Don't force ideology if contrarianism is the question.

## Relationship to Existing Infrastructure

This feature builds on three existing systems:

| System | Role | Status |
|--------|------|--------|
| `external-prior` identification strategy (ADR-0103) | Model mechanism (`xi ~ Normal(prior, sigma)`) | Production |
| `--horseshoe-remediate` robustness flag (ADR-0104) | PC2-based prior (same mechanism, different source) | Production |
| `--init-strategy 2d-dim1` (ADR-0107) | 2D Dim 1 as MCMC initialization | Production |

The Dim 1 informative prior is the logical culmination of these three systems: it uses the
model mechanism from ADR-0103, the robustness flag pattern from ADR-0104, and the 2D Dim 1
data source from ADR-0107 — combining them into a single, principled correction for the
horseshoe effect.

The `--init-strategy 2d-dim1` remains useful as a complement: initialization gets the chains
to the right neighborhood, and the informative prior keeps them there.
