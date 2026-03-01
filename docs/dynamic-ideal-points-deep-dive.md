# Dynamic Ideal Points Deep Dive

**Date:** 2026-02-28
**Scope:** Ecosystem survey, model theory, implementation options, and recommendations for tracking Kansas legislator ideology across bienniums (84th–91st, 2011–2026).

---

## Contents

1. [The Problem: Snapshots vs. Trajectories](#1-the-problem-snapshots-vs-trajectories)
2. [The Martin-Quinn Model](#2-the-martin-quinn-model)
3. [Identification in the Dynamic Setting](#3-identification-in-the-dynamic-setting)
4. [Key Extensions and Alternatives](#4-key-extensions-and-alternatives)
5. [State Legislature Applications](#5-state-legislature-applications)
6. [Software Ecosystem](#6-software-ecosystem)
7. [Performance Comparison](#7-performance-comparison)
8. [Kansas-Specific Considerations](#8-kansas-specific-considerations)
9. [Implementation Options](#9-implementation-options)
10. [Recommendation](#10-recommendation)
11. [References](#11-references)

---

## 1. The Problem: Snapshots vs. Trajectories

Our pipeline currently estimates ideal points independently per biennium (static IRT), then aligns them across sessions via Stocking-Lord IRT linking. Cross-session validation shows this works well (r = 0.940–0.975 for adjacent bienniums), but the approach has fundamental limitations:

- **Scale confounding.** Each biennium's IRT has its own arbitrary scale. Post-hoc affine alignment via bridge legislators corrects location/scale but can't recover information lost by treating sessions independently.
- **No principled uncertainty for change.** If a legislator's ideal point shifts from 0.5 to 0.7 across sessions, is that genuine movement or measurement noise? Static-per-session estimation can't answer this — the confidence intervals from two independent models aren't directly comparable.
- **No partial pooling across time.** A legislator serving one biennium gets no temporal regularization. A legislator serving eight bienniums has eight independent estimates that don't inform each other.
- **Conversion vs. replacement.** Aggregate polarization can increase because moderate members are replaced by extremists (replacement) or because sitting members move to the poles (conversion). Decomposing these requires a single coherent scale across all periods.

Dynamic ideal point models solve these problems by estimating all periods jointly, linking them through a stochastic process on each legislator's latent position.

---

## 2. The Martin-Quinn Model

Martin and Quinn (2002) extended Bayesian IRT to the temporal domain using a state-space framework. The model has two equations:

### Observation equation (measurement model)

For legislator *j* voting on bill *i* at time *t*:

```
P(y_{i,j,t} = 1) = Φ(β_i · θ_{j,t} − α_i)
```

where `α_i` is bill difficulty, `β_i` is bill discrimination, and `θ_{j,t}` is the ideal point. This is the standard 2PL IRT probit model — identical to what we already estimate per biennium.

### State equation (evolution model)

```
θ_{j,t} = θ_{j,t-1} + η_{j,t},    η_{j,t} ~ N(0, τ²_j)
```

A Gaussian random walk. Each legislator's position at time *t* is their previous position plus normally distributed noise. The evolution variance `τ²` controls how much drift is permitted per period.

### Key properties

| τ² value | Behavior |
|----------|----------|
| τ² → 0 | Ideal points frozen (reduces to static IRT across all periods) |
| τ² small (0.01–0.1) | Heavy smoothing — positions change slowly, adjacent periods strongly linked |
| τ² large (1.0+) | Light smoothing — positions can jump substantially between periods |
| τ² → ∞ | No temporal coupling (reduces to independent IRT per period) |

The random walk is **non-stationary**: ideal point uncertainty grows unboundedly as you project further from observed data. End-of-series estimates have wider credible intervals than mid-series.

### Item parameters

Bill parameters `(α_i, β_i)` are **time-period-specific** — each bill belongs to exactly one biennium, and its parameters apply only there. Bills do NOT bridge across periods. The temporal linkage comes entirely from the random walk prior on ideal points.

This is a critical distinction from our current Stocking-Lord linking approach, which explicitly anchors item parameters across sessions. In the dynamic model, items provide within-period measurement while the random walk provides across-period linkage.

### Computation

Martin and Quinn used Gibbs sampling exploiting the conjugate DLM structure:

1. **Forward-filter, backward-sample** ideal point trajectories (Kalman filter + Carter-Kohn backward sampling)
2. **Sample item parameters** via data augmentation (Albert-Chib probit)
3. **Optionally sample τ²** from its inverse-gamma conditional

The original paper ran 200,000 iterations on a Pentium 933 (~11 hours for 9 justices × 47 terms × ~600 cases). Modern samplers (NUTS, variational EM) are dramatically faster.

---

## 3. Identification in the Dynamic Setting

### The static problem (review)

Static IRT is invariant under location shift, scale change, and reflection. Standard fixes: constrain two legislators, standardize to mean 0 / variance 1, or constrain at least one `β > 0` (Bafumi et al. 2005). Our pipeline uses positive `β` constraints plus PCA-informed initialization.

### Dynamic complications

- **Within-period identification.** Each time period faces the same location/scale/reflection problem. The random walk partially addresses this — consecutive periods share a common scale via τ² — but the prior alone doesn't fully identify the model.
- **Cross-period comparability.** If the bill agenda shifts character (e.g., from social to fiscal issues), the latent dimension may not mean the same thing across periods. The random walk assumes scale stability — the same θ = 0.5 means the same thing in 2011 and 2025.
- **Anchor strategy.** MCMCpack uses `theta.constraints` to fix specific legislators' signs or values. idealstan uses `restrict_ind_high`/`restrict_ind_low` to constrain known liberals/conservatives. Both resolve sign and location ambiguity.

### Bridge observations

Bailey (2007) pioneered bridge observations for cross-institutional comparison. For state legislatures, the bridges are **returning legislators** who serve in multiple bienniums. The Markov property means you only need bridges between adjacent periods — each period links to the next through shared legislators, forming a chain.

Our data: ~70–80% overlap for adjacent Kansas bienniums (131 of ~168 legislators matched for 90th→91st). This far exceeds the ~5–10% minimum for identification. The weakest link would be around 2012 redistricting (84th→85th boundary).

---

## 4. Key Extensions and Alternatives

| Model | Key idea | Citation |
|-------|----------|----------|
| **Random walk** (Martin-Quinn) | θ_t = θ_{t-1} + noise | Martin & Quinn 2002 |
| **AR(1)** | θ_t = ρ·θ_{t-1} + noise (mean-reverting) | Kubinec 2019 (idealstan) |
| **Gaussian process** | Continuous-time, squared-exponential kernel | Kubinec 2019 (idealstan) |
| **DW-NOMINATE** | Linear trend: θ_t = θ_0 + d·t (frequentist MLE) | Poole & Rosenthal 1985/1997 |
| **Penalized spline NOMINATE** | Flexible nonlinear trends within NOMINATE | Lewis & Sonnet |
| **Variational EM** | Fast approximate inference (point estimates) | Imai, Lo, Olmsted 2016 |
| **Bridge ideal points** | Cross-institutional comparison via shared observations | Bailey 2007 |
| **Dynamic group-level IRT** | Hierarchical model for aggregate opinion over time | Caughey & Warshaw 2015 |

### Random walk vs. AR(1)

The random walk assumes no mean-reversion — legislators can drift arbitrarily far from center. The AR(1) model (|ρ| < 1) pulls positions back toward a long-run mean, which may be more realistic for legislators constrained by constituency. idealstan implements both; MCMCpack implements only random walk.

Kubinec's guidance: "none of these models is superior to the other. Ideal points do not have any natural time process as they are a latent, unobserved construct." Model selection should be driven by the research question: random walk for unconstrained drift, AR(1) for bounded evolution, GP for flexible nonparametric dynamics.

### DW-NOMINATE vs. Bayesian dynamic IRT

| Feature | DW-NOMINATE | Martin-Quinn / Bayesian |
|---------|-------------|------------------------|
| Dynamics | Linear trend only | Random walk, AR(1), GP |
| Link function | Gaussian kernel | Probit/logit |
| Estimation | MLE (3-step EM) | MCMC or variational |
| Uncertainty | Parametric bootstrap | Posterior credible intervals |
| Dimensionality | Natively 2D | Usually 1D (but see our phase 04b) |
| Flexibility | Rigid linear change | Nonlinear change allowed |

With sufficient data and a well-fitting 1D model, point estimates correlate at r > 0.95. The practical difference is in uncertainty quantification and flexibility of temporal dynamics.

---

## 5. State Legislature Applications

### The gap

**Nobody has published a definitive dynamic ideal point analysis of a state legislature.** The literature is dominated by:

- **U.S. Supreme Court:** Martin-Quinn scores (MCMCpack). 9 justices, long tenures, annual terms.
- **U.S. Congress:** DW-NOMINATE (voteview.com). 535 members, 100+ Congresses.
- **State legislatures:** Shor-McCarty scores — the established standard, but **static** (career-fixed, one score per legislator across their entire tenure). Uses NPAT survey bridging for cross-state comparability.

This gap exists because: (a) state legislature data is messy and hard to scrape (we've solved this), (b) most researchers use Shor-McCarty, and (c) dynamic models are computationally expensive with uncertain payoff at the state level.

### Turnover challenge

Kansas has no formal term limits, but turnover is substantial — ~80% over a decade (Kansas Reflector 2021). Only ~27 of 165 legislators from 2010 were still serving in 2020. This affects dynamic estimation because:

- Most legislators serve 2–4 bienniums (2–4 time points per trajectory)
- The random walk provides meaningful smoothing only when legislators serve 3+ periods
- Short-serving legislators' estimates will be dominated by the within-period data rather than the temporal prior

However, **adjacent-biennium overlap** (~70–80%) is high, which is what the Markov structure needs. The chain of bridges is strong even if individual trajectories are short.

### Time period choice

| Period | Points | Votes per period | Assessment |
|--------|--------|------------------|------------|
| Per biennium | 8 | 500–800 | Natural unit for Kansas; standard in the literature |
| Per session year | 16 | 250–400 | Finer resolution but thinner vote matrices |
| Per month | ~96 | <50 | Too sparse; estimation will fail |

**Biennium is the standard and correct choice.** This is what DW-NOMINATE uses for Congress and what the emIRT `bill.session` parameter expects.

---

## 6. Software Ecosystem

### R packages

| Package | Method | Time-varying | Speed | Uncertainty | Maintained |
|---------|--------|-------------|-------|-------------|------------|
| **MCMCpack** | Gibbs (C++) | Random walk | Slow (hours) | Full posterior | Yes (v1.7-1, 2024) |
| **emIRT** | Variational EM (C++) | Random walk | Fast (minutes) | Point estimates only | Yes (v0.0.14) |
| **idealstan** | NUTS via Stan | RW, AR(1), GP, splines | Moderate | Full posterior | Yes (v0.99.1, 2024) |
| **pscl** | Gibbs | Static only | Moderate | Full posterior | Mature |

**MCMCpack::MCMCdynamicIRT1d** is the canonical Martin-Quinn implementation. Custom C++ Gibbs sampler with Kalman forward-filter/backward-sample. Single-chain only. Handles sparse legislator-time coverage via NA propagation. For our scale (~400 legislators, ~5000 votes, 8 time periods, ~13,600 parameters), expect 6–12 hours.

**emIRT::dynIRT** is the fast alternative. Variational EM produces point estimates comparable to full MCMC (r = 0.93–0.96 vs MCMCpack on Supreme Court data). For our scale, expect minutes. Critical limitation: **variance estimates are unreliable** — "far too small and generally unusable" per the documentation. Requires parametric bootstrap for proper uncertainty.

**idealstan** is the most flexible. Stan/NUTS backend with three time-varying processes, missing data modeling, and Pathfinder for fast approximate inference. The Delaware state legislature example is the closest published precedent to our use case. GitHub-only (not on CRAN), requires cmdstanr.

### Python / PyMC

**No published PyMC dynamic IRT implementation exists.** This is a genuine ecosystem gap. The building blocks are all available:

- `pymc.GaussianRandomWalk` — built-in random walk distribution
- `pm.Bernoulli` / logit link — standard vote likelihood
- nutpie — Rust NUTS sampler already in our pipeline

A dynamic IRT would need to be built from scratch, but the model is straightforward given our existing `build_per_chamber_graph()` infrastructure.

### Stan (language-agnostic)

idealstan's Stan code implements the most complete dynamic IRT in any ecosystem. The Stan programs could theoretically be extracted and run via:

- **cmdstanpy** (Python interface to CmdStan)
- **BridgeStan** — compiles Stan models and exposes log-density/gradient to Python, Rust, or Julia. Can then sample with nutpie: `compiled = nutpie.compile_stan_model(code=stan_code)`, getting Stan's C++ gradients with nutpie's faster adaptation.

### Rust

No native Rust IRT implementations exist. The Rust ecosystem contributes through:

- **nutpie / nuts-rs** — Rust NUTS sampler (already in our pipeline, ~2x faster than Stan's NUTS)
- **BridgeStan** — Rust bindings to compiled Stan models

### JAX / NumPyro

NumPyro on JAX can be 4–11x faster than CPU MCMC on NVIDIA GPUs. However, **JAX Metal on Apple Silicon does not support float64**, which MCMC requires. On Apple Silicon CPU, nutpie's Rust NUTS is faster than NumPyro's JAX NUTS. Not viable for us without cloud GPU access.

### Julia / Turing.jl

Benchmarks show Turing.jl is 5–10x slower than Stan for IRT models due to AD inefficiency. No dynamic IRT package exists. Not recommended.

---

## 7. Performance Comparison

Estimated wall times for our problem (~400 legislators, ~5,000 votes, 8 time periods, 1D dynamic IRT, Apple Silicon M3 Pro):

| Method | Est. Time | Uncertainty | Language |
|--------|-----------|-------------|----------|
| emIRT dynIRT (VI) | **2–5 min** | Point estimates only | R |
| Stan Pathfinder (VI) | **5–15 min** | Approximate | Stan |
| nutpie + PyMC (CPU) | **1–3 hours** | Full posterior | Python |
| nutpie + BridgeStan (CPU) | **45 min – 2 hours** | Full posterior | Stan/Python |
| CmdStan NUTS (CPU) | **2–6 hours** | Full posterior | C++ |
| MCMCpack Gibbs (CPU) | **6–12 hours** | Full posterior | R/C++ |
| NumPyro (NVIDIA GPU) | **10–30 min** | Full posterior | Python/JAX |
| Turing.jl (CPU) | **5–15 hours** | Full posterior | Julia |

### nutpie considerations

nutpie's normalizing flow adaptation provides dramatic speedups on funnel geometry (58x ESS improvement on 100-dim funnels). However, the developer warns it "doesn't work great with models with more than about 1000 parameters." Our dynamic model would have ~13,000+ parameters, exceeding NF capacity. Standard nutpie NUTS (no NF) is still ~2x faster than Stan due to superior mass matrix tuning.

---

## 8. Kansas-Specific Considerations

### Data inventory

| Biennium | Legislature | Chambers | Est. legislators | Est. roll calls |
|----------|------------|----------|------------------|-----------------|
| 2011–12 | 84th | H + S | ~165 | ~400–600 |
| 2013–14 | 85th | H + S | ~165 | ~500–700 |
| 2015–16 | 86th | H + S | ~165 | ~500–700 |
| 2017–18 | 87th | H + S | ~165 | ~500–700 |
| 2019–20 | 88th | H + S | ~165 | ~500–700 |
| 2021–22 | 89th | H + S | ~165 | ~500–700 |
| 2023–24 | 90th | H + S | ~168 | ~600–800 |
| 2025–26 | 91st | H + S | ~168 | ~500+ (in progress) |

Eight bienniums × two chambers = 16 chamber-level models (or 8 pooled models). Total: ~400 unique legislators, ~5,000 roll calls.

### Bridge legislator coverage

Adjacent-biennium overlap is ~70–80%, which far exceeds any published minimum for identification (~5–10% floor, ~20–30% recommended). The chain-of-bridges structure is strong.

Over the full span (84th→91st), only ~5–15% of legislators persist — but the Markov structure doesn't need full-span bridges, only adjacent-period links.

### The 2012 redistricting risk

The 84th→85th transition (2012 redistricting) is the weakest link in the bridge chain. Redistricting causes higher-than-usual turnover and, critically, changes district boundaries. A legislator serving the same district before and after redistricting may face a different constituency, which could drive genuine preference change confounded with scale change. This is a known limitation in the DW-NOMINATE literature as well.

### R supermajority challenge

With ~72% Republicans, the "interesting" variation is intra-party. The dynamic model needs to distinguish genuine ideological drift from agenda-driven scale rotation — harder when the median voter barely moves and the bill pool changes character across bienniums.

### Chamber separation

Kansas House (125 members, 2-year terms) and Senate (40 members, 4-year terms with staggering) should be modeled separately, consistent with our existing per-chamber approach. Senate staggering provides natural multi-period bridges (~50% of senators appear in three consecutive bienniums). Cross-chamber comparison can use IRT linking on the dynamic estimates, same as the current pipeline.

---

## 9. Implementation Options

### Option A: PyMC dynamic IRT (build in current stack)

Extend `build_per_chamber_graph()` to accept a time dimension. Replace `xi ~ Normal(0, 1)` with `xi ~ GaussianRandomWalk(sigma=tau)`. Keep nutpie sampling, RunContext, HTML reports, and all existing infrastructure.

```python
with pm.Model(coords={"time": bienniums, "legislator": names}) as model:
    tau = pm.HalfNormal("tau", sigma=0.5, dims="legislator")
    xi = pm.GaussianRandomWalk(
        "xi", sigma=tau, steps=T - 1,
        init_dist=pm.Normal.dist(0, 1),
        dims=("time", "legislator"),
    )
    alpha = pm.Normal("alpha", mu=0, sigma=5, dims="bill")
    beta = pm.HalfNormal("beta", sigma=2.5, dims="bill")

    eta = beta[bill_idx] * xi[time_idx, leg_idx] - alpha[bill_idx]
    y = pm.Bernoulli("y", logit_p=eta, observed=votes)
```

**Pros:** Full integration with existing pipeline. nutpie sampling. Posterior uncertainty. Familiar toolchain.
**Cons:** No precedent for PyMC dynamic IRT. Convergence unknowns at our scale. ~1–3 hours per run.

### Option B: emIRT for exploration + PyMC for publication

Use `emIRT::dynIRT` via subprocess for fast exploratory analysis (minutes), then build the full Bayesian model in PyMC for final results with proper uncertainty.

```r
# R script called via subprocess
library(emIRT)
result <- dynIRT(.data = data, .starts = starts, .priors = priors,
                 .control = list(threads = 6, verbose = TRUE, thresh = 1e-6))
write.csv(result$means$x, "dynamic_ideal_points.csv")
```

**Pros:** Fast iteration. emIRT is production-tested on Martin-Quinn data. Two-tier workflow (explore fast, publish slow).
**Cons:** Two codebases. emIRT variance estimates are unusable without bootstrap.

### Option C: idealstan via cmdstanr/cmdstanpy

Use the most complete dynamic IRT package available. Supports random walk, AR(1), and GP time processes. Built-in visualization and identification strategies.

**Pros:** Most flexible temporal modeling. Actively maintained. HMC/NUTS. Delaware state legislature precedent.
**Cons:** R/Stan dependency. GitHub-only (beta). Limited community. No nutpie integration (though BridgeStan could bridge this).

### Option D: Stan + BridgeStan + nutpie

Write (or extract from idealstan) a Stan dynamic IRT program. Compile via BridgeStan. Sample with nutpie's Rust NUTS. Gets the best of both worlds: Stan's fast C++ autodiff + nutpie's superior adaptation.

**Pros:** Potentially fastest full-MCMC option. Stan's mature autodiff.
**Cons:** Complex toolchain (Stan + BridgeStan + nutpie). Requires C++ build environment. Less familiar than pure PyMC.

### Option E: Multi-session chained alignment (minimal extension)

Don't build a dynamic model at all. Instead, extend the current per-biennium IRT + Stocking-Lord linking to chain across all 8 bienniums: align 84th→85th, 85th→86th, ..., 90th→91st. This produces a coherent scale without the complexity of state-space estimation.

**Pros:** Minimal implementation. Uses proven infrastructure. No new model to debug.
**Cons:** Not a true dynamic model — no partial pooling across time, no principled uncertainty for change, no conversion/replacement decomposition.

---

## 10. Recommendation

### Primary approach: Option A (PyMC dynamic IRT)

Build the dynamic model in our existing PyMC/nutpie stack. Rationale:

1. **Infrastructure leverage.** We have `build_per_chamber_graph()`, nutpie sampling, RunContext, HTML reports, PCA-informed init, and 1,273 tests. A PyMC dynamic model extends this naturally.
2. **Sampler quality.** nutpie's NUTS with mass matrix adaptation is ~2x faster than Stan and produces better ESS/gradient than Gibbs samplers (MCMCpack). For funnel geometry from the random walk, modern NUTS is substantially better than 2002-era Gibbs.
3. **Posterior uncertainty.** Full Bayesian inference is a core strength of our pipeline. emIRT sacrifices this; MCMCpack provides it but slowly.
4. **Precedent opportunity.** No published PyMC dynamic IRT for a state legislature exists. We would be the first, built on a validated pipeline with 8 bienniums and external validation against Shor-McCarty.

### Secondary: Option B exploration layer

Use `emIRT::dynIRT` as a fast sanity check before committing to multi-hour MCMC runs. A 5-minute emIRT run that shows "legislator X moved 0.3 units rightward across 4 bienniums" gives confidence that the signal exists before estimating the full posterior.

### Non-centered parameterization is essential

The idealstan Stan code and the Stan forums are clear: non-centered parameterization prevents the funnel divergences that plague dynamic IRT with small τ². Instead of:

```
xi[t] ~ Normal(xi[t-1], tau)  # centered
```

Use:

```
xi_raw[t] ~ Normal(0, 1)       # non-centered
xi[t] = xi[t-1] + tau * xi_raw[t]
```

PyMC's `GaussianRandomWalk` may handle this internally — needs verification.

### Per-legislator vs. global τ²

Martin-Quinn and emIRT use per-legislator τ²_j (each legislator can drift at their own rate). With ~400 legislators, estimating 400 evolution variances adds substantial model complexity. Start with a global τ² (or per-party τ²) and add per-legislator variance only if the data demands it.

### Identification strategy

Use the same approach as our static IRT: PCA-informed initialization + positive β constraints on high-discrimination bills. Additionally, constrain two anchor legislators (one known conservative, one known liberal) across their full service window to pin the sign and location.

---

## 11. References

### Core papers

- Martin, A.D. and Quinn, K.M. (2002). "Dynamic Ideal Point Estimation via MCMC for the U.S. Supreme Court, 1953–1999." *Political Analysis*, 10(2), 134–153.
- Clinton, J., Jackman, S., and Rivers, D. (2004). "The Statistical Analysis of Roll Call Data." *APSR*, 98(2), 355–370.
- Bafumi, J., Gelman, A., Park, D., and Kaplan, N. (2005). "Practical Issues in Implementing and Understanding Bayesian Ideal Point Estimation." *Political Analysis*, 13, 171–187.
- Bailey, M.A. (2007). "Comparable Preference Estimates across Time and Institutions for the Court, Congress, and Presidency." *AJPS*, 51(3), 433–448.
- Shor, B. and McCarty, N. (2011). "The Ideological Mapping of American Legislatures." *APSR*, 105(3), 530–551.
- Imai, K., Lo, J., and Olmsted, J. (2016). "Fast Estimation of Ideal Points with Massive Data." *APSR*, 110(4), 631–656.
- Kubinec, R. (2019). "Generalized Ideal Point Models for Time-Varying and Missing-Data Roll Call Votes." OSF Preprint / idealstan package.
- Marshall, B. and Peress, M. (2018). "Dynamic Estimation of Ideal Points for the US Congress." *Public Choice*, 176, 301–320.

### Methodology

- Peress, M. (2018). "Studying Dynamics in Legislator Ideal Points: Scale Matters." *Political Analysis*.
- Caughey, D. and Warshaw, C. (2015). "Dynamic Estimation of Latent Opinion Using a Hierarchical Group-Level IRT Model." *Political Analysis*, 23(2), 197–211.
- Gray, T. (2020). "A Bridge Too Far? Examining Bridging Assumptions." *Legislative Studies Quarterly*.
- Lauderdale, B. and Clark, T. (2014). "Scaling Politically Meaningful Dimensions Using Texts and Votes." *AJPS*, 58(3), 754–771.

### Software

- MCMCpack: https://cran.r-project.org/web/packages/MCMCpack/ (MCMCdynamicIRT1d)
- emIRT: https://cran.r-project.org/web/packages/emIRT/ (dynIRT)
- idealstan: https://github.com/saudiwin/idealstan
- nutpie: https://github.com/pymc-devs/nutpie
- BridgeStan: https://github.com/roualdes/bridgestan
- PyMC GaussianRandomWalk: https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.GaussianRandomWalk.html

### State legislature context

- Kansas Reflector (2021). "No need for term limits: Legislature's turnover rate tops 80% in past decade."
- Shor, B. (2022). "Two Decades of Polarization in American State Legislatures."
- Shor-McCarty individual data: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GZJOT3
