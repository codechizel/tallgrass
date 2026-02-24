# IRT Convergence Investigation: Diagnosing and Fixing Reflection Mode-Splitting

**Date:** 2026-02-23
**Status:** Complete
**Test Subject:** 87th Kansas Legislature (2017-18), Senate chamber

## Abstract

Five of sixteen chamber-sessions in the Kansas Legislature vote dataset exhibit catastrophic convergence failure in the flat (unpooled) 2PL Bayesian IRT model. All five failures share an identical diagnostic signature: R-hat ~1.83, ESS = 3, zero divergences. This paper investigates the root cause, identifies it as a reflection mode-splitting problem caused by unlucky chain initialization, and evaluates four candidate fixes in controlled experiments on the 87th Senate. PCA-informed chain initialization is the only fully effective fix — it eliminates the problem at zero cost while the three alternatives (more chains, more tuning, soft sign constraints) either partially mask or entirely fail to address the underlying issue. We recommend adopting PCA initialization as the default for all IRT runs.

## 1. Background

### 1.1 The IRT Model

We fit a two-parameter logistic (2PL) IRT model to binary roll-call vote data (Yea=1, Nay=0). For legislator $i$ and vote $j$:

$$P(Y_{ij} = 1) = \text{logit}^{-1}(\beta_j \cdot \xi_i - \alpha_j)$$

where $\xi_i$ is the legislator's ideal point (latent ideology), $\beta_j$ is the bill's discrimination (how partisan the vote was), and $\alpha_j$ is the bill's difficulty (threshold for Yea).

### 1.2 The Identification Problem

The 2PL IRT model has a **reflection invariance**: negating all $\xi$ and all $\beta$ simultaneously leaves the likelihood unchanged ($(-\beta)(-\xi) = \beta\xi$). Without constraints, the posterior is bimodal — one mode where conservatives have positive $\xi$ and another where they have negative $\xi$.

We break this symmetry by **anchoring** two legislators: the most conservative (highest PCA PC1 score) is fixed at $\xi = +1$ and the most liberal at $\xi = -1$. This should eliminate the reflected mode, but as we'll show, hard anchoring is sometimes insufficient.

### 1.3 The Failure Pattern

| Session | House | Senate |
|---------|-------|--------|
| 84th (2011-12) | **FAILED** | Passed |
| 85th (2013-14) | Borderline | **FAILED** |
| 86th (2015-16) | **FAILED** | Passed |
| 87th (2017-18) | Passed | **FAILED** |
| 88th (2019-20) | Passed | Passed |
| 89th (2021-22) | **FAILED** | Passed |
| 90th-91st | Passed | Passed |

The failure signature is always identical:
- **R-hat = ~1.83** (far above the 1.01 threshold)
- **ESS = 3** (the absolute minimum for 2 chains x 2000 draws)
- **Zero divergences** (no geometric pathology)
- **Negative PCA-IRT correlation** (r = -0.47 to -0.68, confirming sign inversion)

## 2. Root Cause Analysis

### 2.1 Reflection, Not Divergence

The zero-divergence, bimodal-chain signature rules out geometry-based sampling difficulties (funnels, ridges, etc.). The sampler is working fine — each chain explores its mode efficiently. The problem is that one chain found the "correct" orientation and the other found the reflection.

### 2.2 Why Anchors Aren't Enough

Hard anchors (fixing two $\xi$ values via `pt.set_subtensor`) create a potential energy barrier between the two modes. For a chain initialized in the reflected mode, every free $\xi_i$ wants to be negated, but the two anchored points are pinned. The chain would need to simultaneously flip all free parameters while maintaining the anchor constraints — a coordinated high-dimensional move that NUTS cannot perform.

With sufficient tuning, a chain initialized near the barrier might cross over. But with only 1,000 tuning draws and a 41-legislator model, the probability of crossing depends sensitively on the initial point.

### 2.3 Why It's Session-Dependent

The random seed (42) generates initial parameter values whose distribution depends on the model dimension ($n_{legislators} \times n_{votes}$). Different sessions have different dimensions, so the initial points differ. Some initializations place both chains on the correct side of the barrier; others place one chain on the wrong side. The alternation between House and Senate is coincidental, driven by how seed 42 interacts with each specific problem size.

**Evidence:** The 85th House (R-hat 1.01, ESS 688) is a near-miss — it barely escaped the same fate, consistent with initialization-dependent behavior.

## 3. Test Subject: 87th Senate

### 3.1 Data Characteristics

| Property | Value |
|----------|-------|
| Legislators | 41 |
| Contested votes (post-filter) | 251 |
| Observed cells | 95.2% |
| Yea rate | 0.698 |
| PCA PC1 variance explained | 35.0% |
| Conservative anchor | Mary Pilcher-Cook (PC1 = +20.990) |
| Liberal anchor | Anthony Hensley (PC1 = -10.105) |

### 3.2 Baseline Results (Experiment 0)

| Metric | Value |
|--------|-------|
| R-hat (max) | **1.83** |
| ESS (min) | **3** |
| Divergences | 0 |
| PCA-IRT Pearson r | **-0.470** (inverted) |
| Holdout accuracy | 0.774 (vs 0.699 base rate) |
| Holdout AUC-ROC | 0.818 |
| Sampling time | 55.3s |
| Run label | `2026-02-23` |

The negative PCA correlation confirms the model is sign-flipped. The ideal points are inverted: Democrats appear conservative and Republicans appear liberal.

## 4. Experiments

We test four fixes, each applied independently to the baseline configuration. All experiments use the 87th Senate with identical data and filtering. Only the sampling configuration changes.

### Experiment 1: PCA-Informed Chain Initialization

**Hypothesis:** If both chains start near the PCA solution (correctly oriented), neither will wander into the reflected mode during tuning.

**Method:** Compute approximate initial $\xi$ values from PCA PC1 scores, rescaled to mean-0, variance-1. Pass as `initvals` to `pm.sample()`. Alpha and beta initialized at 0 (default). Chain initialization jitter ensures chains aren't identical.

**Parameters:** Same as baseline (2000 draws, 1000 tune, 2 chains, target_accept=0.9).

**Run label:** `2026-02-23.1`

| Metric | Baseline | Experiment 1 | Change |
|--------|----------|-------------|--------|
| R-hat (max) | 1.83 | **1.004** | Fixed |
| ESS (min) | 3 | **1,263** | 421x better |
| PCA-IRT r | -0.470 | **+0.937** | Sign corrected |
| Holdout accuracy | 0.774 | **0.907** | +0.133 |
| Holdout AUC | 0.818 | **0.958** | +0.140 |
| Sampling time | 55.3s | 48.4s | -12% |

**Result: COMPLETE FIX.** Both chains converge to the correct mode. ESS is excellent (>1,200 for all parameters). PCA correlation is strongly positive. Holdout accuracy jumps 13 percentage points. Sampling is actually slightly faster (the reflected chain was inefficient). This is a zero-cost fix — no additional compute, no model changes.

**Substantive validation:** The converged model correctly identifies Dennis Pyle as a paradox legislator (IRT-PCA rank gap of 21%) — a Republican whose IRT ideal point is more extreme than his PCA score suggests. The baseline model, with its inverted ideal points, detects no paradoxes at all. This confirms that convergence failure doesn't just produce bad diagnostics — it destroys the model's ability to surface real substantive findings.

### Experiment 2: Four Chains

**Hypothesis:** With 4 chains, even if one lands in the reflected mode, the majority (3 of 4) will be correct. R-hat will detect the outlier chain, and the posterior mean will be dominated by the correct-mode chains.

**Method:** Increase `n_chains` from 2 to 4 (with `cores=4`). No other changes.

**Parameters:** 2000 draws, 1000 tune, **4 chains**, target_accept=0.9.

**Run label:** `2026-02-23.2`

| Metric | Baseline | Experiment 2 | Change |
|--------|----------|-------------|--------|
| R-hat (max) | 1.83 | **1.53** | Improved but not fixed |
| ESS (min) | 3 | **7** | Marginal improvement |
| PCA-IRT r | -0.470 | **+0.961** | Sign corrected |
| Holdout accuracy | 0.774 | **0.834** | +0.060 |
| Holdout AUC | 0.818 | **0.899** | +0.081 |
| Sampling time | 55.3s | 60.7s | +10% |

**Result: PARTIAL FIX.** R-hat drops from 1.83 to 1.53 — still failing, but less severely. PCA correlation is positive, suggesting the majority of chains (likely 3 of 4) found the correct mode. The posterior mean is pulled toward the correct orientation by the majority chains, but the outlier chain still contaminates the estimates. ESS remains critically low (7), indicating that averaging across 4 chains where 1 is reflected dilutes the effective sample size. This approach helps but does not solve the fundamental problem.

**The credible intervals reveal the damage.** Consider Dennis Pyle (R), the most conservative senator by PCA. In Experiment 1 (converged): $\xi = +3.54$ [+2.96, +4.13]. In Experiment 2 (4 chains, not converged): $\xi = +1.48$ [-5.13, +4.11]. The point estimate is pulled toward zero by the reflected chain, and the credible interval spans from -5 to +4 — meaningless for inference. The 4-chain approach produces a correctly-signed posterior mean but destroys the precision that makes IRT useful.

### Experiment 3: Extended Tuning

**Hypothesis:** Longer warmup (3000 tuning draws instead of 1000) gives chains more time to find the correct mode and allows the NUTS step-size adaptation to settle.

**Method:** Increase `n_tune` from 1000 to 3000. No other changes.

**Parameters:** 2000 draws, **3000 tune**, 2 chains, target_accept=0.9.

**Run label:** `2026-02-23.3`

| Metric | Baseline | Experiment 3 | Change |
|--------|----------|-------------|--------|
| R-hat (max) | 1.83 | **1.83** | No change |
| ESS (min) | 3 | **3** | No change |
| PCA-IRT r | -0.470 | **-0.465** | No change |
| Holdout accuracy | 0.774 | **0.774** | No change |
| Holdout AUC | 0.818 | **0.818** | No change |
| Sampling time | 55.3s | 80.1s | +45% (more tuning) |

**Result: COMPLETE FAILURE.** Tripling the tuning draws from 1,000 to 3,000 has zero effect on convergence. The chain that initialized in the reflected mode does not cross over even with 3x more warmup. This definitively rules out "insufficient tuning" as the cause — the energy barrier between modes is too high for the NUTS sampler to cross, regardless of tuning duration. The chain is locally well-adapted to its mode (zero divergences), so more adaptation time just makes it more efficient at exploring the wrong mode.

### Experiment 4: Soft Sign Constraint (pm.Potential)

**Hypothesis:** Adding a weak penalty that favors $\text{mean}(\xi_R) > \text{mean}(\xi_D)$ creates a global energy gradient toward the correct orientation without rigidly constraining any individual legislator.

**Method:** After defining $\xi$, compute party-mean ideal points using the known party assignments. Add `pm.Potential("sign_constraint", -penalty * (mean_D - mean_R))` which makes the reflected mode slightly higher energy. The penalty should be weak enough that it doesn't distort the posterior (we use $\lambda = 1.0$ as a light nudge).

**Parameters:** Same as baseline (2000 draws, 1000 tune, 2 chains, target_accept=0.9) plus the sign potential.

**Run label:** `2026-02-23.5` (note: `.4` was an aborted run due to a column-name bug; `.5` is the successful Experiment 4 run)

| Metric | Baseline | Experiment 4 | Change |
|--------|----------|-------------|--------|
| R-hat (max) | 1.83 | **1.83** | No change |
| ESS (min) | 3 | **3** | No change |
| PCA-IRT r | -0.470 | **-0.458** | No change |
| Holdout accuracy | 0.774 | **0.774** | No change |
| Holdout AUC | 0.818 | **0.819** | No change |
| Sampling time | 55.3s | 50.5s | -9% (noise) |

**Result: COMPLETE FAILURE.** The soft sign constraint with $\lambda = 1.0$ has zero effect on convergence. The penalty term adds a linear tilt to the log-posterior, but it is dwarfed by the likelihood — with 9,794 observed vote cells, the data overwhelm the gentle nudge. The reflected mode is a deep, well-separated basin; a potential of magnitude ~1.0 (the party-mean gap) cannot overcome the energy barrier when the likelihood strongly anchors the chain to its local mode. Increasing $\lambda$ would eventually work, but at the cost of distorting the posterior — defeating the purpose of a "soft" constraint. This approach fails for the same fundamental reason as extended tuning: the chain is trapped, and no amount of gradient information within the mode can help it escape.

**Comparison with Experiment 1:** The sign constraint attempts to solve the problem after the chain is already trapped. PCA initialization solves it before the chain ever starts. This is why initialization-based fixes are categorically more effective for mode-splitting than penalty-based fixes.

## 5. Results Comparison

| Metric | Baseline | Exp 1: PCA Init | Exp 2: 4 Chains | Exp 3: More Tune | Exp 4: Sign Potential |
|--------|----------|-----------------|-----------------|-------------------|-----------------------|
| R-hat (max) | 1.83 | **1.004** | 1.53 | 1.83 | 1.83 |
| ESS (min) | 3 | **1,263** | 7 | 3 | 3 |
| PCA-IRT r | -0.470 | **+0.937** | +0.961 | -0.465 | -0.458 |
| Holdout acc | 0.774 | **0.907** | 0.834 | 0.774 | 0.774 |
| AUC-ROC | 0.818 | **0.958** | 0.899 | 0.818 | 0.819 |
| Time (s) | 55.3 | **48.4** | 60.7 | 80.1 | 50.5 |
| Converged? | No | **Yes** | No | No | No |

The results fall into three tiers:

1. **Complete fix (Exp 1):** PCA initialization eliminates the problem entirely — perfect convergence, excellent ESS, strong holdout performance, and no additional cost.
2. **Partial mitigation (Exp 2):** Four chains shift the posterior mean toward the correct mode via majority vote, but convergence diagnostics still fail. The approach masks the problem rather than solving it.
3. **No effect (Exp 3, 4):** Neither extended tuning nor a soft sign constraint can rescue a chain already trapped in the reflected mode.

## 6. Analysis

### 6.1 Why PCA Initialization Works

PCA initialization succeeds because it addresses the problem at its source: chain initialization. The reflected mode exists because the 2PL likelihood is symmetric under sign inversion. Anchors create an energy barrier between the two modes, but they cannot prevent a chain from initializing on the wrong side. PCA provides an approximate but correctly-oriented starting point that places both chains firmly in the correct mode's basin of attraction.

The key insight is that PCA PC1 scores and IRT ideal points are monotonically related for well-behaved roll-call data (Pearson r = 0.937 in the converged model). Rescaling PC1 to mean-0, variance-1 gives a crude but sufficient approximation. The MCMC sampler then refines these starting values toward the true posterior — but crucially, it refines within the correct mode.

### 6.2 Why Extended Tuning and Sign Constraints Fail

Both Experiments 3 and 4 attempt to fix a chain that is already trapped in the reflected mode. This is fundamentally the wrong approach:

- **Extended tuning** gives the sampler more time to adapt its step size and mass matrix, but adaptation is local — it makes the chain more efficient at exploring its current mode, not better at jumping to a distant mode. NUTS is designed for exploration within a single basin, not for mode-hopping.

- **The soft sign constraint** adds a linear energy tilt that favors $\text{mean}(\xi_R) > \text{mean}(\xi_D)$. But with $\lambda = 1.0$, this penalty is negligible compared to the likelihood contribution of 9,794 vote observations. The reflected mode remains a deep basin. Increasing $\lambda$ would eventually break the symmetry, but at the cost of a strongly informative prior on the party structure — which is precisely what we want to avoid in an IRT model designed to discover ideological structure from voting behavior alone.

### 6.3 Why Four Chains Partially Helps

Experiment 2 is instructive because it demonstrates the difference between fixing convergence and masking it. With seed 42 and 4 chains, 3 chains found the correct mode and 1 found the reflection. The posterior mean, averaged across all 4 chains, is pulled toward the correct orientation (PCA-IRT r = +0.961). Holdout accuracy improves. But R-hat remains 1.53 and ESS is 7 — the diagnostics correctly detect that one chain disagrees with the others.

This approach is dangerous in production: it can produce plausible-looking point estimates while hiding a fundamental convergence failure. If the random seed happened to send 2 of 4 chains to the reflected mode, the posterior mean would be contaminated. Convergence diagnostics would still flag the problem, but the point estimates would be misleading. Four chains is a useful diagnostic tool, not a solution.

### 6.4 Literature Context

Our experimental results are consistent with two decades of literature on identification in Bayesian ideal point models. The key references and how they relate to our findings:

**Jackman (2001, 2009) and `pscl::ideal()`.** The reference implementation of Bayesian IRT for legislative voting — used in essentially every political science paper on ideal points — has used eigendecomposition (mathematically equivalent to PCA) as its **default** starting values since 2001. The `startvals="eigen"` option "generates extremely good start values for low-dimensional models fit to recent U.S. congresses, where high rates of party line voting result in excellent fits from low dimensional models." Our Experiment 1 is a reimplementation of this standard practice in the PyMC ecosystem.

**Bafumi, Gelman, Park, & Kaplan (2005).** Rather than anchoring individual legislators, Gelman's group proposed including party affiliation as a regression predictor on ideal points, constraining the coefficient to be positive. This forces the scale orientation structurally rather than through point constraints. Their approach is conceptually what our hierarchical IRT model does with its ordering constraint ($\mu_D < \mu_R$) — and it explains why the hierarchical model converges where the flat model fails. The hierarchical model has structural identification built into the prior; the flat model relies on anchor constraints that create an energy barrier but don't always prevent chains from initializing on the wrong side.

**Betancourt (2017).** His Stan case study warns that **initialization cannot solve label switching in the general case** — for exchangeable mixture components, it merely determines which mode a non-mixing chain explores. However, he acknowledges that ordering constraints work well when components have known distinct roles. Legislative IRT is exactly this favorable case: the two modes are not equally valid hypotheses to average over. One is "correct" (Republicans positive) and the other is its mirror image. This is why initialization works for us but would not work for, say, a Gaussian mixture model with exchangeable components.

**Erosheva & Curtis (2017).** In general Bayesian factor analysis, they found PCA initialization **failed** to prevent reflection mode-splitting — chains still found the wrong polarity. Their failure mode occurs when PCA and the constraint structure point in different directions. This cannot happen in our setup because we select anchors *from* PCA extremes — the initialization and the constraints are aligned by construction.

**Poole & Rosenthal (1985, 2005).** Even in the frequentist W-NOMINATE tradition, eigendecomposition of the agreement matrix has been central to starting value generation from the beginning. Poole and Rosenthal spent "over two years" finding satisfactory starting values in the 1980s; the solution they converged on was eigendecomposition-based. The starting value problem in ideal point estimation is old, and PCA-based solutions are the established answer across both Bayesian and frequentist traditions.

### 6.5 The Cost-Benefit Landscape

| Fix | Effectiveness | Cost | Risk | Verdict |
|-----|--------------|------|------|---------|
| PCA init | Complete | None (faster) | Negligible | **Adopt** |
| 4 chains | Partial | +10% time | Masks failures | Reject |
| More tuning | None | +45% time | Wasted compute | Reject |
| Sign constraint | None | None | Could distort posterior if $\lambda$ increased | Reject |

PCA initialization is the only fix that is simultaneously effective, free, and safe. It requires no additional compute, no model changes, and no prior assumptions about party structure (the PCA scores are computed from the same vote data the model will see). The only theoretical risk is that PCA PC1 could be oriented opposite to the conventional left-right axis, but our existing sign-correction step (used since the PCA phase) eliminates this.

**One caveat from the literature:** PCA initialization depends on PC1 being a reasonable approximation of the ideological dimension. This holds strongly for Kansas (PC1 explains 30-50% of variance, PCA-IRT r > 0.93 in converged models) and for U.S. legislatures generally. It could fail in a legislature with weak party structure or cross-cutting cleavages where PC1 captures something other than ideology. For our purposes this is not a concern, but it would be worth noting if the codebase were extended to non-U.S. legislatures.

## 7. Recommendation

### 7.1 Production Fix

**Adopt PCA-informed chain initialization as the default behavior for all IRT runs.**

Implementation:
1. After anchor selection, compute PCA PC1 scores for all legislators
2. Rescale to mean-0, variance-1 (exclude anchored legislators)
3. Pass as `initvals={"xi_free": rescaled_scores}` to `pm.sample()`
4. Alpha and beta remain default-initialized (0)

This fix is already implemented behind the `--pca-init` CLI flag in `analysis/irt.py`. The recommendation is to make this the default (always-on) behavior, retaining `--no-pca-init` as an opt-out for experimentation.

### 7.2 Validation Plan

After adopting PCA init as default:
1. Re-run all 5 previously-failing chamber-sessions (84th House, 85th Senate, 86th House, 87th Senate, 89th House)
2. Verify R-hat < 1.01 and ESS > 400 for all
3. Verify PCA-IRT Pearson r > 0.90 for all
4. Verify holdout AUC > 0.90 for all
5. Compare ideal point rankings with PCA rankings to confirm substantive reasonableness

### 7.3 What Was Not Tested

This investigation focused on the flat (unpooled) 2PL IRT model. The hierarchical IRT model may have different convergence properties due to partial pooling. However, the same reflection invariance exists in the hierarchical model, so PCA initialization should be equally beneficial there. The hierarchical model already has structural identification via the ordering constraint (Bafumi et al. 2005; Betancourt 2017), which may make PCA initialization redundant for that model. Testing is deferred to a separate investigation.

### 7.4 Broader Lesson

The IRT convergence failure pattern — R-hat ~1.83, ESS 3, zero divergences — is a textbook case of label switching in Bayesian models with discrete symmetries. The fix is equally textbook: break the symmetry at initialization, not through constraints. This principle applies to any latent-variable model with a sign or permutation invariance (factor analysis, mixture models, topic models). The specific technique (PCA-informed initialization) is well-established in the political science IRT literature — Jackman's `pscl::ideal()` has used eigendecomposition as its default `startvals` since 2001, and the broader label switching literature (Stephens 2000, Betancourt 2017) supports initialization-based symmetry breaking for cases where the modes have a known substantive ordering.

For the full literature review supporting this approach, see [lit-review-irt-initialization.md](lit-review-irt-initialization.md).

## Appendix A: Run Labels

| Run Label | Experiment | Description | Converged? |
|-----------|-----------|-------------|------------|
| `2026-02-23` | 0 (Baseline) | Original run: 2 chains, 1000 tune, no init, no constraint | No |
| `2026-02-23.1` | 1 | PCA-informed chain initialization | **Yes** |
| `2026-02-23.2` | 2 | 4 chains (instead of 2) | No (R-hat 1.53) |
| `2026-02-23.3` | 3 | 3000 tuning draws (instead of 1000) | No |
| `2026-02-23.4` | — | Aborted run (column-name bug in sign constraint code) | — |
| `2026-02-23.5` | 4 | Soft sign constraint via pm.Potential ($\lambda = 1.0$) | No |

## Appendix B: Code Changes Per Experiment

Each experiment modifies `build_and_sample()` in `analysis/irt.py`. Changes are reverted between experiments so each test is independent. The production code is not permanently modified until a recommendation is made.

### B.1 Experiment 1: PCA Initialization

Added `xi_initvals` parameter to `build_and_sample()`. In `main()`, PCA PC1 scores are aligned to the model's legislator ordering, standardized to mean-0/variance-1, then the free-parameter subset (excluding anchors) is extracted:

```python
# Align PCA scores to model's legislator ordering
pc1_vals = pca_scores.filter(...).sort(...)["PC1"].to_numpy()
# Standardize to match the Normal(0,1) prior on xi
pc1_std = (pc1_vals - pc1_vals.mean()) / (pc1_vals.std() + 1e-8)
# Extract only free parameters (exclude anchored legislators)
free_pos = [i for i in range(n_legislators) if i not in anchor_set]
xi_init = pc1_std[free_pos].astype(np.float64)
```

Passed to `pm.sample()` via `sample_kwargs`:
```python
sample_kwargs = {}
if xi_initvals is not None:
    sample_kwargs["initvals"] = {"xi_free": xi_initvals}
idata = pm.sample(..., **sample_kwargs)
```

### B.2 Experiment 2: Four Chains

Changed `n_chains=2` to `n_chains=4` and `cores=4` in the CLI invocation. No code changes to `build_and_sample()`.

### B.3 Experiment 3: Extended Tuning

Changed `n_tune=1000` to `n_tune=3000` in the CLI invocation. No code changes to `build_and_sample()`.

### B.4 Experiment 4: Soft Sign Constraint

Added `sign_constraint_parties` parameter to `build_and_sample()`. Inside the model block:

```python
if sign_constraint_parties is not None:
    r_idx, d_idx = sign_constraint_parties
    mean_r = xi[r_idx].mean()
    mean_d = xi[d_idx].mean()
    pm.Potential("sign_constraint", -(mean_d - mean_r))
```

Party indices computed from the legislators DataFrame, mapping Republican → `r_idx` and Democrat → `d_idx` arrays. The penalty term equals $-(\\text{mean}_D - \\text{mean}_R)$, which is positive (lower energy) when Republicans are to the right of Democrats.

## Appendix C: Reproducing These Results

All experiments can be reproduced from the 87th Legislature (2017-18) data:

```bash
# Baseline (Experiment 0) — PCA init is now on by default, so use --no-pca-init to reproduce
uv run python -m analysis.irt --session 2017-18 --no-pca-init

# Experiment 1: PCA initialization (now the default — just run without flags)
uv run python -m analysis.irt --session 2017-18

# Experiment 2: Four chains (with PCA init disabled to isolate the variable)
uv run python -m analysis.irt --session 2017-18 --no-pca-init --n-chains 4

# Experiment 3: Extended tuning (with PCA init disabled)
uv run python -m analysis.irt --session 2017-18 --no-pca-init --n-tune 3000

# Experiment 4: Sign constraint (with PCA init disabled)
uv run python -m analysis.irt --session 2017-18 --no-pca-init --sign-constraint
```

**Note:** After this investigation, PCA initialization was adopted as the default (always-on) behavior. The original experiments were run with the old `--pca-init` opt-in flag. To reproduce the baseline and Experiments 2-4 exactly, use `--no-pca-init` to disable the now-default PCA initialization.

Results are saved in `results/kansas/87th_2017-2018/irt/` with the run labels documented in Appendix A. The same-day run preservation system (`.1`, `.2`, etc.) ensures each experiment's output is retained.
