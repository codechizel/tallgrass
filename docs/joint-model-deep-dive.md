# Joint Hierarchical IRT: Why Cross-Chamber Scaling Fails and How to Fix It

*February 2026*

## The Problem in Plain English

Tallgrass estimates where each Kansas legislator falls on a liberal-to-conservative spectrum using Bayesian Item Response Theory (IRT). The per-chamber models — one for the House, one for the Senate — work well: they produce reliable estimates that correlate strongly with the Shor-McCarty academic benchmark (House r = 0.97, Senate r = 0.95). But the moment we try to put both chambers on a **common scale** using a single "joint" model, everything falls apart.

The joint model fails convergence diagnostics in **all eight bienniums** (2011-2026). The R-hat statistic — which should be below 1.01 for trustworthy results — ranges from 1.007 to 2.42. The effective sample size (ESS) collapses to 5-7 in seven of eight sessions, meaning the model's 8,000 posterior draws contain roughly the same information as 7 independent samples. The error bars on individual legislators' ideal points explode to absurd widths: a legislator whose per-chamber model places them at 0.5 +/- 0.3 might show up in the joint model at 0.5 +/- 3.0.

This matters because the joint model is the only way to directly compare legislators across chambers — to answer questions like "Is the Senate more conservative than the House?" or "Is Senator X further right than Representative Y?" Without a working joint model, we rely on the flat IRT's post-hoc test equating, which makes strong assumptions about the equivalence of shared bills.

This article examines the joint model's architecture, diagnoses the root causes of failure, surveys the political science and psychometrics literature for solutions, and proposes a prioritized improvement plan.

## The Joint Model's Architecture

The joint model (`build_joint_graph()` in `analysis/10_hierarchical/hierarchical.py`, line 562) is a three-level Bayesian hierarchy:

```
Level 3 (global):    mu_global ~ Normal(0, 2)
                          |
Level 2 (chamber):   mu_chamber = mu_global + sigma_chamber * chamber_offset
                          |
Level 1 (group):     mu_group = mu_chamber[chamber] + sigma_party * group_offset
                          |
Level 0 (legislator): xi = mu_group[group] + sigma_within[group] * xi_offset
```

There are four groups (House Democrats, House Republicans, Senate Democrats, Senate Republicans), each with its own mean and within-group variance. Identification is achieved by sorting each chamber's party offsets so that Democrats are to the left of Republicans (`pt.sort(group_offset_raw[:2])` for House, `pt.sort(group_offset_raw[2:])` for Senate).

The vote model is a standard two-parameter logistic (2PL) IRT:

```
P(Yea | xi_i, alpha_j, beta_j) = logistic(beta_j * xi_i - alpha_j)
```

where `alpha_j` is the bill's difficulty (how likely a Yea vote overall) and `beta_j` is the bill's discrimination (how strongly it separates liberals from conservatives).

### Parameter Count

For the 91st Legislature (2025-26), the joint model has approximately **1,042 free parameters**:

| Component | Count | Notes |
|-----------|-------|-------|
| `mu_global` | 1 | Global mean |
| `sigma_chamber` | 1 | Chamber-level spread |
| `chamber_offset` | 2 | House, Senate |
| `sigma_party` | 1 | Party-level spread |
| `group_offset_raw` | 4 | HD, HR, SD, SR (2 sorted pairs) |
| `sigma_within` | 4 | Per-group within-party spread |
| `xi_offset` | 172 | One per legislator (130 House + 42 Senate) |
| `alpha` | 420 | Bill difficulty (71 shared + 179 House-only + 170 Senate-only) |
| `beta` | 420 | Bill discrimination |
| **Total** | **~1,025** | |

This is a large model by MCMC standards. The NUTS sampler's mixing time scales as O(d^{1/4}) with dimension d, meaning a 1,000-parameter model mixes roughly 5.6x slower than a 100-parameter model *even in the best case* — and IRT posteriors are far from the best case.

### The Bill-Matching Bridge

The joint model's primary innovation (ADR-0043, implemented February 2026) is **concurrent calibration**: bills that pass through both chambers share a single set of alpha/beta parameters. When the House votes on HB 2001 and the Senate later votes on the same bill, those observations constrain the same `beta_j`. This creates a mathematical bridge — the shared bill's discrimination parameter forces the two chambers' ideal point scales to be comparable.

In the 91st Legislature, 174 bill numbers appear in both chambers' roll call records. After the EDA near-unanimous filter, 71 survive as shared bridge items. These 71 bills constitute 17% of the total vote set (71 / 420), meaning 83% of the bill parameters have no cross-chamber bridging function. They provide within-chamber information only.

Before the bill-matching fix, the joint model used `vote_id` deduplication, which treated every roll call as unique — meaning **zero shared items** across chambers. The two chambers' likelihoods were completely separable, and the hierarchy alone had to link their scales. The fix improved runtime from 93 minutes to 32 minutes but did not achieve convergence.

### Sign Identification: Necessary but Not Sufficient

The model constrains `mu_group` ordering within each chamber (D < R), which breaks the global reflection symmetry at the *group level*. But the individual bill discrimination parameters (`beta`) can each independently flip sign: if `beta_j` flips to `-beta_j` and the corresponding `xi` interactions adjust, the likelihood is unchanged. With 420 bills, this creates 420 independent axes along which partial reflection can occur.

The production code includes a post-hoc sign correction (`fix_joint_sign_convention()`, line 851) that compares joint ideal points to per-chamber estimates and negates the posterior if they're anti-correlated. But this is a band-aid: if the sampler got stuck exploring reflection modes, the posterior is bimodal and averaging across modes is not meaningful.

### A Critical Gap: No PCA Initialization

The per-chamber models use PCA-informed initialization: the first principal component of the vote matrix provides starting values for `xi_offset`, helping the sampler find the correct mode quickly. This was implemented in ADR-0023 and proved critical for convergence.

The joint model **does not use PCA initialization** in production. The `build_joint_model()` function accepts an `xi_offset_initvals` parameter (line 775), but the production call at line 1753 does not pass it. All chains start from the default (zeros + jitter), meaning the joint model must find the correct mode from scratch — a much harder problem with 1,042 parameters and multimodal posterior.

## What We've Already Tried

### Successful Interventions (Per-Chamber)

| Intervention | Effect | Status |
|--------------|--------|--------|
| PCA-informed init (ADR-0023) | Fixed per-chamber House R-hat (1.0102 → 1.0026) | Production |
| nutpie Rust NUTS (ADR-0051) | Fixed remaining per-chamber failures (12/16 → 12/16 PASS, 4 WARN) | Production |
| Adaptive sigma prior (ADR-0043) | Stabilized small-group Senate Democrats | Production |
| 4-chain (ADR-0044) | Proved jitter mode-splitting; `adapt_diag` > `jitter+adapt_diag` | Diagnosed |

### Experiments on Joint Model

| Intervention | Effect | Outcome |
|--------------|--------|---------|
| Bill-matching (ADR-0043) | Runtime 93 → 32 min, 71 shared bills bridge chambers | Partial improvement |
| Positive beta (LogNormal) | R-hat 1.5 → 1.024, ESS 7 → 243, divergences 0 → 25 | Still fails |
| nutpie Rust NUTS (ADR-0053) | Faster sampling but same convergence profile | No improvement |

### What Has NOT Been Tried

1. **PCA initialization for the joint model** — The infrastructure exists but is not connected in production
2. **Tighter alpha prior** — Currently Normal(0, 5); reducing to Normal(0, 2) would help with identifiability
3. **Sparsity priors on beta** — Horseshoe or half-Laplace to shrink uninformative bills
4. **Normalizing flow adaptation** — nutpie's experimental `flow_model` parameter
5. **Pathfinder initialization** — L-BFGS variational inference for MCMC starting points (available in pymc-extras)
6. **Two-stage linking** — Abandon concurrent calibration; use per-chamber models + Stocking-Lord test equating
7. **Polya-Gamma augmentation** — Data augmentation that turns the logistic likelihood into a conjugate Gaussian
8. **More draws** — Simply running longer (4,000+ draws) with the current model

## Literature Survey: How Does the Field Handle This?

The problem of placing legislators from different chambers on a common ideological scale has been studied extensively in political science and psychometrics. The approaches fall into two broad categories.

### Category 1: Concurrent Calibration (One Big Model)

This is what our joint model attempts: estimate all parameters in a single model where shared items (bills voted on by both chambers) provide the cross-chamber link.

**DW-NOMINATE Common Space** (Poole and Rosenthal 1991, updated through present). The original and still most widely used method. Instead of shared *bills*, it uses **bridge legislators** — members who serve in both chambers across their careers — to anchor the common scale. Kansas has very few senators who previously served in the House within a single biennium, making this approach impractical for within-biennium analysis. DW-NOMINATE also uses a deterministic optimization (not MCMC), sidestepping the convergence problem entirely.

**Shor and McCarty (2011)**. The gold standard for state-level cross-chamber scaling. They use *survey bridging*: Project Vote Smart interest group scores and NPAT questionnaire responses serve as common items across chambers. This is exogenous data — not roll call votes — so the bridging is independent of the legislative process. Their dataset covers all 50 states from 1993-2020 and is externally validated against congressional scores. Our flat IRT correlates strongly with Shor-McCarty (House r = 0.975, Senate r = 0.950 pooled), confirming our per-chamber estimates are sound.

**Bailey (2007)**. Uses "bridge observations" — presidential position-taking on Supreme Court cases and congressional votes — to place presidents, Congress members, and Supreme Court justices on a common scale. Clever bridging design, but not applicable to state legislatures.

**Hierarchical Multi-Group IRT** (Fox 2010, De Boeck et al. 2011). The theoretical basis for our model: treat groups (chambers, parties) as levels in a hierarchy, with partial pooling. Fox (2010) demonstrates this for educational testing where multiple schools take partially overlapping exams. The key difference from our setting: educational tests typically have *many* shared items (50-100% overlap) and relatively few dimensions of variation. Our 17% shared-bill rate may be too low for effective concurrent calibration.

### Category 2: Separate-Then-Link (Test Equating)

Estimate each chamber's ideal points independently (which we know works well), then use a post-hoc transformation to align the scales.

**Test Equating Methods** (Kolen and Brennan 2014). The psychometrics field has developed sophisticated methods for placing scores from different test forms on a common scale. The most relevant are:

- **Mean-Sigma / Mean-Mean**: Simple affine transformation matching the first two moments of shared items' parameters. This is essentially what our flat IRT's test equating does. Fast and robust but assumes the shared items function identically in both chambers.
- **Stocking-Lord**: Minimizes the squared difference between test characteristic curves (the probability-of-correct-response function) evaluated at a grid of ability levels. More robust than mean-sigma because it weights items by their discrimination. Widely used in educational testing for common-item equating.
- **Haebara**: Similar to Stocking-Lord but minimizes differences at the item level rather than the test level. Can be more robust when a few items function differently across groups.

These methods are well-studied, computationally cheap (they're just optimization problems on 2 parameters — slope and intercept), and don't require MCMC. The trade-off is that they assume the shared items measure the same construct in both chambers — an assumption that may not hold if, say, a bill is a routine procedural vote in one chamber but a contentious party-line vote in the other.

**IRT Linking with Differential Item Functioning (DIF)** (Holland and Wainer 1993). An extension that first tests whether each shared item functions equivalently across groups (using Wald, LR, or Lord's chi-squared tests), drops items that show DIF, then equates using the remaining "invariant" items. This addresses the concern about item equivalence head-on. Our EDA filter (removing near-unanimous votes) provides some protection against DIF, but a formal DIF analysis on the shared bills would be more rigorous.

### Category 3: Alternative Estimation Strategies

**emIRT** (Imai, Lo, and Olmsted 2016). Variational EM algorithm for IRT. Trades MCMC for a fast deterministic approximation. Implemented in R only (`emIRT` package). Reported to scale to congressional datasets (500+ legislators, 1,000+ votes). Could handle our joint model's dimensionality but would require an R dependency we've explicitly avoided.

**Pathfinder** (Zhang et al. 2022). L-BFGS-based variational inference that finds multiple approximate posterior modes. Available in `pymc-extras` as `pmx.fit(method="pathfinder")`. Not a full posterior estimate — it provides starting points for MCMC. Two uses: (1) initialize NUTS chains near the correct mode; (2) serve as a standalone approximate posterior when MCMC is intractable.

**Polya-Gamma Augmentation** (Polson, Scott, and Windle 2013). A data augmentation scheme that converts the logistic likelihood into a conditionally Gaussian problem, enabling Gibbs sampling. This replaces the geometric challenge of NUTS on a logistic likelihood with the algebraic efficiency of conjugate Gaussian updates. Published implementations exist (BayesLogit R package, pybayeslogit Python), but integrating with PyMC's model-building framework would require substantial custom code.

**Normalizing Flow Adaptation** (nutpie experimental). nutpie's `flow_model` parameter uses a neural network to learn a nonlinear transformation of the posterior that makes it approximately Gaussian. This directly addresses the geometric pathologies (reflection ridges, funnels) that cause slow mixing. The feature was explicitly designed for models with ~1,000 parameters — almost exactly our joint model's size. However, the feature is experimental and may not be production-ready.

## Root Cause Diagnosis: Two Reinforcing Problems

The joint model's convergence failure has two distinct root causes that compound each other:

### Problem 1: Reflection Mode Multimodality

With `beta ~ Normal(0, 1)`, each of the 420 bill discrimination parameters can be positive or negative. The likelihood is invariant under `(beta_j, xi) -> (-beta_j, -xi)` for each bill independently. The `pt.sort` constraint on group offsets breaks the *global* reflection (all betas flip simultaneously), but it cannot prevent *partial* reflections where a subset of betas flip.

This creates a posterior landscape with a combinatorial number of local modes — not 2^420 distinct modes (most are ruled out by the data), but enough ridges and saddle points to dramatically slow NUTS exploration. The sampler doesn't diverge (the geometry is smooth), but it takes many steps to traverse between regions, inflating autocorrelation and depressing ESS.

**Evidence**: The positive-beta experiment (LogNormal prior) reduced joint R-hat from 1.53 to 1.024 and ESS from 7 to 243 — a 35x improvement in ESS from eliminating the reflection ambiguity alone. This confirms that reflection modes are the dominant cause.

### Problem 2: High-Dimensional Slow Mixing

Even after removing reflection modes, the joint model has ~600 free parameters (1,042 minus the 420 betas that could be removed by marginalizing, though we don't actually marginalize). NUTS mixing time scales as O(d^{1/4}) with dimension, meaning the 91st joint model (~1,042 params) mixes ~3.2x slower than the per-chamber House (~460 params).

But the scaling is worse than theoretical because:

1. **Block-diagonal structure**: ~83% of bills are chamber-specific. Their beta parameters have no gradient information from the other chamber, creating large blocks of the posterior that are only weakly coupled to the cross-chamber structure. The mass matrix adaptation must capture this block structure.

2. **Three-level hierarchy**: The per-chamber model has two levels (party → legislator). The joint model adds a third (global → chamber → party → legislator). Each additional level introduces a funnel geometry where the group-level variance and the individual-level offsets are tightly coupled.

3. **Structural missingness**: The vote matrix is ~50% structurally missing (House members don't vote on Senate-only bills and vice versa). This means the effective data-per-parameter ratio is lower than the raw observation count suggests.

**Evidence**: Even with LogNormal beta (removing Problem 1), the joint model's ESS was 243 — below the 400 threshold. The step size collapsed to 0.007-0.009 (vs. 0.06 for per-chamber House), and draw speed was ~1/second (vs. ~9/second per-chamber). The geometry is navigable but slow.

## Improvement Plan

Based on the literature survey and our experimental evidence, here is a prioritized plan for achieving joint model convergence. The priorities reflect a balance of expected impact, implementation effort, and risk.

### Priority 1: Enable PCA Initialization for Joint Model

**Expected impact:** High. PCA init resolved per-chamber convergence failures (ADR-0023) and is already the proven solution for IRT mode-finding.

**What to do:** Compute PCA PC1 on the combined vote matrix (House + Senate, with shared bills unified), then pass the resulting values as `xi_offset_initvals` to `build_joint_model()`. The infrastructure already exists — the parameter is accepted and plumbed through to nutpie's `initial_points`.

**Effort:** Low. The combined vote matrix is already computed for flat IRT (`build_joint_vote_matrix()` in `irt.py`). Extract PC1, map to legislator slugs, pass to the joint model call at line 1753.

**Risk:** Low. PCA init is a proven technique in our pipeline. Worst case: it helps but is not sufficient alone.

### Priority 2: Constrain Beta > 0 (LogNormal Prior)

**Expected impact:** High. Already demonstrated: LogNormal(0, 0.5) dropped joint R-hat from 1.53 to 1.024 and improved ESS from 7 to 243.

**What to do:** Change `PRODUCTION_BETA` from `Normal(0, 1)` to `LogNormal(0, 0.5)` — or, better, use LogNormal only for the joint model while keeping Normal for per-chamber models (where it works fine). This could mean adding a `JOINT_BETA` constant.

**Effort:** Low. The `BetaPriorSpec` infrastructure already supports LogNormal. Just pass it to the joint model call.

**Risk:** Moderate. LogNormal constrains all bills to discriminate in the "natural" direction (voting Yea = more conservative on conservative bills). This is substantively reasonable for final-action votes (which dominate our filtered dataset) but could be wrong for procedural votes where Yea means "table the bill" or "recommit to committee." The per-chamber experiment showed slight ICC decrease (0.90 → 0.87 House) and moderate Senate ranking shifts (r ≈ 0.92).

### Priority 3: Combine Priorities 1 + 2

**Expected impact:** Very high. PCA init and LogNormal beta attack the two root causes independently (slow mode-finding and reflection multimodality). Combined, they may be sufficient.

**Effort:** Low — it's just enabling both changes at once.

### Priority 4: Increase Draws

**Expected impact:** Moderate. If R-hat is acceptable (as with LogNormal beta, R-hat = 1.024), more draws should push ESS past 400. Doubling draws from 2,000 to 4,000 should roughly double ESS (to ~486 from the LogNormal experiment's 243).

**What to do:** Pass `--n-samples 4000` (or 6000) to the joint model. Consider increasing `--n-tune` proportionally.

**Effort:** Zero code changes. Just longer runtime — the LogNormal joint model took 56 minutes at 2,000 draws, so 4,000 draws would take ~112 minutes.

**Risk:** Low, but expensive in wall-clock time.

### Priority 5: Tighter Alpha Prior

**Expected impact:** Low-moderate. The current `alpha ~ Normal(0, 5)` allows bill difficulty parameters to wander far from zero, creating additional geometric challenges. Tightening to `Normal(0, 2)` or `Normal(0, 3)` regularizes the posterior without changing substantive estimates (difficulty parameters are typically in [-3, 3] for legislative votes).

**Effort:** One line change.

**Risk:** Low. The alpha prior is weakly informative; tightening it brings it closer to the data-generating range.

### Priority 6: Normalizing Flow Adaptation (nutpie)

**Expected impact:** Potentially transformative. nutpie's normalizing flow feature was designed for exactly this use case — models with ~1,000 parameters and complex posterior geometries. It learns a nonlinear transformation that makes the posterior approximately Gaussian, directly addressing both the reflection ridge and the funnel geometry.

**What to do:** Pass `flow_model=True` (or equivalent parameter) to nutpie. This is an experimental feature; check nutpie's current API for the exact parameter name and stability guarantees.

**Effort:** Low code change, moderate validation effort. The feature is experimental and may produce unexpected behavior.

**Risk:** Moderate. Experimental feature. May not work, may change API between releases. Need to validate that flow-adapted samples produce correct posteriors.

### Priority 7: Pathfinder Initialization

**Expected impact:** Moderate. Pathfinder finds approximate posterior modes via L-BFGS variational inference. Better than PCA for finding the correct mode in a multimodal posterior because it uses gradient information.

**What to do:** Install `pymc-extras`, run `pmx.fit(method="pathfinder")` on the joint model, extract the mode, and use it as initial points for nutpie sampling.

**Effort:** Moderate. New dependency (`pymc-extras`), new code to extract and format initial points.

**Risk:** Low-moderate. Pathfinder is published and peer-reviewed but its interaction with nutpie's own initialization is untested.

### Priority 8: Two-Stage Linking (Escape Hatch)

**Expected impact:** High, by sidestepping the problem. If concurrent calibration (the joint model) proves intractable, two-stage linking achieves the same goal — cross-chamber comparable scores — using a completely different approach.

**What to do:** (1) Estimate per-chamber hierarchical models (already works). (2) Match shared bills across chambers. (3) Use Stocking-Lord or Haebara linking to find the affine transformation (slope, intercept) that places Senate scores on the House scale. (4) Transform Senate ideal points. This is a 2-parameter optimization problem, not an MCMC problem.

**Effort:** Moderate. The bill-matching infrastructure (`_match_bills_across_chambers`) already exists. Need to implement the Stocking-Lord criterion function and optimizer. The `equating` Python package may provide this, though the ecosystem is thin.

**Risk:** Low computational risk (it's just optimization). The substantive risk is that shared bills may function differently across chambers (differential item functioning), which would bias the linking constants. A formal DIF test should precede linking.

### Priority 9: Sparsity Priors on Beta

**Expected impact:** Low-moderate. Many bills have near-zero discrimination (they pass nearly unanimously and provide no ideological information). A sparsity prior (horseshoe, half-Laplace) would shrink these uninformative betas toward zero, effectively reducing the model's dimension without pre-filtering.

**What to do:** Implement a horseshoe prior on beta: `beta ~ Horseshoe(tau=1)` or the regularized horseshoe of Piironen and Vehtari (2017). The `BetaPriorSpec` framework would need a new distribution type.

**Effort:** Moderate. New distribution in `BetaPriorSpec`, potential reparameterization needed for MCMC efficiency.

**Risk:** Moderate. Horseshoe priors can create their own sampling difficulties (the "horseshoe funnel"). Need careful parameterization.

## Recommendation

The most promising path forward combines Priorities 1-3: **PCA initialization + LogNormal beta + more draws**. These three interventions are low-effort, address both root causes, and build on proven techniques from our per-chamber work.

If the combined approach achieves convergence, we're done. If not, Priority 6 (normalizing flow adaptation) is the next best bet — it's a fundamentally different approach to the geometric problem, designed for models at exactly our scale.

Priority 8 (two-stage linking) remains the escape hatch. It trades model elegance for computational tractability and is essentially guaranteed to produce usable cross-chamber scores. The flat IRT already implements a basic version of this (mean-sigma equating); upgrading to Stocking-Lord would make it more robust.

The field's consensus is clear: concurrent calibration (one big model) is theoretically ideal but computationally challenging, while separate-then-link is practical and well-validated. Shor and McCarty — the gold standard for state-level scaling — use an exogenous bridge (surveys), not a concurrent model. DW-NOMINATE uses bridge legislators, not shared bills. Our concurrent approach via shared bills is less common in the literature, partly because the bill overlap rate (17% in the 91st) may be too low for effective bridging.

If Priorities 1-4 together do not achieve clean convergence (all diagnostics passing), we should seriously consider adopting two-stage Stocking-Lord linking as the production method, documenting the concurrent model as an aspirational benchmark in the experimental lab.

---

**Related documents:**
- Joint model diagnosis: `docs/joint-hierarchical-irt-diagnosis.md` (bill-matching bug and fix)
- Convergence improvement plan: `docs/hierarchical-convergence-improvement.md` (per-chamber + joint theory)
- Positive beta experiment: `results/experimental_lab/2026-02-27_positive-beta/experiment.md`
- nutpie hierarchical experiment: `results/experimental_lab/2026-02-27_nutpie-hierarchical/experiment.md`
- External validation: `docs/external-validation-results.md` (per-chamber models validated)
- Hierarchical shrinkage: `docs/hierarchical-shrinkage-deep-dive.md`
- PCA init experiment: `docs/hierarchical-pca-init-experiment.md`
- Model specification: `analysis/10_hierarchical/model_spec.py`
