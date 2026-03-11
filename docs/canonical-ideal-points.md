# Canonical Ideal Points: From 1D Fixes to 2D Dim 1 Promotion

**Date:** 2026-03-11
**Status:** Design phase
**Related:** ADR-0103, ADR-0104, ADR-0107, ADR-0108, `docs/horseshoe-effect-and-solutions.md`, `docs/dim1-informative-prior.md`

## The Problem

The 79th Kansas Senate (2001-02, 30R/10D) is the poster child for a structural failure in 1D ideal point estimation. Senator Tim Huelskamp — a documented far-right conservative who was later elected to Congress as a Tea Party Republican — appears as the *most liberal* member of the Senate in the 1D IRT model. This isn't a convergence failure or a coding bug. It's the horseshoe effect: in a supermajority chamber, 1D IRT conflates ideology with establishment-vs-rebel contrarianism because both ultra-conservative rebels and Democrats vote Nay on the same bills for opposite reasons.

The 2D Bayesian IRT model (Phase 06) correctly separates these dimensions. Dim 1 = ideology (left-right), Dim 2 = establishment-vs-rebel. On Dim 1, Huelskamp sits at +1.401 (far-right conservative). On Dim 2, he sits at -2.822 (extreme contrarian). The 2D model has always had the right answer.

This document records our journey through three failed approaches to fixing the 1D model, the discovery that we accidentally broke the 2D model, and the conclusion that the field-standard solution was available all along: **use 2D Dim 1 as the canonical ideology score**.

## Approach 1: Initialization (`--init-strategy 2d-dim1`)

**ADR-0107 (2026-03-10)**

The first idea: use 2D IRT Dim 1 scores as MCMC starting points for the 1D model. If the chains start near the ideology solution, maybe they'll stay there.

**Why it failed:** Initialization only sets where chains begin. The 1D posterior's dominant mode IS the contrarian axis — it has higher likelihood than the ideology mode. During tuning, chains drift from the ideology starting point to the contrarian mode regardless of where they start. The correlation between 1D IRT and 2D Dim 1 remains r = -0.13 even with `--init-strategy 2d-dim1`. The posterior landscape itself is the problem, not the starting point.

## Approach 2: Informative Priors (`--dim1-prior`)

**ADR-0108 (2026-03-11)**

The second idea: constrain the 1D posterior using `xi ~ Normal(dim1_values, sigma)` instead of `xi ~ Normal(0, 1)`. This changes the posterior landscape itself, not just the starting point.

**Partial success:** With `--dim1-prior`, Huelskamp moves to xi=+3.349 (most conservative). The prior pulls hard enough to overcome the contrarian mode. But this raises a question: if the prior must be tight enough to dominate the likelihood, what is the 1D model actually contributing? We're using the 2D model's output as input to a 1D model that then approximately recovers... the 2D model's output. Extra computation, reduced precision, same answer at best.

**The deeper problem:** This approach treats the 1D model as the canonical source and the 2D model as a corrective input. But for horseshoe-affected chambers, the 1D model is measuring the wrong thing. No amount of prior-based correction changes that fundamental mismatch.

## Approach 3: The Accidental Regression

**Commit `1bffb1d` (2026-03-10)**

While implementing the shared init strategy (ADR-0107), we introduced a subtle but devastating bug: the `--init-strategy auto` default for Phase 06 (2D IRT) now prefers 1D IRT scores over PCA for initialization. For most chambers, this is an improvement — the 1D IRT is a better starting point than raw PCA. But for the 79th Senate, the 1D IRT is horseshoe-confounded (r = **-0.94** with PCA PC1, nearly perfectly inverted).

The effect: Phase 06's 2D model starts from poisoned initial values. The Dim 1 posterior gets pulled toward the confounded axis. Dim 1 vs PCA PC1 correlation degrades from r = -0.19 (already weak, due to the Senate's swapped principal components) in the prior run to r = -0.56 in the current run. The clean 2D separation that we were counting on to fix the 1D model is itself now contaminated by the 1D model.

This created a circular dependency of garbage: bad 1D IRT → poisons 2D IRT init → degrades 2D Dim 1 → which we feed back into 1D IRT via `--dim1-prior`. Each step amplifies the original confounding instead of correcting it.

**The fix for the regression:** Phase 06 should never use 1D IRT scores for initialization when the 1D model is horseshoe-affected. At minimum, the `auto` strategy should check the 1D-PCA correlation and fall back to PCA when they disagree. More conservatively, Phase 06 should always use PCA for initialization — PCA is cheap, always available, and doesn't carry model-based confounding.

## The Field-Standard Solution: Use 2D Dim 1

After three attempts to fix the 1D model, we arrived at the insight that the field already solved this problem decades ago.

### The DW-NOMINATE Precedent

DW-NOMINATE — the dominant methodology for measuring legislative ideology in political science — has **always been a 2D model** where Dimension 1 is extracted as "the" ideology score. Keith Poole, Howard Rosenthal, and the Voteview project do not run separate 1D models and agonize over horseshoe artifacts. They fit 2D (or higher) and extract Dim 1. Every published NOMINATE score — the ones cited in thousands of political science papers, used by journalists, referenced in Supreme Court briefs — is Dim 1 from a multidimensional model.

Clinton, Jackman & Rivers (2004) — the IDEAL model, the Bayesian counterpart to NOMINATE — similarly recommend fitting with at least 2 dimensions. Shor & McCarty (2011), whose state legislature scores are the closest analogue to what we're doing, also use a multidimensional framework.

The second dimension in Congress historically captured civil rights/race (1950s-1980s) and collapsed below 5% variance after the 1990s. In the 79th Kansas Senate, the second dimension captures establishment-vs-rebel contrarianism at 13.6% variance — substantively meaningful, comparable to where Congress was in the 1990s.

### Why This Is Better Than Fixing 1D

| Approach | What it does | Result |
|----------|-------------|--------|
| 1D IRT (standard) | Estimates a single latent dimension | Confounded in supermajority chambers |
| `--init-strategy 2d-dim1` | Starts 1D chains at 2D Dim 1 values | Chains drift to confounded mode |
| `--dim1-prior` | Constrains 1D posterior toward 2D Dim 1 | Approximately recovers 2D Dim 1 with extra noise |
| `--horseshoe-remediate` | Filters votes + PC2 prior | Discards 60% of data, uses PCA not model-based |
| **2D Dim 1 directly** | Extract Dim 1 from the 2D model | Clean ideology axis, full data, model-based |

For balanced chambers where 1D works fine (most House sessions, Senate sessions after the 84th), 1D and 2D Dim 1 correlate at r > 0.95. There's no cost to using 2D Dim 1 — it's the same answer. For horseshoe-affected chambers, 2D Dim 1 is the correct answer and 1D is not.

### Statistical Considerations

**Convergence:** The 2D model has relaxed convergence thresholds (R-hat < 1.05 vs < 1.01 for 1D). However, Dim 1 parameters typically converge much better than Dim 2. Per-dimension convergence reporting (which `check_2d_convergence()` already computes) allows us to verify that Dim 1 meets production-grade thresholds even when Dim 2 does not.

**Identification:** The 2D model uses PLT (Promax Loading Target) identification with `beta[0,1]=0` (Dim 1 items load only on Dim 1 for the first item), plus post-hoc party-mean sign checks. This is more complex than 1D anchor-based identification but is field-standard for multidimensional IRT.

**Uncertainty:** The 2D model produces full posteriors for Dim 1, including HDIs. These are directly usable in downstream phases that currently consume `xi_mean`, `xi_sd`, `xi_hdi_2.5`, `xi_hdi_97.5` from the 1D model.

## The Path Forward

### Immediate Fix: Restore the 2D Model

The `auto` init strategy for Phase 06 must not use 1D IRT when it's horseshoe-confounded. Two options:

1. **Conservative:** Phase 06 always uses PCA for initialization (`--init-strategy pca-informed` as default). PCA is cheap, always available, and carries no model-based confounding. For chambers where 1D is good, PCA is nearly as good an initializer. For chambers where 1D is bad, PCA is dramatically better.

2. **Smart fallback:** The `auto` strategy checks the 1D IRT vs PCA correlation before using 1D scores. If |r| < 0.7 (indicating severe disagreement / horseshoe), fall back to PCA.

**Recommendation: Option 1 (conservative).** The marginal benefit of IRT-over-PCA initialization is small (slightly faster convergence for well-behaved chambers), while the downside risk (poisoning the 2D model) is catastrophic. For Phase 06 specifically, PCA should be the default.

### Canonical Ideal Points Pipeline

After fixing the 2D model, implement the canonical output routing:

1. Phase 05 (1D IRT) runs as-is. Its output is the 1D ideal point — correct for most chambers, confounded for some.
2. Phase 06 (2D IRT) runs as-is (with PCA init). Adds a Dim 1 forest plot to its report.
3. A new canonical routing step after Phase 06 writes `canonical_ideal_points_{chamber}.parquet`:
   - For balanced chambers (horseshoe detector negative): copy 1D scores.
   - For horseshoe-affected chambers (horseshoe detector positive AND Dim 1 convergence passes): use 2D Dim 1 scores.
4. Downstream phases (synthesis, profiles, cross-session, etc.) read from the canonical source.

This preserves both models' outputs while automatically routing the best available score to consumers.

### Deprecation

The `--dim1-prior`, `--horseshoe-remediate`, and `--init-strategy 2d-dim1` flags remain available for research but are no longer the recommended workflow for production runs. The canonical routing system subsumes their purpose.

## Lessons Learned

1. **Don't feed confounded outputs into upstream models.** The `auto` init strategy's preference for 1D IRT scores assumed those scores are reliable. For horseshoe-affected chambers, they're anti-reliable — worse than random, actively misleading.

2. **Don't fix models that are measuring the wrong thing.** The 1D model isn't broken for the 79th Senate — it's correctly estimating the dominant 1D latent trait, which happens to be contrarianism, not ideology. Adding priors to force it toward ideology is fighting the model's likelihood.

3. **The field already solved this.** DW-NOMINATE has used 2D Dim 1 as the primary ideology score for 40 years. Our custom approach (1D model + horseshoe patches) was reinventing the wheel, poorly.

4. **Circular dependencies amplify errors.** When 1D → 2D → 1D, a confounded 1D contaminates the 2D initialization, which degrades the 2D output, which we then feed back to 1D. Each step makes things worse.

5. **PCA is a reliable initialization source precisely because it's simple.** PCA doesn't carry model-based confounding. It can have swapped dimensions (as in the 79th Senate where PC1 = establishment and PC2 = ideology), but the 2D model's PLT rotation handles that. Using a confounded model's output as initialization is strictly worse than using the unconfounded but possibly mis-ordered PCA.
