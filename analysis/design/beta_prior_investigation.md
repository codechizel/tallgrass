# Beta Prior Investigation: The D-Yea Blind Spot

**Date:** 2026-02-20
**Status:** Active investigation
**Related:** `analysis/design/irt.md`, `docs/adr/0006-irt-implementation-choices.md`

## The Problem

The IRT model uses a LogNormal(0.5, 0.5) prior on the discrimination parameter (beta), constraining it to be positive. With the parameterization `P(Yea) = logit⁻¹(β·ξ - α)` and `β > 0`, the probability of voting Yea **always increases** with the ideal point ξ. This means the model can only represent bills where more-conservative legislators (higher ξ) prefer Yea.

Bills where the liberal position is Yea — i.e., Democrats vote Yea and Republicans vote Nay — cannot be fitted with positive beta. The model's only recourse is to assign near-zero discrimination, treating these bills as uninformative noise.

## The Evidence

From the 2025-26 House data (297 contested votes, 130 legislators):

| Bill direction | Count | % of bills | Mean β | Max β |
|---|---|---|---|---|
| R-Yea (majority of Rs vote Yea) | 259 | 87% | 3.46 | 7.21 |
| D-Yea (majority of Ds vote Yea) | 37 | 13% | 0.19 | 0.39 |

**100% of D-Yea bills have β < 0.5.** Every single one is treated as uninformative. On a typical D-Yea bill, the model predicts ~28% Yea rate for all legislators regardless of ideology, while the actual pattern is a clean party split (37 Ds vote Yea, 84 Rs vote Nay).

This creates a bimodal beta distribution:
- A cluster at β ≈ 7 (party-line R-Yea bills, hitting a practical ceiling)
- A cluster at β ≈ 0.17 (D-Yea bills, pushed to the LogNormal floor)
- A spread in between (crosscutting or partially-partisan bills)

## Why the Design Doc Was Wrong

The original `analysis/design/irt.md` stated:

> "Bills where the liberal position is Yea will have positive beta (since higher xi still predicts Yea) and negative alpha (the difficulty parameter shifts to make high-xi legislators vote Yea). This is mathematically equivalent to the negative-beta formulation."

This is **incorrect**. Here's the math:

```
P(Yea) = logit⁻¹(β·ξ - α)
```

- The sign of `β` determines the **direction**: whether P(Yea) increases or decreases with ξ.
- The value of `α` determines the **cutpoint**: the overall probability level.

With `β > 0`, no value of α can make P(Yea) *decrease* with ξ. Alpha shifts the entire curve up or down, but the slope is always positive. For a D-Yea bill, you need `β < 0` so that P(Yea) decreases with ξ (i.e., liberals are more likely to vote Yea). Constraining `β > 0` eliminates this possibility entirely.

**The claim that alpha handles directionality conflated two different things:** alpha handles the *threshold* (how easy/hard the bill is to pass), but beta alone determines the *direction* (which end of the spectrum favors Yea).

## Why This Matters

### Information loss
The model discards ideological signal from 37/297 House bills (12.5%). These bills could help differentiate moderate Republicans who occasionally cross party lines from hardliners who never do. That information is treated as noise.

### Asymmetric weighting
Only R-Yea bills contribute to ideal point estimation. A legislator's position is determined almost entirely by their behavior on bills where Republicans vote Yea. Their votes on D-Yea bills barely affect their estimated ideal point.

### HDI widening at the conservative extreme
The D-Yea bills would help distinguish ultra-conservative legislators from mainstream conservatives (based on how consistently they vote Nay even on bills Democrats support). Without this information, the conservative tail has wider credible intervals than necessary.

### Misleading bill parameters
The beta values for D-Yea bills are meaningless — they're artifacts of the prior constraint, not measures of ideological content. Any downstream analysis using bill discrimination values (e.g., selecting high-discrimination bills for network analysis) will systematically exclude D-Yea bills.

## Why It's Not Catastrophic

1. **Only 12.5% of bills affected** (in Kansas). The Republican supermajority means most contested votes have R-Yea majorities. In a more balanced legislature, the proportion of D-Yea bills could be much higher (~50%), making this problem far worse.

2. **Information is correlated.** A legislator who votes Nay on R-Yea bills will typically vote Yea on D-Yea bills. The R-Yea bills alone contain most of the ranking information — the D-Yea bills add precision but not fundamentally different signal.

3. **Validation metrics are strong.** Holdout accuracy: 90.6% (House), 91.5% (Senate). PPC Bayesian p-values: 0.49, 0.42. PCA correlation: r = 0.95, 0.92. The ideal points are approximately correct.

4. **Sensitivity analysis is excellent.** r > 0.99 between 2.5% and 10% threshold runs. The 1D structure is robust.

## Why the Anchors Change the Calculus

The original rationale for LogNormal was sign identification: with Normal priors on beta, flipping the sign of (β_j, ξ_i) leaves the likelihood unchanged, causing sign-switching between chains.

But our model has **hard anchors**: two legislators are fixed at ξ = +1 and ξ = -1. These anchors break the sign symmetry for ξ. Once ξ is identified, the sign of β is also identified:
- If ξ_anchor = +1 is the conservative anchor, and the bill is R-Yea, then β > 0 is the only explanation (higher ξ → higher P(Yea))
- If the bill is D-Yea, then β < 0 is the only explanation (higher ξ → lower P(Yea))

**The anchors make the LogNormal constraint unnecessary.** The sign identification problem that LogNormal solves has already been solved by the anchors. We're paying the cost (D-Yea blind spot) without getting the benefit (sign identification is already handled).

## Fixes to Try

### Fix A: Normal(0, 2.5) prior — unconstrained

Replace LogNormal with a symmetric Normal prior centered at zero. The sign of beta is free to be positive (R-Yea bills) or negative (D-Yea bills). The anchors ensure the sign is identified.

**Expected benefits:** D-Yea bills get properly discriminating beta values. All 297 bills contribute to ideal point estimation.

**Expected risks:** If the anchors provide weaker identification than expected, we might see sign-switching between chains (beta alternating between +/- across chains for the same bill). Check for: bimodal trace plots, R-hat > 1.01 on beta parameters, divergences.

### Fix B: Normal(0, 1) prior — tighter unconstrained

Same as Fix A but with a tighter prior (SD = 1 instead of 2.5). This regularizes more strongly, pulling extreme discrimination values toward zero.

**Expected benefits:** Same as Fix A, with less risk of extreme beta values.

**Expected risks:** May over-regularize — highly discriminating bills might have their beta pulled toward zero.

### Fix C: Pre-flip D-Yea bills

Before fitting, identify D-Yea bills and flip their vote encoding (1→0, 0→1) in the vote matrix. Keep the LogNormal prior. After fitting, note the flipped bills — their beta values represent the absolute discrimination, and the flip indicates direction.

**Expected benefits:** All bills are now "R-Yea direction" in the model, so LogNormal works for all of them. No changes to the sampling algorithm.

**Expected risks:** Adds preprocessing complexity. The flip decision must be correct and consistent. Edge cases (bipartisan bills where neither party has a clear majority direction) could be mishandled.

## Experiment Protocol

Run each fix on House data only (larger chamber, more signal) with reduced MCMC for speed:
- 500 draws, 300 tune, 2 chains (vs. 2000/1000/2 production)
- Compare: beta distributions, D-Yea bill handling, ideal point correlations, holdout accuracy, convergence

Record results below as each experiment completes.

---

## Results

### Experiment setup

- **Data:** House only (130 legislators × 297 votes)
- **MCMC:** 500 draws, 300 tune, 2 chains (reduced for speed; convergence warnings expected)
- **Bill directions:** 259 R-Yea, 37 D-Yea (classified by which party's majority votes Yea)
- **Script:** `analysis/irt_beta_experiment.py`
- **Output:** `results/2025-2026/irt/beta_experiment/`

### Metrics comparison

| Metric | LogNormal(0.5,0.5) | Normal(0,2.5) | Normal(0,1) |
|---|---|---|---|
| Sampling time (s) | 62 | 79 | **51** |
| Divergences | 0 | 0 | 0 |
| ξ R-hat max | 1.070 | 1.014 | **1.013** |
| ξ ESS min | 21 | 123 | **203** |
| β R-hat max | 1.017 | 1.018 | **1.014** |
| PCA Pearson r | 0.950 | 0.963 | **0.972** |
| PCA Spearman ρ | **0.916** | 0.905 | 0.915 |
| Holdout accuracy | 0.908 | **0.944** | 0.943 |
| Holdout AUC-ROC | 0.954 | **0.980** | 0.979 |
| D-Yea \|β\| mean | 0.186 | **4.767** | 2.384 |
| R-Yea β mean | 3.45 | 2.68 | 1.37 |
| Bills with β < 0 | 0 | 63 | 67 |

### Key findings

**1. Both Normal variants dramatically improve holdout accuracy (+3.5%).**

LogNormal: 90.8% accuracy, 0.954 AUC. Normal(0,1): 94.3% accuracy, 0.979 AUC. The improvement comes entirely from properly modeling the 37 D-Yea bills that LogNormal treated as noise. When the model can represent "Democrats vote Yea on this bill," its predictions for those bills improve from chance to correct.

**2. The sign-switching fear was unfounded.**

Zero divergences across all three models. No sign-switching between chains. The hard anchors (ξ fixed at ±1) provide sufficient identification — the LogNormal constraint was solving a problem that the anchors had already solved. This is the central insight: **when anchors fix the sign of ξ, the sign of β is determined by the data**, not the prior.

**3. Normal(0,1) is the best overall.**

- Best convergence: lowest R-hat, highest ESS (10× the LogNormal ESS)
- Fastest sampling: 51s vs 62s (LogNormal) and 79s (Normal(0,2.5))
- Highest PCA correlation: r = 0.972 (vs 0.950 for LogNormal)
- Holdout nearly identical to Normal(0,2.5): 94.3% vs 94.4%
- The tighter prior provides useful regularization — it prevents the β ≈ 7 ceiling seen in LogNormal and the β ≈ 6 spread seen in Normal(0,2.5), while still giving D-Yea bills |β| ≈ 2.4 (well above zero)

**4. D-Yea bills are now properly modeled.**

Under LogNormal, all 37 D-Yea bills had |β| < 0.39 (effectively zero). Under Normal(0,1), they have |β| mean = 2.38 with negative sign — the model correctly identifies that higher ξ predicts Nay on these bills. The beta distribution plot shows clean separation: R-Yea bills cluster on the positive side, D-Yea bills cluster on the negative side.

**5. More bills contribute negative β than expected.**

67 bills (not just 37) have β < 0 under Normal(0,1). The extra ~30 bills are crosscutting votes where the minority Yea coalition skews liberal — not pure D-Yea bills but votes where ideology partially predicts in the reverse direction. These were previously invisible.

**6. The ideal point scale stretches under Normal(0,1).**

The xi comparison scatter shows Normal(0,1) produces a wider spread, especially on the liberal end (Democrats at ξ ≈ -4 instead of ξ ≈ -2). This is because D-Yea bills now provide discriminating information that helps separate Democrats from each other. The overall correlation with LogNormal remains very high (r = 0.983, ρ = 0.974) — the ranking is preserved but the spacing changes.

### Interpretation

The LogNormal(0.5, 0.5) prior was a double-edged sword:

**What it was supposed to do:** Prevent sign-switching by constraining β > 0. This is the standard approach in IRT models without hard anchors (soft identification via priors).

**What it actually did in our model:** Since our anchors already provide hard identification, the positive constraint was redundant — and actively harmful. It silenced 12.5% of bills, degraded holdout accuracy by 3.5 percentage points, and produced worse convergence diagnostics (10× lower ESS).

**Why the literature recommendation didn't apply:** Most IRT implementations in the political science literature use either (a) soft identification (priors only, no anchors) or (b) post-hoc rotation. In these setups, positive-constrained β is genuinely helpful for identification. But we use **hard anchors** (two legislators fixed at ξ = ±1), which is a different identification strategy. The combination of hard anchors + positive β is over-identified and loses information unnecessarily.

### Recommendation

**Switch the production IRT model to Normal(0, 1) for the beta (discrimination) prior.**

This is a one-line code change (`pm.Normal` instead of `pm.LogNormal`) that:
- Uses all 297 bills instead of only 259
- Improves holdout accuracy by ~3.5%
- Improves convergence (10× ESS, lower R-hat)
- Runs faster (18% wall-clock reduction)
- Requires no post-hoc sign correction (anchors handle identification)

The design docs, ADR-0006, and the IRT primer should be updated to reflect this change and document the investigation.

### Plots

See `results/2025-2026/irt/beta_experiment/`:
- `beta_distributions.png` — Side-by-side histograms showing D-Yea bills crushed at zero under LogNormal vs properly negative under Normal variants
- `xi_comparison.png` — Scatter plots of ideal points: Normal variants vs LogNormal baseline (r > 0.98)
- `metrics_comparison.png` — Summary metrics table as figure
