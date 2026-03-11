# The Horseshoe Effect: When One Dimension Isn't Enough

**Date:** 2026-03-08
**Context:** 79th Kansas Legislature (2001–2002), IRT Phase 05

## What Is the Horseshoe Effect?

Imagine lining up every senator from most liberal to most conservative along a single
number line. That's what a one-dimensional IRT model does — it finds the single axis
that best explains how legislators vote. For most sessions, the line runs from liberal
Democrats on the left to conservative Republicans on the right, and the model works
well.

But in a **supermajority chamber** — like the 79th Kansas Senate with 30 Republicans
and 10 Democrats — something goes wrong. The most important divide in the chamber
isn't between the parties anymore. It's *within* the dominant party: establishment
Republicans vs. rebel conservatives. Tim Huelskamp, a staunch fiscal conservative who
later served in the Freedom Caucus in Congress, voted against Republican leadership
on bill after bill. So did the Democrats — but for completely opposite reasons.

The model sees Huelskamp and the Democrats voting the same way and concludes they
must be ideological neighbors. On the recovered number line, Huelskamp ends up scored
as the *most liberal* senator — more liberal than any Democrat.

This is the horseshoe effect. The true ideological landscape has two dimensions (a
left–right axis and an establishment–rebel axis), but we're forcing it onto one
dimension. The endpoints of the real spectrum fold back on themselves, like bending a
stick into a horseshoe shape, and legislators who are ideological opposites end up
overlapping.

### How We Know It's Happening

We built three diagnostic tests that together confirm the distortion:

1. **Democrat wrong-side fraction.** In a correctly recovered dimension, all Democrats
   should be on the liberal (negative) side. In the 79th Senate, 30% of Democrats
   landed on the conservative side. That's a clear signal.

2. **Contested-only refit.** We rerun the model using *only* bills where both parties
   split (at least 10% of each party on each side). In the House, the full model and
   the contested-only model barely agreed (correlation r = 0.41), meaning 63% of
   the votes were intra-party splits that contaminated the ideological dimension.

3. **2D comparison.** We compare 1D rankings against the first dimension from a 2D
   model. In the 79th Senate, they *negatively* correlate (r = −0.13). The 1D and
   2D models don't just disagree about magnitudes — they disagree about which
   direction is "conservative."

### Seeing the Horseshoe: What the 2D Axes Mean

The 2D IRT model produces interactive scatter plots where you can hover over any
dot to identify the legislator. These plots make the horseshoe directly visible —
but only if you understand what each axis is actually measuring.

**PCA PC1 (x-axis on the Dim 1 vs PC1 plot)** is supposed to represent ideology,
but in supermajority chambers it is ideology *confounded with contrarianism*. PCA
is a linear method — it projects onto the direction of maximum variance. When a
legislator like Huelskamp votes Nay on bills that nearly all Republicans support,
PCA cannot distinguish "Nay because liberal" from "Nay because contrarian." The
Nay votes look identical in the binary matrix. So PCA folds the horseshoe flat
and puts far-right contrarians next to far-left Democrats. In the 79th Senate,
Huelskamp lands at PC1 = −31 — past every Democrat.

**IRT Dim 1 (y-axis on the Dim 1 vs PC1 plot)** is closer to pure ideology. The
2D model has a second dimension to absorb the contrarianism variance, so Dim 1
does not have to carry that burden. Huelskamp goes from PC1 = −31 (most liberal!)
to Dim 1 = +1.4 (most conservative). The model correctly infers that his Nay
votes are not coming from the same place as Democrat Nay votes — they are on
different bills, with different discrimination parameters.

**PCA PC2 (x-axis on the Dim 2 vs PC2 plot)** captures the horseshoe arch
itself. It is the residual pattern after PC1 extracts its confounded signal.
Huelskamp at PC2 = +10 is an extreme outlier because the arch is most
pronounced at the endpoints. In the Dim 2 vs PC2 scatter, you can literally see
the horseshoe curve — the Democrat cluster in the bottom-left sweeps up through
moderate Republicans and curves back down to Huelskamp in the bottom-right.

**IRT Dim 2 (y-axis on the Dim 2 vs PC2 plot)** measures orthogonal
contrarianism: how much a legislator deviates from the expected voting pattern
given their ideological position. Huelskamp is extreme on Dim 2 (−2.8) because
he genuinely is a contrarian — but now that is measured on a separate axis from
his ideology, not confounded with it.

The key diagnostic insight: **when Dim 1 and PC1 disagree about a legislator,
that is the horseshoe at work.** The comparison plots are horseshoe detectors.
In the 91st Senate (2025–2026), Caryn Tyson shows a milder version of the same
pattern. In the 79th Senate (2001–2002), Huelskamp is the extreme case where
the horseshoe completely dominates PC1, driving the Dim 1 vs PC1 correlation to
r = −0.19. The two methods do not just disagree about magnitudes — they disagree
about which direction is "conservative."

Even though the 2D model's convergence is poor for the 79th (R-hat up to 1.96,
ESS as low as 5), it is directionally correct as a diagnostic tool. The poor
convergence actually makes sense: the model is trying to separate two dimensions
from data where PCA conflates them, and with only 40 senators that is a hard
estimation problem. The interactive plots let you verify the correction
legislator by legislator.

## Why This Matters

If you're a researcher, journalist, or citizen using ideal-point scores to
understand legislators' positions, the horseshoe effect can lead you badly astray.
A far-right rebel looks liberal. A moderate establishment figure looks extreme.
Rankings that should track ideology instead track loyalty to party leadership.

The 79th biennium is the clearest case in our data, but any session with a large
supermajority (roughly 70% or more single-party) is at risk. Several Kansas
bienniums fall into this territory.

## The Identification Problem in Multiple Dimensions

Before exploring solutions, it helps to understand a deeper problem that makes
multidimensional models difficult: **rotational invariance**.

Think of placing legislators on a 2D map, like pins on a corkboard. The model
figures out the *distances* between legislators — who votes similarly and who
doesn't — but it has no compass. You could rotate the entire map by any angle
and all the distances would stay exactly the same. There's nothing in the voting
data alone to tell the model which direction is "left" and which is "up."

In one dimension, this problem is trivial — you can only flip left and right,
and we handle that with a sign check. In two dimensions, the number of equivalent
configurations is *infinite* (any rotation angle works). Current methods handle
this by either:

- **Anchoring** specific legislators to fixed positions based on prior knowledge
  (fragile — get one anchor wrong and the whole map distorts)
- **Forcing dimensions to be uncorrelated** (a strong assumption that may not
  match reality — economic and social conservatism often correlate)

This is why most political science research sticks to one dimension, even when
two dimensions would tell a richer story. The identification problem in higher
dimensions has been an unsolved nuisance for decades.

## Approaches Under Investigation

We are evaluating six approaches to address the horseshoe effect. They range
from quick diagnostic improvements to fundamental changes in the statistical
model.

### 1. Auto-Promote 2D Results for Supermajority Sessions

**The idea:** When horseshoe diagnostics trigger, automatically flag the 1D results
as unreliable and promote the 2D model's results as the canonical output.

**Why it could work:** The 2D model correctly separates the inter-party and
intra-party dimensions. In the 79th Senate, the 2D model places Democrats at one
end and conservative rebels at the other — exactly right. The 1D model's problem
is that it's the wrong tool for this particular job.

**Trade-off:** We lose the simplicity of a single left–right score. 2D results are
harder to summarize and compare across sessions. But an accurate 2D picture is
better than a misleading 1D one.

**Effort:** Low. The infrastructure already exists (`--promote-2d` flag).

### 2. Default to Contested Votes for Supermajority Sessions

**The idea:** When a chamber has a supermajority, automatically filter to
cross-party contested votes before fitting the IRT model.

**Why it could work:** Contested votes — those where both parties split — are the
votes that actually distinguish liberals from conservatives. Removing near-unanimous
votes strips out the intra-party noise that causes the horseshoe. In the Senate,
the contested-only refit correlates at r = 0.96 with the full model, meaning the
dimension is stable. In the House, the correlation drops to r = 0.41, revealing
that intra-party votes were dominating the result.

**Trade-off:** We discard information. Some near-unanimous votes are genuinely
informative (a bill that passes 39-1 tells you something about the lone dissenter).
Filtering is a blunt instrument.

**Effort:** Low. The `--contested-only` flag already exists.

### 3. The L1-Based Ideal Point Model (Shin, Lim & Park, 2025)

This is the most theoretically interesting approach. Published in the *Journal of
the American Statistical Association* in 2025, it makes a single mathematical
change that dissolves the rotational invariance problem.

#### How Standard Models Measure Distance

Every ideal point model asks the same question: "How far is this bill's outcome
from what legislator X wants?" If the bill is close to what they want, they vote
for it. If it's far away, they vote against it. The question is how you measure
"far."

The standard approach uses **Euclidean distance** — the straight-line distance
between two points, the way a crow flies. In two dimensions, all the points at
the same Euclidean distance from a center form a **circle**. This is the L2 norm,
named after the squaring operation in the Pythagorean theorem:
distance = √(Δx² + Δy²).

The key property of a circle: you can rotate it and it looks the same. This is
exactly why the standard model can't tell which direction is "left" — the
distance formula treats all directions equally.

#### What the L1 Model Changes

The L1 model replaces Euclidean distance with **Manhattan distance** — the
distance you'd walk in a city with a grid of streets, going only north–south
and east–west. Instead of the crow-flies diagonal, you walk the blocks:
distance = |Δx| + |Δy|.

All the points at the same Manhattan distance from a center form a **diamond**
(a square rotated 45 degrees), not a circle.

A diamond has corners. It has flat edges aligned with the axes. Unlike a circle,
a diamond is *not* the same shape after an arbitrary rotation. The only rotations
that map a diamond exactly onto itself are 90°, 180°, and 270° — plus
reflections. That's 8 transformations total, compared to the circle's infinite
number.

This means the L1 model's likelihood function can only be mapped onto itself by
those 8 transformations. The infinite ambiguity of the standard model collapses
to a small, manageable set of discrete equivalences. In practice, these are
trivial to resolve — you just check which of the 8 configurations puts
Republicans on the right.

#### Why This Helps with the Horseshoe

The horseshoe effect happens because the 1D model can't distinguish between two
dimensions that contribute roughly equally to voting variation. The L1 model is
natively multidimensional and, crucially, it can recover *correlated* dimensions
without external anchors. In the 79th Senate, it could potentially recover both
the left–right axis and the establishment–rebel axis simultaneously, without
forcing them to be perpendicular.

In simulations by Shin et al., the L1 model correctly recovered the true
configuration of legislators in scenarios where both W-NOMINATE and standard
Bayesian IRT failed due to rotational invariance.

#### The Trade-Off

Manhattan distance uses absolute values (|Δx|), which have a sharp corner at
zero — the function isn't smooth. Most modern Bayesian samplers (including our
nutpie NUTS sampler) rely on computing gradients, and gradients are undefined at
sharp corners. The original paper uses a specialized gradient-free sampler
(multivariate slice sampling), which is slower.

We can work around this by using a smooth approximation — replacing |x| with
√(x² + ε) for a tiny ε — or by calling the authors' R package as a subprocess,
the way we already handle W-NOMINATE.

**Effort:** Medium to high. New model formulation, sampler considerations, and
validation against known results.

### 4. External Anchoring via Shor-McCarty Scores

**The idea:** Use nationally comparable ideal-point scores from academic datasets
as informative priors, placing legislators on a scale that's pre-calibrated to
left–right ideology.

**Why it could work:** External scores impose the correct ideological structure
before the model even sees the data. If Huelskamp's national score says he's
far-right, the model starts from that belief and the data has to overcome a
strong prior to move him elsewhere.

**Trade-off:** Only works for sessions where external scores exist. Shor-McCarty
coverage of Kansas is incomplete, especially for older sessions like the 79th.

**Effort:** Medium. The `external-prior` identification strategy already exists
in the codebase.

### 5. Regularized Horseshoe Prior on Discrimination

**The idea:** Replace the current Normal(0, 1) prior on bill discrimination
parameters (β) with a regularized horseshoe prior (Piironen & Vehtari, 2017).
This would adaptively shrink uninformative bills' discrimination toward zero
while allowing informative bills to retain large values.

**Why it could work:** Many votes in supermajority chambers are near-unanimous
and carry almost no information about ideology. Including them adds noise. A
shrinkage prior would automatically down-weight these votes without requiring
explicit filtering.

**Trade-off:** The regularized horseshoe adds model complexity and new
hyperparameters (slab width, expected sparsity). It addresses vote weighting
but does not fix the fundamental geometric problem — even with perfect
discrimination estimates, a 1D model in a 2D chamber will still horseshoe.

**Effort:** Medium. Well-documented in the PyMC ecosystem.

### 6. Hierarchical Cross-Session Borrowing

**The idea:** Fit a hierarchical model that shares information across sessions.
Legislators who serve in both a supermajority session and a balanced session
would have their ideal points informed by the balanced session, where the
dimension is well-identified.

**Why it could work:** A legislator's ideology doesn't change drastically
between sessions. If the 80th session (more balanced) places Huelskamp as
far-right, that information can anchor his position in the 79th as well.

**Trade-off:** Computationally expensive. Requires careful modeling of which
parameters are shared vs. session-specific. Assumes ideological stability
across sessions, which may not hold during realignment periods.

**Effort:** High. New model architecture.

## Experimental Results (2026-03-08)

We tested three of the six approaches. Results changed our understanding of
which paths are viable.

### Supermajority Audit

We ran horseshoe diagnostics across all 28 chamber-sessions (78th–91st):

- **3 horseshoe detections:** 79th Senate (severe), 80th Senate (moderate),
  82nd Senate (mild). All in the Kansas Senate, all in the early 2000s.
- **14 borderline cases:** Every Senate from 81st–91st has a supermajority
  but passes horseshoe detection. The identification strategy system
  (anchor-agreement) appears effective for these sessions.
- **5 sessions with 1D-2D disagreement:** The 79th, 80th, 81st, 83rd, and
  88th Senates show concerning discrepancies between 1D rankings and 2D
  Dimension 1 (|r| < 0.70 or negative). This is broader than horseshoe
  detection alone suggests.

### Regularized Horseshoe Prior: RULED OUT

The regularized horseshoe (Piironen & Vehtari 2017) on discrimination
parameters made things **dramatically worse**:

- **House:** Perfectly inverted all ideal points (r = -0.9999 vs baseline).
  97.9% of Democrats on the wrong side. The over-shrinkage of discrimination
  collapsed the ideological signal entirely.
- **Senate:** Failed to converge (R-hat 1.83, ESS 3). The auxiliary variable
  structure created funnel geometry the sampler could not navigate.
- **9x slower** in the House due to ~1,870 additional parameters.

This confirms that the horseshoe effect is a geometric problem, not a
vote-weighting problem. Shrinkage priors on discrimination do not help.

### L1-Based Model (Smooth Approximation): PROMISING BUT IMPRACTICAL

The smooth L1 approximation (√(x² + ε)) with nutpie NUTS sampling showed
directional promise but is not viable in its current form:

- **House:** Perfect party separation (0% Democrats on wrong side) and good
  correlation with standard IRT (r = 0.812). But catastrophic convergence
  failure (2,000 divergences, R-hat 2.26, ESS 3).
- **Senate:** Inverted relative to 1D IRT (r = -0.954), 20% wrong side,
  convergence failure (80 divergences, R-hat 1.87).
- **3.3 hours** for the House alone (vs ~2.5 minutes for standard IRT).

The smooth approximation creates difficult curvature near zero that NUTS
cannot navigate. The original paper uses gradient-free slice sampling for
good reason. **Next step: call the authors' R package via subprocess**
(following the W-NOMINATE pattern), which uses the correct sampler.

## The Swapped Dimensions: PCA Gets It Backward

A closer look at the 79th Senate PCA reveals something striking: **the dimensions
are in the wrong order.** PCA extracts components by explained variance, largest
first. But in the 79th Senate, the largest source of variance is not ideology —
it's establishment loyalty.

| Dimension | Variance | Party Separation (R mean vs D mean) | What It Measures |
|-----------|----------|--------------------------------------|------------------|
| PC1 | 19.6% | R = +0.6, D = −1.7 (gap: 2.3) | Establishment vs. rebel |
| PC2 | 13.6% | R = +4.2, D = −12.5 (gap: 16.7) | **Left–right ideology** |

PC2 has **perfect party separation**: zero Democrats on the Republican side, zero
Republicans on the Democrat side. PC1 has 9 Republicans on the Democrat side and
2 Democrats on the Republican side. PC2 is the clean ideology axis. But because
PC1 explains slightly more variance (ratio 1.45:1), PCA puts it first.

The 1D IRT model follows PCA's lead. It initializes from PC1 scores, which point
it toward the establishment-loyalty axis. It then locks in: the correlation
between 1D IRT and PC1 is r = −0.964, while 1D IRT vs PC2 is r = −0.068 — the
model has **zero correlation** with the actual ideology dimension.

Meanwhile, the 2D IRT model correctly untangles the two. Its correlations with
PCA confirm the swap:

| | PC1 (establishment) | PC2 (ideology) |
|---|---|---|
| **IRT Dim 1** (ideology) | r = −0.27 | **r = 0.95** |
| **IRT Dim 2** (contrarianism) | **r = 0.99** | r = −0.10 |

IRT Dim 1 aligns with PC2 (ideology). IRT Dim 2 aligns with PC1
(establishment). The 2D model figured out which dimension is which — PCA couldn't.

### Can We Nudge the 1D Model to the Right Dimension?

Since the 1D model locks onto PC1 (the wrong dimension), a natural question: can
we point it at PC2 instead? There are four viable mechanisms, and they can be
combined.

#### Approach A: Informative Prior from PC2 Scores

The most principled approach. Instead of the default `xi ~ Normal(0, 1)` prior on
ideal points, use PC2 scores as the prior mean: `xi ~ Normal(PC2_score, sigma)`.
This creates genuine posterior mass favoring the ideology direction. The model
starts believing the PC2 ordering is correct, and the data has to actively
contradict that belief to move a legislator.

The key design parameter is **sigma** (the prior width):
- **sigma = 0.5**: Strong pull. Risk: prior dominates the data in a chamber with
  only 40 senators. Diagnostic: if posterior xi correlates >0.99 with the PC2
  input, the data isn't adding anything and the prior is too tight.
- **sigma = 1.0**: Moderate pull. The data can move legislators away from PC2
  positions, but the PC2 direction is favored over PC1.
- **sigma = 2.0**: Weak pull. May not be enough to prevent drift back to PC1
  during MCMC tuning.

This approach has precedent: Clinton, Jackman & Rivers (2004) use informative
"spike" priors on anchor legislators to fix the dimension. Our
`external-prior` identification strategy (ADR-0103) already implements the
infrastructure — it accepts an array of prior means and sets
`xi ~ Normal(external_priors, 0.5)`. Currently wired for Shor-McCarty scores
but unused. Passing PC2 scores requires no new model code, just a new data path.

#### Approach B: PC2-Informed Initialization

A lighter touch: start the MCMC chains at PC2 scores instead of PC1 scores.
Currently, the IRT model initializes `xi_free` from PC1 via `initial_points`:

```python
pc1_vals = pca_scores["PC1"].to_numpy()
pc1_std = (pc1_vals - pc1_vals.mean()) / (pc1_vals.std() + 1e-8)
xi_init = pc1_std[free_pos]
```

Changing `["PC1"]` to `["PC2"]` is a one-line edit. But initialization alone is a
weak signal — NUTS explores aggressively during tuning and can drift away from the
PC2 basin toward the higher-variance PC1 direction. Initialization matters most
when the posterior is genuinely multimodal (separate basins), not just a ridge.

**Best used in combination with Approach A.** The informative prior holds the model
on the PC2 axis; the initialization ensures chains start there.

#### Approach C: PC2-Filtered Vote Selection

Instead of modifying the model, pre-filter the vote matrix. PCA produces per-bill
loadings on each component. Keep only bills where `|PC2 loading| > |PC1 loading|`
— votes that discriminate more on ideology than on establishment-loyalty.

This follows the same pattern as our existing `--contested-only` flag, which
filters to cross-party contested votes. The difference: contested-only selects
votes by party-split patterns, while PC2 filtering selects votes by which
dimension they measure. The two are complementary — a bill can be contested
(parties split) *and* load primarily on PC1 (establishment-loyalty).

Caughey & Schickler (2016) advocate using specific subsets of votes according to
historical context. The risk is reduced sample size: if filtering removes too many
votes, the model loses statistical power. With only ~40 senators, every vote
matters.

#### Approach D: Projective IRT (2D → 1D)

Already available. Fit the 2D IRT model, then extract Dimension 1 as the
canonical ideology score. This is what the `--promote-2d` flag does. The 2D model
correctly identifies which dimension is ideology (Dim 1 correlates at r = 0.95
with PC2). Extracting Dim 1 gives a 1D score that's grounded in the right axis.

This isn't technically "nudging the 1D model" — it's abandoning the 1D model and
using a 2D model with a 1D projection. But the result is the same: a single
ideology score per legislator that correctly separates Democrats from
conservative rebels.

The drawback: the 2D model has convergence issues for the 79th (R-hat up to 1.96,
ESS as low as 5). The projected Dim 1 scores are directionally correct but have
wide uncertainty. Approaches A–C attempt to get the 1D model — which converges
cleanly — to recover the right dimension directly.

#### Experimental Results (2026-03-09)

We tested all four mechanisms on both the Senate (horseshoe case) and House
(no horseshoe) of the 79th biennium. Full results:
`results/experimental_lab/2026-03-09_pc2-targeted-irt/`.

**Senate** (ground truth r(PC2) = 0.949, 0% D wrong side):

| Variant | r(PC2) | D wrong | R-hat | ESS | Verdict |
|---------|--------|---------|-------|-----|---------|
| baseline | 0.079 | 80% | 1.017 | 263 | Locked on PC1 |
| pc2_init | 0.080 | 80% | 1.005 | 288 | Init alone does nothing |
| pc2_prior (σ=1.0) | −0.054 | 20% | 1.020 | 243 | Flips sign, stays on PC1 |
| pc2_prior (σ=0.5) | 0.042 | 20% | 1.005 | 529 | Same — prior can't redirect |
| **pc2_filtered** | **0.815** | **0%** | 1.012 | 425 | Finds ideology axis |
| **combined** | **0.842** | **0%** | **1.004** | **704** | **Best: ideology + convergence** |

**House** (no swapped dimensions — baseline works fine):

| Variant | r(PC2) | D wrong | R-hat | ESS | Verdict |
|---------|--------|---------|-------|-----|---------|
| baseline | 0.117 | 0% | 1.006 | 307 | Correct — PC1 is ideology |
| pc2_filtered | −0.905 | 29% | 1.007 | 174 | Redirects but hurts party separation |
| combined | 0.904 | 73% | 1.009 | 229 | Damages what was working |

**Three key findings:**

1. **Vote filtering is the essential mechanism.** Initialization and informative
   priors alone cannot overcome the data's pull toward PC1. The prior flips the
   sign (correct R > D) but the dimension remains establishment-loyalty, not
   ideology. Only when the vote matrix is restricted to PC2-dominant bills
   (172/437 for Senate) does the model find the right axis.

2. **Combined approach is the winner.** PC2 prior + filtered votes achieves the
   highest r(PC2) = 0.842, the cleanest convergence (R-hat 1.004, ESS 704),
   and zero Democrats on the wrong side. The prior adds value *on top of*
   filtering — it improves convergence and nudges r(PC2) from 0.815 to 0.842.

3. **Remediation must be chamber-specific.** The House does not have swapped
   dimensions — its PC1 is already ideology. Applying PC2 targeting to the
   House damages results. Any production implementation must be gated on
   horseshoe detection: only chambers that fail the diagnostic get redirected.

#### Production Implementation: `--horseshoe-remediate`

The combined approach is now available as the `--horseshoe-remediate` CLI flag
(ADR-0104). When enabled, the IRT pipeline automatically:

1. Runs `detect_horseshoe()` on each chamber (implies `--horseshoe-diagnostic`)
2. If horseshoe detected: loads PCA per-bill loadings from Phase 02
3. Filters votes to PC2-dominant bills via `filter_pc2_dominant_votes()`
   (keeps only bills where |PC2 loading| > |PC1 loading|)
4. Builds a PC2 informative prior: `xi ~ Normal(PC2_score, 1.0)`
5. Refits the 1D IRT with the `external-prior` strategy
6. Reports whether the horseshoe was resolved

If no horseshoe is detected, the chamber uses standard IRT unchanged. This
means the flag is safe to enable globally — it only fires for affected
chambers.

```bash
just irt 2001-02 --horseshoe-remediate   # remediate the 79th biennium
```

## Summary

| Approach | Effort | Fixes Horseshoe? | Status |
|----------|--------|-------------------|--------|
| Auto-promote 2D | Low | Yes (sidesteps it) | **Recommended** |
| Contested-only default | Low | Partially | Viable |
| **PC2-targeted 1D IRT** | **Low** | **Yes (stays 1D)** | **Confirmed — combined approach wins** |
| L1-based model (Shin 2025) | High | Yes (fundamentally) | R package path viable |
| External anchoring | Medium | Yes (when data exist) | Untested |
| Regularized horseshoe prior | Medium | No | **Ruled out** |
| Cross-session borrowing | High | Yes | Untested |

The horseshoe effect is not a bug in our code. It is a fundamental limitation
of compressing a multidimensional political reality onto a single number line.
The most practical near-term solution is auto-promoting 2D results when
horseshoe diagnostics trigger. The L1 model remains theoretically attractive
but requires the authors' R implementation rather than a PyMC approximation.

## Experiments

Three experiments are planned to evaluate the most promising approaches:

- `results/experimental_lab/2026-03-08_regularized-horseshoe/` — Approach 5: adaptive shrinkage prior on discrimination parameters
- `results/experimental_lab/2026-03-08_l1-ideal-point/` — Approach 3: L1-based ideal point model (Shin et al. 2025)
- `results/experimental_lab/2026-03-08_supermajority-auto-promote/` — Approaches 1 & 2: audit all sessions for horseshoe risk, test 2D promotion and contested-only defaults
- `results/experimental_lab/2026-03-09_pc2-targeted-irt/` — PC2-targeted 1D IRT: informative prior, initialization, and vote filtering to recover the ideology dimension directly

## References

- Shin, S., Lim, J., & Park, J. H. (2025). L1-based Bayesian ideal point
  model for multidimensional politics. *Journal of the American Statistical
  Association*, 120(550), 631–644.
- Piironen, J. & Vehtari, A. (2017). Sparsity information and regularization
  in the horseshoe and other shrinkage priors. *Electronic Journal of
  Statistics*, 11(2), 5018–5051.
- Poole, K. T. & Rosenthal, H. (1997). *Congress: A Political-Economic
  History of Roll Call Voting*. Oxford University Press.
- Clinton, J., Jackman, S., & Rivers, D. (2004). The statistical analysis of
  roll call data. *American Political Science Review*, 98(2), 355–370.
- Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe
  estimator for sparse signals. *Biometrika*, 97(2), 465–480.
- Betancourt, M. (2018). Bayes sparse regression (case study).
  https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html
- Zhang, Y. D., Naughton, B. P., Bondell, H. D., & Reich, B. J. (2022).
  Bayesian regression using a prior on the model fit: The R2-D2 shrinkage
  prior. *Journal of the American Statistical Association*, 117(538), 862–874.
- Caughey, D. & Schickler, E. (2016). Substance and change in congressional
  ideology: NOMINATE and its alternatives. *Studies in American Political
  Development*, 30(2), 128–146.
- Morucci, M., Foster, M., Webster, S., Lee, D. & Siegel, A. (2024). IRT-M:
  Inductive discovery of multi-dimensional ideology. *American Political
  Science Review*. [arxiv:2111.11979](https://arxiv.org/abs/2111.11979)
- Strachan, D. P., et al. (2024). Projective IRT for multidimensional
  assessment. *Multivariate Behavioral Research*.

## Resolution: Canonical Ideal Points

After extensive experimentation with 1D fixes (`--init-strategy 2d-dim1`, `--dim1-prior`, `--horseshoe-remediate`), we concluded that the field-standard solution is correct: **use 2D IRT Dimension 1 as the canonical ideology score** for horseshoe-affected chambers. This is exactly what DW-NOMINATE has done for 40 years — fit a 2D model and extract Dim 1. See `docs/canonical-ideal-points.md` for the full narrative of our journey through three failed approaches and the conclusion.

## Related

- `docs/canonical-ideal-points.md` — **Definitive article:** from 1D fixes to 2D Dim 1 promotion
- `docs/canonical-ideal-points-implementation-plan.md` — Implementation plan for canonical routing
- `docs/dim1-informative-prior.md` — Dim 1 informative prior approach (superseded by canonical routing)
- `docs/79th-horseshoe-robustness-analysis.md` — Empirical diagnostic results
- `docs/adr/0104-irt-robustness-flags.md` — Robustness flags ADR
- `docs/adr/0103-irt-identification-strategy-system.md` — Identification strategies ADR
- `docs/how-irt-works.md` — IRT methodology overview
- `docs/irt-identification-strategies.md` — Detailed strategy documentation
- `analysis/design/irt.md` — IRT design document
- `analysis/design/irt_2d.md` — 2D IRT design document (PLT identification, experimental status)
