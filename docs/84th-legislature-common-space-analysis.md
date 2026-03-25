# The 84th Legislature: Why the Common-Space Chain's Weakest Link Breaks

**Date:** 2026-03-25
**Scope:** 84th Kansas Legislature (2011-2012), Phases 05-07b, 28, 30
**Related:** ADR-0118, ADR-0120, ADR-0122, `docs/pca-ideology-axis-instability.md`, `docs/84th-biennium-analysis.md`

---

## Summary

The 84th Legislature is the common-space chain's weakest link — not because of data quality problems, but because of a **canonical routing error**. The hierarchical 2D IRT model (Phase 07b) was selected as the canonical source despite producing scores that correlate poorly with W-NOMINATE (r=0.392 Senate, r=0.838 House). The flat 2D IRT model (Phase 06) produces far better agreement (r=0.918 Senate, r=0.968 House). The root cause: the hierarchical model's party-pooling prior distorts the first dimension in a chamber where **intra-Republican factionalism dominates the party divide**. This is the only session in the dataset where the hierarchical model is demonstrably worse than the flat model.

---

## Political Context: A Party at War with Itself

The 84th Legislature (2011-2012) captures the Kansas Republican Party in the middle of a historic factional split. Governor Sam Brownback, elected in 2010, brought an aggressive agenda — massive income tax cuts (the "Kansas experiment"), abortion restrictions, and Medicaid restructuring. The Republican caucus fractured into two camps:

- **Conservative faction** (aligned with Brownback, AFP, Kansas Chamber): supply-side tax cuts, reduced state spending, social conservatism
- **Moderate faction** (led by Senate President Steve Morris): fiscal pragmatism, education funding, coalition governance with Democrats

The Senate was ground zero. With 30 Republicans and 10 Democrats, moderates held the balance of power by allying with Democrats on key votes. Party-line votes were rare (2.4% vs 22.4% in the 91st). The Republican caucus had 5.4x more internal ideological variation than Democrats. Senate assortativity was 0.188 — nearly random — meaning party membership was a weak predictor of voting behavior.

**The August 2012 purge** eliminated this faction. Conservative groups funded primary challenges against moderate senators. Steve Morris, Tim Owens, and approximately seven other moderates were defeated. The 85th Legislature (2013-2014) had dramatically higher Republican cohesion. This is why the 84th-85th bridge is the weakest in the dataset (89 bridges, 62.7% overlap).

---

## Per-Session Model Performance

### The problem: 1D IRT fails completely

Flat 1D IRT for the 84th is one of the worst-performing models in the entire pipeline:

| Metric | House | Senate |
|--------|-------|--------|
| R-hat max | 1.832 | 1.828 |
| ESS min | 2.88 | 2.89 |
| Party separation d | 1.42 | **0.02** |
| Sign flipped | Yes | Yes |
| axis_uncertain | True | True |

The Senate party separation of d=0.02 means the 1D model found essentially **zero ideological difference between parties**. This is correct — in a 1D projection, the moderate-conservative Republican split dominates, and both parties' centers collapse to the middle.

### The solution: 2D IRT recovers ideology

Both the flat 2D (Phase 06) and hierarchical 2D (Phase 07b) models converge:

| Model | Chamber | R-hat | ESS | Converged |
|-------|---------|-------|-----|-----------|
| Flat 2D | House | 1.026 | 139 | Yes |
| Flat 2D | Senate | 1.033 | 141 | Yes |
| Hierarchical 2D | House | 1.016 | 261 | Yes (Tier 1) |
| Hierarchical 2D | Senate | 1.015 | 694 | Yes (Tier 1) |

Both converge cleanly. But convergence does not mean correctness.

### The critical finding: which Dim 1 is right?

Cross-validating each model's Dim 1 against W-NOMINATE Dim 1 (the field-standard unsupervised estimator):

| Model | House r | Senate r |
|-------|---------|----------|
| 1D IRT | 0.583 | 0.288 |
| **Flat 2D Dim 1** | **0.968** | **0.918** |
| Hierarchical 2D Dim 1 | 0.838 | 0.392 |

The flat 2D model outperforms the hierarchical model in both chambers, and the gap is enormous in the Senate (0.918 vs 0.392). Within-party correlations for the hierarchical Senate are essentially zero (Republican r=-0.137, Democrat r=0.060).

---

## Root Cause: Party Pooling vs. Factionalism

The hierarchical 2D IRT model (Phase 07b, ADR-0117) uses a **party-pooling prior**: each party's legislators share a hierarchical mean and within-party variance. This works brilliantly in most sessions because party membership IS the dominant predictor of ideology.

In the 84th Senate, party membership is a **weak predictor** (ICC=0.502, assortativity=0.188). The dominant dimension of conflict is the moderate-conservative Republican factional split. The party-pooling prior forces the model to emphasize the party divide, which is the **second-most-important** dimension, not the first. The result: Dim 1 captures a party signal that exists but is not the primary axis of variation, while the true ideology dimension (which cross-cuts the party divide within Republicans) is relegated to Dim 2 or mixed across both dimensions.

W-NOMINATE and the flat 2D IRT model are **unsupervised** — they identify the principal axes of variation without any party information. For the 84th Senate, these unsupervised methods correctly place the factional split on Dim 1.

### Why the routing logic was fooled

The canonical routing system (ADR-0109, ADR-0110) selects the hierarchical model when:
1. Horseshoe detected (yes — intra-R factionalism triggers this correctly)
2. H2D converges at Tier 1 (yes — R-hat 1.015, ESS 694)
3. Party separation d > 1.5 (likely passed — the party-pooling prior *creates* party separation)

The quality gates check convergence and party separation, but they don't check **whether the separated dimension is the right one**. The party-pooling prior guarantees some party separation on Dim 1 by construction, making the party-d gate easy to pass even when the dimension doesn't match the unsupervised ideology axis.

---

## Impact on Common-Space Linking

The incorrect canonical source for the 84th propagates through the common-space chain:

1. **Per-session scores are distorted**: The 84th Senate canonical scores don't align with the ideological continuum. Bridge legislators' positions in the 84th are measured on a different axis than in the 85th.

2. **The 84th-85th affine link is unreliable**: The linking algorithm estimates A and B from bridge legislators' positions in adjacent sessions. If the 84th measures factionalism and the 85th measures ideology, the affine fit is trying to map between incommensurable scales.

3. **All pre-84th sessions are affected**: The chain composes links backward from the 91st. Any error at the 84th-85th link propagates to the 78th-83rd.

This explains the per-session common-space correlation for the 84th (r=0.564 between IRT and W-NOMINATE common-space scores, worst in the dataset). It's not the linking algorithm that failed — it's the input scores.

---

## The Dimension Swap: Ideology Is on Dim 2, Not Dim 1

The dimension swap detection (`dimension_swap_corrected = YES` in the convergence summary) fired for both chambers — but it swapped the **wrong way**. A comprehensive comparison of all available IRT dimensions against W-NOMINATE Dim 1 reveals the error:

| Model | House r | House rho | Senate r | Senate rho |
|-------|---------|-----------|----------|------------|
| 1D IRT | 0.617 | 0.865 | 0.288 | 0.578 |
| Flat 2D Dim 1 | 0.968 | 0.984 | 0.918 | 0.893 |
| Flat 2D Dim 2 | 0.520 | 0.324 | -0.812 | -0.754 |
| H2D Dim 1 (current canonical) | 0.838 | 0.582 | 0.392 | 0.377 |
| **H2D Dim 2** | **0.977** | **0.984** | **0.992** | **0.966** |

**H2D Dim 2 correlates r=0.992 with W-NOMINATE in the Senate** — near-perfect agreement. Within Republicans, the correlation is r=0.994 (n=30). This is the ideology dimension. What the routing system labeled "Dim 1" actually captures the party-pooling prior's forced party separation, not the natural ideology axis.

The hierarchical model's Dim 2 is the **best available option** for the 84th: it has the hierarchical model's superior convergence properties (ESS 261 House, 694 Senate) AND near-perfect agreement with the field-standard unsupervised estimator. The flat 2D Dim 1 (r=0.918/0.968) is the second-best option.

### Why the swap detection failed

The dimension swap detection uses PCA correlation: it checks which IRT dimension aligns better with PC1. But in the 84th, PCA itself has ambiguous axes (PC1 d=1.84, PC2 d=1.30 in the Senate). When PCA can't reliably identify the ideology axis, the swap detection inherits that ambiguity. Using W-NOMINATE as the cross-validation target instead of PCA would catch this case.

---

## Recommendation: Use H2D Dim 2 for the 84th

The primary fix: **override the canonical source for the 84th to use H2D Dim 2 instead of H2D Dim 1.**

This can be implemented as a W-NOMINATE cross-validation gate in the routing logic:

> After selecting the canonical dimension, check its Pearson correlation with W-NOMINATE Dim 1 (when available). If the OTHER dimension of the same model correlates substantially better (delta r > 0.15), swap dimensions. This catches cases where the PCA-based swap detection fails due to ambiguous axes.

This gate would fire for the 84th (and should be checked for the 88th Senate, which also shows PCA axis instability). All other sessions would be unaffected.

### Expected improvement

After switching to H2D Dim 2 for the 84th:
- Per-session Senate IRT-WNOM correlation: 0.392 → **0.992**
- Per-session House IRT-WNOM correlation: 0.838 → **0.977**
- Common-space 84th correlation: expected to improve dramatically
- The 84th-85th bridge link will be far more reliable
- All pre-84th sessions (78th-83rd) will benefit from the improved chain link

---

## Other 84th Characteristics (Not Bugs)

These are **features, not bugs** — they reflect the genuine political complexity of the session:

- **Weakest bridge coverage (62.7%)**: The 2012 purge + redistricting produced the largest turnover. Still well above the 20-bridge psychometric minimum (89 bridges).
- **Ambiguous PCA axes**: Neither PC1 nor PC2 clearly dominates party separation (d=1.84 vs 1.30). This is the transition point between the PC2-ideology era (78th-83rd) and the PC1-ideology era (85th+).
- **ODT-era data limitations**: 29.4% of vote pages are committee-of-the-whole tallies without individual names. This reduces the number of votes but doesn't bias the ones we have.
- **Supermajority adaptive tuning**: The 2D IRT phase automatically doubled N_TUNE from 2000 to 4000 for this session (81% R), per ADR-0112.

---

## The Bigger Picture: When Party Pooling Helps and Hurts

The 84th teaches a general lesson about hierarchical models in political science:

**Party pooling helps when party structure is strong.** In the 91st Legislature (2025-2026), party membership explains most of the ideological variance. The hierarchical prior correctly shrinks within-party estimates toward the party mean, improving precision.

**Party pooling hurts when party structure is weak.** In the 84th Senate (2011-2012), the Republican caucus spans the entire ideological spectrum. The hierarchical prior pulls moderate Republicans toward the Republican mean and conservative Republicans toward the same mean, compressing the factional variation that IS the dominant political dimension. The result is a first dimension that reflects the model's prior, not the data.

This is analogous to the "ecological fallacy" in political science — assuming group-level patterns (party = ideology) hold at the individual level. In the 84th Senate, they don't.

The flat 2D model avoids this problem because it has no party information. Like W-NOMINATE and DW-NOMINATE, it identifies dimensions purely from voting patterns. For sessions where party structure is ambiguous, unsupervised methods are more reliable.

---

## References

- Poole, K. T., & Rosenthal, H. (1997). *Congress: A Political-Economic History of Roll Call Voting*. Oxford.
- Clinton, J., Jackman, S., & Rivers, D. (2004). The statistical analysis of roll call data. *APSR* 98(2): 355-370.
- Shor, B., & McCarty, N. (2011). The ideological mapping of American legislatures. *APSR* 105(3): 530-551.
- Brownback, S. (2012). Interview with AP: described the tax cuts as "a real live experiment."
- Kansas City Star (2012). Coverage of the August 2012 Republican primary results.
