# W-NOMINATE vs IRT Common Space: Divergence Analysis

**Date:** 2026-03-25
**Scope:** Phase 28 (IRT Common Space) vs Phase 30 (W-NOMINATE Common Space)
**Related:** ADR-0120, ADR-0122, `docs/common-space-ideal-points.md`, `docs/wnominate-common-space.md`

---

## Executive Summary

The cross-method validation (Phase 30) compares career scores from two independent common-space methods — Bayesian IRT and W-NOMINATE — across 696 matched legislators spanning 14 bienniums (1999-2026). Overall agreement is strong (Pearson r = 0.961, Spearman rho = 0.950), but 25 legislators show rank disagreements exceeding 100 positions. This analysis identifies three distinct causes: identity resolution artifacts (fixable), scale compression (methodological, expected), and a genuinely problematic session (the 84th Legislature).

---

## The Data

| Metric | Value |
|--------|-------|
| Matched legislators | 696 |
| Pearson r (overall) | 0.961 |
| Spearman rho (overall) | 0.950 |
| Republican within-party r | 0.857 |
| Democrat within-party r | 0.862 |
| IRT score range | [-4.50, +3.87] |
| W-NOMINATE score range | [-1.05, +1.31] |
| IRT standard deviation | 1.508 |
| W-NOM standard deviation | 0.581 |

The IRT common space has 2.6x the variance of the W-NOMINATE common space. This ratio is not constant across sessions — it grows from ~2.5 for early bienniums to ~5.8 for the reference session (91st). This variance amplification is the dominant source of score-level divergence.

---

## Three Causes of Divergence

### Cause 1: Identity Resolution Artifacts (Fixable)

Five of the top 25 divergent legislators have **slug-based** person_keys rather than OCD-based keys. These are chamber-switch orphans where the same legislator got two different identity keys, splitting their career into fragments with incomplete voting histories.

| Legislator | Rank Diff | Root Cause |
|-----------|-----------|------------|
| Caryn Tyson | 329 | `rep_tyson_caryn_1` (84th House, 1 session) split from `ocd-person/cc84252c-...` (85th-91st Senate, 7 sessions). IRT career score for the orphan is -0.376 (based on one House session); her actual 7-session Senate career score is +2.240. |
| Ray Merrick | 432 | `merrick_ray_1` (84th House, 1 session) split from OCD-based Senate entry. Same pattern as Tyson. |
| Greg Smith | 184 | `smith_greg_1` (84th House) split from Senate entry. |
| Mary Pilcher Cook | 216 | `pilcher_cook_mary_1` — slug-based fallback in early sessions. |
| Shari Weber | 229 | `weber_shari_1` — similar pattern. |

**Status:** Fixed by ADR-0122 (cross-chamber OCD expansion + duplicate detection quality gate). Phases 28 and 30 need re-running to pick up the fix.

Additionally, **Charles Roth** (rank diff 405, IRT=+2.068 vs W-NOM=-0.093) is caused by a **slug encoding variant**: the actual legislator slug is `rep_roth_charlie_1` (701 votes) but the common-space data references `rep_roth_charles_1` (0 votes). The `_SLUG_OVERRIDES` table already maps `roth_charles_1 → roth_charlie_1` for KanFocus data, but this mapping needs to be applied consistently in the upstream W-NOMINATE and IRT phases.

### Cause 2: Scale Compression (Methodological, Expected)

W-NOMINATE constrains all legislators to the unit hypersphere ([-1, +1] per dimension). Bayesian IRT has an unbounded latent trait. This creates a **nonlinear relationship at the tails**: legislators near the W-NOMINATE boundary are compressed relative to their IRT scores.

The affine linking (which assumes a linear relationship) is estimated on the interior of the distribution where both methods agree, but extrapolates poorly at the extremes. The result is systematic:

- **Conservative Republicans**: W-NOMINATE compresses them against +1; IRT lets them spread to +3 or +4. After linking, the W-NOMINATE career score understates their conservatism relative to IRT.
- **Liberal Democrats**: Same pattern at -1; IRT extends to -4.5 while W-NOMINATE caps at -1.

This explains why the top divergent Republicans (Josh Powell, Amanda Grosserode, Brett Hildabrand) all show W-NOM > IRT in rank but W-NOM < IRT in score — they're compressed at the W-NOMINATE boundary.

**This is expected behavior, not a bug.** The two methods operationalize ideology differently. W-NOMINATE's bounded scale prevents extreme scores by design; IRT's unbounded scale allows them. Neither is "wrong" — they answer slightly different questions.

### Cause 3: The 84th Legislature (Genuine Problem)

The 84th Legislature (2011-2012) has by far the worst per-session agreement between the two methods:

| Session | N | Pearson r | Spearman rho | MAE | IRT/WN Std Ratio |
|---------|---|-----------|-------------|-----|-------------------|
| 83rd (2009-10) | 170 | 0.807 | 0.847 | 0.97 | 2.79 |
| **84th (2011-12)** | **150** | **0.564** | **0.453** | **1.42** | **3.29** |
| 85th (2013-14) | 156 | 0.679 | 0.537 | 1.66 | 3.93 |

The Spearman rho of 0.453 means the **rank ordering** disagrees substantially — this is not just a scaling issue. The 84th is the session immediately after the 2012 redistricting and the "Brownback purge" that eliminated moderate Republican senators. It represents a structural break in Kansas politics with:

- **Massive turnover**: 62.7% bridge coverage (weakest in the dataset)
- **Extreme supermajority**: Republican dominance so large that intra-party variation dominates
- **Potential horseshoe effects**: The canonical routing may select 2D Dim 1 or 1D IRT differently for this session

The 84th's poor agreement propagates through the chain: it is one link in the 13-link chain from 78th to 91st, and any error at this link affects all earlier sessions' aligned scores.

---

## Per-Session Correlation Table

The full per-session breakdown shows how agreement varies across the 28-year span:

| Session | N | r | rho | MAE | IRT Std | WN Std | Ratio |
|---------|---|---|-----|-----|---------|--------|-------|
| 78th (1999-00) | 167 | 0.740 | 0.630 | 0.74 | 1.18 | 0.46 | 2.58 |
| 79th (2001-02) | 168 | 0.715 | 0.640 | 0.82 | 1.25 | 0.51 | 2.45 |
| 80th (2003-04) | 171 | 0.768 | 0.750 | 0.88 | 1.36 | 0.55 | 2.48 |
| 81st (2005-06) | 171 | 0.776 | 0.801 | 0.95 | 1.45 | 0.58 | 2.52 |
| 82nd (2007-08) | 166 | 0.779 | 0.801 | 0.96 | 1.52 | 0.57 | 2.66 |
| 83rd (2009-10) | 170 | 0.807 | 0.847 | 0.97 | 1.60 | 0.57 | 2.79 |
| 84th (2011-12) | 150 | 0.564 | 0.453 | 1.42 | 1.99 | 0.61 | 3.29 |
| 85th (2013-14) | 156 | 0.679 | 0.537 | 1.66 | 2.43 | 0.62 | 3.93 |
| 86th (2015-16) | 169 | 0.805 | 0.798 | 1.74 | 2.52 | 0.61 | 4.09 |
| 87th (2017-18) | 171 | 0.829 | 0.928 | 1.61 | 2.72 | 0.62 | 4.37 |
| 88th (2019-20) | 168 | 0.720 | 0.722 | 2.28 | 3.59 | 0.66 | 5.44 |
| 89th (2021-22) | 169 | 0.788 | 0.801 | 2.40 | 3.93 | 0.71 | 5.54 |
| 90th (2023-24) | 168 | 0.786 | 0.748 | 2.54 | 4.07 | 0.72 | 5.65 |
| 91st (2025-26) | 173 | 0.787 | 0.680 | 2.74 | 4.12 | 0.71 | 5.78 |

**Key pattern**: The IRT/W-NOM standard deviation ratio grows monotonically from 2.5 to 5.8. This is the chain propagation effect — each link's affine coefficient (A) is >1 for IRT (stretching the wider scale) and <1 for W-NOMINATE (compressing the bounded scale), and these compound multiplicatively through 13 links.

The MAE also grows monotonically, but this is primarily driven by the variance ratio — the two methods are on increasingly different absolute scales as you move further from the reference.

---

## W-NOMINATE Senate Quality Gate Failures

Four Senate sessions fail W-NOMINATE common-space quality gates:

| Session | Party d | Sign | Issue |
|---------|---------|------|-------|
| 78th (1999-00) | 5.41 | Flipped | W-NOMINATE places Democrats right of Republicans |
| 79th (2001-02) | 1.15 | Flipped | Weak separation + sign flip |
| 80th (2003-04) | 1.13 | OK | Party separation below 1.5 threshold |
| 82nd (2007-08) | 1.41 | OK | Party separation below 1.5 threshold |

These are the same sessions affected by the PCA axis instability documented in ADR-0118 and `docs/pca-ideology-axis-instability.md`. In these early Senate sessions, the moderate-vs-conservative Republican factional split dominated the primary dimension, pushing the party divide to a secondary dimension. W-NOMINATE's Dim 1 captures the same faction split, producing either sign flips or weak party separation.

The IRT common space handles this via canonical routing (using Hierarchical 2D Dim 1 for horseshoe-affected sessions). W-NOMINATE does not have an equivalent correction — the sign flip is applied post-hoc at the session level, but if Dim 1 genuinely captures factionalism rather than partisanship, flipping the sign doesn't fix the underlying issue.

---

## Implications

### What the divergences tell us

1. **The correlation of 0.961 is strong evidence of convergent validity.** Two fundamentally different estimation methods (logistic IRT vs geometric spatial model), linked by the same chain methodology, agree on the rank ordering of Kansas legislators. This means the common-space scores are measuring something real, not an artifact of either method.

2. **The tail divergences are predictable from first principles.** W-NOMINATE's bounded scale compresses extremes; IRT's unbounded scale extends them. Researchers should choose based on their question: W-NOMINATE for comparability with the national literature (DW-NOMINATE uses the same scale), IRT for maximum discrimination among Kansas legislators.

3. **The 84th Legislature requires caution.** Any cross-temporal analysis that passes through the 84th-85th link (i.e., comparing pre-2012 to post-2012 legislators) should be interpreted with wider uncertainty than the CIs suggest. The two methods disagree most at this structural break.

4. **Identity resolution bugs inflate the divergence statistics.** After re-running with ADR-0122 fixes, several of the "top 25 divergent" legislators will disappear from the list as their split identities are merged into correct career scores.

### Recommended actions

1. **Re-run Phases 28 and 30** after the ADR-0122 identity resolution fix is confirmed working
2. **Investigate the Charlie/Charles Roth slug encoding** in the upstream Phase 16 W-NOMINATE output — this is a per-session data quality issue, not a common-space issue
3. **Consider a nonlinear linking model** for the W-NOMINATE common space (e.g., rank-based equating or kernel equating from Kolen & Brennan 2014) that can handle the boundary compression
4. **Add the 84th Legislature as a "fragile link" annotation** to common-space reports, similar to the horseshoe warnings

---

## Technical Background

### Why do IRT and W-NOMINATE disagree?

The two methods differ in three fundamental ways:

**1. Likelihood model.** IRT uses a logistic link: `P(yea) = logit^{-1}(beta * theta - alpha)`. W-NOMINATE uses a Gaussian utility function: `P(yea) = f(exp(-d(x, yea_point)^2 / 2sigma^2))` where d is the Euclidean distance between the legislator's ideal point and the bill's "yea" outcome point. The logistic and Gaussian CDFs are similar but not identical, producing different orderings for legislators whose voting patterns are ambiguous.

**2. Scale constraints.** IRT's latent trait is unbounded; identification comes from anchoring (anchor-PCA, anchor-agreement, etc.). W-NOMINATE constrains legislators to [-1, +1] per dimension; identification comes from the boundary itself. This means W-NOMINATE has less estimation uncertainty at the extremes (legislators are "pinned" to the boundary) but less discrimination (all extreme legislators look the same).

**3. Dimensionality treatment.** The canonical IRT routing system (ADR-0109) selects between 1D IRT, flat 2D IRT Dim 1, and hierarchical 2D IRT Dim 1 depending on horseshoe status. W-NOMINATE estimates Dim 1 and Dim 2 jointly in a 2D spatial model. The two methods may extract different first dimensions, especially in supermajority sessions where the party divide is not the dominant axis of variation.

### Why does the variance ratio grow through the chain?

Consider two sessions linked by `xi_ref = A * xi_source + B`. The variance transforms as `Var(xi_ref) = A^2 * Var(xi_source)`. If A > 1, the variance inflates; if A < 1, it compresses.

For IRT common space, each pairwise A coefficient is close to 1 (the IRT scale is unbounded and doesn't compress), so `A_total = A_1 * A_2 * ... * A_13 ≈ 1`. The variance grows slowly through the chain.

For W-NOMINATE common space, each pairwise A coefficient is also close to 1 within the [-1, +1] scale. But after affine transformation to the reference session's scale, the variance is mapped to the reference session's variance — which for IRT is 2.6x larger than for W-NOMINATE. The ratio compounds because the IRT chain stretches the scale while the W-NOMINATE chain preserves it.

The result: the IRT common space spreads legislators across a wider range as you move further from the reference, while the W-NOMINATE common space maintains a narrower range. Career scores (which average across sessions) inherit this divergence.

---

## References

- Battauz, M. (2023). A general framework for chain equating under the IRT paradigm. *Psychometrika* 88(4): 1260-1287.
- Clinton, J., Jackman, S., & Rivers, D. (2004). The statistical analysis of roll call data. *APSR* 98(2): 355-370.
- Groseclose, T., Levitt, S. D., & Snyder, J. M. (1999). Comparing interest group scores across time and chambers. *APSR* 93(1): 33-50.
- Kolen, M. J., & Brennan, R. L. (2014). *Test Equating, Scaling, and Linking* (3rd ed.). Springer.
- Poole, K. T., & Rosenthal, H. (1997). *Congress: A Political-Economic History of Roll Call Voting*. Oxford.
