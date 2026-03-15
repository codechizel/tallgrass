# PCA Ideology Axis Instability in the Kansas Senate

**Date:** 2026-03-14
**Sparked by:** Reviewing the 79th (2001-2002) PCA report, sections 6 and 26 (House and Senate score scatter matrices)

---

## The Observation

In the 79th Senate PCA scatter matrix, PC1 shows no party separation whatsoever — Republicans and Democrats are completely intermixed. But PC2 shows textbook party separation, with Democrats clustered at the negative end and Republicans at the positive end. This is the opposite of the House, where PC1 is clearly the ideology axis.

The initial question: is PC2, not PC1, the ideology factor for the 79th Senate?

## The Answer: Yes, and It's Not a One-Off

Measuring party separation with Cohen's d (effect size: how many pooled standard deviations apart are the party means), we find that **ideology lands on PC2 in 7 of 14 sessions** for the Kansas Senate:

| Session | Senate R% | House PC1 d | Senate PC1 d | Senate PC2 d | Senate ideology on |
|---------|-----------|-------------|--------------|--------------|-------------------|
| 78th (1999-2000) | 68% | 4.97 | 2.11 | 2.56 | PC2 |
| **79th (2001-2002)** | **75%** | **5.37** | **0.28** | **4.98** | **PC2** |
| 80th (2003-2004) | 72% | 3.48 | 0.27 | 2.56 | PC2 |
| 81st (2005-2006) | 75% | 3.91 | 1.72 | 2.25 | PC2 |
| 82nd (2007-2008) | 75% | 3.88 | 0.89 | 2.41 | PC2 |
| 83rd (2009-2010) | 78% | 3.30 | 0.79 | 4.40 | PC2 |
| 84th (2011-2012) | 81% | 5.37 | 1.84 | 1.30 | ambiguous |
| 85th (2013-2014) | 79% | 4.77 | 6.83 | 0.19 | PC1 |
| 86th (2015-2016) | 80% | 3.84 | 5.69 | 0.09 | PC1 |
| 87th (2017-2018) | 71% | 2.66 | 2.10 | 1.87 | PC1 |
| 88th (2019-2020) | 73% | 8.74 | 1.72 | 3.10 | PC2 |
| 89th (2021-2022) | 70% | 7.20 | 7.21 | 0.60 | PC1 |
| 90th (2023-2024) | 72% | 6.69 | 6.75 | 0.38 | PC1 |
| 91st (2025-2026) | 75% | 7.30 | 7.49 | 0.14 | PC1 |

**The 79th is the most extreme case**: Senate PC1 d = 0.28 (essentially zero party separation) while PC2 d = 4.98 (massive separation). The House is completely normal in every session (PC1 d always > 2.5).

The pattern shifts around the 84th-85th (2011-2014). Before that, the Senate almost always has ideology on PC2. After, it's consistently on PC1. The 88th is a late-era exception.

## Why This Happens

PCA extracts components in order of **variance explained**, not in order of **substantive meaning**. In the Kansas Senate:

- **Before ~2013**: The largest source of variance in roll-call votes was intra-Republican factional disagreement (moderate establishment vs. conservative insurgent). With 70-78% Republicans, this intra-caucus axis dominates total variance. The party divide, while sharp, explains less variance because there are fewer Democrats to contribute to it. So PCA captures the intra-R axis as PC1 (19.6% variance in the 79th) and the party axis as PC2 (13.6%).

- **After ~2013**: Something changed in Kansas politics — possibly increased polarization, possibly reduced intra-R factionalism — such that the party divide became the dominant source of variance again. PC1 recaptured ideology.

The eigenvalue ratios confirm this. In the 79th Senate, λ₁/λ₂ = 1.45 — barely above 1, meaning the first two components capture similar amounts of variance and could easily swap. In the 91st Senate, λ₁/λ₂ = 4.42 — PC1 dominates overwhelmingly.

## The Downstream Problem: 1D IRT Ideal Points

This isn't just a PCA display issue. It propagates into the IRT models. Checking IRT ideal point correlation with PCA components:

| Session | Senate PC1 d | Senate PC2 d | IRT ρ(PC1) | IRT ρ(PC2) | IRT party d | Assessment |
|---------|-------------|-------------|------------|------------|-------------|------------|
| 79th | 0.28 | 4.98 | 0.974 | -0.067 | 0.86 | IRT on wrong axis |
| 80th | 0.27 | 2.56 | 0.898 | -0.331 | 0.59 | IRT on wrong axis |
| 82nd | 0.89 | 2.41 | 0.954 | 0.534 | 1.26 | IRT on wrong axis |
| 83rd | 0.79 | 4.40 | 0.911 | 0.500 | 1.69 | IRT on wrong axis |
| 91st | 7.49 | 0.14 | 0.751 | -0.335 | 6.04 | Normal |

In the 79th Senate, the 1D IRT ideal points correlate at ρ = 0.974 with PC1 — the non-ideology component. The IRT party separation is a meager d = 0.86, compared to d = 4.98 on the axis that actually separates parties. The top and bottom of the IRT ranking are all Republicans:

```
Most "liberal" (IRT):           Most "conservative" (IRT):
  -3.259  Republican  Huelskamp    +2.712  Republican  Vratil
  -2.251  Republican  Lyon         +2.645  Republican  Oleen
  -2.223  Republican  Pugh         +2.610  Republican  Teichman
  -1.911  Republican  Tyson        +2.406  Republican  Adkins
  -1.746  Republican  O'Connor     +2.278  Republican  Kerr
```

The IRT is ranking the conservative insurgents (Huelskamp, Tyson) at the "liberal" end and the moderate establishment (Vratil, Oleen) at the "conservative" end. It's measuring intra-Republican factionalism, not ideology. The 10 Democrats are scattered through the middle with a mean of -0.485 — barely distinguishable from the Republican mean of +0.650.

## The Horseshoe Connection

The canonical routing system correctly detects the horseshoe in the 79th Senate (30% of Democrats on the wrong side, 88% party overlap). But the 2D IRT failed to converge for this session (Tier 3), so the canonical source fell back to 1D IRT — which, as shown above, is measuring the wrong axis.

This is the exact failure mode the Hierarchical 2D IRT (Phase 07b) was designed to address. But the deeper issue remains: **the 1D IRT model doesn't just produce noisy estimates in these sessions — it produces estimates on the wrong latent dimension entirely.**

## Implications for the Pipeline

### What's Working

1. The horseshoe detector correctly flags all affected sessions
2. PCA itself is correct — it's doing what PCA does (maximize variance)
3. The sign convention (`orient_pc1()`) works within its assumptions — it flips PC1 so Republicans are positive. The problem is that PC1 isn't ideology in these sessions.

### What Needs Investigation

1. **PCA-informed IRT initialization**: When PC1 isn't ideology, `--init-strategy pca-informed` initializes the IRT sampler on the wrong axis. The MCMC should eventually find the right mode, but starting far from it increases convergence risk. A potential fix: detect the PC swap and use PC2 for init when PC2 has stronger party separation.

2. **1D IRT adequacy**: In 4 of 14 sessions, the 1D IRT Senate estimates have Cohen's d < 1.5 between parties. These aren't just noisy — they're substantively wrong for ideology measurement. For these sessions, the 1D model should probably be flagged as unreliable, independent of the horseshoe detector.

3. **Hierarchical model priors**: The Hierarchical 1D IRT (Phase 07) uses `mu_party_raw ~ Normal(0, 2)` with `sort(mu_party)` for identification. If the discrimination parameters (beta) align the model to the intra-R axis rather than the party axis, the sort constraint (D < R) provides correct party *ordering* but on the wrong *dimension*. The party means will be close together because the model isn't measuring what separates parties.

4. **Cross-session comparability**: IRT ideal points from the 79th-83rd Senates are measuring a fundamentally different construct than those from the 85th-91st Senates. The Dynamic IRT (Phase 27) stitches these together assuming they measure the same thing. This assumption is violated.

### What Doesn't Need Fixing

1. The PCA report itself — PC1 being intra-R variance is a genuine finding, not a bug
2. The House results — PC1 is consistently ideology across all sessions
3. Sessions where PC1 d > 2.0 — the standard pipeline works correctly for these

## Historical Context

The Kansas Senate shifted dramatically during this period. The 78th-83rd Senates (1999-2010) saw intense intra-Republican conflict between moderate establishment Republicans (Vratil, Oleen, Kerr) and the conservative movement (Huelskamp, Tyson, Pyle). This factional war generated more roll-call variance than the R-vs-D party divide, because Republicans controlled the chamber and their internal disagreements produced the contested votes.

The 2012 Kansas Republican primary ("the purge") swept out many moderate Republican senators. After 2013, the surviving Republican caucus was more ideologically homogeneous, reducing intra-R variance. The party divide re-emerged as the dominant axis.

This is exactly the kind of structural shift that a fixed "PC1 = ideology" assumption will miss. The data itself is telling us that the dominant political cleavage in the Kansas Senate changed between 2010 and 2014.
