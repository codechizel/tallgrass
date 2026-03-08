# 79th Kansas Senate: Horseshoe Effect Robustness Analysis

An empirical investigation of the horseshoe effect in the 79th Kansas Legislature (2001-2002)
using the IRT robustness flags system (ADR-0104). Three diagnostic flags were run independently
on the same session to quantify dimension distortion in a 30R/10D supermajority Senate.

**Session:** 79th Kansas Legislature (2001-2002)
**Date:** 2026-03-08
**Run IDs:** 79-260308.2 (horseshoe), 79-260308.3 (contested), 79-260308.4 (2D cross-ref)

## The Problem

In a supermajority legislature, 1D IRT can confuse two distinct voting patterns:

1. **Ultra-conservative rebels** vote Nay because the bill is too moderate (Huelskamp, Oleen)
2. **Democrats** vote Nay because the bill is too conservative (Haley, Hensley)

Both produce Nay votes on the same bills, so a 1D model places them near each other. The
recovered dimension becomes "rebel vs. establishment" rather than "conservative vs. liberal."
This is the horseshoe effect — the ideological spectrum folds back on itself, and the two
ends of the horseshoe overlap in 1D projection.

The identification strategy system (ADR-0103) selects `anchor-agreement` for the 79th Senate
and `validate_sign()` corrects the sign flip, so Huelskamp correctly appears on the
conservative end. But the fundamental dimension distortion remains: the model measures
something closer to "propensity to dissent from the majority coalition" than ideological
position.

## Chamber Configuration

| Chamber | Legislators | Votes (filtered) | R/D Split | Strategy | Anchors | Sign Flipped |
|---------|-------------|-------------------|-----------|----------|---------|--------------|
| House | 128 | 621 | 90R/38D (70%) | anchor-pca | Weber (R) / Spangler (D) | No |
| Senate | 40 | 437 | 30R/10D (75%) | anchor-agreement | Tyson (R) / Haley (D) | Yes |

The House is at the edge of supermajority territory (70%), while the Senate exceeds it (75%).
The identification system correctly auto-selects different strategies for each chamber.

## Diagnostic 1: Horseshoe Detection (`--horseshoe-diagnostic`)

Six quantitative metrics assess whether the 1D ideal points exhibit horseshoe distortion.

### House

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Democrat wrong-side fraction | 0.0% | No Democrats appear on the conservative side |
| Party overlap fraction | 0.0% | Complete party separation |
| PCA eigenvalue ratio (PC1/PC2) | 1.78 | PC1 captures substantially more variance than PC2 |

**Verdict: No horseshoe detected.** The balanced House produces clean party separation, as
expected for a chamber where inter-party dynamics dominate.

### Senate

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Democrat wrong-side fraction | **30.0%** | 3 of 10 Democrats placed on the conservative side (threshold: 20%) |
| Party overlap fraction | **88.3%** | Nearly complete mixing — Republicans and Democrats share 88% of the ideal point range |
| PCA eigenvalue ratio (PC1/PC2) | **1.45** | PC1 barely dominates PC2 — nearly equal variance on two dimensions |
| R mean ideal point | -0.14 | Republicans centered slightly to the "liberal" side |
| D mean ideal point | -0.34 | Democrats centered nearby — only 0.20 separation |
| Most negative Republican | **Lana Oleen (xi = -3.84)** | More "liberal" than the Democrat mean by a factor of 11 |

**Verdict: Horseshoe DETECTED.** Every metric confirms the distortion:

- **30% of Democrats on the wrong side** means the model cannot reliably assign party labels
  from ideal points alone. In a correctly recovered ideological dimension, zero Democrats should
  appear more conservative than the Republican center.

- **88% party overlap** means ideal points provide almost no party-discriminating information.
  For comparison, the House has 0% overlap — complete separation.

- **Eigenvalue ratio of 1.45** is dangerously close to 1.0 (equal dimensions). In well-behaved
  chambers, this ratio is typically 2.0-4.0. A ratio near 1.0 means the "first" dimension
  captures barely more signal than the second, and the choice of which dimension to project
  onto is nearly arbitrary.

- **Oleen at xi = -3.84** is the smoking gun. Oleen was one of the most conservative Republican
  senators — she chaired the Ways and Means committee and was known for fiscal conservatism.
  Her placement as the most "liberal" legislator in the entire Senate (below all 10 Democrats)
  demonstrates that the 1D model is not measuring ideology. It is measuring establishment
  loyalty: Oleen, like Huelskamp, frequently dissented from the majority coalition.

## Diagnostic 2: Contested-Only Refit (`--contested-only`)

This diagnostic re-fits the IRT model using only cross-party contested votes — bills where
both parties split (each party has at least 10% of its members on each side). By stripping
out intra-party votes (where only Republicans split), the model should recover a dimension
closer to inter-party ideology.

### House

| Metric | Value |
|--------|-------|
| Total votes | 621 |
| Contested votes | 232 (37%) |
| Contested model sampling time | 40.2s |
| **Primary vs. contested-only correlation** | **r = 0.41** |

**This is a dramatic finding.** A Pearson correlation of 0.41 between the full model and the
contested-only model means the two are measuring substantially different things. In a healthy
1D model, we would expect r > 0.90 — the contested votes should reinforce the same dimension
that the full model captures.

r = 0.41 means the full House model is dominated by the 389 non-contested votes (63% of the
total), which are mostly intra-Republican splits. When those votes are removed, the ideal
points reorder almost entirely. This is striking because the House is only at 70% Republican
— barely a supermajority — yet the intra-party dynamics already dominate the 1D model.

**Implication:** Even in chambers that appear "balanced enough" for standard IRT, a large
fraction of non-contested votes can bias the recovered dimension toward intra-party factional
dynamics rather than inter-party ideology. The 37% contested rate means only about one in
three votes provides clean left-right signal.

### Senate

| Metric | Value |
|--------|-------|
| Total votes | 437 |
| Contested votes | 117 (27%) |
| Contested model sampling time | 6.8s |
| **Primary vs. contested-only correlation** | **r = 0.96** (sign-corrected) |

The high correlation (r = 0.96) seems to contradict the horseshoe finding, but it reflects
a different phenomenon. In the supermajority Senate, the full model's dimension is already
driven by the rebel-vs-establishment dynamic, which happens to be roughly consistent with the
contested-vote dimension. The rebels (Huelskamp, Oleen, Tyson) dissent from the majority
coalition on both intra-party and cross-party votes, so removing the intra-party votes doesn't
change their ranking much.

The sign flip (the contested model required sign correction) confirms that the two models
settle into opposite posterior modes — but once corrected, they agree. This is because the
79th Senate's contested votes still have the same 30R/10D asymmetry, so the contested-only
model faces a similar (though less severe) identification challenge.

## Diagnostic 3: 2D Cross-Reference (`--promote-2d`)

This diagnostic compares 1D ideal point rankings with the Dimension 1 rankings from the 2D
IRT model (Phase 06, run 79-260307.3). If the 1D model has correctly recovered the primary
ideological dimension, the two should be highly correlated. Large rank shifts indicate
legislators who are misplaced by the 1D model's dimension conflation.

### House

| Metric | Value |
|--------|-------|
| Legislators matched | 128 |
| 1D vs. 2D Dim 1 correlation | **r = 0.16** |
| Legislators with rank shift > 10 | 5 |

| Legislator | 1D Rank | 2D Rank | Shift |
|------------|---------|---------|-------|
| Stanley Dreher | 19 | 121 | 102 |
| Gerry Ray | 26 | 127 | 101 |
| Carol Edward Beggs | 20 | 115 | 95 |
| Al Lane | 35 | 125 | 90 |
| Bob Bethell | 12 | 101 | 89 |

A correlation of r = 0.16 between the 1D and 2D models means they are **essentially
uncorrelated** — the 1D model's single dimension captures something almost orthogonal to the
2D model's first dimension. Five legislators shift by 89-102 rank positions, meaning they are
placed near one end by the 1D model and near the opposite end by the 2D model.

This is consistent with the contested-only finding (r = 0.41): the House 1D model is capturing
a dimension that is neither the 2D model's first dimension nor the contested-vote dimension.
It is being dominated by intra-party factional dynamics embedded in the 63% of non-contested
votes.

### Senate

| Metric | Value |
|--------|-------|
| Legislators matched | 40 |
| 1D vs. 2D Dim 1 correlation | **r = -0.13** |
| Legislators with rank shift > 10 | 5 |

| Legislator | 1D Rank | 2D Rank | Shift |
|------------|---------|---------|-------|
| David Haley | 4 | 38 | 34 |
| Anthony Hensley | 6 | 40 | 34 |
| Paul Feleciano | 7 | 39 | 32 |
| Mark Gilstrap | 8 | 37 | 29 |
| Jim Barone | 9 | 36 | 27 |

A correlation of r = -0.13 confirms the total failure of 1D to recover the ideological
dimension in the Senate. The 1D and 2D models are **negatively correlated** — they disagree
about which direction is "conservative."

All five flagged legislators are Democrats. In the 1D model, they rank 4-9 (appearing
moderately conservative). In the 2D model, they rank 36-40 (the most liberal end). This is
the horseshoe effect made concrete: the Democrats are folded into the middle of the
Republican distribution by the 1D compression, and the 2D model unfolds them back to
their correct positions.

## Cross-Diagnostic Synthesis

The three diagnostics tell a consistent story:

### The 79th Senate is a textbook horseshoe case

Every metric confirms the distortion. The 1D model recovers a dimension that is orthogonal
to ideology (r = -0.13 with 2D Dim 1), with 88% party overlap and 30% of Democrats on the
wrong side. The contested-only refit agrees with the full model (r = 0.96) because both are
dominated by the same rebel-vs-establishment dynamic.

### The 79th House is a surprise

Despite being only 70% Republican — at the threshold of supermajority — the House shows
even more dramatic dimension distortion than the Senate. The contested-only model (r = 0.41)
and 2D cross-reference (r = 0.16) both indicate that the 1D model is measuring something
almost unrelated to inter-party ideology. The difference is that the House's distortion is
not a horseshoe (no party overlap, no Democrats on the wrong side) — it is pure intra-party
factional dynamics dominating a model that has more than enough data to capture them.

### The contested-vote fraction is a warning signal

| Chamber | Contested Fraction | Primary vs. Contested r | 1D vs. 2D r |
|---------|--------------------|-------------------------|-------------|
| House | 37% (232/621) | 0.41 | 0.16 |
| Senate | 27% (117/437) | 0.96 | -0.13 |

When fewer than 40% of votes are cross-party contested, the remaining 60%+ of intra-party
votes can dominate the 1D model and steer the recovered dimension away from ideology. The
contested fraction is a quick, pre-MCMC diagnostic that flags sessions requiring robustness
analysis.

## Recommendations

1. **Always run `--horseshoe-diagnostic` on supermajority sessions.** The computational cost
   is negligible (metrics computed from existing results, no extra MCMC). The six metrics
   provide a clear pass/fail signal.

2. **Use `--contested-only` to check dimension stability.** A correlation below 0.80 between
   the full and contested-only models indicates that intra-party dynamics are distorting
   the 1D dimension. This applies even to "balanced" chambers like the 79th House.

3. **Run `--promote-2d` when 2D results are available.** A 1D-vs-2D correlation below 0.50
   (or negative) means the 1D model is not capturing the primary ideological dimension.

4. **For the 79th specifically:** The Senate ideal points should be interpreted as
   "establishment loyalty" scores, not ideology scores. The House ideal points should be
   interpreted with similar caution. Cross-session comparisons (Phase 26) involving the
   79th should note the dimension distortion.

5. **Future work:** Automatic contested-fraction warnings in Phase 01 (EDA), pre-IRT
   horseshoe risk scoring, and promotion of 2D IRT results when 1D diagnostics fail.

## Appendix: Run Configuration

All three runs used identical MCMC settings:

| Parameter | Value |
|-----------|-------|
| Samples | 2,000 draws per chain |
| Tune | 1,000 draws (discarded) |
| Chains | 2 |
| Target accept | 0.90 |
| Seed | 42 |
| Sampler | nutpie (Rust NUTS) |
| PCA init | Yes |
| Data source | CSV (`--csv`) |

Convergence passed all checks for both chambers across all three runs (R-hat < 1.01,
ESS > 400, 0 divergences, E-BFMI > 0.3).
