# Career Scores: One Number Per Legislator

**Date:** 2026-03-24
**Depends on:** Phase 28 Common Space Ideal Points
**Related:** `docs/common-space-ideal-points.md`, ADR-0120

---

## The Problem

Phase 28 produces per-session ideal points on a common scale — every legislator gets a score for every biennium they served. Barbara Ballard has 14 rows, one per biennium from the 78th through the 91st. This is the right output for tracking trajectories and studying within-career dynamics.

But sometimes you want one number: "How conservative is this legislator?" A journalist writing a profile, a voter checking a scorecard, or a researcher building a cross-sectional model all want a single career summary. DW-NOMINATE's Common Space constant model provides exactly this for Congress — one fixed ideal point per legislator, estimated from their entire voting record pooled across all sessions.

We can produce the same thing without re-running IRT. The per-session common-space scores with their posterior SDs are exactly the inputs to a meta-analysis.

---

## The Approach: Random-Effects Meta-Analysis

Each legislator's career is treated as a collection of "studies" — their per-session ideal point estimates, each with its own precision. The meta-analytic framework combines them into a single pooled estimate with a properly calibrated standard error.

### Why Random Effects, Not Fixed Effects

The fixed-effect model assumes the true ideal point is constant across sessions. The random-effects model allows it to vary:

```
x_t = mu + u_t + e_t
```

where mu is the career-average ideal point, u_t ~ N(0, tau²) is genuine between-session movement, and e_t ~ N(0, sigma_t²) is IRT estimation noise.

Political reality demands the random-effects model. Legislators do change — they respond to redistricting, constituent shifts, party pressure, and personal evolution. The fixed-effect model would produce artificially tight confidence intervals that ignore this variation. The random-effects model correctly reflects both sources of uncertainty: estimation noise and genuine movement.

For a legislator who was genuinely stable (tau² ≈ 0), the RE estimate converges to the FE estimate — no information is lost. For a legislator who moved substantially, the RE standard error is wider, honestly reflecting the ambiguity of summarizing a moving target with a single number.

### The Formulas

**Between-session heterogeneity** (REML estimator for tau²):

Cochran's Q:
```
Q = sum_{t=1}^{T} w_t * (x_t - mu_FE)^2     where w_t = 1/sigma_t^2
```

If Q > (T-1), there is excess heterogeneity beyond estimation noise.

Higgins I²:
```
I² = max(0, (Q - (T-1)) / Q)
```

Interpretation: I² < 25% = stable, 25-75% = moderate movement, > 75% = substantial mover.

**DerSimonian-Laird tau² estimator** (simple, works for small T):
```
tau² = max(0, (Q - (T-1)) / (sum(w_t) - sum(w_t²)/sum(w_t)))
```

**Random-effects weights:**
```
w_t* = 1 / (sigma_t² + tau²)
```

When tau² > 0, all sessions get more equal weight (noisy sessions are penalized less because the between-session variance dominates). When tau² = 0, the RE weights equal the FE weights.

**Pooled career score:**
```
mu_RE = sum(w_t* * x_t) / sum(w_t*)
```

**Standard error:**
```
SE(mu_RE) = sqrt(1 / sum(w_t*))
```

### Single-Session Legislators

Legislators who served only one biennium get their common-space score as their career score, with the common-space SD as the career SE. No pooling is needed — they have T=1.

I² is undefined for T=1 and set to NA. These legislators are neither stable nor movers — we simply don't have enough data to tell.

### Comparison with DW-NOMINATE Common Space

The DW-NOMINATE Common Space constant model pools all of a legislator's votes across their entire career into a single IRT model. This is mathematically equivalent to the precision-weighted fixed-effect mean at the vote level (treating each vote as a separate observation).

Our meta-analytic approach operates one level higher — pooling session-level estimates rather than individual votes. This has two advantages:

1. **It uses the horseshoe-corrected canonical scores.** The per-session IRT already handles dimensionality issues (2D routing for horseshoe sessions). A pooled-vote model would need to solve the horseshoe problem at the career level, which is harder.

2. **It naturally produces heterogeneity diagnostics.** DW-NOMINATE's constant model assumes stability by construction. Our approach tests for it via I² and flags movers.

The disadvantage is slightly lower statistical efficiency — session-level pooling throws away within-session covariance information. In practice, with posterior SDs of 0.1-0.4, the loss is negligible.

---

## What the Output Looks Like

**Per-chamber tables** (`career_scores_house.parquet`, `career_scores_senate.parquet`): one row per unique legislator per chamber. **Unified table** (`career_scores_unified.parquet`): one row per legislator across both chambers, with Senate scores mapped onto House scale via 54 chamber-switcher bridges. Columns:

| Column | Description |
|--------|-------------|
| `full_name` | Legislator name |
| `party` | Most recent party affiliation |
| `chamber` | House or Senate |
| `n_sessions` | Number of bienniums served |
| `first_session` | Earliest biennium |
| `last_session` | Most recent biennium |
| `career_score` | RE pooled ideal point on common scale |
| `career_se` | Standard error (reflects both estimation noise and movement) |
| `career_lo` | 95% CI lower bound |
| `career_hi` | 95% CI upper bound |
| `i_squared` | Heterogeneity: proportion of variance from genuine change |
| `tau_squared` | Between-session variance estimate |
| `movement_flag` | "stable" (I² < 25%), "moderate" (25-75%), "mover" (> 75%) |
| `most_recent_score` | Score from most recent biennium (for comparison) |
| `most_recent_se` | SE from most recent biennium |

---

## Diagnostics

**For the report:**

1. **Distribution of I²** — histogram showing how many legislators are stable vs. movers. In a well-functioning legislature, most should be stable (I² < 25%), with a tail of movers.

2. **Career score vs. most recent session** — scatter plot. Stable legislators cluster on the diagonal. Movers deviate. This shows whether the career summary is a good proxy for the current position.

3. **Biggest movers** — table of legislators with the highest I², showing their trajectory (first session score → last session score, direction of change).

4. **Career score distribution by party** — violin plot or ridge plot showing the ideological spread within each party across all career scores.

---

## Known Limitations

**A career score is a lossy summary.** For a legislator who spent 10 years as a moderate Republican and then shifted hard right after a primary challenge, the career score is a meaningless average that represents no actual period. The I² flag catches this, but the number itself is still misleading. Always prefer the per-session trajectory for serious analysis.

**Small T bias.** With T = 2-3 sessions (typical for Kansas), the DerSimonian-Laird estimator underestimates tau². The REML estimator is more robust but computationally heavier. For T >= 4, DL is adequate.

**Party-switching legislators.** A handful of legislators changed parties during their career. The career score summarizes their entire career under the most recent party label. The per-session trajectory is more informative for these cases.

---

## References

- Cochran, W. G. (1954). The combination of estimates from different experiments. *Biometrics*, 10(1), 101-129.
- DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177-188.
- Higgins, J. P., & Thompson, S. G. (2002). Quantifying heterogeneity in a meta-analysis. *Statistics in Medicine*, 21(11), 1539-1558.
- Poole, K. T. (2005). *Spatial Models of Parliamentary Voting*. Cambridge University Press. [Common Space constant model]
- Shor, B., & McCarty, N. (2011). The ideological mapping of American legislatures. *APSR*, 105(3), 530-551.
