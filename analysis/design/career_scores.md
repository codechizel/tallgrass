# Career Scores Design Choices

**Script:** `analysis/28_common_space/common_space_data.py` (new functions)
**Entry point:** `analysis/28_common_space/common_space.py` (integrated into Phase 28 output)
**Report:** `analysis/28_common_space/common_space_report.py` (new sections)
**Deep dive:** `docs/career-scores.md`

## Purpose

Collapse per-session common-space ideal points into a single career score per legislator using random-effects meta-analysis. Produces one row per legislator per chamber with a pooled estimate, SE, and heterogeneity diagnostics.

## Method

Random-effects meta-analysis (DerSimonian-Laird) on per-session common-space scores. Each session's posterior SD (propagated through the alignment chain) provides the within-session variance. The between-session variance (tau²) captures genuine ideological movement.

## Parameters & Constants

| Constant | Value | Justification |
|----------|-------|---------------|
| `I_SQUARED_STABLE` | 0.25 | Higgins 2002: I² < 25% = low heterogeneity |
| `I_SQUARED_MOVER` | 0.75 | Higgins 2002: I² > 75% = substantial heterogeneity |
| `MIN_SESSIONS_FOR_HETEROGENEITY` | 2 | Need ≥ 2 sessions to compute Q and I² |

## Algorithm

### Step 1: Group by legislator

Group the common-space output by `name_norm` × `chamber`. Each group has T rows (one per biennium served) with `xi_common` and `xi_common_sd`.

### Step 2: For each legislator, compute:

**If T = 1:** Career score = xi_common, SE = xi_common_sd, I² = NA, tau² = NA, flag = NA.

**If T >= 2:**

1. Fixed-effect weights: `w_t = 1 / sigma_t²`
2. Fixed-effect pooled mean: `mu_FE = sum(w_t * x_t) / sum(w_t)`
3. Cochran's Q: `Q = sum(w_t * (x_t - mu_FE)²)`
4. I²: `max(0, (Q - (T-1)) / Q)`
5. DerSimonian-Laird tau²: `max(0, (Q - (T-1)) / (sum(w) - sum(w²)/sum(w)))`
6. RE weights: `w_t* = 1 / (sigma_t² + tau²)`
7. Career score: `mu_RE = sum(w_t* * x_t) / sum(w_t*)`
8. Career SE: `SE = sqrt(1 / sum(w_t*))`
9. Movement flag: stable (I² < 0.25), moderate (0.25-0.75), mover (> 0.75)

### Step 3: Output

- `career_scores_{chamber}.parquet` / `.csv`
- Columns: full_name, party (most recent), chamber, n_sessions, first_session, last_session, career_score, career_se, career_lo, career_hi, i_squared, tau_squared, movement_flag, most_recent_score, most_recent_se

### Step 4: Report sections (added to existing Phase 28 report)

- **Career Score Table** — Interactive (ITables), searchable by name, sortable by score
- **Career Score vs. Most Recent Session** — scatter plot, stable legislators on diagonal
- **I² Distribution** — histogram of heterogeneity across all multi-session legislators
- **Top Movers** — table of legislators with highest I², showing trajectory direction

## Implementation Plan

| # | Task | Scope |
|---|------|-------|
| 1 | Add `compute_career_scores()` to `common_space_data.py` | ~40 lines: DL meta-analysis, per-legislator |
| 2 | Add career score output to `common_space.py` main() | Save parquet/CSV, add to results dict |
| 3 | Add 4 report sections to `common_space_report.py` | Career table, scatter, I² histogram, movers |
| 4 | Add tests to `test_common_space.py` | Career score math, edge cases (T=1, T=2), I² |
| 5 | Update docs | CLAUDE.md reference, roadmap entry |

All changes are within the existing Phase 28 package — no new phase, no new Justfile recipe. Career scores are produced automatically whenever `just common-space` runs.

## Dependencies

```
Phase 28 common space output (common_space_{chamber}.parquet)
  └─→ Career scores (career_scores_{chamber}.parquet)
```

No new external dependencies. Uses only numpy and polars.
