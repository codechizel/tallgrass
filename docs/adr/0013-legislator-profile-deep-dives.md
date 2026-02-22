# ADR-0013: Legislator Profile Deep-Dives

**Date:** 2026-02-22
**Status:** Accepted

## Context

The synthesis report (ADR-0008) identifies notable legislators — mavericks, bridge-builders, metric paradoxes — but gives each only a paragraph and a profile card (6 normalized metrics). For a nontechnical audience, this is enough to name the interesting actors but not enough to understand *what they actually do differently*.

Questions the synthesis report can't answer:
- On which specific bills did Schreiber break ranks?
- Does Tyson's low clustering loyalty come from voting Nay on partisan bills, or from voting differently on routine bills?
- Who does Borjon vote most like — other moderate Republicans, or Democrats?
- Which of Dietrich's votes most surprised the prediction model?

Answering these requires combining vote-level data (raw CSVs) with upstream analysis outputs (IRT bill parameters, prediction surprising votes) — a join that synthesis deliberately avoids because it works only with aggregated parquets.

## Decision

Add a new analysis phase `profiles` (three files + tests) that produces per-legislator deep-dive reports:

| File | Role |
|------|------|
| `analysis/profiles_data.py` | Pure data logic (no plotting, no I/O) |
| `analysis/profiles.py` | CLI, data loading, 5 plotting functions, orchestration |
| `analysis/profiles_report.py` | HTML report builder (intro + per-legislator sections) |
| `tests/test_profiles.py` | 23 tests across 6 test classes |

### Target Selection

Reuses `detect_all()` from `synthesis_detect.py` — no new detection logic. Adds optional `--slugs` CLI flag for user-requested extras. Deduplicates and caps at 8 targets.

### Per-Legislator Analysis (6 views)

1. **Scorecard** — 6 normalized (0-1) metrics with party average markers. Uses `xi_mean_percentile` and `betweenness_percentile` (not raw values) so all bars are visually comparable.

2. **Bill type breakdown** — Yea rates on high-discrimination (|beta_mean| > 1.5) vs low-discrimination (|beta_mean| < 0.5) bills. Uses IRT `bill_params` to classify bills. Requires MIN_BILLS_PER_TIER (3) per tier.

3. **Position in context** — Forest plot of same-party IRT ideal points. Target highlighted with diamond marker and yellow background. HDI error bars when available.

4. **Defection bills** — Specific bills where the legislator voted against their party majority. Sorted by closeness of party margin (tightest first). Joined with rollcalls for bill metadata.

5. **Voting neighbors** — Top 5 most similar and most different legislators by simple agreement rate. Same-chamber only. Pivot-based pairwise computation.

6. **Surprising votes** — Filtered from prediction phase's `surprising_votes` parquet. Top 10 by confidence_error.

### Data Sources

- `load_all_upstream()` + `build_legislator_df()` from `synthesis.py` (reused, not duplicated)
- IRT `bill_params_{chamber}.parquet` (loaded separately — synthesis doesn't load these)
- Raw `votes.csv` and `rollcalls.csv` (for vote-level defection analysis)

### Chamber Normalization

Raw vote CSVs use title-case chamber names ("House", "Senate") while `leg_dfs` keys are lowercase. `prep_votes_long()` normalizes to lowercase via `str.to_lowercase()`.

## Consequences

**Positive:**
- All 6 analysis views are fully data-driven — different sessions surface different legislators with different stories.
- Pure data logic in `profiles_data.py` is independently testable (23 tests).
- Sections degrade gracefully: no defections → skip defection table; no surprising votes → skip table.
- Reuses existing infrastructure (synthesis loading, RunContext, report system) — no new dependencies.
- Scorecard uses only 0-1 metrics, avoiding the scale-mixing problem where raw PCA/UMAP/IRT values dominate the chart.

**Negative:**
- Vote-level analysis (defections, neighbors) requires loading the full raw votes CSV (~76K rows), which is slower than parquet-only phases.
- The neighbor computation uses a pivot to wide format, which is O(n_legislators × n_votes) in memory.
- Bill type thresholds (HIGH_DISC=1.5, LOW_DISC=0.5) are imported from IRT convention but are somewhat arbitrary. A bill with |beta|=1.4 is excluded from both tiers.
- Position plot is very tall for the House (~95 Republicans). Functional but the labels are small.

**Files created:**
- `analysis/profiles_data.py` — ~280 lines. Pure data logic.
- `analysis/profiles.py` — ~410 lines. CLI + plotting + orchestration.
- `analysis/profiles_report.py` — ~215 lines. HTML report builder.
- `tests/test_profiles.py` — ~280 lines. 23 tests.
- `analysis/design/profiles.md` — Design choices document.
