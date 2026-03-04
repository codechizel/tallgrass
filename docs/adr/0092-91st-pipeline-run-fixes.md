# ADR-0092: 91st Pipeline Run Fixes

**Date:** 2026-03-03
**Status:** Accepted

## Context

First full 25-phase pipeline run for the 91st biennium (2025-26) after directory restructuring (ADR-0091) and four new analysis phases (20-23). Nine bugs surfaced across 12 files — all fixed during the run.

## Decision

### Bugs Fixed

| # | Phase | Bug | Root Cause | Fix |
|---|-------|-----|------------|-----|
| 1 | ALEC scraper | 0 bills scraped | alec.org changed HTML from `<article>` to `<li class="media-flex">` | Fallback: try `<article>`, then `<li class="media-flex">` |
| 2 | 01 EDA | `TypeError` in format string | `bill_number` is `None` for some rollcalls | `(row['bill_number'] or '')` |
| 3 | 02 PCA+ | `ColumnNotFoundError: "slug"` | `load_legislators()` was centrally renaming `slug` → `legislator_slug` | Reverted central rename; 3 phases that need `legislator_slug` alias locally (LCA, W-NOMINATE, Bill Text) |
| 4 | 19 TSA | `TypeError: 'int' object is not iterable` | R JSON serializes single Bai-Perron breakpoint as scalar, not list | Wrap scalar in list for breakpoints, ci_lower, ci_upper |
| 5 | 20 Bill Text | `AttributeError: No prediction data` | HDBSCAN missing `prediction_data=True` required by BERTopic `all_points_membership_vectors()` | Added parameter |
| 6 | 20-23 | Double-nested output directories | `RunContext(results_root=session.results_dir)` appends session name again | Removed `results_root=` param (default `results/kansas/` is correct) |
| 7 | 23 Model Legislation | `ValueError: Invalid format specifier` | `make_interactive_table` expects `.3f`, not `{:.3f}` | Fixed format strings |
| 8 | EDA report | Stale missing votes section | `failure_manifest.json` not updated after KanFocus gap-fill | Cross-reference kf_ rollcalls at report time; show only genuinely unrecovered failures |

### Column Naming Convention (Clarified)

ADR-0066 established that scraper CSVs use `slug` and analysis expects `vote` (not `vote_category`). This run revealed ambiguity: `load_legislators()` in `phase_utils.py` was performing a central `slug` → `legislator_slug` rename that broke all phases referencing `slug`.

**Resolution:** `load_legislators()` keeps the CSV column name (`slug`) as-is. The three phases that specifically need `legislator_slug` (LCA, W-NOMINATE, Bill Text) rename locally. This is consistent with the existing pattern where each phase handles column aliasing at load time.

### Missing Votes Report Accuracy

The EDA report's "missing votes" section reads `failure_manifest.json`, which is written by the ksleg scraper and never updated after KanFocus gap-fill. For the 91st biennium: 23 failures recorded, but 20 were already recovered by KanFocus (kf_ prefix rollcalls).

**Resolution:** `_append_missing_votes()` in `run_context.py` now:
1. Loads kf_ rollcalls from the session's `rollcalls.csv`
2. Extracts (bill_number, vote_date) pairs from KanFocus data
3. Extracts vote dates from failure manifest URLs (je_YYYYMMDD pattern)
4. Filters out recovered votes before rendering
5. Reports recovery count in the section header

## Consequences

- All 25 phases complete successfully for 91st biennium
- ALEC scraper resilient to future HTML structure changes (fallback chain)
- Column naming convention is unambiguous: `slug` everywhere, local rename only when needed
- Missing votes report is always accurate regardless of gap-fill timing
- Future pipeline runs for other bienniums may surface additional edge cases in phases 20-23 (first-run bugs are expected for new phases)
