# ADR-0016: State-Level Directory Structure

**Date:** 2026-02-22
**Status:** Accepted

## Context

The project stores scraped data in `data/{biennium}/` and analysis results in `results/{biennium}/`. With multi-state expansion on the horizon (Nebraska analysis planned), all Kansas data needs to be separated from future state data at the directory level.

Additionally, `KSSession` already centralized session/URL logic but callers were manually constructing paths like `Path("data") / ks.output_name` in ~14 files. This scattered the path convention across the codebase.

## Decision

Insert a state-level directory between the root and the biennium directory:

```
data/kansas/91st_2025-2026/          (was: data/91st_2025-2026/)
data/kansas/90th_2023-2024/
results/kansas/91st_2025-2026/01_eda/   (was: results/91st_2025-2026/eda/)
```

Implementation:

- Add `STATE_DIR = "kansas"` module constant to `session.py`
- Add `data_dir` and `results_dir` properties to `KSSession` that include `STATE_DIR`
- Update `KSVoteScraper.__init__` to use `session.data_dir` instead of manual construction
- Update `RunContext` default `results_root` to `Path("results") / STATE_DIR`
- Update all analysis scripts (~14 files) to use `ks.data_dir` / `ks.results_dir`
- Update `data_dir_for_session()` static method to delegate to `ks.data_dir`

## Consequences

- **Good**: All path logic centralized on `KSSession` — callers say `ks.data_dir` not `Path("data") / ks.output_name`
- **Good**: Multi-state expansion requires only adding a new state module with its own `STATE_DIR`
- **Good**: Existing biennium naming (`91st_2025-2026`) is unchanged within the state directory
- **Trade-off**: One more directory level in all paths — minimal impact since everything is gitignored
- **Trade-off**: `STATE_DIR` is Kansas-specific; a true multi-state architecture would need per-session state resolution. Acceptable for now — YAGNI until Nebraska is real.
