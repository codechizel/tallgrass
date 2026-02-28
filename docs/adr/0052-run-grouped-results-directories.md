# ADR-0052: Run-grouped results directories

**Date:** 2026-02-27
**Status:** Accepted

## Context

Each analysis phase writes to its own `results/kansas/{session}/{phase}/{date}/` directory with an independent `latest` symlink. When running the full pipeline, phases clobber each other's `latest` — phase N+1 may read stale data from a previous run's `latest` instead of the current run's output.

For reproducibility and pipeline integrity, we need a "run directory" that groups all phases from a single pipeline execution under one timestamp.

## Decision

Add a `--run-id` parameter to all phase scripts and `RunContext`. When set, output goes to `results/{session}/{run_id}/{phase}/` instead of `results/{session}/{phase}/{date}/`. A session-level `latest` symlink points to the run directory.

### Run ID format

`{bb}-{YYMMDD}` where `bb` is the legislature number (ordinal suffix stripped). Same-day collisions append `.1`, `.2`, etc.

Example: `91-260228`

### Directory layout

**Pipeline runs (run-grouped):**
```
results/kansas/91st_2025-2026/
  91-260228/                         ← run directory
    01_eda/
      plots/  data/  run_info.json  run_log.txt
    02_pca/
      ...
  latest → 91-260228                 ← session-level symlink
  01_eda_report.html → latest/01_eda/01_eda_report.html
```

**Cross-session (flat — no phase nesting):**
```
results/kansas/cross-session/
  90-vs-91/                          ← comparison directory
    260226/                          ← YYMMDD run
      plots/  data/  run_info.json  run_log.txt  90-vs-91_report.html
    latest → 260226
  90-vs-91_report.html → 90-vs-91/latest/90-vs-91_report.html
```

**Legacy mode (individual phase runs):**
Date directories use YYMMDD format (e.g. `260227`). Same-day collisions append `.1`, `.2`, etc.

### New functions in `run_context.py`

- **`generate_run_id(session, results_root=None)`** — creates a run ID from session + date, with optional collision detection
- **`resolve_upstream_dir(phase, results_root, run_id, override)`** — 4-level precedence: (1) explicit CLI override, (2) `results_root/{run_id}/{phase}`, (3) `results_root/{phase}/latest`, (4) `results_root/latest/{phase}`

### Pipeline recipe

`just pipeline 2025-26` generates a run ID and passes it to all 13 phases in dependency order.

## Consequences

**Positive:**
- Full pipeline runs are atomic — no stale cross-phase reads
- Every run is self-contained and traceable via `run_info.json`
- Old results directories untouched; mixed layouts coexist safely

**Negative:**
- Two directory layouts to understand (run-grouped vs legacy per-phase)
- `resolve_upstream_dir()` adds a layer of indirection

**Backward compatibility:**
- `--run-id` is optional everywhere. Omitting it preserves 100% current behavior.
- `load_all_upstream()` accepts an optional `run_id` parameter (defaults to None).
- Experiment runner and cross-session work without changes (fallback path resolution).
