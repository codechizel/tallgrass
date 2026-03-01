# ADR-0052: Run-grouped results directories

**Date:** 2026-02-27
**Status:** Accepted (amended 2026-02-28: auto-generate run_id, eliminate legacy mode)

## Context

Each analysis phase writes to its own `results/kansas/{session}/{phase}/{date}/` directory with an independent `latest` symlink. When running the full pipeline, phases clobber each other's `latest` — phase N+1 may read stale data from a previous run's `latest` instead of the current run's output.

For reproducibility and pipeline integrity, we need a "run directory" that groups all phases from a single pipeline execution under one timestamp.

## Decision

Add a `--run-id` parameter to all phase scripts and `RunContext`. When set, output goes to `results/{session}/{run_id}/{phase}/` instead of `results/{session}/{phase}/{date}/`. A session-level `latest` symlink points to the run directory.

**Amendment (2026-02-28):** For biennium sessions, `RunContext` now auto-generates a run_id when none is provided. This eliminates the legacy per-phase directory layout for biennium sessions — every run (pipeline or standalone) uses the same `{session}/{run_id}/{phase}/` structure. Non-biennium sessions (cross-session, special sessions) retain their flat date-label layout. See ADR-0056.

### Run ID format

`{bb}-{YYMMDD}.{n}` where `bb` is the legislature number (ordinal suffix stripped) and `n` starts at 1. Subsequent same-day runs increment: `.2`, `.3`, etc.

Example: `91-260228.1`

### Directory layout

**Biennium sessions (all runs, pipeline or standalone):**
```
results/kansas/91st_2025-2026/
  91-260228.1/                       ← run directory
    01_eda/
      plots/  data/  run_info.json  run_log.txt
    02_pca/
      ...
  latest → 91-260228.1               ← session-level symlink
  01_eda_report.html → latest/01_eda/01_eda_report.html
```

**Cross-session (flat — no phase nesting):**
```
results/kansas/cross-session/
  90-vs-91/                          ← comparison directory
    260226.1/                        ← YYMMDD.n run
      plots/  data/  run_info.json  run_log.txt  90-vs-91_report.html
    latest → 260226.1
  90-vs-91_report.html → 90-vs-91/latest/90-vs-91_report.html
```

### New functions in `run_context.py`

- **`generate_run_id(session, results_root=None)`** — creates a run ID from session + date, with optional collision detection
- **`resolve_upstream_dir(phase, results_root, run_id, override)`** — 4-level precedence: (1) explicit CLI override, (2) `results_root/{run_id}/{phase}`, (3) `results_root/{phase}/latest` (flat mode), (4) `results_root/latest/{phase}`

### Pipeline recipe

`just pipeline 2025-26` generates a run ID and passes it to all 14 phases in dependency order.

## Consequences

**Positive:**
- Full pipeline runs are atomic — no stale cross-phase reads
- Every run is self-contained and traceable via `run_info.json`
- Single directory layout for all biennium sessions (no orphan phase directories)
- Standalone phase runs share the same structure as pipeline runs

**Negative:**
- `resolve_upstream_dir()` adds a layer of indirection
- Standalone phase runs auto-generate a new run_id, so upstream data resolves through `latest` symlink (precedence 4) rather than same-run lookup

**Backward compatibility:**
- `--run-id` is optional everywhere. For biennium sessions, omitting it triggers auto-generation.
- Non-biennium sessions (cross-session, special) retain flat `{phase}/{date}/` layout.
- `load_all_upstream()` accepts an optional `run_id` parameter (defaults to None).
- Experiment runner and cross-session work without changes (fallback path resolution).
