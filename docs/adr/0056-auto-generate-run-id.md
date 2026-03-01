# ADR-0056: Auto-generate run_id for biennium sessions

**Date:** 2026-02-28
**Status:** Accepted

## Context

ADR-0052 introduced two directory layouts: **run-directory mode** (pipeline runs with `--run-id`) grouping phases under `{session}/{run_id}/{phase}/`, and **legacy mode** (standalone runs without `--run-id`) creating `{session}/{phase}/{date}/` directories.

In practice, running a single phase standalone (e.g. `just hierarchical`) created an orphan phase directory at the session root that sat alongside the run directories from pipeline runs. This violated the single-layout assumption, produced confusing directory listings, and created inconsistent report symlink targets.

## Decision

`RunContext` now auto-generates a `run_id` via `generate_run_id()` when:
1. No explicit `--run-id` is provided, **and**
2. The session is a biennium (normalized name contains `_`, e.g. `91st_2025-2026`)

Non-biennium sessions (`"cross-session"`, `"2024s"`) retain the flat `{analysis}/{date}/` layout since their directory structure is intentionally different (no phase nesting).

### Upstream resolution for standalone runs

When a standalone phase auto-generates a new run_id, its upstream phases won't be under that same run_id. `resolve_upstream_dir()` handles this via precedence 4: `results_root/latest/{phase}`, which resolves through the session-level `latest` symlink to the most recent pipeline run's data.

### Additional fixes in this change

- **`analysis/__init__.py`**: Added `irt_linking` to the PEP 302 meta-path finder's `_MODULE_MAP` — fixes `ModuleNotFoundError` that prevented `test_hierarchical.py` and `test_experiment_runner.py` from collecting (78 tests unblocked).
- **`analysis/experiment_runner.py`**: Removed unused imports (`build_joint_graph`, `build_per_chamber_graph`).

## Consequences

**Positive:**
- Single directory layout for all biennium sessions — no orphan phase directories
- Standalone phase runs and pipeline runs produce identical structure
- 78 previously-blocked tests now run (1273 total, up from 1195)

**Negative:**
- Each standalone phase run creates its own run directory (rather than reusing an existing one) — minor directory proliferation
- Upstream data for standalone runs resolves through `latest` symlink (indirect), not same-run path (direct)
