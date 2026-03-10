# ADR-0105: Experiment Lab Cleanup

**Date:** 2026-03-09
**Status:** Accepted

## Context

A code review of the experiment lab infrastructure (`docs/experiment-lab-code-review.md`) identified seven issues: dead code from pre-framework experiments, duplicated utility functions, redundant import fallbacks, inconsistent use of nutpie vs pm.sample, redeclared constants that could drift from production, and structural experiments bypassing platform safety checks.

## Decision

### 1. Delete `irt_beta_experiment.py`

`analysis/experimental/irt_beta_experiment.py` predated the experiment framework (ADR-0048). It built IRT models from scratch with inline `pm.Model()` blocks and string-based prior dispatch — exactly what `BetaPriorSpec` replaced. Its findings are preserved in `analysis/design/beta_prior_investigation.md` and experiment results directories.

### 2. Fix `_fmt_elapsed` duplication in `experiment_runner.py`

ADR-0067 claimed this was removed but the `experiment_runner.py` copy survived. Replaced with `from analysis.run_context import _format_elapsed`. Removed the duplicate test class from `test_experiment_runner.py` (already covered by `test_run_context.py`).

### 3. Remove try/except import fallbacks

Five try/except blocks in `experiment_runner.py` provided fallback imports for bare-module execution (`from model_spec import ...`). All callers use the `analysis.X` import path via `sys.path` setup. Replaced with direct imports.

### 4. Import production constants

`HIER_N_SAMPLES`, `HIER_N_TUNE`, `HIER_N_CHAINS`, `HIER_TARGET_ACCEPT` were redeclared as literal values. Now imported from `analysis.hierarchical` to prevent silent drift.

### 5. Migrate `irt_2d_experiment.py` to nutpie

The 2D IRT experiment was the last holdout using `pm.sample()`. Migrated to `nutpie.compile_pymc_model()` + `nutpie.sample()` with `jitter_rvs` excluding the PCA-initialized `xi` variable, matching the production pattern in `analysis/06_irt_2d/irt_2d.py`.

### 6. Add platform checks to standalone experiments

Structural variant experiments that bypass `run_experiment()` (e.g., PC2-targeted IRT) now use `PlatformCheck` and `ExperimentLifecycle` directly. This provides thread-cap validation, compiler checks, concurrent-job detection, PID locking, and SIGTERM cleanup without requiring full framework integration.

Pattern for standalone experiments:
```python
from analysis.experiment_monitor import ExperimentLifecycle, PlatformCheck

platform = PlatformCheck.current()
for w in platform.validate(n_chains):
    print(f"  WARNING: {w}")

with ExperimentLifecycle("experiment-name"):
    # ... sampling ...
```

## Consequences

- `analysis/experimental/irt_beta_experiment.py` deleted — 563 lines of dead code removed
- `experiment_runner.py` reduced from 548 to ~500 lines; all imports are direct (no fallbacks)
- 7 duplicate tests removed from `test_experiment_runner.py`
- All MCMC experiments now use nutpie consistently
- Production constant drift is impossible (imported, not redeclared)
- Standalone experiments get platform safety checks with ~5 lines of boilerplate

## Related

- [ADR-0048](0048-experiment-framework.md) — Experiment framework design
- [ADR-0067](0067-open-source-readiness.md) — Open-source readiness (original `_fmt_elapsed` fix)
- [ADR-0051](0051-nutpie-migration.md) — nutpie migration
- [Code Review](../experiment-lab-code-review.md) — Full findings document
