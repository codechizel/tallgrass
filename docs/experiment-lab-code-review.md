# Experiment Lab Code Review

**Date:** 2026-03-09
**Status:** All findings implemented in ADR-0105.
**Scope:** `analysis/experiment_runner.py`, `analysis/experiment_monitor.py`, `analysis/07_hierarchical/model_spec.py`, `analysis/experimental/*.py`, `results/experimental_lab/2026-03-09_pc2-targeted-irt/run_experiment.py`, and associated tests.

## Executive Summary

The experiment framework (ADR-0048) is well-designed. The core architecture — `BetaPriorSpec` as data, production functions with `beta_prior=` parameter, frozen config dataclasses — is clean and effective. The issues below are minor friction points and organic debt from rapid iteration, not structural problems.

Seven findings, roughly ordered by importance:

| # | Finding | Severity | Effort |
|---|---------|----------|--------|
| F1 | `irt_beta_experiment.py` is pre-framework dead code | Low | Small |
| F2 | `_fmt_elapsed` duplicated in `experiment_runner.py` | Low | Trivial |
| F3 | Try/except import dance repeated 5 times | Medium | Medium |
| F4 | PC2 experiment bypasses framework entirely | Medium | Design decision |
| F5 | `irt_2d_experiment.py` uses `pm.sample()` not nutpie | Medium | Small |
| F6 | Tests for `_format_elapsed` duplicated across two files | Low | Trivial |
| F7 | `experiment_runner.py` redeclares production constants | Low | Trivial |

---

## F1: `irt_beta_experiment.py` Is Pre-Framework Dead Code

**File:** `analysis/experimental/irt_beta_experiment.py`

This script builds IRT models from scratch with inline `pm.Model()` blocks — the exact pattern the experiment framework was designed to eliminate. It predates `BetaPriorSpec` and `run_experiment()`. Its `build_and_sample_variant()` function accepts a string like `"lognormal_0.5_0.5"` and dispatches to `pm.LogNormal(...)`, which is precisely what `BetaPriorSpec.build()` does with a frozen dataclass.

The script is referenced only in docs (ADR-0103, design doc, deep dive) as historical context. No code imports from it. Its usage comment still says `uv run python analysis/irt_beta_experiment.py` — the pre-move path.

**Recommendation:** Archive or delete. The experiment's findings are recorded in `analysis/design/beta_prior_investigation.md` and the results live in `results/experimental_lab/`. The script itself serves no ongoing purpose. If kept for posterity, move to a `_legacy/` directory so it's clearly not part of the active experiment surface.

---

## F2: `_fmt_elapsed` Duplicated in `experiment_runner.py`

**File:** `analysis/experiment_runner.py:335-343`

`experiment_runner.py` defines its own `_fmt_elapsed()` that is functionally identical to `_format_elapsed()` in `analysis/run_context.py`. ADR-0067 even documents that this duplication was removed — but it's back (or was never actually removed).

The experiment runner already imports from `analysis.irt`, `analysis.hierarchical`, `analysis.report`, etc. Adding one more import from `analysis.run_context` is trivial.

**Recommendation:** Replace `_fmt_elapsed` with `from analysis.run_context import _format_elapsed`. The test file (`test_experiment_runner.py`) already imports from `run_context` — it's testing the right function but the runner uses the wrong copy.

---

## F3: Try/Except Import Dance Repeated 5 Times

**File:** `analysis/experiment_runner.py:38-110`

Five separate try/except blocks handle the dual import paths (`analysis.X` vs bare `X`):

```python
try:
    from analysis.model_spec import PRODUCTION_BETA, BetaPriorSpec
except ModuleNotFoundError:
    from model_spec import PRODUCTION_BETA, BetaPriorSpec
```

This pattern exists to support running the script both as a module (`python -m analysis.experiment_runner`) and directly (`python analysis/experiment_runner.py`). The same pattern appears in `analysis/07_hierarchical/hierarchical.py`.

The root cause is the `sys.path.insert(0, ...)` on line 36, which makes both import styles work but requires every import to be duplicated.

**Recommendation:** Consolidate the path setup. Options:
1. **Simplest:** Since the experiment runner is always invoked through wrapper scripts (in `results/experimental_lab/*/run_experiment.py`) that already do `sys.path.insert(0, PROJECT_ROOT)`, the bare-module fallback imports are never triggered. Test this by removing the fallbacks and running the test suite. If they pass, the fallbacks are dead code.
2. **Cleanest:** Move to a single `sys.path` strategy (the `PROJECT_ROOT` insert that wrapper scripts already do) and use only `analysis.X` imports. The PEP 302 meta-path finder handles the numbered-directory resolution.

---

## F4: PC2 Experiment Bypasses the Framework Entirely

**File:** `results/experimental_lab/2026-03-09_pc2-targeted-irt/run_experiment.py`

The currently running experiment builds its own `pm.Model()` from scratch (line 183), defines its own `print_header()` (line 105), manages its own `nutpie.compile_pymc_model()` / `nutpie.sample()` calls, and writes its own `metrics.json` without the framework's platform checks or lifecycle management.

This is understandable — the experiment varies the *ideal point prior* and *initialization*, not the *beta prior*, which is all `ExperimentConfig` currently supports. The sort-constraint identification strategy (lines 196-207) is also structurally different from production's anchor-based identification.

However, this means the experiment runs without:
- `PlatformCheck` validation (thread caps, compiler check, concurrent job detection)
- `ExperimentLifecycle` (PID lock, SIGTERM handler, cleanup)
- Standardized `metrics.json` schema

This is the same "799-line script" anti-pattern the framework was designed to prevent, just for a different axis of variation.

**Recommendation:** This is a design decision, not a bug. Two paths forward:

1. **Extend the framework.** Generalize `ExperimentConfig` to support ideal-point prior variation (`xi_prior: XiPriorSpec | None`), custom initialization (`initvals: dict[str, np.ndarray] | None`), and identification strategy (`identification: str`). This is the right move if PC2-targeting experiments become a recurring pattern.

2. **Use lifecycle/platform checks standalone.** Even without full `run_experiment()` integration, wrapper scripts can call `PlatformCheck.current().validate(n_chains)` and use `with ExperimentLifecycle(name):` independently. This gets 80% of the safety benefit with no refactoring. A few lines at the top of `main()`:

```python
from analysis.experiment_monitor import ExperimentLifecycle, PlatformCheck
platform = PlatformCheck.current()
for w in platform.validate(N_CHAINS):
    print(f"  WARNING: {w}")
```

---

## F5: `irt_2d_experiment.py` Uses `pm.sample()`, Not nutpie

**File:** `analysis/experimental/irt_2d_experiment.py:166-175`

The 2D IRT experiment still uses PyMC's built-in `pm.sample()` rather than nutpie's Rust NUTS sampler. Every other MCMC model in the codebase migrated to nutpie (ADR-0051, ADR-0053). The PC2 experiment (F4 above) correctly uses nutpie.

This means the 2D experiment:
- Is significantly slower (PyMC's Python NUTS vs nutpie's Rust NUTS)
- Doesn't benefit from nutpie's `compile_pymc_model()` / `jitter_rvs` / `initial_points` API
- Uses `cores=n_chains` (PyMC multiprocessing) instead of nutpie's internal Rust threading

The model itself is compatible with nutpie — it's a standard PyMC model with Normal, HalfNormal, Deterministic, and Bernoulli variables.

**Recommendation:** Migrate to nutpie. The change is mechanical:

```python
compiled = nutpie.compile_pymc_model(model, initial_points={"xi": xi_initvals_2d})
idata = nutpie.sample(compiled, draws=n_samples, tune=n_tune, chains=n_chains, seed=RANDOM_SEED)
```

This also removes the need for the `init="adapt_diag"` workaround on line 161.

---

## F6: `_format_elapsed` Tests Duplicated

**Files:** `tests/test_experiment_runner.py:165-190` and `tests/test_run_context.py:145-167`

Both test files test `_format_elapsed()` from `analysis.run_context`. The experiment runner tests import and re-test the same function with slightly different values. This is harmless but adds ~7 tests to the suite that test nothing new.

**Recommendation:** Remove `TestFmtElapsed` from `test_experiment_runner.py`. The function is already thoroughly tested in `test_run_context.py`. If the experiment runner's `_fmt_elapsed` is replaced per F2, these tests become doubly redundant.

---

## F7: `experiment_runner.py` Redeclares Production Constants

**File:** `analysis/experiment_runner.py:116-119`

```python
HIER_N_SAMPLES = 2000
HIER_N_TUNE = 1500
HIER_N_CHAINS = 4
HIER_TARGET_ACCEPT = 0.95
```

These are hardcoded copies of production defaults from `analysis/07_hierarchical/hierarchical.py`. The comment says "Import production defaults from hierarchical.py" but they're not imported — they're redeclared. If production defaults change, the experiment runner will silently use stale values.

**Recommendation:** Import them:

```python
from analysis.hierarchical import (
    HIER_N_SAMPLES, HIER_N_TUNE, HIER_N_CHAINS, HIER_TARGET_ACCEPT,
)
```

These names are already imported in the try/except block on line 48 — they just aren't used for the constants. Alternatively, the `ExperimentConfig` defaults could reference `hierarchical.HIER_N_SAMPLES` directly, though frozen dataclass defaults must be compile-time constants (so a module-level import is required either way).

---

## What's Working Well

To be clear: the framework's core design is strong. Specific things worth calling out:

- **`BetaPriorSpec.build()` with match/case dispatch** (model_spec.py:48-68) is exactly right. Prior variation as data, not behavior. The error message on unknown distributions is helpful.
- **`PlatformCheck` as a frozen dataclass with `.current()` classmethod** (experiment_monitor.py:51-125) is well-designed. The validation rules encode real hard-won knowledge (ADR-0022), and the warning/FATAL distinction gives callers control.
- **`ExperimentLifecycle`** (experiment_monitor.py:158-258) handles the full POSIX lifecycle correctly: advisory locks via `fcntl.flock()`, atomic status writes via temp+rename, SIGTERM → `sys.exit(0)` for clean __exit__, and proper cleanup ordering.
- **`write_status()` with `os.fsync()` + `os.replace()`** (experiment_monitor.py:131-152) is textbook atomic file writing. The `BaseException` catch with cleanup prevents leaked temp files on KeyboardInterrupt.
- **Test coverage** for the framework infrastructure is solid. The `test_experiment_monitor.py` tests exercise real PID files and lock behavior, not mocks.
- **The PC2 experiment** (results/experimental_lab/2026-03-09_pc2-targeted-irt/run_experiment.py) is well-structured for a standalone script. Clean variant dispatch, proper sign validation, meaningful evaluation metrics, and a comparison table that answers the research question directly.
