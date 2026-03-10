# ADR-0048: Experiment Framework

**Date:** 2026-02-27
**Status:** Accepted

## Context

Tallgrass has a growing experimental practice. Six experiments have been run since 2026-02-23. The positive-beta experiment (`run_experiment.py`, 799 lines) duplicates ~200 lines of production model-building code to change 3 lines (the beta prior). If production evolves, experiments silently drift. Platform constraints (thread caps, compiler checks) aren't enforced by experiment scripts that bypass the Justfile.

Three structural problems:
1. **Code duplication creates drift risk** — experiment copies of `build_per_chamber_model()` and `build_joint_model()` can diverge from production
2. **Platform constraints are implicit** — M3 Pro MCMC rules (ADR-0022) only enforced by Justfile exports, not by experiment code
3. **No standard experiment lifecycle** — each experiment reinvents data loading, diagnostics, and output structure

## Decision

Implement a lightweight experiment framework with three components:

### 1. BetaPriorSpec (`analysis/07_hierarchical/model_spec.py`)

Factor the bill discrimination (beta) prior into a frozen dataclass. Production functions gain a `beta_prior` parameter with `PRODUCTION_BETA` as the default. Zero behavior change for production callers. Experiments pass alternative specs to the same functions — no code duplication.

Two production constants are defined: `PRODUCTION_BETA = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})` for per-chamber models, and `JOINT_BETA = BetaPriorSpec("lognormal_reparam", {"mu": 0, "sigma": 1})` for the joint model. The `lognormal_reparam` case uses `exp(Normal(0, 1))` to produce positive discrimination without boundary geometry (ADR-0055). Supported distribution types: `normal`, `lognormal`, `lognormal_reparam`, `halfnormal`.

### 2. Experiment Monitoring (`analysis/experiment_monitor.py`)

- **PlatformCheck**: validates Apple Silicon constraints (thread caps, C++ compiler, concurrent MCMC processes) before sampling begins
- **Monitoring callback**: updates process title via `setproctitle` and writes atomic JSON status file every 50 draws during `pm.sample()`
- **ExperimentLifecycle**: context manager with PID lock (`fcntl`), process group isolation, SIGTERM handler, and automatic cleanup

### 3. Experiment Runner (`analysis/experiment_runner.py`)

- **ExperimentConfig**: frozen dataclass bundling all experiment parameters (beta prior, sampling settings, session, chambers)
- **run_experiment()**: orchestrates the full lifecycle: platform check → data load → per-chamber models → optional joint model → HTML report → metrics.json
- All model-building, diagnostics, and reporting logic is imported from production — zero duplication
- HTML reports use `build_hierarchical_report()` from `analysis/07_hierarchical/hierarchical_report.py` — the same 18-22 section report as production `just hierarchical`. Standalone experiment scripts also call this function directly, ensuring experiment output is visually identical to production and can be directly compared.

## Alternatives Considered

- **Sacred/Hydra/MLflow**: too heavy for single-developer, single-machine use; designed for ML hyperparameter sweeps, not structural model variants (see ecosystem survey in `docs/experiment-framework-deep-dive.md`)
- **OOP Strategy/Template Method**: model variants share 90-95% of code and differ by data (which prior), not behavior — functional approach is simpler
- **Pydantic**: 6.5x slower than dataclasses; overkill for internal configs

## Consequences

**Positive:**
- 799-line experiment scripts become ~25-line configs
- Production functions are the single source of truth for model-building logic
- Platform constraints are enforced before hours of sampling begin
- MCMC processes are visible in `ps` via `setproctitle`
- Standardized output format enables cross-variant comparison

**Negative:**
- Structural variants (2D IRT, PC2-targeted IRT) that change model topology or ideal-point priors still need standalone scripts — `BetaPriorSpec` only handles beta-prior variants. Standalone scripts should still use `PlatformCheck` and `ExperimentLifecycle` directly for platform safety and process management (see ADR-0105).
- `setproctitle` adds a dev dependency (~20 KB wheel)

**Dependencies added:** `setproctitle>=1.3`

## Related

- [ADR-0022](0022-analysis-parallelism-and-timing.md) — Apple Silicon MCMC constraints
- [ADR-0044](0044-hierarchical-pca-informed-init.md) — PCA-informed initialization
- [ADR-0045](0045-4-chain-hierarchical-irt.md) — 4-chain with adapt_diag
- [ADR-0047](0047-positive-beta-constraint-experiment.md) — Positive beta experiment (motivation)
- [ADR-0055](0055-reparameterized-beta-and-irt-linking.md) — Reparameterized LogNormal and IRT linking (uses BetaPriorSpec)
- [ADR-0105](0105-experiment-lab-cleanup.md) — Experiment lab cleanup (removes dead code, fixes drift)
- [Experiment Framework Deep Dive](../experiment-framework-deep-dive.md) — Full ecosystem survey and design rationale
