# ADR-0022: Analysis Parallelism and Runtime Timing

**Date:** 2026-02-23
**Status:** Accepted

## Context

The analysis pipeline includes several computationally expensive phases: MCMC sampling (IRT, hierarchical IRT) and ML model training (XGBoost in prediction and cross-session). These were running single-threaded or with sequential chains, leaving most CPU cores idle on modern hardware (e.g., M3 Pro with 12 cores, 36GB RAM).

Separately, when re-running analyses after fixes, there was no easy way to compare runtimes between runs. A phase that suddenly runs much faster or slower can indicate a bug (skipped computation, convergence failure, or an accidental regression).

Same-day re-runs also had a clobbering problem: the second run would overwrite the first day's directory, destroying the baseline for comparison.

## Decision

### 1. Parallel MCMC chain sampling

Added `cores=n_chains` to all `pm.sample()` calls in `irt.py`, `hierarchical.py`, and `irt_beta_experiment.py`.

**Safety:** PyMC uses multiprocessing (not threading) for parallel chains. Each chain runs in its own OS process with its own memory space. Per-chain seeds are derived deterministically from `random_seed + chain_index`. Results are mathematically identical to sequential execution — the posterior samples are independent by construction.

**Alternatives considered:**
- `asyncio` for MCMC — rejected; PyMC's sampler is not async-compatible.
- Free-threaded Python (PEP 703, `python3.14t`) — rejected; PyMC and Polars lack free-threading support. Library incompatibility silently re-enables the GIL, making the optimization invisible and the risk real.
- `InterpreterPoolExecutor` (PEP 734) — rejected; sub-interpreters cannot share complex objects (DataFrames, PyMC models).

### 2. Multi-core XGBoost

Added `n_jobs=-1` to all `XGBClassifier` instances in `prediction.py` and `cross_session.py`. (`RandomForestClassifier` already had `n_jobs=-1`.)

**Safety:** XGBoost's parallelism is internal to its C++ histogram and split-finding code. With `random_state` set, results are deterministic regardless of thread count. `LogisticRegression` with `lbfgs` solver was not changed — it solves a single convex optimization problem and does not benefit from `n_jobs`.

### 3. Wall-clock runtime timing

`RunContext` now records `elapsed_seconds` and `elapsed_display` in `run_info.json` and prints the timing to the console after each phase. The HTML report header displays "Runtime: Xm Ys" alongside the existing session, timestamp, and git hash metadata.

**Implementation:** Simple `datetime.now()` subtraction — no `time.perf_counter()` or CPU timing that could add overhead. Accuracy is ±1 second, which is more than sufficient for detecting order-of-magnitude changes.

### 4. Same-day run preservation

`_next_run_label()` checks for existing date directories before choosing a run label. First run: `2026-02-23/`, second: `2026-02-23.1/`, third: `2026-02-23.2/`. The `latest` symlink always points to the most recent. `run_info.json` records the `run_label` for traceability.

## Consequences

**Benefits:**
- IRT and hierarchical IRT wall-clock time roughly halved (~15 min → ~8 min per chamber with 2 parallel chains).
- XGBoost tree building uses all available cores instead of one.
- Runtime is visible in every report and JSON metadata file, enabling sanity checks across runs.
- Same-day re-runs preserve the original for A/B comparison of results.

**Trade-offs:**
- Parallel MCMC chains double memory usage (each process holds the full model and samples). On 36GB RAM this is not a concern, but on constrained machines (<8GB), `cores=1` can be passed to `pm.sample()` to revert to sequential.
- XGBoost `n_jobs=-1` may compete for CPU with other processes running concurrently. On a shared machine, consider `n_jobs=4` instead.
- Same-day run directories accumulate disk usage. Acceptable given small file sizes (parquet compression, PNGs).
