# Nutpie: Rust-Based NUTS Sampler — Ecosystem Survey, Code Audit, and Integration Plan

*February 2026*

This document evaluates nutpie as a replacement for PyMC's default NUTS sampler in the Tallgrass hierarchical IRT pipeline. It covers architecture, the normalizing flow innovation, practical migration concerns, and a staged integration plan.

## What nutpie Is

Nutpie is a Python wrapper around [nuts-rs](https://github.com/pymc-devs/nuts-rs), a Rust implementation of the No-U-Turn Sampler (NUTS). Created by Adrian Seyboldt (PyMC-Labs, Flatiron Institute), it compiles PyMC model log-probability gradients through Numba or JAX, then executes MCMC sampling entirely in Rust without Python overhead. All chain parallelism uses Rust threads — no Python `multiprocessing`, no child processes, no orphans.

Current version: 0.16.6 (February 18, 2026). Pre-built wheels for macOS ARM64, Linux x86-64/ARM64, Windows x86-64. Python 3.11+. Maintained under the `pymc-devs` GitHub organization.

## Why It Matters for Tallgrass

Our hierarchical IRT convergence problem (House passes 1/8 bienniums, joint passes 0/8) is caused by the reflection-mode ridge and the non-centered correlation plateau — exactly the class of problem nutpie's normalizing flow adaptation was designed to solve. From Seyboldt's Flatiron Institute CCM Colloquium (February 2025):

> "A centered parameterization of a hierarchical IRT 2PL model with around 1000 total parameters... a nice example due to additive non-identifiability, multiplicative non-identifiability, and funnels from the hierarchical priors."

This is our model. The normalizing flow adaptation was literally developed on a hierarchical 2PL IRT with ~1000 parameters — nearly identical to our joint model's ~1,042 parameters.

## Architecture

### The Compilation Pipeline

```
PyMC Model -> PyTensor Graph -> Backend Compilation -> Rust Sampler
                                    |
                           Numba (default) or JAX
```

1. `nutpie.compile_pymc_model(model)` extracts the log-probability and gradient functions from the PyTensor computation graph
2. These functions are compiled to native code via Numba (default) or JAX
3. The compiled functions are wrapped in a `CompiledModel` that the Rust sampler can call
4. `nutpie.sample(compiled_model, ...)` executes sampling entirely in Rust, calling back to the compiled gradient functions

### Two Backends

**Numba** (default): Compiles PyTensor ops to LLVM via Numba's `nopython` mode. Fast for small-to-medium models. Falls back to Python for unsupported ops (with significant performance degradation — "often more than a factor of 2"). Can fail with `TypingError` on complex model structures.

**JAX**: Better for larger models. Required for normalizing flow adaptation. Can use GPU acceleration. Longer initial compilation but potentially faster sampling for high-dimensional models.

### Thread Model

All chain parallelism is handled by Rust threads within a single Python process. This eliminates:
- The `multiprocessing` overhead (process creation, pickling, pipe communication)
- The orphan process problem (no child processes exist)
- The `fork` vs `spawn` debate on macOS
- The need for `setproctitle` on child processes (there's only one process)

On our M3 Pro (6P + 6E cores), nutpie's Rust thread pool is subject to the same macOS QoS scheduling as any other threaded workload. The `OMP_NUM_THREADS=6` cap in our Justfile affects BLAS operations within the gradient computation but does not directly control nutpie's chain-level parallelism.

## The Normalizing Flow Innovation

### Standard NUTS Mass Matrix

PyMC's default sampler estimates a diagonal mass matrix during warmup from sample covariance. This works when parameters are approximately independent after transformation. It fails when the posterior has strong correlations — exactly our hierarchical IRT situation where ~130 `xi_offset` parameters must move in concert with `sigma_within`.

### nutpie's Standard Adaptation

Even without normalizing flows, nutpie's warmup is fundamentally different from PyMC's:
- Uses **gradient outer products** (L-BFGS-like) for mass matrix estimation instead of sample covariance
- Converges to a better mass matrix faster, requiring fewer tuning steps
- Includes Fisher divergence-based adaptation and low-rank plus diagonal preconditioners

### Normalizing Flow Adaptation (the Key Feature)

The normalizing flow learns a **nonlinear reparameterization** of the posterior during warmup. Instead of just estimating a mass matrix (linear transformation), it learns a bijective function that maps the complex posterior geometry to a simpler space where NUTS mixes efficiently.

On a 100-dimensional funnel (the canonical benchmark for hierarchical models):

| Metric | Without NF | With NF |
|--------|-----------|---------|
| Gradient evaluations | 124,219 | 42,527 |
| Min ESS | ~31.5 | ~1,836 |
| Divergences | Present | Zero |

**Requirements for NF adaptation:**
- JAX backend (not Numba)
- Model should have fewer than ~1000 parameters for effective flow training
- Minimum ~10 minutes runtime (flow training overhead)
- Loss values > 6 indicate the posterior is too difficult for the current flow configuration

**Relevance to our models:**

| Model | Parameters | NF Viable? |
|-------|-----------|------------|
| Flat IRT (per-chamber) | ~500-600 | Yes — well within range |
| Hierarchical per-chamber (Senate) | ~524 | Yes |
| Hierarchical per-chamber (House) | ~694 | Yes — right in the sweet spot |
| Hierarchical joint | ~1,042 | Borderline — at the documented limit |

## API Surface

### Two Integration Paths

```python
# Path 1: Through PyMC (simpler, less control)
with model:
    idata = pm.sample(nuts_sampler="nutpie")

# Path 2: Direct nutpie API (more control, recommended)
compiled = nutpie.compile_pymc_model(model)
idata = nutpie.sample(compiled, draws=2000, tune=1500, chains=4)
```

Path 1 is convenient but **silently ignores `initvals` and `init` parameters**. Path 2 gives full control and is the recommended approach.

### Key Parameters

```python
nutpie.sample(
    compiled_model,
    draws=2000,               # Post-warmup draws per chain
    tune=1500,                # Warmup draws per chain
    chains=4,                 # Number of parallel chains
    cores=4,                  # Rust threads for parallel chains
    seed=42,                  # Random seed
    init_mean=None,           # Initialization point (unconstrained space)
    save_warmup=False,        # Retain warmup draws
    blocking=True,            # False for non-blocking API
    progress_bar=True,        # Terminal progress display
    # Diagnostic storage (opt-in):
    store_divergences=True,
    store_mass_matrix=True,   # Useful for identifying convergence issues
    store_unconstrained=True, # Transformed parameters
    store_gradient=True,      # Gradient information
)
```

### The Non-Blocking API

```python
compiled = nutpie.compile_pymc_model(model)
sampler = nutpie.sample(compiled, draws=2000, chains=4, blocking=False)

# Returns immediately. Then:
sampler.is_finished    # Check completion (bool)
sampler.inspect()      # Get partial trace as InferenceData (copy, non-destructive)
sampler.pause()        # Pause all chains
sampler.resume()       # Resume paused chains
sampler.abort()        # Stop and return partial trace
sampler.cancel()       # Stop and discard everything
sampler.wait(timeout=60)  # Block until done or timeout
```

`inspect()` is the standout feature — you can examine the partial posterior at any point during sampling without interrupting it. This replaces the entire file-based monitoring approach described in the experiment framework deep dive: instead of writing a JSON status file every 50 draws, you call `sampler.inspect()` and get the actual partial InferenceData.

### Progress Bar

nutpie uses Rust's `indicatif` crate for terminal progress bars. The display shows per-chain:
- Current draw count
- Step size (small values indicate problems)
- Divergence count
- Gradients per draw (100-1000+ suggests poor parameterization)

## Output Format and Diagnostics

### InferenceData Compatibility

`nutpie.sample()` returns a standard `arviz.InferenceData` object. All standard ArviZ operations work:

```python
az.rhat(idata)           # R-hat convergence
az.ess(idata)            # Bulk ESS
az.ess(idata, method="tail")  # Tail ESS
az.bfmi(idata)           # E-BFMI
az.hdi(idata)            # HDI intervals
az.plot_trace(idata)     # Trace plots
idata.posterior["xi"].mean(dim=["chain", "draw"])  # Point estimates
idata.sample_stats["diverging"]  # Divergence tracking
idata.to_netcdf("trace.nc")     # Persistent storage
```

### Known Gaps

Two InferenceData groups are missing from nutpie's output:

1. **`log_likelihood`** — Element-wise log-likelihood is not computed ([issue #150](https://github.com/pymc-devs/nutpie/issues/150), still open). Workaround: call `pm.compute_log_likelihood(idata)` after sampling. This means `az.loo()`, `az.waic()`, and `az.compare()` require a post-processing step.

2. **`constant_data`** — Not stored in the output ([issue #74](https://github.com/pymc-devs/nutpie/issues/74)).

Neither is a blocker for Tallgrass. We don't currently use LOO-CV or WAIC (our model comparison uses correlation with flat IRT and Shor-McCarty scores). And `constant_data` is not referenced in our downstream analysis.

### Diagnostic Differences

**Important caveat**: nutpie can produce different convergence signatures than PyMC's default sampler for the same model. From a PyMC Discourse thread where a survival analysis model showed divergences with default NUTS but zero divergences with nutpie — yet both posteriors were problematic:

> "Different mass matrices produce different U-turn conditions, affecting trajectory tree size." — Bob Carpenter

This means convergence diagnostics may disagree between samplers. A model that shows R-hat > 1.01 with PyMC might show R-hat < 1.01 with nutpie (or vice versa). This is not a bug — it reflects genuinely different exploration of the same posterior. We should compare resulting ideal point estimates, not just diagnostics.

## The initvals Question

This is the sharpest edge for Tallgrass integration. Our PCA-informed initialization strategy (ADR-0023, ADR-0044, ADR-0045) is critical for hierarchical IRT convergence:

**Current PyMC approach:**
```python
pm.sample(
    initvals={"xi_offset": pca_initvals},  # Per-variable dict
    init="adapt_diag",                      # No jitter — PCA provides orientation
)
```

All chains receive the same initvals. The `adapt_diag` mode applies deterministic perturbation from the random seed, but no random jitter — this prevents the mode-splitting we discovered in the 4-chain experiment (ADR-0045).

**nutpie's approach:**
```python
nutpie.sample(
    compiled_model,
    init_mean=flat_vector,  # Single point in unconstrained space
)
```

Key differences:
- `init_mean` is a **flat vector** in unconstrained (transformed) space, not a per-variable dict
- nutpie applies its own jitter around `init_mean` — this cannot be disabled
- `init_mean` is a single point; all chains are jittered from the same mean
- When `init_mean=None`, defaults to zeros in unconstrained space

**Integration approach:**
1. After building the PyMC model, use `compiled_model.initial_point` to get the default unconstrained starting point
2. Map our PCA initvals to unconstrained space using PyMC's value transforms
3. Pass as `init_mean`
4. Accept that nutpie adds jitter — test whether this causes mode-splitting (it may not, because nutpie's mass matrix adaptation is better at resolving modes than PyMC's `adapt_diag`)

Alternatively, if nutpie's normalizing flow adaptation resolves the reflection mode entirely (as the theory and benchmarks suggest it should), PCA initialization may become unnecessary — the flow would learn the correct reparameterization during warmup regardless of starting point.

## Tallgrass Compatibility Audit

### What Our Models Use

| Feature | Used? | nutpie Support |
|---------|-------|---------------|
| `pm.Normal`, `pm.HalfNormal` | Yes | Full support |
| `pm.LogNormal` | Experimental | Full support |
| `pm.Bernoulli(logit_p=...)` | Yes (likelihood) | Full support |
| `pm.Deterministic` | Yes (11 instances) | Full support |
| `pt.sort()` | Yes (identification) | Should work — standard PyTensor op |
| `pt.set_subtensor()` | Yes (anchor insertion) | Should work |
| `pt.concatenate()` | Yes (joint model) | Should work |
| `dims` and `coords` | Yes (everywhere) | Full support |
| `initvals` dict | Yes (PCA init) | Requires conversion to flat unconstrained vector |
| `init="adapt_diag"` | Yes (no-jitter) | Not supported — nutpie has own initialization |
| `callback` parameter | Not yet (proposed) | Not supported — use non-blocking API instead |
| `cores` parameter | Yes | Maps to Rust thread count |

### Downstream ArviZ Usage

All of these work with nutpie's output:
- `az.rhat()`, `az.ess()`, `az.ess(method="tail")`, `az.bfmi()`, `az.hdi()`
- `az.plot_trace()`
- `idata.posterior[var].mean(dim=[...])`, `.values`, `.sel()`
- `idata.sample_stats["diverging"]`
- `idata.to_netcdf()`

Not used in Tallgrass (and missing from nutpie output): `az.loo()`, `az.waic()`, `az.compare()`.

## Performance Expectations

### Published Benchmarks

On posteriordb benchmarks (heterogeneous collection of Bayesian models), nutpie averages ~2x faster than Stan. On models with correlated posteriors (hierarchical, non-centered), the advantage grows with normalizing flow adaptation.

From the PyMC fast-sampling tutorial, on a probabilistic PCA model (5000 data points, 2 dimensions):

| Sampler | Time | Speedup |
|---------|------|---------|
| PyMC default NUTS | 47.6s | 1x |
| nutpie (Numba) | 16.1s | 3x |
| NumPyro (JAX) | 12.9s | 3.7x |
| BlackJAX (JAX) | 11.6s | 4.1x |

### Expected Impact on Tallgrass

| Model | Current Time | Expected with nutpie | Notes |
|-------|-------------|---------------------|-------|
| Flat IRT (per-chamber) | ~10-20 min | ~3-7 min | Numba backend sufficient |
| Hierarchical per-chamber (Senate) | ~5-8 min | ~2-4 min | Numba; NF optional |
| Hierarchical per-chamber (House) | ~7-12 min | ~3-5 min (Numba) or potentially much better with NF | NF may resolve convergence |
| Hierarchical joint | ~31 min | Unknown | NF required; at parameter limit |
| Full hierarchical pipeline | ~40 min | ~15-25 min (estimate) | Biggest gain from NF on House |

The headline number is not wall-clock speedup — it's that models which currently **fail to converge** (House per-chamber: 7/8 fail, joint: 8/8 fail) might converge with nutpie's normalizing flow adaptation, because the flow learns the reparameterization that resolves the reflection mode and correlation plateau.

## Known Risks and Failure Modes

### PyTensor C Compiler Prerequisite

Before any MCMC experiment (nutpie or PyMC default), verify that PyTensor has a working C++ compiler. Without it, PyTensor falls back to pure Python mode — **18x slower** (131 min/chain vs 7 min/chain, measured on the positive-beta experiment 2026-02-27).

**Root cause**: PyTensor checks for `g++`/`clang++` at import time. Two known failure modes on macOS:

1. **Xcode license not accepted**: After an automatic Xcode update, `g++`/`clang++` exist on `PATH` but return a license error instead of compiling. Fix: open Xcode.app and accept the agreement, or run `sudo xcodebuild -license accept`.
2. **Stripped `PATH`**: Background processes or non-login shells may have a `PATH` missing `/usr/bin`. The Justfile now exports `PATH` with `/usr/bin` prepended.

For scripts run outside the Justfile:

```python
import pytensor
if not pytensor.config.cxx:
    raise RuntimeError(
        "PyTensor has no C++ compiler — sampling will be ~18x slower. "
        "Ensure /usr/bin is on PATH or run via `just`."
    )
```

This check belongs in the `PlatformCheck.validate()` method of the experiment runner (see `docs/experiment-framework-deep-dive.md`).

### Compilation Failures (nutpie-specific)

Numba's `nopython` mode cannot compile all PyTensor operations. Known failures:
- `TypingError: Failed in nopython mode pipeline` — complex dimension handling, pymc-marketing's adstock functions
- `LLVM ERROR: Symbol not found: __powidf2` — kernel crash, Numba/LLVM compatibility
- Dimension-1 arrays can trigger bugs

**Mitigation**: If Numba fails, switch to JAX backend. If both fail, fall back to PyMC default sampler.

### Memory

nutpie stores warmup draws and sampler statistics more aggressively than PyMC. Peak memory reported as ~2x trace size. For our hierarchical joint model (~170 legislators x ~430 votes x 2000 draws x 4 chains), this could be significant. The `zarr_store` parameter (added in 0.16.0) mitigates by streaming to disk.

### Masking Model Problems

nutpie's different mass matrix adaptation can produce zero divergences on models where PyMC shows many divergences — but both posteriors may be problematic. The absence of divergences with nutpie is not proof of convergence. Always check R-hat, ESS, and domain-specific validation (ideal point correlations, Shor-McCarty agreement).

### The 1000-Parameter NF Limit

Our joint model has ~1,042 parameters — right at the documented limit for normalizing flow effectiveness. This could go either way. The per-chamber models (524-694 parameters) are safely within range.

## Integration Plan

### Experiment 1: Flat IRT Baseline (Low Risk) — COMPLETE, PASS

**Run date:** 2026-02-27. **Full results:** `docs/nutpie-flat-irt-experiment.md`, **ADR:** ADR-0049.

Compiled the 91st House flat 2PL IRT (130 legislators × 297 votes, 722 free parameters) with nutpie 0.16.6 Numba backend. Compilation: 13.6s, clean. Sampling: 2×2000 draws in 112.8s. All convergence diagnostics pass with wide margins (R-hat max 1.004, ESS min 1,950, zero divergences). Ideal points correlate |r| = 0.994 with PyMC baseline (sign flip due to IRT reflection invariance — correctable post-hoc).

Key findings:
- Numba `nopython` mode handles all PyTensor ops in our IRT models (`pt.set_subtensor`, `pm.Bernoulli(logit_p=...)`, `pm.Deterministic` with dims)
- InferenceData fully compatible with ArviZ (R-hat, ESS, BFMI, HDI, NetCDF)
- Single-process execution confirmed (Rust threads, no child processes)
- `log_likelihood` group absent as expected (nutpie issue #150)

### Experiment 2: Hierarchical Per-Chamber with Numba — COMPLETE

**Script:** `results/experimental_lab/2026-02-27_nutpie-hierarchical/run_experiment.py`

Tested the hierarchical IRT on both chambers (4 chains, 2000 draws, 1500 tune, no PCA init).

**Results:** House ALL PASSED (R-hat 1.004, ESS 1,294). Senate FAILED — reflection mode-splitting without PCA init (R-hat 1.53, ESS 7). The House has enough data (130 legislators, 35k observations) for nutpie's mass matrix adaptation to break the reflection symmetry. The Senate's smaller dataset (42 legislators, 7.7k observations) does not.

### Experiment 2b: Hierarchical + PCA Init — COMPLETE

**Script:** `results/experimental_lab/2026-02-27_nutpie-hierarchical/run_experiment_pca_init.py`

Added PCA-informed `xi_offset` initialization to fix the Senate mode-splitting.

**Key finding — `jitter_rvs` matters:** Setting `jitter_rvs=set()` (disable all jitter) causes `sigma_within` (HalfNormal) to initialize at its support point (~0), producing `log(0) = -inf` in unconstrained space → "Invalid initial point." The fix: `jitter_rvs = {rv for rv in model.free_RVs if rv.name != "xi_offset"}` — jitter all RVs except the PCA-initialized one.

**Results:** Both chambers ALL PASSED. Senate: R-hat 1.001, ESS 1,658 (was 1.53/7 without PCA). House: R-hat 1.003, ESS 1,204. Ideal points identical to PyMC production (|r| = 1.0000 both chambers). Sampling time comparable (House 201s, Senate 42s).

**Conclusion:** nutpie + PCA init is ready for production use on per-chamber models. NF adaptation is unnecessary for per-chamber — only needed for the joint model.

### Experiment 3: Normalizing Flow on Joint Model (High Value, Medium Risk)

Test whether NF adaptation resolves the joint cross-chamber convergence failure (0/8 bienniums pass with PyMC).

1. Compile with JAX backend (`gradient_backend="jax"`)
2. Enable normalizing flow adaptation
3. Run on 91st joint model (1,042 parameters — at NF limit)
4. Check: R-hat (target < 1.01), ESS (target > 400), divergences, NF loss value
5. Compare ideal points with per-chamber hierarchical baselines

**Key question**: Can NF push the joint model past the convergence threshold, or is 1,042 parameters beyond the limit?

### Experiment 4: Joint Model with NF (High Risk, High Reward)

Test the joint cross-chamber model that currently fails 8/8.

1. Compile with JAX backend + NF adaptation
2. Run on 91st joint model (1,042 parameters — at NF limit)
3. Monitor NF loss value — if > 6, the model may be too difficult
4. Compare ideal points with per-chamber hierarchical baselines

**Key question**: Can NF push the joint model past the convergence threshold, or is 1,042 parameters beyond the limit?

### Production Migration (Only After Per-Chamber Experiments Pass)

Per-chamber experiments 1-2b have all succeeded. Production migration path:

1. Add `nutpie` as a required dependency (not optional)
2. Create a `sampler` module in `analysis/` that wraps the PyMC-to-nutpie transition
3. Update `build_per_chamber_model()` and `build_and_sample()` to use nutpie
4. Replace `pm.sample(callback=...)` monitoring with `sampler.inspect()` non-blocking pattern
5. Update convergence baselines in `docs/analytic-flags.md` (R-hat/ESS numbers will change)
6. Update Apple Silicon tuning docs (no multiprocessing concerns)
7. Run the full 8-biennium pipeline and compare with PyMC baseline
8. Write ADR documenting the migration decision

### What We Keep Regardless

- PCA-informed initialization logic (may still be useful as `init_mean`)
- ArviZ-based diagnostics pipeline
- Production convergence thresholds (R-hat < 1.01, ESS > 400)
- The experiment framework's `ExperimentConfig` and `BetaPriorSpec` patterns
- `setproctitle` on the parent process (single process, but still useful for `ps` visibility)

## The Monitoring Story Simplifies

With nutpie, the three-layer monitoring approach from the experiment framework deep dive collapses:

| Layer | With PyMC (multiprocessing) | With nutpie (single process) |
|-------|---------------------------|------------------------------|
| Process titles | Parent + 4 child titles | One process title |
| Status file | PyMC callback writes JSON | `sampler.inspect()` returns actual InferenceData |
| PID/process group | PID lock + `os.setpgrp()` for 5 processes | PID lock for 1 process |
| Orphan detection | `pgrep -f tallgrass` for stray workers | Not possible — no child processes |
| Emergency kill | `kill -TERM -$PGID` (process group) | `kill $PID` (single process) |

The non-blocking API (`blocking=False` + `sampler.inspect()`) is strictly more powerful than the file-based heartbeat approach, because you get the actual partial posterior — not just a progress counter.

## References

- Seyboldt, A. (2025). Normalizing flow adaptation for NUTS. Flatiron Institute CCM Colloquium. [Slides](https://indico.flatironinstitute.org/event/4037/)
- Sountsov, P. & Suter, C. (2022). Automatically adapting the number of state variables for non-centered parameterizations. *Bayesian Analysis* (forthcoming).
- [nutpie GitHub](https://github.com/pymc-devs/nutpie)
- [nuts-rs GitHub](https://github.com/pymc-devs/nuts-rs)
- [nutpie documentation](https://pymc-devs.github.io/nutpie/)
- [nutpie NF adaptation docs](https://pymc-devs.github.io/nutpie/nf-adapt.html)
- [nutpie sampling options](https://pymc-devs.github.io/nutpie/sampling-options.html)
- [PyMC fast sampling tutorial](https://www.pymc.io/projects/examples/en/latest/samplers/fast_sampling_with_jax_and_numba.html)
- [nutpie on PyPI](https://pypi.org/project/nutpie/)
- [Stan discourse: nutpie with NF](https://discourse.mc-stan.org/t/nutpie-now-with-normalizing-flow-adaptation/37293)
- [PyMC discourse: nutpie sampling issues](https://discourse.pymc.io/t/nutpie-sampling-issue/14366)
- [PyMC discourse: fastest sampler for hierarchical MMMs](https://discourse.pymc.io/t/what-sampler-should-i-use-for-the-fastest-inference-in-hierarchical-bayesian-mmm/17447)
- [Gelman blog: improvements to NUTS](https://statmodeling.stat.columbia.edu/2025/12/11/a-slew-of-improvements-to-nuts/)
- [nutpie log-likelihood issue #150](https://github.com/pymc-devs/nutpie/issues/150)

## Related Documents

- [Experiment Framework Deep Dive](experiment-framework-deep-dive.md) — monitoring, code duplication, platform constraints
- [Hierarchical Convergence Improvement](hierarchical-convergence-improvement.md) — the 9-priority plan; nutpie may leapfrog priorities 1-6
- [Apple Silicon MCMC Tuning](apple-silicon-mcmc-tuning.md) — P/E core scheduling (partially obsoleted by nutpie's single-process model)
- [Hierarchical 4-Chain Experiment](hierarchical-4-chain-experiment.md) — jitter mode-splitting discovery (nutpie may handle differently)
- [ADR-0044](adr/0044-hierarchical-pca-informed-init.md) — PCA init (needs different plumbing with nutpie)
- [ADR-0045](adr/0045-4-chain-hierarchical-irt.md) — adapt_diag no-jitter strategy (nutpie has own init)
