# Apple Silicon Tuning for MCMC Workloads

How we tuned Bayesian MCMC sampling on an M3 Pro, and what we learned about performance and efficiency cores the hard way.

**Hardware:** MacBook Pro M3 Pro, 36 GB unified memory, 6 performance (P) cores + 6 efficiency (E) cores.

---

## The Problem

On February 23, 2026, we ran the full 12-phase analysis pipeline across all eight bienniums (2011-2026) in a batch job. The hierarchical Bayesian IRT phase — the most expensive step, using PyMC's NUTS sampler — ground to a halt. The largest model (91st Legislature, 172 legislators, 491 roll calls, 43,612 observations) was killed after 4 hours without completing. Per-chamber models that normally finish in 7 minutes took 17 minutes. Something was badly wrong.

The immediate suspect was a code change: we had added `cores=n_chains` to all `pm.sample()` calls that same day. We reverted it. The problem persisted. The investigation that followed revealed that the bottleneck wasn't in Python at all — it was in how Apple Silicon schedules work across its two types of CPU cores.

## How Apple Silicon Differs

Apple's M-series chips use a big.LITTLE architecture: fast performance (P) cores paired with power-efficient efficiency (E) cores. On the M3 Pro, that's 6P + 6E. The E-cores deliver roughly **50% of the single-thread throughput** of the P-cores. For web browsing and email, this is a great tradeoff — background tasks run on efficient cores while interactive work gets full speed. For sustained numerical computation, it's a trap.

The critical constraint: **macOS does not expose CPU core affinity APIs on Apple Silicon.** There is no way to pin a process to a specific core. The mechanisms that work on Linux and Windows are all unavailable:

| API | Status on Apple Silicon |
|-----|------------------------|
| `os.sched_setaffinity()` | Linux-only, not available |
| `psutil.Process().cpu_affinity()` | Linux/Windows only |
| Thread Affinity API (`ml_get_max_affinity_sets`) | Returns 0 on arm64 |

The only scheduling lever is Quality of Service (QoS) hints via `taskpolicy`, which are suggestions the OS can override:

| QoS Class | Core Preference | Effect |
|-----------|----------------|--------|
| `background` | E-cores only | 2-3x slower for compute |
| `utility` | Mild P-core preference | Adequate for light work |
| `default` | Strong P-core preference | Normal process behavior |
| `interactive` | Aggressive P-core preference | UI/foreground work |

You cannot force a process onto P-cores. You can only avoid forcing it onto E-cores.

## Three Failure Modes We Discovered

### 1. Thread Pool Spillover

NumPy, SciPy, and PyTensor create internal thread pools (via OpenMP or BLAS) sized to `os.cpu_count()`. On the M3 Pro, that's 12 threads. Since there are only 6 P-cores, at least 6 of those threads land on E-cores running at half speed. For MCMC sampling, which hammers linear algebra on every gradient evaluation, this quietly halves throughput on a significant fraction of the work.

**Fix:** Cap thread pools to the P-core count.

```bash
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
```

We set these globally in our [Justfile](https://github.com/casey/just) command runner so every analysis recipe inherits the cap automatically.

### 2. Sequential Chain Thermal Throttling

MCMC samplers draw multiple independent chains for convergence diagnostics. With `cores=1` (sequential chains), chain 1 runs at full speed on a P-core, saturating it for minutes. By the time chain 2 starts, two things have happened: the chip has accumulated thermal debt and may downclock, and the macOS scheduler may migrate the process to an E-core after extended high-CPU usage.

The result: chain 2 consistently runs at about **half the speed** of chain 1.

We measured this across every model type:

| Model | Chain 1 (draws/s) | Chain 2 (draws/s) | Throttle |
|-------|-------------------|-------------------|----------|
| 91st House | 8.61 | 4.60 | 1.9x |
| 91st Senate | 50.26 | 24.82 | 2.0x |
| 88th House | 23.98 | 11.91 | 2.0x |
| 88th Senate | 75.18 | 16.94 | 4.4x |
| 88th Joint | 2.72 | 1.38 | 2.0x |

**Fix:** Run chains in parallel (`cores=n_chains`). Both chains start simultaneously while the chip is cool and P-cores are available. Neither chain accumulates the thermal/scheduling debt that degrades sequential execution. This gives a consistent **1.8-1.9x wall-clock speedup** — almost perfectly utilizing both cores.

| Model | Sequential | Parallel | Speedup |
|-------|-----------|----------|---------|
| 91st per-chamber | 15m 7s | 8m 6s | 1.87x |
| 88th full (incl. joint) | 50.8 min | 27.7 min | 1.83x |

### 3. Batch Job CPU Saturation

The original failure. Running 8 bienniums simultaneously meant 16+ MCMC processes competing for 6 P-cores. The macOS scheduler distributed them across all 12 cores, and at least half the chains landed on E-cores at 50% speed. Combined with thermal throttling from sustained 100% utilization across every core, each chain ran at roughly 40% of its normal speed — a 2.5x slowdown that turned a 2-hour pipeline into an unfinishable one.

**Fix:** Run bienniums sequentially, never simultaneously. Parallel chains *within* a single biennium are fine (2-4 chains on 6 P-cores has plenty of headroom), but multiple bienniums create the kind of system-wide CPU pressure that triggers E-core scheduling.

### 4. Jitter Mode-Splitting with 4+ Chains

Discovered during the 4-chain experiment (2026-02-26). PyMC's default `jitter+adapt_diag` initialization adds random perturbation to starting values. With PCA-informed initialization and 4 chains, one chain's jitter pushed it past the reflection mode boundary of the IRT posterior, causing R-hat ~1.53 and ESS of 7. The same configuration with 2 chains was fine — the sorted party means constraint provided sufficient mode-breaking.

**Fix:** Use `init='adapt_diag'` (no jitter) when PCA initvals are provided. PCA scores already orient chains correctly; jitter adds noise without benefit. See ADR-0045 and `docs/hierarchical-4-chain-experiment.md`.

## The Rules

These are the five rules we now follow for all CPU-intensive analysis on Apple Silicon:

1. **Cap thread pools to the P-core count.** Set `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to the number of performance cores (6 on M3 Pro). This prevents NumPy/SciPy/PyTensor from spawning threads that land on E-cores.

2. **Run MCMC chains in parallel.** Use `cores=n_chains` (or just accept PyMC's default, which already does this). Parallel chains avoid the thermal throttling that slows sequential chain 2 by ~50%.

3. **Run bienniums sequentially.** Never run multiple MCMC jobs at the same time. The macOS scheduler will push overflow work to E-cores, and the resulting slowdown is worse than just waiting.

4. **Never use `taskpolicy -c background`.** This forces processes onto E-cores exclusively. A background-priority MCMC job runs 2-3x slower with no way to promote it back to P-cores.

5. **Use `adapt_diag` (no jitter) with PCA-informed init and 4+ chains.** Jitter can flip chains past the IRT reflection mode boundary. With 2 chains, the sorted party constraint suffices; with 4+, it doesn't.

## Verifying Determinism

A natural concern with changing `cores`: does it affect results? We verified that all convergence diagnostics are **bit-identical** between `cores=1` and `cores=2`. This was checked across every model type and every diagnostic category:

- R-hat (per-parameter maximums)
- Effective sample size (per-parameter minimums)
- Energy BFMI (per-chain)
- Divergence counts
- Intraclass correlation coefficients
- Group-level parameter estimates and credible intervals
- Correlation with flat IRT ideal points

The `cores` parameter is purely a scheduling decision. PyMC derives per-chain seeds deterministically from `random_seed + chain_index`, so the sampler's trajectory is identical regardless of whether chains execute sequentially or in parallel.

## Practical Impact

Before tuning (batch job, no thread caps):

| Phase | Time |
|-------|------|
| Hierarchical IRT, 91st (per-chamber) | 17+ minutes |
| Hierarchical IRT, 91st (joint) | Killed after 4 hours |
| Full 8-biennium pipeline | Did not complete |

After tuning (sequential bienniums, thread caps, parallel chains):

| Phase | Time |
|-------|------|
| Hierarchical IRT, 91st (per-chamber) | 8 minutes |
| Hierarchical IRT, 91st (joint) | 93 minutes |
| Full 12-phase pipeline, single biennium | ~1h 50m |
| Full 8-biennium pipeline (sequential) | ~15 hours |

The joint model for the 91st Legislature remains the bottleneck at 93 minutes — that's genuine model complexity (491 votes, max tree depth), not a scheduling issue. Everything else runs at or near the hardware's theoretical throughput.

## Applicability Beyond This Project

These findings generalize to any sustained numerical workload on Apple Silicon:

- **PyMC / Stan / NumPyro** — any MCMC sampler that uses multiprocessing for parallel chains
- **scikit-learn** — random forests, gradient boosting with `n_jobs=-1`
- **PyTorch / JAX training loops** — extended GPU-free computation
- **Large matrix operations** — anything that triggers OpenMP/BLAS threading

The core insight: Apple Silicon's heterogeneous architecture is invisible to most Python libraries. They see 12 cores and spawn 12 threads, unaware that half of them run at half speed. The fix is simple — cap your thread pools and avoid concurrent heavy jobs — but you have to know to do it.

## References

- [ADR-0022: Analysis parallelism and runtime timing](adr/0022-analysis-parallelism-and-timing.md) — the architectural decision record
- [Parallel chains experiment](../results/experiments/2026-02-23_parallel-chains-performance/experiment.md) — full data tables and methodology
- Apple Developer Documentation: [Energy Efficiency Guide for Mac Apps](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/)
- PyMC Documentation: [Parallel Sampling](https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/pymc_overview.html)
