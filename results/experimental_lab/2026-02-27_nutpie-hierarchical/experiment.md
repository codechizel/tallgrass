# Experiment: nutpie Hierarchical Per-Chamber IRT (Numba)

**Date:** 2026-02-27
**Status:** Planning
**Author:** Claude Code + Joseph Claeys

## The Short Version

*Does switching from PyMC's built-in NUTS sampler to nutpie's Rust-based NUTS fix the House convergence failure in our hierarchical IRT model? We compile the same model, sample with nutpie (no PCA init, no normalizing flows), and compare convergence diagnostics and ideal points against both the PyMC hierarchical and flat IRT baselines. Results below.*

## Why We Ran This Experiment

Our hierarchical Bayesian IRT model — which estimates where each Kansas legislator falls on a liberal-to-conservative spectrum while accounting for party membership — produces reliable results for the Senate but fails to converge for the House in 7 of 8 legislative sessions. The House has more legislators (130 vs 40), more bills (~280 vs ~240), and a more complex posterior geometry. The PyMC sampler gets stuck exploring a ridge created by the reflection symmetry in the model (the same bill can be interpreted as "discriminating conservative" or "discriminating liberal" — mathematically equivalent solutions).

Experiment 1 showed that nutpie compiles and samples our *flat* IRT model correctly (|r| = 0.994 vs PyMC). Now we test the model that actually matters — the hierarchical per-chamber model that currently fails on House.

nutpie uses a Rust implementation of the NUTS sampler with a different mass matrix adaptation algorithm than PyMC's. Even without normalizing flows (Experiment 3), the different adaptation may handle the posterior geometry better.

## What We Expected to Find

We believe that nutpie's Rust NUTS implementation, even without normalizing flow adaptation, may resolve the House convergence failure because:
1. Different mass matrix adaptation could better navigate the non-centered hierarchy's "funnel" geometry
2. Rust-native chain parallelism eliminates Python multiprocessing overhead
3. The absence of PCA initialization (which we're deliberately omitting) tests whether nutpie can find the mode on its own

If nutpie does NOT fix convergence, the problem is genuinely model-structural (the reflection ridge), not sampler-related — and Experiment 3 (normalizing flows) becomes critical.

## What We Tested

### Baseline (Current Production Model)

- **What it is:** Per-chamber hierarchical IRT with `beta ~ Normal(0, 1)`, PCA-informed initialization, 4 chains, `adapt_diag`, PyMC NUTS sampler.
- **Command:** `just hierarchical`
- **Output directory:** `results/kansas/91st_2025-2026/hierarchical/latest/`

### Run 1: nutpie House

- **What changed:** Sampler switched from PyMC NUTS to nutpie Rust NUTS (Numba backend). PCA initialization removed — chains start from zeros.
- **Why:** House is the problematic chamber. If nutpie fixes convergence here, it's the primary finding.
- **Command:** `uv run python results/experiments/2026-02-27_nutpie-hierarchical/run_experiment.py`
- **Output directory:** `run_01_house/`

### Run 2: nutpie Senate

- **What changed:** Same as Run 1, but on Senate.
- **Why:** Senate already converges with PyMC — this confirms nutpie doesn't break what already works and provides a timing comparison.
- **Command:** (runs automatically after Run 1)
- **Output directory:** `run_02_senate/`

## How We Measured Success

| Metric | What It Tells Us | Passing Value | Source |
|--------|------------------|---------------|--------|
| R-hat(xi) max | Whether chains agree on ideal point estimates | < 1.01 | Vehtari et al. 2021 |
| R-hat(mu_party) max | Whether chains agree on party mean estimates | < 1.01 | Vehtari et al. 2021 |
| R-hat(sigma_within) max | Whether chains agree on within-party spread | < 1.01 | Vehtari et al. 2021 |
| ESS(xi) min | Effective number of independent samples for ideal points | > 400 (100/chain) | Vehtari et al. 2021 |
| Divergences | Sampler found pathological geometry | 0 | NUTS theory |
| E-BFMI | Sampler energy transitions are efficient | > 0.3 | Betancourt 2016 |
| \|r\| vs PyMC hierarchical | Agreement with current production estimates | > 0.95 | Internal validation |
| \|r\| vs flat IRT | Agreement with simpler, independently validated model | > 0.90 | External validation (Shor-McCarty) |

## Results

### Summary Table

*To be filled after experiment runs.*

| Metric | PyMC Baseline (House) | nutpie House | PyMC Baseline (Senate) | nutpie Senate |
|--------|----------------------|--------------|----------------------|---------------|
| R-hat(xi) max | | | | |
| R-hat(mu_party) max | | | | |
| R-hat(sigma_within) max | | | | |
| ESS(xi) min | | | | |
| Divergences | | | | |
| E-BFMI min | | | | |
| \|r\| vs PyMC hier | — | | — | |
| \|r\| vs flat IRT | | | | |
| Compile time (s) | — | | — | |
| Sample time (s) | | | | |

### What We Observed

*To be filled after experiment runs.*

### Impact on Rankings and Scores

*To be filled after experiment runs.*

## What We Learned

*To be filled after experiment runs.*

## Changes Made

*To be filled after experiment runs.*

---

## Default Session

Unless otherwise noted, all experiments use the **91st biennium (2025-26)** as the test session. This is the current session; the primary analyst (Sen. Joseph Claeys) has content knowledge of the legislators and can spot anomalies in the results. Each experiment produces a full HTML report so that downstream impacts on rankings, tables, and plots can be visually inspected.

## File Organization

Experiment directories use: `YYYY-MM-DD_short-description/`

Each experiment directory contains:
- `experiment.md` — this document (copy from TEMPLATE.md)
- `run_experiment.py` — the script that runs the experiment
- `run_NN_description/` — output directories for each run (numbered sequentially)
- Any supporting scripts, logs, or data specific to this experiment

Results are append-only: new runs add rows to the results table; old results are never deleted.
