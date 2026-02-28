# Experiment: nutpie Flat IRT Baseline (Numba)

**Date:** 2026-02-27
**Status:** Complete
**Author:** Claude Code + Joseph Claeys

## The Short Version

nutpie compiles and samples our flat 2PL IRT model without issue. Ideal points correlate |r| = 0.994 with the PyMC baseline (sign-flipped due to IRT reflection invariance — functionally identical rankings). All convergence diagnostics pass with wide margins. This is a green light for Experiment 2 (hierarchical model).

## Why We Ran This Experiment

Before testing nutpie on the hierarchical IRT models that actually need fixing (House convergence: 1/8 sessions pass, joint: 0/8), we need to verify basic compatibility. The flat 2PL IRT model is our simplest MCMC model and has a well-established PyMC baseline. If nutpie can't compile and sample this cleanly, there's no point testing harder models.

Specific questions:
1. Does `nutpie.compile_pymc_model()` succeed on our model structure (`pt.set_subtensor` anchors, `pm.Bernoulli(logit_p=...)`, `pm.Deterministic` with dims)?
2. Does `nutpie.sample()` produce valid InferenceData with standard ArviZ diagnostics?
3. Do the resulting ideal points agree with the PyMC baseline?

## What We Expected to Find

We believe nutpie will compile and sample the flat IRT model without issues because the model uses only standard PyTensor operations, and nutpie's Numba backend supports all of them. We expect ideal points to correlate |r| > 0.99 with the PyMC baseline, with a possible sign flip due to IRT reflection invariance.

## What We Tested

### Baseline (Current Production Model)

- **What it is:** Flat 2PL IRT with anchor-based identification, PyMC NUTS sampler. 91st House: 130 legislators x 297 votes, 722 free parameters.
- **Command:** `just irt`
- **Output directory:** `results/kansas/91st_2025-2026/irt/latest/`

### Run 1: nutpie House

- **What changed:** Sampler switched from PyMC NUTS to nutpie Rust NUTS (Numba backend). Same model, same anchors, same priors.
- **Why:** Verify basic compatibility before testing harder hierarchical models.
- **Command:** `uv run python results/experimental_lab/2026-02-27_nutpie-flat-irt/run_experiment.py`
- **Output directory:** `run_01_house/`

## How We Measured Success

| Metric | What It Tells Us | Passing Value | Source |
|--------|------------------|---------------|--------|
| Compilation | Whether nutpie can compile our PyTensor graph via Numba | Success (no error) | — |
| R-hat(xi) max | Whether chains agree on ideal point estimates | < 1.01 | Vehtari et al. 2021 |
| R-hat(alpha) max | Whether chains agree on bill difficulty | < 1.01 | Vehtari et al. 2021 |
| R-hat(beta) max | Whether chains agree on bill discrimination | < 1.01 | Vehtari et al. 2021 |
| ESS(xi) min | Effective number of independent samples | > 400 | Vehtari et al. 2021 |
| Divergences | Sampler found pathological geometry | 0 | NUTS theory |
| E-BFMI | Sampler energy transitions are efficient | > 0.3 | Betancourt 2016 |
| \|r\| vs PyMC baseline | Agreement with production ideal points | > 0.99 | Internal validation |

## Results

### Summary Table

Values in **bold** pass the threshold.

| Metric | PyMC Baseline | nutpie |
|--------|--------------|--------|
| Compilation | — | **SUCCESS** (13.6s) |
| R-hat(xi) max | — | **1.0036** |
| R-hat(alpha) max | — | **1.0073** |
| R-hat(beta) max | — | **1.0039** |
| ESS(xi) min | — | **1,950** |
| Divergences | — | **0** |
| E-BFMI chain 0 | — | **0.970** |
| E-BFMI chain 1 | — | **1.005** |
| \|r\| vs PyMC | — | **0.9935** |
| Sign flip | — | Yes |
| Compile time | — | 13.6s |
| Sample time | — | 112.8s |
| Total time | — | 126.4s |
| **Verdict** | — | **PASS** |

### What We Observed

**Compilation succeeded cleanly.** nutpie compiled the 722-parameter model through Numba `nopython` mode in 13.6 seconds. Every PyTensor operation — `pt.set_subtensor` for anchor insertion, `pm.Bernoulli(logit_p=...)`, `pm.Deterministic` with coordinate dims — compiled without error or Python fallback.

**All convergence diagnostics pass with wide margins.** R-hat max across all parameters is 1.007 (threshold: 1.01). Minimum ESS is 1,950 (threshold: 400). Zero divergences. E-BFMI values near 1.0 (threshold: 0.3). The flat model's well-conditioned geometry poses no challenge.

**Ideal points match the PyMC baseline (|r| = 0.994).** The raw correlation is negative (r = -0.994) because nutpie found the reflected solution — the mirror image where liberals score positive and conservatives negative. This is a well-known IRT identification artifact, not a nutpie problem. Multiply by -1 and the rankings are identical.

**InferenceData is fully ArviZ-compatible.** R-hat, ESS, BFMI, HDI, NetCDF export all work. `log_likelihood` group is absent (known nutpie gap, issue #150) — not needed for Tallgrass.

### Impact on Rankings and Scores

No impact. The sign-flipped ideal points produce identical legislator rankings after correction. This confirms nutpie is a drop-in replacement for the flat model.

## What We Learned

The hypothesis was confirmed. nutpie compiles and samples flat IRT cleanly. Key takeaways:

1. **Numba backend handles all our PyTensor ops** — no need for JAX on flat models
2. **Sign flip requires post-hoc correction** — production code should check anchor polarity
3. **13.6s compilation overhead** is acceptable (one-time cost per model)
4. **Single-process execution** — Rust threads, no child processes, no orphan risk

This clears the path for Experiment 2 (hierarchical per-chamber) — the model that actually fails convergence.

## Changes Made

No production code changes. ADR-0049 documents the decision.

## Artifacts

| File | Description |
|------|-------------|
| `run_experiment.py` | Experiment script |
| `run_01_house/metrics.json` | Machine-readable results |
| `run_01_house/scatter_nutpie_vs_pymc.png` | Scatter plot (anti-diagonal confirms sign flip) |
| `run_01_house/data/idata_nutpie_house.nc` | Full posterior trace (NetCDF) |

## Related

- `docs/nutpie-deep-dive.md` — Architecture, NF innovation, integration plan
- `docs/nutpie-flat-irt-experiment.md` — Extended write-up with full analysis
- `docs/adr/0049-nutpie-flat-irt-baseline.md` — Decision record

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
