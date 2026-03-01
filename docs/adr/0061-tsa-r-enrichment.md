# ADR-0061: TSA R Enrichment (CROPS + Bai-Perron)

**Date:** 2026-02-28
**Status:** Accepted

## Context

Phase 15 (TSA) uses Python's `ruptures` library for PELT changepoint detection, which provides point estimates of break locations with a manually-selected penalty parameter. The TSA deep dive identified two gaps:

1. **CROPS penalty selection** — the 25-point `np.linspace(1, 50, 25)` grid approximates the exact solution path. True CROPS (Haynes et al. 2017) finds the *exact* penalty thresholds where the optimal segmentation changes. Python's `ruptures` does not implement CROPS; R's `changepoint` package does.

2. **Bai-Perron confidence intervals** — PELT gives point estimates of break locations. Bai-Perron (1998, 2003) provides formal 95% CIs on break dates, the econometric gold standard. R's `strucchange` package implements this.

Phase 17 (W-NOMINATE) established the R subprocess pattern: temp CSV input → `Rscript` → JSON output.

## Decision

Add R-based CROPS and Bai-Perron as optional enrichments to Phase 15 TSA, using the Phase 17 subprocess pattern:

- **New file `tsa_r_data.py`** — pure parsing/conversion logic (no I/O), fully testable
- **New file `tsa_strucchange.R`** — single R script handling both CROPS (`changepoint::cpt.mean`) and Bai-Perron (`strucchange::breakpoints` + `confint`)
- **`--skip-r` flag** — explicit opt-out; auto-detection when flag not set
- **R is optional** — Python-only PELT always runs; R adds enrichment when available
- **Per-party analysis** — CROPS and Bai-Perron operate on 1D signals (per-party weekly Rice)
- **JSON output** — R writes JSON; Python parses it (matches Phase 17 pattern)

## Consequences

**Benefits:**
- Completes all 7 TSA deep dive recommendations
- CROPS provides exact penalty solution path, replacing the 25-point approximation
- Bai-Perron provides formal 95% CIs on break dates
- Cross-referencing PELT with Bai-Perron confirms breaks via independent methods
- R is optional — CI/pipeline always works without it

**Trade-offs:**
- Requires R + 3 packages for full functionality (not required for basic TSA)
- R subprocess adds ~2-5 seconds per party per chamber
- Two methods (PELT + Bai-Perron) may produce slightly different break locations (this is expected — different statistical frameworks)

**New test coverage:** 21 new tests covering all `tsa_r_data` functions + `check_tsa_r_packages`. All synthetic data, no R calls.
