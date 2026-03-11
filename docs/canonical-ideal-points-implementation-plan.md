# Implementation Plan: Canonical Ideal Points

**Date:** 2026-03-11
**Prereqs:** ADR-0103, ADR-0104, ADR-0107, ADR-0108
**Article:** `docs/canonical-ideal-points.md`

## Overview

Three changes that flow from the investigation documented in the article:

1. **Fix the regression** — Phase 06 must default to PCA initialization, not horseshoe-confounded 1D IRT
2. **Add Dim 1 forest plot** — the key visual deliverable in the 2D report
3. **Canonical ideal point routing** — auto-select 1D or 2D Dim 1 per chamber for downstream consumption

## Step-by-Step

### Step 1: Fix Phase 06 Init Strategy Default

**File:** `analysis/06_irt_2d/irt_2d.py`

Change the `--init-strategy` default from `auto` to `pca-informed` for the 2D IRT phase specifically. The `auto` strategy prefers 1D IRT scores, which are horseshoe-confounded in exactly the chambers where the 2D model matters most. PCA is always safe.

**Change:**

In `parse_args()`, change the default:

```python
parser.add_argument(
    "--init-strategy",
    default="pca-informed",  # was "auto" — PCA is always safe for 2D init
    ...
)
```

Also update `analysis/init_strategy.py` to add a validation: when the caller is Phase 06 (or more generally, when the caller is a 2D model), log a warning if `auto` selects 1D IRT scores with |r(IRT, PCA)| < 0.7, indicating likely horseshoe contamination.

**Testing:** Existing `test_init_strategy.py` tests pass with default change. Add one test verifying the new warning when correlation is low.

**Complexity:** Low — default value change + optional diagnostic warning.

### Step 2: Add Dim 1 Forest Plot to Phase 06 Report

**Files:** `analysis/06_irt_2d/irt_2d.py`, `analysis/06_irt_2d/irt_2d_report.py`

The 2D IRT report currently has scatter plots (2D, Dim1 vs PC1, Dim2 vs PC2) but no forest plot showing Dim 1 ideal points with HDI bars. This is the single most important visual for the 2D model — it shows the ideology axis as a ranked list with uncertainty.

**Implementation:**

Reuse the `plot_forest()` function from Phase 05 (`analysis/05_irt/irt.py`). It takes a DataFrame with `xi_mean`, `xi_hdi_2.5`, `xi_hdi_97.5`, `full_name`, `party` columns. The 2D parquet already has `xi_dim1_mean`, `xi_dim1_hdi_3%`, `xi_dim1_hdi_97%` — just rename columns and pass through.

Add to `irt_2d_report.py`:

```python
# Build forest-plot-compatible DataFrame from 2D Dim 1 scores
dim1_forest_df = ideal_points_2d.select(
    "full_name", "party", "chamber",
    pl.col("xi_dim1_mean").alias("xi_mean"),
    pl.col("xi_dim1_hdi_3%").alias("xi_hdi_2.5"),  # close enough for display
    pl.col("xi_dim1_hdi_97%").alias("xi_hdi_97.5"),
)
```

Add one forest plot section per chamber to the 2D report, positioned after the 2D scatter plots. Title: "Senate Dim 1 Ideal Points (Ideology Axis)".

**Testing:** Visual inspection on 79th Senate. Huelskamp should be at the far-right end. Smoke test in `test_irt_2d.py`.

**Complexity:** Low — column rename + existing `plot_forest()` reuse.

### Step 3: Canonical Ideal Point Routing

**New file:** `analysis/canonical_ideal_points.py`
**Modified files:** `analysis/06_irt_2d/irt_2d.py` (call the router after sampling), downstream consumers

This is the core architectural change. After Phase 06 completes, a routing function examines both 1D and 2D results and writes a canonical output.

**API:**

```python
from analysis.canonical_ideal_points import write_canonical_ideal_points

write_canonical_ideal_points(
    chamber="Senate",
    irt_1d_dir=irt_dir,        # Phase 05 output
    irt_2d_dir=irt_2d_dir,     # Phase 06 output
    output_dir=canonical_dir,  # Where to write canonical output
    horseshoe_threshold=0.20,  # Democrat wrong-side fraction
    dim1_rhat_threshold=1.05,  # Max acceptable R-hat for 2D Dim 1
    dim1_ess_threshold=200,    # Min acceptable ESS for 2D Dim 1
)
```

**Logic:**

1. Load 1D ideal points from Phase 05
2. Load 2D Dim 1 ideal points from Phase 06
3. Run horseshoe detection on the 1D scores (reuse `detect_horseshoe()` from `irt.py`)
4. If horseshoe detected AND 2D Dim 1 convergence meets thresholds:
   - Write `canonical_ideal_points_{chamber}.parquet` with 2D Dim 1 scores
   - Column mapping: `xi_dim1_mean` → `xi_mean`, `xi_dim1_hdi_3%` → `xi_hdi_2.5`, etc.
   - Add `source` column: `"2d_dim1"` for provenance
   - Log: "Canonical source: 2D Dim 1 (horseshoe detected in 1D, Dim 1 R-hat OK)"
5. If horseshoe NOT detected OR 2D Dim 1 convergence fails:
   - Write `canonical_ideal_points_{chamber}.parquet` with 1D scores
   - Add `source` column: `"1d_irt"` for provenance
   - Log: "Canonical source: 1D IRT"

**Output schema** (matches Phase 05 exactly — zero downstream changes needed):

| Column | Type | Description |
|--------|------|-------------|
| `legislator_slug` | str | Unique identifier |
| `full_name` | str | Display name |
| `party` | str | Republican/Democrat/Independent |
| `chamber` | str | House/Senate |
| `district` | int | District number |
| `xi_mean` | float | Posterior mean ideal point |
| `xi_sd` | float | Posterior standard deviation |
| `xi_hdi_2.5` | float | 2.5% HDI bound |
| `xi_hdi_97.5` | float | 97.5% HDI bound |
| `source` | str | `"1d_irt"` or `"2d_dim1"` (new column) |

**Pipeline integration:**

The canonical output writes to `{run_dir}/canonical_irt/`. This step runs automatically at the end of Phase 06 (since it needs both Phase 05 and Phase 06 results). If Phase 06 is skipped (e.g., `--skip-2d`), canonical falls back to Phase 05 scores.

**Downstream changes:**

Update `analysis/24_synthesis/synthesis_data.py` to load from `canonical_irt/` instead of `05_irt/`. The column schema is identical — the only new column is `source`, which synthesis can optionally report. All other downstream phases (profiles, cross-session, coalition labeling) consume synthesis output and need no changes.

**Testing:**
- Unit tests for the routing logic (horseshoe detected → 2D, not detected → 1D)
- Unit test for column schema matching
- Integration test: 79th Senate → canonical source should be `"2d_dim1"`
- Integration test: 91st Senate → canonical source should be `"1d_irt"` (no horseshoe)

**Complexity:** Medium — new module + synthesis loader change + tests.

### Step 4: Documentation and ADR

**New ADR:** `docs/adr/0109-canonical-ideal-points.md`

Document:
- Context: three failed approaches to fixing 1D IRT, the init strategy regression
- Decision: use 2D Dim 1 as canonical for horseshoe-affected chambers
- Consequences: downstream phases get correct scores automatically, `--dim1-prior` / `--horseshoe-remediate` become research-only flags

**Updates:**
- `CLAUDE.md`: add canonical routing to architecture section
- `docs/adr/README.md`: add ADR-0109
- `.claude/rules/analysis-framework.md`: update pipeline description
- `docs/horseshoe-effect-and-solutions.md`: add "Resolution" section pointing to canonical routing
- `analysis/design/irt.md`: note canonical routing
- `analysis/design/irt_2d.md`: note Dim 1 forest plot, canonical output

### Step 5: Deprecation Flags

Mark `--dim1-prior`, `--horseshoe-remediate`, and `--init-strategy 2d-dim1` as research-only:

- Add deprecation warnings when these flags are used: "Note: this flag is retained for research. Production pipelines use canonical ideal point routing (ADR-0109)."
- Do NOT remove the flags — they're useful for experimentation and the implementation is already tested.
- Update help strings to note "research only" status.

**Complexity:** Low — print warnings + help string updates.

## Execution Order

| Step | Dependency | Estimated Effort |
|------|-----------|-----------------|
| 1. Fix Phase 06 init default | None | Low |
| 2. Dim 1 forest plot | None | Low |
| 3. Canonical routing | Steps 1-2 (needs correct 2D results) | Medium |
| 4. Documentation + ADR | Steps 1-3 | Low |
| 5. Deprecation flags | Step 4 (needs ADR number) | Low |

Steps 1 and 2 can be done in parallel. Step 3 depends on having correct 2D results (Step 1). Steps 4 and 5 are documentation/cleanup.

## Validation

After implementation, re-run the 79th pipeline and verify:

1. **Phase 06 2D scatter** matches the prior (good) run: Democrats on the left, Huelskamp far-right on Dim 1
2. **Phase 06 Dim 1 forest plot** shows clean ideology ordering: Huelskamp and rebels at conservative end, Democrats at liberal end
3. **Canonical output** selects `"2d_dim1"` for Senate, `"1d_irt"` for House
4. **Synthesis** correctly identifies Huelskamp as "Senate Maverick" based on canonical scores
5. **91st biennium** (balanced chamber): canonical output selects `"1d_irt"` for both chambers

## Non-Goals

- **Replacing Phase 05.** The 1D model stays in the pipeline. It's useful as a diagnostic — the divergence between 1D and 2D Dim 1 IS the horseshoe signal. And for balanced chambers, it's the simplest correct model.
- **Auto-promoting for all bienniums without validation.** The initial rollout should be validated on all 14 bienniums (77th-91st) before enabling globally.
- **Modifying the 2D model internals.** Phase 06's PLT identification, convergence criteria, and Bayesian model structure are unchanged.
