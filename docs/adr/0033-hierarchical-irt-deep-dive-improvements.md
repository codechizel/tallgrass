# ADR-0033: Hierarchical IRT Deep Dive Improvements

**Date:** 2026-02-25
**Status:** Accepted

## Context

A comprehensive code audit and ecosystem survey of the Hierarchical Bayesian IRT implementation (Phase 10) was conducted as `docs/hierarchical-irt-deep-dive.md`. The audit confirmed mathematical correctness but identified 9 code issues (2 substantive, 1 code defect, 6 code quality), 8 test gaps (including 1 defective existing test), and 2 internal documentation inaccuracies.

The ecosystem survey covered 16 packages (7 R, 5 Python, 4 PPL frameworks) and confirmed that no Python library offers party-level hierarchical IRT for legislative analysis. The closest R equivalents are `emIRT::hierIRT()` and `MCMCpack::MCMCirtHier1d()`.

## Decision

Implement all identified fixes:

### Code Changes

1. **Small-group warning** (`MIN_GROUP_SIZE_WARN = 15`): `prepare_hierarchical_data` prints a WARNING when a party has fewer than 15 legislators. The James-Stein estimator guarantees shrinkage dominance only for J >= 3 groups; with J=2 and small N, the hierarchical model may over-shrink (see `docs/hierarchical-shrinkage-deep-dive.md` for the Senate case study).

2. **Retain `flat_xi_rescaled`**: Removed `.drop("flat_xi_rescaled")`. The linearly rescaled flat IRT ideal points now persist in the output parquet, enabling consistent-scale scatter plots and downstream cross-session comparison.

3. **Scatter plot uses rescaled values**: `plot_shrinkage_scatter` now uses `flat_xi_rescaled` for both the x-axis and annotation positions, eliminating the mixed-scale methodology where axes showed raw values but "top 5 movers" labels used scale-corrected deltas.

4. **`SHRINKAGE_MIN_DISTANCE = 0.5`**: Extracted magic number for the minimum flat-to-party-mean distance below which `shrinkage_pct` is null.

5. **`HIER_CONVERGENCE_VARS` / `JOINT_EXTRA_VARS`**: Extracted hardcoded variable name lists to module-level constants.

6. **ICC columns renamed** `icc_hdi_*` → `icc_ci_*`: The ICC credible interval is computed via `np.percentile` (equal-tailed), not `az.hdi()` (highest-density). Column names now accurately reflect the computation. Updated in `hierarchical.py`, `hierarchical_report.py`, and test assertions.

7. **`extract_group_params` guard**: Now raises `ValueError` with a clear message when called with joint model InferenceData (which has `mu_group` instead of `mu_party`), replacing a silent `KeyError`.

8. **Docstring fixes**: `compute_variance_decomposition` docstring corrected from "mean" to "group-size-weighted mean". `HIERARCHICAL_PRIMER` updated to include the ordering constraint in the joint model formula.

### Test Changes (26 → 35 tests)

| Test | What it covers |
|------|---------------|
| `test_small_group_triggers_warning` | Party with < 15 legislators prints WARNING |
| `test_large_groups_no_warning` | Parties at >= 15 legislators produce no WARNING |
| `test_joint_ordering_per_chamber` | Joint model sorts each chamber's pair independently |
| `test_joint_ordering_preserves_already_sorted` | Already-sorted pairs pass through unchanged |
| `test_fallback_with_two_matches` | Rescaling fallback to slope=1.0 with <= 2 matches |
| `test_highly_unequal_groups` | ICC valid with 20R/3D (Kansas-like proportions) |
| `test_joint_model_data_raises` | `extract_group_params` raises ValueError for joint data |
| `test_independent_excluded` | Independent legislators correctly excluded |
| `test_no_independents_no_exclusion` | No exclusion when no Independents present |

Fixed existing tests:
- `test_correlation_handles_missing`: Replaced tautological `assert not np.isnan(r) or True` with `assert np.isnan(r)`
- `test_shrinkage_toward_party_mean`: Now uses explicitly shrunk fixture data and verifies majority shrink toward party mean
- `test_icc_schema`: Updated for `icc_ci_*` column names
- `test_shrinkage_columns_with_flat`: Added `flat_xi_rescaled` assertion

## Consequences

**Benefits:**
- Users see a proactive warning when hierarchical shrinkage is unreliable for small groups, instead of discovering the problem only via convergence diagnostics or external validation
- Scatter plots are now visually accurate (same scale on both axes)
- `flat_xi_rescaled` available for downstream phases (cross-session, profiles)
- Column names accurately describe their computation method
- `extract_group_params` fails clearly instead of with a cryptic KeyError
- 35 tests provide comprehensive regression coverage

**Trade-offs:**
- ICC column rename (`icc_hdi_*` → `icc_ci_*`) is a breaking change for any downstream code reading variance decomposition parquets. All known consumers (report builder, ICC plot) have been updated.
- `flat_xi_rescaled` adds one column to the output parquet (negligible storage cost)

**No behavioral changes:** All fixes are code quality improvements. The model specification, priors, sampling strategy, and identification approach are unchanged.
