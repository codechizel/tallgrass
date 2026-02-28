# ADR-0042: Joint Model Sign Fix, Pipeline Hardening, and Joint Model Diagnosis

**Date:** 2026-02-26
**Status:** Accepted

## Context

Running the full 12-phase analysis pipeline across all 8 bienniums (84th-91st, 2011-2026) revealed three runtime crashes and a fundamental design flaw in the joint cross-chamber hierarchical IRT model.

### Joint model sign indeterminacy

The joint hierarchical IRT model combines House and Senate data into a single MCMC model. Due to the multiplicative sign symmetry of the 2PL IRT likelihood (`beta * xi = (-beta) * (-xi)`), the model's sign convention can flip independently for each chamber's subset of votes. The within-chamber ordering constraint (`pt.sort` ensures D < R) is necessary but not sufficient: it constrains relative ordering within a chamber but not the absolute sign convention across chambers.

In both the 91st and 90th sessions, the Senate consistently converged with the opposite sign convention (Democrats positive, Republicans negative). Post-hoc comparison with the per-chamber hierarchical models (which have correct sign convention) confirmed the flip.

### Joint model bill-matching bug

A deeper investigation revealed that the joint model has **zero shared bill parameters** between chambers. The data combination step (`build_joint_model()`, line 418) deduplicates by `vote_id`, which is always unique per roll call. Even when both chambers vote on the same bill, they have different vote_ids. This means the two chambers' likelihoods are completely separable — connected only by the soft hierarchical prior, which is insufficient to establish a common measurement scale.

The flat IRT's `build_joint_vote_matrix()` correctly matches bills across chambers by `bill_number`, finding 71-133 shared bills per session. The hierarchical model should do the same but does not. See `docs/joint-hierarchical-irt-diagnosis.md` for the full analysis.

### Prediction single-class crash (84th session)

The 84th biennium's holdout test set for the passage model contained only passed bills (no failures). `roc_auc_score` requires both classes present; `log_loss` needs explicit labels to avoid degenerate probability estimates.

### Synthesis Independent crash (89th session)

Dennis Pyle (Independent, 89th Senate) had null `unity_score` because party unity is undefined for Independents. The `detect_chamber_maverick()` function did not filter nulls, and `_minority_parties()` included "Independent" in its results, causing a format string crash.

### sys.path portability fix

All 15 analysis phase scripts add `sys.path.insert(0, ...)` to ensure the project root is on the Python path before the `try/except` import block. This provides a fallback when scripts are invoked outside the `uv run` environment.

## Decision

### 1. Joint model sign fix

Add `fix_joint_sign_convention()` to `hierarchical.py`:

- Compare joint model xi (per chamber) against per-chamber hierarchical xi via Pearson correlation
- If r < 0, negate all posterior xi samples for that chamber
- Return `(idata, flipped_chambers)` so downstream code knows which chambers were corrected
- Do NOT attempt to fix `mu_group` — it was estimated jointly with the flipped xi and cannot be recovered by arithmetic. Instead, `extract_hierarchical_ideal_points()` computes empirical group means from corrected xi for flipped chambers.

### 2. Prediction single-class fix

In `evaluate_holdout()`:
- Use `float("nan")` for AUC when only one class is present in the test set
- Pass `labels=[0, 1]` to `log_loss` to handle single-class holdout sets

### 3. Synthesis Independent fix

- `_minority_parties()` now excludes "Independent" from results (party-unity metrics are undefined for Independents)
- `detect_chamber_maverick()` filters out null `unity_score` values before computing percentiles

### 4. sys.path portability

All 15 main analysis scripts (`eda.py` through `external_validation.py`) add:
```python
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
```

### 5. Joint model diagnosis documented

Full diagnosis of the bill-matching bug written to `docs/joint-hierarchical-irt-diagnosis.md`. The joint model's results should be treated as unreliable for cross-chamber comparisons until the data combination is refactored to match bills by `bill_number`.

## Consequences

**Benefits:**
- Joint model sign convention is corrected post-hoc, producing correct within-chamber rankings (r = 0.999 vs per-chamber models)
- Pipeline runs cleanly on all 8 bienniums without crashes
- Root cause of cross-chamber distortion is documented with a clear fix path
- 4 new tests cover sign convention logic (35 → 39 hierarchical tests)

**Trade-offs:**
- The sign fix is a band-aid: it corrects direction but not the scale distortion caused by zero shared bill parameters. The joint model remains unreliable for cross-chamber placement until the bill-matching refactor.
- `party_mean` for flipped chambers is computed from empirical xi means rather than the mu_group posterior. This is less principled but necessarily so — the mu_group posterior is in the wrong sign convention.
- The AUC `nan` for single-class holdouts means some 84th-session prediction metrics are missing. This is correct behavior (AUC is undefined for a single class) but means the prediction report must handle nans gracefully.

**Next step:** ~~Refactor `build_joint_model()` to match bills by `bill_number` and create shared beta parameters for matched bills, as described in `docs/joint-hierarchical-irt-diagnosis.md`.~~ **Done in ADR-0043** — bill-matching and group-size-adaptive priors implemented.

**Further evolution:** ADR-0055 (2026-02-28) adds a reparameterized LogNormal beta prior (`exp(Normal(0, 1))`) to the joint model, eliminating the reflection mode multimodality that contributed to sign flips. With the `lognormal_reparam` prior and PCA initialization, `fix_joint_sign_convention()` is no longer triggered (r = 0.97 House, r = 0.89 Senate on the 84th). Stocking-Lord IRT linking was also added as a production alternative to the joint model for cross-chamber placement.
