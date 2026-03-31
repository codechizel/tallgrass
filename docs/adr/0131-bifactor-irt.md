# ADR-0131: Bifactor IRT for Roll-Call Analysis

**Date:** 2026-03-30
**Status:** Experimental

## Context

The pipeline's 2D M2PL model (Phase 06) separates ideology from the establishment-contrarian axis using Positive Lower Triangular (PLT) rotation constraints. While this resolves the Tyson paradox, the dimension separation is statistical (rotation convention) rather than structural. CFA literature shows that a bifactor model — one general factor loading on all items, plus specific factors loading on item subsets — provides a principled structural separation where the general factor captures "pure" shared variance across all items.

Investigation (see `docs/cfa-irt-dimensionality-deep-dive.md`) confirmed that standard 2-factor CFA is mathematically equivalent to 2D M2PL IRT for binary data and would require impractical bill-level a priori classification. The bifactor variant avoids this: the general factor loads on all bills with no classification needed, and only the specific factors require bill groupings.

## Decision

Add Phase 06b (bifactor IRT) to the single-biennium pipeline:

1. **General factor (theta_G):** Loads on ALL contested bills. Captures what is common to all voting behavior — the broadest ideological dimension. Identified by post-hoc sign flip (Republican mean > 0).

2. **Specific factors:** Load on bill subsets classified by Phase 05 IRT discrimination:
   - **theta_S1 (partisan):** Bills with |beta| > 1.5 (HIGH_DISC_THRESHOLD)
   - **theta_S2 (bipartisan):** Bills with |beta| < 0.5 (LOW_DISC_THRESHOLD)
   - Medium-discrimination bills load on the general factor only.

3. **Identification:** Orthogonal by construction (independent Normal(0,1) priors on all three theta factors). No PLT constraints needed — simpler than Phase 06.

4. **Key diagnostics:**
   - ECV (Explained Common Variance): `sum(a_G^2) / [sum(a_G^2) + sum(a_S1^2) + sum(a_S2^2)]`
   - omega_h (Omega Hierarchical): reliability of the general factor

5. **Bill classification is not circular:** Uses 1D beta *magnitude* from Phase 05 (a different model), not bifactor parameters. Medium-disc bills anchor the general factor as "pure ideology" items.

6. **Fallback:** When Phase 05 hasn't run, falls back to EDA party-line classification (`vote_alignment.parquet`).

## Consequences

- **New phase:** `analysis/06b_bifactor/` with `bifactor.py` and `bifactor_report.py`. Runs after Phase 06 in the pipeline sequence.
- **Parameters:** ~3x the parameter count of 1D IRT (3 theta per legislator + 4 parameters per bill). Expected runtime 2-4x Phase 06.
- **Convergence:** Uses same relaxed thresholds as Phase 06 (R-hat < 1.05, ESS > 200).
- **If ECV > 0.70:** The 1D model is already adequate. Bifactor adds complexity without meaningful gain. This is a valid empirical finding, not a failure.
- **If ECV < 0.60:** The specific factors carry substantial variance. The bifactor general factor may be a better canonical ideology score than 2D Dim 1.
- **Canonical routing integration (deferred):** Phase 06b does not yet feed into `canonical_ideal_points.py`. This will be revisited based on empirical results.
- **PPC support (deferred):** Bifactor log-likelihood not yet in Phase 08 PPC.

## Empirical Results (14 Bienniums, 2026-03-30)

Phase 06b was run on all 14 Kansas bienniums (78th-91st). The 82nd failed due to missing EDA data for its run-id.

### House — Promising

11/13 chambers converged on the general factor. ECV ranged from 0.42 (85th, strong multidimensionality) to 0.98 (86th, essentially unidimensional).

| Pattern | Sessions | ECV Range | Assessment |
|---------|----------|-----------|------------|
| 1D adequate (ECV > 0.70) | 78th, 79th, 86th-89th | 0.78-0.98 | Bifactor adds no value; 1D IRT sufficient |
| Meaningful structure (ECV 0.60-0.70) | 80th, 81st, 83rd, 90th, 91st | 0.60-0.73 | Specific factors carry real variance; theta_G may improve over 2D Dim 1 |
| Strong multidimensionality (ECV < 0.60) | 84th, 85th | 0.42-0.62 | Transitional supermajority era; convergence weak (low omega_h) |

The 90th and 91st achieved **full convergence** (no warnings) with ECV ~0.62 — the cleanest bifactor results. theta_G correlates r > 0.96 with 1D IRT in all converged House chambers.

### Senate — Struggles

Only 6/13 converged. Root cause: small Democrat caucus (8-12 members) provides insufficient signal for specific factors.

| Issue | Sessions | Details |
|-------|----------|---------|
| Collapsed (ECV < 0.15) | 88th, 90th | General factor captured near-zero variance; model poorly identified |
| Sign flip (r < -0.50 with 1D) | 79th, 86th, 91st | Post-hoc sign check failed to correct; horseshoe-era contamination |
| Converged well | 81st, 83rd, 85th, 87th | Senate works when party balance is more moderate |

### Implications for Deferred Items

- **BF1 (canonical routing):** House-only integration is viable for ECV < 0.70 sessions. Senate not recommended at current convergence rates.
- **BF2 (PPC LOO-CV):** Worthwhile for the 90th/91st House (full convergence, meaningful ECV).
- **BF3 (threshold sensitivity):** Lower priority — the discrimination-based classification worked across all runnable sessions.
- **BF4 (hierarchical bifactor):** Risk is high — Senate already struggles with flat bifactor; adding hierarchy would compound convergence problems.
