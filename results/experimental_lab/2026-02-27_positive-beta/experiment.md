# Experiment: Positive Beta Constraint for Hierarchical IRT Convergence

**Date:** 2026-02-27
**Status:** Complete
**Author:** Claude Code + Joseph Claeys

## The Short Version

We tested whether forcing bill discrimination parameters to be positive would fix convergence failures in our hierarchical IRT model. It helped — the LogNormal variant fixed the House's R-hat diagnostic (1.0103 → 1.0058) — but it traded one failure for another, dropping the effective sample size below threshold (564 → 362). The joint cross-chamber model still fails badly (R-hat 1.024, 25 divergences). Positive beta is a necessary ingredient but not sufficient on its own; more draws, better initialization, or a different sampler are likely needed alongside it.

## Why We Ran This Experiment

Our hierarchical Bayesian IRT model — which estimates where each Kansas legislator falls on a liberal-to-conservative spectrum — produces reliable results for the Senate but struggles with the House. Specifically, the statistical "convergence diagnostics" (quality checks that tell us whether the model's estimates can be trusted) pass for the Senate in 6 of 8 legislative sessions but only 1 of 8 for the House. The joint model that puts both chambers on a common scale fails for every session.

The root cause, identified in `docs/hierarchical-convergence-improvement.md`, is a mathematical symmetry in the model. Each bill has a "discrimination" parameter (beta) that measures how well it separates liberals from conservatives. When beta is allowed to be negative, the model can't tell the difference between "conservative legislator votes Yea on a conservative bill" and "liberal legislator votes Yea on a liberal bill" — both look the same mathematically. This creates a mirror-image solution that the sampler wastes time exploring.

The House has ~280 bills (each with its own beta), creating ~280 axes of ambiguity. The Senate has ~240, fewer legislators to track (40 vs 130), and a simpler geometry overall. This explains the asymmetry.

## What We Expected to Find

We believe that forcing all beta values to be positive — meaning every bill is required to discriminate in the "natural" direction (voting Yea = more conservative on a conservative bill) — will eliminate the mirror-image problem and allow the House model to converge.

We test two variants:
- **LogNormal(0, 0.5)**: Soft positive constraint. Prior median = 1.0, allowing betas near zero but not negative.
- **HalfNormal(1)**: Hard zero floor. Wider spread, less informative about the magnitude.

We expect both to substantially improve R-hat and ESS for the House. We also expect ideal point rankings to be highly correlated (r > 0.95) with the current model — the constraint shouldn't change *who* ranks where, just make the estimates more reliable.

## What We Tested

### Baseline (Current Production Model)

- **What it is:** Per-chamber hierarchical IRT with `beta ~ Normal(0, 1)` (symmetric, allows negative values). PCA-informed initialization, 4 chains, `adapt_diag`.
- **Command:** `uv run python results/experimental_lab/2026-02-27_positive-beta/run_experiment.py --variant baseline`
- **Output directory:** `run_01_baseline/`

### Run 2: LogNormal Beta

- **What changed:** `beta ~ LogNormal(0, 0.5)` instead of `Normal(0, 1)`. Prior median = 1.0, prior mean ≈ 1.13. All other settings identical.
- **Why:** LogNormal is the standard positive-constraint prior in educational IRT (De Ayala 2009). It places most mass near 1.0 (moderate discrimination) while allowing near-zero values for uninformative bills.
- **Command:** `uv run python results/experimental_lab/2026-02-27_positive-beta/run_experiment.py --variant lognormal`
- **Output directory:** `run_02_lognormal/`

### Run 3: HalfNormal Beta

- **What changed:** `beta ~ HalfNormal(1)` instead of `Normal(0, 1)`. Hard zero floor, wider spread than LogNormal.
- **Why:** Less informative than LogNormal about the expected magnitude. Good for checking sensitivity to the prior shape.
- **Command:** `uv run python results/experimental_lab/2026-02-27_positive-beta/run_experiment.py --variant halfnormal`
- **Output directory:** `run_03_halfnormal/`

### Run 4: Best Variant + Joint Model

- **What changed:** Whichever variant performs best in Runs 2-3, applied to the full pipeline including the joint cross-chamber model.
- **Why:** The joint model fails in all 8 sessions with the current parameterization. If positive beta fixes the per-chamber House, does it also fix the joint model?
- **Command:** `uv run python results/experimental_lab/2026-02-27_positive-beta/run_experiment.py --variant lognormal --include-joint`
- **Output directory:** `run_02_lognormal/` (joint results stored alongside per-chamber LogNormal outputs)

## How We Measured Success

| Metric | What It Tells Us | Passing Value | Source |
|--------|------------------|---------------|--------|
| R-hat (xi) | Whether chains agree on ideal point estimates | < 1.01 | Vehtari et al. 2021 |
| R-hat (mu_party) | Whether chains agree on party-level means | < 1.01 | Vehtari et al. 2021 |
| ESS (xi) | Effective number of independent samples for ideal points | > 400 (100/chain) | Vehtari et al. 2021 |
| ESS (mu_party) | Effective samples for party means | > 400 (100/chain) | Vehtari et al. 2021 |
| Divergences | Sampler geometry problems | < 5 | PyMC default |
| E-BFMI | Sampler energy sufficiency | > 0.3 | Betancourt 2016 |
| Pearson r (vs baseline) | Whether rankings changed | > 0.95 | Internal |
| Pearson r (vs flat IRT) | Agreement with independently validated model | > 0.95 | Shor-McCarty external validation |
| ICC | Fraction of ideological variance explained by party | Report only | — |
| Sampling time | Wall-clock cost | Report only | — |

## Results

### Summary Table

Values in **bold** pass the threshold. Thresholds: R-hat < 1.01, ESS > 400, divergences < 5.

| Metric | Baseline | LogNormal | HalfNormal | Joint (LN) |
|--------|----------|-----------|------------|------------|
| House R-hat(xi) max | 1.0103 | **1.0058** | 1.0122 | — |
| House ESS(xi) min | **564** | 362 | **450** | — |
| House R-hat(mu_party) max | **1.0070** | **1.0058** | **1.0071** | — |
| House ESS(mu_party) min | **512** | 397 | **529** | — |
| House divergences | **0** | **0** | **0** | — |
| House sampling time | 423s | 329s | 337s | — |
| House r vs baseline | — | 0.9948 | 0.9928 | — |
| House r vs flat IRT | 0.9868 | 0.9816 | 0.9778 | — |
| House ICC | 0.902 | 0.871 | 0.857 | — |
| Senate R-hat(xi) max | **1.0058** | **1.0084** | **1.0028** | — |
| Senate ESS(xi) min | **1002** | **555** | **1717** | — |
| Senate R-hat(mu_party) max | **1.0055** | **1.0075** | **1.0025** | — |
| Senate ESS(mu_party) min | **928** | **534** | **1700** | — |
| Senate divergences | **0** | **0** | **0** | — |
| Senate sampling time | 75s | 61s | 72s | — |
| Senate r vs baseline | — | 0.9199 | 0.9267 | — |
| Senate r vs flat IRT | 0.9763 | 0.8915 | 0.8988 | — |
| Senate ICC | 0.922 | 0.863 | 0.845 | — |
| Joint R-hat(xi) max | — | — | — | 1.0235 |
| Joint R-hat(mu_group) max | — | — | — | 1.0251 |
| Joint ESS(xi) min | — | — | — | 243 |
| Joint ESS(mu_group) min | — | — | — | 269 |
| Joint divergences | — | — | — | 25 |
| Joint sampling time | — | — | — | 3,358s |
| Total time | 505s | 396s | 415s | 3,762s |
| **House verdict** | **FAIL** (R-hat) | **FAIL** (ESS) | **FAIL** (R-hat) | — |
| **Senate verdict** | **PASS** | **PASS** | **PASS** | — |
| **Joint verdict** | — | — | — | **FAIL** (R-hat, ESS, div) |

### What We Observed

#### Per-Chamber Results (Runs 1-3)

**The positive beta constraint improved House R-hat but not enough to pass.** LogNormal dropped the worst-case House R-hat from 1.0103 to 1.0058 — well past the 1.01 threshold, which is the headline result. However, it simultaneously dropped the minimum ESS from 564 to 362, failing a different diagnostic. HalfNormal went the other direction: worse R-hat (1.0122) but adequate ESS (450).

Neither variant achieves a clean pass for the House.

**The Senate was already fine and stayed fine.** Both variants pass all convergence checks in the Senate. HalfNormal produced especially good Senate results: R-hat 1.0028, ESS 1,717 — the best Senate convergence we've ever measured. LogNormal was slightly weaker in the Senate (ESS 555, down from 1,002 at baseline).

**Zero divergences across all runs.** The sampler geometry is clean — the convergence issues are about chain mixing speed, not pathological curvature.

**Rankings are highly correlated with baseline but not with flat IRT.** Both variants produce House ideal points correlated r > 0.99 with the baseline, confirming the positive beta constraint doesn't change *who* ranks *where*. Senate correlations with baseline are lower (r ≈ 0.92), which is expected: the Senate has fewer legislators, so each one's estimate shifts more when the prior changes. Both variants' Senate flat-IRT correlations dropped from 0.976 to ~0.89-0.90, suggesting the positive constraint pushes the hierarchical model further from the unconstrained flat IRT.

**ICC (party-explained variance) decreased slightly.** From 0.902 to 0.857-0.871 for the House, and 0.922 to 0.845-0.863 for the Senate. This makes sense: constraining beta to be positive restricts how much the model can shrink ideal points toward party means.

#### Joint Model (Run 4)

**The joint model still fails, but less catastrophically.** With LogNormal beta, the joint model produced R-hat(xi) = 1.024 and R-hat(mu_group) = 1.025 — well above the 1.01 threshold but dramatically better than production's typical R-hat of 1.5-2.4 and ESS of 5-7. ESS improved to 243 (from ~7 at production), and the model produced 25 divergences (the first time we've seen non-zero divergences in this experiment, and well above the < 5 threshold).

The joint model took 56 minutes to sample — about 10x longer than per-chamber House. The step size was extremely small (0.007-0.009, vs 0.06 for per-chamber House), indicating difficult geometry. The chains moved slowly: ~1 draw/second vs ~9 draws/second for per-chamber House.

**No sign correction was needed.** Both chambers' joint ideal points correlated positively with the per-chamber results (House r = 1.000, Senate r = 0.998), confirming the positive beta eliminated the sign-flip problem at the joint level too.

### Impact on Rankings and Scores

**Per-chamber House rankings are essentially unchanged** (r = 0.995 LogNormal vs baseline, r = 0.993 HalfNormal). This confirms the positive constraint doesn't alter the substantive conclusions — the same legislators end up in the same relative positions.

**Senate rankings shifted moderately** (r ≈ 0.92 vs baseline). The hierarchical Senate was already affected by over-shrinkage with 10 Democrats (documented in `docs/hierarchical-shrinkage-deep-dive.md`), and the positive beta constraint appears to interact with this issue. The Senate's flat-IRT correlation dropped from 0.976 to 0.89-0.90, suggesting more shrinkage.

**ICC decreased across the board.** The fraction of ideological variance explained by party dropped from 0.90 to 0.86 (House) and 0.92 to 0.85 (Senate). This is not necessarily a problem — it means the model is allowing more within-party variation rather than pulling everyone toward their party mean. Whether this is "better" depends on whether the baseline was over-shrinking.

## What We Learned

**The hypothesis was partially confirmed.** Positive beta helps — LogNormal clearly fixes the R-hat diagnostic for per-chamber House — but it's not a silver bullet. The improvement came at the cost of ESS, and the joint model remains intractable.

**The convergence problem has at least two components:**

1. **Reflection mode (fixed by positive beta).** LogNormal eliminated the β sign-flip ambiguity and dropped House R-hat from 1.0103 to 1.0058 — a clean pass. This confirms the theory in `docs/hierarchical-convergence-improvement.md`.

2. **Slow mixing in high dimensions (not fixed).** Even with the reflection mode gone, ESS dropped below 400 for the House. The chains agree on the answer (good R-hat) but haven't produced enough independent samples yet. For the joint model (172 legislators, 420 votes), this problem dominates: the step size shrinks to 0.007 and drawing speed drops to 1/second.

**LogNormal beats HalfNormal for per-chamber House.** LogNormal fixed R-hat while HalfNormal didn't (1.0058 vs 1.0122). For the Senate, HalfNormal was actually better (ESS 1,717 vs 555). This makes sense: the Senate's smaller parameter space benefits from HalfNormal's wider variance, while the House needs the tighter LogNormal to break the reflection mode efficiently.

**Next steps for convergence improvement** (from the priority list in `docs/hierarchical-convergence-improvement.md`):

- **More draws** (Priority 2): Increase from 2,000 to 4,000+ draws with LogNormal beta. This directly addresses the ESS shortfall — if R-hat is fine, more draws should push ESS past 400.
- **nutpie sampler** (Priority 3): Drop-in replacement for PyMC's default sampler. Uses Rust-compiled gradient evaluation, reported 2-5x speedup. Would address both per-chamber ESS and joint model tractability.
- **Reduce bill set** (Priority 6): Filter low-discrimination bills before modeling. Fewer beta parameters = faster sampling and better ESS.

## Changes Made

No production code changes. The positive beta constraint is promising but needs to be combined with more draws or a faster sampler before it's ready to deploy. The experiment artifacts (run outputs, HTML reports, metrics) are preserved in this directory for reference.

---

**Default session:** 91st biennium (2025-26). The primary analyst (Sen. Joseph Claeys) has content knowledge of the legislators in this session and can verify that rankings and groupings look correct. Each run produces a full HTML report for visual inspection.

**Related documents:**
- Theory and improvement plan: `docs/hierarchical-convergence-improvement.md`
- ADR: `docs/adr/0047-positive-beta-constraint-experiment.md`
- Previous experiment (4-chain): `results/experimental_lab/2026-02-26_hierarchical-4-chains/`
- Over-shrinkage analysis: `docs/hierarchical-shrinkage-deep-dive.md`
