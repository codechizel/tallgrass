# ADR-0032: Beta-Binomial Deep Dive Improvements

**Date:** 2026-02-25
**Status:** Accepted

## Context

A code audit and literature review of the Beta-Binomial party loyalty phase (Phase 9) identified five issues ranging from a variance estimator bug to missing diagnostics. See `docs/beta-binomial-deep-dive.md` for the full analysis.

## Decision

Implement all five recommendations from the deep dive:

1. **Fix `ddof=0` → `ddof=1`** in `estimate_beta_params`. The method of moments should use sample variance (dividing by N-1), not population variance (dividing by N). With Senate Democrats (N≈10), population variance underestimates by ~11%, producing an overly tight prior. Casella (1985) and Gelman BDA3 Ch. 5 both use sample variance.

2. **Add Tarone's overdispersion test.** Tests H0 (Binomial) vs H1 (Beta-Binomial overdispersion) per party-chamber group. Results written to the filtering manifest. Provides evidence for the model choice rather than assuming it.

3. **Add `votes_with_party` column** to output parquet. The integer y_i was previously reconstructable only via `n * raw_loyalty`, introducing floating-point noise. Now stored directly and used in `plot_posterior_distributions`.

4. **Add `prior_kappa` column** (`alpha + beta`) to output parquet and cross-chamber comparison table. This is the effective prior sample size — directly interpretable as "how many pseudo-observations does the prior contribute?"

5. **Clarify duplicated constants.** `MIN_PARTY_VOTES` and `CI_LEVEL` remain duplicated in `beta_binomial_report.py` with an explicit comment explaining the circular import that prevents sharing.

## Consequences

### Benefits

- **More accurate priors for small groups.** The ddof fix produces slightly wider priors for Senate Democrats, reducing over-shrinkage.
- **Evidence-based model choice.** Tarone's test now confirms overdispersion, strengthening ADR-0015's rationale.
- **Better data completeness.** `votes_with_party` and `prior_kappa` make the output self-documenting.
- **8 new tests (34 total).** Coverage for ddof, variable sample sizes, Tarone's test, and the previously-broken fallback test.

### Trade-offs

- Output schema change: downstream consumers reading `posterior_loyalty_*.parquet` will see two new columns. These are additive — no existing columns changed.
- Tarone's test adds ~5ms per run. Negligible.
