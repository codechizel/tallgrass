# ADR-0015: Empirical Bayes for Beta-Binomial Party Loyalty

**Date:** 2026-02-22
**Status:** Accepted

## Context

The indices phase (Phase 7) computes CQ-standard party unity as a raw fraction: `votes_with_party / party_votes_present`. This is noisy for legislators with few party votes (e.g., Miller with 30 votes, or mid-session replacements). The Beta-Binomial model applies Bayesian shrinkage to produce more reliable estimates.

The method doc (`Analytic_Methods/14_BAY_beta_binomial_party_loyalty.md`) describes two approaches:

1. **Empirical Bayes** — Estimate Beta(alpha, beta) hyperparameters from data via method of moments, then compute closed-form posteriors.
2. **Full hierarchical Bayes** — Use PyMC to fit a hierarchical Beta-Binomial model with priors on the hyperparameters, producing full posterior distributions via MCMC.

## Decision

Use **empirical Bayes (method of moments)** for the Beta-Binomial phase. Reserve full hierarchical MCMC for the roadmap's hierarchical Bayesian legislator model (item #3).

## Consequences

### Benefits

- **Instant execution** (~1 second) — no MCMC convergence to worry about.
- **No PyMC dependency** for this phase — only scipy.stats for Beta PDF/PPF.
- **Closed-form posteriors are exact** — no sampling error, no chain diagnostics, no divergences.
- **Simple implementation** — ~400 lines of analysis code, easy to test with synthetic data.
- **Perfectly adequate for the use case** — with ~170 legislators and 4 party-chamber groups, the difference between empirical Bayes and full Bayes point estimates is negligible.

### Trade-offs

- **Underestimates hyperparameter uncertainty** — the credible intervals are slightly too narrow because they treat the estimated prior as known. This matters for formal inference but not for the exploratory purpose here.
- **Does not propagate prior uncertainty** — if the true party loyalty distribution is poorly estimated (e.g., Senate Democrats with n≈10), the posteriors may be overconfident.
- **Cannot do model comparison** — empirical Bayes doesn't produce a marginal likelihood, so we can't formally compare this model to alternatives using Bayes factors.

### Future path

The hierarchical Bayesian model (roadmap item #3) will use PyMC for full posterior inference, including hyperparameter uncertainty, model comparison, and posterior predictive checks. That phase will supersede this one for formal analysis; this phase remains as the fast, exploratory baseline.
