# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the Tallgrass project.

ADRs document significant technical decisions so future contributors understand *why* things are the way they are, not just *what* they are.

## Format

Each ADR follows a lightweight MADR-style template:

```markdown
# ADR-NNNN: Title

**Date:** YYYY-MM-DD
**Status:** Accepted | Superseded | Deprecated

## Context
What prompted this decision?

## Decision
What did we decide?

## Consequences
What are the trade-offs?
```

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-results-directory-structure.md) | Results directory structure | Accepted | 2026-02-19 |
| [0002](0002-polars-over-pandas.md) | Polars over pandas | Accepted | 2026-02-19 |
| [0003](0003-two-phase-fetch-parse.md) | Two-phase fetch/parse pattern | Accepted | 2026-02-19 |
| [0004](0004-html-report-system.md) | HTML report system | Accepted | 2026-02-19 |
| [0005](0005-pca-implementation-choices.md) | PCA implementation choices | Accepted | 2026-02-19 |
| [0006](0006-irt-implementation-choices.md) | IRT implementation choices | Accepted | 2026-02-19 |
| [0007](0007-clustering-implementation-choices.md) | Clustering implementation choices | Accepted | 2026-02-20 |
| [0008](0008-data-driven-synthesis-detection.md) | Data-driven synthesis detection | Accepted | 2026-02-21 |
| [0009](0009-retry-waves-for-transient-failures.md) | Retry waves for transient failures | Accepted | 2026-02-21 |
| [0010](0010-data-driven-network-edge-analysis.md) | Data-driven network edge weight analysis | Accepted | 2026-02-21 |
| [0011](0011-umap-implementation-choices.md) | UMAP implementation choices | Accepted | 2026-02-22 |
| [0012](0012-nlp-bill-text-features-for-prediction.md) | NLP bill text features for prediction | Accepted | 2026-02-22 |
| [0013](0013-legislator-profile-deep-dives.md) | Legislator profile deep-dives | Accepted | 2026-02-22 |
| [0014](0014-clustering-visualization-overhaul.md) | Clustering visualization overhaul | Accepted | 2026-02-22 |
| [0015](0015-empirical-bayes-beta-binomial.md) | Empirical Bayes for Beta-Binomial party loyalty | Accepted | 2026-02-22 |
| [0016](0016-state-level-directory-structure.md) | State-level directory structure | Accepted | 2026-02-22 |
| [0017](0017-hierarchical-bayesian-irt.md) | Hierarchical Bayesian IRT | Accepted | 2026-02-22 |
| [0018](0018-ty-type-checker-adoption.md) | ty type checker adoption (beta) | Accepted | 2026-02-22 |
| [0019](0019-cross-session-validation.md) | Cross-session validation | Accepted | 2026-02-22 |
| [0020](0020-historical-session-support.md) | Historical session support (2011-2026) | Accepted | 2026-02-22 |
| [0021](0021-independent-party-handling.md) | Independent party handling across analysis pipeline | Accepted | 2026-02-22 |
| [0022](0022-analysis-parallelism-and-timing.md) | Analysis parallelism and runtime timing | Accepted | 2026-02-23 |
| [0023](0023-pca-informed-irt-initialization.md) | PCA-informed IRT chain initialization (default) | Accepted | 2026-02-23 |
| [0024](0024-instruction-file-restructure.md) | Instruction file restructure | Accepted | 2026-02-24 |
| [0025](0025-external-validation-shor-mccarty.md) | External validation against Shor-McCarty scores | Accepted | 2026-02-24 |
| [0026](0026-eda-literature-diagnostics.md) | EDA literature-backed diagnostics | Accepted | 2026-02-24 |
| [0027](0027-umap-validation-and-robustness.md) | UMAP validation and robustness improvements | Accepted | 2026-02-24 |
| [0028](0028-clustering-deep-dive-improvements.md) | Clustering deep dive improvements | Accepted | 2026-02-24 |
| [0029](0029-leiden-community-detection.md) | Leiden community detection (replacing Louvain) | Accepted | 2026-02-24 |
| [0030](adr-0030-analysis-directory-restructuring.md) | Analysis directory restructuring (numbered subdirs + PEP 302) | Accepted | 2026-02-25 |
| [0031](0031-prediction-deep-dive-improvements.md) | Prediction deep dive improvements (holdout eval, Brier/log-loss, IRT caveat) | Accepted | 2026-02-25 |
| [0032](0032-beta-binomial-deep-dive-improvements.md) | Beta-Binomial deep dive improvements (ddof fix, Tarone's test, output columns) | Accepted | 2026-02-25 |
| [0033](0033-hierarchical-irt-deep-dive-improvements.md) | Hierarchical IRT deep dive improvements (small-group warning, ICC rename, 9 tests) | Accepted | 2026-02-25 |
| [0034](0034-synthesis-deep-dive-improvements.md) | Synthesis deep dive improvements (data extraction, minority mavericks, dynamic AUC, 47 tests) | Accepted | 2026-02-25 |
| [0035](0035-cross-session-deep-dive-improvements.md) | Cross-session deep dive improvements (3 bug fixes, affine transform for turnover, 18 new tests) | Accepted | 2026-02-25 |
| [0036](0036-scraper-hardening.md) | Scraper hardening (assert removal, config constants, title truncation warning, 44 HTTP tests) | Accepted | 2026-02-25 |
| [0037](0037-pipeline-review-fixes.md) | Pipeline review fixes (except syntax, RunContext failure safety, PC1 sign, label alignment) | Accepted | 2026-02-25 |
| [0038](0038-python-314-modernization.md) | Python 3.14 modernization (__future__ removal, CalVer, typing cleanup) | Accepted | 2026-02-25 |
| [0039](0039-cross-session-validation-enhancements.md) | Cross-session validation enhancements (PSI, ICC, fuzzy matching, percentile thresholds) | Accepted | 2026-02-25 |
| [0040](0040-rename-to-tallgrass.md) | Rename package to Tallgrass | Accepted | 2026-02-25 |
| [0041](0041-mca-implementation-choices.md) | MCA implementation choices (prince, Greenacre, categorical encoding) | Accepted | 2026-02-25 |
| [0042](0042-joint-model-sign-fix-and-pipeline-hardening.md) | Joint model sign fix, pipeline hardening, and joint model diagnosis | Accepted | 2026-02-26 |
| [0043](0043-hierarchical-irt-bill-matching-and-adaptive-priors.md) | Hierarchical IRT bill-matching and group-size-adaptive priors | Accepted | 2026-02-26 |
| [0044](0044-hierarchical-pca-informed-init.md) | PCA-informed initialization for hierarchical IRT | Accepted | 2026-02-26 |
| [0045](0045-4-chain-hierarchical-irt.md) | 4-chain hierarchical IRT with adapt_diag initialization | Accepted | 2026-02-26 |
| [0046](0046-2d-irt-experimental.md) | 2D IRT as experimental extension (PLT identification, Tyson paradox) | Accepted | 2026-02-26 |
| [0047](0047-positive-beta-constraint-experiment.md) | Positive beta constraint experiment for hierarchical IRT convergence | Accepted | 2026-02-27 |
| [0048](0048-experiment-framework.md) | Experiment framework (BetaPriorSpec, PlatformCheck, experiment runner) | Accepted | 2026-02-27 |
| [0049](0049-nutpie-flat-irt-baseline.md) | nutpie flat IRT baseline experiment (compilation, sampling, sign flip) | Accepted | 2026-02-27 |

## Creating a New ADR

1. Copy the template above
2. Use the next sequential number (`NNNN`)
3. Add an entry to the index table
