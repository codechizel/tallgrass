# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the KS Vote Scraper project.

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

## Creating a New ADR

1. Copy the template above
2. Use the next sequential number (`NNNN`)
3. Add an entry to the index table
