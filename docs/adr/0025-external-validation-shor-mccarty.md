# ADR-0025: External Validation Against Shor-McCarty Ideology Scores

**Date:** 2026-02-24
**Status:** Accepted

## Context

Every validation in the analysis pipeline is internal: IRT correlates with PCA, holdout prediction accuracy is high, cross-session ideal points are stable. The field standard for state legislator ideology research requires external validation against an independent measure. This was identified as the single biggest credibility gap in `docs/method-evaluation.md`.

The Shor-McCarty Individual State Legislator Ideology dataset (Harvard Dataverse, CC0 license) provides career-level ideal points for 610 Kansas legislators covering 1996-2020. This overlaps our 84th-88th bienniums (2011-2020), giving five independent correlation estimates.

## Decision

1. **Validate against Shor-McCarty CC0 data** via name matching. Compute Pearson r and Spearman ρ per biennium, chamber, and IRT model (flat and hierarchical).

2. **Cache the `.tab` file at `data/external/shor_mccarty.tab`** with auto-download from the Dataverse API and graceful failure with manual download instructions.

3. **Two-phase name matching:** exact normalized "first last" match (Phase 1), then last-name-only with deduplication (Phase 2). A `NICKNAME_MAP` dict handles edge cases iteratively.

4. **Report structure:** Matching summary, correlation summary table, scatter plots, intra-party correlations, outlier analysis, pooled analysis across all bienniums.

5. **Pure data logic separated from I/O:** `external_validation_data.py` contains all parsing, matching, and correlation functions (testable with synthetic data). `external_validation.py` handles download, file I/O, and orchestration.

## Consequences

**Benefits:**
- Closes the pipeline's largest credibility gap
- Provides the first independent check that our IRT scores measure what political scientists mean by "ideology"
- CC0 license eliminates data access concerns
- Five bienniums give five independent correlation estimates, not just one
- Pure logic / I/O separation enables comprehensive unit testing without network access

**Trade-offs:**
- SM scores are career-fixed; our scores vary by session. This limits the comparison to rank ordering, not scale equivalence.
- SM coverage ends at 2020. The 89th-91st bienniums (2021-2026) cannot be validated this way.
- Name matching is inherently fragile. The NICKNAME_MAP must be maintained as edge cases are discovered.
- 84th biennium has known data quality issues (~30% missing ODT votes), which may reduce correlation regardless of model quality.
