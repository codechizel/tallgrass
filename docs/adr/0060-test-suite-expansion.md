# ADR-0060: Test Suite Expansion (Markers, Integration, Report Structure)

**Date:** 2026-02-28
**Status:** Accepted

## Context

The test suite had 1421 tests (all passing) but three gaps identified in the roadmap backlog:

1. **No end-to-end pipeline integration tests.** Phases were tested in isolation — no test verified that EDA output feeds correctly into PCA, or that RunContext produces the expected directory structure with real phase data.

2. **No HTML report structural tests.** The existing `test_report.py` verified rendering logic (section types, builder API) but not output stability — no tests checked that TOC anchors matched section IDs, that numbering was sequential, or that the CSS was embedded. Full-HTML snapshots (via syrupy or similar) would be brittle due to great_tables inline CSS changes across library versions.

3. **No pytest markers for selective test runs.** Running only scraper tests required listing files manually in the Justfile. No way to skip slow data-integrity tests or select integration tests by category.

## Decision

Three changes, no new dependencies:

### 1. Pytest Markers

Registered three markers in `pyproject.toml` (`scraper`, `integration`, `slow`). Applied via module-level `pytestmark` variables (not per-class decorators) to minimize diff noise:

- `@pytest.mark.scraper` on 8 scraper test files (~264 tests)
- `@pytest.mark.integration` + `@pytest.mark.slow` on `test_data_integrity.py` (~24 tests)
- `@pytest.mark.integration` on `TestRunContextLifecycle` in the new integration test file

Updated Justfile: `test-scraper` now uses `-m scraper` instead of listing files; added `test-fast` recipe (`-m "not slow"`).

### 2. Integration Pipeline Tests (`test_integration_pipeline.py`, 26 tests)

End-to-end test with synthetic data: 20 legislators (12R House, 4D House, 3R Senate, 1D Senate) × 30 roll calls. Programmatic: R vote Yea 80%, D vote Nay 80%, mix of unanimous and contested votes.

Tests chain EDA → PCA: `build_vote_matrix()` → `filter_vote_matrix()` → `compute_agreement_matrices()` / `compute_rice_cohesion()` → `impute_vote_matrix()` → `fit_pca()` → `orient_pc1()`. Verifies the contract between phases (shapes, schemas, value ranges, PC1 orientation).

Also tests `resolve_upstream_dir()` round-trips and full `RunContext` lifecycle (directory structure, primer, run_info.json, latest symlink, failure safety).

### 3. Report Structure Tests (`test_report_structure.py`, 22 tests)

Structural assertions on the rendered HTML skeleton without snapshotting full HTML:

- TOC `<li>` count matches section count; TOC `href` anchors match `<section id>`
- Section numbering is sequential (1, 2, 3...)
- All three container types present (`table-container`, `figure-container`, `text-container`)
- Title in `<h1>` and `<title>`; session in header meta; footer contains title
- Timestamp present; duplicate IDs don't crash; empty report renders valid shell
- `REPORT_CSS` embedded in `<style>` tag; `make_gt()` output integrates correctly

## Consequences

- Test count increases from 1421 to 1469 (48 new tests). All passing.
- Selective test runs: `just test-scraper` (~264 tests, 2s), `just test-fast` (~1445 tests, skips data integrity I/O).
- No `PytestUnknownMarkWarning` — all markers registered in `pyproject.toml`.
- No new dependencies — structural HTML assertions use `re` patterns, not snapshot libraries.
- The synthetic data fixtures in `test_integration_pipeline.py` can be extended to test additional phase chains (e.g., EDA → PCA → IRT) as needed.
