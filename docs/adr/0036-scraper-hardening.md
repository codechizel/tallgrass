# ADR-0036: Scraper Hardening

**Date:** 2026-02-25
**Status:** Accepted

## Context

A deep dive audit of the scraper codebase (see `docs/scraper-deep-dive.md`) identified several minor hardening opportunities and a significant test coverage gap. The scraper's HTTP fetch layer — including retry logic, error classification, cache behavior, and the KLISS API pre-filter — had zero test coverage despite being the most critical reliability infrastructure.

The audit also compared the project against OpenStates (the gold standard for legislative scraping) and found our implementation significantly more robust, but identified concrete improvements.

## Decision

Five changes:

1. **Replace `assert` with defensive check** in `_filter_bills_with_votes()`. The `assert result.html is not None` was a development guard that would raise `AssertionError` in production. Replaced with an explicit check that falls back to the full bill list.

2. **Extract magic numbers to config.py**. Two hardcoded values (`200` for cache filename truncation, `500` for bill title truncation) are now named constants: `CACHE_FILENAME_MAX_LENGTH` and `BILL_TITLE_MAX_LENGTH`.

3. **Add warning on bill title truncation**. Both `scraper.py` and `odt_parser.py` now log a warning when a bill title is truncated, preventing silent data loss.

4. **Add safety comment on `self.delay` mutation**. The `_fetch_many()` retry wave code mutates `self.delay` during execution. A comment now documents why this is safe (wave runs after all Phase 1 futures complete, so no concurrent readers).

5. **Add HTTP layer tests** (`tests/test_scraper_http.py`, 44 tests). Covers `_get()` success paths, error classification (404/5xx/timeout/connection), error page detection (Bug #9 defense), retry counts, cache read/write/clear, `_fetch_many()` retry waves, `_rate_limit()` thread safety, and `_filter_bills_with_votes()` JSON format handling.

## Consequences

- Test count increases from 1052 to 1096.
- The retry strategy (the scraper's most critical reliability feature) now has direct test coverage for all error paths.
- Config constants improve discoverability of truncation limits.
- No behavioral changes to scraper output — all changes are defensive improvements.
