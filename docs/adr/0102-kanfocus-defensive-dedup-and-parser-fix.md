# ADR-0102: KanFocus Defensive Deduplication and Parser Fix

**Date:** 2026-03-06
**Status:** Accepted

## Context

The 83rd biennium KanFocus CSV had 4 duplicate rollcalls, 279 duplicate votes, and 1 malformed truncated line. Deep investigation across all 8 archived bienniums (78th-91st) confirmed:

1. **Not a reproducible code bug.** A clean re-run from the 83rd biennium cache produces 1,186 rollcalls with 0 duplicate `vote_id`s and 66,815 votes with 0 duplicate `(slug, vote_id)` pairs. The original duplicates were an operational issue — likely a write interruption during the original scrape.

2. **Parser regex bleed.** The `_parse_metadata()` result regex `Result:\s*(\S+(?:\s+\S+)*?)(?:\s+All\s+Members|\s*\n)` matched "All Members" (a table header) when the Result field was empty. Affected 8 records across 5 bienniums (79th, 80th, 81st, 90th, 91st).

3. **Content-duplicate confirmation votes.** 33-103 per biennium are legitimate separate votes on gubernatorial appointments that look identical (same `bill_number=""`, same `motion="On confirmation"`, same tally) but have distinct `vote_num`s — NOT bugs.

4. **Tally mismatches.** Only 2/12,570 records across all bienniums had count mismatches (voice votes where individual names aren't listed).

## Decision

Add defensive layers rather than change the core pipeline:

### 1. Rollcall deduplication in `save_csvs()` (`output.py`)
The existing vote deduplication by `(legislator_slug, vote_id)` had no rollcall counterpart. Added `vote_id`-based rollcall dedup as a safety net, matching the existing vote dedup pattern. Logs count when duplicates are removed.

### 2. Early dedup in `convert_to_standard()` (`kanfocus/output.py`)
Added `seen_vote_ids` set to skip records with duplicate `vote_num/year/chamber` coordinates before conversion. Also added tally mismatch warning when parsed legislator count doesn't match the sum of `yea + nay + present + not_voting`.

### 3. Parser regex guard (`kanfocus/parser.py`)
Post-match guard: if `result == "All Members"`, set to empty string. Simpler than rewriting the regex and handles the edge case cleanly.

### 4. Cache corruption recovery (`kanfocus/fetcher.py`)
`fetch_page()` now catches `OSError`/`UnicodeDecodeError` when reading cache files, deletes the corrupted file, and re-fetches from network.

### 5. Safe integer parsing (`kanfocus/output.py`, `kanfocus/crossval.py`)
`_safe_int()` helper replaces 12 bare `int()` calls that could raise on malformed CSV fields. Handles empty strings, `None`, whitespace, and non-numeric values by returning 0.

## Consequences

- **Idempotent safety net.** Even if upstream produces duplicates (write interruption, re-run glitch), the CSV output is clean. Both `save_csvs()` and `convert_to_standard()` deduplicate independently.
- **No behavior change for clean data.** All dedup layers are no-ops when input has no duplicates.
- **16 new tests** across 4 test files covering all defensive improvements.
- **No performance impact.** Dedup uses set lookups (O(1) per record).
