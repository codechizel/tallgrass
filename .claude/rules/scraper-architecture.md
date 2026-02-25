---
paths:
  - "src/**/*.py"
---

# Scraper Architecture

## Session Coverage (2011-2026)

| Biennium | Bill Discovery | Vote Format | Party Detection | Member Dir |
|----------|---------------|-------------|-----------------|------------|
| 2025-26 (91st) | HTML links | vote_view | `<h2>District N - Party` | HTML links |
| 2023-24 (90th) | HTML links | vote_view | `<h2>` | HTML links |
| 2021-22 (89th) | HTML links | vote_view | `<h2>` | HTML links |
| 2019-20 (88th) | JS (quoted keys) | vote_view | `<h2>` | JS fallback |
| 2017-18 (87th) | JS (unquoted keys) | vote_view | `<h2>` | JS fallback |
| 2015-16 (86th) | JS (unquoted keys) | vote_view | `<h3>Party:` | JS fallback |
| 2013-14 (85th) | JS (unquoted keys) | odt_view | `<h3>Party:` | JS fallback |
| 2011-12 (84th) | JS (unquoted keys) | odt_view* | `<h3>Party:` | JS fallback |

\* 84th session ODTs: ~30% contain only tally metadata (no individual legislator names).

## Concurrency Pattern

Two-phase: concurrent fetch via ThreadPoolExecutor (MAX_WORKERS=5), then sequential parse. Rate limiting is thread-safe via `threading.Lock()`. Never mutate shared state during fetch.

## Key Data Structures

- `FetchResult` — typed HTTP result; `content_bytes` for binary downloads (ODT); cached as `.bin`
- `FetchFailure` — recorded with bill context (number, motion, path); written to `failure_manifest.json`
- `VoteLink` — frozen dataclass for vote page links; `is_odt=True` routes to ODT parser
- Module constants: `_BILL_URL_RE` (compiled regex), `VOTE_CATEGORIES` (5-tuple), `_normalize_bill_code()`

## Retry Strategy

`_get()` returns `FetchResult` and classifies errors:

- **404** -> `"permanent"`, max 2 attempts
- **5xx** -> `"transient"`, exponential backoff with jitter
- **Timeout** -> `"timeout"`, exponential backoff with jitter
- **Connection error** -> `"connection"`, fixed delay
- **Other 4xx** -> `"permanent"`, no retry
- **HTTP 200 error page** -> `"permanent"`, detected by HTML heuristics

All HTTP requests go through `_get()` (including KLISS API) for consistent retries, rate limiting, caching, and error-page detection.

## Retry Waves

When sustained server failures cause thundering herd at the `_fetch_many()` level:

1. Collect transient failures (5xx, timeout, connection)
2. Wait `WAVE_COOLDOWN` (90s)
3. Re-dispatch with reduced load: `WAVE_WORKERS=2`, `WAVE_DELAY=0.5s`
4. Repeat up to `RETRY_WAVES=3` times

Jitter on per-URL backoff prevents thundering herd within each wave. See ADR-0009.

## ODT Parser (2011-2014)

`odt_parser.py` — pure functions, no I/O, stdlib only (zipfile + xml.etree):
- `parse_odt_votes()`: bytes -> (RollCall, IndividualVote, legislators)
- Extracts `content.xml` from ZIP, parses `<text:user-field-decl>` metadata
- Maps House/Senate vote category variants to canonical names
- Last-name-only resolution via member directory (handles initials, ambiguity)

## Vote Deduplication

ODT sessions (2011-2014) can link the same vote page from multiple bills. `save_csvs()` deduplicates by `(legislator_slug, vote_id)`, keeping first occurrence.
