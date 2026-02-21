# ADR-0009: Retry waves for transient failures

**Date:** 2026-02-21
**Status:** Accepted

## Context

A fresh scrape of 2025-26 failed on 55 of 882 vote pages — all transient 5xx errors clustered in a 3-second window. The KS Legislature website is old and fragile; sustained concurrent traffic causes it to buckle. The per-URL retry logic (3 attempts, exponential backoff) handles isolated hiccups but cannot handle a server that needs minutes to recover. All 5 workers fail simultaneously, retry simultaneously (thundering herd), exhaust their 3 attempts within ~35 seconds, and give up forever.

Alternatives considered:

- **Global circuit breaker**: Adds complexity (failure-rate tracking, state machine). The wave pattern achieves the same effect more simply.
- **Adaptive worker scaling**: Dynamic concurrency adds tuning knobs. A fixed step-down to 2 workers is sufficient.
- **Increasing `MAX_RETRIES`**: More per-URL retries don't help when the server needs minutes, not seconds, to recover. They just extend the thundering herd.

## Decision

Add **retry waves** to `_fetch_many()`. After the initial concurrent pass, if transient failures remain (error_type in `transient`, `timeout`, `connection`):

1. Wait `WAVE_COOLDOWN` (90s) — let the server recover
2. Re-dispatch only the failed URLs with reduced load:
   - `WAVE_WORKERS=2` (down from `MAX_WORKERS=5`)
   - `WAVE_DELAY=0.5s` (down from `REQUEST_DELAY=0.15s`)
3. Merge results: successes overwrite old failures
4. Repeat up to `RETRY_WAVES=3` times (or until no transient failures remain)

Additionally, add **jitter** to per-URL backoff in `_get()` for 5xx and timeout errors: multiply the delay by `(1 + random.uniform(0, 0.5))`. This spreads retry timing across workers within a wave.

The wave loop lives entirely inside `_fetch_many()`. Callers (`get_vote_links`, `parse_vote_pages`, `enrich_legislators`) are unaffected.

## Consequences

- **Good**: The scraper becomes a better citizen — backs off when the server is struggling instead of hammering it
- **Good**: Transient 5xx clusters that previously caused 55+ permanent failures should now resolve within 1-2 waves
- **Good**: No changes to the pipeline orchestration or caller code
- **Good**: Jitter prevents thundering herd within each wave
- **Trade-off**: Worst case adds up to 4.5 minutes (3 waves x 90s cooldown) to a scrape when the server is persistently down. This is acceptable — the scraper already takes several minutes, and getting complete data is worth waiting.
- **Trade-off**: `self.delay` is mutated during waves (restored in `finally`). This is safe because `_fetch_many()` is the only caller of `_get()` during concurrent fetches, and waves run sequentially.
