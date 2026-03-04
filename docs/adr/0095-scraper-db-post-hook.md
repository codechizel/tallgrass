# ADR-0095: Scraper Post-Hook — Auto-Load CSVs into PostgreSQL (DB3)

**Date:** 2026-03-04
**Status:** Accepted

## Context

DB2 (ADR-0094) provides management commands to load scraped CSVs into PostgreSQL. After every scrape run, users must manually run `just db-load <session>` to sync the database. This is tedious and easy to forget.

DB3 adds an optional `--auto-load` flag to scraper CLIs so they can automatically load CSVs after a successful run.

## Decision

### Subprocess, not import

The scraper invokes `manage.py load_session` via `subprocess.run()` rather than importing Django. This keeps the core scraper Django-free — users who don't need the database never pull in Django dependencies. The `web` dependency group remains optional.

### Fail soft

If PostgreSQL is down, Django isn't installed, or `uv` isn't found, the hook prints a warning and continues. The scraper's primary job (writing CSVs) must never fail due to database issues.

### Three CLIs

`--auto-load` is available on:
- `tallgrass` (vote scraper)
- `tallgrass-text` (bill text fetcher)
- `tallgrass-kanfocus` (KanFocus vote scraper)

Not on `tallgrass-alec` — the ALEC corpus is an external dataset loaded via `just db-load-alec`.

### Shared helper

`src/tallgrass/db_hook.py` provides `try_load_session()` and `try_load_alec()`. Both use `_run_manage()` internally, which handles subprocess invocation, environment setup, timeout, and error reporting.

### No new Justfile recipes

Existing recipes pass `*args` through, so `just scrape 2025 --auto-load` works already.

## Consequences

- **Positive:** One-command workflow: scrape + load in a single invocation
- **Positive:** Zero coupling between scraper and Django — subprocess boundary
- **Positive:** Fail-soft guarantees scrape output is never lost due to DB issues
- **Negative:** Subprocess adds ~2-5 seconds overhead (Python startup + Django init)
- **Negative:** Error diagnostics are limited to last 3 lines of stderr

## Files

- `src/tallgrass/db_hook.py` — shared subprocess helper (new)
- `src/tallgrass/cli.py` — `--auto-load` flag
- `src/tallgrass/text/cli.py` — `--auto-load` flag
- `src/tallgrass/kanfocus/cli.py` — `--auto-load` flag
- `tests/test_db_hook.py` — 28 tests (all subprocess mocked)
