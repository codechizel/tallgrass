# ADR-0040: Rename Package to Tallgrass

**Date:** 2026-02-25
**Status:** Accepted

## Context

The project started as a single-purpose Kansas Legislature vote scraper (`ks-vote-scraper`), but has grown into a full analysis platform with 14 analysis phases, Bayesian models, and cross-session validation. The name "ks-vote-scraper" undersells the scope and is awkward as a CLI command.

## Decision

Rename the package from `ks-vote-scraper` (Python: `ks_vote_scraper`) to `tallgrass` — short, memorable, distinctly Kansas.

Changes:

- **Package directory**: `src/ks_vote_scraper/` → `src/tallgrass/`
- **pyproject.toml**: `name = "tallgrass"`, entrypoint `tallgrass = "tallgrass.cli:main"`
- **CLI**: `uv run tallgrass` (was `uv run ks-vote-scraper`)
- **All imports**: `from tallgrass.session import KSSession` etc. (~32 files updated)
- **Justfile**: 3 command invocations updated
- **Documentation**: CLAUDE.md, MEMORY.md, ADR index, roadmap, deep dives

No changes to data model, output paths, class names (e.g. `KSVoteScraper`, `KSSession`), or analysis pipeline behavior. This is purely a naming change.

## Consequences

- **CLI invocation changes** — `uv run tallgrass` replaces `uv run ks-vote-scraper`. All `just` recipes are unaffected (they wrap the CLI).
- **Import paths change** — any external code importing `ks_vote_scraper` must update to `tallgrass`.
- **No data migration needed** — output directories, CSV filenames, and cache paths are unchanged.
- **Class names preserved** — `KSVoteScraper`, `KSSession`, etc. retain their `KS` prefix since they still scrape the Kansas Legislature specifically.
