# ADR-0024: Instruction File Restructure

**Date:** 2026-02-24
**Status:** Accepted

## Context

The project's Claude Code instruction files had grown to ~718 lines across 5 files, loading ~56K tokens into every conversation. Anthropic's guidance recommends keeping CLAUDE.md under 150-200 lines for reliable instruction-following. Key problems:

1. **CLAUDE.md was 383 lines** — 2x the recommended maximum. Detailed architecture, test inventories, and analysis phase descriptions loaded in every session regardless of task.
2. **~40% of MEMORY.md duplicated CLAUDE.md** — HTML parsing pitfalls, session coverage, biennium naming, analysis phases, and technology preferences appeared in both files.
3. **MEMORY.md exceeded the 200-line auto-memory limit** (212 lines) — the last 12 lines were silently truncated every session.
4. **No path scoping on rule files** — analysis-specific rules loaded during pure scraper work and vice versa.
5. **documentation.md was 19 lines** — too small for a standalone file; its content was closely related to commit-workflow.md.

## Decision

Restructure the instruction files following Anthropic's recommended patterns:

**CLAUDE.md (383 → 146 lines):** Keep only essentials that apply to every session — commands, code style, HTML parsing pitfalls, session URL logic, data model, concurrency summary, and testing commands. Replace detailed sections with cross-references to scoped rule files.

**New path-scoped rule files** (using `paths:` frontmatter so they load only when relevant files are touched):
- `scraper-architecture.md` (paths: `src/**/*.py`) — session coverage table, retry strategy, concurrency details, ODT parser, vote deduplication
- `analysis-framework.md` (paths: `analysis/**/*.py`) — 12-phase pipeline, report system, design doc index, technology preferences, Kansas-specific notes
- `testing.md` (paths: `tests/**/*.py`) — test file inventory, conventions, manual verification

**Merged documentation.md into commit-workflow.md** — documentation standards (format, primers, ADRs) merged into the post-feature workflow section where they naturally belong.

**MEMORY.md (212 → 37 lines):** Slimmed to a concise index with cross-references to three new topic files:
- `analysis-phases.md` — phase constants, IRT convergence failures, pipeline status, field evaluation
- `hardware.md` — Apple Silicon CPU scheduling, thread pool rules, measured performance
- `historical-sessions.md` — JS discovery, ODT parsing, member directory fallback, per-biennium caveats

## Consequences

**Benefits:**
- Always-loaded context reduced from ~56K to ~26K tokens (53% reduction)
- CLAUDE.md at 146 lines (within Anthropic's recommended range)
- MEMORY.md at 37 lines (well under 200-line limit, no truncation)
- Scraper-specific detail only loads when working on `src/`; analysis detail only loads when working on `analysis/`; test inventory only loads when working on `tests/`
- Zero duplication between CLAUDE.md and MEMORY.md
- Documentation standards now live alongside the commit workflow they're part of

**Trade-offs:**
- More files to maintain (5 rule files instead of 3, 4 memory files instead of 1)
- Cross-references require keeping file names stable
- Path scoping means context may be absent when working on files outside the scoped paths (mitigated by CLAUDE.md containing the critical guardrails like HTML parsing pitfalls)
