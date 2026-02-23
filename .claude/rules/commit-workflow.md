# Commit Workflow

## Version Tag Format
`[vYYYY.MM.DD.N]` where N is the sequential number for that day (starting at 1).

## Commit Message Format
```
type(scope): description [vYYYY.MM.DD.N]
```

**No Co-Authored-By lines.** Never append co-author trailers to commits.

### Types
- feat: New feature
- fix: Bug fix
- refactor: Code restructuring
- docs: Documentation only
- test: Adding/updating tests
- chore: Maintenance, deps, config
- style: Formatting, no logic change

### Scopes
- scraper: Core scraping pipeline
- models: Data models
- output: CSV export
- session: Session/biennium URL logic
- config: Settings/config changes
- cli: Command-line interface
- docs: Documentation files
- infra: CI, hooks, tooling

## Post-Feature Workflow

After completing a feature or fix, **always update documentation before committing:**

1. **Update existing docs** — check CLAUDE.md, relevant ADRs, roadmap, and design docs for sections that reference the changed code. Update them to reflect the new behavior.
2. **Create a new ADR if warranted** — if the change introduces a new architectural pattern, a non-obvious technical decision, or a trade-off worth recording for future contributors, create `docs/adr/NNNN-title.md` and add it to the ADR index. Not every feature needs one — bug fixes, small enhancements, and straightforward additions don't. Use judgment.
3. **Include doc changes in the same commit** — code and its documentation ship together, not separately.

## Pre-Commit Checklist
1. Run `just lint` — must pass
2. Stage relevant files (NOT `data/`, `.env`, or secrets)
3. Commit with conventional message and version tag

## Pushing
**Never push without explicit permission.** Commit as often as useful; push only when asked. We are often on networks that don't allow SSH, so pushing is done manually or by explicit request.
