# ADR-0085: OpenStates OCD Person IDs for Stable Legislator Identity

**Date:** 2026-03-02
**Status:** Accepted

## Context

Cross-session legislator matching uses `normalize_name(full_name)` everywhere — Phase 13 (cross-session), Phase 16 (dynamic IRT), Phase 14/14b (external validation). This breaks for same-name legislators (two Mike Thompsons: one Senate, one House), and districts change with redistricting (Chase Blasi: District 27 → 26). The KS Legislature slug (`sen_blasi_chase_1`) is session-scoped. With planned multi-state expansion, name collisions will only get worse.

[OpenStates](https://github.com/openstates/people) maintains canonical legislator data for all US states, using OCD (Open Civic Data) person IDs (`ocd-person/{uuid}`) as persistent identifiers. Their Kansas coverage spans 2011-present (487 legislators: 165 current + 322 retired), the data is CC0-licensed, and their YAML files contain KS Legislature member URLs from which we can extract our exact slugs for joining.

## Decision

Adopt OpenStates OCD person IDs as the canonical cross-biennium legislator identifier.

### Implementation

1. **New module `src/tallgrass/roster.py`** — Downloads `openstates/people` repo as a GitHub tarball (single HTTP request), parses YAML files for `data/ks/` (both `legislature/` and `retired/`), extracts slugs from KS Legislature member URLs in the `links:` field, builds a slug→ocd_id mapping cached as JSON.

2. **Scraper integration** — At the end of `enrich_legislators()`, loads the slug lookup and attaches `ocd_id` to each legislator. If roster not synced, all legislators get `ocd_id=""` with a hint message.

3. **Output** — `ocd_id` added as 8th column in legislators CSV.

4. **Backward compatibility** — `phase_utils._clean_legislators()` adds an empty `ocd_id` column when loading older CSVs that lack it.

5. **Cross-session matching** — `match_legislators()` uses a 3-phase strategy: (0) OCD ID join, (1) name-norm join on unmatched remainder, (2) optional fuzzy matching. Output schema unchanged — downstream consumers need zero changes.

6. **Dynamic IRT** — `build_global_roster()` groups by `ocd_id` when available, falling back to `name_norm`. This correctly separates same-name legislators into distinct `global_idx` entries.

### Data flow

```
just roster-sync
  → downloads openstates/people tarball
  → parses data/ks/*.yml (legislature/ + retired/)
  → writes data/external/openstates/ks_roster.json (full records)
  → writes data/external/openstates/ks_slug_to_ocd.json (flat lookup)

just scrape 2025
  → enrich_legislators() reads ks_slug_to_ocd.json
  → sets info["ocd_id"] per legislator
  → save_csvs() writes ocd_id column

just cross-session / just dynamic-irt
  → loads legislators CSV with ocd_id column
  → Phase 0: join on ocd_id (when both sessions have it)
  → Phase 1: fall back to name_norm join (backward compat)
```

### Dependencies

- `pyyaml>=6.0` added to `[dependency-groups].dev` (only needed for `sync_roster()`, not at scrape time)
- OpenStates data: CC0 license, GitHub tarball (no API key needed)

## Consequences

**Positive:**
- Same-name legislators (two Mike Thompsons) correctly separated across sessions
- District changes (redistricting) no longer break matching
- Multi-state ready — `state` parameter defaults to "ks", cache files namespaced
- Backward compatible — older CSVs without `ocd_id` still work via name matching
- No runtime cost at scrape time — fast JSON read only

**Negative:**
- Requires one-time `just roster-sync` before first use
- Depends on OpenStates data quality for slug extraction (mitigated: well-maintained CC0 project)
- ~487 entries in lookup for Kansas — scales linearly with state count

**Neutral:**
- Existing analysis results unchanged — OCD ID matching produces same results for non-colliding names, and correctly handles the edge cases that name matching gets wrong
