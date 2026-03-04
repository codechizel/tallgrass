# ADR-0098: KanFocus Cross-Validation — Multi-Motion Matching and Slug Resolution Fixes

## Status

Accepted (2026-03-04)

## Context

The initial cross-validation run on the 91st biennium (ADR-0097) revealed two bugs in our matching code — not in the source data:

1. **Multi-motion matching**: Multiple rollcalls share `(bill, chamber, date)` — both KanFocus and kslegislature.gov have multiple amendment votes on the same bill/day. The original `find_matches()` used first-seen-wins, so N KF rollcalls all compared against the same 1 JE rollcall, inflating "genuine mismatches" with junk comparisons.

2. **Slug mismatches**: ~10 legislators had different slugs in KF vs JE because `match_to_existing()` didn't handle: (a) nickname↔formal name (Brad↔Bradley, Bill↔William), (b) chamber prefix validation (Senate slug used for House vote), (c) hyphen normalization (Faust-Goudeau vs faust_goudeau).

Combined, these produced a misleading 0% "all votes match" rate and 1,757 "genuine mismatches" when the underlying data is highly concordant.

## Decision

### Fix 1: Tally-Based Multi-Motion Matching (`crossval.py`)

Replace first-seen-wins dict in `find_matches()` with two-level grouping:

1. Group KF and JE rollcalls by `(normalized_bill_number, chamber, date)` — same key as before
2. For 1:1 groups (803 of 824 keys), match directly — no overhead
3. For multi-motion groups, sub-match by **tally vector** `(yea, nay, nv_total)` where JE `nv_total = not_voting + absent_not_voting`
4. 1:1 per tally = unique match. N:N per tally = pair positionally (ambiguous but both valid). Leftover KF → unmatched.

New helpers: `_tally_key(rc)` returns the tally vector; `_match_by_tally()` groups each side by tally key, pairs matching groups positionally, tracks consumed JE vote_ids to prevent double-matching.

### Fix 2: Slug Cross-Reference Improvements (`slugs.py`)

Three targeted fixes to `match_to_existing()` and supporting code:

**2a. Nickname table**: `_NICKNAMES` dict mapping 16 common short names to formal names (bill→william, brad→bradley, mike→michael, etc.). Reverse map `_FORMAL_TO_NICK` built at module load. `_try_aliases()` tries all alias variants as a fourth matching strategy in `match_to_existing()`.

**2b. Chamber prefix validation**: `_chamber_compatible()` verifies the matched slug's prefix (`sen_`/`rep_`) is compatible with `kf_chamber`. Reject mismatches — "Mike Thompson" in a House vote no longer returns `sen_thompson_mike_1`. Applied to all four matching strategies.

**2c. Hyphen normalization**: `normalize_name()` replaces hyphens with spaces before slug generation. "Oletha Faust-Goudeau" → "Oletha Faust Goudeau" → `sen_faust_goudeau_oletha_1`.

### Fix 3: Name-Based Fallback (`crossval.py`)

For any remaining slug mismatches in `compare_individual_votes()`, a name-based fallback matches KF-only slugs to JE-only slugs by `normalize_name()`. This catches edge cases where slug resolution wasn't sufficient but the legislator is clearly the same person (same normalized name).

## Consequences

- Multi-motion matching correctly disambiguates amendment votes on the same bill/day
- Slug resolution handles real-world name variations (nicknames, hyphens, chamber confusion)
- Name fallback provides a safety net for remaining edge cases in individual vote comparison
- 21 new tests (12 slug, 9 crossval) — total scraper tests: ~624, total project tests: ~2722
- Existing test for `generate_slug("Oletha Faust-Goudeau", "S")` updated: now produces `sen_faust_goudeau_oletha_1` (was `sen_faust-goudeau_oletha_1`)
- No breaking changes to public API — all changes are internal to matching/comparison logic
