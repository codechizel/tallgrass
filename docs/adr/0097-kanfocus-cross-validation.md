# ADR-0097: KanFocus Cross-Validation Diagnostic

## Status

Accepted (2026-03-04)

## Context

We have two independent sources for Kansas Legislature vote data: kslegislature.gov (`je_` prefix) and KanFocus (`kf_` prefix). For bienniums 84th-91st, both sources cover the same rollcalls. The gap-fill merge (ADR-0088) discards overlapping KanFocus data, trusting kslegislature.gov as authoritative. However, we had no way to verify this assumption or measure agreement between the two sources.

Understanding the degree of concordance is important for:
1. **Data quality assurance** — confirming both sources record the same votes consistently
2. **Gap-fill confidence** — if sources agree on overlap, KanFocus-only data (78th-83rd) is more trustworthy
3. **Category mapping validation** — the KanFocus→tallgrass vote category mapping (ADR-0088) combines "Absent and Not Voting" into "Not Voting"; we need to verify this is the only systematic difference

## Decision

Add `--mode crossval` to `tallgrass-kanfocus` as a read-only diagnostic that re-parses the KanFocus cache and compares overlapping rollcalls against kslegislature.gov CSVs. No data mutation, no network access.

### New Module: `src/tallgrass/kanfocus/crossval.py`

Three frozen dataclasses (`VoteMismatch`, `RollCallComparison`, `CrossValReport`) and pure functions for each stage:

**Bill number normalization**: `normalize_bill_number()` strips "Sub for" prefixes (including nested: "S Sub for Sub for HB 2007" → "HB 2007") and normalizes whitespace. KanFocus and kslegislature.gov sometimes differ on whether to include the substitution prefix.

**Rollcall matching**: `find_matches()` joins on `(normalized_bill_number, chamber, vote_date)`. For multi-motion groups (multiple rollcalls sharing the same key), sub-matches by tally vector `(yea, nay, nv_total)` to disambiguate. See ADR-0098 for the multi-motion fix.

**Vote category comparison rules**:

| KF | JE | Result |
|----|-----|--------|
| Yea | Yea | Match |
| Nay | Nay | Match |
| Present and Passing | Present and Passing | Match (already mapped by `convert_to_standard`) |
| Not Voting | Not Voting | Match |
| Not Voting | Absent and Not Voting | Compatible (ANV/NV ambiguity — not a genuine error) |
| Anything else | Different | Genuine mismatch |

**Tally comparison**: Yea, Nay, Present require exact match. Not Voting has two levels: exact (`kf_nv == je_nv`) and compatible (`kf_nv == je_nv + je_anv`).

**Cache re-parsing**: `load_kf_from_cache()` iterates the same `(vote_num, year, chamber)` streams as the fetcher, reconstructing URLs to find hash-keyed cache files. Calls `parse_vote_page()` + `convert_to_standard()` — reuses all existing parsing infrastructure.

### CLI Integration

`crossval` added to `--mode` choices. Handler skips fetcher creation (no network needed), loads existing slugs for cross-reference, calls `run_crossval()`, writes markdown report to data dir.

### Output

`data/kansas/{name}/crossval_report.md` with sections: Summary table, Tally Agreement, Passed/Failed Agreement, Individual Vote Agreement, Tally Mismatches (detail), Individual Vote Mismatches (detail), Bill Number Normalizations, Unmatched KF Rollcalls.

### Justfile

`just kanfocus-crossval 2025` recipe as a convenience alias.

## Consequences

- Pure diagnostic — zero impact on existing data or pipeline
- Quantifies agreement between two independent data sources for the first time
- ANV/NV ambiguity explicitly modeled as "compatible" rather than "mismatch"
- Re-parses cache without network access — safe to run anytime
- Bill number normalization handles KanFocus-specific substitution prefixes
- 37 new tests covering normalization, matching, comparison, and report formatting
- Multi-motion matching and slug resolution bugs fixed in ADR-0098
- Future use: run across all 8 overlapping bienniums to build a data quality baseline
