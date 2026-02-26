# ADR-0021: Independent Party Handling Across Analysis Pipeline

**Date:** 2026-02-22
**Status:** Accepted

## Context

The scraper's party detection only recognized "Republican" and "Democrat" (via string matching on the `<h2>District N - Party</h2>` element). When Dennis Pyle appeared as "District 1 - Independent" in the 89th biennium (2021-22), the scraper stored an empty string for his party. This caused cascading failures across the analysis pipeline:

1. **EDA crash**: `f"{row['party']:12s}"` on a None value
2. **Prediction crash**: null `vote_type` formatting (separate but related data quality issue)
3. **Hierarchical IRT**: `PARTY_IDX_MAP` only had Republican/Democrat keys — Independent would produce a KeyError
4. **Party unity/loyalty**: computed against wrong party direction
5. **PARTY_COLORS**: only defined for R/D — Independent legislators rendered with fallback gray only in some files
6. **Plots/legends**: hardcoded `["Republican", "Democrat"]` loops skipped Independents entirely

Kansas has had Independent legislators in historical sessions (Pyle switched affiliation before the 2022 governor's race). As the project extends back to 2011, more non-major-party legislators may appear.

## Decision

1. **Fill null/empty party to "Independent" at every CSV load point** — each analysis phase's `load_data()` / `load_metadata()` now includes `pl.col("party").fill_null("Independent").replace("", "Independent")` alongside the existing `strip_leadership_suffix()` call. This handles both null values (Polars null) and empty strings (CSV empty field).

2. **Add "Independent" to PARTY_COLORS** in all 12 analysis modules: `"Independent": "#999999"` (gray). Also added `"Independent": "#CCCCCC"` to `PARTY_COLORS_LIGHT` in synthesis and profiles.

3. **Exclude Independents from party-specific models** rather than trying to shoehorn them in:
   - **Hierarchical IRT**: filter Independent legislators from the vote matrix before model fitting (the 2-party partial pooling model requires party membership). Print a warning listing excluded legislators. Independents still appear in flat IRT results.
   - **Party unity/maverick** (indices.py): skip non-R/D legislators in `compute_unity_and_maverick()` (`party not in ("Republican", "Democrat")`). CQ party unity is undefined for Independents.
   - **Beta-binomial**: already loops `for party in ["Republican", "Democrat"]` — Independents naturally excluded.

4. **Make plots and text dynamic**:
   - Party vote breakdown loop and plot now iterate over parties actually present in data (not hardcoded R/D list).
   - Legends built from parties present in the data, not hardcoded R/D.
   - Synthesis cluster column detection is dynamic (`cluster_k*` instead of hardcoded `cluster_k2`).

5. **Fix umap_viz.py Python 2 except syntax** (found during audit): `except FileNotFoundError, OSError:` → `except (FileNotFoundError, OSError):` at 4 locations.

## Consequences

**Benefits:**
- Pipeline runs cleanly on sessions with Independent legislators (validated on 89th, 2021-22)
- All 12 analysis phases produce correct output for Dennis Pyle
- PARTY_COLORS consistent across all modules — Independent renders as gray in all plots
- Party-specific models correctly exclude legislators who don't belong to a party
- Dynamic plots adapt to any party composition without code changes

**Trade-offs:**
- Independents are excluded from hierarchical IRT, party unity, and beta-binomial — they get flat IRT ideal points but no party-specific Bayesian estimates. This is analytically correct (partial pooling by party is undefined for a one-member "party") but means Independent profiles in synthesis have null values for party-specific metrics.
- The `ideology_label()` function in synthesis_detect.py returns "moderate" for Independents (neither "conservative Republican" nor "liberal Democrat"). This is a reasonable default but loses information.
- The scraper itself still only detects R/D/empty — the analysis pipeline handles the mapping. A future scraper fix to detect "Independent" directly would be cleaner but is not blocking.

**Risk:**
- If a future session has many Independents (e.g., 5+), the hierarchical model's party-level estimates would be unaffected (Independents are excluded), but synthesis and profiles would have more null party-specific metrics. Consider a 3-party hierarchical model if this becomes common.

## Update: Synthesis Detect Crash Fix (2026-02-26)

Running the 89th biennium synthesis revealed that Independents still caused a crash in `synthesis_detect.py`. Dennis Pyle had null `unity_score` (party unity is undefined for a one-member party), and:

1. `_minority_parties()` included "Independent" in its results, causing `detect_chamber_maverick()` to attempt maverick analysis on a party with undefined unity scores.
2. `detect_chamber_maverick()` did not filter null `unity_score` values before computing percentiles.

**Fix:** `_minority_parties()` now excludes "Independent" from results, and `detect_chamber_maverick()` adds a null filter on `unity_score`. See ADR-0042.
