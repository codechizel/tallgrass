# ADR-0014: Clustering Visualization Overhaul

**Date:** 2026-02-22
**Status:** Accepted

## Context

The clustering phase produced dendrograms for both chambers, but these were unreadable for the nontechnical audience:

- **House dendrogram**: truncated to 12 nodes showing only cluster sizes like "(52)" — conveyed almost no information about individual legislators.
- **Senate dendrogram**: tall and dense, requiring statistical training to interpret branch heights and leaf ordering.
- **Name extraction bug**: `full_name.split()[-1]` failed on leadership suffixes like "Tim Shallenburger - Vice President of the Senate" → produced "Senate" instead of "Shallenburger". Also failed silently for duplicate last names (two Claeys senators, two Carpenters in the House).

The project's analytic workflow rules state: "Prefer concrete, narrative-friendly visualizations (ranked bar charts, annotated scatter plots) over abstract statistical plots (dendrograms, eigenvalue scree plots)."

## Decision

1. **Keep existing dendrograms** as supplementary figures (unchanged).
2. **Add three new alternative visualizations**, each showing the same hierarchical clustering data in a different format:
   - **Voting Blocs (sorted dot plot)**: legislators ordered by dendrogram leaf position, x-axis = IRT ideal point, party-colored dots. Most readable for nontechnical audiences.
   - **Polar Dendrogram (circular tree)**: the full dendrogram rendered in polar coordinates. Compact and visually striking, but labels crowd at 130+ members. Uses radial staggering (alternating inner/outer label lanes) for collision avoidance.
   - **Icicle Chart (flame chart)**: top-down rectangular hierarchy showing merge distances as vertical position, subtrees colored by majority party. Good for showing the hierarchical structure without requiring dendrogram literacy.
3. **Centralize name extraction** in a `_build_display_labels()` helper that:
   - Strips leadership suffixes (splits on " - ")
   - Detects duplicate last names via Counter
   - Disambiguates with abbreviated first-name prefix (e.g., "Jo. Claeys" vs "J.R. Claeys")

All three functions share the same signature: `(Z, slugs, ideal_points, chamber, out_dir)`.

## Consequences

**Benefits:**
- Nontechnical audiences can understand voting bloc structure without dendogram literacy
- Three different views let consumers pick the one that resonates
- Centralized name extraction prevents the `.split()[-1]` bug across all new plots
- Original dendrograms preserved for statistical audiences

**Trade-offs:**
- Polar dendrogram labels are still hard to read at 130 members (House) — the voting blocs and icicle charts handle House better
- `_build_display_labels()` is local to clustering.py; other modules that extract last names (prediction, profiles) use their own `.split(" - ")[0].split()[-1]` pattern and would benefit from a shared utility in future

**Risk:**
- The radial staggering heuristic (gap < 0.8 × even_sep → outer lane) is empirical. Future sessions with different legislator counts may need tuning.
