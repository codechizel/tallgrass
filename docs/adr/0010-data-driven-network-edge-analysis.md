# ADR-0010: Data-Driven Network Edge Weight Analysis

**Date:** 2026-02-21
**Status:** Accepted

## Context

The network analysis (`network.py`) originally contained `check_tyson_thompson_edge_weights()`, which hardcoded two Senate slugs (`sen_tyson_caryn_1`, `sen_thompson_mike_1`) and only ran for the Senate chamber. The function compared their within-Republican edge weights to the party median — a valuable analysis, but one that would produce no output for any other biennium where those specific legislators don't exist.

This parallels the synthesis hardcoding problem solved in ADR-0008. The analytical concept — "do the most ideologically extreme majority-party members have weaker intra-party connections?" — is session-independent. The specific legislators should be detected from data.

## Decision

Replace `check_tyson_thompson_edge_weights()` with `check_extreme_edge_weights()`:

1. **Majority party detection:** Determined by node count in the graph (handles any party composition, not just Republican supermajority).
2. **Legislator selection:** Top `top_n` (default 2) majority-party legislators by highest `|xi_mean|` (IRT ideal point magnitude). This naturally selects the most ideologically extreme members.
3. **Both chambers:** Runs for House and Senate (the old function was Senate-only).
4. **Output:** Per-legislator intra-party edge weight statistics (mean, median, min) and gap versus party median, plus the legislator's IRT ideal point for context.

Additionally, network visualization was enhanced with data-driven annotations:
- **Bridge legislator rings:** Top 3 betweenness centrality nodes get red ring annotation on network layout plots.
- **Centrality ranking bar chart:** New `plot_centrality_ranking()` shows all legislators ranked by betweenness, party-colored.
- **Threshold sweep enhancements:** Default threshold line (blue dashed) and stability zone shading (green) on sweep plots.
- **Edge weight distribution:** Threshold line and interpretation text box added.

## Consequences

**Positive:**
- Network edge weight analysis works on any biennium without code changes.
- Both chambers are analyzed (previously Senate-only), surfacing House outliers that were missed.
- Bridge legislator annotations make network plots self-explanatory for nontechnical audiences.
- Stability zone and threshold line on sweep plots communicate robustness without requiring statistical expertise.

**Negative:**
- `top_n=2` is a fixed default. Sessions with more or fewer extreme legislators may need adjustment.
- IRT ideal points must be present as node attributes (`xi_mean`). If IRT hasn't been run, the function returns `None`.

**Files changed:**
- `analysis/network.py` — Renamed function, generalized logic, added visualization enhancements (+398/-80 lines).
- `analysis/network_report.py` — Renamed `_add_tyson_thompson` to `_add_extreme_edge_analysis`, added `_add_centrality_ranking_figure`, updated captions (+78/-80 lines).

## Update: Community Composition n_other Column (2026-02-23)

A cross-biennium audit discovered that `analyze_community_composition()` only tracked `n_republican` and `n_democrat` columns. In the 89th biennium (2021-22), Dennis Pyle (Independent) was invisible in community composition tables — community 2 showed 7 Republicans but the 8th member was unaccounted for. Added `n_other` column that counts non-R/D legislators, ensuring `n_republican + n_democrat + n_other == n_legislators` always holds. Four tests added to verify the invariant.
