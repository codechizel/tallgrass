# ADR-0029: Leiden community detection (replacing Louvain)

**Date:** 2026-02-24
**Status:** Accepted

## Context

Network analysis (Phase 6) used Louvain community detection via the `python-louvain` package. The original rationale (ADR-0007 design doc) was that Louvain and Leiden produce nearly identical results at ~170 nodes, and `igraph` (required by `leidenalg`) was a C-library dependency that complicated installation.

Three factors prompted reconsideration:

1. **Louvain quality guarantee issue.** Traag et al. (2019) proved that Louvain can produce badly connected communities — up to 25% in some cases. Leiden guarantees well-connected communities via an additional refinement phase.

2. **igraph is now pip-installable.** Since igraph v0.10 (2022), `pip install igraph` works on all platforms without separate C library installation. The dependency concern is obsolete.

3. **Kansas three-faction structure.** Kansas historically had three factions: conservative Republicans, moderate Republicans, and Democrats. The moderate R wing (~30-40 legislators) falls below modularity's theoretical resolution limit of sqrt(2m) ≈ 50 edges (Fortunato & Barthelemy, 2007). CPM (Constant Potts Model), available through `leidenalg`, is resolution-limit-free and can detect subcaucuses of any size.

Additionally, a code audit found three `except` syntax bugs (`except ValueError, ZeroDivisionError:` — Python 2 syntax that only catches the first exception type in Python 3) and identified opportunities for a shared graph builder helper, polarization metric, and backbone extraction.

## Decision

1. **Replace `python-louvain` with `igraph>=0.11` + `leidenalg>=0.11`** in pyproject.toml.

2. **Use Leiden with RBConfigurationVertexPartition** (modularity optimization) as a drop-in replacement for Louvain at the same 8 resolution parameters.

3. **Add CPM (CPMVertexPartition) sweep** at 8 gamma values (0.05–0.50) alongside the modularity sweep. CPM is resolution-limit-free and directly addresses the three-faction detection question.

4. **Add party modularity** (Waugh et al., 2009) as a quantitative polarization metric.

5. **Add disparity filter backbone** (Serrano et al., 2009) for statistically significant edge extraction.

6. **Fix `except` tuple syntax bugs** at three locations (Python 2 syntax that only catches the first exception type).

7. **Extract `_graph_from_kappa_matrix()` shared helper** used by both `build_kappa_network()` and `_build_network_from_vote_subset()`.

## Consequences

- **Dependencies change:** `python-louvain` removed, `igraph` + `leidenalg` added. Both are pip-installable.
- **Constant renamed:** `LOUVAIN_RESOLUTIONS` → `LEIDEN_RESOLUTIONS`. New constant `CPM_GAMMAS`.
- **ty config updated:** `community.**` replaced with `igraph.**` + `leidenalg.**` in replace-imports-with-any.
- **New output files:** `cpm_resolution_{chamber}.parquet` alongside existing community resolution files.
- **Backward compatible:** All existing function signatures unchanged. Tests pass (34 existing + 19 new = 53 total).
- **Network report text updated:** All "Louvain" references changed to "Leiden".
- **Bug fix:** Three `except` clauses now correctly catch all specified exception types. Previously, `ZeroDivisionError` was silently ignored in Kappa computation and assortativity.
