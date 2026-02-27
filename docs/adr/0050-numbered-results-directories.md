# ADR-0050: Numbered Results Directories

**Date:** 2026-02-27
**Status:** Accepted

## Context

Analysis source directories use numbered prefixes (`analysis/01_eda/`, `analysis/04_irt/`, etc.) that convey execution order at a glance. Results directories used bare names (`results/.../eda/`, `results/.../irt/`), creating an inconsistency that made `ls` output harder to read and broke the visual connection between source and output.

## Decision

Prefix all results directories with the same phase numbers used by source directories. The `analysis_name=` parameter passed to `RunContext` now includes the prefix (e.g. `"01_eda"` instead of `"eda"`). All cross-phase path references (`UPSTREAM_PHASES`, `plot_map`, per-script upstream lookups) updated to match.

Mapping:
- `01_eda`, `02_pca`, `02c_mca`, `03_umap`, `04_irt`, `05_clustering`
- `06_network`, `07_indices`, `08_prediction`, `09_beta_binomial`
- `10_hierarchical`, `11_synthesis`, `12_profiles`, `13_cross_session`, `14_external_validation`

Old results directories remain on disk as historical artifacts. New runs create numbered directories alongside them, with `latest` symlinks in the new locations.

## Consequences

- Directory listings now sort by execution order: `01_eda/`, `02_pca/`, ..., `14_external_validation/`
- Source and results directory names match, reducing cognitive overhead
- Report symlinks change (e.g. `01_eda_report.html` instead of `eda_report.html`)
- Old un-numbered results are not migrated — they remain as archives of prior runs
- Justfile recipe names (`just eda`, `just pca`) are unchanged — they're user-facing shortcuts
