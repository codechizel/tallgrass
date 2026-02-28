# Plan: Run-grouped results directories

## Problem
Each phase writes independently to `results/kansas/{session}/{phase}/{date}/` and updates a `latest` symlink. When running a full pipeline, phase N+1 finds phase N via `{phase}/latest` — but if an older run already exists, `latest` points to stale data until the new phase completes. Concurrent or re-runs clobber each other.

## New directory structure

```
results/kansas/84th_2011-2012/
  84th-2026-02-27T19-30-00/     ← run directory (biennium-sqldate)
    01_eda/
      plots/
      data/
      run_info.json
      run_log.txt
    02_pca/
      ...
    04_irt/
      ...
  latest → 84th-2026-02-27T19-30-00   ← symlink to most recent run
```

Format: `{legislature}{ordinal}-{YYYY}-{MM}-{DD}T{HH}-{MM}-{SS}` (hyphens instead of colons for filesystem safety). Example: `84th-2026-02-27T19-30-00`.

## Changes required

### 1. `analysis/run_context.py` — RunContext

- Add optional `run_id: str | None` parameter to `__init__`
- If `run_id` is provided, path becomes: `results/kansas/{session}/{run_id}/{analysis_name}/`
- If `run_id` is None (standalone run), keep current behavior: `results/kansas/{session}/{analysis_name}/{date}/`
- Update `latest` symlink: when `run_id` is set, the session-level `latest` symlink points to the run directory (updated by the LAST phase that finishes)
- Phase-level `latest` symlinks: still update within the run (not strictly needed but keeps backward compat)
- Report convenience symlink: adjust path to account for new nesting level

### 2. Each analysis phase script (~13 files)

- Add `--run-id` CLI argument (optional, default None)
- Pass `run_id=args.run_id` to `RunContext()`
- When `run_id` is set, look for upstream phases at `results/kansas/{session}/{run_id}/{phase}/data/` instead of `{phase}/latest/data/`
- When `run_id` is None, keep current `latest` lookup (backward compatible)

### 3. Inter-phase references

Every phase that reads upstream data has a pattern like:
```python
eda_dir = results_root / "01_eda" / "latest"
```
When `run_id` is set, these become:
```python
eda_dir = run_root / "01_eda"  # run_root = results_root / session / run_id
```

Files affected (upstream lookups):
- `analysis/02_pca/pca.py` — reads EDA
- `analysis/02c_mca/mca.py` — reads PCA
- `analysis/03_umap/umap_viz.py` — reads EDA, PCA, IRT
- `analysis/04_irt/irt.py` — reads EDA, PCA
- `analysis/05_clustering/clustering.py` — reads EDA, IRT, PCA
- `analysis/06_network/network.py` — reads EDA, IRT, clustering
- `analysis/07_indices/indices.py` — reads EDA, IRT, network, clustering
- `analysis/08_prediction/prediction.py` — reads IRT, clustering, network, PCA
- `analysis/09_beta_binomial/beta_binomial.py` — reads indices
- `analysis/10_hierarchical/hierarchical.py` — reads EDA, PCA, IRT
- `analysis/11_synthesis/synthesis_data.py` — reads all 10 upstream phases
- `analysis/12_profiles/profiles.py` — reads IRT bill params
- `analysis/13_cross_session/cross_session.py` — reads prediction
- `analysis/14_external_validation/external_validation.py` — reads IRT, hierarchical

### 4. Helper: generate run_id

Add a function to `run_context.py`:
```python
def generate_run_id(session: str) -> str:
    """Generate a run ID like '84th-2026-02-27T19-30-00'."""
    normalized = _normalize_session(session)
    legislature = normalized.split("_")[0]  # "84th"
    now = datetime.now(_CT).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{legislature}-{now}"
```

### 5. Backward compatibility

- `--run-id` is optional. Without it, everything works exactly as before.
- Old results directories are untouched.
- `synthesis_data.py` needs the most careful treatment since it iterates all upstream phases.

## Implementation order

1. `run_context.py` — add `run_id` param, `generate_run_id()`, path logic
2. Update each phase script (add `--run-id` arg, adjust upstream paths)
3. Update `synthesis_data.py` to accept run_id
4. Test with one biennium
5. ADR documenting the change
