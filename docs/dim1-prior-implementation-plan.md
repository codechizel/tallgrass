# Implementation Plan: `--dim1-prior` Flag

**Date:** 2026-03-11
**Prereqs:** ADR-0103, ADR-0104, ADR-0107
**Article:** `docs/dim1-informative-prior.md`

## Overview

Add a `--dim1-prior` robustness flag that uses 2D IRT Dimension 1 scores as informative priors
for 1D IRT ideal points. Reuses the existing `external-prior` identification strategy for flat
models and adds a `pm.Potential` for hierarchical models.

## Scope

| Component | Change | Complexity |
|-----------|--------|------------|
| Phase 05 (`irt.py`) | New CLI flag, wire 2D Dim 1 into `external-prior` | Low (plumbing only) |
| Phase 07 (`hierarchical.py`) | New CLI flag, `pm.Potential` on composed xi | Medium |
| Experimental joint (`joint_irt_experiment.py`) | New CLI flag, same as Phase 05 | Low |
| `init_strategy.py` | No changes (already loads 2D scores) | None |
| ADR | New ADR documenting the decision | Low |
| Docs | CLAUDE.md update, existing article covers rationale | Low |

## Step-by-Step

### Step 1: Phase 05 — Flat IRT (`analysis/05_irt/irt.py`)

**1a. Add CLI flags**

In `parse_args()`, add to the robustness group (after `--promote-2d`):

```python
robustness.add_argument(
    "--dim1-prior",
    action="store_true",
    help="Use 2D IRT Dimension 1 as informative prior for ideology recovery "
    "(requires Phase 06 results; implies --promote-2d)",
)
robustness.add_argument(
    "--dim1-prior-sigma",
    type=float,
    default=1.0,
    help="Width of the Dim 1 informative prior (default: 1.0; lower = stronger constraint)",
)
```

After parsing, add implication: `--dim1-prior` implies `--promote-2d` (for automatic
cross-referencing in the report):

```python
if parsed.dim1_prior:
    parsed.promote_2d = True
```

**1b. Add constant**

```python
DIM1_PRIOR_SIGMA_DEFAULT = 1.0
```

**1c. Load 2D Dim 1 scores**

The existing `--promote-2d` / `--init-strategy 2d-dim1` code already loads 2D scores. Extend
the loading block (around line 3827) to also load when `--dim1-prior` is set:

```python
need_2d = args.promote_2d or args.init_strategy == "2d-dim1" or args.dim1_prior
```

This single-line change ensures 2D scores are loaded for the new flag.

**1d. Wire into the per-chamber loop**

In the per-chamber sampling loop, after anchor selection and before `build_and_sample()`,
add a conditional block:

```python
if args.dim1_prior:
    ch_lower = chamber.lower()
    dim1_scores = irt_2d_scores.get(ch_lower)
    if dim1_scores is None:
        print(f"  WARNING: 2D IRT results not found for {chamber} — skipping dim1-prior")
    else:
        # Build per-legislator prior means from 2D Dim 1
        dim1_map = {
            row["legislator_slug"]: row["xi_dim1_mean"]
            for row in dim1_scores.iter_rows(named=True)
        }
        dim1_raw = np.array([dim1_map.get(s, 0.0) for s in data["leg_slugs"]])
        dim1_std_val = dim1_raw.std()
        if dim1_std_val > 0:
            dim1_std = (dim1_raw - dim1_raw.mean()) / dim1_std_val
        else:
            dim1_std = dim1_raw

        matched_dim1 = sum(1 for s in data["leg_slugs"] if s in dim1_map)
        print(
            f"  Dim 1 prior: {matched_dim1}/{data['n_legislators']} matched, "
            f"sigma={args.dim1_prior_sigma}"
        )

        # Override strategy and anchors for external-prior
        strategy = IS.EXTERNAL_PRIOR
        anchors = []  # no hard anchors — prior identifies
        external_priors = dim1_std.astype(np.float64)
        external_prior_sigma = args.dim1_prior_sigma

        # Also use as init (belt and suspenders)
        xi_init = dim1_std.astype(np.float64)
```

Then pass these into `build_and_sample()`:

```python
idata, sampling_time = build_and_sample(
    data=data,
    anchors=anchors,
    n_samples=args.n_samples,
    n_tune=args.n_tune,
    n_chains=args.n_chains,
    xi_initvals=xi_init,
    strategy=strategy,
    external_priors=external_priors,
    external_prior_sigma=external_prior_sigma,
)
```

When `--dim1-prior` is not set, these variables keep their existing values (anchor-based
strategy, PCA init, no external priors). No changes to the non-dim1-prior path.

**1e. Report integration**

Add a conditional section to the HTML report (alongside the existing horseshoe remediation
section) that documents:
- That `--dim1-prior` was used
- The prior sigma
- Match count (how many legislators had 2D Dim 1 scores)
- Comparison with the standard model (if both were run)

### Step 2: Phase 07 — Hierarchical IRT (`analysis/07_hierarchical/hierarchical.py`)

**2a. Add CLI flags**

Same two flags as Phase 05: `--dim1-prior` and `--dim1-prior-sigma`.

**2b. Modify `build_per_chamber_graph()`**

Add optional parameter:

```python
def build_per_chamber_graph(
    data,
    beta_prior=PRODUCTION_BETA,
    dim1_prior: np.ndarray | None = None,
    dim1_prior_sigma: float = 1.0,
) -> pm.Model:
```

Inside the model context, after the `xi = mu_party + sigma_within * xi_offset` composition,
add:

```python
if dim1_prior is not None:
    pm.Potential(
        "dim1_prior",
        pm.logp(pm.Normal.dist(mu=dim1_prior, sigma=dim1_prior_sigma), xi),
    )
```

This adds a log-probability contribution equivalent to observing `xi` through a Normal
likelihood centered on the 2D Dim 1 values. It does not change the hierarchical
decomposition — `mu_party`, `sigma_within`, and `xi_offset` retain their original priors
and structure. The Potential simply adds a soft pull toward the ideology axis.

**2c. Modify `build_joint_graph()`**

Same pattern: add `dim1_prior` and `dim1_prior_sigma` parameters. After the composed `xi`
is built (from the concatenated House + Senate offset vectors), add the same `pm.Potential`.

The caller is responsible for concatenating the 2D Dim 1 scores in the correct order
(House first, Senate second) to match `build_joint_graph()`'s legislator ordering.

**2d. Wire into `main()`**

In the per-chamber loop, load 2D scores and pass to `build_per_chamber_graph()`.

In the joint model section, concatenate 2D Dim 1 scores from both chambers and pass to
`build_joint_graph()`.

### Step 3: Experimental Joint IRT (`analysis/experimental/joint_irt_experiment.py`)

**3a. Add CLI flags**

Same two flags.

**3b. Load 2D scores**

Add upstream resolution for Phase 06:

```python
if args.dim1_prior:
    irt_2d_dir = resolve_upstream_dir("06_irt_2d", results_root)
```

**3c. Build prior array**

Merge 2D Dim 1 scores from both chambers into a single array matching the joint matrix's
legislator ordering:

```python
dim1_map = {}
for ch in ("house", "senate"):
    scores = load_2d_scores(irt_2d_dir / "data", ch)
    if scores is not None:
        for row in scores.iter_rows(named=True):
            dim1_map[row["legislator_slug"]] = row["xi_dim1_mean"]

dim1_raw = np.array([dim1_map.get(s, 0.0) for s in joint_slugs])
# ... standardize ...
```

**3d. Override strategy**

Same as Phase 05: switch to `EXTERNAL_PRIOR` strategy with dim1 values as priors, no hard
anchors.

### Step 4: Documentation

**4a. New ADR**

Create `docs/adr/adr-0108-dim1-informative-prior.md`:
- Context: horseshoe effect in supermajority chambers, limitation of initialization-only approach
- Decision: use 2D IRT Dim 1 as informative prior via existing `external-prior` strategy
- Consequences: requires Phase 06 to have run first (reverse dependency), adds a new
  robustness flag, subsumes `--horseshoe-remediate` for cases where 2D results are available

**4b. Update CLAUDE.md**

Add `--dim1-prior` to the MCMC flags list:

```
- Robustness flags: `--horseshoe-diagnostic`, `--horseshoe-remediate`, `--contested-only`,
  `--promote-2d`, `--dim1-prior` (ADR-0108)
```

**4c. Update ADR index**

Add entry in `docs/adr/README.md` under the IRT section.

### Step 5: Testing

**5a. Unit tests**

- Test that `--dim1-prior` implies `--promote-2d`
- Test that `build_per_chamber_graph(dim1_prior=...)` adds a Potential named `"dim1_prior"`
- Test that `build_joint_graph(dim1_prior=...)` adds a Potential named `"dim1_prior"`
- Test that the flat model with `external_priors=dim1_values` produces a model with
  per-legislator prior means (existing external-prior tests cover this)

**5b. Integration test**

Run the 79th Senate with `--dim1-prior` and verify:
- Huelskamp and Oleen appear on the conservative end
- Democrat wrong-side fraction = 0%
- Correlation with 2D Dim 1 > 0.85
- Convergence passes (R-hat < 1.01, ESS > 400)

## Recommendations

1. **Default sigma = 1.0.** This matches the validated PC2 prior sigma from
   `--horseshoe-remediate` and provides moderate constraint. Expose as `--dim1-prior-sigma`
   for sensitivity analysis.

2. **Don't auto-trigger.** Like `--horseshoe-remediate`, this should be an explicit user
   decision. The report can *recommend* it (when `--horseshoe-diagnostic` detects distortion
   and 2D results are available), but the user runs it manually.

3. **Combine with `--init-strategy 2d-dim1`.** Use Dim 1 for both initialization AND priors.
   Initialization gets chains to the right neighborhood; priors keep them there. Belt and
   suspenders.

4. **Phase 05 first, then Phase 07.** The flat IRT change is a pure wiring exercise (no new
   model code). The hierarchical change requires the `pm.Potential` addition. Ship Phase 05
   first, validate on the 79th, then extend to Phase 07.

5. **Report the prior's influence.** When `--dim1-prior` is used, always report:
   - Prior sigma used
   - Correlation between prior-informed and standard ideal points (if standard was also run)
   - Posterior predictive accuracy comparison
   This lets readers assess how much the prior changed the results.

## Non-Goals

- **Replacing the 1D model with 2D Dim 1 directly.** The 1D model with informative priors
  still refines the estimates using the full vote matrix. Direct substitution would lose this
  refinement.

- **Automatic 2D-to-1D promotion.** The 2D model has relaxed convergence thresholds
  (R-hat < 1.05 vs. < 1.01) and is considered experimental. It should inform the 1D model,
  not replace it.

- **Modifying the 2D model.** The 2D IRT (Phase 06) is the upstream source. Its PLT
  identification and convergence criteria are unchanged by this work.

## File Checklist

| File | Action |
|------|--------|
| `analysis/05_irt/irt.py` | Add `--dim1-prior`, `--dim1-prior-sigma` flags; wire 2D Dim 1 into `external-prior` |
| `analysis/07_hierarchical/hierarchical.py` | Add flags; add `dim1_prior` param to graph builders; add `pm.Potential` |
| `analysis/experimental/joint_irt_experiment.py` | Add flags; load 2D scores; override strategy |
| `docs/adr/adr-0108-dim1-informative-prior.md` | New ADR |
| `docs/adr/README.md` | Add ADR-0108 to index |
| `CLAUDE.md` | Add `--dim1-prior` to flags list |
| `tests/` | Unit tests for new graph builder parameters; integration test on 79th |
