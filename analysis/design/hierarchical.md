# Hierarchical Bayesian IRT Design Choices

**Script:** `analysis/hierarchical.py`
**Constants defined at:** `analysis/hierarchical.py` (top of file)

## Assumptions

1. **Partial pooling by party.** Legislators within the same party (and chamber) share a common party-level mean ideal point. Individual ideal points are drawn from a Normal distribution centered on their party mean, with party-specific within-group variance. This is the standard 2-level hierarchical IRT specification.

2. **Two parties only.** The model assumes exactly two parties (Republican, Democrat). Independents or third-party legislators would need to be either assigned to the closest party or excluded. Kansas currently has none.

3. **Per-chamber primary, joint secondary.** The primary analysis runs separate 2-level models for House and Senate (consistent with the flat IRT phase). A secondary 3-level joint model adds a chamber level but may fail to converge due to sparse cross-chamber overlap (only ~71 shared bills for ~169 legislators).

4. **Non-centered parameterization.** The hierarchical model uses non-centered parameterization (`xi = mu_party + sigma_within * xi_offset`) to avoid the "funnel of hell" pathology that plagues centered hierarchical models. This is standard practice for hierarchical Bayesian models.

5. **Ordering constraint for identification.** Instead of hard anchors (as in the flat IRT), party means are constrained via `pt.sort(mu_party_raw)` so that the lower-indexed party (Democrat, index 0) always has the lower mean ideal point. This is cleaner than `pm.Potential` with `-inf` penalties and avoids label switching.

6. **Flat IRT ideal points are available.** Shrinkage comparison requires the flat IRT phase to have been run first. The hierarchical phase reads from `irt/latest/data/ideal_points_{chamber}.parquet`.

## Parameters & Constants

| Constant | Value | Justification |
|----------|-------|---------------|
| `HIER_N_SAMPLES` | 2000 | Same as flat IRT; sufficient for posterior summaries |
| `HIER_N_TUNE` | 1500 | Higher than flat's 1000 — hierarchical funnels need more warmup |
| `HIER_N_CHAINS` | 2 | Same as flat IRT |
| `HIER_TARGET_ACCEPT` | 0.95 | Higher than flat's 0.9 — reduces divergences in hierarchical geometry |
| `RHAT_THRESHOLD` | 1.01 | Reused from flat IRT |
| `ESS_THRESHOLD` | 400 | Reused from flat IRT |
| `MAX_DIVERGENCES` | 10 | Reused from flat IRT |
| `RANDOM_SEED` | 42 | Reused from flat IRT |
| `PARTY_COLORS` | R=#E81B23, D=#0015BC | Consistent across all phases |

## Methodological Choices

### Ordering constraint vs hard anchors

**Decision:** Use `pt.sort` on party means instead of fixing individual legislator ideal points.

**Rationale:** Hard anchors (as in flat IRT) fix two specific legislators to xi=+1 and xi=-1. This works well for identification but breaks the partial pooling story — anchored legislators can't be shrunk toward their party mean. The ordering constraint (`mu_party[D] < mu_party[R]`) provides sign identification without constraining any individual. During warmup, `pt.sort` is continuous and differentiable, avoiding the gradient discontinuities of `pm.Potential(..., -inf)`.

**Trade-off:** Without anchors, the scale is determined only by the prior on `sigma_within` and the data. This is fine because we compare relative positions (ranks, party separation) rather than absolute scale values.

### Non-centered parameterization

**Decision:** Model `xi_offset ~ Normal(0, 1)` with `xi = mu_party[party_idx] + sigma_within[party_idx] * xi_offset`.

**Rationale:** Centered parameterization (`xi ~ Normal(mu_party, sigma_within)`) creates a "funnel" in the posterior geometry when `sigma_within` is small — the sampler struggles to explore. Non-centered parameterization decouples the individual parameters from the group-level parameters, giving NUTS a smoother geometry. This is the standard recommendation for hierarchical models (Betancourt 2017, Stan manual).

**Trade-off:** Non-centered is less efficient when groups are large and `sigma_within` is well-identified. For Kansas (40-90 R, 10-38 D per chamber), the groups are large enough that centered *might* work, but non-centered is safer and the computational penalty is minimal.

### Per-chamber primary, joint secondary

**Decision:** Run separate 2-level models for House and Senate. Optionally run a 3-level joint model with `--skip-joint` flag.

**Rationale:** Chambers vote on different bills and have different compositions. The flat IRT already established that per-chamber analysis is the norm. The joint model (chamber → party → legislator) is theoretically interesting (common ideological scale across chambers) but practically difficult: only ~71 shared bills link the chambers, and bridging legislators are rare. Per-chamber models are the trustworthy primary output.

**Risk:** The joint model may produce divergences or poor ESS due to the sparse link between chambers. The `--skip-joint` flag allows bypassing it without losing the primary results.

**Identification:** The joint model uses the same `pt.sort` ordering constraint within each chamber's pair of party offsets (House D < House R, Senate D < Senate R) to prevent sign flips. Without this, the sampler could place Senate Democrats above Senate Republicans while correctly ordering the House — a label-switching pathology observed in the 90th biennium audit.

### Variance decomposition via ICC

**Decision:** Compute the intraclass correlation coefficient (ICC) as `sigma_between² / (sigma_between² + sigma_within²)` where `sigma_between` is derived from the variance of party means and `sigma_within` is the pooled within-party standard deviation (weighted by group size). Computed per posterior draw (vectorized with numpy) to propagate uncertainty.

**Rationale:** ICC directly answers "what fraction of ideological variance is explained by party?" This is the most interpretable summary of the hierarchical structure. Values near 1 mean party explains almost everything; near 0 means legislators vary independently of party.

**Expected range:** 0.6-0.8 for Kansas (strong party structure but meaningful within-party variation). Observed: ~0.90 for both chambers in the 91st session.

### Shrinkage comparison with flat IRT

**Decision:** Compare hierarchical ideal points to flat IRT ideal points by computing delta and percent shrinkage toward party mean. Because the two models produce ideal points on different scales (flat ≈ [-4, 3], hierarchical ≈ [-11, 9]) due to different identification constraints (hard anchors vs ordering constraint), flat estimates are rescaled to the hierarchical scale via linear regression (`np.polyfit`) before computing deltas.

**Rationale:** The whole point of the hierarchical model is partial pooling. Showing *which* legislators moved and *how much* they moved toward their party mean is the primary deliverable. Legislators with sparse records should show the most shrinkage. Shrinkage percent is computed as `(1 - hier_dist/flat_dist) * 100` where distances are measured to the party mean; positive values indicate the estimate moved closer to the party mean.

**Edge cases:**
- Anchored legislators in the flat IRT (fixed at xi=±1) have xi_sd≈0, which makes percent shrinkage undefined. These are excluded (`flat_xi_sd > 0.01` guard).
- Legislators near the party mean have unstable shrinkage ratios. These are excluded (`flat_dist > 0.5` guard).

## Downstream Implications

- **Synthesis** should join `hier_xi_mean`, `hier_xi_sd`, and `shrinkage_pct` from the hierarchical ideal points parquet. Column names are prefixed with `hier_` to avoid collision with the flat IRT's `xi_mean`/`xi_sd`.
- **Profiles** can use hierarchical ideal points as a more robust alternative to flat IRT, especially for low-participation legislators.
- **Cross-session comparison** (future): hierarchical ideal points are more comparable across sessions because partial pooling reduces noise in sparse records.
- **ICC** provides a single-number summary of party polarization that can be tracked over time.
- The hierarchical model **supersedes** the Beta-Binomial for formal party loyalty analysis, but Beta-Binomial remains useful as an instant exploratory baseline (no MCMC required).
