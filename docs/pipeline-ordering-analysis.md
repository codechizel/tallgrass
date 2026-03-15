# Pipeline Ordering Analysis

**Date:** 2026-03-14
**Status:** Actionable — reordering + new phase planned

## Summary

A dependency audit of the 27-phase analysis pipeline identified three ordering problems, two missed prior opportunities, and one new phase worth adding: **Hierarchical 2D IRT**. The core Bayesian prior chain (PCA → 1D IRT → 2D IRT → Hierarchical → canonical routing) is correctly ordered with no circular dependencies. The problems are in the positioning of non-Bayesian phases relative to phases that could consume their output, and in the absence of a model that combines the 2D and hierarchical structures.

## Methodology

Every phase was traced for:

1. **Upstream data loads** — calls to `resolve_upstream_dir()`, explicit parquet reads
2. **Prior specifications** — Bayesian priors, initialization strategies, anchor selection
3. **Downstream consumers** — which later phases read this phase's output
4. **Minimum required inputs** — what a phase truly needs vs. what it optionally enriches with

A literature survey (March 2026) identified relevant recent work:

- Lu & Wang 2011 (JRSS-C): Hierarchical ideal point estimation for multidimensional legislative behavior with party group structure
- Li, Gibbons & Ročková 2025 (JASA): Sparse Bayesian multidimensional IRT — identification theory for hierarchical MIRT
- Shin 2024: Issue-specific ideal points with partial pooling across dimensions
- Bucchianeri, Volden & Wiseman 2025 (APSR): Legislative Effectiveness Scores for state legislatures
- Wang et al. 2025 (J. Social Computing): Attention-based bill similarity for legislative prediction (+6.69% accuracy)
- Nature Scientific Data 2025: Transformer-based CAP coding of 1.36M state bills

## Current Execution Order

```
01 EDA → 02 PCA → 03 MCA → 04 UMAP → 05 IRT → 06 2D IRT → 07 Hierarchical
→ 08 PPC → 09 Clustering → 10 LCA → 11 Network → 12 Bipartite → 13 Indices
→ 14 Beta-Binomial → 15 Prediction → 16 W-NOMINATE → 17 External Validation
→ 18 DIME → 19 TSA → 20 Bill Text → 21 TBIP → 22 Issue IRT
→ 23 Model Legislation → 24 Synthesis → 25 Profiles
```

Cross-biennium: `26 Cross-Session → 27 Dynamic IRT`

## Core Prior Chain (Correctly Ordered)

The Bayesian information flow is sound:

| Flow | Mechanism | Status |
|------|-----------|--------|
| 01 EDA → 02 PCA → 05 IRT | PCA PC1 for anchor selection + chain init | Correct |
| 02 PCA → 06 2D IRT | Mandatory `pca-informed` init (avoids horseshoe contamination) | Correct |
| 05 IRT → 06 2D IRT | Horseshoe detection from 1D ideal point distributions | Correct |
| 06 2D IRT → 07 Hierarchical | Canonical routing output; optional dim1 soft prior (ADR-0108) | Correct |
| 05/06/07 → 08 PPC | LOO-CV model comparison on all three fitted posteriors | Correct |
| 13 Indices → 14 Beta-Binomial | Party unity scores feed empirical Bayes priors | Correct |
| 05 IRT → 27 Dynamic IRT | Informative `xi_init ~ Normal(xi_static, 1.5)` prior | Correct |
| 20 Bill Text → 21 TBIP / 22 Issue IRT / 23 Model Legislation | Topic assignments and embeddings | Correct |

No circular dependencies exist anywhere in the pipeline. All information flows forward.

## Ordering Problems

### Problem 1: Phase 20 (Bill Text) runs too late — starves Phases 12 and 15

**Severity: High**

Phase 20 (Bill Text Analysis) depends only on raw CSVs (`bill_texts.csv`). It has zero upstream phase dependencies. Yet it runs at position 20, after phases that want its output:

- **Phase 15 (Prediction)** contains a built-in `nlp_features.py` module that fits TF-IDF + NMF on bill `short_title` as a workaround. This is a 6-topic NMF model — strictly inferior to Phase 20's BERTopic (FastEmbed + HDBSCAN + c-TF-IDF) with 384-dimensional embeddings. The richer features are unavailable because Phase 20 hasn't run yet.

- **Phase 12 (Bipartite)** can optionally consume bill text data for enriched bill community analysis. It runs 8 positions before Phase 20.

**Impact:** Prediction uses degraded NLP features. Bipartite analysis misses text enrichment entirely during pipeline runs.

### Problem 2: Phase 04 (UMAP) runs before Phase 05 (IRT) — can't validate against IRT

**Severity: Medium**

Phase 04 explicitly loads IRT ideal points via `resolve_upstream_dir("05_irt", ...)` for Procrustes validation of UMAP embeddings against IRT. But IRT runs at position 05, one step after UMAP at position 04. During a pipeline run, UMAP can only validate against PCA.

UMAP is visualization-only with zero downstream consumers, so moving it later in the pipeline breaks nothing.

### Problem 3: Phase 13/14 (Indices/Beta-Binomial) could theoretically run earlier

**Severity: Low**

Phase 13 computes Rice Index, party unity, and maverick scores from raw votes. The IRT dependency is for comparison columns only, not for the core index computations. Phase 14's empirical Bayes similarly needs only raw party-line vote counts. Both could run immediately after EDA.

In practice this doesn't matter — no upstream phase needs their output before position 13. The first consumer is Phase 15 (Prediction), which is correctly ordered after both.

## Missed Prior Opportunities

### A. Phase 22 (Issue IRT): weakly informative priors despite available posteriors

Phase 22 fits independent 1D IRT per topic with `xi ~ Normal(0, 1)` priors. It already loads Phase 05's full-model ideal points for anchor selection and sign alignment. It could instead use them as **informative priors**: `xi_topic ~ Normal(xi_full_model, sigma)`.

**Rationale:** Topic subsets have 10–30 bills. Weakly informative priors produce wide, unreliable posteriors. An informative prior from the full model would:

- Improve convergence on sparse topic subsets
- Shrink poorly-identified topic scores toward the full-model estimate (sensible default)
- Make deviations interpretable: posterior pushback against the prior signals genuine topic-specific divergence

This is methodologically equivalent to partial pooling across topics — the same principle that makes hierarchical models work.

### B. Phase 19 (TSA) drift → Phase 27 (Dynamic IRT) evolution_sd

TSA estimates within-session ideological drift via rolling PCA. Dynamic IRT uses a fixed `HalfNormal(0.15)` or `HalfNormal(0.5)` for evolution_sd based on chamber size. If TSA detects high intra-session drift, Dynamic IRT's evolution prior could be wider. The ordering supports this flow (19 runs in single-biennium, 27 in cross-biennium), but the information doesn't currently transfer.

### C. Phase 08 (PPC) Q3 → canonical routing

Yen's Q3 statistic detects local dependence (unmodeled dimensions). High Q3 is a more principled signal that 2D IRT is needed than the current heuristic horseshoe detection (wrong-side fraction, party overlap). Currently PPC runs after 2D IRT is already fitted, so it validates rather than informs. Using Q3 to drive canonical routing would require an architectural change: fit 1D → PPC on 1D → decide 2D necessity → conditionally fit 2D.

## New Phase: Hierarchical 2D IRT

### Motivation

The pipeline currently has two complementary IRT refinements over the flat 1D baseline:

- **Phase 06 (Flat 2D IRT):** Resolves the horseshoe effect by adding a second dimension (ideology + establishment-contrarian). PLT identification. But suffers from convergence problems on Dim 2 (weak signal, ~11% variance, high R-hat, low ESS).

- **Phase 07 (Hierarchical 1D IRT):** Adds party-level partial pooling for shrinkage. Non-centered parameterization. Excellent convergence. But single-dimensional — can't resolve horseshoe.

Neither model has what the other needs. A **Hierarchical 2D IRT** combines both: party-level pooling on a 2D ideal point space.

### Literature Support

Lu & Wang (2011, JRSS-C) proposed hierarchical ideal point estimation specifically for multidimensional legislative analysis with party group structure — the exact use case. Their framework demonstrated that party-group hierarchy improves both identification and convergence for multi-dimensional ideal points.

Li, Gibbons & Ročková (2025, JASA) provide modern identification theory for sparse Bayesian multidimensional IRT, establishing that sparse factor loadings with hierarchical structure achieve identifiability without the rotational indeterminacy that plagues flat MIRT models. Their PLT + sparsity approach is compatible with our existing PLT identification.

The non-centered parameterization literature (Betancourt & Girolami 2015) confirms that hierarchical models with thin data — exactly the situation on Dim 2 — benefit most from non-centered parameterization. Our existing Phase 07 already uses non-centering.

### Model Structure

```
Party-level parameters (per dimension):
    mu_party_dim1[p] ~ Normal(mu_07_party[p], 1.0)   # informative from Phase 07
    mu_party_dim2[p] ~ Normal(dim2_party_avg[p], 2.0) # soft prior from Phase 06
    sigma_party_dim1[p] ~ HalfNormal(sigma_scale)      # adaptive for small groups
    sigma_party_dim2[p] ~ HalfNormal(sigma_scale)      # adaptive for small groups

Legislator ideal points (non-centered, per dimension):
    xi_offset_dim1[i] ~ Normal(0, 1)
    xi_offset_dim2[i] ~ Normal(0, 1)
    xi_dim1[i] = mu_party_dim1[party[i]] + sigma_party_dim1[party[i]] * xi_offset_dim1[i]
    xi_dim2[i] = mu_party_dim2[party[i]] + sigma_party_dim2[party[i]] * xi_offset_dim2[i]

Bill parameters (PLT identification, same as Phase 06):
    alpha[j] ~ Normal(0, 5)
    beta: PLT-constrained (n_votes, 2)

Likelihood:
    eta[i,j] = sum_d(beta[j,d] * xi[i,d]) - alpha[j]
    y[i,j] ~ Bernoulli(logit(eta[i,j]))
```

### Prior Chain — Three Upstream Sources

This is the key innovation: the Hierarchical 2D model consumes informative priors from **three** upstream phases, each contributing something the others cannot:

```
Phase 02 (PCA)
    └─→ PC1/PC2 for chain initialization (init strategy: pca-informed)
        Provides starting values for MCMC chains — same role as in Phase 06.

Phase 06 (Flat 2D IRT)
    └─→ Party-level averages on Dim 2 → soft prior on mu_party_dim2
        Flat 2D gives us a first estimate of where parties sit on the
        establishment-contrarian axis. Wide sigma (2.0) lets the
        hierarchical model refine this.

Phase 07 (Hierarchical 1D IRT)
    └─→ mu_party posteriors → informative hyperprior on mu_party_dim1
    └─→ sigma_party posteriors → informative scale for sigma_party_dim1
        Phase 07 has already learned the party structure on the ideology
        dimension. We transfer this knowledge as an informative prior
        rather than re-discovering it from scratch.
```

This is methodologically sound: Phase 07's posterior becomes the Hierarchical 2D's prior. This is the standard "posterior-as-prior" sequential Bayesian update. The information flows strictly forward (no circular dependency).

### Why Party Pooling Fixes Dim 2 Convergence

Phase 06's main problem is Dim 2 convergence. With ~165 legislators and ~11% variance signal, the flat model struggles to identify all legislator positions on Dim 2 independently.

Party-level pooling on Dim 2 reduces effective dimensionality dramatically. Instead of estimating 165 independent Dim 2 positions, we estimate ~2–3 party means + within-party deviations. For the 72% Republican supermajority in Kansas, the Republican party mean on Dim 2 determines the centroid of the supermajority on the establishment-contrarian axis, and individual Rs are shrunk toward it. This is exactly the regularization Dim 2 needs.

**Expected improvement:** Dim 2 R-hat should drop below 1.10 (from ~1.05 in flat 2D) because:
- Fewer effective parameters (party means + offsets vs. independent positions)
- Non-centered parameterization avoids the funnel geometry
- Informative hyperpriors on Dim 1 stabilize the model, letting the sampler focus on Dim 2

### Identification Strategy

Triple identification:
1. **PLT on beta** (rotational identification) — same as Phase 06
2. **Sorted party means on Dim 1** (sign identification) — D < R, same as Phase 07
3. **Hierarchical shrinkage** (regularization) — partial pooling prevents extreme estimates

The combination is stronger than any one alone. PLT handles rotation, sorted means handle sign, and shrinkage handles the weak-signal regime on Dim 2.

### Impact on Canonical Routing

The canonical routing (currently in `canonical_ideal_points.py`) would be extended:

```
For each chamber:
    1. Detect horseshoe from Phase 05 (1D IRT) — existing logic
    2. If horseshoe detected:
        a. Prefer Hierarchical 2D Dim 1 (if converged at Tier 1/2)
        b. Fall back to Flat 2D Dim 1 (if Hierarchical 2D failed but Flat 2D converged)
        c. Fall back to 1D IRT (if both 2D models failed)
    3. If no horseshoe:
        a. Use 1D IRT (balanced chambers don't need 2D)
```

Hierarchical 2D is preferred over Flat 2D when both converge because:
- Better-calibrated uncertainty (partial pooling produces more honest credible intervals)
- More interpretable (party structure on both dimensions)
- Likely better convergence (regularization)

### Impact on PPC (Phase 08)

PPC would evaluate four models instead of three:
1. Flat 1D IRT (Phase 05)
2. Flat 2D IRT (Phase 06)
3. Hierarchical 1D IRT (Phase 07)
4. **Hierarchical 2D IRT (new phase)**

LOO-CV model comparison across all four provides a principled model selection criterion.

### Position in Pipeline

After Phase 07 (Hierarchical 1D), before Phase 08 (PPC):

```
... → 06 2D IRT → 07 Hierarchical 1D → [NEW] Hierarchical 2D → 08 PPC → ...
```

This is the only correct position because:
- Needs Phase 06 output (flat 2D ideal points for Dim 2 party averages)
- Needs Phase 07 output (party-level mu/sigma posteriors for informative hyperpriors)
- PPC must run after all IRT models are fitted

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Over-parameterization on Dim 2 when signal is weak | Informative prior from Phase 06; adaptive sigma for small groups; graceful skip if < 3 parties |
| Non-centered 2D hierarchical + PLT is a complex model | nutpie Rust NUTS handles high-dimensional models; proven in Phase 07 (1D hierarchical) |
| Canonical routing complexity (4 models instead of 3) | Clear tier system; Hierarchical 2D preferred only when converged; existing tiered quality gate applies |
| Runtime cost on Apple Silicon | Supermajority tuning (ADR-0112) already handles this; expect ~15–25 min/chamber based on Phase 06 + Phase 07 timings |

## Dependency Graph (Updated)

```
Raw CSVs ──→ 01 EDA ──→ 02 PCA ──→ 05 IRT ──→ 06 2D IRT ──→ 07 Hierarchical 1D
                │            │          │            │               │
                │            │          │            │    ┌──────────┘
                │            │          │            │    ↓
                │            │          │            └→ [NEW] Hierarchical 2D
                │            │          │                    │
                │            │          │            ┌───────┘
                │            │          │            ↓
                │            │          ├──→ 08 PPC (validates all 4 models)
                │            │          │
                │            │          ├──→ 09 Clustering ──→ 11 Network
                │            │          │                          │
                │            │          ├──→ 13 Indices ──→ 14 Beta-Binom
                │            │          │         │              │
                │            │          └────┬────┴──────────→ 15 Prediction
                │            │               │                     │
                │            └───────────────┘                     │
                │                                                  │
Raw CSVs ──→ 20 Bill Text ──→ 21 TBIP                            │
                │           ──→ 22 Issue IRT                      │
                │           ──→ 23 Model Legislation              │
                │                                                  │
                └──────────────────→ 12 Bipartite                 │
                └─────────────────────────────────────────────────┘
                                                                   │
                ┌──────────────────────────────────────────────────┘
                ↓
             24 Synthesis ──→ 25 Profiles

Cross-biennium:
Multiple IRT outputs ──→ 26 Cross-Session ──→ 27 Dynamic IRT
```

## Validation-Only Phases (No Downstream Data Dependencies)

These phases produce reports but no data consumed by later phases. Their position matters only for their own upstream availability:

- 03 MCA (validates PCA via categorical encoding)
- 04 UMAP (visualization; validates against PCA and optionally IRT)
- 10 LCA (validates clustering via Bernoulli mixture)
- 16 W-NOMINATE (field-standard comparison)
- 17 External Validation (Shor-McCarty correlation)
- 18 DIME (CFscore correlation)
- 19 TSA (drift detection; could inform Phase 27 but currently doesn't)

## Future Feature Compatibility

A literature survey identified potential future additions. All fit naturally into the proposed reorder:

| Feature | Depends On | Natural Position | Benefits from Reorder? |
|---------|-----------|------------------|----------------------|
| Legislative Effectiveness Scores (Volden-Wiseman) | Bill Text CAP categories | After Indices (13) | Yes — needs Phase 20 early for bill categorization |
| Interest Group Scorecard Alignment | External data + IRT | Alongside Ext. Validation (17/18) | No — independent |
| Cosponsorship Network | Raw CSVs (sponsor_slugs) | Alongside Network (11) | No — independent |
| Attention-Based Bill Similarity | Bill embeddings + votes | Enhancement to Prediction (15) | Yes — needs Phase 20 early for embeddings |
| Gridlock / Productivity Metrics | bill_actions | Alongside Indices (13) | No — independent |
| Temporal Community Detection | Multi-biennium networks | Cross-biennium pipeline (after 26) | No — independent |
| Bill Passage Survival Analysis | bill_actions + timestamps | After Prediction (15) | Slight — benefits from topic features |

No future feature benefits from Bill Text staying at position 20. Every text-dependent feature is better served by the proposed reorder.

## Recommended Pipeline Order (Final)

Three changes from current:
1. `text-analysis`: position 20 → 4 — unlocks BERTopic topics + embeddings for all downstream phases
2. `umap`: position 4 → 9 — enables Procrustes validation against IRT and canonical ideal points
3. `hierarchical-2d` (new): insert after `hierarchical` — party-pooled 2D IRT with informative priors

```
01 EDA → 02 PCA → 03 MCA → 20 Bill Text → 05 IRT → 06 2D IRT
→ 07 Hierarchical 1D → [NEW] Hierarchical 2D → 08 PPC → 04 UMAP
→ 09 Clustering → 10 LCA → 11 Network → 12 Bipartite → 13 Indices
→ 14 Beta-Binomial → 15 Prediction → 16 W-NOMINATE → 17 External Validation
→ 18 DIME → 19 TSA → 21 TBIP → 22 Issue IRT → 23 Model Legislation
→ 24 Synthesis → 25 Profiles
```

Cross-biennium (unchanged): `26 Cross-Session → 27 Dynamic IRT`

## Implementation Plan

### Step 1: Reorder Justfile (no code changes)

Move `text-analysis` and `umap` in the pipeline recipe. Zero code changes required — only the execution order in the shell script changes.

### Step 2: Create Hierarchical 2D IRT phase

New directory: `analysis/XX_hierarchical_2d/` (number TBD based on convention).

**Model builder:** Combine Phase 06's PLT-constrained 2D structure with Phase 07's non-centered hierarchical parameterization. Key additions:
- Per-dimension party means and sigma
- Informative hyperpriors from Phase 07 (mu_party, sigma_party) on Dim 1
- Soft priors from Phase 06 (flat 2D party averages) on Dim 2
- Same PLT identification as Phase 06

**Data loading:** Standard `resolve_upstream_dir()` for Phases 01, 02, 06, and 07.

**Canonical routing update:** Extend `canonical_ideal_points.py` to prefer Hierarchical 2D Dim 1 over Flat 2D Dim 1 for horseshoe-affected chambers when both converge.

**PPC update:** Phase 08 loads and evaluates the new model's idata alongside the existing three.

**Report:** HTML report with party-level Dim 1 and Dim 2 posteriors, shrinkage comparison vs. flat 2D, convergence diagnostics, 2D scatter with party structure.

### Step 3: Update Synthesis (Phase 24)

Add Hierarchical 2D to the upstream phase list in `synthesis_data.py`. Read canonical ideal points from the new routing (which may now come from Hierarchical 2D).

### Step 4: Documentation

- ADR for the Hierarchical 2D IRT model (priors, identification, prior chain)
- ADR for pipeline reordering (rationale, dependency audit)
- Update CLAUDE.md pipeline section
- Update `analysis-framework.md` with new phase
- Design doc: `analysis/design/hierarchical_2d.md`
