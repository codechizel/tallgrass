# Pipeline Ordering Analysis

**Date:** 2026-03-14
**Status:** Actionable — reordering planned

## Summary

A dependency audit of the 27-phase analysis pipeline identified three ordering problems and two missed prior opportunities. The core Bayesian prior chain (PCA → 1D IRT → 2D IRT → Hierarchical → canonical routing) is correctly ordered with no circular dependencies. The problems are in the positioning of non-Bayesian phases relative to phases that could consume their output.

## Methodology

Every phase was traced for:

1. **Upstream data loads** — calls to `resolve_upstream_dir()`, explicit parquet reads
2. **Prior specifications** — Bayesian priors, initialization strategies, anchor selection
3. **Downstream consumers** — which later phases read this phase's output
4. **Minimum required inputs** — what a phase truly needs vs. what it optionally enriches with

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

## Dependency Graph

```
Raw CSVs ──→ 01 EDA ──→ 02 PCA ──→ 05 IRT ──→ 06 2D IRT ──→ 07 Hierarchical
                │            │          │            │               │
                │            │          │            └─── canonical ─┘
                │            │          │                    │
                │            │          ├──→ 08 PPC ←───────┘
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
                └────── (should feed) ──→ 12 Bipartite            │
                └────── (should feed) ────────────────────────────┘
                                                                   │
                ┌──────────────────────────────────────────────────┘
                ↓
             24 Synthesis ──→ 25 Profiles

Cross-biennium:
Multiple 05 IRT outputs ──→ 26 Cross-Session ──→ 27 Dynamic IRT
```

Dashed connections (`should feed`) represent the ordering problems — data exists but isn't available at runtime because the producer runs after the consumer.

## Validation-Only Phases (No Downstream Data Dependencies)

These phases produce reports but no data consumed by later phases. Their position matters only for their own upstream availability:

- 03 MCA (validates PCA via categorical encoding)
- 04 UMAP (visualization; validates against PCA and optionally IRT)
- 10 LCA (validates clustering via Bernoulli mixture)
- 16 W-NOMINATE (field-standard comparison)
- 17 External Validation (Shor-McCarty correlation)
- 18 DIME (CFscore correlation)
- 19 TSA (drift detection; could inform Phase 27 but currently doesn't)

## Recommended Reorder

Move `text-analysis` to position 4 (after MCA, before IRT). Move `umap` to position 9 (after PPC, before Clustering). No phase renumbering needed — directory numbers are organizational, the Justfile controls execution order.

```
01 EDA → 02 PCA → 03 MCA → 20 Bill Text → 05 IRT → 06 2D IRT
→ 07 Hierarchical → 08 PPC → 04 UMAP → 09 Clustering → 10 LCA
→ 11 Network → 12 Bipartite → 13 Indices → 14 Beta-Binomial
→ 15 Prediction → 16 W-NOMINATE → 17 External Validation → 18 DIME
→ 19 TSA → 21 TBIP → 22 Issue IRT → 23 Model Legislation
→ 24 Synthesis → 25 Profiles
```

**Two moves, zero breakage:**
1. `text-analysis`: position 20 → 4 — unlocks BERTopic topics + embeddings for all downstream phases
2. `umap`: position 4 → 9 — enables Procrustes validation against IRT and canonical ideal points
