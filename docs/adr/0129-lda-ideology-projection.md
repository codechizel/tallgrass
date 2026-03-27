# ADR-0129: Fisher's LDA Ideology Projection in Phase 02

**Date:** 2026-03-26
**Status:** Accepted

## Context

PCA extracts components in order of variance explained, not substantive meaning. In Kansas Senate supermajority sessions, the largest source of variance can be intra-Republican factionalism (moderate vs. conservative) rather than the party divide, causing PC1 to capture the wrong axis. This affected 4-7 of 14 Senate sessions depending on the contested-vote threshold.

The pipeline managed this with three stacked mechanisms:
1. Manual PCA overrides (`pca_overrides.yaml`) — 4-8 hardcoded session/chamber entries
2. Automated party-correlation detection (`detect_ideology_pc()`) — point-biserial correlation swap
3. Canonical routing Layer 1 — override-driven fallback to 2D IRT Dim 2

Each mechanism was a patch on the previous one's failure mode (see ADR-0118, ADR-0123, ADR-0127). The architecture was fragile, required re-auditing whenever `CONTESTED_THRESHOLD` changed, and could only select one PC at a time — missing party-relevant signal spread across multiple components.

### Empirical finding

Fisher's Linear Discriminant Analysis on PCA scores (PC1-PC5) finds the optimal party-separating direction automatically. Across all 28 chamber-sessions:
- Mean Cohen's d improved from 4.97 (best single PC) to 9.36 (LDA)
- The "unsolvable" 84th Senate improved from d=1.89 to d=5.03
- Even clean sessions (91st) improved by 20% due to party signal in PC3+
- LOO cross-validated accuracy: 97.3-100% (no overfitting)

## Decision

Add Fisher's LDA as a post-PCA computation inside Phase 02. Use shrinkage LDA (Ledoit-Wolf) for stability with small Democrat groups (n ~ 8-12). Produce two new columns in the PCA scores parquet:

- `ideology_score`: projection onto the optimal party-separating direction
- `establishment_score`: first principal component of the orthogonal complement (captures intra-party factionalism)

Downstream phases (`init_strategy.py`, Phase 05, Phase 06) prefer `ideology_score` when available, falling back to the old PC1/override/detection logic for backward compatibility with pre-LDA pipeline results.

Remove canonical routing Layer 1 (manual PCA override). Keep Layers 2-3 (horseshoe detection, tiered convergence) as safety nets.

Deprecate `pca_overrides.yaml` — no longer read by the active pipeline, retained for historical reference.

### Why shrinkage LDA

Classical LDA requires inverting the within-class covariance matrix `S_w`. With 5 PCA features and ~10 Democrats, `S_w` is poorly conditioned. Ledoit-Wolf shrinkage (Friedman 1989, Ledoit & Wolf 2004) pulls `S_w` toward the identity: `S_shrunk = alpha * I + (1-alpha) * S_sample`, where `alpha` is determined analytically. scikit-learn implements this as `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')`.

Guard: LDA is skipped entirely when either party has fewer than 10 members.

### Why not replace PCA entirely

LDA is supervised — it uses party labels to define the ideology dimension. This means:
1. It cannot discover when the factional axis is more important than the party axis
2. It compresses within-party variation by construction
3. It provides no information about dimensionality

PCA remains the unsupervised discovery step. LDA is a party-oriented post-processing step for downstream IRT initialization and routing.

## Consequences

- PCA overrides (`pca_overrides.yaml`) deprecated — LDA subsumes manual intervention
- `detect_ideology_pc()` superseded — LDA subsumes automated detection
- Canonical routing simplified from 3 layers to 2
- IRT initialization improved for every session (especially problematic ones)
- Report gains 4 new sections: plain-language LDA explanation, ideology-vs-establishment scatter, weight table, comparison card
- Establishment score available as bonus output for intra-party analysis

### Limitations

- Circularity: LDA defines party as the primary organizing dimension by construction. It cannot discover that factionalism exceeds partisanship.
- Small-sample instability: with ~10 Democrats, the LDA direction has non-trivial variance. Shrinkage mitigates but does not eliminate this.
- Party is not ideology: the score measures the direction that best separates party labels, not "ideology" per se.

### References

- Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.
- Friedman, J. H. (1989). Regularized discriminant analysis. *JASA*, 84(405).
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices.
- Aldrich, Montgomery & Sparks (2014). Polarization and ideology: partisan sources of low dimensionality. *Political Analysis*.
- See `docs/lda-ideology-projection.md` for the full analysis.
