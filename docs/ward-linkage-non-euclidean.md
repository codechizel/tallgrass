# Ward Linkage on Non-Euclidean Distances

An analysis of why Ward linkage with Kappa-derived distances is methodologically impure, why it works in practice for our data, and the fix we applied.

**Date:** 2026-02-24

---

## The Problem

Ward's method minimizes the total within-cluster variance at each merge step. Its update formula (Lance-Williams) assumes the input distances satisfy the Euclidean metric — specifically, that they embed isometrically in Euclidean space. The [scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) explicitly warns:

> *Methods 'centroid', 'median', and 'ward' are correctly defined only if Euclidean pairwise metric is used.*

Our hierarchical clustering used Ward linkage on a Kappa-derived distance matrix: `distance = 1 - Cohen's Kappa`. Cohen's Kappa corrects for chance agreement, making it an excellent similarity measure for binary voting data with Kansas's 82% Yea base rate. But 1 - Kappa is not a Euclidean distance. It may not even satisfy the triangle inequality in all cases, because Kappa can produce values where `d(A,C) > d(A,B) + d(B,C)` when the marginal distributions of voters A, B, and C differ enough.

### What Can Go Wrong

When Ward is applied to non-Euclidean distances, several problems can arise:

1. **Negative branch heights.** The Lance-Williams update formula can produce negative merge distances when the triangle inequality is violated. This makes the dendrogram uninterpretable — a merge at height -0.02 has no meaning.

2. **Distorted merge ordering.** Ward assumes that merging two clusters increases the total sum-of-squares. With non-Euclidean distances, merges can actually *decrease* the objective, leading to inversions where a child node appears at a higher height than its parent.

3. **Invalid cophenetic interpretation.** The cophenetic correlation measures how faithfully the dendrogram preserves the original distances. With Ward on non-Euclidean input, even a high cophenetic correlation doesn't guarantee the dendrogram is meaningful — the validation metric shares the same flawed assumptions as the linkage method.

4. **Silhouette scores remain valid.** The silhouette computation uses the original precomputed distance matrix (`metric="precomputed"`), not the Ward-derived dendrogram distances. This means our silhouette-based model selection was correct even when the linkage method was impure.

### Why It Worked Anyway

For our data, this issue never produced visible artifacts for three reasons:

1. **The Kappa distance matrix is well-behaved.** For Kansas legislators, the Kappa distances are approximately Euclidean — the matrix is positive semi-definite after symmetrization, and triangle inequality violations are rare and small.

2. **The k=2 structure is overwhelming.** The party split creates such strong separation (silhouette ~0.75) that minor metric distortions don't change the optimal partition. Ward, average, and complete linkage all recover the same two clusters.

3. **Cross-method validation caught any divergence.** With ARI > 0.93 between hierarchical, K-Means, and GMM, any dendrogram distortion that changed cluster assignments would have surfaced as an ARI drop.

## The Fix

We switched the hierarchical linkage method from Ward to **average linkage** for the Kappa distance matrix. Average linkage:

- Is valid with any distance metric, including precomputed non-Euclidean distances
- Defines the distance between clusters as the mean of all pairwise distances between members — intuitive and well-defined regardless of metric properties
- Performs well when the number of clusters is potentially overspecified (Hands & Everitt), which applies here since we evaluate k=2 through k=7
- Is the standard choice in the political science literature for agreement-based hierarchical clustering of legislative data

The constant `LINKAGE_METHOD` was changed from `"ward"` to `"average"`, and the design doc was updated to document the rationale.

### Why Not Complete Linkage?

Complete linkage (max distance between cluster members) is also valid with non-Euclidean metrics and produces more compact clusters than average linkage. However, it tends to be sensitive to outliers — a single extreme legislator can prevent two otherwise-similar clusters from merging. Given that our data includes known outliers (Miller with 0.90 HDI, Hill with 2.03 HDI), average linkage is more robust.

### Why Not Keep Ward on PCA Scores?

An alternative approach would be to add a second hierarchical clustering run using Ward linkage on PCA scores (where Euclidean distance is valid). This would provide a methodologically pure Ward comparison point. We chose not to do this because:

- The PCA scores are already used as input to other methods (K-Means, HDBSCAN)
- Adding a second hierarchical method increases complexity without clear analytical gain
- Average linkage on Kappa distances is a single, well-justified method

## Impact

In practice, the switch from Ward to average linkage does not change the k=2 finding or the cross-method ARI scores. The cophenetic correlation may change slightly (average linkage typically produces higher cophenetic correlations than Ward on the same data). The dendrogram structure is preserved at the macro level — Republicans and Democrats still form two clear clusters — but fine-grained merge ordering within parties may differ.

## References

- [scipy.cluster.hierarchy.linkage documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
- Hands, S. & Everitt, B. (1987). A Monte Carlo study of the recovery of cluster structure in binary data by hierarchical clustering techniques. *Multivariate Behavioral Research*, 22(2), 235-243.
- Murtagh, F. & Legendre, P. (2014). Ward's hierarchical agglomerative clustering method: which algorithms implement Ward's criterion? *Journal of Classification*, 31, 274-295.
