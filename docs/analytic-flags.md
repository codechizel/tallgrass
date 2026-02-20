# Analytic Flags

Observations, outliers, and data points flagged during quantitative analysis that warrant qualitative investigation or special handling in downstream phases. Each entry records **what** was observed, **where** (which analysis phase), **why** it matters, and **what to do about it**.

This is a living document — add entries as each analysis phase surfaces new findings.

## Flagged Legislators

### Sen. Caryn Tyson (R, District 12)

- **Phase:** PCA, IRT
- **Observation:** Extreme PC2 outlier at -24.8 (3x the next senator). PCA PC1 rank: 23rd of 32 Rs. **IRT rank: 1st (most conservative, xi=+4.17).** Jumped 22 ranks from PCA to IRT — the largest shift in the chamber.
- **Explanation:** Tyson has a 61.9% Yea rate and 74 Nay votes — more than double the Republican median. But her contrarian Nay votes are concentrated on low-discrimination bills (|beta| <= 1.5), which IRT downweights. On high-discrimination bills (|beta| > 1.5), she is 100% conservative: 63/63 Yea on R-Yea, 18/18 Nay on D-Yea. No other senator has a perfect record. Additionally, 31 of her 41 dissent votes (Nay where >80% Rs = Yea) are on negative-beta bills, meaning her dissent actually *reinforces* her conservative score rather than moderating it. This is a 1D model limitation: her two-dimensional behavior (ideology + contrarianism) is compressed into a single axis, and the axis captures the dimension that's most informative about ideology.
- **Downstream:**
  - **Clustering:** Tyson's IRT position will pull any cluster she's in toward an extreme. Consider supplementing IRT with a party loyalty metric to distinguish "ideologically extreme" from "unreliable caucus member."
  - **Prediction:** IRT ideal points will predict her partisan votes well but miss her contrarian dissent on routine bills. A 2D model would improve predictions for Tyson specifically.
  - **Interpretation:** Always present Tyson's ranking with the caveat that 1D IRT conflates "consistently conservative on partisan votes" with "most conservative overall." See `analysis/design/tyson_paradox.md` for full investigation.

### Sen. Mike Thompson (R, District 10)

- **Phase:** PCA
- **Observation:** Third-most extreme PC2 (-8.0). Same direction as Tyson but milder.
- **Explanation:** Similar pattern — higher-than-typical Nay rate on routine bills (73.4% Yea rate). Shows a softer version of the Tyson contrarian tendency.
- **Downstream:** Same as Tyson. Check if Thompson and Tyson form a recognizable caucus or voting bloc. Clustering phase should reveal whether they consistently co-vote.

### Sen. Silas Miller (D, District ?)

- **Phase:** PCA
- **Observation:** Second-most extreme PC2 (-10.9). Only 30/194 votes (15.5%) — dead last in Senate participation.
- **Explanation:** Mid-session replacement. Previously served in the House with a normal voting record. Row-mean imputation filled 85% of his Senate matrix with his average Yea rate, producing an artificial PC2 extreme. **This is an imputation artifact, not a real voting pattern.**
- **Downstream:**
  - **IRT (Phase 4):** Use Miller as a **bridging legislator**. He served in both chambers, so a joint IRT model can use his ~300+ House votes to tightly constrain his ideal point, with the Senate votes further refining it. This is the standard "bridging observations" technique in the ideal-point literature.
  - **Clustering:** Exclude from Senate clustering or flag his cluster assignment as low-confidence.
  - **General:** Any analysis with a minimum-participation filter should note that Miller barely clears the 20-vote threshold. His estimates carry much more uncertainty than typical senators.

## Flagged Voting Patterns

### PC2 as "Contrarianism on Routine Legislation"

- **Phase:** PCA (Senate)
- **Observation:** PC2 (11.2% of variance) is driven by a cluster of near-unanimous bills where 1-2 senators dissent. The top PC2 loadings are routine bills (consent calendar, waterfowl hunting regs, bond validation, National Guard education).
- **Interpretation:** This is not a traditional ideological dimension. It captures a tendency to vote against the chamber consensus on uncontroversial legislation. Tyson is the primary driver, Thompson secondary, with Miller's position artifactual.
- **Downstream:** When interpreting Senate clustering results, the Tyson/Thompson pattern may create a spurious "cluster" that is really just two contrarian voters, not a substantive ideological faction. Consider whether PC2 should be downweighted or excluded in clustering inputs.

### Sen. Silas Miller (D) — IRT Update

- **Phase:** IRT
- **Observation:** IRT ideal point xi=-0.892, HDI=[-1.341, -0.439], width=0.902 (13th widest of 42 senators). Despite having only 30/194 votes (15.5%), his HDI is not the widest — extreme conservative senators have wider intervals due to fewer discriminating bills at the tail.
- **Explanation:** IRT handles Miller's sparse data natively (absences absent from likelihood, no imputation). His 30 observed votes are consistent enough to produce a reasonably constrained estimate. The PC2 artifact from PCA does not carry over — this is exactly the improvement IRT provides over PCA for sparse legislators.
- **Downstream:**
  - **Clustering:** HDI width of 0.902 means his ideal point is less certain than most Democrats. Consider weighting by 1/xi_sd or flagging his cluster assignment.
  - **Bridging:** A joint cross-chamber IRT model could use his ~300+ House votes to tighten the Senate estimate further. Deferred to future enhancement.

### Sen. Scott Hill (R)

- **Phase:** IRT
- **Observation:** Widest HDI in Senate: width=2.028 (xi=+1.329, HDI=[+0.398, +2.426]). Well-separated from the pack (next widest is 1.412).
- **Explanation:** Likely low participation on contested votes or voting pattern that doesn't align cleanly with the 1D model. Warrants investigation.
- **Downstream:** Cluster assignment is lowest-confidence in Senate. Flag in any ranking or comparison.

### House ESS Warning — Resolved

- **Phase:** IRT
- **Observation:** With LogNormal beta prior, minimum ESS for House ideal points was 214 (threshold: 400). **After switching to Normal(0,1) beta prior, ESS min improved to 894 — well above threshold.** All convergence checks now pass.
- **Explanation:** The LogNormal prior created poor posterior geometry (bimodal beta distribution, ceiling effects). The unconstrained Normal prior resolved this, improving ESS by 4x and eliminating the only convergence warning.
- **Status:** Resolved as of 2026-02-20 (Normal(0,1) beta prior).

## Flagged Voting Patterns — IRT

### Sensitivity Analysis: Highly Robust

- **Phase:** IRT
- **Observation:** Ideal points are extremely stable across minority thresholds. Pearson r between 2.5% and 10% runs: House r=0.9982, Senate r=0.9930.
- **Interpretation:** The 1D ideological structure is not driven by borderline-contested votes. Removing them barely changes legislator positions. This validates the 2.5% default threshold.

### PCA-IRT Agreement

- **Phase:** IRT
- **Observation:** Pearson r with PCA PC1: House r=0.972, Senate r=0.939. Both above 0.90; House exceeds the 0.95 "strong" threshold.
- **Interpretation:** High agreement confirms both methods recover the same 1D structure. The Senate's lower r=0.939 reflects two things: (1) IRT weights discriminating bills more heavily (inflating Tyson/Thompson), and (2) IRT handles sparse data (Miller) without imputation artifacts. See `analysis/design/tyson_paradox.md` for a detailed investigation of the largest PCA-IRT rank divergences.
- **Downstream:** Use IRT ideal points (not PCA scores) as the primary input for clustering and network analysis. IRT provides uncertainty estimates and handles missing data properly. But be aware that IRT ideal points systematically inflate the ranking of contrarian legislators (Tyson, Thompson) relative to PCA.

## Flagged Voting Patterns — Clustering

### k=2 Optimal, k=3 Hypothesis Rejected

- **Phase:** Clustering
- **Observation:** Both hierarchical (Ward on Kappa) and k-means (on IRT) selected k=2 as optimal for both chambers. Silhouette at k=2 = 0.82 (House), 0.79 (Senate); at k=3 = 0.64 (House), 0.57 (Senate). GMM selected k=4 by BIC.
- **Explanation:** The moderate/conservative Republican distinction is continuous, not discrete. The party boundary is the dominant clustering structure. With a ~72% Republican supermajority, intra-R variation is spread smoothly across the ideal-point spectrum. GMM's k=4 likely captures distributional shape (e.g., the long right tail) rather than genuine factions.
- **Downstream:**
  - **Network:** Community detection may find finer structure than k-means because it operates on pairwise agreement edges, not centroids. Test whether Louvain/modularity recovers 3+ communities.
  - **Prediction:** Cluster labels at k=2 are equivalent to party and won't add predictive power. Consider party loyalty as a continuous feature instead.

### Tyson and Thompson Cluster Assignments

- **Phase:** Clustering
- **Observation:** Tyson (xi=+4.17, loyalty=0.417) and Thompson (xi=+3.44, loyalty=0.472) both cluster with conservative Rs (Cluster 0) at k=2. They have the two lowest party loyalty scores in the Senate.
- **Explanation:** Their extreme IRT positions dominate the 1D clustering — they're far from Democrats and firmly in the R cluster. The 2D (IRT x loyalty) scatter plot visually separates them from core party members despite same cluster assignment.
- **Downstream:** The party loyalty metric successfully distinguishes "ideologically extreme" from "reliable caucus member." For network analysis, Tyson and Thompson may have lower within-cluster edge weights than typical Rs.

### Miller and Hill Cluster Confidence

- **Phase:** Clustering
- **Observation:** Miller (xi=-2.19, loyalty=1.000) clusters with Democrats; Hill (xi=+1.44, loyalty=1.000) clusters with Republicans. Miller has perfect loyalty on his contested votes; Hill has widest HDI in Senate.
- **Explanation:** Miller's sparse data (30/194 votes) produces wider IRT uncertainty, but his cluster assignment is unambiguous (firmly D). Hill's wide HDI (2.03) means his ideal point could range from moderate to solidly conservative, but he still clusters with Rs.
- **Downstream:** Both assignments are stable across methods (ARI = 0.90+ across all pairs). No special handling needed.

### Cross-Method Agreement — Very Strong

- **Phase:** Clustering
- **Observation:** Mean ARI = 0.958 (House), 0.935 (Senate) across hierarchical/k-means/GMM. Hierarchical and k-means are perfectly aligned (ARI = 1.0 for Senate).
- **Explanation:** The 2-cluster structure is extremely robust. The high ARI despite different input spaces (Kappa distance vs IRT ideal points) and algorithms confirms the party split is the overwhelming signal.
- **Downstream:** High confidence that any network community detection finding >2 groups represents genuinely finer structure, not algorithmic noise.

### Veto Overrides — Strictly Party-Line

- **Phase:** Clustering
- **Observation:** 17 veto override votes per chamber. R cluster: 98% Yea (House), 98% Yea (Senate). D cluster: 1% Yea (House), 1% Yea (Senate). No cross-party coalition detected.
- **Explanation:** Unlike Congress where veto overrides often produce bipartisan coalitions, Kansas overrides in the 2025-26 session are strictly partisan. The R supermajority can override unilaterally without D votes.
- **Downstream:** Veto override subgroup adds no novel clustering structure. Network analysis may find override votes are among the most party-line (highest discrimination).

## Template

```
### [Legislator Name or Pattern]

- **Phase:** [EDA | PCA | IRT | Clustering | Network]
- **Observation:** What was seen in the data.
- **Explanation:** Why it happened (if known).
- **Downstream:** What to do about it in future phases.
```
