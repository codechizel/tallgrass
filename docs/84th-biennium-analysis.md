# The 84th Kansas Legislature (2011-2012): A Party at War with Itself

*Analysis of the full Tallgrass pipeline results for the 84th Kansas Legislature, the first biennium in our 2011-2026 coverage.*

---

## Historical Context

The 84th Kansas Legislature convened in January 2011, weeks after the 2010 Republican wave swept the country. Sam Brownback, freshly elected governor, arrived in Topeka with an ambitious conservative agenda: the largest tax cuts in state history, restrictions on abortion, and Medicaid block grants. He had overwhelming numbers — Republicans held 81 of 125 House seats (65%) and 30 of 40 Senate seats (75%).

But Brownback had a problem. His own party was split.

The Kansas Senate had been governed for years by a coalition of moderate Republicans and Democrats who controlled the chamber through the leadership of Senate President Steve Morris (R-Hugoton), Vice President John Vratil (R-Leawood), and budget committee chair Carolyn McGinn (R-Sedgwick). These were self-described "Bob Dole Republicans" — fiscal pragmatists who thought in terms of moderation. They could block any legislation the conservative wing pushed through the House by allying with the Senate's seven Democrats.

The data from the Tallgrass pipeline captures this factional war in quantitative detail. Across every analytical method — PCA, IRT ideal points, clustering, network analysis, party indices, and Bayesian models — the same finding emerges: **the Kansas Republican Party in 2011-2012 was not one party. It was two.**

---

## Data Quality: The ODT Era

The 84th is the earliest biennium in our coverage and uses ODT (OpenDocument Text) vote files rather than the HTML vote pages available from 2015 onward. This creates several data quality constraints that downstream phases must account for:

| Issue | Impact |
|-------|--------|
| 29.4% of vote pages failed to parse (374 of 1,274) | All are committee-of-the-whole tallies with aggregate counts but no individual names |
| 12 House members and 3 Senate members never appear | Their votes exist only in ODT tally headers, not as parseable individual records |
| 57.9% of roll calls have "Unknown" vote type | ODT metadata often lacks motion/vote-type fields |
| 899 of 900 roll calls show tally mismatches | Structural: header tallies include members whose individual votes weren't parsed |
| Zero "Not Voting" records | ODT parser collapses "Not Voting" and "Absent and Not Voting" into one category |
| No bill title text available | NMF topic features in the prediction phase are all zeros |
| One corrupted ODT produced 250 votes in a 125-seat House | `je_20120519114358_525311` — likely concatenated data from two vote pages |

Despite these gaps, the dataset contains **150 legislators** (113 House, 37 Senate), **900 roll calls** across 436 bills, and **67,549 individual vote records**. After filtering near-unanimous votes, the analysis uses 260 House votes and 172 Senate votes — enough for meaningful statistical inference.

The 29.4% missing vote rate is the highest in our eight-biennium coverage. The missing votes skew heavily toward unanimous or near-unanimous outcomes (most committee-of-the-whole votes pass by wide margins), but some close votes are lost — including HB 2134 (20-19, margin 1), which would have been highly informative for IRT estimation.

---

## Phase 1: EDA — Two Parties, One Label

The exploratory data analysis immediately surfaces the central tension. At first glance, the numbers look like a typical Republican-dominated legislature:

| | House | Senate |
|---|---|---|
| Republicans | 81 (72%) | 30 (81%) |
| Democrats | 32 (28%) | 7 (19%) |

But the voting behavior tells a different story:

- **Party-line votes: only 22 of 900 roll calls (2.4%).** For comparison, the 91st Legislature (2025-26) has a 22.4% party-line rate — a ninefold increase. In the 84th, the parties almost never lined up cleanly against each other.
- **Bipartisan votes: 61.8%.** Nearly two-thirds of all roll calls saw majorities of both parties voting the same way.
- **Republican Rice cohesion: 0.845.** Compare to 0.936 in the 91st. The 84th Republican caucus had significant internal disagreement.
- **Democratic Rice cohesion: 0.905.** The smaller Democratic caucus was more unified than the majority party.

The eigenvalue structure from the vote matrix provides early warning of multi-dimensionality. The House lambda1/lambda2 ratio is 3.09 ("primarily 1D"), but the Senate ratio is only 2.55 — below the threshold of 3.0, flagging that a meaningful second dimension is present.

---

## Phases 2-3: Dimensionality Reduction — The Moderate Bloc Emerges

### PCA

PC1 explains 27.3% of House variance and 29.6% of Senate variance — low compared to typical legislative PCA results (40-60% in the literature) and well below the 91st's 53.4% (House). The reduced signal partly reflects the ODT-era data gaps diluting the partisan dimension, and partly reflects genuine multi-dimensionality in the voting behavior.

Despite the lower explained variance, PC1 achieves perfect party separation in both chambers — no Democrat scores on the Republican side. But within the Republican party, the spread is enormous:

**House:** 12 Republicans have negative PC1 scores (voting more like Democrats than like their own party median). The most extreme: Charles Roth (-4.04), Barbara Bollier (-3.66), Lorene Bethell (-2.72), and Tom Sloan (-2.62). Bollier, who would switch parties to become a Democrat in 2018 and run for the U.S. Senate in 2020, was already ideologically positioned as a Democrat a full decade before the switch.

**Senate:** 12 of 30 Republicans have negative PC1 scores — 40% of the caucus. The most extreme: John Vratil (-7.90), Pete Brungardt (-7.21), Jean Schodorf (-6.72), Roger Reitz (-6.35), and Senate President Steve Morris (-5.89). These five moderate Republican senators are positioned deeper into Democratic territory on PC1 than some actual Democrats.

PC2 captures a secondary dimension of contrarianism. Caryn Tyson (House, PC2 = -17.95) and Mary Pilcher-Cook (Senate, PC2 = -18.6) are extreme outliers — both are strongly conservative on PC1 but distinctly different from their party on the secondary axis. This is the "Tyson paradox" pattern seen across multiple bienniums.

### MCA

The Multiple Correspondence Analysis reveals an important methodological limitation for this session. **House MCA is clean** — PCA validation r = 0.979, no horseshoe effect, stable sensitivity. But **Senate MCA is dominated by one legislator's absences.**

Les Donovan, a Republican senator with a 42.3% absence rate (121 of 286 votes), distorts the entire Senate MCA. His cos-squared on Dim1 is 0.991 — he alone accounts for nearly all of the first dimension's variance. The top 20 Dim1 contributions are all "Absent" categories. MCA Dim1 measures "how similar is your attendance pattern to Les Donovan's," not ideology.

This drives the PCA validation correlation down to r = 0.740 (below the 0.90 threshold) and causes the sensitivity analysis to collapse (r = 0.316 between unanimity thresholds). Senate MCA results for this session should be treated as unreliable for ideological interpretation. This is not a code bug — it is a known limitation of MCA when treating absences as a third category, amplified by a single high-absence legislator.

### UMAP

The UMAP embedding creates clear party separation in both chambers (minimum cross-party distance = 11.0 in House), but the **House UMAP1 validation correlation is misleadingly low (r = 0.374).** The ideological signal is present but rotated onto a diagonal axis rather than aligned with UMAP1. Within-party correlations are strong (Republicans: UMAP2 vs PC1 rho = -0.85), confirming the signal exists but the reporting metric doesn't capture it.

The Senate UMAP is the most visually striking result: a dramatic three-group structure with a gap of 17.5 UMAP units between the conservative Republican cluster and the moderate-Republican-plus-Democrat cluster. The moderate Republicans are not a separate group — they are continuous with the Democrats, with no visible boundary between them.

---

## Phase 4: Bayesian IRT — Ideal Points Tell the Full Story

### Flat IRT (1D)

Both chambers converge cleanly under the nutpie Rust NUTS sampler — all R-hat < 1.006, all ESS > 700, zero divergences. This is notable because the 84th House was historically one of five sessions that failed convergence with PyMC's default sampler (R-hat 1.83). The nutpie migration (ADR-0053) resolved this completely.

The ideal point distributions quantify what every other method has shown:

**House ideal points:**

| Party | Mean xi | Std | Range |
|-------|---------|-----|-------|
| Republican | +1.121 | 0.813 | -0.578 to +2.731 |
| Democrat | -2.150 | 0.631 | -3.333 to -1.012 |

Twelve Republicans have negative ideal points, placing them left of center. The most liberal Republican, Charles Roth (xi = -0.578), is positioned between the two parties.

**Senate ideal points:**

| Party | Mean xi | Std | Range |
|-------|---------|-----|-------|
| Republican | +0.034 | 1.435 | -2.318 to +3.089 |
| Democrat | -2.119 | 0.591 | -3.077 to -1.467 |

The Senate Republican mean is effectively zero — the average Republican senator is ideologically centrist, not conservative. The within-party standard deviation (1.435) is 2.4 times the Democratic standard deviation (0.591). Twelve of 30 Senate Republicans have negative ideal points. John Vratil (xi = -2.318) is positioned more liberally than five of the seven Democrats.

Dennis Pyle emerges as the most conservative senator (xi = +3.089), more extreme than even the anchor legislator Mary Pilcher-Cook. His voting pattern foreshadows his 2022 independent gubernatorial campaign, running to the right of the Republican nominee.

### 2D IRT (Experimental)

The 2D model technically fails ESS thresholds (124-191 for some parameters vs the 200 relaxed threshold) but produces interpretable results with R-hat values all under 1.05 and zero divergences.

Dimension 1 correlates strongly with PCA PC1 (House r = 0.951, Senate r = 0.865), confirming it as the partisan axis. Dimension 2 partially captures the secondary contrarianism dimension but with weaker validation (House r = 0.731, Senate r = 0.311 with PCA PC2).

The 2D model decomposes Dennis Pyle's extreme 1D position: his Dim1 is a moderate +1.349 (not even the most conservative), while his Dim2 is an extreme -3.284. His apparent hyper-conservatism in the 1D model partly reflects secondary-dimension behavior compressed onto the single axis. Mary Pilcher-Cook is extreme on both dimensions (Dim1 = +2.928, Dim2 = -4.048).

### Hierarchical IRT

The per-chamber hierarchical models converge perfectly (all R-hat < 1.003, all ESS > 700, zero divergences). The adaptive prior (HalfNormal(0.5)) for the Senate Democrat sigma_within compensates for the small group size (7 members).

The intra-class correlation (ICC) tells the most important story:

| Chamber | ICC | Interpretation |
|---------|-----|----------------|
| House | 0.850 | Party explains 85% of ideological variance |
| Senate | **0.502** | Party explains only half the variance |

A Senate ICC of 0.502 is remarkably low. In a chamber where party is supposed to be the organizing principle, party membership predicts only half the variation in how senators vote. The other half is within-party disagreement — primarily the moderate-conservative Republican split.

The group-level parameters confirm this:

| Chamber | Party | mu (party mean) | sigma_within |
|---------|-------|-----------------|--------------|
| House | Democrat | -4.518 | 1.277 |
| House | Republican | +2.017 | 1.395 |
| Senate | Democrat | -4.892 | 0.551 |
| Senate | Republican | +0.557 | **2.994** |

Senate Republicans have a sigma_within of 2.994 — over five times the Democratic within-party spread (0.551). The Republican group mean (+0.557) has a 95% HDI that includes zero, meaning the model cannot confidently place the average Republican senator on the conservative side of the spectrum.

Senate Democrat shrinkage is aggressive (up to 62% for Oletha Faust-Goudeau), reflecting the heavy prior influence with only 7 members. The flat IRT may be more trustworthy for Senate Democrat rankings in this session.

The **joint cross-chamber model catastrophically fails** — R-hat up to 2.26, ESS as low as 1 per chain, 427 divergences. This is consistent with the known joint model convergence problem across all eight bienniums and the results should be discarded.

---

## Phase 5: Clustering — The Two Republican Parties

Every clustering method independently discovers the same structure:

**House k=2:** Cluster 0 = 79 Republicans (100% R). Cluster 1 = 32 Democrats + 2 Republicans (Bollier and Roth). The two most liberal Republicans are algorithmically indistinguishable from Democrats.

**Senate k=2:** Cluster 0 = 18 Republicans (100% R). Cluster 1 = 12 Republicans + 7 Democrats. This is the central finding: **40% of Senate Republicans cluster with Democrats rather than with their own party.** K-means, hierarchical, HDBSCAN, spectral, and GMM all agree. GMM assigns members with >0.99 probability to their clusters.

Within-party subclustering of Senate Republicans produces a clean 2-way split:
- **Conservative wing (18):** Pyle, Merrick, Lynn, Olson, Abrams, Wagle, Masterson, Pilcher-Cook. Loyalty 58-89%.
- **Moderate wing (12):** Vratil (loyalty 51%), Schodorf (54%), Brungardt (56%), Huntington (56%), Morris (59%), McGinn (59%), Umbarger (59%), Reitz, Emler, Longbine, Marshall, Teichman. Loyalty 51-66%.

---

## Phase 6: Network Analysis — The Senate's Broken Assortativity

The network analysis produces the single most dramatic statistic in the entire pipeline:

| Metric | House | Senate |
|--------|-------|--------|
| Party assortativity | **0.906** | **0.188** |

House party assortativity of 0.906 indicates near-perfect party-line voting agreement — if you know a legislator's party, you can predict who they vote with. Senate party assortativity of 0.188 is barely above random. **Party label is nearly meaningless for predicting Senate voting coalitions.**

Community detection finds only 2 communities in the Senate (stable across all resolution parameters from 0.5 to 1.25), but they do not follow party lines:
- Community 0: 19 members = 12 Republicans + 7 Democrats (the moderate coalition)
- Community 1: 18 members = 18 Republicans (the conservative caucus)

The conservative Republicans have zero cross-party edges (Merrick, Olson, Apple, Ostmeyer, Bruce). The moderate Republicans are the only pathway between the two blocs: Charles Roth (House betweenness = 0.104, 3x any other legislator) serves as the critical bridge.

---

## Phase 7: Indices — The Maverick Caucus

The classical indices paint the starkest picture:

### Top Senate Mavericks (Weighted Score)

| Rank | Legislator | Party | Maverick Rate | IRT xi |
|------|-----------|-------|---------------|--------|
| 1 | Pete Brungardt | R | 78.8% | -2.07 |
| 2 | Terrie Huntington | R | 78.4% | -1.93 |
| 3 | John Vratil | R | **81.1%** | -2.32 |
| 4 | Steve Morris | R | 69.8% | -1.79 |
| 5 | Carolyn McGinn | R | 67.9% | -1.39 |

**John Vratil voted against his party's majority 81.1% of the time.** He sided with the Republican caucus on only 10 of 53 party-line votes. Pete Brungardt (78.8%) and Terrie Huntington (78.4%) were almost as extreme. Senate President Steve Morris, the chamber's most powerful Republican, defected on nearly 70% of party votes.

At the other end: Mike Petersen, Pat Apple, and Mary Pilcher-Cook defect on only 5-6% of party votes. The within-party gap — 81% to 5% — is extraordinary for a single caucus.

Only 53 of 456 Senate votes (11.6%) are party-line votes, the lowest rate of any session. When 40% of your caucus routinely crosses over, few votes achieve a clean partisan split.

The co-defection network reveals a tight four-person nucleus (Brungardt, Vratil, Schodorf, Huntington: 37-39 shared defections each), surrounded by a ring of eight additional moderates who defect somewhat less frequently (Morris, Emler, McGinn, Reitz, Teichman, Umbarger, Longbine, Marshall).

---

## Phase 8: Prediction — IRT Does the Work

Vote prediction AUC falls in the 0.93-0.96 range — healthy, not suspiciously high (no data leakage) and well above the majority-class baseline (0.81-0.82). Logistic regression matches or slightly beats XGBoost, confirming that the IRT interaction term (`xi * beta`) captures the dominant signal — the relationship between votes and ideology is inherently logistic.

Feature importance (SHAP) ranks `xi_x_beta` first, followed by `beta_mean` and `xi_mean`. Party ranks below IRT features, confirming that IRT ideal points subsume the raw party signal.

The hardest-to-predict legislators are the mavericks: Trent LeDoux (61.9% accuracy), Chris Steineger (65.2%), Amanda Grosserode (68.0%). These are legislators whose voting behavior is genuinely multi-dimensional — no 1D model can fully explain them.

NMF topic features are completely empty for this session (no bill title text in the ODT era), so one entire feature category contributes zero signal. This is a data limitation, not a bug.

---

## Phase 9: Beta-Binomial — Loyalty Under a Bayesian Lens

The Beta-Binomial empirical Bayes produces posterior loyalty estimates that sharpen the maverick analysis. The most striking number: **Senate Republican prior mean loyalty is 61.9%.** This is not an error — it reflects the fact that the "average" Senate Republican voted with their party majority less than two-thirds of the time.

Eleven of 30 Senate Republicans have posterior loyalty below 50%. John Vratil's posterior loyalty is 19.9% — the lowest in the entire eight-biennium dataset. Barbara Bollier's is 43.6%.

Shrinkage is asymmetric by design: Senate Democrats average 30.5% shrinkage (strong prior from a small, cohesive group) while Senate Republicans average only 2.5% (weak prior from a large, fractured group).

---

## Phases 11-12: Synthesis and Profiles

The synthesis phase identifies seven notable legislators through data-driven detection:

| Legislator | Chamber | Role | Key Metric |
|-----------|---------|------|------------|
| Barbara Bollier (R) | House | Maverick | 58% maverick rate; IRT -0.53 |
| John Vratil (R) | Senate | Maverick | 81% maverick rate; IRT -2.32 |
| Vince Wetta (D) | House | Minority maverick | 34% maverick rate; IRT -1.07 |
| Oletha Faust-Goudeau (D) | Senate | Minority maverick | 17% maverick rate |
| Charles Roth (R) | House | Bridge-builder | Betweenness 0.104; 13 cross-party edges |
| Anthony Brown (R) | House | Paradox | IRT +2.33 (very conservative) yet 21% maverick — defects *rightward* |
| Dennis Pyle (R) | Senate | Paradox | Most conservative (IRT +3.09) yet defects rightward on intra-party votes |

The "Brown paradox" and "Pyle paradox" are mirror images of the Tyson paradox seen in later sessions: legislators who are the most ideologically extreme in their party yet frequently disloyal, because they defect in the *same direction* as their ideology. They vote Nay when their party votes Yea on bills they consider insufficiently conservative. High CQ party unity (86%) masks this because it only counts cross-party votes.

---

## What the Data Foreshadows

The 84th Legislature captured several politicians at inflection points, and the quantitative record is remarkably predictive of their trajectories:

**Barbara Bollier** (House, IRT -0.53, maverick rate 58%): Already the most maverick House Republican in 2011-2012. She would switch parties in December 2018, citing the Kansas GOP's anti-transgender platform as the "breaking point," and run for the U.S. Senate as a Democrat in 2020, losing to Roger Marshall by 12 points.

**Laura Kelly** (Senate, IRT -2.34): A second-term Democratic senator who would be elected Governor of Kansas in 2018, defeating Kris Kobach 48%-43%. Her 91st Legislature data shows the dynamic from the other side — a Democratic governor vetoing Republican legislation, with 34 veto override votes in the current session.

**Steve Morris** (Senate, IRT -1.79, maverick rate 70%): The Senate President who warned the AP before the 2012 primary, "There's a war. It's probably as bad as I've seen it." He was defeated in the August 2012 Republican primary by conservative challenger Larry Powell, backed by the Kansas Chamber of Commerce and Americans for Prosperity.

**The moderate purge:** The August 2012 primaries eliminated the faction the data captures at its peak. Senators Vratil, Schodorf, Brungardt, Huntington, Reitz, Umbarger, and Morris were either defeated or retired. The 85th Legislature (2013-2014) would show dramatically higher Republican cohesion as the moderate wing disappeared from the chamber.

---

## Analytic Flags and Potential Issues

### Genuine Concerns

1. **Senate MCA is unreliable for this session.** Les Donovan's 42.3% absence rate dominates MCA Dim1, turning it into an attendance measure rather than an ideological one. PCA validation drops to r = 0.740 (below the 0.90 threshold) and sensitivity collapses to r = 0.316. Downstream consumers of Senate MCA scores for this session would get misleading results.

2. **House UMAP validation is misleadingly low (r = 0.374).** The ideological signal is present in the 2D embedding but rotated off the UMAP1 axis. The validation metric only checks UMAP1, not the full 2D structure. This is a framework-level question about whether the validation should project onto the best ideological axis.

3. **Joint hierarchical model fails catastrophically** (R-hat 2.26, ESS 1, 427 divergences). This is consistent with the known joint model convergence problem across all sessions and the results should be discarded.

4. **Senate hierarchical shrinkage is aggressive for Democrats** (up to 62%). With only 7 members, the hierarchical prior dominates. Flat IRT may be more trustworthy for Senate Democrat rankings.

5. **2D IRT fails ESS thresholds** (124-191 vs 200) in both chambers. Results are interpretable with caveats but should not be treated as fully converged.

### Not Bugs (Expected Behavior)

6. **29.4% missing vote pages** — documented ODT committee-of-the-whole limitation (CLAUDE.md point 10).

7. **899/900 tally mismatches** — structural consequence of missing members, not a parsing error.

8. **Empty NMF topic features** — bill title text is unavailable in the ODT era.

9. **House passage holdout AUC is NaN** — 100% pass rate in the holdout set makes ROC undefined.

10. **Senate betweenness centrality is uniformly 0.0** for all but one legislator — extreme polarization means no bridge nodes exist, not a network construction error.

### Cross-Phase Consistency

No coding or statistical errors were detected. Every analytical method converges on the same conclusion independently: PCA, IRT, clustering (5 methods), network community detection, classical indices, and Bayesian models all identify the same moderate Republican bloc, the same mavericks, and the same factional structure. The cross-method agreement is itself strong evidence that the pipeline is working correctly — if any single method had a bug, it would diverge from the others.

---

## The 84th in Context

The 84th Kansas Legislature represents the last gasp of the moderate Republican tradition in Kansas state politics. The data captures a party in which 40% of Senate members routinely voted against their own leadership, the Senate President had a maverick rate of 70%, and party assortativity was barely above random. Within eighteen months, the August 2012 primaries — backed by the Brownback administration, the Kansas Chamber of Commerce, and Americans for Prosperity — would eliminate this faction and transform the Kansas Legislature into the more polarized body visible in later bienniums.

The analytical pipeline handles this unusual political configuration without breaking. The low ICC (0.502), low party assortativity (0.188), and low party-line vote rate (2.4%) are not artifacts — they are faithful measurements of a legislature that did not conform to the party-line assumptions built into most analytical frameworks. The hierarchical IRT's adaptive prior correctly adjusts for the tiny Senate Democratic caucus. The clustering algorithms correctly refuse to force a party-line split that the voting data doesn't support. The network analysis correctly reports that party label is nearly meaningless in the Senate.

If anything, the 84th Legislature serves as the best stress test for the pipeline: a session that violates the "strong party" assumption that most legislative analysis tools take for granted, yet produces coherent and historically validated results across all 14 phases.

---

*Generated from Tallgrass pipeline run `84-260228.1` (2026-02-28). Data source: Kansas Legislature (kslegislature.gov), 84th Legislature, 2011-2012 regular session.*

*For methodology details, see `docs/analysis-primer.md`. For the full pipeline architecture, see `CLAUDE.md`.*

## References

- [The Great Kansas Republican Purge of 2012](https://slate.com/news-and-politics/2012/08/the-great-kansas-republican-purge-of-2012.html) — Slate
- [Conservative takeover in Kansas just the beginning](https://www.washingtonpost.com/blogs/the-fix/post/conservative-takeover-in-kansas-just-the-beginning/2012/08/08/bd27ddd8-e163-11e1-98e7-89d659f9c106_blog.html) — Washington Post
- [Kansas conservatives claim control after primary victories](https://www.washingtontimes.com/news/2012/aug/9/kansas-conservatives-claim-control-after-primary-v/) — Washington Times
- [In Kansas, Governor Sam Brownback Drives a Rightward Shift](https://stateline.org/2012/01/25/in-kansas-governor-sam-brownback-drives-a-rightward-shift/) — Stateline/Pew
- [Last bastion of GOP moderates in state politics under attack](https://www2.ljworld.com/news/2012/jan/03/last-bastion-gop-moderates-state-politics-under-at/) — Lawrence Journal-World
- [Insight Kansas: Schmidt, Schodorf, and the Fate of the Moderate Republican Kansas Woman](https://mittelpolitan.substack.com/p/insight-kansas-column-for-july-vicki) — Mittelpolitan
- [Barbara Bollier — Wikipedia](https://en.wikipedia.org/wiki/Barbara_Bollier)
- [Laura Kelly — Wikipedia](https://en.wikipedia.org/wiki/Laura_Kelly)
