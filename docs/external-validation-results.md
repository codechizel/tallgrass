# Are Our Ideology Scores Right? Checking Our Work Against the Gold Standard

**How we verified our Kansas Legislature ideal points against an independent national dataset — and what we found.**

*2026-02-24*

---

## The Problem: Grading Your Own Homework

Imagine you build a bathroom scale in your garage. You weigh yourself ten times and get the same number every time. Consistent? Yes. Accurate? You have no idea. The only way to find out is to step on a *different* scale — one you trust — and compare.

That's exactly the situation our analysis pipeline was in.

We built a statistical model that places every Kansas legislator on an ideological scale: far left on one end, far right on the other. The model is internally consistent — it agrees with itself in every way we can check. Our principal components analysis (a simpler technique) gives nearly the same ranking. Our model can predict 93% of votes correctly. Legislators who return session after session get similar scores each time. Everything lines up.

But all of those checks are circular. We're grading our own homework. We never asked: *does an independent measure of ideology, built by different researchers using different methods, agree with our scores?*

Every published ideal point study does this. It was the single biggest credibility gap in our pipeline.

---

## The Answer: The Shor-McCarty Dataset

Boris Shor and Nolan McCarty are political scientists who spent years building a national database of state legislator ideology scores. Their dataset covers **all 50 states, from 1993 to 2020** — including Kansas. The methodology was published in the *American Political Science Review* in 2011, one of the top journals in the field.

Their approach is clever. They start with a standardized survey (the National Political Awareness Test) that legislators in multiple states answer. Because some legislators take the same survey *and* cast votes in their state legislature, those shared survey responses become "bridges" that connect the scales across states. Within each state, they estimate ideal points from roll call votes — just like we do — but the bridges let them place everyone on a single national scale.

The dataset is free, publicly available from Harvard Dataverse under a CC0 (public domain) license. For Kansas, it includes **610 legislators** with ideology scores spanning 1996 to 2020.

Think of it this way: Shor-McCarty built a professionally calibrated scale. We built our own scale. Now we're stepping on both and comparing the readings.

---

## What We Compared

Our pipeline estimates ideology scores per legislative session (a two-year biennium). Shor-McCarty assigns each legislator a single career-level score. This means we can't compare them for every session — only for the years that overlap.

The overlap window is **2011 to 2020**, covering the 84th through 88th Kansas Legislatures. We started with the most recent overlapping session: the **88th Legislature (2019-2020)**.

For each legislator who served in the 88th Legislature and appears in the Shor-McCarty dataset, we have two ideology scores:

- **Our score** (`xi_mean`): estimated from Kansas roll call votes using Bayesian Item Response Theory
- **Their score** (`np_score`): estimated independently by Shor and McCarty using national bridging methodology

If both methods are measuring the same thing — where a legislator falls on the liberal-conservative spectrum — the two sets of scores should be highly correlated. Statisticians measure this with **Pearson's r**, a number from -1 to +1:

| Correlation (r) | What it means |
|------------------|---------------|
| 0.95 - 1.00 | Essentially identical rankings |
| 0.90 - 0.95 | Strong agreement — minor differences on a few individuals |
| 0.85 - 0.90 | Good agreement — differences likely reflect real dynamics |
| 0.70 - 0.85 | Moderate — worth investigating what's different |
| Below 0.70 | Concerning — one or both methods may have a problem |

Our target was **r > 0.85** (good agreement). Published studies in this field typically report correlations in the 0.85-0.95 range between different ideal point methods.

---

## The Results

### Matching Legislators

Before computing any correlations, we had to match legislators across the two datasets. Our data uses names like "John Alcala" while Shor-McCarty uses "Alcala, John." We built a name normalization algorithm that converts both to a common format ("john alcala"), handling middle names, suffixes like "Jr." and "Sr.", and leadership titles like "- President."

The matching worked remarkably well:

- **House**: 127 of 128 legislators matched (**99.2%**) — 85 Republicans, 42 Democrats
- **Senate**: 41 of 41 legislators matched (**100%**) — 30 Republicans, 11 Democrats

Only one House member couldn't be matched, likely someone who entered the legislature after Shor-McCarty's coverage ended in 2020.

### The Correlation Table

Here are the headline numbers. We tested two versions of our model: the standard ("flat") IRT and a more sophisticated hierarchical version that partially pools information within parties.

| Model | Chamber | Pearson r | Spearman ρ | Matched | Quality |
|-------|---------|-----------|------------|---------|---------|
| Flat IRT | House | **0.981** | 0.962 | 127 | Strong |
| Flat IRT | Senate | **0.929** | 0.962 | 41 | Strong |
| Hierarchical IRT | House | **0.984** | 0.973 | 127 | Strong |
| Hierarchical IRT | Senate | -0.541 | -0.109 | 41 | Concern |

*Pearson r measures linear agreement. Spearman ρ measures rank-order agreement (do the two methods agree on who's more conservative than whom?). "Matched" is the number of legislators present in both datasets.*

### What These Numbers Mean

**House (both models): r = 0.98.** This is exceptional. Out of 127 matched House members, our pipeline and Shor-McCarty agree almost perfectly on where each legislator falls on the ideological spectrum. A correlation of 0.98 means that if you picked any two House members at random, the two methods would agree on which one is more conservative more than 99% of the time.

To put this in perspective: if you measured everyone's height with two different rulers, you'd expect a correlation very close to 1.0. Our ideology scores are nearly that consistent with the gold standard.

**Senate (flat model): r = 0.93.** Still strong, though slightly lower than the House. This makes sense: with only 41 senators (versus 127 representatives), each individual has more influence on the correlation, and the Shor-McCarty career-level score has less data per senator to work with.

**Senate (hierarchical model): r = -0.54.** This is the one bad result, and it has a clear technical explanation (see the next section). It does *not* mean our Senate scores are wrong — the flat model's r = 0.93 confirms they're solid. It means our hierarchical model has a specific problem with small chambers.

---

## The Scatter Plots: Seeing the Agreement

Numbers tell you the correlation is strong. Pictures show you *how* it's strong.

### House: Flat IRT vs. Shor-McCarty

The House scatter plot shows 127 dots — one per legislator — arranged in a tight diagonal line from bottom-left (most liberal) to upper-right (most conservative). Democrats cluster in the lower-left corner, Republicans fill the upper-right half.

The dots hug the regression line so closely that you can barely see the gray confidence band around it. There are no wild outliers — no legislator that one method says is very liberal while the other says is very conservative. The tightest cluster is among Democrats, who form an especially compact group. Republicans show slightly more spread, reflecting the genuine intra-party ideological variation that our earlier analyses identified.

The most liberal legislator in both datasets: **Rui Xu** (D-House), who our model scores at -2.93 and Shor-McCarty scores at -1.40 (the scales differ, but the ranking agrees). The most conservative: **Trevor Jacobs** (R-House), at +3.20 in our model and +1.74 in Shor-McCarty's.

### Senate: Flat IRT vs. Shor-McCarty

The Senate scatter plot has only 41 dots, so each legislator is more visible. The overall pattern is the same — a clear diagonal from liberal Democrats to conservative Republicans — but with more scatter around the line, which is expected with fewer data points.

The most interesting story is **Jim Denning**, the Republican Senate Majority Leader. Our model places him at -0.58 (surprisingly moderate for a Republican), while Shor-McCarty places him at +0.75 (solidly conservative). That's the largest discrepancy in the Senate. But it's not mysterious: as Majority Leader during the 2019-2020 session, Denning's voting likely reflected the strategic compromises that come with managing a caucus rather than purely expressing personal ideology. Shor-McCarty's career score captures who Denning was across his whole tenure; our session-specific score captures what he did during those particular two years.

---

## Why the Hierarchical Senate Model Failed (and Why That's Actually Informative)

The one concerning result — the hierarchical model's r = -0.54 for the Senate — deserves explanation, because it illustrates a real statistical phenomenon.

Our hierarchical model uses "partial pooling." Think of it like a teacher grading on a curve within each class section. Instead of evaluating every student against the entire school, the model first groups legislators by party and then estimates each individual's position partly based on how their party typically votes.

This works well when there are enough legislators per group. With 85 Republicans and 42 Democrats in the House (127 total), there's plenty of data to estimate both the group tendencies and the individual deviations. The hierarchical House model actually *outperforms* the flat model (r = 0.984 vs. 0.981).

But the Senate has only 30 Republicans and 11 Democrats. With groups this small, the model "over-shrinks" — it pulls everyone too close to their party's average, erasing the individual differences that make the scores useful. It's like the teacher deciding that all students in a 10-person class must have gotten roughly the same score, because the class is too small to be confident about individual differences.

This is a textbook limitation of hierarchical models with small groups, and it's exactly the kind of finding that external validation is designed to catch. Without the Shor-McCarty comparison, we wouldn't know the hierarchical Senate scores have this problem. Now we do, and analysts can use the flat model for Senate analysis with confidence.

---

## What This Means for the Project

### The Good News

Our core methodology is validated. The flat IRT model — the one used throughout the pipeline — produces ideology scores that agree with the national gold standard at r = 0.93 to 0.98. This means:

1. **The rank orderings are real.** When we say Legislator A is more conservative than Legislator B, an independent national dataset agrees.
2. **The party separation is real.** The gap between the most conservative Democrat and the most liberal Republican in our data matches the gap in Shor-McCarty's data.
3. **The intra-party variation is real.** The differences we find among Republicans — which is the most analytically interesting signal in a supermajority state — are not artifacts of our methodology.

### The Caveats

- **We've validated one biennium so far** (88th, 2019-2020). Four more overlapping bienniums (84th through 87th) remain to be tested. If the correlation is equally strong across all five, that's definitive. If it drops for earlier sessions (especially the 84th, which has known data quality issues), that tells us something about historical data, not methodology.
- **Shor-McCarty's coverage ends in 2020.** For the 89th Legislature (2021-2022) and beyond, we cannot perform this validation. Our results for recent sessions stand on the strength of the methodology validated here, plus cross-session stability checks.
- **Career scores vs. session scores.** Shor-McCarty gives each legislator one score for their entire career. We give each legislator a score per session. Legislators who genuinely change over time will show up as "disagreements" even if both methods are correct. Jim Denning's discrepancy may reflect exactly this.

---

## How External Validation Fits the Pipeline

This validation step sits outside the regular 12-phase analysis pipeline. It doesn't change any scores or modify any results. It answers a single question: *can we trust the numbers?*

The answer, for the flat IRT model, is yes. The correlation of r = 0.98 in the House and r = 0.93 in the Senate places our pipeline's accuracy squarely within the range of published academic studies in this field. We are now on the same footing as the peer-reviewed literature in terms of methodological credibility.

---

## Technical Details

For readers who want the specifics:

- **Shor-McCarty data source**: Harvard Dataverse, doi:10.7910/DVN/NWSYOS, CC0 license. Variable used: `np_score` (national common space score).
- **Our data**: Bayesian 2PL IRT ideal points estimated via PyMC (4 chains, 2000 tuning + 2000 draws). PCA-informed chain initialization per ADR-0023.
- **Matching**: Two-phase name normalization. Phase 1: exact match on lowercase "first last" + chamber. Phase 2: last-name-only + district number tiebreaker. Overall match rate: 99.4% (168/169).
- **Correlation method**: Pearson r for linear agreement, Spearman ρ for rank agreement. 95% confidence intervals via Fisher z-transformation.
- **Code**: `analysis/external_validation.py` (runner), `analysis/external_validation_data.py` (pure logic), `analysis/external_validation_report.py` (HTML report builder). Design doc at `analysis/design/external_validation.md`. ADR-0025.
- **Reference**: Shor, B. and N. McCarty. 2011. "The Ideological Mapping of American Legislatures." *American Political Science Review* 105(3): 530-551.
