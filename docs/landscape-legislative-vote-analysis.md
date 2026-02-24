# The Landscape of Legislative Vote Analysis

**How political scientists, data journalists, and open-source projects measure what legislators do — and where our pipeline fits.**

*Last updated: 2026-02-24*

---

## Introduction

Analyzing legislative voting records is one of the oldest problems in quantitative political science. Since the 1980s, researchers have developed increasingly sophisticated methods for extracting ideological positions from the simple act of voting Yea or Nay. This document surveys the field: the dominant methods, the key researchers, the software ecosystem, and the emerging frontier — then locates our Kansas Legislature pipeline within that landscape.

The field divides roughly into **frequentist** approaches (NOMINATE, Optimal Classification, PCA), **Bayesian** approaches (IRT ideal points, hierarchical models), **machine learning** approaches (XGBoost, neural networks, LLM agents), and **network** approaches (co-voting graphs, community detection). Most published work focuses on the U.S. Congress; state legislatures remain dramatically understudied.

---

## The Frequentist Tradition

### NOMINATE: The 40-Year Workhorse

The dominant method in legislative analysis is the **NOMINATE** family, created by Keith T. Poole and Howard Rosenthal beginning in the early 1980s. NOMINATE (Nominal Three-Step Estimation) is a maximum-likelihood spatial voting model that places legislators and roll call votes in a low-dimensional space (usually 1-2 dimensions).

The core idea: each legislator has an **ideal point**, each vote has two **outcome points** (Yea and Nay positions), and legislators vote for the outcome closer to their ideal point, with some probabilistic error. The algorithm iterates between estimating legislator positions and vote parameters until convergence.

**Key variants:**

| Variant | Year | Key Feature |
|---------|------|-------------|
| D-NOMINATE | 1983 | Original dynamic model |
| W-NOMINATE | 1990s | Within-session, bootstrapped standard errors |
| DW-NOMINATE | 2000s | Dynamic + weighted; cross-Congress comparison |
| alpha-NOMINATE | 2013 | Bayesian, mixture utilities |

The first dimension captures the liberal-conservative spectrum with ~85-93% classification accuracy in modern Congresses. The second dimension historically captured cross-cutting issues (slavery, civil rights) but is less informative in the polarized modern era.

The current DW-NOMINATE scores are maintained by the **Voteview** project (Jeffrey B. Lewis, Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, Luke Sonnet) at [voteview.com](https://voteview.com). Scores cover every Congress from the 1st (1789) through the present and are updated live as new votes are taken.

**Foundational references:**
- Poole, K.T. and H. Rosenthal. 1985. "A Spatial Model for Legislative Roll Call Analysis." *American Journal of Political Science* 29(2): 357-384.
- Poole, K.T. and H. Rosenthal. 1997. *Congress: A Political-Economic History of Roll Call Voting*. Oxford University Press.
- Poole, K.T. 2005. *Spatial Models of Parliamentary Voting*. Cambridge University Press.

### Optimal Classification: The Nonparametric Alternative

Keith Poole also developed **Optimal Classification (OC)**, a nonparametric method that makes no assumptions about utility functions or error distributions. Instead, it finds cutting lines (hyperplanes) that best separate Yea from Nay voters, and simultaneously optimizes legislator positions to maximize total correct classifications.

OC typically achieves higher classification accuracy than NOMINATE (since it makes fewer assumptions) but cannot produce standard errors or probabilistic predictions. It is best used when you want to make minimal distributional assumptions or when you suspect NOMINATE's parametric assumptions are violated.

**Reference:** Poole, K.T. 2000. "Nonparametric Unfolding of Binary Choice Data." *Political Analysis* 8(3): 211-237.

### PCA/SVD: The Quick Baseline

Principal Component Analysis on the legislator-by-vote binary matrix is the fastest approach. Heckman and Snyder (1997) showed that SVD of the agreement matrix yields ideal points highly correlated with NOMINATE scores (typically r > 0.95 for the first dimension). PCA requires imputation for missing data and produces no uncertainty estimates, but it runs in seconds rather than minutes or hours.

PCA is best used as a **sanity check** before investing in more computationally expensive methods — which is how our pipeline uses it (Phase 2, before IRT in Phase 4).

**Reference:** Heckman, J.J. and J.M. Snyder Jr. 1997. "Linear Probability Models of the Demand for Attributes." *RAND Journal of Economics* 28(S): S142-S189.

---

## The Bayesian Revolution

### Clinton-Jackman-Rivers: The Foundation

The landmark paper that brought Bayesian methods to legislative analysis:

> Clinton, J., S. Jackman, and D. Rivers. 2004. "The Statistical Analysis of Roll Call Data." *American Political Science Review* 98(2): 355-370.

They developed a two-parameter IRT (Item Response Theory) model for roll call voting, estimated via MCMC (Gibbs sampling with data augmentation). The model treats each legislator's ideal point as a latent variable with a prior distribution, and each vote's difficulty and discrimination as item parameters — borrowing the framework from educational testing (where students answer test questions) and applying it to legislators voting on bills.

The key advantages over NOMINATE:
- **Full posterior distributions** — not just point estimates, but credible intervals for every ideal point
- **Natural handling of missing data** — the Bayesian framework treats absences as missing at random
- **Flexible priors** — informative priors can encode substantive knowledge
- **Hierarchical extensions** — partial pooling across groups, time periods, or chambers

Simon Jackman's textbook *Bayesian Analysis for the Social Sciences* (Wiley, 2009) provides the definitive treatment.

### Martin-Quinn Scores: Dynamic Ideal Points

Andrew D. Martin and Kevin M. Quinn extended Bayesian IRT to allow ideal points to change over time via a random walk prior. Their **Martin-Quinn scores** are the standard measure of Supreme Court justice ideology, updated annually at [mqscores.wustl.edu](https://mqscores.wustl.edu/).

**Reference:** Martin, A.D. and K.M. Quinn. 2002. "Dynamic Ideal Point Estimation via Markov Chain Monte Carlo for the U.S. Supreme Court, 1953-1999." *Political Analysis* 10(2).

### emIRT: Speed Without Sacrifice

Kosuke Imai, James Lo, and Jonathan Olmsted developed EM-based algorithms that produce estimates essentially identical to MCMC but within minutes instead of hours:

> Imai, K., J. Lo, and J. Olmsted. 2016. "Fast Estimation of Ideal Points with Massive Data." *American Political Science Review* 110(4): 631-656.

Their **emIRT** R package includes `binIRT` (binary), `dynIRT` (dynamic), `hierIRT` (hierarchical), and `ordIRT` (ordinal) models. It is 84% C++ code and runs orders of magnitude faster than MCMC. For large-scale work (all state legislatures, multi-decade panels), emIRT's speed advantage is decisive.

### The Cutting Edge (2024-2025)

Recent methodological advances:

**Issue-Specific IRT.** Sooahn Shin (2024) developed **IssueIRT**, a hierarchical IRT model that estimates separate ideal point axes per policy area (monetary policy, social issues, etc.) using issue labels on bills. Demonstrated that legislators may shift on one dimension while remaining stable on another. R package: [github.com/sooahnshin/issueirt](https://github.com/sooahnshin/issueirt).

**L1 Ideal Points.** Shin, Lim, and Park (2025) solved the rotational invariance problem in multidimensional ideal point estimation using L1 distance instead of L2, published in *Journal of the American Statistical Association* 120(550).

**Cross-Domain Hierarchical Models.** Lipman, Moser, and Rodriguez (2025) extended Bayesian spatial voting models with a hierarchical structure that jointly estimates ideal points across two voting domains (e.g., procedural vs. final passage votes). Published in *Political Analysis*. This is structurally similar to our hierarchical IRT's partial pooling by party.

**Generalized Ideal Points.** Robert Kubinec's **idealstan** R package (built on Stan) handles binary, ordinal, count, and continuous responses, plus strategic missing data via a hurdle model — recognizing that legislative absences are often informative, not random.

**Unfolding Models.** Duck-Mayr and Montgomery developed the Bayesian Generalized Graded Unfolding Model (GGUM), which handles cases where legislators at both extremes vote together against the center. The standard IRT assumption of monotonic response functions fails for single-peaked preferences, and the GGUM accommodates this. R package: **bggum** on CRAN.

---

## The Software Ecosystem

### R Dominates

The field is overwhelmingly R-based. The major packages:

| Package | Authors | Method | Speed |
|---------|---------|--------|-------|
| **pscl** | Jackman | Bayesian IRT (Gibbs) | Slow |
| **MCMCpack** | Martin, Quinn, Park | Bayesian IRT (Gibbs) | Slow |
| **emIRT** | Imai, Lo, Olmsted | EM algorithm | Very fast |
| **wnominate** | Poole, Lewis, Lo, Carroll | W-NOMINATE (MLE) | Medium |
| **oc** | Poole, Lewis, Lo, Carroll | Optimal Classification | Medium |
| **idealstan** | Kubinec | Bayesian IRT (Stan/HMC) | Moderate |
| **dgo** | Dunham, Caughey, Warshaw | Hierarchical group-level IRT | Slow |
| **issueirt** | Shin | Issue-specific hierarchical IRT | Moderate |
| **anominate** | Lo, Poole, Carroll | alpha-NOMINATE (Bayesian) | Moderate |

### Python: A Distant Second

Python is far behind R in this specific domain:

- **PyMC** — You can implement Clinton-Jackman-Rivers directly (as our project does), but there is no pre-built legislative ideal point module.
- **py-irt** (Lalor & Rodriguez, 2023) — Scalable IRT on Pyro/PyTorch with GPU acceleration. Built for educational testing but applicable to ideal points. Published in *INFORMS Journal on Computing*.
- **pynominate** (Boche, Lewis, Sonnet) — Python implementation of DW-NOMINATE from the Voteview team. Partial equivalent of the R package.
- **PyStan** — The Python interface to Stan, enabling any Stan model (including Jeffrey Arnold's legislator examples) to run from Python.

There is no Python equivalent of `pscl`, `MCMCpack`, or `emIRT`. If you are working in Python, you are implementing models from scratch — which is what our project does.

---

## Machine Learning and Deep Learning

### Vote Prediction

Tree-based methods (XGBoost, Random Forests) achieve high accuracy (often 90%+) for predicting individual votes given legislator and bill features, but they don't produce interpretable ideological dimensions the way NOMINATE or IRT do. SHAP values provide post-hoc interpretation. This is the approach our pipeline takes in `analysis/prediction.py`.

### Text-Based Scaling

Several approaches use legislative text rather than (or alongside) votes:

- **Wordfish** (Slapin & Proksch, 2008) — Unsupervised Poisson scaling of word frequencies. R only.
- **Text-Based Ideal Points (TBIP)** (Vafa, Naidu & Blei, 2020) — Unsupervised probabilistic topic model that estimates ideal points from text alone. Produces ideal points correlated with vote-based NOMINATE scores. Code: [github.com/keyonvafa/tbip](https://github.com/keyonvafa/tbip).
- **LLM-based positioning** (2024-2025) — Instruction-tuned LLMs can position political texts on ideological dimensions with r > 0.90 correlation to expert-coded benchmarks. Published in *Political Analysis*, 2025.

### LLM Agents for Legislative Simulation

The **Political Actor Agent** (Li, Gong & Jiang, AAAI 2025) uses LLM-based agents to simulate legislative actors and predict roll call votes. Each agent has a profile built from historical voting records and simulates leadership influence hierarchies. Still experimental, but it represents the frontier where NLP meets legislative analysis.

---

## Network Analysis

### Co-Voting Networks

The landmark application of network science to legislative polarization:

> Waugh, Pei, Fowler, and Mucha. 2009. "Party Polarization in Congress: A Network Science Approach." arXiv:0907.3509.

They used **modularity** from community detection to measure polarization without assuming party structure, tracing it back to the 1st Congress. Network methods complement ideal point estimation by revealing coalition structure, bridge actors, and influence patterns that unidimensional scaling cannot capture.

**James Fowler** (2006) applied cosponsorship networks to 280,000 pieces of legislation, showing that network centrality predicts amendment passage and vote choice even after controlling for ideology. Published in *Political Analysis* 14(4).

Our pipeline's network phase (`analysis/network.py`) uses Louvain community detection and betweenness centrality on co-voting graphs — standard methods in this literature.

---

## State Legislature Analysis: The Gap

### The Shor-McCarty Project

The definitive state-level work:

> Shor, B. and N. McCarty. 2011. "The Ideological Mapping of American Legislatures." *American Political Science Review* 105(3): 530-551.

They estimated ideal points for 27,629 state legislators across all 50 states (1993-2020) using **bridge observations** — legislators who also responded to Project Vote Smart's survey, providing a common set of items across states. Data at [americanlegislatures.com](https://americanlegislatures.com/) and [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NWSYOS), last updated April 2023.

### The DIME/CFscores Alternative

Adam Bonica's **Database on Ideology, Money in Politics, and Elections** (Stanford) estimates ideology from campaign finance records rather than votes. Donors who give to both state and federal candidates create a common scale across institutions and levels of government. Covers candidates who never serve (no roll calls needed). Data at [data.stanford.edu/dime](https://data.stanford.edu/dime).

**Reference:** Bonica, A. 2014. "Mapping the Ideological Marketplace." *American Journal of Political Science* 58(2): 367-386.

### The Data Access Problem

State legislatures present unique challenges:
- **No common scale** — each state votes on different bills, so ideal points aren't directly comparable without bridging
- **Data availability** — roll call votes are not centrally archived for most states
- **Smaller chambers** — fewer legislators means more estimation uncertainty (Kansas House = 125 seats)
- **Supermajorities** — lopsided party ratios make intra-party variation more interesting than inter-party (Kansas is ~72% Republican)

The major data providers for state legislatures:

| Source | Coverage | Type |
|--------|----------|------|
| **Open States** ([openstates.org](https://openstates.org)) | 50 states + DC + PR | Scrapers, API, bulk data |
| **LegiScan** ([legiscan.com](https://legiscan.com)) | 50 states + Congress | API, weekly snapshots |
| **Shor-McCarty** | 50 states (1993-2020) | Ideology scores |
| **DIME/CFscores** | All levels | Campaign-finance ideology |
| **BillTrack50** ([billtrack50.com](https://www.billtrack50.com)) | 50 states + Congress | Bill tracking |

None of these provide **analytical pipelines** — they offer raw data or pre-computed scores, not the tools to go from roll call records to statistical conclusions.

---

## Party Discipline and Loyalty

### Classical Measures

- **Rice Index** (Stuart Rice, 1925) — |% Yea − % Nay| within a party. The oldest party unity measure.
- **CQ Party Unity Scores** — Congressional Quarterly's annual scores, computed since 1953. The journalism standard.
- **WELDON Index** — Accounts for abstentions, unlike Rice.

### Bayesian Approaches

Most party loyalty analysis uses frequentist counts (CQ scores) or simple proportions. Our pipeline's **Beta-Binomial empirical Bayes** approach (`analysis/beta_binomial.py`) applies Bayesian shrinkage to party unity scores — pulling legislators with few party-line votes toward the party average, which produces more reliable estimates for low-participation members. The **hierarchical IRT** (`analysis/hierarchical.py`) with partial pooling by party is the state-of-the-art Bayesian approach to modeling intra-party variation.

### Intra-Party Factions

Recent research has moved beyond the party-vs-party frame to study **factions within parties**:

- Spirling and Quinn (2010) used Dirichlet process mixture models to discover voting blocs within UK parties (*JASA*).
- Clarke, Volden, and Wiseman (2024) studied the nine largest ideological caucuses in Congress and their effect on legislative effectiveness (*Political Research Quarterly*).
- Kolln and Polk (2024) used cluster analysis to identify ideological and hierarchical factions (*Comparative Political Studies*).

---

## Cross-Institutional and Temporal Comparison

### Making Scores Comparable

The fundamental problem: if Kansas and California vote on different bills, how do you compare their legislators? The field has developed several bridging strategies:

- **Survey bridges** — Shor-McCarty use NPAT survey responses answered by legislators in multiple states
- **Legislator bridges** — Members who serve in both chambers or move between state and federal office
- **Campaign finance bridges** — Donors who give to candidates in multiple jurisdictions (DIME/CFscores)
- **Affine alignment** — Our cross-session validation uses returning members as bridge actors and applies an affine transform to align IRT scales across bienniums

Michael Bailey's **Bridge Ideal Points** (Georgetown) represent the gold standard for cross-institutional comparison, linking Congress, the President, and the Supreme Court using 42,415+ bridging observations from amicus briefs and position-taking. Data at [michaelbailey.georgetown.domains](https://michaelbailey.georgetown.domains/bridge-ideal-points-2020/).

### Dynamic Models

Methods for tracking ideological change over time:

- **DW-NOMINATE** — Assumes legislators occupy a fixed position across their career (strong assumption)
- **Nokken-Poole scores** — Hold DW-NOMINATE cutting lines fixed, re-estimate ideal points per Congress
- **Martin-Quinn scores** — Random walk prior allows term-by-term change (Supreme Court)
- **emIRT dynIRT** — Fast EM-based dynamic ideal points
- **Penalized splines** — Lewis and Sonnet developed smooth temporal trends for NOMINATE

---

## Journalism and Public-Facing Tools

### The Federal Focus

Major data journalism tools are almost exclusively federal:

- **Voteview** ([voteview.com](https://voteview.com)) — The academic gold standard, live-updated DW-NOMINATE scores
- **GovTrack** ([govtrack.us](https://www.govtrack.us)) — Bill tracking, annual report cards, ideology statistics. Founded by Joshua Tauberer in 2004.
- **FiveThirtyEight** — Congressional voting analysis (Trump Score, Biden Score). Now part of ABC News.
- **ProPublica Represent** — Previously the most important public tool. **Now discontinued.**

### The State-Level Gap

State-level legislative data journalism is sparse. **Plural Policy** ([pluralpolicy.com](https://pluralpolicy.com)) acquired Open States in 2020 and provides bill tracking across 50 states with AI-powered summaries — but no statistical analysis of voting patterns. The same is true of BillTrack50 and FastDemocracy: tracking, not analysis.

The state legislature voting analysis space for journalism is essentially vacant.

### International

- **TheyWorkForYou** ([theyworkforyou.com](https://www.theyworkforyou.com)) — mySociety's open-source UK parliamentary monitoring platform. In 2025 launched a dedicated vote analysis platform with party alignment metrics. Covers Westminster, Scottish Parliament, Senedd, and Northern Ireland Assembly.
- **HowTheyVote.eu** ([howtheyvote.eu](https://howtheyvote.eu)) — Open-source European Parliament roll call vote tracker, weekly-updated. Funded by the German Federal Ministry of Education and Research.

Standard scaling methods (NOMINATE, IRT) perform poorly on the UK House of Commons due to extreme party discipline. Spirling (2014) showed that Optimal Classification produces misleading orderings for Westminster. Researchers have turned to alternative data sources — Early Day Motions (non-binding motions MPs can sign) and social media followership — to estimate British ideal points.

---

## Where Our Pipeline Fits

### What We're Doing

Our Kansas Legislature pipeline is a 13-phase analytical system that goes from raw HTML on kslegislature.gov to narrative synthesis reports:

1. **Scraping** — Custom Python scraper covering 2011-2026 (8 bienniums, HTML and ODT vote formats)
2. **EDA** — Descriptive statistics, filtering, agreement matrices
3. **PCA** — Fast ideological baseline
4. **UMAP** — Nonlinear dimensionality reduction
5. **Bayesian IRT** — 2PL ideal points with PCA-informed initialization (PyMC)
6. **Clustering** — K-means, hierarchical, GMM (k=2 finding: party is the dominant structure)
7. **Network** — Co-voting graphs, Louvain community detection, betweenness centrality
8. **Prediction** — XGBoost vote prediction with SHAP interpretation and NMF topic features
9. **Classical Indices** — Rice, party unity, CQ-style scores
10. **Beta-Binomial** — Empirical Bayes shrinkage on party loyalty
11. **Hierarchical IRT** — 2-level partial pooling by party, optional 3-level joint cross-chamber model
12. **Synthesis** — Data-driven detection of notable legislators, narrative report
13. **Profiles** — Deep-dive per-legislator analysis

Plus a **cross-session validation** module for comparing bienniums.

### What Makes It Unusual

Surveying the field, several aspects of our pipeline are uncommon or unique:

**Single-state, full-pipeline, Python-native.** The academic world has mature R-based tools for individual analytical steps (NOMINATE, pscl, emIRT) and data providers for raw data (Voteview, Open States, LegiScan). But an integrated pipeline from HTML scraping through Bayesian analysis to narrative synthesis reports does not exist in the open-source landscape. The closest analogs are HowTheyVote.eu (EU Parliament, scraping + display, no statistical analysis) and TheyWorkForYou (UK Parliament, similar monitoring scope but no Bayesian modeling).

**Python over R.** We implement the Clinton-Jackman-Rivers IRT model directly in PyMC rather than using `pscl` or `MCMCpack` in R. This is a minority approach — the field overwhelmingly uses R. Our choice is driven by ecosystem coherence (Polars, scikit-learn, XGBoost, NetworkX all in one language) rather than methodological preference.

**PCA-informed chain initialization.** Our IRT uses standardized PC1 scores to seed both MCMC chains' ideal point parameters, preventing the reflection mode-splitting that caused 5 of 16 historical chamber-sessions to fail convergence. This addresses the same identification problem the literature solves with informative priors or ordering constraints, but via initialization rather than the prior.

**Hierarchical IRT with joint cross-chamber model.** Our 3-level cross-chamber model (chamber → party → legislator) with non-centered parameterization is structurally similar to Lipman et al. (2025) but pools by party rather than by voting domain. The 91st Legislature's joint model (172 legislators, 491 votes, 43,612 observations) converges cleanly with 0 divergences in ~93 minutes on Apple Silicon.

**Empirical Bayes for party loyalty.** The Beta-Binomial shrinkage on CQ-style party unity scores is a principled alternative to raw proportions, pulling low-participation legislators toward their party average. Most published work reports raw CQ scores without shrinkage.

**State legislature supermajority analysis.** Kansas's ~72% Republican legislature makes intra-party variation more analytically interesting than the inter-party polarization that dominates the congressional literature. Our clustering analysis confirmed this (k=2 optimal, mapping to party; the initial k=3 intra-Republican hypothesis was rejected — the intra-party variation is continuous, not factional).

### Methods We Don't Use (and Why)

| Method | Why Not |
|--------|---------|
| W-NOMINATE | R-only; PCA + Bayesian IRT covers the same ground in Python |
| Optimal Classification | R-only; no Python implementation exists |
| DW-NOMINATE | Designed for cross-Congress comparison; we use affine alignment instead |
| emIRT | R-only; PyMC is fast enough for single-state analysis |
| Text-Based Ideal Points (TBIP) | Full bill text not available from our scraper; `short_title` NMF features are a lightweight substitute |
| Dynamic IRT (random walk) | Bienniums are analyzed independently with cross-session validation; within-biennium dynamics are minimal for a 2-year window |

### External Validation Opportunities

The Shor-McCarty dataset includes Kansas-specific scores (1993-2020) that could serve as an external benchmark for our IRT ideal points. DIME/CFscores provide an independent campaign-finance-based ideology estimate. Neither has been systematically compared to our results yet.

---

## References

### Foundational Papers

- Clinton, J., S. Jackman, and D. Rivers. 2004. "The Statistical Analysis of Roll Call Data." *APSR* 98(2): 355-370.
- Poole, K.T. and H. Rosenthal. 1985. "A Spatial Model for Legislative Roll Call Analysis." *AJPS* 29(2): 357-384.
- Poole, K.T. and H. Rosenthal. 1997. *Congress: A Political-Economic History of Roll Call Voting*. Oxford.
- Poole, K.T. 2005. *Spatial Models of Parliamentary Voting*. Cambridge.
- Jackman, S. 2001. "Multidimensional Analysis of Roll Call Data via Bayesian Simulation." *Political Analysis* 9(3).
- Jackman, S. 2009. *Bayesian Analysis for the Social Sciences*. Wiley.
- Martin, A.D. and K.M. Quinn. 2002. "Dynamic Ideal Point Estimation via MCMC for the U.S. Supreme Court." *Political Analysis* 10(2).

### Scalability and Modern Methods

- Imai, K., J. Lo, and J. Olmsted. 2016. "Fast Estimation of Ideal Points with Massive Data." *APSR* 110(4): 631-656.
- Lalor, J. and P. Rodriguez. 2023. "py-irt: A Scalable Item Response Theory Library." *INFORMS Journal on Computing* 35(1).
- Shin, S. 2025. "L1-based Bayesian Ideal Point Model for Multidimensional Politics." *JASA* 120(550).
- Lipman, M., S. Moser, and A. Rodriguez. 2025. "Explaining Differences in Voting Patterns Across Voting Domains Using Hierarchical Bayesian Models." *Political Analysis*.

### State-Level Analysis

- Shor, B. and N. McCarty. 2011. "The Ideological Mapping of American Legislatures." *APSR* 105(3): 530-551.
- Bonica, A. 2014. "Mapping the Ideological Marketplace." *AJPS* 58(2): 367-386.
- Handan-Nader, C., A. Myers, and A.B. Hall. 2025. "Polarization and State Legislative Elections." *AJPS* (accepted January 2025).
- Caughey, D. and C. Warshaw. 2015. "Dynamic Estimation of Latent Opinion Using a Hierarchical Group-Level IRT Model." *Political Analysis* 23(2).

### Network Analysis

- Waugh, A., L. Pei, J. Fowler, and P. Mucha. 2009. "Party Polarization in Congress: A Network Science Approach." arXiv:0907.3509.
- Fowler, J. 2006. "Connecting the Congress: A Study of Cosponsorship Networks." *Political Analysis* 14(4): 456-487.
- Ringe, N. and S. Wilson. 2016. "Pinpointing the Powerful: Covoting Network Centrality as a Measure of Political Influence." *Legislative Studies Quarterly* 41(3): 739-769.

### Text and ML

- Vafa, K., S. Naidu, and D. Blei. 2020. "Text-Based Ideal Points." ACL 2020.
- Gerrish, S. and D. Blei. 2011. "Predicting Legislative Roll Calls from Text." ICML 2011.
- Li, R., Y. Gong, and G. Jiang. 2025. "Political Actor Agent: Simulating Legislative System for Roll Call Votes Prediction with Large Language Models." AAAI 2025.

### Party Discipline

- Rice, S. 1925. "The Behavior of Legislative Groups." *Political Science Quarterly* 40(1).
- Spirling, A. and K. Quinn. 2010. "Identifying Intra-Party Voting Blocs in the UK House of Commons." *JASA*.
- Clarke, A., C. Volden, and A. Wiseman. 2024. "Ideological Caucuses in Congress." *Political Research Quarterly*.

### International

- Spirling, A. 2014. "UK OC OK? Interpreting Optimal Classification Scores for the UK House of Commons." *Political Analysis*.
- Kellermann, M. 2012. "Estimating Ideal Points in the British House of Commons Using Early Day Motions." *AJPS*.

### Software and Data

- Voteview: [voteview.com](https://voteview.com)
- Shor-McCarty scores: [americanlegislatures.com](https://americanlegislatures.com)
- Martin-Quinn scores: [mqscores.wustl.edu](https://mqscores.wustl.edu)
- Bailey Bridge Ideal Points: [michaelbailey.georgetown.domains](https://michaelbailey.georgetown.domains/bridge-ideal-points-2020/)
- Open States: [openstates.org](https://openstates.org)
- LegiScan: [legiscan.com](https://legiscan.com)
- DIME/CFscores: [data.stanford.edu/dime](https://data.stanford.edu/dime)
- pscl (R): [CRAN](https://cran.r-project.org/web/packages/pscl/)
- MCMCpack (R): [CRAN](https://cran.r-project.org/web/packages/MCMCpack/)
- emIRT (R): [CRAN](https://cran.r-project.org/web/packages/emIRT/) / [GitHub](https://github.com/kosukeimai/emIRT)
- wnominate (R): [CRAN](https://cran.r-project.org/web/packages/wnominate/)
- idealstan (R/Stan): [GitHub](https://github.com/saudiwin/idealstan)
- py-irt (Python): [PyPI](https://pypi.org/project/py-irt/)
- TBIP (Python): [GitHub](https://github.com/keyonvafa/tbip)
- HowTheyVote.eu: [howtheyvote.eu](https://howtheyvote.eu)
- TheyWorkForYou: [theyworkforyou.com](https://www.theyworkforyou.com)
