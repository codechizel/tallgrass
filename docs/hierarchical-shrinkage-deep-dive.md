# The Small-Chamber Problem: Why Hierarchical Models Struggle with 11 Democrats

**A deep dive into over-shrinkage, the two-group problem, and what the statistics literature says about fixing it.**

*2026-02-24*

---

## Background: What Happened

Our [external validation results](external-validation-results.md) showed something striking. When we compared our ideology scores against the Shor-McCarty gold standard for the 88th Kansas Legislature (2019-2020), three of four model/chamber combinations validated beautifully:

| Model | Chamber | Pearson r | n | Quality |
|-------|---------|-----------|---|---------|
| Flat IRT | House | 0.981 | 127 | Strong |
| Flat IRT | Senate | 0.929 | 41 | Strong |
| Hierarchical IRT | House | 0.984 | 127 | Strong |
| **Hierarchical IRT** | **Senate** | **-0.541** | **41** | **Concern** |

The hierarchical model's Senate scores were not just wrong — they were *inverted*. A negative correlation means the model ranked legislators backwards: people it called liberal, the gold standard called conservative, and vice versa.

The flat model — the simpler one, which treats each legislator independently — worked perfectly. So this isn't a data problem. It's a model problem, and a well-documented one. This article explains what went wrong, why it was predictable, and what can be done about it.

---

## Part 1: What Hierarchical Models Do (and Why We Use Them)

### The Basic Idea

Imagine you're a teacher with two classes of students, and you want to estimate each student's ability from a test. You have two choices:

**Option A (Flat model):** Grade each student independently. Each test score stands on its own. A student who happened to have a bad day gets a low score, period.

**Option B (Hierarchical model):** Recognize that students within the same class share a common experience — same teacher, same curriculum, same classroom. Use that shared structure to "borrow strength" across students. If one student's score seems anomalously low compared to their classmates, nudge it slightly upward, because the class average provides information about what we'd expect.

That nudging is called **shrinkage**, and it's the whole point of hierarchical modeling. It produces better estimates on average because it uses *all available information*, not just each individual's data. In our case, the "classes" are political parties, and the "students" are legislators.

### Our Model, Specifically

Our hierarchical IRT model works like this:

1. Each **party** (Democrat, Republican) has an average ideology score (`mu_party`)
2. Each party has a measure of internal spread (`sigma_within`) — how diverse the party is
3. Each **legislator's** ideology score is drawn from their party's distribution:

```
legislator_score = party_mean + party_spread × individual_offset
```

The individual offset captures what makes each legislator unique beyond their party label. A moderate Republican has a small negative offset (they're slightly left of their party's mean). A hardline Republican has a large positive offset.

This is a good model for a legislature. Party is the single strongest predictor of voting behavior — it explains about 89% of the variance in House voting. The hierarchical model makes that structure explicit.

### Where the Shrinkage Happens

When the model estimates a legislator's score, it balances two sources of information:

- **The legislator's own voting record** — what they actually did
- **Their party's average** — what legislators like them tend to do

When the voting record is strong (many votes, clear pattern), the model mostly trusts the individual data. When the voting record is noisy or sparse, the model leans more on the party average. This is shrinkage: pulling uncertain estimates toward the group mean.

For the Kansas House, with 85 Republicans and 42 Democrats, this works beautifully. There's plenty of data to estimate both the party averages and the individual deviations.

For the Kansas Senate, with 30 Republicans and 11 Democrats, it breaks down.

---

## Part 2: What Went Wrong in the Senate

### The Diagnostics

Before even looking at the external validation, the model itself was screaming that something was wrong. The convergence diagnostics for the Senate hierarchical model:

| Diagnostic | Value | Healthy Range | Verdict |
|-----------|-------|---------------|---------|
| R-hat (ideal points) | 1.83 | < 1.01 | Catastrophic failure |
| R-hat (party means) | 1.83 | < 1.01 | Catastrophic failure |
| Effective sample size | 3 | > 400 | Catastrophic failure |
| ICC (party explains) | 0.41 | 0.85-0.95 expected | Prior-dominated |
| ICC uncertainty | [0.00, 0.88] | Should be narrow | Completely uninformative |

**R-hat** measures whether the MCMC sampling chains agree with each other. A value of 1.83 means the two chains explored completely different regions of the parameter space — they never converged on the same answer. An **effective sample size** of 3 (out of 4,000 draws) means the sampler was stuck, producing nearly identical values over and over.

The **ICC** (intraclass correlation) measures how much of the ideological variance is explained by party. In the House, the ICC is 0.89 with a narrow uncertainty band — party clearly matters. In the Senate, the ICC is 0.41 with an interval from 0.00 to 0.88 — the model has no idea whether party matters a lot or barely at all.

### The Root Cause: Not Enough Democrats

The hierarchical model needs to estimate three things per party:

1. The party's average ideology (1 number)
2. The party's internal diversity (1 number)
3. Each legislator's individual offset (1 number per person)

For Senate Democrats, that's: 1 mean + 1 spread + 11 individual offsets = **13 parameters from 11 people**.

It's like trying to characterize a "typical" family's eating habits by observing a single dinner party of 11 guests. You'd get one snapshot, but you couldn't reliably estimate both the average *and* the variety — those two quantities would be entangled in the noise.

Technically, each legislator has voted on roughly 124 roll calls, so the total data isn't tiny (11 legislators × 124 votes ≈ 1,300 observations). But the problem is that those 1,300 observations need to simultaneously inform the group-level parameters (party mean, party spread) and the individual-level parameters (11 offsets), and there isn't enough information to go around.

### What the Sampler Sees: Two Contradictory Explanations

When the MCMC sampler tries to explore the Senate Democrat parameter space, it encounters a fundamental ambiguity. The same voting patterns can be explained in two completely different ways:

**Explanation A:** The party spread is small (Democrats are ideologically similar), and each legislator has a large individual offset to explain their specific votes.

**Explanation B:** The party spread is large (Democrats are ideologically diverse), and each legislator's offset is small because the party distribution already covers the range.

Both explanations fit the data equally well. With only 11 Democrats, there isn't enough information to distinguish between them. The sampler oscillates between these two modes — sometimes favoring Explanation A, sometimes Explanation B — and never settles. This is what produces the R-hat of 1.83 and the effective sample size of 3.

In the House, 42 Democrats provide enough data to clearly distinguish between these explanations. The model confidently identifies the party mean, the party spread, and each individual's position. In the Senate, it can't.

---

## Part 3: The Two-Group Problem

### It Gets Worse: J = 2

The small-sample problem is compounded by something deeper: we only have **two parties**. In statistics, the number of groups is conventionally called J. Our model has J = 2.

This matters because the hierarchical model is trying to estimate not just the party means, but also the *variability between parties*. How different are Democrats and Republicans? That between-group variance is a crucial parameter that controls the strength of shrinkage.

With J = 2, you are estimating a variance from two data points. The party means. That's it. Two numbers to estimate a spread. Any introductory statistics course teaches that you need at least 3 data points to get a meaningful variance estimate — and even then it's unstable.

This isn't an edge case that statisticians recently discovered. It traces back to a foundational result from 1955.

### The James-Stein Threshold

In 1955, Charles Stein proved a startling theorem: when you have J = 3 or more groups, shrinkage estimators *always* produce better estimates on average than treating each group independently. Always. This is called the "Stein paradox" because it's unintuitive — how can biasing each estimate toward the average possibly help?

The answer is that the bias you introduce (pulling toward the mean) is more than offset by the reduction in noise, *but only when J >= 3*. With J = 2, this guarantee vanishes. Shrinkage might help, or it might make things worse. You don't know in advance, and the data can't tell you.

James and Stein formalized this in 1960, and it remains one of the most important results in statistical theory. It's the reason that hierarchical models shine with many groups (schools within districts, patients within hospitals, players within teams) but can stumble with very few.

**Reference:** James, W. and Stein, C. (1960). "Estimation with Quadratic Loss." *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 361-379.

---

## Part 4: What the Literature Says

The J = 2 problem in hierarchical models has been discussed extensively. Here are the key contributions, in chronological order.

### Gelman (2006): "Prior Distributions for Variance Parameters"

This landmark paper in *Bayesian Analysis* addressed the question of what prior to put on the group-level variance (sigma or tau) in hierarchical models. Andrew Gelman showed that the once-standard inverse-gamma prior (popular in textbooks at the time) is pathological for small J: it either collapses the variance to zero (producing extreme over-shrinkage) or inflates it to infinity (producing no shrinkage at all), depending on how you set the parameters.

His recommendation: use a **half-Cauchy** distribution as the prior on the group-level standard deviation. The half-Cauchy has a broad peak near zero but heavy tails, allowing the data to push the variance wherever it needs to go. This became the standard recommendation for hierarchical models for nearly a decade.

**Reference:** Gelman, A. (2006). "Prior distributions for variance parameters in hierarchical models." *Bayesian Analysis*, 1(3), 515-534.

### Gelman (2015): "Hierarchical Modeling When You Have Only 2 Groups"

In a blog post that became widely cited, Gelman directly addressed the J = 2 case:

> "If you simply put the default uniform prior on the group-level scale parameter, all the posterior mass will be at tau = infinity, and you'll do no pooling at all."

His recommendation for J = 2 is blunt: **don't try to estimate the between-group variance from the data.** Instead, either:

1. **Fix it** to a plausible value based on domain knowledge, or
2. Use a **strongly informative prior** that encodes what you know about how different the groups should be

This is a pragmatic concession. The hierarchical model is still useful for J = 2 — the within-group partial pooling still helps — but you must supply the between-group structure yourself rather than asking the data to estimate it.

**Reference:** Gelman, A. (2015). "Hierarchical modeling when you have only 2 groups." *Statistical Modeling, Causal Inference, and Social Science* (blog), December 8, 2015.

### Papaspiliopoulos, Roberts, and Skoeld (2007): Centered vs. Non-Centered

Our model uses a "non-centered parameterization" — a technical rearrangement of the model that helps the MCMC sampler navigate the difficult geometry of hierarchical posteriors. This paper provides the theory:

- **Non-centered** parameterization works best when data per group is sparse (the model struggles to pin down individual values)
- **Centered** parameterization works best when data per group is abundant

Our Senate (sparse data for the Democrat group) is exactly the case where non-centered parameterization is appropriate — and we use it. But the parameterization helps with the *geometry* of the posterior, not the *identifiability* of the parameters. No amount of clever sampling can rescue a model that doesn't have enough data to separate its parameters.

**Reference:** Papaspiliopoulos, O., Roberts, G. O., and Skoeld, M. (2007). "A General Framework for the Parametrization of Hierarchical Models." *Statistical Science*, 22(1), 59-73.

### Peress (2009): "Small Chamber Ideal Point Estimation"

This paper directly addresses the problem we encountered — estimating ideology scores for small legislative chambers. Michael Peress applied the analysis to the U.S. Supreme Court, which has only 9 justices.

His key insight: standard IRT estimators suffer from the **incidental parameters problem** in small chambers. Each legislator's ideal point is a parameter that must be estimated, and as the number of legislators shrinks relative to the number of votes, these estimates become inconsistent — they don't improve even as you add more roll calls.

Peress developed an estimator specifically designed for small chambers that avoids this problem. His work confirms that standard approaches (both maximum likelihood and basic Bayesian IRT) can fail with small N, and that specialized techniques are needed.

**Reference:** Peress, M. (2009). "Small Chamber Ideal Point Estimation." *Political Analysis*, 17(3), 276-290.

### Betancourt and Girolami (2013): The Funnel of Hell

Michael Betancourt (one of the core developers of the Stan probabilistic programming language) described the "funnel geometry" that plagues hierarchical models when the group-level variance is poorly constrained.

Picture a funnel: at the narrow end, the group-level variance is near zero, and all individual parameters are squeezed together. At the wide end, the variance is large, and individuals spread out. The MCMC sampler must smoothly transition between these extremes. When the data strongly constrains the variance, this funnel is gentle. When the data barely constrains it (as with J = 2 or small group sizes), the funnel is extreme — a narrow throat connected to a cavernous opening — and the sampler gets stuck.

This is exactly what we see in the Senate model: the sampler oscillates between two modes (tight funnel throat vs. wide opening) and never converges.

**Reference:** Betancourt, M. and Girolami, M. (2013). "Hamiltonian Monte Carlo for Hierarchical Models." arXiv:1312.0906.

### The Folk Theorem of Statistical Computing

Gelman has a saying that he calls the "folk theorem of statistical computing":

> "When you have computational problems, often there's a problem with your model."

An R-hat of 1.83 and an effective sample size of 3 are not sampler bugs. They're the sampler faithfully reporting that the posterior has no single well-defined peak. Throwing more computing power at the problem — more chains, more tuning, longer runs — won't help. The model itself needs to change.

**Reference:** Gelman, A. (2008). "The folk theorem of statistical computing." *Statistical Modeling, Causal Inference, and Social Science* (blog), May 13, 2008.

### Bailey (2001): Random Effects with Covariates

Michael Bailey tackled a related problem — estimating ideal points when individual legislators have few votes. His solution: use a random-effects model with covariates (like party) to bring additional information into the estimation. This is conceptually similar to our hierarchical approach, and his finding that covariate-enriched models outperform flat models for sparse data aligns with our House results.

But his work also implicitly acknowledges the same limitation: when groups are small, the hierarchical structure can only help if the group-level parameters are well-estimated.

**Reference:** Bailey, M. (2001). "Ideal Point Estimation with a Small Number of Votes: A Random-Effects Approach." *Political Analysis*, 9(3), 192-210.

### The Stan Prior Choice Recommendations (Current)

The Stan development team maintains a living document on prior choice. Their current guidance has moved away from Gelman's 2006 half-Cauchy recommendation for small-J settings:

> "If the number of groups is small, the data don't provide much information on the group-level variance, and so it can make sense to use stronger prior information."

Their current default recommendation is **half-normal(0, 1)** — tighter and more regularizing than the half-Cauchy. For very small J, they suggest going further: use domain knowledge to set the scale.

**Reference:** Stan Development Team. "Prior Choice Recommendations." GitHub wiki, stan-dev/stan (ongoing).

---

## Part 5: The Paradox — Our Model Actually Handles J = 2 Correctly

Here's the twist: our model's *between-party* prior structure already follows Gelman's 2015 recommendation for J = 2.

Look at the model specification:

```
mu_party_raw ~ Normal(0, sigma=2)     # Direct prior on party means
sigma_within ~ HalfNormal(sigma=1)    # Within-party spread
```

The party means (`mu_party_raw`) have a **direct prior** — `Normal(0, 2)`. The model does *not* try to estimate a between-party variance hyperparameter from data. It says: "party means are plausibly within about +/-4 of zero on the logit scale." This is a fixed, domain-informed choice, exactly what Gelman recommends for J = 2.

So why does the Senate still fail?

The problem isn't the between-party structure. It's the **within-party** estimation. With only 11 Senate Democrats, the model can't reliably estimate the `sigma_within` for Democrats. The posterior for `sigma_within_D` has a mean of 3.86 with enormous uncertainty — almost four times the prior's mode of zero. The model is desperately trying to accommodate 11 data points, and the prior isn't strong enough to constrain it.

Compare to the House, where 42 Democrats produce a tight `sigma_within_D` estimate of 1.28 — close to the prior and well-constrained.

---

## Part 6: What Can Be Done

The literature and our diagnostics point to several concrete remedies, from quick fixes to principled redesigns.

### Remedy 1: Minimum Group Size Threshold (Simplest)

The most straightforward fix: **don't use the hierarchical model for chambers where any party has fewer than ~20 legislators.** Flag those results as unreliable and default to the flat model, which works perfectly at r = 0.93.

This is the approach most published papers take implicitly — they apply hierarchical models to the U.S. House (435 members) or Senate (100 members), not to the Kansas Senate's 11-member minority caucus. The technique was designed for settings with larger groups.

Our external validation proved the flat model is trustworthy. There's no shame in using it where it works.

### Remedy 2: Tighter Prior on Within-Party Spread

The `sigma_within ~ HalfNormal(1)` prior is "weakly informative" — it gently nudges the within-party spread to be moderate but doesn't insist. With 11 Democrats, "gentle" isn't enough. A tighter prior like `HalfNormal(0.5)` would more strongly regularize the within-party spread, preventing the model from wandering into implausible regions.

The tradeoff: a tighter prior imposes more of the modeler's assumptions and less of the data's voice. For large groups, this is unwelcome — you want the data to speak. For groups of 11, the data speaks too softly to be heard regardless, so a stronger prior is the pragmatic choice.

### Remedy 3: Informative Prior Calibration from Shor-McCarty

Now that we have external validation data, we could use the Shor-McCarty scores to *calibrate* the hierarchical model's priors. For example:

- Compute the actual between-party distance in Shor-McCarty scores for Kansas (typically 1-3 on their scale)
- Compute the actual within-party spread for Kansas Democrats and Republicans
- Use these empirically grounded values as the basis for the prior scales

This is an example of **empirical Bayes** — using one dataset to inform the priors for a model applied to a related dataset. It's a well-established technique, and having the Shor-McCarty data makes it natural.

### Remedy 4: Group-Size-Adaptive Priors (Implemented — ADR-0043)

Rather than using the same `HalfNormal(1)` for both parties regardless of size, scale the prior based on the group:

```
For n >= 20 legislators:  sigma_within ~ HalfNormal(1.0)   # Let data speak
For n < 20:               sigma_within ~ HalfNormal(0.5)   # Stronger guidance
```

**This remedy was implemented in ADR-0043** with constants `SMALL_GROUP_THRESHOLD = 20` and `SMALL_GROUP_SIGMA_SCALE = 0.5`, applied in both `build_per_chamber_model()` and `build_joint_model()`. The tighter prior prevents the catastrophic convergence failures (R-hat > 1.8, ESS < 10) that occurred with the standard prior on small groups like Senate Democrats (~11 members), while allowing the data to speak for larger groups.

This acknowledges the statistical reality that smaller groups need more prior regularization without abandoning the hierarchical structure entirely.

### Remedy 5: Fix sigma_within for Small Groups

The most aggressive fix, following Gelman's J = 2 advice: for groups below a size threshold, don't estimate `sigma_within` at all. Fix it to a domain-reasonable value (e.g., 1.0, based on the House estimates). This eliminates the parameter that the model can't identify, removing the source of the failure.

The downside: you're no longer learning within-party variation from the data for that group. The upside: the individual ideal points will be well-estimated, which is what we actually care about.

### Remedy 6: Use the Joint Model

Our 3-level joint model — which combines House and Senate into a single estimation — converged perfectly, with R-hat at 1.004 and zero issues. By pooling 168 legislators across both chambers, it has enough data to estimate all group-level parameters reliably.

The joint model already exists and works. For analyses that require hierarchical estimates for Senate legislators, use the joint model's results rather than the per-chamber model.

### What We Wouldn't Recommend

- **Running more chains or longer chains.** The convergence failure is not a sampling problem but a model identification problem. More compute won't help.
- **Switching to half-Cauchy priors.** The heavy tails of the half-Cauchy would make the Senate problem *worse* by allowing `sigma_within` to wander to even more extreme values. The current half-normal is the better choice for small groups.
- **Dropping the hierarchical model entirely.** It works for the House and provides genuine insight (variance decomposition, shrinkage analysis). The fix should be targeted, not wholesale.

---

## Part 7: The Broader Lesson

The hierarchical Senate failure is actually a success story for our methodology. Here's why:

**Before external validation**, we had convergence diagnostics (R-hat = 1.83) that flagged a problem, but we couldn't tell whether the resulting ideal points were usable despite the warnings or genuinely corrupted. Convergence warnings sometimes occur in models that still produce reasonable point estimates.

**After external validation**, we know definitively: the hierarchical Senate scores are not just noisy — they're inverted. The flat model validates at r = 0.93. The hierarchical model produces r = -0.54. This is a real, consequential failure, not a harmless warning.

This is exactly what external validation is for. Internal diagnostics can flag potential problems. External validation can confirm whether those problems matter. The combination is what gives us confidence in our results.

The flat IRT model — our workhorse — is externally validated at r = 0.93-0.98 across both chambers. That's the number that matters. The hierarchical model adds value in the House (r = 0.984, variance decomposition, shrinkage insight) and should be used there. In the Senate, with its 11-member minority caucus, the flat model is the right tool.

As the statistics saying goes: **all models are wrong, but some are useful.** The hierarchical model is useful for the House. For the Senate, it needs either the fixes described above or an honest acknowledgment that 11 legislators aren't enough for reliable partial pooling.

---

## Summary of Key References

| Author(s) | Year | Key Contribution |
|-----------|------|-----------------|
| Stein; James & Stein | 1955; 1960 | Shrinkage dominance only guaranteed for J >= 3 groups |
| Bailey | 2001 | Random-effects ideal points with covariates for sparse data |
| Clinton, Jackman & Rivers | 2004 | Foundational Bayesian IRT for ideal points |
| Bafumi, Gelman, Park & Kaplan | 2005 | Practical issues in hierarchical ideal point estimation |
| Gelman | 2006 | Prior distributions for variance parameters; half-Cauchy recommendation |
| Papaspiliopoulos, Roberts & Skoeld | 2007 | Centered vs. non-centered parameterization theory |
| Gelman | 2008 | Folk theorem: computational problems signal model problems |
| Peress | 2009 | Small chamber ideal point estimation; incidental parameters problem |
| Shor & McCarty | 2011 | National state legislature ideal points (our external validation source) |
| Polson & Scott | 2012 | Theoretical justification for half-Cauchy priors |
| Betancourt & Girolami | 2013 | Funnel geometry in hierarchical HMC sampling |
| Gelman | 2015 | J = 2 groups requires informative prior or fixed variance |
| Stan Development Team | ongoing | Current prior recommendations: half-normal for small J |

---

## Technical Appendix: The Exact Model

For readers who want to see precisely what's under the hood, here is the hierarchical model as implemented in `analysis/hierarchical.py`:

```
# Party-level parameters
mu_party_raw ~ Normal(0, sigma=2)           # 2 party mean ideal points
mu_party     = sort(mu_party_raw)           # Identification: D < R
sigma_within ~ HalfNormal(sigma=1)          # Per-party within-group spread

# Legislator ideal points (non-centered parameterization)
xi_offset_i  ~ Normal(0, sigma=1)           # Raw offset per legislator
xi_i         = mu_party[party] + sigma_within[party] × xi_offset_i

# Bill parameters
alpha_j      ~ Normal(0, sigma=5)           # Bill difficulty (how hard to pass)
beta_j       ~ Normal(0, sigma=1)           # Bill discrimination (how much it separates)

# Likelihood
P(Yea) = logistic(beta_j × xi_i - alpha_j)
```

MCMC settings: 2000 draws, 1500 tuning, 2 chains, target acceptance 0.95, NUTS sampler (PyMC).

The flat IRT model replaces the first four lines with a simple `xi_i ~ Normal(0, 1)` and uses hard anchor constraints (one known liberal = -1, one known conservative = +1) instead of the ordering constraint.
