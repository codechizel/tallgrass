# How We Analyze Kansas Legislature Voting Data

**A Plain-English Guide to the Analysis Pipeline**

---

## What This Project Does

Every time a bill comes up for a vote in the Kansas Legislature, each legislator casts a vote: Yea, Nay, or they're absent. That's public record. We collect every one of those votes — roughly 78,000 individual votes across 882 roll calls for the current 2025-2026 session alone — and then ask a series of increasingly sophisticated questions about what the data reveals.

This document walks through each step of our analysis, in order, explaining what we're doing and why. No statistics background is required. Where we use technical methods, we'll explain them through analogies and real examples from the Kansas data.

---

## Why Order Matters

Imagine you're a detective arriving at a crime scene. You wouldn't start by running DNA tests. First, you'd walk through the scene and take notes. Then you'd photograph the evidence. Then you'd dust for fingerprints. Then maybe you'd send something to the lab. Each step builds on the last, and skipping ahead can mean you miss something important or waste time on the wrong lead.

Our analysis follows the same logic. We start with simple counting and work our way up to sophisticated modeling. Each step serves two purposes: it produces its own insights, and it feeds information into the steps that follow. When a later step produces a surprising result, the earlier steps give us the foundation to understand whether the surprise is real or an artifact.

Here's the full pipeline at a glance:

1. **Exploratory Data Analysis** — Count things. Look for problems.
2. **PCA** — Find the main axis of disagreement.
3. **UMAP** — Draw a map of the legislature.
4. **Ideal Point Estimation (IRT)** — Place every legislator on a precise ideological scale.
5. **Clustering** — Ask: are there distinct voting blocs?
6. **Network Analysis** — Ask: who votes with whom, and who's the bridge?
7. **Classical Indices** — Compute standard measures like party loyalty.
8. **Prediction** — Can a computer predict how each legislator will vote?
9. **Beta-Binomial** — Adjust loyalty scores for legislators with small track records.
10. **Hierarchical IRT** — Measure how much of ideology comes from party vs. the individual.
11. **Synthesis** — Combine everything into a narrative.
12. **Profiles** — Deep-dive into the most interesting legislators.
13. **Cross-Session Validation** — Check whether our findings hold up over time.

Let's walk through each one.

---

## Step 1: Exploratory Data Analysis — Taking Inventory

### What it is

Before doing anything clever, we count things. How many legislators are there? How many voted on each bill? How many bills passed unanimously? How many were close calls?

This is the equivalent of a doctor taking your vitals before running any tests. It would be irresponsible to skip it.

### What we learn

For the current 91st Kansas Legislature (2025-2026):

- **172 legislators** cast votes: 130 in the House, 42 in the Senate
- **124 Republicans, 48 Democrats** — a Republican supermajority at roughly 72%
- Of 882 roll calls, **322 were unanimous** (zero Nay votes). Another 69 were nearly unanimous. That means about 44% of votes carried no disagreement at all.
- The overall Yea rate is about 84%. Kansas legislators agree with each other far more often than they disagree.
- Only **11 votes out of 882 had a margin of 10 or less**. The tightest: HB 2366 tied 58-58 in the House.

### Why it matters for the next steps

Those unanimous votes are a problem for analysis. If every single legislator voted the same way, that vote tells us nothing about who disagrees with whom. It's like giving a test where every student gets every question right — you can't tell who's smarter. So we filter those out, keeping only the "contested" votes where at least 2.5% of voters were in the minority. We also drop any legislator who cast fewer than 20 contested votes (not enough data to say anything meaningful about them).

After filtering, we're left with about 120 active House members voting on roughly 200 contested bills, and 40 Senators on about 150 contested bills. This cleaned-up dataset is what every subsequent step works with.

### One more thing: measuring agreement

We also compute how often every pair of legislators votes the same way. But there's a trap. If 84% of all votes are Yea, then two legislators who both just vote Yea on everything would appear to "agree" 84% of the time — even if they have nothing in common besides following the herd. To correct for this, we use a measure called Cohen's Kappa, which adjusts for the rate of agreement you'd expect by chance. Think of it as asking not "how often did they agree?" but "how often did they agree *beyond what you'd expect from random chance*?"

This agreement matrix becomes the foundation for several later steps.

---

## Step 2: PCA — Finding the Main Storyline

### What it is

Each legislator has cast hundreds of votes. That's hundreds of data points per person. PCA (Principal Component Analysis) asks: *can we summarize all of this with just one or two numbers?*

### The analogy

Imagine 172 people are standing in a large room. Each person's location in the room represents their complete voting record — hundreds of dimensions, one for each contested bill. We can't visualize a room with 200 dimensions. But PCA finds the single direction through that room where people are *most spread out*. If you shined a flashlight from the side and looked at the shadows on the wall, PCA finds the angle that creates the longest, most informative shadow.

In a legislature, that first direction almost always turns out to be the left-right ideological spectrum. It's the single summary that captures the most information about how people differ.

### What we learn

- The first principal component (PC1) captures about 35% of all voting variation in the House and 32% in the Senate. That's the "main storyline" of Kansas politics — it's the liberal-conservative divide.
- A second dimension exists but explains only about 11% of variation — much weaker, and harder to interpret.
- We orient the scale so that higher numbers mean more conservative (matching the convention used by national projects like VoteView/DW-NOMINATE).

### Why it comes before the next steps

PCA is fast and cheap to compute. It gives us a quick sketch of the political landscape before we invest in the more expensive, more precise methods that follow. Think of PCA as a quick pencil sketch before bringing out the oil paints.

Critically, PCA scores from this step are used to set up the next major analysis (IRT) — they help tell the model which end of the scale is "conservative" and which is "liberal."

### The limitation

PCA gives every legislator a score, but it doesn't come with any measure of *uncertainty*. It can't tell you "we're pretty confident about Senator A's score, but Senator B missed a lot of votes so we're less sure about theirs." That's what Step 4 addresses.

---

## Step 3: UMAP — Drawing the Map

### What it is

UMAP is a technique for creating a two-dimensional map of the legislature. If PCA finds the best "shadow on the wall," UMAP is more like a cartographer creating a map of a country — it tries to preserve the *neighborhoods*. Legislators who vote similarly end up close together. Legislators who vote differently end up far apart.

### The analogy

Imagine you have a list of how far apart every pair of cities in Kansas is. Could you draw a map from just the distances? That's essentially what UMAP does, except instead of geographic distance, it uses voting similarity.

### What we learn

The resulting map shows the legislature as two distinct clusters — one red (Republican), one blue (Democrat) — with a clear gap between them. Within the Republican cluster, you can see legislators spread along a spectrum from moderate to very conservative, but there's no hard boundary splitting them into subgroups. It's a gradient, not two separate clusters.

### Why it's useful

This map becomes the most intuitive visualization for a general audience. You can literally point at a dot and say "this is Senator Tyson, way out on the far right edge" or "this is Rep. Schreiber, the Republican dot sitting closest to the Democrats." It makes the abstract concept of "ideology" into something you can see.

---

## Step 4: Ideal Point Estimation (IRT) — The Gold Standard

### What it is

This is the most important analytical step in the pipeline, and it deserves a careful explanation.

IRT stands for Item Response Theory. It was originally developed for educational testing — if you've ever taken a standardized test like the SAT, your score was calculated using a version of this method. The core idea is simple: by looking at which questions a student gets right and wrong, you can figure out how skilled they are. But you can also figure out something about each question — how hard it is, and how well it distinguishes strong students from weak ones.

We apply the same logic to voting. Instead of students and test questions, we have **legislators and bills**. Instead of right and wrong answers, we have **Yea and Nay votes**. The method simultaneously estimates two things:

1. **Each legislator's "ideal point"** — a number representing where they sit on the ideological spectrum. Think of it as their address on a number line running from liberal (left) to conservative (right).

2. **Each bill's characteristics** — how "hard" or contentious the bill is (its difficulty), and how well it separates liberals from conservatives (its discrimination). A bill where party-line voting is strict has high discrimination. A bill where the vote is mixed regardless of ideology has low discrimination.

### A step-by-step example

Let's make this concrete with a simplified example using Kansas data:

- **SB 1** (Emergency Final Action): 36 Yea, 2 Nay in the Senate. Nearly everyone voted for it. This bill has *low discrimination* — it doesn't tell you much about who's liberal or conservative, because almost everyone agreed.

- **A veto override vote**: 28 Republicans vote Yea, 0 Democrats vote Yea. This bill has *high discrimination* — the vote perfectly separates the two parties. It's like a test question that only the top students get right.

- **Senator Caryn Tyson** votes Yea on nearly every high-discrimination partisan bill but votes Nay on many low-discrimination routine bills. The model places her far to the right (very conservative) because when it *matters* — when a bill sharply divides the parties — she's reliably on the conservative side.

- **Rep. Mark Schreiber** (Republican) frequently votes with Democrats. The model places him near zero — ideologically at the boundary between the two parties.

### Why it's better than just counting

You might think: why not just count how often each legislator votes with their party? The problem is that not all votes are equal. A legislator who defects on a unanimous 130-0 vote is doing something very different from a legislator who defects on a close 65-55 vote. IRT naturally accounts for this. Bills that split the legislature sharply carry more weight in determining ideal points than bills where everyone agrees.

### The Bayesian part

We use a Bayesian version of IRT, which means the model doesn't just give us a single number for each legislator — it gives us a range of plausible values. This is like a weather forecast saying "the high will be 75, give or take 3 degrees" instead of just "the high will be 75." For a legislator who cast 200 votes, the range is narrow (we're quite confident). For a legislator who missed many votes, the range is wide (we're less sure). This honesty about uncertainty is one of the great virtues of the Bayesian approach.

### Validation

Here's a reassuring finding: the PCA scores from Step 2 and the IRT ideal points from Step 4 correlate at r > 0.95 (where 1.0 would be perfect agreement). Two completely different methods — one a simple linear technique, the other a full probabilistic model — arrive at essentially the same picture. When independent methods agree, that's strong evidence that both are capturing something real.

---

## Step 5: Clustering — Are There Distinct Factions?

### What it is

In Kansas politics, everyone assumes there are at least three voting blocs: conservative Republicans, moderate Republicans, and Democrats. Clustering analysis asks: *does the voting data actually support this?*

### The analogy

Imagine you dump a bag of M&Ms on a table and ask: how many colors are there? That's easy — you can see them. Now imagine the M&Ms are all slightly different shades, blending from one color to the next. Is that two colors with a sharp boundary, or one color with a gradient? Clustering algorithms are mathematical ways to answer that question.

We use three different methods (not just one) so we can see whether they agree. If all three say "two groups," we can be much more confident than if they disagreed.

### What we learn

This is one of the most interesting findings in the entire analysis:

**The data says two groups, not three.** All three methods agree: the clearest structure in the Kansas Legislature is the party divide — Republicans and Democrats. There is no statistically meaningful "moderate Republican" cluster.

That doesn't mean all Republicans vote the same. They don't. But the variation among Republicans is *continuous* — a smooth spectrum from center-right to far-right — rather than two distinct camps. It's the difference between a light dimmer and a light switch. The dimmer has a range of brightness but no "click" point that separates two settings.

### Why this matters

If there were a distinct moderate Republican bloc, it might vote with Democrats on certain issues, forming predictable coalitions. But a continuous spectrum means that which Republicans break ranks depends on the specific issue, not on membership in a defined faction.

---

## Step 6: Network Analysis — Mapping Relationships

### What it is

Network analysis treats the legislature as a social network. Each legislator is a node (a dot). If two legislators agree with each other at a meaningful rate (above that chance-adjusted Kappa threshold), we draw a line between them. The result is a web of connections — a voting network.

### The analogy

Think of a high school cafeteria. Students cluster into friend groups — the athletes at one table, the band kids at another. Some students sit at the edge of a group and talk to kids at other tables. In our legislative network, "sitting together" means "voting together," and the students who bridge different tables are the most interesting legislators.

### What we learn

At the standard threshold for meaningful agreement, **the Republican and Democratic networks are completely disconnected**. There is literally zero agreement between the parties that exceeds what you'd expect from chance alone. The two parties vote as entirely separate worlds on contested bills.

Within each party, though, the network reveals something useful: **who are the most connected legislators?** Who sits at the center of their party's network, voting reliably with the most colleagues? And who sits at the edge, loosely connected even within their own party?

The most important measure is **betweenness centrality** — how often a legislator sits on the shortest path between two other legislators. A legislator with high betweenness is a "bridge" — they connect parts of the network that would otherwise be further apart. In the House, Rep. Jesse Borjon emerges as a key bridge figure within the Republican network. In the Senate, Sen. Brenda Dietrich is an important connector.

### One striking finding

Senator Caryn Tyson — the most conservative member of the Senate by IRT score — has only 5 connections out of 31 possible Republican-to-Republican links. She is the *most isolated* Republican senator despite being the *most conservative*. This tells us something PCA and IRT alone don't capture: she's an ideological purist who doesn't build voting coalitions, even within her own party.

---

## Step 7: Classical Indices — Standard Yardsticks

### What it is

Political scientists have been measuring party discipline for decades using standardized indices. We compute these to connect our work to the established literature and to provide intuitive, easily understood metrics.

The key measures:

- **Party Unity Score**: On votes where the two parties oppose each other (a majority of Republicans on one side, a majority of Democrats on the other), how often does a legislator vote with their own party? A score of 0.95 means they vote the party line 95% of the time on these divisive votes.

- **Maverick Score**: The inverse of unity. A maverick is someone who frequently breaks from their party.

- **Rice Index**: For each party on each vote, how unified are they? A Rice Index of 1.0 means every party member voted the same way. A Rice Index of 0.0 means the party split evenly.

### What we learn

- The average Republican unity score in the House is about 0.95 — very disciplined.
- The average Democrat unity score is about 0.93 — slightly less disciplined, but still high.
- **Rep. Mark Schreiber** stands out: a Republican with a unity score of only 0.62. He breaks from his party on over a third of divisive votes — making him by far the most independent Republican in the House.
- **Sen. Brenda Dietrich** is the Senate counterpart: a Republican with a unity score of about 0.79.

### Why these come after IRT, not before

You might wonder: why not just compute party unity first and skip all the complicated modeling? Because unity scores treat all "party votes" equally — a vote that barely qualifies (51% vs 49% within each party) counts the same as a vote where the split is 98% vs 2%. IRT naturally weights votes by how much information they carry. Unity is a useful summary, but it's a blunter instrument.

Still, unity scores are easy to explain and immediately meaningful ("this legislator votes with their party X% of the time"), so they complement the IRT scores nicely.

---

## Step 8: Prediction — Testing Understanding

### What it is

If we truly understand how the Kansas Legislature works, we should be able to *predict* votes. We train a machine learning model on all the information gathered so far — ideal points, party, loyalty scores, network position — and ask: given a specific legislator facing a specific bill, will they vote Yea or Nay?

### The analogy

Imagine you know a friend very well — their tastes, their habits, their values. Someone describes a new restaurant to you and asks: "Will your friend like it?" The better you know them, the more accurate your prediction. That's what we're doing, but with votes instead of restaurant reviews, and for 172 legislators at once.

### What we learn

The model predicts individual votes with **98% accuracy** and an AUC-ROC of 0.99 (a measure of how well the model distinguishes Yea from Nay votes, where 1.0 is perfect). For context, simply guessing "Yea" every time would get you 84% accuracy. Our model is dramatically better than that naive approach.

The most important finding is *which features matter most*. The model tells us, in order: a bill's IRT discrimination score matters most (how partisan the bill is), followed by the legislator's ideal point (where they sit on the spectrum), followed by their uncertainty (how many votes they've cast). Party membership and loyalty scores matter, but less — because they're largely redundant with the ideal point.

This confirms that the IRT model from Step 4 really is capturing the core of what drives legislative voting in Kansas.

### The interesting failures

The few votes the model gets wrong are often the most revealing. When the model is 99% confident a legislator will vote Yea but they vote Nay, something unusual happened. These "surprising votes" are candidates for deeper investigation — they might reflect a personal stake in a bill, constituent pressure, or a deal cut behind the scenes.

---

## Step 9: Beta-Binomial — Correcting for Small Samples

### What it is

This step addresses a specific problem with the party unity scores from Step 7. Imagine two legislators:

- **Legislator A** voted in 200 party-line votes and agreed with their party 190 times (95% unity).
- **Legislator B** voted in only 8 party-line votes and agreed with their party 6 times (75% unity).

Is Legislator B really less loyal? Maybe. But 8 votes is a tiny sample. If they'd voted just one more time with the party, their score would jump to 78%. With one more the other way, it drops to 71%. The raw percentage is *noisy* — it bounces around a lot with small samples.

### The analogy

This is the same problem as early-season batting averages in baseball. A player who goes 4-for-10 in the first week has a .400 batting average — seemingly incredible. But no one believes they'll hit .400 all season. What we do, instinctively, is pull that early estimate back toward something more typical. If the league average is .270, we might mentally adjust that .400 to something like .330 — still above average, but not as extreme as the raw number suggests.

That instinct is exactly what this analysis formalizes. It's called **shrinkage** — estimates based on small samples are "shrunk" toward the group average. Estimates based on large samples barely move. Legislator A's 95% (based on 200 votes) stays almost exactly where it is. Legislator B's 75% (based on 8 votes) gets pulled significantly toward their party's average.

### What we learn

For most Kansas legislators, the adjustment is small — they've cast enough votes that their raw scores are reliable. But for newly appointed members or those who missed many sessions, the adjustment can be significant. This gives us a fairer picture of their true loyalty, rather than one distorted by a small sample.

---

## Step 10: Hierarchical IRT — Nature vs. Nurture

### What it is

The IRT model from Step 4 treats every legislator as a completely independent individual. But legislators aren't independent — they belong to parties that coordinate positions, share information, and sometimes enforce discipline. The hierarchical model adds a layer of structure: it says "legislators within a party are *similar* to each other, but not identical."

### The analogy

There are three ways to approach this kind of modeling, and they mirror a familiar story:

- **Complete pooling** ("all legislators are the same"): Ignore party entirely and treat everyone as one big group. This is like Mama Bear's porridge — too cold. It misses the obvious fact that Democrats and Republicans vote very differently.
- **No pooling** ("every legislator is an island"): Treat each person completely independently, learning nothing from their colleagues. This is Papa Bear's porridge — too hot. A legislator with only 15 votes gets a wildly unreliable estimate.
- **Partial pooling** ("learn from your group, but keep your individuality"): Each legislator gets their own estimate, but that estimate is gently informed by what their party colleagues look like. This is Baby Bear's porridge — just right.

The key question this model answers is: **how much of a legislator's ideology is explained by their party, and how much is individual?**

### What we learn

About **90% of the variation in ideal points is between parties** (Democrats vs. Republicans), and only about 10% is variation within parties. In other words, party membership is an overwhelmingly strong predictor of voting behavior. Knowing that a legislator is a Kansas Republican tells you almost everything you need to know about how they'll vote. The remaining 10% is where the interesting individual stories live — the mavericks, the moderates, the ideological purists.

This model also has a practical benefit: it produces better estimates for legislators with sparse voting records. By "borrowing strength" from their party colleagues, it can give a reasonable estimate even when an individual's data alone would be too thin. Think of it like estimating a new student's likely performance by considering both their own test scores and the school's overall track record — you use both sources of information.

---

## Step 11: Synthesis — Telling the Story

### What it is

At this point, we have a dozen different measurements for each legislator: their ideal point, their uncertainty, their party unity, their network position, their prediction accuracy, their adjusted loyalty, their hierarchical estimate, and more. The synthesis step joins all of these into a single unified profile for each legislator and then scans the data for the most interesting stories.

### What stories does it find?

The system automatically identifies three types of notable legislators:

**Mavericks** — legislators in the majority party (Republicans, in Kansas) who break ranks most often. In the House, **Rep. Mark Schreiber** stands out with a party unity of just 0.62. In the Senate, it's **Sen. Brenda Dietrich** at 0.79. These are the most independent voices within the Republican supermajority.

**Bridge-builders** — legislators who sit near the ideological midpoint and have high network connectivity. They're the ones who, in theory, could broker compromise. In the House, **Rep. Jesse Borjon** occupies this role — ideologically moderate among Republicans with high betweenness centrality. In the Senate, **Sen. Rick Hill** plays a similar role.

**Paradoxes** — legislators whose metrics seem to contradict each other. The most fascinating is **Sen. Caryn Tyson**, who ranks as the *most conservative* senator by IRT ideal point but has *below-average* party loyalty. How is that possible? Because she's a principled ideological purist: on high-stakes partisan votes, she's 100% party-line. But on routine, low-controversy bills, she frequently votes Nay on her own — not because she's moderate, but because she holds a stricter standard than her party colleagues for what legislation she's willing to support. She's the most extreme *and* the most independent, which is a rare combination that a single number can't capture.

---

## Step 12: Profiles — Deep Dives

### What it is

For each legislator identified as notable in the synthesis step, we produce a detailed dossier with five analyses:

1. **Scorecard** — A normalized report card showing where they rank on six dimensions (ideology, unity, loyalty, maverick score, network position, and prediction accuracy).

2. **Bill-type breakdown** — How do they vote on highly partisan bills vs. routine bills? (This is what reveals the Tyson paradox.)

3. **Defection analysis** — On which specific bills did they break from their party, and how close was the overall vote?

4. **Voting neighbors** — Which 5 legislators vote most like them? Which 5 are most opposite?

5. **Surprising votes** — The specific votes where the prediction model was most wrong about this legislator. These are the votes that don't fit the pattern — and therefore the most interesting ones to investigate.

---

## Step 13: Cross-Session Validation — Does It Hold Up?

### What it is

Everything above analyzes a single two-year legislative session. But we can also compare across sessions. We run the same pipeline on the previous session (2023-2024, the 90th Legislature) and then ask: are the results consistent?

### What we check

- **Returning legislators**: About 78% of legislators serve in both sessions. Do their ideology scores stay roughly the same? (Mostly yes — ideology is quite stable.)
- **Who moved?** A few legislators shift noticeably between sessions. These shifts might reflect genuine political evolution, changes in the bills being voted on, or external pressures.
- **Prediction transfer**: If we train the prediction model on the 2023-2024 data, how well does it predict votes in 2025-2026? A modest drop in accuracy (about 5-10 percentage points) — not bad, given that an entirely different set of bills is being voted on.
- **Threshold validation**: The cutoffs we use to detect mavericks and bridges — do they produce sensible results in both sessions, or were they accidentally tuned to one specific set of data?

### Why this matters

Any analysis can look impressive on the data it was built from. The real test is whether it generalizes. Cross-session validation is our way of checking that the patterns we found are features of Kansas politics, not artifacts of our methods.

---

## Wrapping Up: What the Numbers Can and Can't Tell You

### What they can tell you

- Where each legislator sits on the ideological spectrum, with honest uncertainty
- Whether the legislature has distinct factions or a smooth spectrum (smooth spectrum, in Kansas)
- Which legislators are the most independent, the most connected, and the most surprising
- That a single left-right dimension explains the vast majority — about 90% — of voting behavior
- That party membership is the dominant predictor, but individual variation exists within each party

### What they can't tell you

- **Why** anyone votes the way they do. Voting data shows *what happened*, not the motivations behind it. A Nay vote could reflect deep ideological conviction, a favor owed to a colleague, constituent pressure, or a misunderstanding of the bill.
- **The quality of legislation.** Higher party unity doesn't mean better governance. A maverick isn't necessarily principled, and a party loyalist isn't necessarily a rubber stamp. The data is silent on these value judgments.
- **What happens off the floor.** Committee negotiations, caucus meetings, and informal conversations shape legislation long before it reaches a vote. Our analysis sees only the final, public act.
- **The second and third dimensions.** Our primary model is one-dimensional (left-right). Real politics has multiple dimensions — fiscal vs. social policy, urban vs. rural interests, institutional vs. anti-establishment attitudes. The one-dimensional model captures the dominant pattern, but it simplifies the rest.

These limitations aren't failures — they're the honest boundaries of what quantitative analysis of roll-call votes can achieve. The value is in what the data *does* reveal: a clear, replicable, evidence-based picture of how the Kansas Legislature actually votes, free from anecdote and selective memory.
