# Experiment: [Short Title]

**Date:** YYYY-MM-DD
**Status:** Planning | In Progress | Complete
**Author:** [name]

## The Short Version

*2-3 sentences a non-technical reader can understand. What did we try, and did it work? Fill this in last, after results are in.*

## Why We Ran This Experiment

*What problem motivated this experiment? Describe the behavior we observed, why it matters, and what we hoped to improve. Write for a reader who doesn't know what MCMC or R-hat means — explain the practical consequence (e.g., "our model couldn't produce reliable results for the House") before the technical detail.*

## What We Expected to Find

*State the hypothesis plainly: "We believe X causes Y, and changing Z should fix it." Include the reasoning — why do we expect this to work?*

## What We Tested

*Describe each run. Each run should change exactly one variable from a baseline so we can attribute any improvement to that specific change.*

### Baseline (Current Production Model)

- **What it is:** [Plain description of the current approach]
- **Command:** `[exact command to reproduce]`
- **Output directory:** `[path relative to this experiment]`

### Run N: [Descriptive Name]

- **What changed:** [Exactly one change from the baseline, in plain English]
- **Why:** [Brief rationale for this specific change]
- **Command:** `[exact command to reproduce]`
- **Output directory:** `[path relative to this experiment]`

## How We Measured Success

*Define the metrics we compared across runs. For each metric, explain what it means in plain terms and where the pass/fail threshold comes from.*

| Metric | What It Tells Us | Passing Value | Source |
|--------|------------------|---------------|--------|
| | | | |

## Results

*Fill in after each run completes. Do NOT delete previous results when adding new ones — the full history is the record.*

### Summary Table

| Metric | Baseline | Run 1 | Run 2 | ... |
|--------|----------|-------|-------|-----|
| | | | | |

### What We Observed

*Describe the results in narrative form. Lead with the headline finding, then detail. Highlight surprises. Include specific numbers — this is the evidence section.*

### Impact on Rankings and Scores

*Did the experiment change who ranks where? Show correlation between baseline and treatment ideal points. If rankings shifted, identify who moved and by how much. Readers care about whether the results they see in the main reports would change.*

## What We Learned

*Plain-English conclusion. Was the hypothesis confirmed? What does this mean for the project going forward? What should we do next?*

## Changes Made

*What code or configuration changes were applied to production as a result? Include commit hash if applicable. "No changes" is a valid answer — not every experiment leads to a code change.*

---

## Default Session

Unless otherwise noted, all experiments use the **91st biennium (2025-26)** as the test session. This is the current session; the primary analyst (Sen. Joseph Claeys) has content knowledge of the legislators and can spot anomalies in the results. Each experiment produces a full HTML report so that downstream impacts on rankings, tables, and plots can be visually inspected.

## File Organization

Experiment directories use: `YYYY-MM-DD_short-description/`

Each experiment directory contains:
- `experiment.md` — this document (copy from TEMPLATE.md)
- `run_experiment.py` — the script that runs the experiment
- `run_NN_description/` — output directories for each run (numbered sequentially)
- Any supporting scripts, logs, or data specific to this experiment

Results are append-only: new runs add rows to the results table; old results are never deleted.
