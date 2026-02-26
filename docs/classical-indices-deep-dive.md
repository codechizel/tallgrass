# Classical Indices Deep Dive

A literature review, code audit, and gap analysis of the Phase 7 (Classical Indices) implementation in the Tallgrass analysis pipeline.

## Summary of Findings

**The implementation is methodologically sound.** All four core indices — Rice, Party Unity, ENP, and Maverick — use the correct canonical formulas from the political science literature. There are no correctness bugs. The main opportunities are (1) a handful of missing standard indices that would strengthen the analysis, (2) test coverage gaps, (3) a code smell involving duplicated constants, and (4) the question of directory restructuring.

## 1. Literature Survey: What Are "Classical Indices"?

The term "classical indices" in legislative voting analysis refers to descriptive, assumption-light measures that predate model-based approaches (IRT, NOMINATE). They require no latent variable estimation — just counting.

### 1.1 Rice Index (Rice 1925)

**Formula:** `Rice_j = |Yea_j - Nay_j| / (Yea_j + Nay_j)` per party per roll call.

**Range:** 0 (perfect 50-50 split) to 1 (unanimous). A Rice of 0.50 means a 75-25 split.

**Known bias:** Artificially inflated for small parties (Desposato 2005, BJPS). A party of 3 can only achieve Rice values of {1/3, 1}, never 0. Our EDA phase already implements the Desposato bootstrap correction.

**Reference:** Rice, S.A. (1925). "The Behavior of Legislative Groups." *Political Science Quarterly* 40(1).

### 1.2 Party Unity Score (CQ Standard)

**Step 1 — Identify party votes:** A roll call is a "party vote" iff the majority of Republicans oppose the majority of Democrats, with each party having >= 2 Yea+Nay voters.

**Step 2 — Per-legislator score:** `Unity_i = (votes with party majority on party votes) / (party votes where i was present)`.

This is the Congressional Quarterly (CQ) standard, used since the 1950s. Absences are excluded from the denominator (modern convention).

**Reference:** CQ Roll Call Vote Studies; Voteview Party Unity Scores.

### 1.3 Effective Number of Parties (Laakso & Taagepera 1979)

**Formula:** `ENP = 1 / sum(p_i^2)` where `p_i` is each party's seat share (or vote share).

This is the reciprocal of the Herfindahl-Hirschman Index (HHI) from industrial economics. Our implementation computes both seat-based (static) and vote-based (per-roll-call, using (party, direction) blocs).

**Alternative:** The Golosov index (`N_eff = sum(p_i / (p_i + p_1^2 - p_i^2))`) handles dominant-party systems better. Relevant for Kansas's Republican supermajority (~72%).

**Reference:** Laakso, M. & Taagepera, R. (1979). "'Effective' Number of Parties." *Comparative Political Studies* 12(1).

### 1.4 Maverick Scores

**Unweighted:** `Maverick_i = 1 - Unity_i` (fraction of defections on party votes).

**Weighted:** Defections weighted by chamber vote closeness: `w_v = 1 / max(margin, floor)`. Close votes receive higher weight because defecting on a close vote is more consequential.

This weighted variant is a project-original contribution — well-motivated but not from a specific published index.

### 1.5 UNITY Index (Carey 2007)

**Formula:** `UNITY = |Yea - Nay| / (Yea + Nay + Abstain + Absent)`

Differs from Rice in the denominator: UNITY penalizes absences. With Kansas's "Absent and Not Voting" category, this would capture whether abstention is a form of soft dissent.

**Not currently implemented.** Would be a one-line addition.

**Reference:** Carey, J.M. (2007). "Competing Principals, Political Institutions, and Party Unity in Legislative Voting." *American Journal of Political Science* 51(1).

### 1.6 Agreement Index (Hix, Noury & Roland 2005)

**Formula:** `AI_j = (3 * max(Y, N, A) - (Y + N + A)) / (2 * (Y + N + A))`

A three-way generalization of Rice that handles Yea/Nay/Abstain. When there are only two categories, it reduces to Rice. Developed for the European Parliament where abstentions are common and meaningful.

Kansas has 5 vote categories (Yea, Nay, Present and Passing, Absent and Not Voting, Not Voting). The AI could capture "soft dissent" via Present and Passing.

**Not currently implemented.** Would require modest effort — the formula is simple but the vote category mapping needs thought.

**Reference:** Hix, S., Noury, A. & Roland, G. (2005). "Power to the Parties." *British Journal of Political Science* 35(2).

### 1.7 Polarization Metrics (McCarty, Poole & Rosenthal 2006)

Three standard metrics from the Voteview tradition:

- **Party Mean Difference:** `mean(ideal_point | R) - mean(ideal_point | D)` on the first dimension.
- **Proportion of Moderates:** Legislators with `|ideal_point| < threshold`.
- **Party Overlap:** Fraction of legislators from each party falling within the other party's ideal-point range.

These are computed from IRT ideal points (or NOMINATE scores). Our pipeline has the IRT data — the cross-referencing phase (Phase 8) computes a Spearman correlation between unity and IRT, but does not compute the standard polarization summary metrics.

**Not currently implemented as standalone metrics.** Partially addressed via cross-referencing.

### 1.8 Roll and Stuff Rates (Carey 2007)

- **Roll rate:** How often a party's majority opposes the winning side when the party supported the status quo.
- **Stuff rate:** How often a party's majority opposes the winning side when the party opposed the status quo.

Interesting for Kansas's supermajority dynamics (how often does the minority party get rolled?), but lower priority.

## 2. Python Ecosystem Assessment

**There is no Python equivalent of R's `pscl`.** The R ecosystem dominates legislative voting analysis tooling:

| R Package | Capability |
|-----------|-----------|
| `pscl` | IRT/ideal points, roll call analysis |
| `wnominate` / `dwnominate` | W-NOMINATE / DW-NOMINATE |
| `anominate` | Alpha-NOMINATE (bridges IRT and NOMINATE) |
| `politicsR` | Rice, ENP, 31 functions |

Python has fragments:

| Python Package | Capability | Limitation |
|----------------|-----------|-----------|
| [`voting`](https://github.com/crflynn/voting) (crflynn) | Diversity (Laakso-Taagepera, Golosov, Shannon), disproportionality, apportionment | Electoral/seat-share only. No roll call analysis. |
| [`py-irt`](https://github.com/nd-ball/py-irt) | Bayesian IRT (1PL, 2PL, 4PL) via Pyro/PyTorch | General IRT, not legislative-specific |
| [`dw-nominate`](https://github.com/jeremyjbowers/dw-nominate) (parser) | Downloads pre-computed NOMINATE scores | Parser only, Python 2-era |
| [`poli-sci-kit`](https://github.com/andrewtavis/poli-sci-kit) | Seat apportionment, parliament plots | No roll call analysis |
| [`pyopenstates`](https://openstates.github.io/pyopenstates/) | Open States API client | Data access only |

**Bottom line:** For classical indices (Rice, Party Unity, ENP, Maverick), there is no existing Python package. Our `analysis/indices.py` is, as far as we can determine, one of the few standalone Python implementations of these measures.

Our choice to use PyMC for Bayesian IRT rather than attempting DW-NOMINATE is validated: at alpha=0, alpha-NOMINATE reduces to IRT 2PL, and for state legislatures, IRT is preferred over NOMINATE (smaller N, full posterior uncertainty).

## 3. Code Audit

### 3.1 Correctness: All Formulas Are Correct

| Index | Formula in Code | Canonical Formula | Verdict |
|-------|----------------|-------------------|---------|
| Rice | `\|yea - nay\| / total_voters` | `\|Yea - Nay\| / (Yea + Nay)` | Correct. `total_voters = yea + nay` by construction. |
| Party Unity | votes with majority / party votes present | CQ standard | Correct. Modern convention (absences excluded). |
| ENP (seats) | `1 / sum(share^2)` | `1 / sum(p_i^2)` | Correct. |
| ENP (votes) | Per-vote HHI on (party, direction) blocs | Literature extension | Correct. Novel but well-motivated. |
| Maverick (unweighted) | `1 - unity` | `1 - Unity` | Correct. |
| Maverick (weighted) | `sum(w * defection) / sum(w)` | No canonical formula | Correct as implemented. Original contribution. |

### 3.2 Code Smells and Refactoring Opportunities

**1. Duplicated constants in `indices_report.py`** (lines 31-38)

The report builder re-declares all 8 constants from `indices.py` with a comment: "Constants duplicated here to avoid circular import." This is a maintenance risk — if someone changes a threshold in `indices.py` but not `indices_report.py`, the report would show stale values.

**Fix:** Extract constants to a shared module (e.g., `analysis/indices_constants.py`) or pass them as parameters to `build_indices_report()`. The constants are already available in the `results` dict via the filtering manifest — the report could read them from there.

**2. Row-by-row iteration in `compute_unity_and_maverick()`** (line 702)

```python
for row in indiv.iter_rows(named=True):
```

This iterates Python-side over every individual vote on every party vote. For the 91st session (~68K votes, ~300 party votes), this is fast enough (~50K qualifying rows), but it's the only non-vectorized computation in the module. The pattern is: accumulate per-legislator stats and defection vote IDs. The defection-set tracking makes pure Polars vectorization awkward, which justifies the current approach — but a comment explaining this trade-off would help.

**3. `PARTY_VOTE_THRESHOLD` is declared but never used in code logic**

The constant `PARTY_VOTE_THRESHOLD = 0.50` is defined at line 156 and appears in the manifest and report, but the actual party vote logic on line 311 uses a hard-coded `pl.col("yea_count") > pl.col("nay_count")` comparison. The threshold is effectively 0.50 by virtue of the `>` operator, but the constant doesn't drive the logic. This isn't a bug (the behavior is correct), but it's misleading — the constant suggests the threshold is configurable when it isn't.

**4. `find_fractured_votes()` determines majority party by average voter count** (line 487)

This uses `avg_size` across all votes per party, which could theoretically differ from seat count if participation rates differ by party. Using `compute_enp_seats()` or the legislators DataFrame directly would be more precise. In practice, for Kansas, this makes no difference.

**5. `from __future__ import annotations` style**

This is used consistently for forward references and is correct. No issue.

### 3.3 Dead Code

**None found.** Every function is called from `main()` or from other functions. The code is clean.

### 3.4 Missing Features (Ranked by Value)

1. **Carey UNITY index** — High value, trivial to add. One-line formula change in denominator. Captures whether abstention is soft dissent.

2. **Polarization summary metrics** — High value, easy to add. Party mean difference, proportion moderates, and overlap on IRT ideal points. Already has the data from cross-referencing.

3. **Agreement Index (Hix-Noury-Roland)** — Medium value. Captures "Present and Passing" as a distinct signal. More relevant for European Parliament-style analysis, but Kansas does have 5 vote categories.

4. **Golosov ENP** — Low-medium value. Better for dominant-party systems. Easy to add alongside Laakso-Taagepera.

5. **Roll/Stuff rates (Carey)** — Medium value for Kansas's supermajority dynamics, but adds complexity.

### 3.5 Test Coverage Gaps

The test file (`tests/test_indices.py`) has **4 test classes with ~15 tests**. This is notably thin compared to other phases (IRT has 73 tests, clustering has 70, network has 53).

**Missing test coverage:**

| Function | Current Coverage | Gap |
|----------|-----------------|-----|
| `_rice_from_counts()` | 5 tests | Good |
| `compute_party_majority_positions()` | 3 tests | Missing: multi-chamber filtering, Independent party handling |
| `identify_party_votes()` | 3 tests | Missing: edge case with no Democrats, tied margins, MIN_PARTY_VOTERS guard |
| `compute_enp_seats()` | 3 tests | Missing: three-party case, single-party chamber |
| `compute_enp_per_vote()` | 0 tests | **Untested** |
| `compute_unity_and_maverick()` | 0 tests | **Untested** — most complex function in the module |
| `compute_co_defection_matrix()` | 0 tests | **Untested** |
| `find_fractured_votes()` | 0 tests | **Untested** |
| `compute_rice_summary()` | 0 tests | Untested but simple |
| `compute_rice_by_vote_type()` | 0 tests | Untested but simple |
| `run_sensitivity_analysis()` | 0 tests | Untested |
| `cross_reference_upstream()` | 0 tests | Untested |
| Plot functions | 0 tests | Acceptable (visual output) |

**The most critical gaps are `compute_unity_and_maverick()` and `compute_enp_per_vote()`.** Unity/maverick is the most complex function (100+ lines, Python-level iteration, z-score computation, multiple output DataFrames) and feeds directly into downstream phases (beta-binomial, synthesis, profiles). ENP per-vote is the novel vote-based extension and should have at least basic formula verification.

## 4. Directory Restructuring Assessment

The user asked whether the `analysis/` directory should reflect the pipeline ordering (e.g., `01_eda/`, `02_pca/`, etc.).

### Current Structure

```
analysis/
  __init__.py
  run_context.py          # shared infrastructure
  report.py               # shared infrastructure
  nlp_features.py         # shared utility (used by prediction)
  eda.py                  # Phase 1
  eda_report.py
  pca.py                  # Phase 2
  pca_report.py
  umap_viz.py             # Phase 3
  umap_report.py
  irt.py                  # Phase 4
  irt_report.py
  clustering.py           # Phase 5
  clustering_report.py
  network.py              # Phase 6
  network_report.py
  prediction.py           # Phase 7 (original numbering varies)
  prediction_report.py
  indices.py              # Phase 7
  indices_report.py
  beta_binomial.py        # Phase 9
  beta_binomial_report.py
  hierarchical.py         # Phase 10
  hierarchical_report.py
  synthesis.py            # Phase 11
  synthesis_report.py
  synthesis_detect.py
  profiles.py             # Phase 12
  profiles_report.py
  profiles_data.py
  cross_session.py        # Phase 13
  cross_session_report.py
  cross_session_data.py
  external_validation.py  # Phase 14
  external_validation_report.py
  external_validation_data.py
  irt_beta_experiment.py  # experimental
  design/                 # 17 design docs
```

### Recommendation: Don't Restructure

**Arguments against numbered directories:**

1. **Import complexity.** Every phase imports from shared modules (`run_context`, `report`). Moving files into subdirectories would require updating every `try/except` import pattern across all 30+ files. Cross-phase imports (e.g., IRT constants imported by hierarchical, indices consumed by beta-binomial) would become messier.

2. **Justfile stability.** All 14 `just` recipes would need path updates.

3. **Test path stability.** All 18 test files import from `analysis.{module}`. Subdirectories would break these.

4. **Phase ordering is already documented.** The pipeline order is clearly specified in CLAUDE.md, the analysis primer, and the Justfile recipes. Adding it to directory names is redundant.

5. **Flat is better than nested** (Zen of Python). With 30+ files, the directory is large but not unmanageable. Each file follows a clear naming convention (`{phase}.py`, `{phase}_report.py`, `{phase}_data.py`).

6. **Risk of churn.** The numbering could change (phases have been renumbered before — indices was originally "Phase 7" but prediction precedes it in some orderings). Numbered directories would need renaming if the pipeline order shifts.

**What would actually help:** A `PIPELINE.md` or a comment block at the top of the analysis `__init__.py` listing the phases in order with one-line descriptions. This gives the at-a-glance ordering without the import breakage.

## 5. Recommendations Summary

### Should Do (High Value, Low Risk)

1. **Add tests for `compute_unity_and_maverick()` and `compute_enp_per_vote()`** — These are the most critical untested functions. Target: ~20 new tests covering formula correctness, edge cases (no party votes, single-party chamber, tied votes), and downstream contract (output schema matches what beta-binomial expects).

2. **Extract duplicated constants** from `indices_report.py` — Either pass them via the results dict or create a shared constants location to eliminate the maintenance risk.

3. **Add Carey UNITY index** — One additional column alongside Rice: `|Yea - Nay| / (Yea + Nay + Absent + Not Voting)`. Captures abstention as soft dissent.

### Could Do (Medium Value)

4. **Add polarization summary metrics** to the cross-referencing phase — Party mean difference, proportion moderates, party overlap from IRT ideal points. Already has the data.

5. **Add tests for `find_fractured_votes()`, `compute_co_defection_matrix()`** — Lower risk of bugs but worth covering for completeness.

6. **Comment the `compute_unity_and_maverick()` iteration pattern** — Explain why Python-level iteration is used instead of vectorization (defection set tracking).

### Defer (Low Priority or High Risk)

7. **Agreement Index (Hix-Noury-Roland)** — Only relevant if "Present and Passing" is analytically interesting in Kansas. Worth investigating but not urgent.

8. **Golosov ENP** — Minor improvement over Laakso-Taagepera for dominant-party systems.

9. **Directory restructuring** — Not recommended. The flat structure works, and renumbering would cause significant churn for marginal benefit.

10. **Roll/Stuff rates** — Interesting but adds complexity to an already large module.

## 6. Comparison with the Field

Our indices implementation compares favorably with the available landscape:

| Aspect | Our Implementation | R's `politicsR` | Python Ecosystem |
|--------|-------------------|-----------------|-----------------|
| Rice Index | Per-vote, per-party, vectorized | Per-vote, per-party | Nothing |
| Party Unity | CQ-standard | Not included | Nothing |
| ENP | Seat-based + vote-based blocs | Seat-based only | `voting` package (seat-based only) |
| Maverick | Unweighted + closeness-weighted | Not included | Nothing |
| Co-defection | Pairwise matrix | Not included | Nothing |
| Cross-referencing | IRT, network, clustering | Not included | Nothing |
| Sensitivity | EDA-filtered comparison | Not included | Nothing |
| Veto override subgroup | Separate analysis | Not included | Nothing |

The weighted maverick score, vote-based ENP with (party, direction) blocs, and the cross-referencing with IRT/network/clustering are all original contributions not found in standard packages.

## Sources

- Rice, S.A. (1925). "The Behavior of Legislative Groups." *Political Science Quarterly* 40(1).
- Laakso, M. & Taagepera, R. (1979). "'Effective' Number of Parties." *Comparative Political Studies* 12(1).
- Desposato, S.W. (2005). ["Correcting for Small Group Inflation of Roll-Call Cohesion Scores."](https://pages.ucsd.edu/~sdesposato/cohesionbjps.pdf) *British Journal of Political Science* 35(2).
- Hix, S., Noury, A. & Roland, G. (2005). ["Power to the Parties."](https://eml.berkeley.edu/~groland/pubs/hnrbjps.pdf) *British Journal of Political Science* 35(2).
- McCarty, N., Poole, K.T. & Rosenthal, H. (2006). *Polarized America.* MIT Press.
- Carey, J.M. (2007). "Competing Principals, Political Institutions, and Party Unity." *AJPS* 51(1). [Data site.](https://sites.dartmouth.edu/jcarey/legislative-voting-data/)
- Poole, K.T. & Rosenthal, H. (2007). *Ideology and Congress.* Transaction Publishers. [Voteview.](https://voteview.com)
- [`voting` Python package](https://github.com/crflynn/voting) — Diversity/disproportionality measures.
- [`py-irt` Python package](https://github.com/nd-ball/py-irt) — Bayesian IRT via Pyro.
- [CQ Roll Call Vote Studies](https://rollcall.com/2025/02/18/congress-party-unity-vote-studies/)
- [Voteview Polarization](https://voteview.com/articles/party_polarization)
- [Voteview Party Unity](https://legacy.voteview.com/Party_Unity.htm)
