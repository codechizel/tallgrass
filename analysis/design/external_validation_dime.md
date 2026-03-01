# External Validation (DIME/CFscores) Design Choices

## Assumptions

1. **Different constructs require different expectations.** CFscores measure donor ideology (who funds you), not voting ideology (how you vote). Overall correlation r ≈ 0.75-0.90 is expected for state legislatures — lower than Shor-McCarty because SM and our IRT both measure the same construct (voting). This is not a failure; it is the expected behavior of a cross-construct comparison.

2. **Name matching reuses Phase 14 infrastructure.** The same `normalize_our_name()` function from Phase 14 handles our legislator names. DIME provides separate `lname`/`fname` fields (already lowercase), so `normalize_dime_name()` is simpler than SM's "Last, First" parser.

3. **Incumbent-only matching.** Our IRT ideal points are for legislators who actually cast votes. DIME includes challengers and open-seat candidates who have CFscores but no voting record. Matching against non-incumbents would inflate unmatched rates without adding value.

4. **Cycle-to-biennium mapping is many-to-one.** Kansas House members serve 2-year terms (one cycle per biennium). Kansas Senate members serve 4-year staggered terms (two cycles overlap each biennium). A biennium maps to 2 election cycles. When a legislator appears in multiple cycles for the same biennium, we keep the most recent cycle's record.

5. **Static CFscore is the primary target.** The static (career) CFscore is analogous to Shor-McCarty's career-fixed `np_score` — one score across all cycles. The dynamic (per-cycle) CFscore is noisier but more temporally aligned with our session-specific IRT scores. Both are reported.

## Parameters & Constants

| Constant | Value | Justification | Code location |
|----------|-------|---------------|---------------|
| `MIN_MATCHED` | 10 | Inherited from Phase 14; minimum for reliable correlation | `external_validation_data.py` |
| `MIN_GIVERS` | 5 | DIME documentation suggests 10; we use 5 because Kansas state legs have fewer donors | `external_validation_dime_data.py` |
| `STRONG_CORRELATION` | 0.90 | Same thresholds as Phase 14 for comparability | `external_validation_data.py` |
| `GOOD_CORRELATION` | 0.85 | Same thresholds as Phase 14 | `external_validation_data.py` |
| `CONCERN_CORRELATION` | 0.70 | Same thresholds as Phase 14 | `external_validation_data.py` |
| `OUTLIER_TOP_N` | 5 | Same as Phase 14 | `external_validation_data.py` |

## Methodological Choices

### Minimum Donor Threshold

**Decision:** `MIN_GIVERS = 5` (configurable via `--min-givers` CLI flag).

**Alternatives considered:**
- `num.givers >= 10`: DIME documentation's suggestion. Too aggressive for Kansas — many state legislators have 5-15 donors per cycle.
- `num.givers >= 1`: No filter. Risk of unreliable CFscores from 1-2 donor records.
- Weighted correlation by donor count: More principled but adds complexity without clear benefit for our validation question.

**Impact:** The `--min-givers` flag allows sensitivity analysis. Run with 5 (default), 10, and 20 to see if results are robust.

### Incumbent Filter

**Decision:** Only match `ico_status == "I"` (incumbents).

**Rationale:** Our IRT ideal points are for legislators who cast votes in a specific biennium. DIME records for challengers (C) and open-seat candidates (O) have CFscores but no corresponding voting record. Including them would create unmatched records that inflate error rates without testing validation.

### Biennium Coverage Decision

**Decision:** Validate 84th-89th (6 bienniums). Skip 90th-91st.

**Rationale:** The 89th biennium (2021-2022) is the key gain — one extra biennium beyond SM coverage. For the 90th (2023-2024) and 91st (2025-2026), the most recent DIME cycle is 2022. Using 2-4 year old CFscores for current legislators is too stale for meaningful validation. A "good" correlation could reflect stable ideology, and a "bad" one could reflect actual ideological change — neither is informative.

### SM Side-by-Side Comparison

**Decision:** For bienniums 84th-88th (where both SM and DIME are available), load SM correlations from Phase 14 and display side-by-side.

**Rationale:** Triangulation — the same legislator's IRT ideal point should correlate positively with both their SM score (voting) and their CFscore (donors). If SM r = 0.93 and DIME r = 0.82, that's expected and informative: our scores are validated by two independent constructs.

## Downstream Implications

1. **Triangulation argument.** When both SM and DIME correlations are positive and significant, the pipeline's credibility is substantially strengthened. No single external source could be this convincing.

2. **89th biennium validation.** This is the only external validation available for 2021-2022 data. SM coverage ends at 2020.

3. **Intra-party interpretation.** CFscore intra-party correlations are expected to be lower than SM intra-party correlations. This is a known limitation of campaign finance data, not a pipeline issue.

4. **Democratic divergence.** If Kansas Democrats show systematically more liberal CFscores than their IRT ideal points, that is consistent with Bonica & Tausanovitch (2022) and does not indicate a measurement failure.
