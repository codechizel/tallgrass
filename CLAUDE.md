# KS Vote Scraper

Scrapes Kansas Legislature roll call votes from kslegislature.gov into CSV files for statistical/Bayesian analysis. Coverage: 2011-2026 (84th-91st legislatures).

## Commits

- **No Co-Authored-By lines.** Never append co-author trailers.
- Use conventional commits with version tags: `type(scope): description [vYYYY.MM.DD.N]`
- **After every feature/fix:** update relevant docs (CLAUDE.md, ADRs, design docs) before committing. Code and docs ship in the same commit.
- Never push without explicit permission.
- See `.claude/rules/commit-workflow.md` for types, scopes, and full details.

## Commands

[Just](https://github.com/casey/just) is used as a command runner — thin aliases over `uv run` commands. The `Justfile` also sets `OMP_NUM_THREADS=6` and `OPENBLAS_NUM_THREADS=6` globally to cap thread pools on Apple Silicon (ADR-0022). Run `just --list` to see all recipes, or use the underlying `uv run` commands directly.

```bash
just scrape 2025                             # → uv run ks-vote-scraper 2025
just scrape-fresh 2025                       # → uv run ks-vote-scraper --clear-cache 2025
just lint                                    # → ruff check --fix + ruff format
just lint-check                              # → ruff check + ruff format --check
just typecheck                               # → ty check src/ + ty check analysis/
just sessions                                # → uv run ks-vote-scraper --list-sessions
just check                                   # → lint-check + typecheck + test (quality gate)
just test                                    # → uv run pytest tests/ -v (~1125 tests)
just test-scraper                            # → pytest on scraper test files only
uv run ks-vote-scraper 2023                  # historical session (direct)
uv run ks-vote-scraper 2024 --special        # special session (direct)
```

Analysis recipes (all pass `*args` through to the underlying script):

`just eda`, `just pca`, `just umap`, `just irt`, `just indices`, `just betabinom`, `just hierarchical`, `just synthesis`, `just profiles`, `just cross-session`, `just external-validation`.

Each maps to `uv run python analysis/NN_phase/phase.py`. Example: `just profiles --names "Masterson"` runs `uv run python analysis/12_profiles/profiles.py --names "Masterson"`.

## Build Philosophy

- **Check for existing open source solutions first.** Don't reinvent the wheel, but don't force a shoehorned dependency either.

## Code Style

- Python 3.14+, modern type hints (`list[str]` not `List[str]`, `X | None` not `Optional[X]`)
- Ruff: line-length 100, rules E/F/I/W
- ty: type checking (beta) — `src/` must pass clean; `analysis/` warnings-only for third-party stub noise
- Frozen dataclasses for data models; type hints on all function signatures
- Libraries with incomplete stubs configured as `replace-imports-with-any` in `pyproject.toml`

## Architecture

```
src/ks_vote_scraper/
  config.py     - Constants (BASE_URL, delays, workers, user agent)
  session.py    - KSSession: biennium URL resolution, STATE_DIR, data_dir/results_dir
  models.py     - IndividualVote + RollCall dataclasses
  scraper.py    - KSVoteScraper: 4-step pipeline (bill URLs -> API filter -> vote parse -> enrich)
  odt_parser.py - ODT vote file parser (2011-2014): pure functions, no I/O
  output.py     - CSV export (3 files: votes, rollcalls, legislators)
  cli.py        - argparse CLI entry point
```

Pipeline: `get_bill_urls()` -> `_filter_bills_with_votes()` -> `get_vote_links()` -> `parse_vote_pages()` -> `enrich_legislators()` -> `save_csvs()`

See `.claude/rules/scraper-architecture.md` for session coverage table, retry strategy, concurrency details, and ODT parsing.

## HTML Parsing Pitfalls (Hard-Won Lessons)

These are real bugs that were found and fixed. Do NOT regress on them:

1. **Tag hierarchy on vote pages is NOT what you'd expect.** `<h2>` = bill number, `<h4>` = bill title, `<h3>` = chamber/date/motion AND vote category headings.

2. **Party detection via full page text will always match "Republican".** Every legislator page has a party filter dropdown. Must parse the specific `<h2>` containing "District \d+".

2b. **Legislator `<h1>` is NOT the member name.** First `<h1>` is a nav heading. Must use `soup.find("h1", string=re.compile(r"^(Senator|Representative)\s+"))`. Also strip leadership suffixes.

3. **Vote category parsing requires scanning BOTH h2 and h3.** Uses `soup.find_all(["h2", "h3", "a"])` — do not simplify.

4. **KLISS API response structure varies.** Raw list or `{"content": [...]}`. Always handle both.

5. **Pre-2015 party detection uses `<h3>Party: Republican</h3>`** instead of `<h2>District N - Republican</h2>`.

6. **Pre-2021 bill lists are JavaScript-rendered.** JS fallback fetches `bills_li_{end_year}.js` data files.

6b. **Pre-2021 JS data uses two key formats.** 88th uses quoted JSON keys; 87th and earlier use unquoted JS object literal syntax.

6c. **JS data files live at `/m/` not `/s/` for all sessions except the 88th.**

7. **Pre-2015 vote pages are ODT files, not HTML.** ZIP archives with `content.xml`. House/Senate use different vote category names.

8. **Pre-2021 member directories are JavaScript-rendered.** Same unquoted-key issue as bill data.

9. **KS Legislature server returns HTML error pages with HTTP 200 for binary URLs.** `_get()` checks `content[:5]` for `<html` prefix.

10. **84th session ODTs often lack individual vote data.** ~30% are committee-of-the-whole (tally-only). Not a parser bug.

## Session URL Logic

- Current (2025-26): `/li/b2025_26/...`
- Historical (2023-24): `/li_2024/b2023_24/...`
- Special (2024): `/li_2024s/...`
- API: current uses `/li/api/v13/rev-1`, historical uses `/li_{end_year}/api/v13/rev-1`
- `CURRENT_BIENNIUM_START` in session.py must be updated when a new biennium becomes current (next: 2027).

## Data Model

- `vote_id` encodes a timestamp: `je_20250320203513` -> `2025-03-20T20:35:13`
- `passed`: passed/adopted/prevailed/concurred -> True; failed/rejected/sustained -> False; else null
- Vote categories: Yea, Nay, Present and Passing, Absent and Not Voting, Not Voting (exactly 5)
- Legislator slugs: `sen_` = Senate, `rep_` = House
- Independent party handling: scraper outputs empty string; all analysis fills to "Independent" at load time (ADR-0021)

## Output

Three CSVs in `data/kansas/{legislature}_{start}-{end}/`:
- `{name}_votes.csv` — one per legislator per roll call (deduped by `legislator_slug` + `vote_id`)
- `{name}_rollcalls.csv` — one per roll call
- `{name}_legislators.csv` — one per legislator

Directory naming: `(start_year - 1879) // 2 + 18` -> legislature number. Special sessions: `{year}s`.
Cache: `data/kansas/{name}/.cache/`. Failed fetches -> `failure_manifest.json` + `missing_votes.md`.
External data: `data/external/shor_mccarty.tab` (Shor-McCarty scores, auto-downloaded from Harvard Dataverse).

## Results Directory

Analysis outputs in `results/kansas/{session}/{analysis}/{date}/` with `latest` symlink. Same-day runs append `.1`, `.2`, etc. `RunContext` manages structured output, elapsed timing, HTML reports, and auto-primers.

## Analysis Pipeline

Phases live in numbered subdirectories (`analysis/01_eda/`, `analysis/07_indices/`, etc.). A PEP 302 meta-path finder in `analysis/__init__.py` redirects `from analysis.eda import X` to `analysis/01_eda/eda.py` — zero import changes needed (ADR-0030). Shared infrastructure (`run_context.py`, `report.py`) stays at the root.

See `.claude/rules/analysis-framework.md` for the full 12-phase pipeline, report system architecture, and design doc index. See `.claude/rules/analytic-workflow.md` for methodology rules, validation requirements, and audience guidance.

Key references:
- Design docs: `analysis/design/README.md`
- ADRs: `docs/adr/README.md` (38 decisions)
- Analysis primer: `docs/analysis-primer.md` (plain-English guide)
- External validation: `docs/external-validation-results.md` (general-audience results article)
- Hierarchical deep dive: `docs/hierarchical-shrinkage-deep-dive.md` (over-shrinkage analysis with literature)
- PCA deep dive: `docs/pca-deep-dive.md` (literature review, code audit, implementation recommendations)
- UMAP deep dive: `docs/umap-deep-dive.md` (literature survey, code review, implementation recommendations)
- IRT deep dive: `docs/irt-deep-dive.md` (field survey, code audit, test gaps, recommendations)
- IRT field survey: `docs/irt-field-survey.md` (identification problem, unconstrained β contribution, Python ecosystem gap)
- Clustering deep dive: `docs/clustering-deep-dive.md` (literature survey, code audit, test gaps, recommendations)
- Hierarchical IRT deep dive: `docs/hierarchical-irt-deep-dive.md` (ecosystem survey, code audit, 9 issues fixed, 35 tests, ADR-0033)
- Prediction deep dive: `docs/prediction-deep-dive.md` (literature survey, code audit, IRT circularity analysis, test gaps)
- Beta-Binomial deep dive: `docs/beta-binomial-deep-dive.md` (ecosystem survey, code audit, ddof fix, Tarone's test)
- Synthesis deep dive: `docs/synthesis-deep-dive.md` (field survey, code audit, detection algorithms, test gaps, refactoring)
- Profiles deep dive: `docs/profiles-deep-dive.md` (code audit, ecosystem survey, name-based lookup)
- Cross-session deep dive: `docs/cross-session-deep-dive.md` (ecosystem survey, code audit, 3 bugs fixed, 18 new tests)
- Scraper deep dive: `docs/scraper-deep-dive.md` (ecosystem comparison, code audit, data quality review, test gap analysis)
- Future bill text analysis: `docs/future-bill-text-analysis.md` (bb25, topic modeling, retrieval, open questions)
- Apple Silicon MCMC tuning: `docs/apple-silicon-mcmc-tuning.md` (P/E core scheduling, thread pool caps, parallel chains, batch job rules)
- Ward linkage article: `docs/ward-linkage-non-euclidean.md` (why Ward on Kappa distances is impure, the fix)
- Analytic flags: `docs/analytic-flags.md` (living document of observations)
- Field survey: `docs/landscape-legislative-vote-analysis.md`
- Method evaluation: `docs/method-evaluation.md`

## Concurrency

- **Scraper**: concurrent fetch via ThreadPoolExecutor (MAX_WORKERS=5), sequential parse. Never mutate shared state during fetch.
- **MCMC**: `cores=n_chains` for parallel chains. PCA-informed init default (ADR-0023).
- **Apple Silicon (M3 Pro, 6P+6E)**: run bienniums sequentially; cap thread pools (`OMP_NUM_THREADS=6`); never use `taskpolicy -c background`. See ADR-0022.

## Testing

```bash
just test                    # 1125 tests
just test-scraper            # scraper tests only
just check                   # full check (lint + typecheck + tests)
```

See `.claude/rules/testing.md` for test file inventory and conventions.
