# KS Vote Scraper

Scrapes Kansas Legislature roll call votes from kslegislature.gov into CSV files for statistical/Bayesian analysis.

## Commits

- **No Co-Authored-By lines.** Never append co-author trailers.
- Use conventional commits with version tags: `type(scope): description [vYYYY.MM.DD.N]`
- **After every feature/fix:** update relevant docs (CLAUDE.md, ADRs, roadmap, design docs) before committing. Create a new ADR when the change involves a non-obvious architectural decision. Code and docs ship in the same commit.
- Never push without explicit permission — push is manual or by request.
- See `.claude/rules/commit-workflow.md` for full details.

## Commands

```bash
just scrape 2025                             # scrape (cached)
just scrape-fresh 2025                       # scrape (fresh)
just lint                                    # lint + format
just lint-check                              # check only
just typecheck                               # ty type check (src + analysis)
just sessions                                # list available sessions
just check                                   # full check (lint + typecheck + tests)
just umap                                    # UMAP ideological landscape
just indices                                 # classical indices analysis
just betabinom                               # Beta-Binomial Bayesian loyalty
just hierarchical                            # hierarchical Bayesian IRT
just cross-session                           # cross-session validation
just synthesis                               # run synthesis report
just profiles                                # legislator profiles
uv run ks-vote-scraper 2023                  # historical session
uv run ks-vote-scraper 2024 --special        # special session
```

## Build Philosophy

- **Check for existing open source solutions first.** Before building any new feature, analysis method, or tooling, search for well-maintained open source packages that already solve the problem. No need to reinvent the wheel — but don't force it either. If the off-the-shelf options don't meet our needs, require heavy finagling to integrate, or are in the wrong language/ecosystem, build from scratch. A clean custom implementation that fits the project is better than a shoehorned dependency.

## Code Style

- Python 3.14+, use modern type hints (`list[str]` not `List[str]`, `X | None` not `Optional[X]`)
- Ruff: line-length 100, rules E/F/I/W
- ty: type checking (beta, see below)
- Frozen dataclasses for data models
- Type hints on all function signatures

### ty Type Checker (Beta)

We use [ty](https://github.com/astral-sh/ty) (Astral's Rust-based type checker) as an early adopter. It's in beta — we're both using it for real type error detection and testing its readiness for scientific Python projects.

**Policy:**
- `src/` (scraper): must pass with **zero errors**. All diagnostics are real or fixable.
- `analysis/`: errors demoted to **warnings** for categories dominated by third-party stubs noise (Polars scalars, matplotlib kwargs, sklearn/PyMC return types). Real errors should still be fixed.
- Libraries with incomplete stubs (`pymc`, `sklearn`, `bs4`, etc.) are configured as `replace-imports-with-any` in `pyproject.toml` to prevent cascading false positives.

Run `just typecheck` or `just check` (which includes it).

## Architecture

```
src/ks_vote_scraper/
  config.py    - Constants (BASE_URL, delays, workers, user agent)
  session.py   - KSSession: biennium URL resolution, STATE_DIR, data_dir/results_dir properties
               - Properties: uses_odt (pre-2015), js_data_paths (pre-2021 JS bill discovery)
  models.py    - IndividualVote + RollCall dataclasses
  scraper.py   - KSVoteScraper: 4-step pipeline (bill URLs → API filter → vote parse → legislator enrich)
               - FetchResult/FetchFailure/VoteLink dataclasses: typed HTTP results + vote page links
               - FetchResult.content_bytes for binary downloads (ODT files)
               - VoteLink.is_odt flag for routing to appropriate parser
               - JS bill discovery fallback (_get_bill_urls_from_js, _parse_js_bill_data)
               - Member directory (_load_member_directory) for ODT name resolution
               - Pre-2015 party detection (h3 "Party:" fallback)
               - Module constants: _BILL_URL_RE, VOTE_CATEGORIES, _normalize_bill_code()
  odt_parser.py - ODT vote file parser (2011-2014): pure functions, no I/O
               - parse_odt_votes(): main entry point, bytes → (RollCall, IndividualVote, legislators)
               - Extracts content.xml from ZIP, parses user-field-decl metadata
               - Maps House/Senate vote category variants to canonical names
               - Last-name resolution via member directory (handles initials and ambiguity)
  output.py    - CSV export (3 files: votes, rollcalls, legislators)
  cli.py       - argparse CLI entry point
```

Pipeline: `get_bill_urls()` → `_filter_bills_with_votes()` → `get_vote_links()` → `parse_vote_pages()` → `enrich_legislators()` → `save_csvs()`

### Session Coverage (2011-2026)

| Biennium | Bill Discovery | Vote Format | Party Detection | Member Dir |
|----------|---------------|-------------|-----------------|------------|
| 2025-26 (91st) | HTML links | vote_view | `<h2>District N - Party` | HTML links |
| 2023-24 (90th) | HTML links | vote_view | `<h2>` | HTML links |
| 2021-22 (89th) | HTML links | vote_view | `<h2>` | HTML links |
| 2019-20 (88th) | JS (quoted keys) | vote_view | `<h2>` | JS fallback |
| 2017-18 (87th) | JS (unquoted keys) | vote_view | `<h2>` | JS fallback |
| 2015-16 (86th) | JS (unquoted keys) | vote_view | `<h3>Party:` | JS fallback |
| 2013-14 (85th) | JS (unquoted keys) | odt_view | `<h3>Party:` | JS fallback |
| 2011-12 (84th) | JS (unquoted keys) | odt_view* | `<h3>Party:` | JS fallback |

\* 84th session ODTs: ~30% of vote pages contain only tally metadata (no individual legislator names in body text). These are committee-of-the-whole and conference committee votes. The remaining 70% parse normally.

## Concurrency Pattern

All HTTP fetching uses a two-phase pattern: concurrent fetch via ThreadPoolExecutor (MAX_WORKERS=5), then sequential parse. Rate limiting is thread-safe via `threading.Lock()`. Never mutate shared state during the fetch phase.

### Analysis Parallelism

- **MCMC chains** (`irt.py`, `hierarchical.py`): `cores=n_chains` runs chains in parallel via multiprocessing. Each chain gets its own process and deterministic per-chain seed. Results are mathematically identical to sequential execution. `hierarchical.py` also accepts `--cores N` CLI flag for explicit override (used in performance experiments).
- **IRT chain initialization** (`irt.py`): PCA-informed initialization is on by default (`--no-pca-init` to disable). Standardized PC1 scores seed both chains' ideal point parameters, preventing reflection mode-splitting that caused 5 of 16 historical chamber-sessions to fail convergence. See ADR-0023.
- **XGBoost** (`prediction.py`, `cross_session.py`): `n_jobs=-1` enables multi-core tree building via XGBoost's internal C++ thread pool. Deterministic with `random_state`.
- **Random Forest** (`prediction.py`): `n_jobs=-1` for parallel tree fitting.

### Apple Silicon CPU Scheduling (M3 Pro)

The development machine has **6 performance (P) cores + 6 efficiency (E) cores** (12 total). macOS controls core assignment via QoS — **there is no way to pin Python processes to specific cores on Apple Silicon** (`os.sched_setaffinity()` and `psutil.cpu_affinity()` are Linux-only; the Thread Affinity API returns 0 on arm64).

**Key rules for CPU-intensive analysis:**

1. **Run bienniums sequentially, not simultaneously.** Running multiple MCMC jobs at once forces the OS to schedule some chains on E-cores (~50% the IPC of P-cores), causing ~2.5x slowdown. One biennium at a time keeps all chains on P-cores.
2. **Keep parallel chains within a biennium.** `cores=n_chains` (2 chains on 6 P-cores) is optimal — measured 1.83-1.90x faster than sequential. Sequential causes chain 2 to run at ~50% speed due to thermal throttling and E-core migration after chain 1 heats the chip.
3. **Cap internal thread pools.** Justfile recipes for CPU-intensive phases set `OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6` to prevent NumPy/SciPy/PyTensor thread pools (which default to 12 = all cores) from spilling onto E-cores.
4. **Never use `taskpolicy -c background`** for analysis — it forces E-cores only, ~2-3x slower.

See ADR-0022 and the parallel chains performance experiment (`results/experiments/2026-02-23_parallel-chains-performance/`) for measured data.

### Retry Strategy

`_get()` returns a `FetchResult` (not raw HTML) and classifies errors for differentiated retry behavior:

- **404** → `"permanent"`, max 2 attempts (one retry for transient routing guards)
- **5xx** → `"transient"`, exponential backoff (`RETRY_DELAY * 2^attempt`) with jitter
- **Timeout** → `"timeout"`, exponential backoff with jitter
- **Connection error** → `"connection"`, fixed delay
- **Other 4xx** → `"permanent"`, no retry
- **HTTP 200 error page** → `"permanent"`, detected by HTML heuristics (short body, error `<title>`); JSON responses bypass this check

All HTTP requests go through `_get()`, including the KLISS API call. This ensures consistent retries, rate limiting, caching, and error-page detection everywhere.

Failed fetches and parse failures (e.g., 0 votes on a page) are recorded as `FetchFailure` with bill context (bill number, motion text, bill path) and written to a JSON failure manifest at the end of the run.

### Retry Waves

Per-URL retries handle isolated hiccups, but when the server buckles under sustained load (e.g., 55 simultaneous 5xx errors), all workers fail and retry in lockstep (thundering herd). Retry waves solve this at the `_fetch_many()` level:

1. After the initial fetch pass, collect transient failures (5xx, timeout, connection)
2. Wait `WAVE_COOLDOWN` (90s) for the server to recover
3. Re-dispatch failed URLs with reduced load: `WAVE_WORKERS=2`, `WAVE_DELAY=0.5s`
4. Repeat up to `RETRY_WAVES=3` times or until all transient failures resolve

Jitter (`* (1 + random.uniform(0, 0.5))`) on per-URL backoff prevents thundering herd within each wave. See ADR-0009 for rationale.

## HTML Parsing Pitfalls (Hard-Won Lessons)

These are real bugs that were found and fixed. Do NOT regress on them:

1. **Tag hierarchy on vote pages is NOT what you'd expect.** The vote page uses `<h2>` for bill number, `<h4>` for bill title, and `<h3>` for chamber/date/motion AND vote category headings (Yea, Nay, etc). If you search `<h2>` for title or motion data, you get nothing.

2. **Party detection via full page text will always match "Republican".** Every legislator page has a party filter dropdown containing `<option value="republican">Republican</option>`. Searching `page.get_text()` for "Republican" matches this dropdown for ALL legislators. Must parse the specific `<h2>` containing "District \d+" (e.g., `"District 27 - Republican"`).

2b. **Legislator `<h1>` is NOT the member name.** The first `<h1>` on member pages is a generic "Legislators" nav heading. The actual name is in a later `<h1>` starting with "Senator " or "Representative ". Must use `soup.find("h1", string=re.compile(r"^(Senator|Representative)\s+"))`. Also strip leadership suffixes like " - House Minority Caucus Chair".

3. **Vote category parsing requires scanning BOTH h2 and h3.** The `Yea - (33):` heading can appear as either `<h2>` or `<h3>` depending on the page. The parser correctly uses `soup.find_all(["h2", "h3", "a"])` — do not simplify this to only one tag.

4. **KLISS API response structure varies.** Sometimes it's a raw list, sometimes `{"content": [...]}`. Always handle both: `data if isinstance(data, list) else data.get("content", [])`.

5. **Pre-2015 party detection uses a different tag.** Sessions before 2015 show party as `<h3>Party: Republican</h3>` instead of encoding it in the `<h2>District N - Republican</h2>` tag. The parser checks `<h2>` first (current format), then falls back to `<h3>Party:` if empty.

6. **Pre-2021 bill lists are JavaScript-rendered.** HTML listing pages for sessions before 2021 have zero `<a>` tags for bills. The JS fallback fetches `bills_li_{end_year}.js` data files and extracts `measures_url` from the JSON array.

6b. **Pre-2021 JS data uses two different key formats.** The 88th (2019-2020) uses quoted JSON keys (`"measures_url":`), but all older sessions (87th and earlier) use unquoted JavaScript object literal syntax (`measures_url:`). The parser applies `re.sub(r"(?m)^(\s+)(\w+):", ...)` to quote bare keys before `json.loads()`.

6c. **JS data files live at `/m/` not `/s/` for all sessions except the 88th.** The scraper tries both paths (`/s/` then `/m/`) and uses whichever responds. The 88th (2019-2020) JS is at `/li_2020/s/js/data/`; all others are at `/li_{year}/m/js/data/`.

7. **Pre-2015 vote pages are ODT files, not HTML.** Sessions 2011-12 and 2013-14 link to `.odt` (OpenDocument) files via `odt_view` URLs. These are ZIP archives with `content.xml` containing user-field metadata and comma-separated legislator names. House and Senate use different vote category names ("Present but not voting" vs "Present and Passing"). Names are last-name-only, requiring a member directory lookup for slug resolution.

8. **Pre-2021 member directories are JavaScript-rendered.** Same as bill lists — the `/members/` page has zero `<a>` tags with slugs. The fallback parses `<script src="...members_list...">` tags and fetches `senate_members_list_li_{end_year}.js` and `house_members_list_li_{end_year}.js`. Same unquoted-key issue as bill data.

9. **The KS Legislature server returns HTML error pages with HTTP 200 for binary URLs.** Some `.odt` URLs return a 169-byte HTML 404 page with HTTP 200 status. For binary fetches, the `_get()` method checks if content starts with `<html` to detect these. Without this check, the bad bytes get cached as `.bin` and the ODT parser silently returns 0 votes.

10. **84th session (2011-12) ODTs often lack individual vote data.** ~30% of vote pages contain only tally metadata (e.g., "41:73" in `T_JE_T_VOTE`) with no "Yeas:/Nays:" section in the body text. These are committee-of-the-whole and conference committee votes. They produce "0 votes parsed from ODT" failures — this is a data limitation, not a parser bug.

## Session URL Logic

The KS Legislature uses different URL prefixes per session — this is the single trickiest part of the scraper:

- Current (2025-26): `/li/b2025_26/...`
- Historical (2023-24): `/li_2024/b2023_24/...`
- Special (2024): `/li_2024s/...`
- API paths also differ: current uses `/li/api/v13/rev-1`, historical uses `/li_{end_year}/api/v13/rev-1`

`CURRENT_BIENNIUM_START` in session.py must be updated when a new biennium becomes current (next: 2027).

## Data Model Notes

- `vote_id` encodes a timestamp: `je_20250320203513` → `2025-03-20T20:35:13`
- `bill_metadata` (short_title, sponsor) comes from the KLISS API — already fetched during pre-filtering, no extra requests needed
- `passed` is derived from result text: passed/adopted/prevailed/concurred → True; failed/rejected/sustained → False; procedural motions that don't match any pattern → null
- Vote categories: defined in `VOTE_CATEGORIES` constant — Yea, Nay, Present and Passing, Absent and Not Voting, Not Voting (exactly these 5)
- Legislator slugs encode chamber: `sen_` prefix = Senate, `rep_` prefix = House

## Output

Three CSVs in `data/kansas/{legislature}_{start}-{end}/` (e.g. `data/kansas/91st_2025-2026/`):
- `{output_name}_votes.csv` — ~68K rows, one per legislator per roll call (deduplicated by `legislator_slug` + `vote_id`)
- `{output_name}_rollcalls.csv` — ~500 rows, one per roll call
- `{output_name}_legislators.csv` — ~172 rows, one per legislator

Vote deduplication: ODT sessions (2011-2014) can link the same vote page from multiple bills, producing duplicate `(legislator_slug, vote_id)` pairs. `save_csvs()` deduplicates by this key, keeping the first occurrence. The 85th session went from 132K to 92K after dedup; the 84th from 74K to 68K.

Cache lives in `data/kansas/{output_name}/.cache/`. Use `--clear-cache` to force fresh fetches.

The directory naming uses the Kansas Legislature numbering scheme: `(start_year - 1879) // 2 + 18` gives the legislature number (e.g., 91 for 2025-2026). Special sessions use `{year}s` (e.g., `2024s`).

When vote page fetches fail, a `failure_manifest.json` is written alongside the CSVs with per-failure context (bill number, motion, URL, status code, error type). A companion `missing_votes.md` is also written with a human-readable table sorted by vote margin (closest votes first, bolded if margin ≤ 10). Failed pages are never cached, so re-running the scraper automatically retries them.

## Results Directory

Analysis outputs go in `results/` (gitignored), organized by state, session, analysis type, and run date:

```
results/
  kansas/
    91st_2025-2026/
      eda_report.html → eda/latest/eda_report.html      ← Convenience symlinks
      pca_report.html → pca/latest/pca_report.html
      ...
      synthesis_report.html → synthesis/latest/...
      eda/
        2026-02-19/
          plots/                  ← PNGs
          data/                   ← Parquet intermediates
          filtering_manifest.json ← What was filtered and why
          run_info.json           ← Git hash, timestamp, parameters, run_label
          run_log.txt             ← Captured console output
        2026-02-19.1/             ← Second run same day (preserves first)
        2026-02-19.2/             ← Third run same day, etc.
        latest → 2026-02-19.2/   ← Symlink to most recent run
      pca/                        ← Same structure
      irt/
      clustering/
      network/
      prediction/
      indices/
      synthesis/                  ← Joins all phases into one narrative report
```

Same-day runs are never clobbered. The first run uses the bare date (`2026-02-19`), subsequent runs append `.1`, `.2`, etc. The `latest` symlink always points to the most recent run. This preserves earlier runs for comparison after fixes or reruns.

All analysis scripts use `RunContext` from `analysis/run_context.py` as a context manager to get structured output. Downstream scripts read from `results/kansas/<session>/<analysis>/latest/`. The module also exports `strip_leadership_suffix()` — a shared utility that removes " - President of the Senate" and similar titles from legislator names. Every phase that reads the legislators CSV applies this at load time so display names are clean throughout the pipeline.

Results paths use the biennium naming scheme: `91st_2025-2026` (matching the data directory). The `kansas/` state directory is controlled by `STATE_DIR` in `session.py` (see ADR-0016).

### Independent Party Handling

The scraper only detects "Republican" and "Democrat" — legislators with other affiliations (e.g., Dennis Pyle as "Independent" in 2021-22) get an empty party string. Every analysis phase fills null/empty party to "Independent" at CSV load time: `pl.col("party").fill_null("Independent").replace("", "Independent")`. All 12 modules define `PARTY_COLORS` with `"Independent": "#999999"`. Party-specific models (hierarchical IRT, party unity/maverick, beta-binomial) exclude Independents. Plots and legends dynamically iterate over parties present in data rather than hardcoding R/D. See ADR-0021.

## Architecture Decision Records

Significant technical decisions are documented in `docs/adr/`. See `docs/adr/README.md` for the full index and template.

## Analysis Primer

`docs/analysis-primer.md` is a plain-English guide to the full 13-step analysis pipeline, written for a general audience (journalists, policymakers, engaged citizens). It walks through each phase in order with analogies, real Kansas data examples, and an honest discussion of limitations. Share this with anyone who wants to understand what the analysis does without reading code or design docs.

## Field Landscape and Method Evaluation

`docs/landscape-legislative-vote-analysis.md` surveys the legislative analysis field: NOMINATE (Poole & Rosenthal), Bayesian IRT (Clinton-Jackman-Rivers), emIRT (Imai-Lo-Olmsted), state-level projects (Shor-McCarty, DIME), ML approaches, network analysis, and the software ecosystem. ~50 references.

`docs/method-evaluation.md` evaluates every major method against our data and pipeline. **Key finding:** external validation with Shor-McCarty scores is the highest-priority gap. Nine other methods were evaluated and rejected with data-specific rationale (vote-type-stratified IRT, strategic absence modeling, GGUM, dynamic within-biennium IRT, etc.).

## Analysis Design Choices

Each analysis phase has a design document in `analysis/design/` recording the statistical assumptions, priors, thresholds, and methodological choices that shape results and carry forward into downstream phases. **Read these before interpreting results or adding a new phase.** See `analysis/design/README.md` for the index.

- `analysis/design/eda.md` — Binary encoding, filtering thresholds, agreement metrics
- `analysis/design/pca.md` — Imputation, standardization, sign convention, holdout design
- `analysis/design/irt.md` — Priors (Normal(0,1) discrimination, anchors), MCMC settings, missing data handling, PCA-informed chain initialization (default on)
- `analysis/design/clustering.md` — Three methods for robustness, party loyalty metric, k=2 finding
- `analysis/design/prediction.md` — XGBoost primary, IRT features dominate, NLP topic features (NMF on short_title), target leakage assessment
- `analysis/design/beta_binomial.md` — Empirical Bayes, per-party-per-chamber priors, method of moments, shrinkage factor
- `analysis/design/synthesis.md` — Data-driven detection thresholds, graceful degradation, template narratives
- `analysis/design/cross_session.md` — Affine IRT alignment, name matching, shift metrics, prediction transfer, detection validation

## Analytics

The `Analytic_Methods/` directory contains 28 documents covering every analytical method applicable to our data. See `Analytic_Methods/00_overview.md` for the full index and recommended pipeline.

### Document Naming Convention
```
NN_CAT_method_name.md
```
Categories: `DATA`, `EDA`, `IDX`, `DIM`, `BAY`, `CLU`, `NET`, `PRD`, `TSA`

### Key Data Structures for Analysis

The **vote matrix** (legislators x roll calls, binary) is the foundation. Build it from `votes.csv` by pivoting `legislator_slug` x `vote_id` with Yea=1, Nay=0, absent=NaN.

**Critical preprocessing:**
- Filter near-unanimous votes (minority < 2.5%) — they carry no ideological signal
- Filter legislators with < 20 votes — unreliable estimates
- Analyze chambers separately (House and Senate vote on different bills)

### Technology Preferences
- **Polars over pandas** for all data manipulation
- **Python over R** — no rpy2 or Rscript; use Python-native methods (PCA, Bayesian IRT) instead of R-only ones (W-NOMINATE, OC)

### Analysis Libraries
```bash
uv add polars numpy scipy matplotlib seaborn scikit-learn  # Phase 1-2
uv add networkx python-louvain prince umap-learn           # Phase 3-5
uv add pymc arviz                                          # Phase 4 (Bayesian)
uv add xgboost shap ruptures                               # Phase 6
```

### HTML Report System

Each analysis phase produces a self-contained HTML report (`{analysis}_report.html`) with SPSS/APA-style tables and embedded plots. Architecture:

- `analysis/report.py` — Generic: section types (`TableSection`, `FigureSection`, `TextSection`), `ReportBuilder`, `make_gt()` helper, Jinja2 template + CSS.
- `analysis/eda_report.py` — EDA-specific: `build_eda_report()` adds ~19 sections.
- `analysis/umap_viz.py` + `analysis/umap_report.py` — UMAP: nonlinear dimensionality reduction on vote matrix (cosine metric, n_neighbors=15). Produces ideological landscape plots, validates against PCA/IRT via Spearman, Procrustes sensitivity sweep.
- `analysis/nlp_features.py` — NLP: TF-IDF + NMF topic modeling on bill `short_title` text. Pure data logic (no I/O). `TopicModel` frozen dataclass, `fit_topic_features()`, `get_topic_display_names()`, `plot_topic_words()`. See ADR-0012.
- `analysis/beta_binomial.py` + `analysis/beta_binomial_report.py` — Beta-Binomial: empirical Bayes shrinkage on CQ party unity scores. Closed-form Beta posteriors per legislator, 4 plots per chamber. Reads from indices `party_unity_{chamber}.parquet`. See ADR-0015.
- `analysis/hierarchical.py` + `analysis/hierarchical_report.py` — Hierarchical IRT: 2-level partial pooling by party on full IRT model. Non-centered parameterization, ordering constraint, variance decomposition (ICC), shrinkage comparison vs flat IRT. 5 plots per chamber + optional joint model. See ADR-0017.
- `analysis/synthesis_detect.py` — Detection: pure data logic that identifies notable legislators (mavericks, bridge-builders, metric paradoxes) from upstream DataFrames. Returns frozen dataclasses with pre-formatted titles and subtitles.
- `analysis/synthesis.py` + `analysis/synthesis_report.py` — Synthesis: loads upstream parquets from all 9 phases (incl. beta_binomial), joins into unified legislator DataFrames, runs data-driven detection, produces 29-33 section narrative HTML report for nontechnical audiences. No hardcoded legislator names.
- `analysis/profiles_data.py` — Profiles data logic: `ProfileTarget`/`BillTypeBreakdown` dataclasses, `gather_profile_targets()`, `build_scorecard()`, `compute_bill_type_breakdown()`, `find_defection_bills()`, `find_voting_neighbors()`, `find_legislator_surprising_votes()`.
- `analysis/profiles.py` + `analysis/profiles_report.py` — Profiles: deep-dive per-legislator reports with scorecard, bill-type breakdown, defection analysis, voting neighbors, surprising votes. 5 plots per legislator. Reuses `load_all_upstream()`/`build_legislator_df()` from synthesis.
- `analysis/cross_session_data.py` — Cross-session: pure data logic for legislator matching (by normalized name), IRT scale alignment (robust affine transform), ideology shift metrics, metric stability correlations, turnover impact, prediction transfer helpers (feature alignment, z-score standardization, SHAP comparison). No I/O. See ADR-0019.
- `analysis/cross_session.py` + `analysis/cross_session_report.py` — Cross-session validation: compares two bienniums. 6 plot types (ideology scatter, biggest movers, shift distribution, turnover impact, prediction AUC comparison, feature importance comparison), detection threshold validation, metric stability correlations, cross-session prediction transfer. CLI: `--session-a`, `--session-b`, `--chambers`, `--skip-prediction`. Output: `results/kansas/cross-session/<pair>/validation/` (e.g., `cross-session/90th-vs-91st/validation/`).
- `RunContext` auto-writes the HTML in `finalize()` if sections were added. Records wall-clock elapsed time (`elapsed_seconds`, `elapsed_display`) in `run_info.json` and displays "Runtime: Xm Ys" in the HTML report header. If a `failure_manifest.json` exists in the session's data directory, a "Missing Votes" section is automatically appended to every report (sorted by margin, close votes bolded).

Tables use great_tables with polars DataFrames (no pandas conversion). Plots are base64-embedded PNGs. See ADR-0004 for rationale.

### Analytic Flags

`docs/analytic-flags.md` is a living document that accumulates observations across analysis phases. Add an entry whenever you encounter:
- **Outlier legislators** — extreme scores, unusual voting patterns, imputation artifacts
- **Methodological flags** — legislators needing special handling (e.g., bridging observations for cross-chamber members, low-participation exclusions)
- **Unexpected patterns** — dimensions or clusters that need qualitative explanation
- **Downstream actions** — things to check or handle differently in later phases (IRT, clustering, network)

Each entry records what was observed, which phase found it, why it matters, and what to do about it. Check this file at the start of each new analysis phase for prior flags that affect the current work.

### Kansas-Specific Analysis Notes
- Republican supermajority (~72%) means intra-party variation is more interesting than inter-party
- Clustering (2026-02-20) found k=2 optimal (party split); initial k=3 hypothesis rejected — intra-R variation is continuous
- 34 veto override votes are analytically rich (cross-party coalitions, 2/3 threshold)
- Beta-Binomial and Bayesian IRT are the recommended Bayesian starting points

## Testing

```bash
just test                    # run all tests
just test-scraper            # scraper tests only (all scraper modules)
uv run pytest tests/ -v      # pytest directly
uv run ruff check src/       # lint clean
```

### Scraper Test Files
- `tests/conftest.py` — shared KSSession fixtures (current, historical, special)
- `tests/test_session.py` — session URL resolution, biennium logic, uses_odt, js_data_paths (~40 tests)
- `tests/test_scraper_pure.py` — pure functions: bill codes, datetime parsing, result derivation, JS bill parsing, binary FetchResult (~45 tests)
- `tests/test_scraper_html.py` — HTML parsing with inline BeautifulSoup fixtures, pre-2015 party detection, odt_view link detection (~35 tests)
- `tests/test_models.py` — dataclass construction and immutability, VoteLink.is_odt (~8 tests)
- `tests/test_odt_parser.py` — ODT vote parsing: XML extraction, metadata, body text, vote categories, name resolution, integration (~47 tests)
- `tests/test_output.py` — CSV export: filenames, headers, row counts, roundtrip (~10 tests)
- `tests/test_cli.py` — argument parsing with monkeypatched scraper (~17 tests)

### Analysis Infrastructure Test Files
- `tests/test_run_context.py` — TeeStream, session normalization, strip_leadership_suffix, RunContext lifecycle, same-day run labeling (~41 tests)
- `tests/test_report.py` — section rendering, format parsing, ReportBuilder, make_gt, elapsed display (~38 tests)
- `tests/test_irt.py` — IRT data prep, anchor selection, sensitivity filter, joint model, forest highlights, paradox detection (~45 tests)
- `tests/test_umap_viz.py` — imputation, orientation, embedding construction, Procrustes, validation correlations (~21 tests)
- `tests/test_nlp_features.py` — TF-IDF + NMF fitting, edge cases, display names, TopicModel dataclass (~16 tests)
- `tests/test_pca.py` — imputation, PC1 orientation, extreme PC2 detection (~16 tests)
- `tests/test_prediction.py` — vote/bill features, model training, SHAP, per-legislator, NLP integration, hardest-to-predict detection, surprising bills schema (~38 tests)
- `tests/test_beta_binomial.py` — method of moments estimation, Bayesian posteriors, shrinkage properties, edge cases (~26 tests)
- `tests/test_hierarchical.py` — hierarchical data prep, model structure, result extraction, variance decomposition, shrinkage comparison (~26 tests)
- `tests/test_profiles.py` — profile target selection, scorecard, bill-type breakdown, defections, voting neighbors, surprising votes (~23 tests)
- `tests/test_cross_session.py` — legislator matching, IRT alignment, ideology shift, metric stability, turnover impact, feature alignment, standardization, SHAP comparison (~55 tests)

### Manual Verification
- Run scraper with `--clear-cache`, check that `vote_date`, `chamber`, `motion`, `bill_title` are populated
- Check legislators CSV: party distribution includes both Republican and Democrat
- Spot-check SB 1: should show Senate, Emergency Final Action, Passed as amended
