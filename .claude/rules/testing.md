---
paths:
  - "tests/**/*.py"
---

# Testing

## Commands

```bash
just test                    # run all tests (~1125)
just test-scraper            # scraper tests only
just check                   # full check (lint + typecheck + tests)
uv run pytest tests/ -v      # pytest directly
```

## Conventions

- Class-based test organization with docstrings including run command
- `# -- Section --` headers to group related tests
- Inline fixtures (HTML parsing uses inline BeautifulSoup, not separate files)
- CLI tests use monkeypatched FakeScraper class
- RunContext tests use `tmp_path` for isolated filesystem operations

## Scraper Test Files

- `tests/conftest.py` — shared KSSession fixtures (current, historical, special)
- `tests/test_session.py` — session URL resolution, biennium logic, uses_odt, js_data_paths (~40 tests)
- `tests/test_scraper_pure.py` — pure functions: bill codes, datetime parsing, result derivation, JS parsing (~45 tests)
- `tests/test_scraper_html.py` — HTML parsing with inline fixtures, pre-2015 party detection, odt_view links (~35 tests)
- `tests/test_models.py` — dataclass construction and immutability, VoteLink.is_odt (~8 tests)
- `tests/test_odt_parser.py` — ODT vote parsing: XML, metadata, body text, name resolution (~47 tests)
- `tests/test_scraper_http.py` — HTTP layer: _get() retries, error classification, cache, _fetch_many() waves, rate limiting, KLISS API (~44 tests)
- `tests/test_output.py` — CSV export: filenames, headers, row counts, roundtrip (~10 tests)
- `tests/test_cli.py` — argument parsing with monkeypatched scraper (~17 tests)

## Analysis Test Files

- `tests/test_run_context.py` — TeeStream, session normalization, strip_leadership_suffix, lifecycle (~41 tests)
- `tests/test_eda.py` — vote matrix, filtering, agreement, Rice, party-line, integrity, new diagnostics (~28 tests)
- `tests/test_report.py` — section rendering, format parsing, ReportBuilder, make_gt, elapsed (~38 tests)
- `tests/test_irt.py` — IRT data prep, anchor selection, sensitivity, forest, paradox detection, convergence diagnostics, posterior extraction, equating (~73 tests)
- `tests/test_umap_viz.py` — imputation, orientation, embedding, Procrustes, validation, trustworthiness, sensitivity sweep, stability, three-party (~40 tests)
- `tests/test_nlp_features.py` — TF-IDF + NMF fitting, edge cases, display names (~16 tests)
- `tests/test_pca.py` — imputation, PC1 orientation, extreme PC2 detection (~16 tests)
- `tests/test_prediction.py` — vote/bill features, model training, SHAP, NLP integration, holdout eval, baselines, proper scoring rules (~54 tests)
- `tests/test_beta_binomial.py` — method of moments, posteriors, shrinkage, edge cases (~26 tests)
- `tests/test_hierarchical.py` — hierarchical data prep, model structure, variance decomposition, small-group warning, joint ordering, rescaling fallback, Independent exclusion (~35 tests)
- `tests/test_profiles.py` — profile targets, scorecard, bill-type breakdown, defections, name resolution (~36 tests)
- `tests/test_cross_session.py` — matching, IRT alignment, shift, stability, prediction transfer, detection, plot smoke tests, report (~73 tests)
- `tests/test_clustering.py` — party loyalty, cross-method ARI, within-party, kappa distance, hierarchical, spectral, HDBSCAN, characterization (~70 tests)
- `tests/test_network.py` — network construction, centrality, Leiden/CPM community detection, bridges, threshold sweep, polarization, disparity filter backbone, extreme edge weights (~53 tests)
- `tests/test_indices.py` — Rice formula, party votes, ENP, unity/maverick, co-defection, Carey UNITY, fractured votes (~37 tests)
- `tests/test_synthesis.py` — synthesis data loading, build_legislator_df joins, _extract_best_auc, detect_all integration, minority mavericks, Democrat-majority paradox (~47 tests)
- `tests/test_synthesis_detect.py` — maverick, bridge-builder, metric paradox detection, annotation slugs (~25 tests)
- `tests/test_external_validation.py` — SM name normalization, parsing, biennium filtering, matching, correlations, outliers (~65 tests)

## Manual Verification

- Run scraper with `--clear-cache`, check `vote_date`, `chamber`, `motion`, `bill_title` populated
- Check legislators CSV: party includes both Republican and Democrat
- Spot-check SB 1: Senate, Emergency Final Action, Passed as amended
