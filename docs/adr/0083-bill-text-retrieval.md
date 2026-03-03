# ADR-0083: Bill Text Retrieval — Multi-State-Ready Architecture

**Date:** 2026-03-02
**Status:** Accepted

## Context

The tallgrass pipeline had no bill text — only `short_title` (5-15 words from KLISS API) used for NMF topic modeling in Phase 08. Full bill text is the prerequisite for BERTopic, CAP policy classification, bill embeddings, TBIP, and issue-specific IRT (roadmap items BT2-BT5). Bill text retrieval (BT1) is the data acquisition layer that unlocks all downstream NLP work.

Key constraints:
- Multi-state-ready from day one — Kansas first, but adding Nebraska, Missouri, etc. should require only a thin per-state adapter
- No LegiScan dependency (too expensive at scale)
- Kansas Legislature serves bill text exclusively as PDF at deterministic URLs
- Bill text retrieval is independent of vote scraping (different cadence, different data)

## Decision

### Separate subpackage: `src/tallgrass/text/`

New subpackage alongside the existing scraper with its own CLI entry point (`tallgrass-text`), not bolted onto the vote scraper.

```
src/tallgrass/text/
  __init__.py      — public API re-exports
  models.py        — BillDocumentRef, BillText frozen dataclasses
  protocol.py      — StateAdapter Protocol (structural typing)
  kansas.py        — KansasAdapter: bill URL discovery + PDF URL construction
  fetcher.py       — BillTextFetcher: concurrent download + text extraction
  extractors.py    — PDF text extraction, legislative text cleaning
  output.py        — CSV export (bill_texts.csv)
  cli.py           — tallgrass-text entry point
```

### StateAdapter Protocol

Structural typing via `typing.Protocol` defines the contract a state must fulfill:

```python
@runtime_checkable
class StateAdapter(Protocol):
    state_name: str
    def discover_bills(self, session_id: str) -> list[BillDocumentRef]: ...
    def data_dir(self, session_id: str) -> Path: ...
    def cache_dir(self, session_id: str) -> Path: ...
```

Adding a state means writing one new file implementing this protocol. No base class inheritance, no registration — structural subtyping handles dispatch.

### Shared bill discovery module: `src/tallgrass/bills.py`

The vote scraper's `get_bill_urls()` contained ~70 lines of bill URL discovery logic (HTML listing + JS fallback) that the text adapter also needs. Rather than duplicating, extracted into a shared module:

- `discover_bill_urls()` — HTML + JS discovery with injected `get_fn` callable
- `discover_bills()` — returns `list[BillInfo]` with bill numbers and URLs
- `parse_js_array()`, `parse_js_bill_data()` — JS data parsing
- `bill_sort_key()`, `url_to_bill_number()` — utility functions

`KSVoteScraper.get_bill_urls()` delegates to the shared module. `KansasAdapter.discover_bills()` calls the same functions. The `get_fn` callable injection avoids coupling to scraper instance state.

### PDF extraction via pdfplumber

`pdfplumber` (added to core dependencies) handles Kansas Legislature's structured legislative PDFs. Pure-function extractors in `extractors.py`:

- `extract_pdf_text(pdf_bytes)` — `pdfplumber.open(io.BytesIO(pdf_bytes))` extracts text
- `clean_legislative_text(raw_text)` — removes page numbers, extra whitespace, artifacts
- Kansas-specific cleaning (KSA references, boilerplate) in `kansas.py`

### Document types (Phase 1 scope)

Introduced version (`_00_0000.pdf`) and supplemental notes (`supp_note_*_00_0000.pdf`). These cover the primary NLP use cases (topic modeling, policy classification). Committee-amended versions and enrolled versions deferred to Phase 2.

### Output: 5th CSV

`{name}_bill_texts.csv` with columns: `session`, `bill_number`, `document_type`, `version`, `text`, `page_count`, `source_url`. Joins to existing data on `bill_number`.

## Consequences

### Benefits

- **Multi-state ready**: Adding a new state requires one file implementing `StateAdapter`. No framework rewrites.
- **Decoupled from vote scraper**: Separate CLI, separate cadence, separate tests. Vote scraper changes don't break text retrieval.
- **Shared bill discovery**: Zero code duplication between scraper and text adapter. Single source of truth for bill URL patterns.
- **Pure-function extractors**: Testable without I/O, composable, easy to extend.

### Trade-offs

- **New dependency**: `pdfplumber` (+ transitive `pdfminer.six`) added to core deps. Justified: PDF extraction is a core pipeline function, not dev-only.
- **Separate entry point**: Users must know about `tallgrass-text` in addition to `tallgrass`. Mitigated by `just text` recipe.
- **Introduced-only scope**: Phase 1 doesn't capture committee amendments or enrolled versions. The text at time of vote may differ from the introduced text. Acceptable for topic modeling; requires expansion for vote-text matching.

### Test coverage

108 new tests across 7 test files:
- `test_text_models.py` (14), `test_text_extractors.py` (16), `test_text_kansas.py` (29)
- `test_text_fetcher.py` (9), `test_text_output.py` (8), `test_text_cli.py` (9)
- `test_bills.py` (23) for the shared bill discovery module

All 2084 tests pass. 312 scraper tests pass (bill discovery refactor caused zero regressions).
