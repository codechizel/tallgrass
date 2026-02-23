# ADR-0004: HTML Report System

**Date:** 2026-02-19
**Status:** Accepted

## Context

EDA outputs parquet files and PNGs — machine intermediates, not human-readable results. A stats practitioner (grad-level, SPSS background) needs formatted tables with labels, *N* counts, test statistics, effect sizes, and figures — the kind of output you'd get from SPSS's output viewer.

We considered several output formats:
- **PDF (via LaTeX/WeasyPrint):** Heavy dependencies, complex installation, brittle on macOS.
- **Markdown:** No native table formatting, no inline images without external rendering.
- **Jupyter notebook:** Requires Jupyter runtime, not a static deliverable.
- **HTML:** Zero dependencies for viewing, self-contained via base64 images and inline CSS, opens in any browser.

## Decision

Build a component-based HTML report system with three layers:

1. **`analysis/report.py`** — Generic report infrastructure: section dataclasses (`TableSection`, `FigureSection`, `TextSection`), `ReportBuilder`, Jinja2 template, CSS, and `make_gt()` helper for SPSS/APA-style tables.

2. **`analysis/eda_report.py`** (and future `pca_report.py`, `irt_report.py`, etc.) — Analysis-specific report builders. Each adds sections to a `ReportBuilder` instance. The report module knows nothing about EDA.

3. **`analysis/run_context.py`** — Integration point. `RunContext` holds a `ReportBuilder` and writes the HTML in `finalize()` if sections were added.

Key technology choices:
- **great_tables** for table rendering — accepts polars DataFrames natively, produces inline-CSS HTML, APA-style borders via `tab_options()`.
- **Jinja2** for HTML templating — lightweight, no file dependencies (template is a string constant).
- **Base64 PNG embedding** — all plots are embedded inline, producing a single self-contained HTML file.

## Consequences

**Benefits:**
- One-file output that opens anywhere (no server, no runtime).
- SPSS-style tables with proper formatting, labels, and footnotes.
- Component architecture: each analysis phase adds sections independently.
- great_tables handles table styling; report.py only manages layout.
- Reusable across all future analysis scripts.

**Auto-injected sections:**
- `RunContext.finalize()` appends a "Missing Votes" section to every report when `failure_manifest.json` exists in the session's data directory. This ensures data gaps from failed vote page fetches are surfaced in every analysis output without modifying individual analysis scripts. The table is sorted by margin (closest votes first) with close votes (≤ 10) bolded.

**Trade-offs:**
- File size grows with embedded PNGs (~2-5 MB per report with 8 plots).
- great_tables and jinja2 are new dev dependencies.
- No interactivity (static HTML). For interactive exploration, use the parquet files directly.
