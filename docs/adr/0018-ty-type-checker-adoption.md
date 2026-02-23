# ADR-0018: ty Type Checker Adoption (Beta)

**Date:** 2026-02-22
**Status:** Accepted

## Context

The project has type hints on all function signatures and uses frozen dataclasses throughout. Without a type checker, these annotations are documentation-only — they don't prevent type errors at development time. The existing toolchain (Ruff) only lints for style and basic errors, not type correctness.

Options considered:
- **mypy**: Mature, widely used, but slow (~10-60x slower than ty) and requires extensive configuration for scientific Python libraries.
- **Pyright**: Fast, good IDE integration, but tightly coupled to VS Code/Pylance ecosystem.
- **ty**: New (beta, Dec 2025), from Astral (same team as Ruff and uv). 10-60x faster than mypy, Rust-based, designed for the same ecosystem we already use.

## Decision

Adopt ty as the project type checker with a **two-tier policy**:

1. **Scraper (`src/`)**: must pass with **zero errors**. All diagnostics are either real bugs or fixable type narrowing issues. This code has well-typed dependencies (stdlib, requests) and benefits from strict checking.

2. **Analysis (`analysis/`)**: errors demoted to **warnings** for three categories dominated by third-party stubs noise:
   - `invalid-argument-type` — Polars scalar-to-float conversions, matplotlib kwargs
   - `unresolved-attribute` — cascading from `replace-imports-with-any` (sklearn `.fit()`, `.predict()`, etc.)
   - `unsupported-operator` — Polars arithmetic on extracted scalars

Libraries with missing or incomplete stubs are configured as `replace-imports-with-any`: PyMC, ArviZ, sklearn, XGBoost, SHAP, UMAP, community, great_tables, css_inline, bs4.

We accept that ty is beta (0.0.x versioning, no stable API). We're both using it for real type error detection and testing its readiness for scientific Python codebases. Configuration may need adjustment as ty matures.

## Consequences

**Benefits:**
- Caught two real type bugs on first run: wrong return type annotation in `report.py`, overly narrow `TextIOBase` in `run_context.py`.
- Sub-second feedback on scraper type correctness (vs minutes for mypy on this codebase).
- Natural fit with existing Ruff + uv toolchain (same vendor, same config patterns).
- `just check` now runs lint + typecheck + tests as a single quality gate.

**Trade-offs:**
- 186 warnings in analysis code are noise (third-party stubs). This will improve as ty and library stubs mature.
- Beta tool — diagnostic behavior may change between versions, requiring config updates.
- No plugin system — can't add custom type inference for Polars or PyMC (ty plans to handle popular libraries directly).
- Must maintain `allowed-unresolved-imports` list as new analysis scripts are added (analysis/ is not a package).
