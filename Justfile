# KS Vote Scraper — Command Runner

# Cap thread pools to P-core count (6) to prevent E-core spillover on Apple Silicon.
# See ADR-0022 and results/experiments/2026-02-23_parallel-chains-performance/.
export OMP_NUM_THREADS := "6"
export OPENBLAS_NUM_THREADS := "6"

# Default: show available commands
default:
    @just --list

# Scrape current session (cached)
scrape *args:
    uv run ks-vote-scraper {{args}}

# Scrape with fresh cache
scrape-fresh *args:
    uv run ks-vote-scraper --clear-cache {{args}}

# Lint and format
lint:
    uv run ruff check --fix src/
    uv run ruff format src/

# Lint check only (no fix)
lint-check:
    uv run ruff check src/
    uv run ruff format --check src/

# Install dependencies
install:
    uv sync

# List available sessions
sessions:
    uv run ks-vote-scraper --list-sessions

# Run EDA analysis
eda *args:
    uv run python analysis/eda.py {{args}}

# Run PCA analysis
pca *args:
    uv run python analysis/pca.py {{args}}

# Run Bayesian IRT analysis
irt *args:
    uv run python analysis/irt.py {{args}}

# Run clustering analysis
clustering *args:
    uv run python analysis/clustering.py {{args}}

# Run network analysis
network *args:
    uv run python analysis/network.py {{args}}

# Run prediction analysis
prediction *args:
    uv run python analysis/prediction.py {{args}}

# Run UMAP analysis
umap *args:
    uv run python analysis/umap_viz.py {{args}}

# Run classical indices analysis
indices *args:
    uv run python analysis/indices.py {{args}}

# Run Beta-Binomial Bayesian party loyalty
betabinom *args:
    uv run python analysis/beta_binomial.py {{args}}

# Run hierarchical Bayesian IRT
hierarchical *args:
    uv run python analysis/hierarchical.py {{args}}

# Run synthesis report
synthesis *args:
    uv run python analysis/synthesis.py {{args}}

# Run cross-session validation
cross-session *args:
    uv run python analysis/cross_session.py {{args}}

# Run legislator profiles
profiles *args:
    uv run python analysis/profiles.py {{args}}

# Run external validation against Shor-McCarty scores
external-validation *args:
    uv run python analysis/external_validation.py {{args}}

# Run all tests
test *args:
    uv run pytest tests/ {{args}} -v

# Run scraper tests only
test-scraper *args:
    uv run pytest tests/test_session.py tests/test_scraper_pure.py tests/test_scraper_html.py tests/test_models.py tests/test_output.py tests/test_cli.py {{args}} -v

# Type check with ty (scraper must pass clean, analysis warnings-only)
typecheck:
    uvx ty check src/
    uvx ty check analysis/

# Full check (lint + typecheck + tests)
check:
    just lint-check
    just typecheck
    just test
