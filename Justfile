# Tallgrass — Command Runner (Just: https://github.com/casey/just)
#
# Thin aliases over `uv run` commands. The main value-adds:
#   1. `just check` sequences lint + typecheck + tests as a single quality gate
#   2. OMP/OPENBLAS thread caps below prevent E-core spillover on Apple Silicon
#   3. `just --list` documents every runnable command in the project
#
# All recipes pass *args through, so `just profiles --names "Masterson"`
# is equivalent to `uv run python analysis/12_profiles/profiles.py --names "Masterson"`.
#
# Cap thread pools to P-core count (6) to prevent E-core spillover on Apple Silicon.
# See ADR-0022 and results/experiments/2026-02-23_parallel-chains-performance/.
export OMP_NUM_THREADS := "6"
export OPENBLAS_NUM_THREADS := "6"

# Ensure /usr/bin is on PATH so PyTensor can find clang++/g++ for C compilation.
# Without this, background processes or stripped shells fall back to pure Python (~18x slower).
export PATH := "/usr/bin:/bin:" + env("PATH")

# Default: show available commands
default:
    @just --list

# Scrape current session (cached)
scrape *args:
    uv run tallgrass {{args}}

# Scrape with fresh cache
scrape-fresh *args:
    uv run tallgrass --clear-cache {{args}}

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
    uv run tallgrass --list-sessions

# Run EDA analysis
eda *args:
    uv run python analysis/01_eda/eda.py {{args}}

# Run PCA analysis
pca *args:
    uv run python analysis/02_pca/pca.py {{args}}

# Run MCA analysis
mca *args:
    uv run python analysis/02c_mca/mca.py {{args}}

# Run Bayesian IRT analysis
irt *args:
    uv run python analysis/04_irt/irt.py {{args}}

# Run clustering analysis
clustering *args:
    uv run python analysis/05_clustering/clustering.py {{args}}

# Run network analysis
network *args:
    uv run python analysis/06_network/network.py {{args}}

# Run prediction analysis
prediction *args:
    uv run python analysis/08_prediction/prediction.py {{args}}

# Run UMAP analysis
umap *args:
    uv run python analysis/03_umap/umap_viz.py {{args}}

# Run classical indices analysis
indices *args:
    uv run python analysis/07_indices/indices.py {{args}}

# Run Beta-Binomial Bayesian party loyalty
betabinom *args:
    uv run python analysis/09_beta_binomial/beta_binomial.py {{args}}

# Run hierarchical Bayesian IRT
hierarchical *args:
    uv run python analysis/10_hierarchical/hierarchical.py {{args}}

# Run synthesis report
synthesis *args:
    uv run python analysis/11_synthesis/synthesis.py {{args}}

# Run cross-session validation
cross-session *args:
    uv run python analysis/13_cross_session/cross_session.py {{args}}

# Run legislator profiles
profiles *args:
    uv run python analysis/12_profiles/profiles.py {{args}}

# Run external validation against Shor-McCarty scores
external-validation *args:
    uv run python analysis/14_external_validation/external_validation.py {{args}}

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

# Monitor running experiment
monitor:
    @cat /tmp/tallgrass/experiment.status.json 2>/dev/null | python3 -m json.tool || echo "No experiment running"

# Full check (lint + typecheck + tests)
check:
    just lint-check
    just typecheck
    just test
