# M1: Test Infrastructure

Improve test ergonomics: mark slow tests so `just test-fast` skips them, and consolidate duplicated test factory functions into shared fixtures.

**Roadmap items:** T1 (`@pytest.mark.slow`), R6 (test helper consolidation)
**Estimated effort:** 1-2 sessions

---

## Part A: Mark Slow Tests (`@pytest.mark.slow`)

### Problem

The `@pytest.mark.slow` marker is registered in `pyproject.toml` and `just test-fast` skips it (`-m "not slow"`), but **zero tests in `test_scraper_http.py` are marked**. The retry/backoff tests in `TestGetRetries` and `TestGetErrorClassification` call `_get()` with mocked HTTP responses, triggering real `time.sleep()` backoff delays. Each retry wave sleeps 5-33 seconds depending on the backoff multiplier.

### What to Do

Add `@pytest.mark.slow` to these test classes and methods in `tests/test_scraper_http.py`:

| Target | Line | Tests | Why Slow |
|--------|------|-------|----------|
| `TestGetErrorClassification` (entire class) | 139 | 7 tests | Transient error tests (500, 502, timeout, connection, generic) trigger retry backoff sleeps |
| `TestGetRetries` (entire class) | 273 | 6 tests | Explicitly tests retry counts — each retry includes backoff sleep |
| Individual tests in `TestFetchMany` with retry waves | 327+ | 2 tests | Tests that exercise `max_waves > 0` trigger retry backoff |

**Implementation:**

```python
# At the class level for entire classes:
@pytest.mark.slow
class TestGetErrorClassification:
    ...

@pytest.mark.slow
class TestGetRetries:
    ...

# For individual tests in TestFetchMany, mark only the retry-wave tests:
class TestFetchMany:
    def test_all_succeed(self, scraper):  # NOT slow — no retries
        ...

    @pytest.mark.slow
    def test_retry_wave_recovers_failures(self, scraper):  # slow — retries with sleep
        ...
```

**Note:** Tests that only mock successful responses or permanent errors (404) don't sleep significantly — only mark the ones that trigger transient error retry loops.

### Verification

```bash
# Before: test-fast runs all scraper HTTP tests
just test-fast -k "test_scraper_http" --collect-only | wc -l

# After: test-fast should skip the marked tests
just test-fast -k "test_scraper_http" --collect-only | wc -l

# Confirm slow tests still pass when run directly
just test -k "test_scraper_http" -v
```

### Key Files

- `tests/test_scraper_http.py` — add markers (lines 139, 273, 327+)
- `pyproject.toml` — marker already registered (no change needed)

---

## Part B: Test Helper Consolidation (R6)

### Problem

Five test files define local `_make_legislators()`, `_make_votes()`, and `_make_rollcalls()` factory functions with near-identical signatures but inconsistent column naming (`slug` vs `legislator_slug`). This creates maintenance burden and masks the schema split documented in ADR-0066.

### Current Helpers by File

| File | Helper | Line | Slug Column | Notes |
|------|--------|------|-------------|-------|
| `tests/test_cross_session.py` | `_make_legislators()` | 40 | `slug` | Has `prefix`, `party`, `chamber`, `start_district` params |
| `tests/test_cross_session.py` | `_make_ideal_points()` | 60 | `legislator_slug` | Also builds `full_name`, `party`, `district`, `chamber` |
| `tests/test_cross_session.py` | `_make_large_matched()` | 82 | mixed | Composes `_make_legislators()` + `match_legislators()` |
| `tests/test_dynamic_irt.py` | `_make_legislators()` | 37 | `legislator_slug` | Different schema: `legislator_slug`, `full_name`, `party`, `chamber` |
| `tests/test_dynamic_irt.py` | `_make_vote_matrix()` | 57 | `legislator_slug` | Builds wide-format vote matrix |
| `tests/test_dynamic_irt.py` | `_make_irt_data()` | 74 | n/a | Returns dict (not DataFrame), numpy arrays |
| `tests/test_dynamic_irt.py` | `_make_multi_biennium_data()` | 110 | mixed | Composes `_make_irt_data()` + `_make_legislators()` |
| `tests/test_integration_pipeline.py` | `_make_legislators()` | 52 | `slug` | Hardcoded 10 legislators, `name`/`slug`/`party`/`chamber`/`district` |
| `tests/test_integration_pipeline.py` | `_make_rollcalls()` | 69 | n/a | `vote_id`, `bill_number`, `chamber`, `passed`, `motion`, etc. |
| `tests/test_integration_pipeline.py` | `_make_votes()` | 112 | `slug` | Wide-format: `slug` + `vote_id` columns |
| `tests/test_tsa.py` | `_make_legislators()` | 47 | `legislator_slug` | Has `n_rep`, `n_dem`, `chamber` params |
| `tests/test_tsa.py` | `_make_votes()` | 72 | `legislator_slug` | Generates random Yea/Nay with timestamps |
| `tests/test_mca.py` | `_make_votes()` | 37 | `legislator_slug` | Wide-format with 3 vote categories (Yea/Nay/Absent) |
| `tests/test_mca.py` | `_make_rollcalls()` | 64 | n/a | `vote_id`, `bill_number`, `chamber`, `vote_date` |
| `tests/test_mca.py` | `_make_legislators()` | 75 | `slug` | Basic: `slug`, `full_name`, `party`, `chamber`, `district` |

### Design

Create shared factory functions in `tests/conftest.py` with a `slug_column` parameter to handle the schema split:

```python
# tests/conftest.py — new shared factories

def make_legislators(
    names: list[str] | None = None,
    n: int = 10,
    *,
    prefix: str = "rep",
    party: str = "Republican",
    chamber: str = "House",
    start_district: int = 1,
    slug_column: str = "slug",
) -> pl.DataFrame:
    """Build a legislators DataFrame.

    Args:
        slug_column: "slug" for scraper-schema tests, "legislator_slug" for analysis-schema tests.
    """
    if names is None:
        names = [f"Member {i}" for i in range(n)]
    slugs = [f"{prefix}_{n.split()[-1].lower()}" for n in names]
    return pl.DataFrame({
        slug_column: slugs,
        "full_name": names,
        "party": [party] * len(names),
        "chamber": [chamber] * len(names),
        "district": list(range(start_district, start_district + len(names))),
    })


def make_votes(
    legislators: pl.DataFrame,
    n_votes: int = 30,
    *,
    slug_column: str = "legislator_slug",
    categories: list[str] | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Build a votes DataFrame (wide format: slug column + vote_id columns)."""
    ...


def make_rollcalls(
    n_votes: int = 30,
    *,
    chamber: str = "House",
    seed: int = 42,
) -> pl.DataFrame:
    """Build a rollcalls DataFrame matching the CSV schema."""
    ...
```

### Scope Rules

**Consolidate** (move to `conftest.py`):
- `_make_legislators()` — all 5 files use nearly identical logic
- `_make_votes()` — 3 files (test_mca, test_tsa, test_integration_pipeline)
- `_make_rollcalls()` — 2 files (test_mca, test_integration_pipeline)

**Keep local** (too domain-specific):
- `_make_ideal_points()` in test_cross_session.py — IRT-specific schema
- `_make_irt_data()` in test_dynamic_irt.py — returns numpy dict, not DataFrame
- `_make_multi_biennium_data()` in test_dynamic_irt.py — composes IRT-specific helpers
- `_make_large_matched()` in test_cross_session.py — uses `match_legislators()`
- `_make_vote_matrix()` in test_dynamic_irt.py — dynamic IRT-specific wide format
- All external validation helpers (test_external_validation.py, test_external_validation_dime.py)
- All scraper dataclass builders (test_scraper_*.py)

### Migration Steps

1. Add `make_legislators()`, `make_votes()`, `make_rollcalls()` to `tests/conftest.py`
2. For each consumer file:
   a. Import the shared factory
   b. Replace calls, passing `slug_column="slug"` or `slug_column="legislator_slug"` as appropriate
   c. Delete the local `_make_*` function
   d. Run that file's tests to confirm: `uv run pytest tests/test_<name>.py -v`
3. Run full suite: `just test`

### Column Naming Reference (ADR-0066)

| Context | Column | Used By |
|---------|--------|---------|
| Scraper CSVs | `slug` | test_integration_pipeline, test_cross_session, test_mca |
| Analysis phases | `legislator_slug` | test_dynamic_irt, test_tsa, test_mca (votes/rollcalls) |

The schema split is intentional — scraper outputs `slug`, analysis phases rename to `legislator_slug` at load time. The `slug_column` parameter makes the factory functions work for both schemas without masking this distinction.

---

## Verification Checklist

- [ ] `just test` — all ~1805 tests pass
- [ ] `just test-fast` — skips the newly marked slow tests
- [ ] `just test-fast --collect-only | grep "slow"` — confirms no slow tests collected
- [ ] No duplicate `_make_legislators` definitions remain (grep: `def _make_legislators` should only appear in conftest.py and domain-specific files)
- [ ] `just lint-check` passes

## Documentation

- Update `docs/roadmap.md` items T1 and R6 to "Done"
- Update `.claude/rules/testing.md` marker section to reflect actual usage
- No ADR needed (internal test infrastructure change)

## Commit

```
refactor(infra): test infrastructure — @pytest.mark.slow + shared factories [vYYYY.MM.DD.N]
```
