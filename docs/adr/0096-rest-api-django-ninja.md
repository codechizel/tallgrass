# ADR-0096: REST API via Django Ninja (DB4)

**Date:** 2026-03-04
**Status:** Accepted

## Context

DB1-DB3 are complete: Django models, CSV-to-PostgreSQL loader, and scraper post-hook. The next step (DB4) is a read-only public REST API for external consumers — researchers, journalists, civic tech apps — to query legislative vote data programmatically.

The original roadmap specified Django REST Framework (DRF). Before implementing, we surveyed the landscape and reviewed our own production Django Ninja usage in the BaseRally project (expense tracking SaaS at `Codechizel/baserally`).

## Decision

Use **Django Ninja** (`>=1.5,<2`) instead of DRF for the REST API.

### Why Ninja over DRF

1. **Prior art.** BaseRally already uses Django Ninja in production (JWT auth, batch operations, rate limiting, Pydantic schemas, OpenAPI auto-docs). The patterns are proven and familiar.

2. **Type-hint alignment.** Ninja uses Pydantic v2 schemas derived from Python type annotations — a natural fit for our Python 3.14+ codebase (`list[str]`, `X | None`, frozen dataclasses). DRF's `Serializer` class hierarchy is more verbose.

3. **Built-in batteries for our scope.** Ninja includes pagination (`LimitOffsetPagination`), filtering (`FilterSchema`), throttling (`AnonRateThrottle`), and OpenAPI/Swagger UI — all built-in. DRF would need `drf-spectacular` for comparable OpenAPI generation.

4. **Simpler for read-only.** Function-based views with decorators (`@router.get()`, `@paginate`) are more concise than DRF's class-based ViewSets for a read-only API with no mutations.

5. **Single dependency.** `django-ninja` is the only new package needed. DRF would also be a single package, but Ninja bundles more functionality (OpenAPI, filtering) that DRF requires third-party add-ons for.

### What we're NOT adopting from BaseRally

BaseRally is a multi-tenant SaaS with JWT auth, org context middleware, permission service, batch mutations, webhook delivery, and Celery async tasks. None of these apply to a read-only public data API. We're adopting only:

- Router-per-resource organization
- Pydantic schemas with `from_attributes = True`
- Health check endpoint
- Comprehensive API-level OpenAPI description
- Test patterns (`ninja.testing.TestClient` + pytest)

### API scope

- **10 read-only endpoints** across 7 resources: sessions, legislators, rollcalls, votes, bill-actions, bill-texts, ALEC model bills (+ health check)
- **No authentication** — all legislative data is public
- **LimitOffset pagination** (default 100, max 1000)
- **FilterSchema per resource** — session, chamber, party, bill_number, date ranges, text search
- **IP-based rate limiting** — 60/min lists, 120/min details
- **List vs Detail schemas** — list endpoints omit large text fields; rollcall detail nests votes
- **Semicolon-joined fields** (`sponsor_slugs`, `committee_names`) parsed into JSON arrays

### Project structure

```
src/web/legislature/api/
  __init__.py          # NinjaAPI instance, router registration, health check
  schemas.py           # Pydantic v2 response schemas (12 schemas, list/detail split)
  filters.py           # FilterSchema classes (7 filters, FilterLookup annotations)
  pagination.py        # TallgrassPagination (LimitOffset, default 100, max 1000)
  throttling.py        # ListRateThrottle (60/m), DetailRateThrottle (120/m)
  endpoints/
    __init__.py
    sessions.py        # GET /sessions/, /sessions/{id}/
    legislators.py     # GET /legislators/, /legislators/{id}/
    rollcalls.py       # GET /rollcalls/, /rollcalls/{id}/ (detail nests votes)
    votes.py           # GET /votes/ (annotates related fields via F())
    bill_actions.py    # GET /bill-actions/
    bill_texts.py      # GET /bill-texts/, /bill-texts/{id}/
    alec.py            # GET /alec/, /alec/{id}/
```

## Implementation Notes

### FilterLookup annotations (not deprecated Field pattern)

Django Ninja 1.5+ deprecates `Field(None, q="...")` for filter schemas. We use the new `FilterLookup` annotation style:

```python
from typing import Annotated
from ninja import FilterLookup, FilterSchema

class LegislatorFilter(FilterSchema):
    session: Annotated[int | None, FilterLookup(q="session_id")] = None
    search: Annotated[str | None, FilterLookup(q=["name__icontains", "slug__icontains"])] = None
```

### Ninja Schema base class handles `from_attributes`

Django Ninja's `Schema` base class sets `from_attributes = True` by default. Do NOT add `class Config: from_attributes = True` to schemas — Pydantic v2 deprecates class-based config and the inner `Config` class triggers `PydanticDeprecatedSince20` warnings.

### Python 3.14 asyncio deprecation (upstream)

Django Ninja 1.5.3 uses `asyncio.iscoroutinefunction()` internally, which Python 3.14 deprecates in favor of `inspect.iscoroutinefunction()`. This produces 13 `DeprecationWarning` from `ninja/signature/utils.py:60` during tests. Not our code — will be fixed upstream.

### Vote endpoint uses F() annotations

The votes endpoint annotates related fields (`legislator__slug`, `rollcall__vote_id`, etc.) via `django.db.models.F()` to flatten the response without N+1 queries:

```python
qs.annotate(
    legislator_slug=F("legislator__slug"),
    legislator_name=F("legislator__name"),
    rollcall_vote_id=F("rollcall__vote_id"),
    rollcall_bill_number=F("rollcall__bill_number"),
)
```

### Django Ninja does NOT need INSTALLED_APPS

Unlike DRF, Django Ninja registers via URL routing only. No entry in `INSTALLED_APPS` required.

### Database indexes (migration 0005)

7 composite indexes added for filter performance:

| Model | Index Fields |
|-------|-------------|
| Legislator | `(session, chamber)`, `(session, party)` |
| RollCall | `(session, chamber)`, `(session, bill_number)`, `(vote_date)` |
| BillAction | `(session, bill_number)` |
| BillText | `(session, bill_number)` |

### Test coverage

64 tests in `tests/test_django_api.py`:
- Health check (1)
- Session endpoints + filters (6)
- Legislator endpoints + filters (9)
- RollCall endpoints + filters (9)
- Vote endpoints + filters (6)
- BillAction endpoints + filters (7)
- BillText endpoints + filters (6)
- ALEC endpoints + filters (7)
- Pagination (5)
- Schema validators (7)
- Empty committee_names (1)

All use `ninja.testing.TestClient` for fast router-level testing.

## Consequences

### Positive

- Automatic OpenAPI 3.x docs at `/api/v1/docs` — always accurate because they derive from type hints
- Pydantic validation catches invalid filter parameters and returns 422 with clear error messages
- Performance advantage over DRF for JSON serialization (though PostgreSQL query time dominates)
- Consistent with existing codebase tooling (modern type hints, function-based, minimal boilerplate)
- Single new dependency (`django-ninja`) added to `web` group — no impact on core scraper users

### Negative

- Smaller ecosystem than DRF (8.8K vs 28K stars, 85K vs 3.9M weekly downloads) — less relevant for our simple read-only scope
- Single primary maintainer (Vitaliy Kucheryaviy) — mitigated by small codebase (~5K lines, forkable)
- No browsable HTML API (DRF's unique feature) — Swagger UI serves the same purpose for API exploration
- Python 3.14 asyncio deprecation warning (upstream, not blocking)

### Neutral

- Database indexes needed for filter performance (new migration) — would be required with either framework
- `select_related`/`prefetch_related` optimization needed for Vote and RollCall queries — ORM-level, framework-independent
