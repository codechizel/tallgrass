# REST API Deep Dive (DB4)

Read-only public API for Tallgrass legislative vote data. Framework selection, endpoint design, and implementation.

**Last updated:** 2026-03-04 (implementation complete)

---

## 1. Framework Decision: Django Ninja

**Decision:** Use [Django Ninja](https://django-ninja.dev/) (`>=1.5,<2`), not Django REST Framework.

### Why Not DRF?

DRF is the Django ecosystem default (28K stars, 3.9M weekly PyPI downloads, 1,400+ contributors), but its advantages — vast plugin library, browsable API with HTML forms, hyperlinked serializers — are irrelevant for a read-only public API with ~6 resource types.

### Why Ninja?

| Factor | Django Ninja | DRF |
|--------|-------------|-----|
| **Prior art** | Already used in BaseRally | — |
| **Type hints** | Native — Pydantic v2 schemas from Python types | Requires `drf-spectacular` for OpenAPI |
| **OpenAPI docs** | Built-in at `/api/docs` (Swagger UI) | Requires third-party package |
| **Filtering** | Built-in `FilterSchema` (no `django-filter`) | `django-filter` dependency |
| **Pagination** | Built-in `@paginate` decorator | Built-in |
| **Rate limiting** | Built-in throttling classes | Built-in |
| **Performance** | Faster serialization (Pydantic) | Adequate but slower |
| **Testing** | `TestClient` + pytest | APIClient + pytest |
| **Dependencies** | 1 package (`django-ninja`) | 1 package (`djangorestframework`) |
| **Code style** | Function-based, type-annotated | Class-based ViewSets |
| **Django 5.2 LTS** | Supported | Supported |

Django Ninja's function-based views with type annotations align with tallgrass's code style (Python 3.14+, modern type hints, frozen dataclasses). Pydantic v2 schemas are the natural counterpart to our frozen dataclass models.

### BaseRally Patterns to Adopt

Our BaseRally project (expense tracking SaaS) uses Django Ninja in production. Relevant patterns:

1. **Router-per-resource**: `api.add_router("/expenses/", expenses.router, tags=["Expenses"])` — clean separation, auto-tagged OpenAPI groups.

2. **Pydantic schemas with `from_attributes = True`**: Read Django ORM objects directly — no manual serialization.

3. **Rate limiting decorator**: `@api_ratelimit(key="ip", rate="100/m")` wrapping `django-ratelimit`. For our read-only public API, IP-based limiting is sufficient.

4. **Health check endpoint**: `/api/v1/health` returning `{"status": "ok"}` — useful for monitoring and load balancer probes.

5. **Comprehensive API docstring**: Top-level `NinjaAPI(description=...)` with auth flow, rate limits, pagination examples — becomes the OpenAPI landing page.

### BaseRally Patterns to Skip

These are SaaS-specific and don't apply to a read-only public data API:

- JWT authentication (we have no auth initially)
- Organization context middleware (single-tenant)
- Permission service (all data is public)
- Batch create/update/delete operations (read-only)
- Webhook delivery (no mutations to notify about)
- Celery async tasks (no background processing needed)

### Gotchas (from BaseRally experience + web research)

1. **CSP conflicts with Swagger UI** — Swagger's inline JS may conflict with Content-Security-Policy headers. Not an issue if API is on a separate subdomain or CSP is relaxed for docs.

2. **Manual JSON parsing** — BaseRally works around Pydantic v2 query parameter issues by parsing `request.body` manually. For read-only GET endpoints with query params, this shouldn't be necessary — `FilterSchema` handles it.

3. **Single primary maintainer** — Vitaliy Kucheryaviy drives most Django Ninja development. The codebase is small (~5K lines) and could be forked if needed. Not a blocking concern for a project this size.

---

## 2. API Design

### Base URL

`/api/v1/` — versioned from the start. `v1` is read-only; future `v2` could add write endpoints.

### Resources

Six read-only endpoints mapping to the 8 Django models (State and Session are lightweight enough to inline):

| Endpoint | Model | Records | Key Filters |
|----------|-------|---------|-------------|
| `GET /api/v1/sessions/` | Session (+ State) | ~16 | `state`, `is_special` |
| `GET /api/v1/sessions/{id}/` | Session | 1 | — |
| `GET /api/v1/legislators/` | Legislator | ~2,000 | `session`, `chamber`, `party`, `slug` |
| `GET /api/v1/legislators/{id}/` | Legislator | 1 | — |
| `GET /api/v1/rollcalls/` | RollCall | ~8,000 | `session`, `chamber`, `bill_number`, `passed`, `date_range` |
| `GET /api/v1/rollcalls/{id}/` | RollCall | 1 | — |
| `GET /api/v1/votes/` | Vote | ~650,000 | `session`, `legislator`, `rollcall`, `vote` |
| `GET /api/v1/bill-actions/` | BillAction | ~23,000 | `session`, `bill_number`, `chamber`, `action_code` |
| `GET /api/v1/bill-texts/` | BillText | ~1,600 | `session`, `bill_number`, `document_type` |
| `GET /api/v1/alec/` | ALECModelBill | ~1,057 | `category`, `task_force`, `search` |
| `GET /api/v1/health` | — | — | — |

**No endpoint for State** — only Kansas exists. The `state` field appears in Session for forward-compatibility (DB6 multi-state), but doesn't need its own endpoint yet.

### Detail Endpoints

Each resource has a detail view (`/{id}/`) that returns the full object. For rollcalls, the detail view includes nested votes (the individual legislator votes for that roll call) — this is the most common access pattern ("show me how everyone voted on SB 55").

### Response Schemas

Pydantic v2 schemas with `from_attributes = True` to read directly from Django ORM objects.

**List vs Detail separation**: List endpoints return lightweight schemas (no large text fields). Detail endpoints return full schemas. This matters especially for:
- `RollCall`: list omits `bill_title`, `motion`, `result`; detail includes them + nested votes
- `BillText`: list omits `text` (can be very large); detail includes it
- `ALECModelBill`: list omits `text`; detail includes it

**Semicolon-joined fields parsed to lists**: `sponsor_slugs` and `committee_names` are stored as semicolon-joined text in the DB (CSV round-trip fidelity). The API parses them into proper JSON arrays:
```json
{"sponsor_slugs": ["sen_tyson_caryn_1", "sen_alley_larry_1"]}
```

### Pagination

`LimitOffsetPagination` (Django Ninja built-in):
- Default: `limit=100`, max: `1000`
- Response: `{"items": [...], "count": 8234}`
- Query params: `?limit=50&offset=200`

LimitOffset is the natural fit for data exploration. The `/votes/` endpoint with ~650K records benefits most — clients can page through a legislator's full voting record or a bill's vote breakdown.

### Filtering

Django Ninja `FilterSchema` classes per resource. All filters are optional — omitting returns all records (paginated).

**Session filters:**
- `state: str | None` — 2-letter state code (default: all, currently only "KS")
- `is_special: bool | None` — filter special sessions

**Legislator filters:**
- `session: int | None` — Session PK
- `session_name: str | None` — e.g. "91st_2025-2026" (convenience)
- `chamber: str | None` — "Senate" or "House"
- `party: str | None` — "Republican", "Democrat", "Independent"
- `search: str | None` — name/slug icontains

**RollCall filters:**
- `session: int | None` — Session PK
- `chamber: str | None`
- `bill_number: str | None` — exact match (e.g. "SB 55")
- `passed: bool | None`
- `date_from: date | None` — vote_date >= (inclusive)
- `date_to: date | None` — vote_date <= (inclusive)
- `search: str | None` — bill_number/short_title icontains

**Vote filters:**
- `session: int | None` — via rollcall__session
- `legislator: int | None` — Legislator PK
- `legislator_slug: str | None` — e.g. "sen_masterson_ty_1"
- `rollcall: int | None` — RollCall PK
- `vote: str | None` — "Yea", "Nay", etc.

**BillAction filters:**
- `session: int | None`
- `bill_number: str | None`
- `chamber: str | None`
- `action_code: str | None`

**BillText filters:**
- `session: int | None`
- `bill_number: str | None`
- `document_type: str | None` — "introduced", "supp_note", etc.

**ALECModelBill filters:**
- `category: str | None`
- `task_force: str | None`
- `search: str | None` — title icontains

### Rate Limiting

IP-based anonymous throttling (no auth):
- **List endpoints**: 60 requests/minute per IP
- **Detail endpoints**: 120 requests/minute per IP
- **Health check**: no limit
- Returns `429 Too Many Requests` with `Retry-After` header

Application-level only — not a DDoS defense. Upstream reverse proxy (nginx/Cloudflare) handles volumetric attacks in production.

### Error Responses

Consistent JSON error format:
```json
{"detail": "Not found"}
{"detail": "Rate limit exceeded. Retry after 42 seconds."}
{"detail": "Invalid filter value for 'chamber': must be 'Senate' or 'House'"}
```

HTTP status codes:
- `200` — success
- `404` — resource not found
- `422` — invalid filter parameters
- `429` — rate limit exceeded
- `500` — server error (should never happen for read-only)

---

## 3. Project Structure

```
src/web/
  legislature/
    api/
      __init__.py          # NinjaAPI instance, router registration, health check
      schemas.py           # Pydantic v2 response schemas (12 schemas, list/detail)
      filters.py           # FilterSchema classes (7 filters, FilterLookup annotations)
      pagination.py        # TallgrassPagination (LimitOffset, default 100, max 1000)
      throttling.py        # ListRateThrottle (60/m), DetailRateThrottle (120/m)
      endpoints/
        __init__.py
        sessions.py        # GET /sessions/, /sessions/{id}/
        legislators.py     # GET /legislators/, /legislators/{id}/
        rollcalls.py       # GET /rollcalls/, /rollcalls/{id}/ (detail nests votes)
        votes.py           # GET /votes/ (annotated related fields)
        bill_actions.py    # GET /bill-actions/
        bill_texts.py      # GET /bill-texts/, /bill-texts/{id}/
        alec.py            # GET /alec/, /alec/{id}/
    models.py              # (existing — added composite indexes)
    admin.py               # (existing — unchanged)
  tallgrass_web/
    urls.py                # path("api/v1/", api.urls)
```

Django Ninja does NOT require adding to `INSTALLED_APPS` — it registers via URL routing only.

This mirrors the BaseRally pattern: `api/` package with `__init__.py` holding the `NinjaAPI` instance, `schemas.py` for Pydantic models, and `endpoints/` for router modules.

### Key Differences from BaseRally

| Aspect | BaseRally | Tallgrass |
|--------|-----------|-----------|
| Auth | JWT (7-day tokens) | None (public data) |
| Mutations | Full CRUD + batch | Read-only |
| Multi-tenant | Org context middleware | Single-state (KS) |
| Permissions | PermissionService | None needed |
| Webhooks | Celery + HMAC | None |
| Schemas | ~190 lines (Create/Update/Detail) | ~150 lines (List/Detail only) |

---

## 4. Database Considerations

### Indexes

The existing models have indexes via `ordering` and `UniqueConstraint`, but the API's filter patterns need additional indexes for performance:

```python
# On Legislator
models.Index(fields=["session", "chamber"])
models.Index(fields=["session", "party"])

# On RollCall
models.Index(fields=["session", "chamber"])
models.Index(fields=["session", "bill_number"])
models.Index(fields=["vote_date"])

# On Vote
models.Index(fields=["rollcall"])  # already via FK
models.Index(fields=["legislator"])  # already via FK

# On BillAction
models.Index(fields=["session", "bill_number"])

# On BillText
models.Index(fields=["session", "bill_number"])
```

Django auto-creates indexes for ForeignKey fields, so Vote's FK indexes already exist. The additional composite indexes on Legislator, RollCall, BillAction, and BillText will help the most common filter patterns.

### Query Optimization

- **`select_related`** for FK joins: `Vote.objects.select_related("rollcall", "legislator")` avoids N+1
- **`only()`** for list endpoints: skip large text fields (`bill_title`, `motion`, `text`)
- **`prefetch_related`** for rollcall detail: prefetch all votes + legislators in 2 queries instead of N+1

At ~650K votes, unoptimized queries would be slow. With indexes and `select_related`, response times should be <100ms for filtered queries.

---

## 5. Testing Strategy

### Test Structure

All 64 API tests live in a single file following existing Django test conventions:

```
tests/
  test_django_api.py     # All API tests: endpoints, filters, pagination, schemas (64 tests)
```

All API tests get `@pytest.mark.web` + `@pytest.mark.django_db` (existing markers for Django tests requiring PostgreSQL). Uses `ninja.testing.TestClient` for fast router-level testing (no middleware overhead).

### Test Patterns

```python
from ninja.testing import TestClient
from legislature.api import api

client = TestClient(api)

class TestSessions:
    def test_list_with_data(self, session):
        response = client.get("/sessions/")
        assert response.status_code == 200
        items = response.json()["items"]
        assert len(items) == 1

    def test_detail(self, session, legislator, rollcall):
        response = client.get(f"/sessions/{session.id}/")
        data = response.json()
        assert data["legislator_count"] == 1
        assert data["rollcall_count"] == 1
```

### Coverage (64 tests)

| Category | Tests | Description |
|----------|-------|-------------|
| Health check | 1 | 200 + status/version fields |
| Session endpoints + filters | 6 | list, detail, not-found, state filter, is_special filter |
| Legislator endpoints + filters | 9 | list, detail, session/chamber/party/search/session_name filters |
| RollCall endpoints + filters | 9 | list, detail with nested votes, sponsor_slugs parsing, 5 filters |
| Vote endpoints + filters | 6 | list with annotated fields, slug/vote/rollcall/session filters |
| BillAction endpoints + filters | 7 | list, committee_names parsing, 4 filters |
| BillText endpoints + filters | 6 | list omits text, detail includes text, 2 filters |
| ALEC endpoints + filters | 7 | list omits text, detail includes text, category/task_force/search |
| Pagination | 5 | default, custom limit, offset, beyond-data, count-with-filters |
| Schema validators | 7 | semicolon parsing, empty strings, nullable passed, empty committee_names |

---

## 6. Dependencies

Add to `web` group in `pyproject.toml`:

```toml
[dependency-groups]
web = [
    "django>=5.2,<6",
    "psycopg[binary]>=3.2",
    "django-ninja>=1.5,<2",
]
```

**No other new dependencies.** Django Ninja includes:
- Pydantic v2 (schema validation)
- OpenAPI/Swagger UI (auto-generated docs)
- Pagination, filtering, throttling (all built-in)

Django Ninja's built-in `AnonRateThrottle` handles per-endpoint rate limiting. No `django-ratelimit` needed.

---

## 7. Justfile Recipes

```bash
just api                     # → runserver at localhost:8000 (same as db-admin)
```

The existing `just db-admin` recipe already runs `manage.py runserver`. The API will be available at `localhost:8000/api/v1/` alongside the admin at `localhost:8000/admin/`. No new recipe needed unless we want a distinct name.

---

## 8. Documentation

### OpenAPI Auto-Documentation

Django Ninja generates OpenAPI 3.x schema automatically from type hints and Pydantic schemas:
- **Swagger UI**: `localhost:8000/api/v1/docs` (interactive, try-it-out)
- **OpenAPI JSON**: `localhost:8000/api/v1/openapi.json` (for client generation)
- **Tags**: group endpoints by resource (Sessions, Legislators, Roll Calls, etc.)

The `NinjaAPI` instance gets a comprehensive description (following BaseRally's pattern) covering:
- Available resources and their relationships
- Filtering examples
- Pagination format
- Rate limits
- Data coverage (2011-2026, 84th-91st Legislature)

### ADR

ADR-0096: REST API via Django Ninja. Records the framework decision (Ninja over DRF), endpoint design, and scope (read-only, no auth).

---

## 9. Future Considerations

### Authentication (if needed later)

If we ever need authenticated endpoints (e.g., write access, higher rate limits for registered users), Django Ninja supports:
- API key auth (`ApiKey` class)
- JWT auth (BaseRally pattern)
- Session auth (Django built-in)

Adding auth to specific endpoints is a one-line decorator change: `@router.get("/", auth=ApiKeyAuth())`.

### CORS

For browser-based API consumers (JavaScript frontends), add `django-cors-headers`:
```python
CORS_ALLOW_ALL_ORIGINS = True  # Public data, no restriction needed
```

Not needed for server-to-server or CLI consumers.

### Caching

For high-traffic deployments, response caching with `django.views.decorators.cache.cache_page` or Django Ninja's middleware. Legislative data changes infrequently (scraper runs are manual), so aggressive caching (1 hour+) is safe.

### GraphQL

Not planned. REST with filtering covers all known access patterns. GraphQL would add complexity without clear benefit for this data shape.

---

## 10. Implementation Notes

### What shipped (2026-03-04)

All 12 implementation steps completed in a single session:

1. `django-ninja>=1.5,<2` added to `web` dependency group
2. 12 Pydantic schemas in `schemas.py` (list/detail split, no `class Config` — Ninja handles `from_attributes`)
3. 7 `FilterSchema` classes using `FilterLookup` annotations (not deprecated `Field(q=...)` pattern)
4. `pagination.py` with `TallgrassPagination` (max 1000) and `throttling.py` with list/detail rate classes
5. 7 endpoint modules in `endpoints/` — all using `@paginate(TallgrassPagination)` and throttle decorators
6. `urls.py` wired with `path("api/v1/", api.urls)`
7. 7 composite database indexes (migration 0005)
8. 64 tests in `test_django_api.py`
9. ADR-0096, CLAUDE.md, roadmap, testing.md all updated

### Key implementation decisions

- **Single test file** instead of 4 separate files — follows existing `test_django_models.py` pattern
- **Votes use `F()` annotations** to flatten related fields without N+1 queries
- **Rollcall detail manually constructs response** because it nests votes from `prefetch_related`
- **No `class Config`** on schemas — Ninja's `Schema` base class sets `from_attributes = True` already; inner `Config` classes trigger Pydantic v2 deprecation warnings
- **`FilterLookup` annotations** instead of `Field(q=...)` — the latter is deprecated in django-ninja >= 1.5

---

## References

- [Django Ninja documentation](https://django-ninja.dev/)
- [Django Ninja — Pagination](https://django-ninja.dev/guides/response/pagination/)
- [Django Ninja — Filtering](https://django-ninja.dev/guides/input/filtering/)
- [Django Ninja — Throttling](https://django-ninja.dev/guides/throttling/)
- [Django Ninja — Testing](https://django-ninja.dev/guides/testing/)
- [Django Ninja — OpenAPI Docs](https://django-ninja.dev/guides/api-docs/)
- [BaseRally API](https://github.com/Codechizel/baserally) — production Django Ninja reference
- [Jujens — My Opinion on Django Ninja (July 2025)](https://www.jujens.eu/posts/en/2025/Jul/06/django-ninja/)
- ADR-0090: Django project scaffolding (DB1)
- ADR-0094: CSV-to-PostgreSQL loader (DB2)
- ADR-0095: Scraper post-hook (DB3)
- `docs/data-storage-deep-dive.md` — ecosystem survey, PostgreSQL recommendation
