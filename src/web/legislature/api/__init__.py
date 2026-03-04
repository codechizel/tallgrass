"""Tallgrass REST API — read-only public access to Kansas legislative vote data.

Django Ninja API instance with router registration and health check.
"""

from ninja import NinjaAPI

from .endpoints.alec import router as alec_router
from .endpoints.bill_actions import router as bill_actions_router
from .endpoints.bill_texts import router as bill_texts_router
from .endpoints.legislators import router as legislators_router
from .endpoints.rollcalls import router as rollcalls_router
from .endpoints.sessions import router as sessions_router
from .endpoints.votes import router as votes_router

API_VERSION = "1.0.0"

api = NinjaAPI(
    title="Tallgrass API",
    version=API_VERSION,
    description="""
Read-only public API for Kansas legislative roll call vote data (2011-2026).

## Resources

- **Sessions** — Legislative bienniums and special sessions
- **Legislators** — Members with chamber, party, and district
- **Roll Calls** — Individual vote events with tallies and outcomes
- **Votes** — How each legislator voted on each roll call
- **Bill Actions** — Legislative history events (introduced, passed, signed, etc.)
- **Bill Texts** — Extracted text from bill PDFs (introduced + supplemental notes)
- **ALEC Model Bills** — ALEC model policy corpus for similarity analysis

## Pagination

All list endpoints use limit/offset pagination.
- Default: `?limit=100&offset=0`
- Maximum: `?limit=1000`
- Response includes `count` (total matching records) and `items` (results array).

## Filtering

Each resource supports query parameter filters. Omit a filter to skip it.
See endpoint documentation for available filters per resource.

## Rate Limits

- List endpoints: 60 requests/minute per IP
- Detail endpoints: 120 requests/minute per IP

## Data Coverage

- **84th–91st Legislatures** (2011-2026)
- ~649K individual votes, ~8K roll calls, ~2K legislators
- Updated after each legislative session scrape
""",
    auth=None,
)

api.add_router("/sessions/", sessions_router, tags=["Sessions"])
api.add_router("/legislators/", legislators_router, tags=["Legislators"])
api.add_router("/rollcalls/", rollcalls_router, tags=["Roll Calls"])
api.add_router("/votes/", votes_router, tags=["Votes"])
api.add_router("/bill-actions/", bill_actions_router, tags=["Bill Actions"])
api.add_router("/bill-texts/", bill_texts_router, tags=["Bill Texts"])
api.add_router("/alec/", alec_router, tags=["ALEC Model Bills"])


@api.get("/health", tags=["System"])
def health(request):
    """Health check endpoint for monitoring and load balancer probes."""
    return {"status": "ok", "version": API_VERSION}
