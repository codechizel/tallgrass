"""Session endpoints."""

from django.db.models import Count
from django.shortcuts import get_object_or_404
from ninja import Query, Router
from ninja.pagination import paginate

from legislature.models import Session

from ..filters import SessionFilter
from ..pagination import TallgrassPagination
from ..schemas import SessionDetail, SessionOut
from ..throttling import DetailRateThrottle, ListRateThrottle

router = Router()


@router.get("/", response=list[SessionOut], throttle=[ListRateThrottle()])
@paginate(TallgrassPagination)
def list_sessions(request, filters: SessionFilter = Query(...)):
    return filters.filter(Session.objects.all())


@router.get("/{int:session_id}/", response=SessionDetail, throttle=[DetailRateThrottle()])
def get_session(request, session_id: int):
    session = get_object_or_404(
        Session.objects.annotate(
            legislator_count=Count("legislators"),
            rollcall_count=Count("rollcalls"),
        ),
        pk=session_id,
    )
    return session
