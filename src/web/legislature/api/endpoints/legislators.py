"""Legislator endpoints."""

from django.shortcuts import get_object_or_404
from ninja import Query, Router
from ninja.pagination import paginate

from legislature.models import Legislator

from ..filters import LegislatorFilter
from ..pagination import TallgrassPagination
from ..schemas import LegislatorOut
from ..throttling import DetailRateThrottle, ListRateThrottle

router = Router()


@router.get("/", response=list[LegislatorOut], throttle=[ListRateThrottle()])
@paginate(TallgrassPagination)
def list_legislators(request, filters: LegislatorFilter = Query(...)):
    return filters.filter(Legislator.objects.select_related("session"))


@router.get("/{int:legislator_id}/", response=LegislatorOut, throttle=[DetailRateThrottle()])
def get_legislator(request, legislator_id: int):
    return get_object_or_404(Legislator, pk=legislator_id)
