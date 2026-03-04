"""ALEC model bill endpoints."""

from django.shortcuts import get_object_or_404
from ninja import Query, Router
from ninja.pagination import paginate

from legislature.models import ALECModelBill

from ..filters import ALECFilter
from ..pagination import TallgrassPagination
from ..schemas import ALECDetail, ALECOut
from ..throttling import DetailRateThrottle, ListRateThrottle

router = Router()


@router.get("/", response=list[ALECOut], throttle=[ListRateThrottle()])
@paginate(TallgrassPagination)
def list_alec(request, filters: ALECFilter = Query(...)):
    return filters.filter(ALECModelBill.objects.all())


@router.get("/{int:alec_id}/", response=ALECDetail, throttle=[DetailRateThrottle()])
def get_alec(request, alec_id: int):
    return get_object_or_404(ALECModelBill, pk=alec_id)
