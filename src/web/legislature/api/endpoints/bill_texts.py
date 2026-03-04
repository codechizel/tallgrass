"""Bill text endpoints."""

from django.shortcuts import get_object_or_404
from ninja import Query, Router
from ninja.pagination import paginate

from legislature.models import BillText

from ..filters import BillTextFilter
from ..pagination import TallgrassPagination
from ..schemas import BillTextDetail, BillTextOut
from ..throttling import DetailRateThrottle, ListRateThrottle

router = Router()


@router.get("/", response=list[BillTextOut], throttle=[ListRateThrottle()])
@paginate(TallgrassPagination)
def list_bill_texts(request, filters: BillTextFilter = Query(...)):
    return filters.filter(BillText.objects.select_related("session"))


@router.get("/{int:bill_text_id}/", response=BillTextDetail, throttle=[DetailRateThrottle()])
def get_bill_text(request, bill_text_id: int):
    return get_object_or_404(BillText, pk=bill_text_id)
