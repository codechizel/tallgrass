"""Bill action endpoints."""

from ninja import Query, Router
from ninja.pagination import paginate

from legislature.models import BillAction

from ..filters import BillActionFilter
from ..pagination import TallgrassPagination
from ..schemas import BillActionOut
from ..throttling import ListRateThrottle

router = Router()


@router.get("/", response=list[BillActionOut], throttle=[ListRateThrottle()])
@paginate(TallgrassPagination)
def list_bill_actions(request, filters: BillActionFilter = Query(...)):
    return filters.filter(BillAction.objects.select_related("session"))
