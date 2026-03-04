"""Custom pagination with max_limit cap."""

from ninja.pagination import LimitOffsetPagination


class TallgrassPagination(LimitOffsetPagination):
    """Limit/offset pagination with a 1000-item cap."""

    class Input(LimitOffsetPagination.Input):
        limit: int = 100

    max_limit = 1000
