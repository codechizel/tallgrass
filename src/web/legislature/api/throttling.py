"""Rate limiting for the Tallgrass API.

IP-based throttling: 60/min for list endpoints, 120/min for detail endpoints.
Uses Django Ninja's built-in Throttle classes.
"""

from ninja.throttling import AnonRateThrottle


class ListRateThrottle(AnonRateThrottle):
    rate = "60/m"


class DetailRateThrottle(AnonRateThrottle):
    rate = "120/m"
