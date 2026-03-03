"""Shared bill discovery for the Kansas Legislature.

Extracts bill URL discovery logic used by both the vote scraper and the bill
text adapter. All functions take a ``get_fn`` callable for HTTP fetching to
avoid coupling to any specific scraper or adapter instance.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from tallgrass.config import BASE_URL
from tallgrass.session import KSSession

# Compiled regex for extracting bill type and number from URLs
BILL_URL_RE = re.compile(r"/(sb|hb|scr|hcr|sr|hr)(\d+)/", re.I)


@dataclass(frozen=True)
class BillInfo:
    """Minimal bill information from discovery (bill URL path + number)."""

    url_path: str  # e.g., "/li/b2025_26/measures/sb55/"
    bill_number: str  # e.g., "SB 55"


def parse_js_array(js_content: str) -> list[dict]:
    """Extract the first JSON array from JS source, quoting bare keys.

    Finds ``[...]`` in the source, quotes unquoted JS object-literal keys
    (``measures_url:`` -> ``"measures_url":``), and parses as JSON.
    Returns empty list on failure.
    """
    start = js_content.find("[")
    end = js_content.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    array_text = js_content[start : end + 1]
    array_text = re.sub(r"(?m)^(\s+)(\w+):", r'\1"\2":', array_text)
    try:
        data = json.loads(array_text)
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def parse_js_bill_data(js_content: str) -> list[str]:
    """Extract bill URLs from a ``measures_data = [...]`` JS assignment."""
    data = parse_js_array(js_content)
    urls: list[str] = []
    for entry in data:
        url = entry.get("measures_url", "") if isinstance(entry, dict) else ""
        if url:
            urls.append(url)
    return urls


def bill_sort_key(url: str) -> tuple[int, int]:
    """Sort bills: SB before HB, then numerically."""
    match = BILL_URL_RE.search(url)
    if match:
        prefix = match.group(1).lower()
        number = int(match.group(2))
        order = {"sb": 0, "sr": 1, "scr": 2, "hb": 3, "hr": 4, "hcr": 5}
        return (order.get(prefix, 9), number)
    return (99, 0)


def url_to_bill_number(url: str) -> str:
    """Extract a human-readable bill number from a bill URL path.

    ``"/li/b2025_26/measures/sb55/"`` -> ``"SB 55"``
    ``"/li_2024/b2023_24/measures/hb2084/"`` -> ``"HB 2084"``
    """
    match = BILL_URL_RE.search(url)
    if match:
        prefix = match.group(1).upper()
        number = match.group(2)
        return f"{prefix} {number}"
    return ""


def discover_bill_urls(
    session: KSSession,
    get_fn: Callable[[str], object],
    *,
    verbose: bool = True,
) -> list[str]:
    """Discover all bill URL paths for a session.

    Uses HTML listing pages first, then falls back to JS data files for
    pre-2021 sessions. The ``get_fn`` callable must accept a full URL string
    and return an object with ``.ok`` (bool) and ``.html`` (str | None)
    attributes (matching the ``FetchResult`` interface).

    Returns sorted bill URL paths (relative, e.g., ``/li/b2025_26/measures/sb55/``).
    """
    bill_urls: set[str] = set()
    pattern = session.bill_url_pattern

    for label, path in [
        ("All Bills", session.bills_path),
        ("Senate Bills", session.senate_bills_path),
        ("House Bills", session.house_bills_path),
    ]:
        if verbose:
            print(f"  Fetching {label}...")
        result = get_fn(BASE_URL + path)
        if not result.ok:  # type: ignore[union-attr]
            continue

        # Import here to avoid top-level dependency for non-HTML callers
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(result.html, "lxml")  # type: ignore[union-attr]
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if pattern.match(href):
                bill_urls.add(href)

    # Fallback: sessions before 2021 load bill lists via JavaScript
    if not bill_urls and session.js_data_paths:
        if verbose:
            print("  HTML listing yielded 0 bills — trying JS data fallback...")
        bill_urls = _discover_bill_urls_from_js(session, get_fn, verbose=verbose)

    return sorted(bill_urls, key=bill_sort_key)


def _discover_bill_urls_from_js(
    session: KSSession,
    get_fn: Callable[[str], object],
    *,
    verbose: bool = True,
) -> set[str]:
    """Discover bill URLs from JavaScript data files (pre-2021 sessions)."""
    bill_urls: set[str] = set()
    pattern = session.bill_url_pattern

    for js_path in session.js_data_paths:
        if verbose:
            print(f"  Trying JS data file: {js_path}")
        result = get_fn(BASE_URL + js_path)
        if not result.ok or not result.html:  # type: ignore[union-attr]
            continue
        parsed = parse_js_bill_data(result.html)  # type: ignore[union-attr]
        for url in parsed:
            if pattern.match(url):
                bill_urls.add(url)
        if bill_urls:
            if verbose:
                print(f"  JS fallback found {len(bill_urls)} bill URLs")
            break

    return bill_urls


def discover_bills(
    session: KSSession,
    get_fn: Callable[[str], object],
    *,
    verbose: bool = True,
) -> list[BillInfo]:
    """Discover bills and return structured ``BillInfo`` objects.

    Higher-level wrapper around ``discover_bill_urls()`` that also extracts
    bill numbers from the URL paths. Used by the text adapter.
    """
    url_paths = discover_bill_urls(session, get_fn, verbose=verbose)
    bills: list[BillInfo] = []
    for url_path in url_paths:
        bill_number = url_to_bill_number(url_path)
        if bill_number:
            bills.append(BillInfo(url_path=url_path, bill_number=bill_number))
    return bills
