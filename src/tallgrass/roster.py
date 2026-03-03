"""OpenStates roster sync — stable cross-biennium legislator identity.

Downloads the openstates/people repo as a GitHub tarball, parses YAML files
for the requested state, extracts KS Legislature member URL slugs, and builds
a slug→ocd_id mapping cached as JSON.

The fast path at scrape time reads only the pre-built JSON lookup — no YAML
parsing, no network access.  Call ``sync_roster()`` once (``just roster-sync``)
to populate the cache; after that ``load_slug_lookup()`` is a cheap JSON read.

Multi-state ready: all functions accept a ``state`` parameter (default "ks").
"""

import io
import json
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

# ── Constants ────────────────────────────────────────────────────────────────

OPENSTATES_TARBALL_URL = "https://github.com/openstates/people/archive/refs/heads/main.tar.gz"
"""Single HTTP request to download the full openstates/people repo."""

CACHE_DIR = Path("data/external/openstates")
"""Cache directory for roster JSON files (relative to project root)."""

_KS_MEMBER_URL_RE = re.compile(
    r"https?://(?:www\.)?kslegislature\.(?:gov|org)/li(?:_\d+s?)?/(?:b\d{4}_\d{2}/)?members/([^/]+)/?"
)
"""Extract slug from KS Legislature member URLs in OpenStates links."""

# ── Data Model ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OpenStatesPerson:
    """Parsed subset of an OpenStates person YAML file."""

    ocd_id: str
    name: str
    given_name: str
    family_name: str
    links: list[str] = field(default_factory=list)
    slugs: list[str] = field(default_factory=list)


# ── Pure Helpers ─────────────────────────────────────────────────────────────


def extract_slugs_from_links(links: list[str]) -> list[str]:
    """Extract KS Legislature slugs from a list of OpenStates person links.

    Handles both .gov and .org domains, all session URL patterns.

    >>> extract_slugs_from_links(["https://kslegislature.gov/li/b2025_26/members/sen_blasi_chase_1/"])
    ['sen_blasi_chase_1']
    """
    slugs: list[str] = []
    for url in links:
        m = _KS_MEMBER_URL_RE.search(url)
        if m:
            slugs.append(m.group(1))
    return slugs


def parse_person_yaml(content: str) -> OpenStatesPerson | None:
    """Parse an OpenStates person YAML file into an OpenStatesPerson.

    Returns None if the file cannot be parsed or lacks an ``id`` field.
    Requires PyYAML (dev dependency — only needed for ``sync_roster``).
    """
    import yaml

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError:
        return None

    if not isinstance(data, dict) or "id" not in data:
        return None

    ocd_id = data["id"]
    name = data.get("name", "")
    given_name = data.get("given_name", "")
    family_name = data.get("family_name", "")

    # Extract URLs from links list
    raw_links = data.get("links", [])
    link_urls: list[str] = []
    if isinstance(raw_links, list):
        for entry in raw_links:
            if isinstance(entry, dict) and "url" in entry:
                link_urls.append(entry["url"])
            elif isinstance(entry, str):
                link_urls.append(entry)

    slugs = extract_slugs_from_links(link_urls)

    return OpenStatesPerson(
        ocd_id=ocd_id,
        name=name,
        given_name=given_name,
        family_name=family_name,
        links=link_urls,
        slugs=slugs,
    )


# ── Sync (Network) ──────────────────────────────────────────────────────────


def sync_roster(
    state: str = "ks",
    *,
    cache_dir: Path | None = None,
    tarball_url: str = OPENSTATES_TARBALL_URL,
) -> dict[str, str]:
    """Download OpenStates people data and build slug→ocd_id mapping.

    Downloads the openstates/people repo as a tarball, filters to
    ``data/{state}/`` YAML files (both ``legislature/`` and ``retired/``),
    parses each, and writes two cache files:

    - ``{state}_roster.json``: full person records (for inspection)
    - ``{state}_slug_to_ocd.json``: flat slug→ocd_id lookup (for scrape time)

    Args:
        state: Two-letter state abbreviation (lowercase).
        cache_dir: Override cache directory (default: ``data/external/openstates``).
        tarball_url: Override tarball URL (for testing).

    Returns:
        The slug→ocd_id mapping dict.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading OpenStates people data for {state.upper()}...")
    resp = requests.get(tarball_url, timeout=60, stream=True)
    resp.raise_for_status()

    # Parse tarball in memory
    people: list[dict] = []
    slug_to_ocd: dict[str, str] = {}

    # The tarball has a top-level directory like "people-main/"
    # We want files matching "*/data/{state}/legislature/*.yml" and "*/data/{state}/retired/*.yml"
    state_path_re = re.compile(
        rf"[^/]+/data/{re.escape(state)}/(?:legislature|retired)/[^/]+\.ya?ml$"
    )

    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if not state_path_re.match(member.name):
                continue

            f = tf.extractfile(member)
            if f is None:
                continue

            content = f.read().decode("utf-8", errors="replace")
            person = parse_person_yaml(content)
            if person is None:
                continue

            # Determine if retired based on path
            is_retired = "/retired/" in member.name

            person_record = {
                "ocd_id": person.ocd_id,
                "name": person.name,
                "given_name": person.given_name,
                "family_name": person.family_name,
                "links": person.links,
                "slugs": person.slugs,
                "retired": is_retired,
            }
            people.append(person_record)

            # Map each slug to this person's OCD ID
            for slug in person.slugs:
                slug_to_ocd[slug] = person.ocd_id

    # Write cache files
    roster_path = cache_dir / f"{state}_roster.json"
    roster_path.write_text(
        json.dumps(people, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lookup_path = cache_dir / f"{state}_slug_to_ocd.json"
    lookup_path.write_text(
        json.dumps(slug_to_ocd, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"  {len(people)} people, {len(slug_to_ocd)} slug mappings")
    print(f"  Cache: {roster_path}")
    print(f"  Lookup: {lookup_path}")

    return slug_to_ocd


# ── Load (Fast Path) ────────────────────────────────────────────────────────


def load_slug_lookup(
    state: str = "ks",
    *,
    cache_dir: Path | None = None,
) -> dict[str, str]:
    """Load the pre-built slug→ocd_id lookup from cache.

    Returns an empty dict if the cache file doesn't exist (roster not synced).
    This is the fast path — no YAML parsing, no network access.

    Args:
        state: Two-letter state abbreviation (lowercase).
        cache_dir: Override cache directory.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR

    lookup_path = cache_dir / f"{state}_slug_to_ocd.json"
    if not lookup_path.exists():
        return {}

    return json.loads(lookup_path.read_text(encoding="utf-8"))
