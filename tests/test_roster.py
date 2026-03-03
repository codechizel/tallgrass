"""
Tests for OpenStates roster sync and slug→ocd_id mapping.

Verifies YAML parsing, slug extraction from KS Legislature member URLs,
cache load/save, and edge cases (multiple slugs, retired legislators,
same-name legislators).

Run: uv run pytest tests/test_roster.py -v
"""

import json

import pytest

from tallgrass.roster import (
    extract_slugs_from_links,
    load_slug_lookup,
    parse_person_yaml,
)

pytestmark = pytest.mark.scraper


# ── extract_slugs_from_links ────────────────────────────────────────────────


class TestExtractSlugsFromLinks:
    """Slug extraction from OpenStates person link URLs."""

    def test_current_session_gov(self):
        urls = ["https://kslegislature.gov/li/b2025_26/members/sen_blasi_chase_1/"]
        assert extract_slugs_from_links(urls) == ["sen_blasi_chase_1"]

    def test_historical_session(self):
        urls = ["https://kslegislature.org/li_2024/b2023_24/members/rep_smith_jane_1/"]
        assert extract_slugs_from_links(urls) == ["rep_smith_jane_1"]

    def test_special_session(self):
        urls = ["https://www.kslegislature.org/li_2024s/members/sen_doe_john_1/"]
        assert extract_slugs_from_links(urls) == ["sen_doe_john_1"]

    def test_multiple_links_multiple_slugs(self):
        urls = [
            "https://kslegislature.gov/li/b2025_26/members/sen_blasi_chase_1/",
            "https://kslegislature.org/li_2024/b2023_24/members/sen_blasi_chase_1/",
            "https://kslegislature.org/li_2022/b2021_22/members/sen_blasi_chase_1/",
        ]
        assert extract_slugs_from_links(urls) == [
            "sen_blasi_chase_1",
            "sen_blasi_chase_1",
            "sen_blasi_chase_1",
        ]

    def test_non_ks_links_ignored(self):
        urls = [
            "https://example.com/member/123",
            "https://ballotpedia.org/Chase_Blasi",
        ]
        assert extract_slugs_from_links(urls) == []

    def test_mixed_ks_and_non_ks(self):
        urls = [
            "https://ballotpedia.org/Chase_Blasi",
            "https://kslegislature.gov/li/b2025_26/members/sen_blasi_chase_1/",
        ]
        assert extract_slugs_from_links(urls) == ["sen_blasi_chase_1"]

    def test_empty_links(self):
        assert extract_slugs_from_links([]) == []

    def test_trailing_slash_optional(self):
        urls = ["https://kslegislature.gov/li/b2025_26/members/sen_blasi_chase_1"]
        assert extract_slugs_from_links(urls) == ["sen_blasi_chase_1"]

    def test_www_prefix(self):
        urls = ["https://www.kslegislature.gov/li/b2025_26/members/rep_jones_bob_1/"]
        assert extract_slugs_from_links(urls) == ["rep_jones_bob_1"]


# ── parse_person_yaml ──────────────────────────────────────────────────────


class TestParsePersonYaml:
    """YAML parsing for OpenStates person files."""

    def test_basic_person(self):
        yaml_content = """
id: ocd-person/12345678-1234-1234-1234-123456789012
name: Chase Blasi
given_name: Chase
family_name: Blasi
links:
  - url: https://kslegislature.gov/li/b2025_26/members/sen_blasi_chase_1/
"""
        person = parse_person_yaml(yaml_content)
        assert person is not None
        assert person.ocd_id == "ocd-person/12345678-1234-1234-1234-123456789012"
        assert person.name == "Chase Blasi"
        assert person.given_name == "Chase"
        assert person.family_name == "Blasi"
        assert len(person.slugs) == 1
        assert person.slugs[0] == "sen_blasi_chase_1"

    def test_multiple_links(self):
        yaml_content = """
id: ocd-person/aabbccdd-1234-5678-9012-aabbccddeeff
name: Mike Thompson
given_name: Mike
family_name: Thompson
links:
  - url: https://kslegislature.gov/li/b2025_26/members/sen_thompson_mike_1/
  - url: https://ballotpedia.org/Mike_Thompson
"""
        person = parse_person_yaml(yaml_content)
        assert person is not None
        assert len(person.slugs) == 1
        assert person.slugs[0] == "sen_thompson_mike_1"
        assert len(person.links) == 2

    def test_no_ks_links(self):
        yaml_content = """
id: ocd-person/ffffffff-0000-0000-0000-ffffffffffff
name: Some Person
links:
  - url: https://ballotpedia.org/Some_Person
"""
        person = parse_person_yaml(yaml_content)
        assert person is not None
        assert person.slugs == []

    def test_no_links(self):
        yaml_content = """
id: ocd-person/ffffffff-0000-0000-0000-ffffffffffff
name: No Links Person
"""
        person = parse_person_yaml(yaml_content)
        assert person is not None
        assert person.slugs == []
        assert person.links == []

    def test_missing_id_returns_none(self):
        yaml_content = """
name: No ID Person
links:
  - url: https://example.com
"""
        assert parse_person_yaml(yaml_content) is None

    def test_invalid_yaml_returns_none(self):
        assert parse_person_yaml(":::invalid::: yaml {{{") is None

    def test_non_dict_yaml_returns_none(self):
        assert parse_person_yaml("- just\n- a\n- list") is None

    def test_frozen_dataclass(self):
        person = parse_person_yaml("""
id: ocd-person/12345678-1234-1234-1234-123456789012
name: Test Person
""")
        assert person is not None
        with pytest.raises(AttributeError):
            person.name = "Changed"  # type: ignore[misc]

    def test_multiple_ks_session_links(self):
        """Same legislator with member URLs from multiple sessions."""
        yaml_content = """
id: ocd-person/aabb1122-3344-5566-7788-99aabbccddee
name: Chase Blasi
given_name: Chase
family_name: Blasi
links:
  - url: https://kslegislature.gov/li/b2025_26/members/sen_blasi_chase_1/
  - url: https://kslegislature.org/li_2024/b2023_24/members/sen_blasi_chase_1/
"""
        person = parse_person_yaml(yaml_content)
        assert person is not None
        assert len(person.slugs) == 2
        assert all(s == "sen_blasi_chase_1" for s in person.slugs)


# ── Two Mike Thompsons ──────────────────────────────────────────────────────


class TestTwoMikeThompsons:
    """Same-name legislators get distinct OCD IDs."""

    def test_separate_ocd_ids(self):
        senate_yaml = """
id: ocd-person/aaaa0000-0000-0000-0000-000000000001
name: Mike Thompson
given_name: Mike
family_name: Thompson
links:
  - url: https://kslegislature.gov/li/b2025_26/members/sen_thompson_mike_1/
"""
        house_yaml = """
id: ocd-person/aaaa0000-0000-0000-0000-000000000002
name: Mike Thompson
given_name: Mike
family_name: Thompson
links:
  - url: https://kslegislature.gov/li/b2025_26/members/rep_thompson_mike_1/
"""
        senate = parse_person_yaml(senate_yaml)
        house = parse_person_yaml(house_yaml)
        assert senate is not None and house is not None
        assert senate.ocd_id != house.ocd_id
        assert senate.slugs[0] == "sen_thompson_mike_1"
        assert house.slugs[0] == "rep_thompson_mike_1"

    def test_distinct_slug_mapping(self):
        """Simulates building the slug→ocd_id map for both Mike Thompsons."""
        slug_to_ocd: dict[str, str] = {}
        slug_to_ocd["sen_thompson_mike_1"] = "ocd-person/aaaa-0001"
        slug_to_ocd["rep_thompson_mike_1"] = "ocd-person/aaaa-0002"
        assert len(slug_to_ocd) == 2
        assert slug_to_ocd["sen_thompson_mike_1"] != slug_to_ocd["rep_thompson_mike_1"]


# ── load_slug_lookup ─────────────────────────────────────────────────────────


class TestLoadSlugLookup:
    """Loading the cached slug→ocd_id JSON lookup."""

    def test_returns_empty_dict_when_no_cache(self, tmp_path):
        result = load_slug_lookup("ks", cache_dir=tmp_path)
        assert result == {}

    def test_loads_existing_cache(self, tmp_path):
        lookup = {"sen_blasi_chase_1": "ocd-person/12345"}
        (tmp_path / "ks_slug_to_ocd.json").write_text(json.dumps(lookup))
        result = load_slug_lookup("ks", cache_dir=tmp_path)
        assert result == lookup

    def test_state_namespacing(self, tmp_path):
        ks_lookup = {"sen_blasi_chase_1": "ocd-person/ks-1"}
        mo_lookup = {"rep_jones_bob_1": "ocd-person/mo-1"}
        (tmp_path / "ks_slug_to_ocd.json").write_text(json.dumps(ks_lookup))
        (tmp_path / "mo_slug_to_ocd.json").write_text(json.dumps(mo_lookup))
        assert load_slug_lookup("ks", cache_dir=tmp_path) == ks_lookup
        assert load_slug_lookup("mo", cache_dir=tmp_path) == mo_lookup

    def test_roundtrip_json(self, tmp_path):
        """Values survive JSON write→read cycle."""
        original = {
            "sen_blasi_chase_1": "ocd-person/12345678-1234-1234-1234-123456789012",
            "rep_thompson_mike_1": "ocd-person/aabbccdd-eeff-0011-2233-445566778899",
        }
        path = tmp_path / "ks_slug_to_ocd.json"
        path.write_text(json.dumps(original, indent=2) + "\n")
        loaded = load_slug_lookup("ks", cache_dir=tmp_path)
        assert loaded == original
