"""
Tests for ODT vote file parser (2011-2014 sessions).

Inline ODT fixtures are built with zipfile.ZipFile(BytesIO()) to create minimal
valid ODT archives containing content.xml with structured metadata and vote text.

Run: uv run pytest tests/test_odt_parser.py -v
"""

import json
import zipfile
from io import BytesIO

import pytest

pytestmark = pytest.mark.scraper

from tallgrass.odt_parser import (  # noqa: E402
    OdtMetadata,
    _classify_motion,
    _derive_passed,
    _extract_bill_title,
    _extract_body_text,
    _extract_content_xml,
    _format_vote_date,
    _parse_occurred_datetime,
    _parse_odt_body_votes,
    _parse_odt_metadata,
    _resolve_last_name,
    parse_odt_votes,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_odt(content_xml: str) -> bytes:
    """Create a minimal ODT (ZIP) archive with the given content.xml."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("content.xml", content_xml)
    return buf.getvalue()


def _make_content_xml(
    chamber: str = "house",
    bill_numbers: list[str] | None = None,
    occurred: str = "2013/03/27 12:22:40",
    vote_tally: str = "40:0",
    action_code: str = "fa_fabc_341",
    body_paragraphs: list[str] | None = None,
) -> str:
    """Build a minimal ODF content.xml with user-field metadata and body text."""
    if bill_numbers is None:
        bill_numbers = ["SB1"]
    if body_paragraphs is None:
        body_paragraphs = [
            "SB 1, AN ACT concerning taxation.",
            "On roll call, the vote was: Yeas 40; Nays 0.",
            "Yeas: Smith, Jones, Brown.",
            "The bill passed.",
        ]

    fields = {
        "T_JE_S_CHAMBER": chamber,
        "T_JE_T_BILLNUMBER": json.dumps(bill_numbers),
        "T_JE_DT_OCCURRED": occurred,
        "T_JE_T_VOTE": vote_tally,
        "T_JE_S_ACTIONCODE": action_code,
    }

    from xml.sax.saxutils import quoteattr

    field_decls = "\n".join(
        f"<text:user-field-decl text:name={quoteattr(k)} "
        f'office:value-type="string" office:string-value={quoteattr(v)}/>'
        for k, v in fields.items()
    )

    paras = "\n".join(
        f'<text:p text:style-name="P3"><text:span>{p}</text:span></text:p>' for p in body_paragraphs
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
  <office:body>
    <office:text>
      <text:user-field-decls>
        {field_decls}
      </text:user-field-decls>
      {paras}
    </office:text>
  </office:body>
</office:document-content>"""


# ── _extract_content_xml() ───────────────────────────────────────────────────


class TestExtractContentXml:
    """Extract content.xml from ODT ZIP archives."""

    def test_valid_odt(self):
        xml = _make_content_xml()
        odt = _make_odt(xml)
        result = _extract_content_xml(odt)
        assert "office:document-content" in result

    def test_invalid_zip(self):
        result = _extract_content_xml(b"not a zip file")
        assert result == ""

    def test_missing_content_xml(self):
        """ZIP without content.xml returns empty string."""
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        result = _extract_content_xml(buf.getvalue())
        assert result == ""


# ── _parse_odt_metadata() ───────────────────────────────────────────────────


class TestParseOdtMetadata:
    """Parse structured metadata from user-field-decl elements."""

    def test_all_fields(self):
        xml = _make_content_xml(
            chamber="senate",
            bill_numbers=["HB2101"],
            occurred="2011/02/25 11:43:27",
            vote_tally="66:53",
            action_code="fa_fabc_341",
        )
        meta = _parse_odt_metadata(xml)
        assert meta.chamber == "senate"
        assert meta.bill_numbers == ["HB2101"]
        assert meta.occurred == "2011/02/25 11:43:27"
        assert meta.vote_tally == "66:53"
        assert meta.action_code == "fa_fabc_341"

    def test_empty_xml(self):
        meta = _parse_odt_metadata("")
        assert meta.chamber == ""
        assert meta.bill_numbers == []

    def test_malformed_bill_numbers(self):
        """Non-JSON bill_numbers field falls back to empty list."""
        xml = _make_content_xml()
        xml = xml.replace(json.dumps(["SB1"]), "not json")
        meta = _parse_odt_metadata(xml)
        assert meta.bill_numbers == []

    def test_invalid_xml(self):
        meta = _parse_odt_metadata("<broken>")
        assert meta.chamber == ""


# ── _extract_body_text() ────────────────────────────────────────────────────


class TestExtractBodyText:
    """Extract paragraph text from content.xml body."""

    def test_paragraphs_extracted(self):
        xml = _make_content_xml(body_paragraphs=["First line.", "Second line."])
        text = _extract_body_text(xml)
        assert "First line." in text
        assert "Second line." in text

    def test_empty_xml(self):
        assert _extract_body_text("") == ""

    def test_multiline_output(self):
        xml = _make_content_xml(body_paragraphs=["A", "B", "C"])
        text = _extract_body_text(xml)
        lines = text.split("\n")
        assert len(lines) == 3


# ── _parse_odt_body_votes() ─────────────────────────────────────────────────


class TestParseOdtBodyVotes:
    """Parse vote categories from ODT body text."""

    def test_house_format(self):
        """House uses 'Yeas:', 'Nays:', 'Present but not voting:', 'Absent or not voting:'."""
        body = "\n".join(
            [
                "On roll call, the vote was: Yeas 3; Nays 1.",
                "Yeas: Smith, Jones, Brown.",
                "Nays: Lee.",
                "Present but not voting: None.",
                "Absent or not voting: Ray.",
                "The bill passed.",
            ]
        )
        categories, result = _parse_odt_body_votes(body, "House")
        assert len(categories["Yea"]) == 3
        assert len(categories["Nay"]) == 1
        assert len(categories["Absent and Not Voting"]) == 1
        assert "passed" in result.lower()

    def test_senate_format(self):
        """Senate uses 'Present and Passing' and 'Absent or Not Voting'."""
        body = "\n".join(
            [
                "On roll call, the vote was: Yeas 40; Nays 0.",
                "Yeas: Smith, Jones.",
                "The bill passed.",
            ]
        )
        categories, result = _parse_odt_body_votes(body, "Senate")
        assert len(categories["Yea"]) == 2
        assert len(categories["Nay"]) == 0

    def test_none_in_category(self):
        """'None.' in a category means zero members."""
        body = "Nays: None.\nThe bill passed."
        categories, _ = _parse_odt_body_votes(body, "House")
        assert len(categories["Nay"]) == 0

    def test_empty_body(self):
        categories, result = _parse_odt_body_votes("", "House")
        assert all(len(v) == 0 for v in categories.values())
        assert result == ""

    def test_result_extraction(self):
        body = "Yeas: Smith.\nThe bill passed, as amended."
        _, result = _parse_odt_body_votes(body, "House")
        assert "passed, as amended" in result

    def test_motion_failed_result(self):
        body = "Yeas: Smith.\nMotion failed."
        _, result = _parse_odt_body_votes(body, "House")
        assert "failed" in result.lower()


# ── _resolve_last_name() ────────────────────────────────────────────────────


class TestResolveLastName:
    """Resolve last names (with optional initials) via member directory."""

    @pytest.fixture
    def directory(self) -> dict[tuple[str, str], dict]:
        return {
            ("house", "smith"): {"slug": "rep_smith_john_1", "name": "Smith, John"},
            ("house", "jones"): {"slug": "rep_jones_bob_1", "name": "Jones, Bob"},
            ("senate", "doe"): {"slug": "sen_doe_jane_1", "name": "Doe, Jane"},
            ("house", "holmes"): {
                "slug": "rep_holmes_c_1",
                "name": "Holmes, C.",
                "ambiguous": True,
            },
            ("house", "c. holmes"): {"slug": "rep_holmes_carol_1", "name": "Holmes, Carol"},
        }

    def test_simple_match(self, directory):
        slug, name = _resolve_last_name("Smith", "house", directory)
        assert slug == "rep_smith_john_1"

    def test_no_directory(self):
        slug, name = _resolve_last_name("Smith", "house", None)
        assert slug == ""
        assert name == "Smith"

    def test_no_match(self, directory):
        slug, name = _resolve_last_name("Unknown", "house", directory)
        assert slug == ""

    def test_ambiguous_without_initial(self, directory):
        """Ambiguous last name without initial returns empty slug."""
        slug, name = _resolve_last_name("Holmes", "house", directory)
        assert slug == ""

    def test_ambiguous_with_initial(self, directory):
        """Ambiguous last name with initial tries initial-qualified key."""
        slug, name = _resolve_last_name("C. Holmes", "house", directory)
        assert slug == "rep_holmes_carol_1"

    def test_wrong_chamber(self, directory):
        slug, name = _resolve_last_name("Smith", "senate", directory)
        assert slug == ""


# ── Datetime helpers ────────────────────────────────────────────────────────


class TestDatetimeHelpers:
    """Convert ODT timestamps to ISO 8601 and MM/DD/YYYY."""

    def test_parse_occurred_datetime(self):
        assert _parse_occurred_datetime("2013/03/27 12:22:40") == "2013-03-27T12:22:40"

    def test_parse_occurred_empty(self):
        assert _parse_occurred_datetime("") == ""

    def test_format_vote_date(self):
        assert _format_vote_date("2013/03/27 12:22:40") == "03/27/2013"

    def test_format_vote_date_empty(self):
        assert _format_vote_date("") == ""


# ── _extract_bill_title() ───────────────────────────────────────────────────


class TestExtractBillTitle:
    """Extract bill title from ODT body text."""

    def test_an_act(self):
        body = "SB 1, AN ACT concerning taxation.\nYeas: Smith."
        title = _extract_bill_title(body)
        assert title.startswith("AN ACT")
        assert "taxation" in title

    def test_no_title(self):
        assert _extract_bill_title("Yeas: Smith.") == ""

    def test_empty(self):
        assert _extract_bill_title("") == ""


# ── _classify_motion() ──────────────────────────────────────────────────────


class TestClassifyMotion:
    """Classify ODT result text into vote_type."""

    def test_final_action_code(self):
        vtype, _ = _classify_motion("The bill passed.", "fa_fabc_341")
        assert vtype == "Final Action"

    def test_emergency_final_action(self):
        vtype, _ = _classify_motion("Emergency final action passed.", "fa_fabc_341")
        assert vtype == "Emergency Final Action"

    def test_concurrence(self):
        vtype, _ = _classify_motion("The Senate concurred.", "cur_con_335")
        assert vtype == "Concurrence"

    def test_misc_code(self):
        vtype, _ = _classify_motion("Motion to table.", "misc_bs_100")
        assert vtype == "Procedural Motion"

    def test_empty(self):
        assert _classify_motion("", "") == ("", "")


# ── _derive_passed() (ODT) ─────────────────────────────────────────────────


class TestOdtDerivePassed:
    """Derive passed boolean from ODT result text."""

    def test_passed(self):
        assert _derive_passed("The bill passed.") is True

    def test_passed_as_amended(self):
        assert _derive_passed("The bill passed, as amended.") is True

    def test_failed(self):
        assert _derive_passed("Motion failed.") is False

    def test_not_passed(self):
        assert _derive_passed("Not passed") is False

    def test_concurred(self):
        assert _derive_passed("The Senate concurred.") is True

    def test_empty(self):
        assert _derive_passed("") is None


# ── OdtMetadata ─────────────────────────────────────────────────────────────


class TestOdtMetadata:
    """OdtMetadata frozen dataclass."""

    def test_construction(self):
        meta = OdtMetadata(
            chamber="house",
            bill_numbers=["HB2101"],
            occurred="2011/02/25 11:43:27",
            vote_tally="66:53",
            action_code="fa_fabc_341",
        )
        assert meta.chamber == "house"
        assert meta.bill_numbers == ["HB2101"]


# ── parse_odt_votes() (integration) ────────────────────────────────────────


class TestParseOdtVotes:
    """End-to-end ODT parsing integration tests."""

    def test_basic_vote(self):
        """Minimal ODT with 3 yeas produces correct output."""
        xml = _make_content_xml(
            chamber="house",
            bill_numbers=["SB1"],
            occurred="2013/03/27 12:22:40",
            vote_tally="3:0",
            action_code="fa_fabc_341",
            body_paragraphs=[
                "SB 1, AN ACT concerning taxation.",
                "On roll call, the vote was: Yeas 3; Nays 0.",
                "Yeas: Smith, Jones, Brown.",
                "Nays: None.",
                "The bill passed.",
            ],
        )
        odt = _make_odt(xml)
        rollcalls, votes, legs = parse_odt_votes(
            odt_bytes=odt,
            bill_number="SB 1",
            bill_path="/li_2014/b2013_14/measures/sb1/",
            vote_url="https://example.com/odt_view/je_20130327122240_207704.odt",
            session_label="85th (2013-2014)",
        )
        assert len(rollcalls) == 1
        assert len(votes) == 3
        assert rollcalls[0].chamber == "House"
        assert rollcalls[0].yea_count == 3
        assert rollcalls[0].nay_count == 0
        assert rollcalls[0].passed is True

    def test_vote_id_from_url(self):
        """vote_id extracted from the odt_view URL."""
        xml = _make_content_xml(body_paragraphs=["Yeas: Smith.", "The bill passed."])
        odt = _make_odt(xml)
        rollcalls, _, _ = parse_odt_votes(
            odt_bytes=odt,
            bill_number="SB 1",
            bill_path="/p",
            vote_url="https://example.com/odt_view/je_20130327122240_207704.odt",
            session_label="85th (2013-2014)",
        )
        assert rollcalls[0].vote_id == "je_20130327122240_207704"

    def test_with_member_directory(self):
        """Member directory enables slug resolution."""
        directory: dict[tuple[str, str], dict] = {
            ("house", "smith"): {"slug": "rep_smith_john_1", "name": "Smith, John"},
        }
        xml = _make_content_xml(body_paragraphs=["Yeas: Smith.", "The bill passed."])
        odt = _make_odt(xml)
        _, votes, legs = parse_odt_votes(
            odt_bytes=odt,
            bill_number="SB 1",
            bill_path="/p",
            vote_url="https://example.com/odt_view/je_123.odt",
            session_label="85th (2013-2014)",
            member_directory=directory,
        )
        assert len(votes) == 1
        assert votes[0].legislator_slug == "rep_smith_john_1"
        assert len(legs) == 1

    def test_empty_odt(self):
        """Invalid ODT bytes returns empty results."""
        rollcalls, votes, legs = parse_odt_votes(
            odt_bytes=b"not a zip",
            bill_number="SB 1",
            bill_path="/p",
            vote_url="https://example.com/odt_view/je_123.odt",
            session_label="85th (2013-2014)",
        )
        assert len(rollcalls) == 0
        assert len(votes) == 0

    def test_bill_metadata_lookup(self):
        """Bill metadata (short_title, sponsor) is looked up by normalized code."""
        xml = _make_content_xml(body_paragraphs=["Yeas: Smith.", "The bill passed."])
        odt = _make_odt(xml)
        rollcalls, _, _ = parse_odt_votes(
            odt_bytes=odt,
            bill_number="SB 1",
            bill_path="/p",
            vote_url="https://example.com/odt_view/je_123.odt",
            session_label="85th (2013-2014)",
            bill_metadata={"sb1": {"short_title": "Taxation", "sponsor": "Sen. Smith"}},
        )
        assert rollcalls[0].short_title == "Taxation"
        assert rollcalls[0].sponsor == "Sen. Smith"

    def test_datetime_fields(self):
        """vote_datetime and vote_date populated from ODT metadata."""
        xml = _make_content_xml(
            occurred="2013/03/27 12:22:40",
            body_paragraphs=["Yeas: Smith.", "The bill passed."],
        )
        odt = _make_odt(xml)
        rollcalls, votes, _ = parse_odt_votes(
            odt_bytes=odt,
            bill_number="SB 1",
            bill_path="/p",
            vote_url="https://example.com/odt_view/je_123.odt",
            session_label="85th (2013-2014)",
        )
        assert rollcalls[0].vote_datetime == "2013-03-27T12:22:40"
        assert rollcalls[0].vote_date == "03/27/2013"
        assert votes[0].vote_datetime == "2013-03-27T12:22:40"
