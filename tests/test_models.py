"""
Tests for data model contracts (frozen dataclasses).

Verifies construction and immutability of IndividualVote, RollCall,
FetchFailure, and VoteLink.

Run: uv run pytest tests/test_models.py -v
"""

import dataclasses

import pytest

from tallgrass.models import IndividualVote, RollCall
from tallgrass.scraper import FetchFailure, VoteLink

pytestmark = pytest.mark.scraper

# ── IndividualVote ───────────────────────────────────────────────────────────


class TestIndividualVote:
    """IndividualVote frozen dataclass."""

    def test_construction(self):
        iv = IndividualVote(
            session="91st (2025-2026)",
            bill_number="SB 1",
            bill_title="AN ACT concerning taxation",
            vote_id="je_20250320203513",
            vote_datetime="2025-03-20T20:35:13",
            vote_date="03/20/2025",
            chamber="Senate",
            motion="Emergency Final Action",
            legislator_name="Doe, John",
            legislator_slug="sen_doe_john_1",
            vote="Yea",
        )
        assert iv.bill_number == "SB 1"
        assert iv.vote == "Yea"

    def test_frozen(self):
        iv = IndividualVote(
            session="s",
            bill_number="SB 1",
            bill_title="t",
            vote_id="v",
            vote_datetime="dt",
            vote_date="d",
            chamber="Senate",
            motion="m",
            legislator_name="n",
            legislator_slug="s",
            vote="Yea",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            iv.vote = "Nay"  # type: ignore[misc]


# ── RollCall ─────────────────────────────────────────────────────────────────


class TestRollCall:
    """RollCall frozen dataclass with defaults."""

    def test_defaults(self):
        rc = RollCall(
            session="91st (2025-2026)",
            bill_number="SB 1",
            bill_title="AN ACT",
            vote_id="je_20250320203513",
            vote_url="https://example.com/vote",
            vote_datetime="2025-03-20T20:35:13",
            vote_date="03/20/2025",
            chamber="Senate",
            motion="Final Action",
            vote_type="Final Action",
            result="Passed",
            short_title="Taxation",
            sponsor="Senator Steffen",
        )
        assert rc.yea_count == 0
        assert rc.nay_count == 0
        assert rc.total_votes == 0
        assert rc.passed is None

    def test_frozen(self):
        rc = RollCall(
            session="s",
            bill_number="SB 1",
            bill_title="t",
            vote_id="v",
            vote_url="u",
            vote_datetime="dt",
            vote_date="d",
            chamber="Senate",
            motion="m",
            vote_type="vt",
            result="r",
            short_title="st",
            sponsor="sp",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rc.yea_count = 99  # type: ignore[misc]


# ── VoteLink ─────────────────────────────────────────────────────────────────


class TestVoteLink:
    """VoteLink frozen dataclass."""

    def test_construction(self):
        vl = VoteLink(
            bill_number="SB 1",
            bill_path="/li/b2025_26/measures/sb1/",
            vote_url="https://example.com/vote_view/je_123/",
            vote_text="Emergency Final Action",
        )
        assert vl.bill_number == "SB 1"
        assert vl.vote_text == "Emergency Final Action"
        assert vl.is_odt is False

    def test_odt_link(self):
        """VoteLink with is_odt=True for ODT vote pages."""
        vl = VoteLink(
            bill_number="SB 1",
            bill_path="/li_2014/b2013_14/measures/sb1/",
            vote_url="https://example.com/odt_view/je_123.odt",
            vote_text="Final Action",
            is_odt=True,
        )
        assert vl.is_odt is True

    def test_frozen(self):
        vl = VoteLink(
            bill_number="SB 1",
            bill_path="/p",
            vote_url="http://u",
            vote_text="t",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            vl.bill_number = "HB 2"  # type: ignore[misc]


# ── FetchFailure ─────────────────────────────────────────────────────────────


class TestFetchFailure:
    """FetchFailure frozen dataclass."""

    def test_construction(self):
        ff = FetchFailure(
            bill_number="SB 1",
            vote_text="Final Action",
            vote_url="https://example.com/vote",
            bill_path="/li/b2025_26/measures/sb1/",
            status_code=500,
            error_type="transient",
            error_message="Internal Server Error",
            timestamp="2025-03-20T20:35:13",
        )
        assert ff.error_type == "transient"
        assert ff.status_code == 500

    def test_frozen(self):
        ff = FetchFailure(
            bill_number="SB 1",
            vote_text="t",
            vote_url="u",
            bill_path="/p",
            status_code=None,
            error_type="timeout",
            error_message="Timed out",
            timestamp="ts",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ff.error_type = "permanent"  # type: ignore[misc]
