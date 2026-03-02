"""
Tests for BillAction dataclass and bill lifecycle CSV export.

Verifies BillAction construction, immutability, HISTORY extraction logic,
and CSV output including committee_names serialization.

Run: uv run pytest tests/test_bill_actions.py -v
"""

import csv
import dataclasses

import pytest

from tallgrass.models import BillAction, IndividualVote, RollCall
from tallgrass.output import save_csvs

pytestmark = pytest.mark.scraper


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_action(
    bill_number: str = "sb1",
    status: str = "Referred to Committee on Taxation",
    committee_names: tuple[str, ...] = ("Committee on Assessment and Taxation",),
) -> BillAction:
    return BillAction(
        session="91st_2025-2026",
        bill_number=bill_number,
        action_code="cr_rsc_282",
        chamber="Senate",
        committee_names=committee_names,
        occurred_datetime="2025-01-13T14:00:00",
        session_date="2025-01-13",
        status=status,
        journal_page_number="42",
    )


def _make_vote(slug: str = "sen_doe_john_1") -> IndividualVote:
    return IndividualVote(
        session="91st (2025-2026)",
        bill_number="SB 1",
        bill_title="AN ACT",
        vote_id="je_20250320203513",
        vote_datetime="2025-03-20T20:35:13",
        vote_date="03/20/2025",
        chamber="Senate",
        motion="Emergency Final Action",
        legislator_name="Doe, John",
        legislator_slug=slug,
        vote="Yea",
    )


def _make_rollcall() -> RollCall:
    return RollCall(
        session="91st (2025-2026)",
        bill_number="SB 1",
        bill_title="AN ACT",
        vote_id="je_20250320203513",
        vote_url="https://example.com/vote",
        vote_datetime="2025-03-20T20:35:13",
        vote_date="03/20/2025",
        chamber="Senate",
        motion="Emergency Final Action",
        vote_type="Emergency Final Action",
        result="Passed",
        short_title="Taxation",
        sponsor="Senator Steffen",
    )


# ── BillAction dataclass ────────────────────────────────────────────────────


class TestBillAction:
    """BillAction frozen dataclass construction and immutability."""

    def test_construction(self):
        action = _make_action()
        assert action.session == "91st_2025-2026"
        assert action.bill_number == "sb1"
        assert action.action_code == "cr_rsc_282"
        assert action.chamber == "Senate"
        assert action.committee_names == ("Committee on Assessment and Taxation",)
        assert action.occurred_datetime == "2025-01-13T14:00:00"
        assert action.session_date == "2025-01-13"
        assert action.status == "Referred to Committee on Taxation"
        assert action.journal_page_number == "42"

    def test_immutability(self):
        action = _make_action()
        with pytest.raises(dataclasses.FrozenInstanceError):
            action.status = "Changed"  # type: ignore[misc]

    def test_tuple_committee_names(self):
        """committee_names must be a tuple (hashable for frozen dataclass)."""
        action = _make_action(committee_names=("Committee A", "Committee B"))
        assert isinstance(action.committee_names, tuple)
        assert len(action.committee_names) == 2

    def test_empty_committee_names(self):
        action = _make_action(committee_names=())
        assert action.committee_names == ()


# ── HISTORY extraction logic ────────────────────────────────────────────────


class TestHistoryExtraction:
    """Tests for KLISS HISTORY -> BillAction conversion logic."""

    def test_history_to_bill_actions(self):
        """Simulate the extraction loop from _filter_bills_with_votes()."""
        history = [
            {
                "action_code": "cr_rsc_282",
                "chamber": "Senate",
                "committee_names": ["Committee on Assessment and Taxation"],
                "occurred_datetime": "2025-01-13T14:00:00",
                "session_date": "2025-01-13",
                "status": "Referred to Committee on Assessment and Taxation",
                "journal_page_number": "42",
            },
            {
                "action_code": "fa_282",
                "chamber": "Senate",
                "committee_names": [],
                "occurred_datetime": "2025-03-20T10:00:00",
                "session_date": "2025-03-20",
                "status": "Emergency Final Action; Yea: 33; Nay: 5",
                "journal_page_number": "256",
            },
        ]
        actions = []
        for entry in history:
            actions.append(
                BillAction(
                    session="91st_2025-2026",
                    bill_number="sb1",
                    action_code=entry.get("action_code", ""),
                    chamber=entry.get("chamber", ""),
                    committee_names=tuple(entry.get("committee_names", [])),
                    occurred_datetime=entry.get("occurred_datetime", ""),
                    session_date=entry.get("session_date", ""),
                    status=entry.get("status", ""),
                    journal_page_number=str(entry.get("journal_page_number", "")),
                )
            )
        assert len(actions) == 2
        assert actions[0].status == "Referred to Committee on Assessment and Taxation"
        assert actions[1].committee_names == ()

    def test_empty_history(self):
        """Bills without HISTORY produce no actions."""
        history: list[dict] = []
        actions = [
            BillAction(
                session="s",
                bill_number="sb1",
                action_code=e.get("action_code", ""),
                chamber=e.get("chamber", ""),
                committee_names=tuple(e.get("committee_names", [])),
                occurred_datetime=e.get("occurred_datetime", ""),
                session_date=e.get("session_date", ""),
                status=e.get("status", ""),
                journal_page_number=str(e.get("journal_page_number", "")),
            )
            for e in history
        ]
        assert actions == []

    def test_all_bills_captured(self):
        """Actions should be captured for ALL bills, not just those with votes."""
        # Bill with no vote-indicating status
        history_no_vote = [
            {
                "action_code": "intro",
                "chamber": "Senate",
                "committee_names": [],
                "occurred_datetime": "2025-01-13T14:00:00",
                "session_date": "2025-01-13",
                "status": "Introduced",
                "journal_page_number": "10",
            },
        ]
        # Bill with vote
        history_with_vote = [
            {
                "action_code": "fa",
                "chamber": "Senate",
                "committee_names": [],
                "occurred_datetime": "2025-03-20T10:00:00",
                "session_date": "2025-03-20",
                "status": "Emergency Final Action; Yea: 33; Nay: 5",
                "journal_page_number": "256",
            },
        ]
        all_actions: list[BillAction] = []
        for history in [history_no_vote, history_with_vote]:
            for entry in history:
                all_actions.append(
                    BillAction(
                        session="s",
                        bill_number="sb1",
                        action_code=entry.get("action_code", ""),
                        chamber=entry.get("chamber", ""),
                        committee_names=tuple(entry.get("committee_names", [])),
                        occurred_datetime=entry.get("occurred_datetime", ""),
                        session_date=entry.get("session_date", ""),
                        status=entry.get("status", ""),
                        journal_page_number=str(entry.get("journal_page_number", "")),
                    )
                )
        # Both bills captured (funnel includes non-vote bills)
        assert len(all_actions) == 2

    def test_missing_fields_default(self):
        """Missing HISTORY entry fields default to empty strings."""
        entry: dict = {}
        action = BillAction(
            session="s",
            bill_number="sb1",
            action_code=entry.get("action_code", ""),
            chamber=entry.get("chamber", ""),
            committee_names=tuple(entry.get("committee_names", [])),
            occurred_datetime=entry.get("occurred_datetime", ""),
            session_date=entry.get("session_date", ""),
            status=entry.get("status", ""),
            journal_page_number=str(entry.get("journal_page_number", "")),
        )
        assert action.action_code == ""
        assert action.chamber == ""
        assert action.committee_names == ()
        assert action.status == ""


# ── CSV export ──────────────────────────────────────────────────────────────


class TestBillActionsCsv:
    """Tests for bill_actions.csv export via save_csvs()."""

    def test_actions_csv_written(self, tmp_path):
        """save_csvs() creates _bill_actions.csv when actions are provided."""
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[_make_vote()],
            rollcalls=[_make_rollcall()],
            legislators={},
            bill_actions=[_make_action()],
        )
        assert (tmp_path / "test_bill_actions.csv").exists()

    def test_no_actions_csv_when_none(self, tmp_path):
        """No file created when bill_actions is None."""
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[],
            rollcalls=[],
            legislators={},
            bill_actions=None,
        )
        assert not (tmp_path / "test_bill_actions.csv").exists()

    def test_no_actions_csv_when_empty(self, tmp_path):
        """No file created when bill_actions is empty list."""
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[],
            rollcalls=[],
            legislators={},
            bill_actions=[],
        )
        assert not (tmp_path / "test_bill_actions.csv").exists()

    def test_actions_csv_headers(self, tmp_path):
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[],
            rollcalls=[],
            legislators={},
            bill_actions=[_make_action()],
        )
        with open(tmp_path / "test_bill_actions.csv") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "session",
                "bill_number",
                "action_code",
                "chamber",
                "committee_names",
                "occurred_datetime",
                "session_date",
                "status",
                "journal_page_number",
            ]

    def test_committee_names_joined(self, tmp_path):
        """Tuple committee_names are semicolon-joined in CSV."""
        action = _make_action(committee_names=("Committee A", "Committee B"))
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[],
            rollcalls=[],
            legislators={},
            bill_actions=[action],
        )
        with open(tmp_path / "test_bill_actions.csv") as f:
            row = next(csv.DictReader(f))
            assert row["committee_names"] == "Committee A; Committee B"

    def test_empty_committee_names_joined(self, tmp_path):
        """Empty tuple produces empty string in CSV."""
        action = _make_action(committee_names=())
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[],
            rollcalls=[],
            legislators={},
            bill_actions=[action],
        )
        with open(tmp_path / "test_bill_actions.csv") as f:
            row = next(csv.DictReader(f))
            assert row["committee_names"] == ""

    def test_actions_csv_roundtrip(self, tmp_path):
        """Write -> read preserves all fields."""
        action = _make_action()
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[],
            rollcalls=[],
            legislators={},
            bill_actions=[action],
        )
        with open(tmp_path / "test_bill_actions.csv") as f:
            row = next(csv.DictReader(f))
            assert row["session"] == "91st_2025-2026"
            assert row["bill_number"] == "sb1"
            assert row["action_code"] == "cr_rsc_282"
            assert row["chamber"] == "Senate"
            assert row["occurred_datetime"] == "2025-01-13T14:00:00"
            assert row["session_date"] == "2025-01-13"
            assert row["status"] == "Referred to Committee on Taxation"
            assert row["journal_page_number"] == "42"

    def test_actions_csv_row_count(self, tmp_path):
        """Multiple actions produce correct row count."""
        actions = [
            _make_action(bill_number="sb1"),
            _make_action(bill_number="sb2"),
            _make_action(bill_number="hb100"),
        ]
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[],
            rollcalls=[],
            legislators={},
            bill_actions=actions,
        )
        with open(tmp_path / "test_bill_actions.csv") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 3

    def test_existing_csvs_still_created(self, tmp_path):
        """Adding bill_actions doesn't break the existing 3 CSVs."""
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[_make_vote()],
            rollcalls=[_make_rollcall()],
            legislators={"sen_doe_john_1": {"name": "Doe", "slug": "sen_doe_john_1"}},
            bill_actions=[_make_action()],
        )
        assert (tmp_path / "test_votes.csv").exists()
        assert (tmp_path / "test_rollcalls.csv").exists()
        assert (tmp_path / "test_legislators.csv").exists()
        assert (tmp_path / "test_bill_actions.csv").exists()
