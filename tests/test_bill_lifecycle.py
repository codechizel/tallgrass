"""
Tests for bill lifecycle classification and Sankey visualization.

Verifies status text classification, transition computation, Died inference,
and Plotly Sankey figure generation.

Run: uv run pytest tests/test_bill_lifecycle.py -v
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.bill_lifecycle import (  # noqa: E402
    LIFECYCLE_STAGES,
    STAGE_COLORS,
    STAGE_ORDER,
    classify_action,
    compute_bill_transitions,
    plot_bill_lifecycle_sankey,
)

# ── classify_action() ──────────────────────────────────────────────────────


class TestClassifyAction:
    """Map KLISS status strings to canonical lifecycle stages."""

    def test_introduced(self):
        assert classify_action("Introduced") == "Introduced"

    def test_committee_referral(self):
        assert classify_action("Referred to Committee on Taxation") == "Committee Referral"

    def test_committee_referral_case_insensitive(self):
        assert classify_action("REFERRED TO COMMITTEE ON TAXATION") == "Committee Referral"

    def test_hearing(self):
        assert classify_action("Scheduled for hearing on March 5") == "Hearing"

    def test_committee_report(self):
        result = classify_action("Committee report recommending bill be passed as amended")
        assert result == "Committee Report"

    def test_committee_of_the_whole(self):
        assert classify_action("Committee of the Whole") == "Committee of the Whole"

    def test_floor_vote_final_action(self):
        assert classify_action("Final Action") == "Floor Vote"

    def test_floor_vote_emergency(self):
        assert classify_action("Emergency Final Action") == "Floor Vote"

    def test_floor_vote_with_tally(self):
        assert classify_action("Emergency Final Action; Yea: 33; Nay: 5") == "Floor Vote"

    def test_cross_chamber_received(self):
        assert classify_action("Received by House") == "Cross-Chamber"

    def test_cross_chamber_transmitted(self):
        assert classify_action("Transmitted to Senate") == "Cross-Chamber"

    def test_governor_enrolled(self):
        assert classify_action("Enrolled and presented to Governor") == "Governor"

    def test_signed_into_law(self):
        assert classify_action("Approved by Governor") == "Signed into Law"

    def test_vetoed(self):
        assert classify_action("Vetoed by Governor") == "Vetoed"

    def test_line_item_veto(self):
        assert classify_action("Line item veto") == "Vetoed"

    def test_unknown_returns_other(self):
        assert classify_action("Some unknown legislative action") == "Other"

    def test_empty_string(self):
        assert classify_action("") == "Other"


# ── compute_bill_transitions() ──────────────────────────────────────────────


class TestComputeTransitions:
    """Bill stage transition counting."""

    def _make_actions_df(self, rows: list[dict]) -> pl.DataFrame:
        """Build a minimal actions DataFrame."""
        return pl.DataFrame(rows)

    def test_basic_transitions(self):
        """Two-step bill: Intro -> Committee Referral."""
        df = self._make_actions_df(
            [
                {
                    "bill_number": "sb1",
                    "occurred_datetime": "2025-01-10T10:00:00",
                    "status": "Introduced",
                },
                {
                    "bill_number": "sb1",
                    "occurred_datetime": "2025-01-13T14:00:00",
                    "status": "Referred to Committee on Taxation",
                },
            ]
        )
        result = compute_bill_transitions(df)
        assert result.height >= 1
        row = result.filter(
            (pl.col("source") == "Introduced") & (pl.col("target") == "Committee Referral")
        )
        assert row.height == 1
        assert row["value"][0] == 1

    def test_deduplicates_consecutive_same_stage(self):
        """Consecutive same-stage entries should be collapsed."""
        df = self._make_actions_df(
            [
                {
                    "bill_number": "sb1",
                    "occurred_datetime": "2025-01-10T10:00:00",
                    "status": "Referred to Committee on Taxation",
                },
                {
                    "bill_number": "sb1",
                    "occurred_datetime": "2025-01-11T10:00:00",
                    "status": "Referred to Committee on Assessment",
                },
                {
                    "bill_number": "sb1",
                    "occurred_datetime": "2025-02-05T10:00:00",
                    "status": "Committee report recommending passage",
                },
            ]
        )
        result = compute_bill_transitions(df)
        # Should have Committee Referral -> Committee Report, not two Committee Referral entries
        referral_to_report = result.filter(
            (pl.col("source") == "Committee Referral") & (pl.col("target") == "Committee Report")
        )
        assert referral_to_report.height == 1
        assert referral_to_report["value"][0] == 1

    def test_died_inference(self):
        """Bill that stalls in committee should get a 'Died' transition."""
        df = self._make_actions_df(
            [
                {
                    "bill_number": "sb99",
                    "occurred_datetime": "2025-01-10T10:00:00",
                    "status": "Introduced",
                },
                {
                    "bill_number": "sb99",
                    "occurred_datetime": "2025-01-13T14:00:00",
                    "status": "Referred to Committee on Taxation",
                },
            ]
        )
        result = compute_bill_transitions(df)
        died = result.filter(pl.col("target") == "Died")
        assert died.height == 1
        assert died["source"][0] == "Committee Referral"

    def test_no_died_for_signed_bill(self):
        """Bill that reaches Signed into Law should not get Died."""
        df = self._make_actions_df(
            [
                {
                    "bill_number": "sb1",
                    "occurred_datetime": "2025-01-10T10:00:00",
                    "status": "Introduced",
                },
                {
                    "bill_number": "sb1",
                    "occurred_datetime": "2025-03-20T10:00:00",
                    "status": "Emergency Final Action; Yea: 33; Nay: 5",
                },
                {
                    "bill_number": "sb1",
                    "occurred_datetime": "2025-04-01T10:00:00",
                    "status": "Approved by Governor",
                },
            ]
        )
        result = compute_bill_transitions(df)
        died = result.filter(pl.col("target") == "Died")
        assert died.height == 0

    def test_multiple_bills(self):
        """Transitions aggregate across multiple bills."""
        df = self._make_actions_df(
            [
                {
                    "bill_number": "sb1",
                    "occurred_datetime": "2025-01-10T10:00:00",
                    "status": "Introduced",
                },
                {
                    "bill_number": "sb1",
                    "occurred_datetime": "2025-01-13T14:00:00",
                    "status": "Referred to Committee on Taxation",
                },
                {
                    "bill_number": "sb2",
                    "occurred_datetime": "2025-01-10T10:00:00",
                    "status": "Introduced",
                },
                {
                    "bill_number": "sb2",
                    "occurred_datetime": "2025-01-14T14:00:00",
                    "status": "Referred to Committee on Education",
                },
            ]
        )
        result = compute_bill_transitions(df)
        intro_to_referral = result.filter(
            (pl.col("source") == "Introduced") & (pl.col("target") == "Committee Referral")
        )
        assert intro_to_referral.height == 1
        assert intro_to_referral["value"][0] == 2

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty transitions."""
        df = pl.DataFrame({"bill_number": [], "occurred_datetime": [], "status": []}).cast(
            {"bill_number": pl.Utf8, "occurred_datetime": pl.Utf8, "status": pl.Utf8}
        )
        result = compute_bill_transitions(df)
        assert result.height == 0
        assert "source" in result.columns
        assert "target" in result.columns
        assert "value" in result.columns


# ── plot_bill_lifecycle_sankey() ────────────────────────────────────────────


class TestSankeyPlot:
    """Plotly Sankey figure generation."""

    def _make_actions_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "bill_number": "sb1",
                    "occurred_datetime": "2025-01-10T10:00:00",
                    "status": "Introduced",
                },
                {
                    "bill_number": "sb1",
                    "occurred_datetime": "2025-01-13T14:00:00",
                    "status": "Referred to Committee on Taxation",
                },
                {
                    "bill_number": "sb1",
                    "occurred_datetime": "2025-03-20T10:00:00",
                    "status": "Emergency Final Action; Yea: 33; Nay: 5",
                },
            ]
        )

    def test_produces_figure(self):
        """Valid actions produce a Plotly Figure object."""
        fig = plot_bill_lifecycle_sankey(self._make_actions_df())
        assert fig is not None
        # Check it's a Plotly figure
        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)

    def test_sankey_has_data(self):
        """Figure contains Sankey trace with nodes and links."""
        fig = plot_bill_lifecycle_sankey(self._make_actions_df())
        assert fig is not None
        trace = fig.data[0]
        assert len(trace.node.label) > 0
        assert len(trace.link.source) > 0

    def test_empty_data_returns_none(self):
        """Empty DataFrame returns None."""
        df = pl.DataFrame({"bill_number": [], "occurred_datetime": [], "status": []}).cast(
            {"bill_number": pl.Utf8, "occurred_datetime": pl.Utf8, "status": pl.Utf8}
        )
        result = plot_bill_lifecycle_sankey(df)
        assert result is None

    def test_custom_title(self):
        """Custom title is applied to the figure."""
        fig = plot_bill_lifecycle_sankey(self._make_actions_df(), title="Custom Title")
        assert fig is not None
        assert fig.layout.title.text == "Custom Title"


# ── Constants ───────────────────────────────────────────────────────────────


class TestConstants:
    """Sanity checks on module constants."""

    def test_all_stages_have_colors(self):
        """Every stage in STAGE_ORDER has a color defined."""
        for stage in STAGE_ORDER:
            assert stage in STAGE_COLORS, f"Missing color for stage: {stage}"

    def test_lifecycle_stages_non_empty(self):
        """LIFECYCLE_STAGES has entries."""
        assert len(LIFECYCLE_STAGES) > 0

    def test_stage_order_includes_died(self):
        """STAGE_ORDER includes the inferred Died stage."""
        assert "Died" in STAGE_ORDER

    def test_stage_order_includes_other(self):
        """STAGE_ORDER includes the Other fallback."""
        assert "Other" in STAGE_ORDER
