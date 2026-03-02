"""Tests for analysis/viz_helpers.py — hemicycle chart layout and rendering."""

import plotly.graph_objects as go
import pytest

from analysis.viz_helpers import (
    PARTY_COLORS,
    VOTE_COLORS,
    _compute_seat_positions,
    make_hemicycle_chart,
    make_vote_hemicycle,
)


class TestComputeSeatPositions:
    """Tests for the hemicycle seat layout algorithm."""

    def test_total_seats_matches_input_house(self):
        """All 125 House seats are placed."""
        positions = _compute_seat_positions(125)
        assert len(positions) == 125

    def test_total_seats_matches_input_senate(self):
        """All 40 Senate seats are placed."""
        positions = _compute_seat_positions(40)
        assert len(positions) == 40

    def test_all_y_positive(self):
        """Hemicycle is above x-axis (semicircle)."""
        positions = _compute_seat_positions(40)
        assert all(y >= 0 for _, y in positions)

    def test_all_y_positive_large(self):
        """Y >= 0 for large chambers too."""
        positions = _compute_seat_positions(125)
        assert all(y >= 0 for _, y in positions)

    def test_single_seat(self):
        """Edge case: single seat is placed."""
        positions = _compute_seat_positions(1)
        assert len(positions) == 1

    def test_two_seats(self):
        """Edge case: two seats are placed."""
        positions = _compute_seat_positions(2)
        assert len(positions) == 2

    def test_zero_seats(self):
        """Zero seats returns empty list."""
        positions = _compute_seat_positions(0)
        assert positions == []

    def test_negative_seats(self):
        """Negative seats returns empty list."""
        positions = _compute_seat_positions(-5)
        assert positions == []

    def test_custom_rows(self):
        """Custom row count is respected."""
        positions = _compute_seat_positions(50, n_rows=3)
        assert len(positions) == 50

    def test_x_spans_negative_to_positive(self):
        """Seats span from left (negative x) to right (positive x)."""
        positions = _compute_seat_positions(40)
        xs = [x for x, _ in positions]
        assert min(xs) < 0
        assert max(xs) > 0

    def test_positions_are_unique(self):
        """No two seats occupy the same position."""
        positions = _compute_seat_positions(125)
        # Use rounded positions to avoid floating point noise
        rounded = [(round(x, 6), round(y, 6)) for x, y in positions]
        assert len(set(rounded)) == len(rounded)

    @pytest.mark.parametrize("n", [10, 25, 40, 80, 125, 165, 200])
    def test_various_chamber_sizes(self, n: int):
        """Layout works for a range of chamber sizes."""
        positions = _compute_seat_positions(n)
        assert len(positions) == n
        assert all(y >= 0 for _, y in positions)


class TestMakeHemicycleChart:
    """Tests for the Plotly hemicycle chart builder."""

    def test_returns_plotly_figure(self):
        seats = [
            {"label": "Republican", "color": "#E81B23", "count": 85},
            {"label": "Democrat", "color": "#0015BC", "count": 40},
        ]
        fig = make_hemicycle_chart(seats, "Test")
        assert isinstance(fig, go.Figure)

    def test_marker_count_matches_total(self):
        seats = [
            {"label": "Republican", "color": "#E81B23", "count": 85},
            {"label": "Democrat", "color": "#0015BC", "count": 40},
        ]
        fig = make_hemicycle_chart(seats, "Test")
        # First trace is the scatter; subsequent traces are legend entries
        assert len(fig.data[0].x) == 125

    def test_legend_traces_match_groups(self):
        seats = [
            {"label": "Republican", "color": "#E81B23", "count": 85},
            {"label": "Democrat", "color": "#0015BC", "count": 40},
        ]
        fig = make_hemicycle_chart(seats, "Test")
        # 1 scatter + 2 legend entries
        assert len(fig.data) == 3

    def test_three_party_chart(self):
        seats = [
            {"label": "Republican", "color": "#E81B23", "count": 85},
            {"label": "Democrat", "color": "#0015BC", "count": 38},
            {"label": "Independent", "color": "#808080", "count": 2},
        ]
        fig = make_hemicycle_chart(seats, "Test")
        assert len(fig.data[0].x) == 125
        assert len(fig.data) == 4  # scatter + 3 legend

    def test_empty_seats_returns_figure(self):
        fig = make_hemicycle_chart([], "Empty")
        assert isinstance(fig, go.Figure)

    def test_custom_dimensions(self):
        seats = [{"label": "A", "color": "red", "count": 10}]
        fig = make_hemicycle_chart(seats, "Test", width=600, height=400)
        assert fig.layout.width == 600
        assert fig.layout.height == 400

    def test_title_is_set(self):
        seats = [{"label": "A", "color": "red", "count": 10}]
        fig = make_hemicycle_chart(seats, "My Title")
        assert fig.layout.title.text == "My Title"


class TestMakeVoteHemicycle:
    """Tests for the per-vote hemicycle variant."""

    def test_returns_plotly_figure(self):
        vote_counts = [
            {"label": "Yea", "color": "#2ecc71", "count": 80},
            {"label": "Nay", "color": "#e74c3c", "count": 40},
            {"label": "Absent and Not Voting", "color": "#95a5a6", "count": 5},
        ]
        fig = make_vote_hemicycle(vote_counts, "HB 2001 — Final Action")
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].x) == 125


class TestColorConstants:
    """Verify color constant completeness."""

    def test_party_colors_has_three_entries(self):
        assert len(PARTY_COLORS) == 3
        assert "Republican" in PARTY_COLORS
        assert "Democrat" in PARTY_COLORS
        assert "Independent" in PARTY_COLORS

    def test_vote_colors_has_five_entries(self):
        assert len(VOTE_COLORS) == 5
        expected = {"Yea", "Nay", "Absent and Not Voting", "Present and Passing", "Not Voting"}
        assert set(VOTE_COLORS.keys()) == expected
