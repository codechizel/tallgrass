# M4: Hemicycle Charts

Add parliament-style hemicircle seat charts to visualize chamber composition and per-vote breakdowns.

**Roadmap item:** R21 (parliament/hemicircle charts for vote composition)
**Estimated effort:** 1 session
**Dependencies:** None

---

## Goal

Hemicycle (semicircle) diagrams are the standard way to visualize legislative body composition. Seats arranged in concentric arcs, colored by party (or by vote on a specific roll call). Visually striking and immediately legible to a general audience.

---

## Design

### New File: `analysis/viz_helpers.py`

Create a shared visualization helper module at the `analysis/` root (not inside a phase directory) so multiple phases can reuse it.

```python
"""Shared visualization helpers for legislative analysis."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def make_hemicycle_chart(
    seats: list[dict[str, str | int]],
    title: str,
    *,
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """Create a hemicycle (semicircle parliament) chart.

    Args:
        seats: List of dicts with keys "label" (str), "color" (str), "count" (int).
            Each entry represents a group (e.g., party) with its seat count and color.
        title: Chart title.

    Returns:
        Plotly Figure with hemicycle layout.
    """
    ...
```

### Algorithm: Seat Layout

Distribute N seats across concentric semicircular arcs:

1. **Total seats:** Sum of all group counts (Kansas House = 125, Senate = 40)
2. **Number of rows:** `ceil(sqrt(N / pi))` — gives a visually balanced aspect ratio
3. **Seats per row:** Distribute proportionally, with outer rows holding more seats
4. **Seat positions:** For row `r` with `k` seats, place at angles `theta_j = pi * (j + 0.5) / k` for `j in range(k)`, radius `= r_inner + r * row_spacing`
5. **Assignment:** Fill seats left-to-right (conservative → liberal convention) by group order

```python
def _compute_seat_positions(total_seats: int, n_rows: int | None = None) -> list[tuple[float, float]]:
    """Compute (x, y) positions for seats in a hemicycle layout.

    Returns positions in order from left-to-right, bottom-to-top.
    """
    if n_rows is None:
        n_rows = max(2, round(math.sqrt(total_seats / math.pi)))

    # Distribute seats across rows (outer rows get more)
    row_radii = [1.0 + i * 0.4 for i in range(n_rows)]
    row_weights = row_radii  # proportional to circumference
    total_weight = sum(row_weights)
    row_counts = [max(1, round(total_seats * w / total_weight)) for w in row_weights]

    # Adjust to match exact total
    diff = total_seats - sum(row_counts)
    for i in range(abs(diff)):
        idx = -(i + 1) % n_rows  # adjust outer rows first
        row_counts[idx] += 1 if diff > 0 else -1

    positions = []
    for r_idx, (radius, count) in enumerate(zip(row_radii, row_counts)):
        for j in range(count):
            theta = math.pi * (j + 0.5) / count
            x = -radius * math.cos(theta)  # left-to-right
            y = radius * math.sin(theta)
            positions.append((x, y))

    return positions
```

### Plotly Rendering

```python
# Inside make_hemicycle_chart():
positions = _compute_seat_positions(total_seats)

# Assign colors by group order
colors = []
for group in seats:
    colors.extend([group["color"]] * group["count"])

xs = [p[0] for p in positions]
ys = [p[1] for p in positions]

fig = go.Figure(
    data=go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker={"size": max(6, min(14, 600 // total_seats)), "color": colors},
        hoverinfo="skip",
    )
)
fig.update_layout(
    title=title,
    showlegend=False,
    xaxis={"visible": False, "scaleanchor": "y"},
    yaxis={"visible": False},
    width=width, height=height,
    margin={"l": 20, "r": 20, "t": 50, "b": 20},
)
```

### Color Scheme

Use project-standard party colors (already defined across phases):

```python
PARTY_COLORS = {
    "Republican": "#E81B23",  # red
    "Democrat": "#0015BC",    # blue
    "Independent": "#808080", # gray
}

# Per-vote coloring:
VOTE_COLORS = {
    "Yea": "#2ecc71",                    # green
    "Nay": "#e74c3c",                    # red
    "Absent and Not Voting": "#95a5a6",  # gray
    "Present and Passing": "#f39c12",    # amber
    "Not Voting": "#bdc3c7",             # light gray
}
```

---

## Integration: EDA Report

### Insertion Point

`analysis/01_eda/eda_report.py` — function `_add_chamber_party_composition()` at line 146.

Currently renders a `TableSection` showing a chamber × party crosstab. Insert the hemicycle immediately after the table:

```python
def _add_chamber_party_composition(report, plots_dir, metadata, legislators):
    # ... existing table logic (lines 146-172) ...

    # NEW: hemicycle chart per chamber
    for chamber in ["House", "Senate"]:
        chamber_legs = legislators.filter(pl.col("chamber") == chamber)
        party_counts = chamber_legs.group_by("party").len().sort("party")

        seats = [
            {"label": row["party"], "color": PARTY_COLORS.get(row["party"], "#808080"),
             "count": row["len"]}
            for row in party_counts.iter_rows(named=True)
        ]
        fig = make_hemicycle_chart(seats, f"{chamber} — Party Composition")
        html = fig.to_html(include_plotlyjs="cdn", div_id=f"hemicycle-{chamber.lower()}")
        report.add(InteractiveSection(
            id=f"hemicycle-{chamber.lower()}",
            title=f"{chamber} Party Composition (Hemicycle)",
            html=html,
            caption=f"Each dot represents one {chamber} seat, colored by party.",
        ))
```

### Optional: Per-Vote Hemicycle

For specific roll calls (e.g., veto overrides), color seats by vote category instead of party. This could be added to the Profiles report or a standalone vote viewer:

```python
def make_vote_hemicycle(
    vote_df: pl.DataFrame,  # one row per legislator for a single roll call
    title: str,
) -> go.Figure:
    """Color seats by vote category for a specific roll call."""
    seats = [
        {"label": cat, "color": VOTE_COLORS[cat],
         "count": vote_df.filter(pl.col("vote") == cat).height}
        for cat in VOTE_CATEGORIES
        if vote_df.filter(pl.col("vote") == cat).height > 0
    ]
    return make_hemicycle_chart(seats, title)
```

---

## Tests

Add tests in `tests/test_viz_helpers.py`:

```python
class TestHemicycleLayout:
    def test_total_seats_matches_input(self):
        """All seats are placed."""
        positions = _compute_seat_positions(125)
        assert len(positions) == 125

    def test_all_y_positive(self):
        """Hemicycle is above x-axis."""
        positions = _compute_seat_positions(40)
        assert all(y >= 0 for _, y in positions)

    def test_small_chamber(self):
        """Works for Senate (40 seats)."""
        positions = _compute_seat_positions(40)
        assert len(positions) == 40

    def test_single_seat(self):
        """Edge case: single seat."""
        positions = _compute_seat_positions(1)
        assert len(positions) == 1


class TestMakeHemicycleChart:
    def test_returns_plotly_figure(self):
        seats = [{"label": "R", "color": "red", "count": 85},
                 {"label": "D", "color": "blue", "count": 40}]
        fig = make_hemicycle_chart(seats, "Test")
        assert isinstance(fig, go.Figure)

    def test_marker_count_matches_total(self):
        seats = [{"label": "R", "color": "red", "count": 85},
                 {"label": "D", "color": "blue", "count": 40}]
        fig = make_hemicycle_chart(seats, "Test")
        assert len(fig.data[0].x) == 125
```

---

## Verification

```bash
just test -k "test_viz_helpers" -v    # new tests pass
just lint-check                       # formatting
just eda 2025-26                      # regenerate EDA report, inspect hemicycle
```

## Documentation

- Update `docs/roadmap.md` item R21 to "Done"
- No ADR needed (additive visualization, no architectural decision)

## Commit

```
feat(infra): hemicycle parliament charts for chamber composition [vYYYY.MM.DD.N]
```
