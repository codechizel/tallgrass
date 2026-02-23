"""Cross-session validation HTML report builder.

Builds ~15 sections (tables, figures, and text) for the cross-session
comparison report. Each section is a small function that slices/aggregates
polars DataFrames and calls make_gt() or FigureSection.from_file().

Usage (called from cross_session.py):
    from analysis.cross_session_report import build_cross_session_report
    build_cross_session_report(ctx.report, results=..., plots_dir=...)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

try:
    from analysis.report import FigureSection, ReportBuilder, TableSection, TextSection, make_gt
except ModuleNotFoundError:
    from report import (  # type: ignore[no-redef]
        FigureSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
    )

try:
    from analysis.cross_session_data import ALIGNMENT_TRIM_PCT, CORRELATION_WARN, SHIFT_THRESHOLD_SD
except ModuleNotFoundError:
    from cross_session_data import (  # type: ignore[no-redef]
        ALIGNMENT_TRIM_PCT,
        CORRELATION_WARN,
        SHIFT_THRESHOLD_SD,
    )


def build_cross_session_report(
    report: ReportBuilder,
    *,
    results: dict,
    plots_dir: Path,
    session_a_label: str,
    session_b_label: str,
) -> None:
    """Build the full cross-session HTML report by adding sections."""
    _add_overview(report, results, session_a_label, session_b_label)
    _add_matching_summary(report, results)

    for chamber in sorted(results["chambers"]):
        cr = results[chamber]
        _add_ideology_scatter(report, plots_dir, chamber)
        _add_biggest_movers_figure(report, plots_dir, chamber)
        _add_biggest_movers_table(report, cr["shifted"], chamber)
        _add_shift_distribution(report, plots_dir, chamber)
        _add_turnover_figure(report, plots_dir, chamber)
        _add_metric_stability_table(report, cr["stability"], chamber)

    _add_detection_validation(report, results)
    _add_methodology(report, results, session_a_label, session_b_label)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_overview(
    report: ReportBuilder,
    results: dict,
    session_a_label: str,
    session_b_label: str,
) -> None:
    matched = results["matched"]
    n_matched = matched.height
    n_chamber_switch = int(matched["is_chamber_switch"].sum())
    n_party_switch = int(matched["is_party_switch"].sum())
    total = n_matched + results["n_departing"] + results["n_new"]
    overlap_pct = n_matched / total * 100

    report.add(
        TextSection(
            id="overview",
            title="Overview",
            html=(
                f"<p>This report compares the <strong>{session_a_label}</strong> "
                f"and <strong>{session_b_label}</strong> Kansas Legislature sessions "
                "to answer three questions: <em>Who moved ideologically?</em> "
                "<em>Are our predictive models honest?</em> "
                "<em>Do our detection methods generalize?</em></p>"
                f"<p><strong>{n_matched} legislators</strong> served in both "
                f"sessions ({overlap_pct:.0f}% "
                f"overlap). {results['n_departing']} departed after "
                f"{session_a_label}, "
                f"and {results['n_new']} are new in {session_b_label}."
                + (
                    f" {n_chamber_switch} legislator(s) switched chambers."
                    if n_chamber_switch
                    else ""
                )
                + (f" {n_party_switch} legislator(s) switched parties." if n_party_switch else "")
                + "</p>"
            ),
        )
    )


def _add_matching_summary(report: ReportBuilder, results: dict) -> None:
    matched = results["matched"]
    rows = []
    for chamber in ["House", "Senate"]:
        n = matched.filter(pl.col("chamber_b") == chamber).height
        if n > 0:
            rows.append({"Chamber": chamber, "Returning": n})

    if rows:
        df = pl.DataFrame(rows)
        html = make_gt(
            df,
            title="Returning Legislators by Chamber",
            source_note="Matched by normalized full_name across sessions.",
        )
        report.add(TableSection(id="matching-summary", title="Matching Summary", html=html))


def _add_ideology_scatter(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"ideology_shift_scatter_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-shift-scatter-{chamber.lower()}",
                f"{chamber} — Who Moved?",
                path,
                caption=(
                    "Each dot is a returning legislator. The diagonal line marks 'no change' — "
                    "legislators above the line moved rightward (more conservative), below moved "
                    "leftward (more liberal). Labeled names are the biggest movers."
                ),
            )
        )


def _add_biggest_movers_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"biggest_movers_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-movers-{chamber.lower()}",
                f"{chamber} — Biggest Ideological Movers",
                path,
                caption=(
                    "Horizontal bars show the magnitude and direction of ideology shift. "
                    "Red bars = moved rightward (more conservative). Blue bars = moved leftward "
                    "(more liberal). Only the top movers are shown."
                ),
            )
        )


def _add_biggest_movers_table(report: ReportBuilder, shifted: pl.DataFrame, chamber: str) -> None:
    movers = shifted.filter(pl.col("is_significant_mover")).sort("abs_delta_xi", descending=True)
    if movers.height == 0:
        report.add(
            TextSection(
                id=f"movers-table-{chamber.lower()}",
                title=f"{chamber} — Significant Movers",
                html="<p>No legislators moved more than 1 standard deviation.</p>",
            )
        )
        return

    display = movers.select(
        "full_name",
        "party",
        "xi_a_aligned",
        "xi_b",
        "delta_xi",
        "shift_direction",
        "rank_shift",
    )

    html = make_gt(
        display,
        title=f"{chamber} — Significant Ideological Movers ({display.height} legislators)",
        subtitle="Legislators who shifted more than 1 SD between sessions",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "xi_a_aligned": "Previous Ideology",
            "xi_b": "Current Ideology",
            "delta_xi": "Shift",
            "shift_direction": "Direction",
            "rank_shift": "Rank Change",
        },
        number_formats={
            "xi_a_aligned": ".2f",
            "xi_b": ".2f",
            "delta_xi": "+.2f",
        },
        source_note=(
            "Previous Ideology is the aligned IRT ideal point from the earlier session. "
            "Positive shift = moved rightward (more conservative). "
            "Rank Change: positive = moved rightward in ranking."
        ),
    )
    report.add(
        TableSection(
            id=f"movers-table-{chamber.lower()}",
            title=f"{chamber} Significant Movers",
            html=html,
        )
    )


def _add_shift_distribution(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"shift_distribution_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-shift-dist-{chamber.lower()}",
                f"{chamber} — Distribution of Ideology Shifts",
                path,
                caption=(
                    "Histogram of ideology shifts for all returning legislators. The dashed lines "
                    "mark the 'significant mover' threshold (1 SD). Most legislators cluster near "
                    "zero (no change); outliers are the movers highlighted above."
                ),
            )
        )


def _add_turnover_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"turnover_impact_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-turnover-{chamber.lower()}",
                f"{chamber} — Turnover Impact on Ideology",
                path,
                caption=(
                    "Ideology distributions of departing, returning, and new legislators. "
                    "If new legislators are further right (or left) than departing ones, "
                    "the chamber's overall composition has shifted."
                ),
            )
        )


def _add_metric_stability_table(
    report: ReportBuilder, stability: pl.DataFrame, chamber: str
) -> None:
    if stability.height == 0:
        return

    display_names = {
        "unity_score": "Party Unity (CQ)",
        "maverick_rate": "Maverick Rate",
        "weighted_maverick": "Weighted Maverick",
        "betweenness": "Network Influence",
        "eigenvector": "Eigenvector Centrality",
        "pagerank": "PageRank",
        "loyalty_rate": "Clustering Loyalty",
        "PC1": "PCA Dimension 1",
    }

    display = stability.with_columns(
        pl.col("metric")
        .map_elements(lambda m: display_names.get(m, m), return_dtype=pl.Utf8)
        .alias("Metric")
    ).select("Metric", "pearson_r", "spearman_rho", "n_legislators")

    html = make_gt(
        display,
        title=f"{chamber} — Metric Stability Across Sessions",
        subtitle="How consistent are legislative metrics for returning legislators?",
        column_labels={
            "Metric": "Metric",
            "pearson_r": "Pearson r",
            "spearman_rho": "Spearman rho",
            "n_legislators": "N",
        },
        number_formats={"pearson_r": ".3f", "spearman_rho": ".3f"},
        source_note=(
            f"Pearson r < {CORRELATION_WARN} would indicate weak stability. "
            "Spearman rho measures rank-order consistency (less sensitive to outliers)."
        ),
    )
    report.add(
        TableSection(
            id=f"stability-{chamber.lower()}",
            title=f"{chamber} Metric Stability",
            html=html,
        )
    )


def _add_detection_validation(report: ReportBuilder, results: dict) -> None:
    rows = []
    for chamber in sorted(results["chambers"]):
        det = results[chamber].get("detection", {})
        for role in ["maverick", "bridge", "paradox"]:
            name_a = det.get(f"{role}_a")
            name_b = det.get(f"{role}_b")
            rows.append(
                {
                    "Chamber": chamber,
                    "Role": role.title(),
                    "Previous Session": name_a or "Not detected",
                    "Current Session": name_b or "Not detected",
                    "Same Person?": "Yes" if (name_a and name_b and name_a == name_b) else "No",
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Detection Threshold Validation",
        subtitle="Do the synthesis detection methods identify the same roles across sessions?",
        source_note=(
            "Mavericks, bridge-builders, and metric paradoxes are detected using the same "
            "thresholds on both sessions. Consistency suggests the thresholds generalize."
        ),
    )
    report.add(
        TableSection(
            id="detection-validation",
            title="Detection Validation",
            html=html,
        )
    )


def _add_methodology(
    report: ReportBuilder,
    results: dict,
    session_a_label: str,
    session_b_label: str,
) -> None:
    a_coeff = results.get("alignment_coefficients", {})
    report.add(
        TextSection(
            id="methodology",
            title="Methodology Notes",
            html=(
                "<p><strong>IRT Scale Alignment:</strong> IRT ideal points from each session "
                "are fitted independently, producing scores on different scales. To compare them, "
                "we use a robust affine transformation (xi_aligned = A &times; xi + B) fitted on "
                f"the {results.get('n_matched', '?')} returning legislators. The top/bottom "
                f"{ALIGNMENT_TRIM_PCT}% of residuals are trimmed before the final fit to prevent "
                "genuine movers from distorting the alignment.</p>"
                "<p><strong>Alignment coefficients:</strong> "
                + ", ".join(
                    f"{ch}: A={coefs['A']:.3f}, B={coefs['B']:.3f}"
                    for ch, coefs in sorted(a_coeff.items())
                )
                + ".</p>"
                if a_coeff
                else ""
                "<p><strong>Significant mover threshold:</strong> A legislator is flagged as a "
                f"significant mover if |shift| > {SHIFT_THRESHOLD_SD} &times; SD(all shifts). "
                "This adapts to the overall session-to-session variability.</p>"
                "<p><strong>Legislator matching:</strong> Legislators are matched by normalized "
                "full name (lowercased, leadership suffixes removed). Fuzzy matching is not used "
                "— only exact name matches are included.</p>"
                f"<p><strong>Reference session:</strong> {session_b_label} is the reference scale. "
                f"{session_a_label} ideal points are transformed onto that scale.</p>"
                "<p>See <code>analysis/design/cross_session.md</code> and "
                "<code>docs/adr/0019-cross-session-validation.md</code> for full details.</p>"
            ),
        )
    )
