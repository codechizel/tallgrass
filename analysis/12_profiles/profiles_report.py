"""Legislator profiles report builder — per-legislator deep-dive sections.

Assembles scorecard, bill-type breakdown, defection analysis, voting neighbors,
and surprising votes into a self-contained HTML report. Called by profiles.py.
"""

from pathlib import Path

import polars as pl

try:
    from analysis.profiles_data import ProfileTarget
except ModuleNotFoundError:
    from profiles_data import ProfileTarget  # type: ignore[no-redef]

try:
    from analysis.report import FigureSection, TableSection, TextSection, make_gt
except ModuleNotFoundError:
    from report import FigureSection, TableSection, TextSection, make_gt  # type: ignore[no-redef]


def build_profiles_report(
    report: object,
    *,
    targets: list[ProfileTarget],
    all_data: dict[str, dict],
    plots_dir: Path,
    session: str,
) -> None:
    """Build the full profiles report with intro + per-legislator sections."""
    _add_intro(report, targets, session)

    for target in targets:
        data = all_data.get(target.slug, {})
        slug_short = target.slug.replace("rep_", "").replace("sen_", "")

        _add_target_header(report, target)
        _add_scorecard_figure(report, target, slug_short, plots_dir)
        _add_bill_type_figure(report, target, slug_short, plots_dir)
        _add_position_figure(report, target, slug_short, plots_dir)
        _add_defections_table(report, target, data.get("defections"))
        _add_surprising_votes_table(report, target, data.get("surprising"))
        _add_neighbors_figure(report, target, slug_short, plots_dir)

    print(f"  Report: {len(report._sections)} sections added")


# ── Intro ────────────────────────────────────────────────────────────────────


def _add_intro(report: object, targets: list[ProfileTarget], session: str) -> None:
    """Opening section explaining what this report is."""
    n = len(targets)
    role_list = []
    for t in targets:
        role_list.append(f"<strong>{t.full_name}</strong> ({t.role})")
    names_html = ", ".join(role_list)

    report.add(
        TextSection(
            id="profiles-intro",
            title="Legislator Profiles — Deep Dives",
            html=(
                f"<p>This report profiles <strong>{n} legislators</strong> who stand "
                "out statistically from the Kansas Legislature's "
                f"{session} session. Each profile combines findings from eight "
                "analysis phases into a single deep-dive.</p>"
                f"<p>Profiled legislators: {names_html}.</p>"
                "<p>Flagged legislators were detected automatically using data-driven "
                "thresholds — different sessions will surface different individuals. "
                "Each profile includes a scorecard, bill-type voting breakdown, "
                "key defection votes, prediction surprises, and voting neighbors.</p>"
            ),
        )
    )


# ── Per-Legislator Sections ─────────────────────────────────────────────────


def _add_target_header(report: object, target: ProfileTarget) -> None:
    """Section header for one legislator with role and narrative."""
    slug_short = target.slug.replace("rep_", "").replace("sen_", "")
    chamber = target.chamber.title()

    report.add(
        TextSection(
            id=f"header-{slug_short}",
            title=f"{target.title} — {target.role}",
            html=(
                f"<p><strong>{chamber}</strong> &middot; "
                f"{target.party} &middot; District {target.district}</p>"
                f"<p><em>{target.subtitle}</em></p>"
            ),
        )
    )


def _add_scorecard_figure(
    report: object, target: ProfileTarget, slug_short: str, plots_dir: Path
) -> None:
    """Enhanced scorecard bar chart."""
    path = plots_dir / f"scorecard_{slug_short}.png"
    if not path.exists():
        return
    report.add(
        FigureSection.from_file(
            f"scorecard-{slug_short}",
            f"{target.full_name} — At a Glance",
            path,
            caption=(
                "Horizontal bars show this legislator's metric values (colored) "
                "alongside their party average (gray dashed line). Metrics include "
                "IRT ideology, party unity, loyalty, maverick rate, network centrality, "
                "and prediction accuracy."
            ),
        )
    )


def _add_bill_type_figure(
    report: object, target: ProfileTarget, slug_short: str, plots_dir: Path
) -> None:
    """Bill type breakdown grouped bar chart."""
    path = plots_dir / f"bill_type_{slug_short}.png"
    if not path.exists():
        return
    report.add(
        FigureSection.from_file(
            f"bill-type-{slug_short}",
            f"How {target.full_name} Votes by Bill Type",
            path,
            caption=(
                "Yea rates on high-discrimination bills (partisan, contested) vs "
                "low-discrimination bills (routine, bipartisan). The gap between a "
                "legislator and their party average on partisan bills reveals how "
                "much they break ranks on the votes that matter most."
            ),
        )
    )


def _add_position_figure(
    report: object, target: ProfileTarget, slug_short: str, plots_dir: Path
) -> None:
    """Forest-style position plot among same-party members."""
    path = plots_dir / f"position_{slug_short}.png"
    if not path.exists():
        return
    report.add(
        FigureSection.from_file(
            f"position-{slug_short}",
            f"Where {target.full_name} Stands Among {target.party}s",
            path,
            caption=(
                "IRT ideal point estimates for all same-party members in the same "
                "chamber. The profiled legislator is highlighted with a diamond "
                "marker and yellow background. Horizontal lines show 95% credible "
                "intervals (uncertainty)."
            ),
        )
    )


def _add_defections_table(
    report: object, target: ProfileTarget, defections: pl.DataFrame | None
) -> None:
    """Table of key votes where this legislator broke ranks."""
    if defections is None or defections.height == 0:
        return

    slug_short = target.slug.replace("rep_", "").replace("sen_", "")

    display = defections.select(
        pl.col("bill_number").alias("Bill"),
        pl.col("short_title").alias("Title"),
        pl.col("motion").alias("Motion"),
        pl.col("legislator_vote").alias("Their Vote"),
        pl.col("party_majority_vote").alias("Party Majority"),
        pl.col("party_yea_pct").alias("Party Yea %"),
    )

    html = make_gt(
        display,
        title=f"Key Votes Where {target.full_name} Broke Ranks",
        subtitle="Sorted by closeness of the party margin (tightest first)",
        number_formats={"Party Yea %": ".1f"},
        source_note=(
            "A 'defection' is any vote where this legislator disagreed with their "
            "party's majority. Party Yea % shows what fraction of the party voted Yea."
        ),
    )

    report.add(
        TableSection(
            id=f"defections-{slug_short}",
            title=f"Defection Votes — {target.full_name}",
            html=html,
        )
    )


def _add_surprising_votes_table(
    report: object, target: ProfileTarget, surprising: pl.DataFrame | None
) -> None:
    """Table of votes where the prediction model was most wrong about this legislator."""
    if surprising is None or surprising.height == 0:
        return

    slug_short = target.slug.replace("rep_", "").replace("sen_", "")

    display = surprising.select(
        pl.col("bill_number").alias("Bill"),
        pl.col("motion").alias("Motion"),
        pl.when(pl.col("actual") == 1)
        .then(pl.lit("Yea"))
        .otherwise(pl.lit("Nay"))
        .alias("Actual Vote"),
        pl.when(pl.col("predicted") == 1)
        .then(pl.lit("Yea"))
        .otherwise(pl.lit("Nay"))
        .alias("Predicted"),
        (pl.col("y_prob") * 100).round(1).alias("Model Confidence (%)"),
        (pl.col("confidence_error") * 100).round(1).alias("Surprise Score"),
    )

    html = make_gt(
        display,
        title=f"Most Surprising Votes — {target.full_name}",
        subtitle="Votes where the prediction model was most confident and most wrong",
        number_formats={"Model Confidence (%)": ".1f", "Surprise Score": ".1f"},
        source_note=(
            "The Surprise Score is how wrong the model was (|confidence - actual|). "
            "Higher = more surprising. These are the votes where this legislator's "
            "behavior defied the patterns learned from the full legislature."
        ),
    )

    report.add(
        TableSection(
            id=f"surprising-{slug_short}",
            title=f"Surprising Votes — {target.full_name}",
            html=html,
        )
    )


def _add_neighbors_figure(
    report: object, target: ProfileTarget, slug_short: str, plots_dir: Path
) -> None:
    """Voting neighbors bar chart."""
    path = plots_dir / f"neighbors_{slug_short}.png"
    if not path.exists():
        return
    report.add(
        FigureSection.from_file(
            f"neighbors-{slug_short}",
            f"Who Does {target.full_name} Vote Like?",
            path,
            caption=(
                "Top 5 most similar legislators (highest agreement) and top 5 most "
                "different (lowest agreement) by simple vote-matching rate across "
                "all shared votes. Same-chamber only."
            ),
        )
    )
