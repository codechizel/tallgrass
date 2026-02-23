"""
Kansas Legislature — Cross-Session Validation

Compares two bienniums (default: 2023-24 and 2025-26) across three dimensions:
1. Ideology stability: IRT ideal point shifts for returning legislators
2. Metric consistency: Party loyalty, network influence, maverick rates
3. Detection validation: Do synthesis thresholds generalize across sessions?

Usage:
  uv run python analysis/cross_session.py
  uv run python analysis/cross_session.py --session-a 2023-24 --session-b 2025-26
  uv run python analysis/cross_session.py --chambers house

Outputs (in results/kansas/cross-session/validation/<date>/):
  - data/:   Parquet files (ideology_shift, metric_stability per chamber)
  - plots/:  PNG visualizations (shift scatter, movers, turnover, stability)
  - filtering_manifest.json, run_info.json, run_log.txt
  - validation_report.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

try:
    from analysis.cross_session_data import (
        CORRELATION_WARN,
        SHIFT_THRESHOLD_SD,
        align_irt_scales,
        classify_turnover,
        compute_ideology_shift,
        compute_metric_stability,
        compute_turnover_impact,
        match_legislators,
    )
except ModuleNotFoundError:
    from cross_session_data import (  # type: ignore[no-redef]
        CORRELATION_WARN,
        SHIFT_THRESHOLD_SD,
        align_irt_scales,
        classify_turnover,
        compute_ideology_shift,
        compute_metric_stability,
        compute_turnover_impact,
        match_legislators,
    )

try:
    from analysis.cross_session_report import build_cross_session_report
except ModuleNotFoundError:
    from cross_session_report import build_cross_session_report  # type: ignore[no-redef]

try:
    from analysis.run_context import RunContext
except ModuleNotFoundError:
    from run_context import RunContext  # type: ignore[no-redef]

try:
    from analysis.synthesis import build_legislator_df, load_all_upstream
except ModuleNotFoundError:
    from synthesis import build_legislator_df, load_all_upstream  # type: ignore[no-redef]

try:
    from analysis.synthesis_detect import (
        detect_bridge_builder,
        detect_chamber_maverick,
        detect_metric_paradox,
    )
except ModuleNotFoundError:
    from synthesis_detect import (  # type: ignore[no-redef]
        detect_bridge_builder,
        detect_chamber_maverick,
        detect_metric_paradox,
    )

# ── Constants ────────────────────────────────────────────────────────────────

PARTY_COLORS: dict[str, str] = {"Republican": "#E81B23", "Democrat": "#0015BC"}
TOP_MOVERS_N: int = 15
ANNOTATE_N: int = 5

# ── Primer ───────────────────────────────────────────────────────────────────

CROSS_SESSION_PRIMER = """\
# Cross-Session Validation

## Purpose

Compares two Kansas Legislature bienniums to answer three questions:

1. **Who moved ideologically?** IRT ideal points for returning legislators are
   placed on a common scale via affine transformation, then compared.
2. **Are our metrics stable?** Party loyalty, maverick rates, and network
   influence are correlated across sessions for returning legislators.
3. **Do our detection methods generalize?** Synthesis detection thresholds
   (maverick, bridge-builder, metric paradox) are run on both sessions.

## Method

IRT ideal points are fitted independently per session. To compare them, we use
a robust affine transformation fitted on the overlapping legislators, trimming
the most extreme residuals (genuine movers) before the final fit.

## Inputs

Reads from both sessions' `results/<session>/` directories:
- IRT ideal points (per chamber)
- Synthesis legislator DataFrames (all upstream phases joined)
- Raw legislator CSVs (for matching)

## Outputs

- `ideology_shift_{chamber}.parquet` — per-legislator shift metrics
- `metric_stability_{chamber}.parquet` — cross-session correlations
- `turnover_impact_{chamber}.json` — cohort distribution comparison
- `detection_validation.json` — detection threshold comparison
- `validation_report.html` — narrative HTML report

## Interpretation Guide

- **Ideology shift scatter:** Dots on the diagonal = no change. Dots above =
  moved rightward (more conservative). Dots below = moved leftward.
- **Significant movers:** Flagged when |shift| > 1 SD of all shifts.
- **Metric stability:** Pearson r > 0.7 = good stability. < 0.5 = weak.
- **Detection validation:** Same role flagged in both sessions = threshold
  generalizes. Different people in the same role = expected turnover.
"""


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Cross-Session Validation")
    parser.add_argument("--session-a", default="2023-24", help="Earlier session (default: 2023-24)")
    parser.add_argument("--session-b", default="2025-26", help="Later session (default: 2025-26)")
    parser.add_argument(
        "--chambers",
        default="both",
        choices=["house", "senate", "both"],
        help="Which chambers to analyze (default: both)",
    )
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────


def print_header(title: str) -> None:
    width = 80
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _majority_party(leg_df: pl.DataFrame) -> str | None:
    """Return the party with the most legislators."""
    counts = leg_df.group_by("party").len().sort("len", descending=True)
    if counts.height == 0:
        return None
    return counts["party"][0]


def _extract_name(full_name: str) -> str:
    """Extract last name for plot annotation."""
    parts = full_name.strip().split()
    return parts[-1] if parts else full_name


# ── Plot Functions ───────────────────────────────────────────────────────────


def plot_ideology_shift_scatter(
    shifted: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
    session_a_label: str,
    session_b_label: str,
) -> None:
    """Scatter: previous ideology vs current ideology, colored by party."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Diagonal (no change)
    all_xi = shifted.select("xi_a_aligned", "xi_b").to_numpy().flatten()
    lo, hi = float(np.min(all_xi)) - 0.3, float(np.max(all_xi)) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color="#999999", linewidth=1, zorder=1)

    # Points by party
    for party, color in PARTY_COLORS.items():
        subset = shifted.filter(pl.col("party") == party)
        if subset.height == 0:
            continue
        ax.scatter(
            subset["xi_a_aligned"].to_numpy(),
            subset["xi_b"].to_numpy(),
            c=color,
            label=party,
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidth=0.5,
            zorder=2,
        )

    # Annotate top movers
    top = shifted.sort("abs_delta_xi", descending=True).head(ANNOTATE_N)
    for row in top.iter_rows(named=True):
        name = _extract_name(row["full_name"])
        ax.annotate(
            name,
            (row["xi_a_aligned"], row["xi_b"]),
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
            xytext=(5, 5),
            textcoords="offset points",
            zorder=3,
        )

    ax.set_xlabel(f"Ideology — {session_a_label} (aligned)", fontsize=11)
    ax.set_ylabel(f"Ideology — {session_b_label}", fontsize=11)
    ax.set_title(f"{chamber}: Who Moved Between Sessions?", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    fig.tight_layout()
    save_fig(fig, plots_dir / f"ideology_shift_scatter_{chamber.lower()}.png")


def plot_biggest_movers(
    shifted: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Horizontal bar chart of top N biggest movers by |delta_xi|."""
    top = shifted.sort("abs_delta_xi", descending=True).head(TOP_MOVERS_N)
    if top.height == 0:
        return

    names = [_extract_name(n) for n in top["full_name"].to_list()]
    deltas = top["delta_xi"].to_numpy()
    colors = ["#E81B23" if d > 0 else "#0015BC" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, deltas, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Ideology Shift (rightward →, ← leftward)", fontsize=10)
    ax.set_title(
        f"{chamber}: Top {top.height} Biggest Ideological Movers",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()
    save_fig(fig, plots_dir / f"biggest_movers_{chamber.lower()}.png")


def plot_shift_distribution(
    shifted: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Histogram of ideology shifts with threshold lines."""
    deltas = shifted["delta_xi"].to_numpy()
    std = float(np.std(deltas))
    threshold = SHIFT_THRESHOLD_SD * std

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(deltas, bins=25, color="#666666", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="#333333", linewidth=1, linestyle="-")
    ax.axvline(threshold, color="#E81B23", linewidth=1.5, linestyle="--", label="Significant mover")
    ax.axvline(-threshold, color="#0015BC", linewidth=1.5, linestyle="--")

    n_movers = int(np.sum(np.abs(deltas) > threshold))
    ax.set_xlabel("Ideology Shift", fontsize=11)
    ax.set_ylabel("Number of Legislators", fontsize=11)
    ax.set_title(
        f"{chamber}: Distribution of Ideology Shifts ({n_movers} significant movers)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    save_fig(fig, plots_dir / f"shift_distribution_{chamber.lower()}.png")


def plot_turnover_impact(
    xi_returning: np.ndarray,
    xi_departing: np.ndarray,
    xi_new: np.ndarray,
    chamber: str,
    plots_dir: Path,
    session_a_label: str,
    session_b_label: str,
) -> None:
    """Strip plot comparing ideology distributions by turnover cohort."""
    fig, ax = plt.subplots(figsize=(8, 4))

    cohorts = [
        (f"Departing\n(left after {session_a_label})", xi_departing, "#999999"),
        ("Returning", xi_returning, "#555555"),
        (f"New\n(joined in {session_b_label})", xi_new, "#333333"),
    ]

    for i, (label, data, color) in enumerate(cohorts):
        if len(data) == 0:
            continue
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.15, 0.15, size=len(data))
        ax.scatter(
            data,
            np.full_like(data, i) + jitter,
            c=color,
            alpha=0.6,
            s=20,
            edgecolors="white",
            linewidth=0.3,
        )
        ax.plot(
            [np.mean(data)],
            [i],
            marker="D",
            color=color,
            markersize=10,
            zorder=5,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

    ax.set_yticks(range(len(cohorts)))
    ax.set_yticklabels([c[0] for c in cohorts], fontsize=10)
    ax.set_xlabel("IRT Ideology (liberal ← → conservative)", fontsize=11)
    ax.set_title(
        f"{chamber}: Who Left and Who Replaced Them?",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()
    save_fig(fig, plots_dir / f"turnover_impact_{chamber.lower()}.png")


# ── Detection Validation ────────────────────────────────────────────────────


def validate_detection(
    leg_df_a: pl.DataFrame,
    leg_df_b: pl.DataFrame,
    chamber: str,
) -> dict:
    """Run synthesis detection on both sessions, compare results."""
    result: dict = {}

    majority_a = _majority_party(leg_df_a)
    majority_b = _majority_party(leg_df_b)

    # Maverick
    mav_a = detect_chamber_maverick(leg_df_a, majority_a, chamber) if majority_a else None
    mav_b = detect_chamber_maverick(leg_df_b, majority_b, chamber) if majority_b else None
    result["maverick_a"] = mav_a.full_name if mav_a else None
    result["maverick_b"] = mav_b.full_name if mav_b else None

    # Bridge-builder
    bridge_a = detect_bridge_builder(leg_df_a, chamber)
    bridge_b = detect_bridge_builder(leg_df_b, chamber)
    result["bridge_a"] = bridge_a.full_name if bridge_a else None
    result["bridge_b"] = bridge_b.full_name if bridge_b else None

    # Paradox
    paradox_a = detect_metric_paradox(leg_df_a, chamber)
    paradox_b = detect_metric_paradox(leg_df_b, chamber)
    result["paradox_a"] = paradox_a.full_name if paradox_a else None
    result["paradox_b"] = paradox_b.full_name if paradox_b else None

    return result


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from ks_vote_scraper.session import KSSession

    ks_a = KSSession.from_session_string(args.session_a)
    ks_b = KSSession.from_session_string(args.session_b)
    session_a_label = ks_a.output_name
    session_b_label = ks_b.output_name

    chambers = ["house", "senate"] if args.chambers == "both" else [args.chambers]

    with RunContext(
        session="cross-session",
        analysis_name="validation",
        params=vars(args),
        primer=CROSS_SESSION_PRIMER,
    ) as ctx:
        print_header("Cross-Session Validation")
        print(f"  Session A: {session_a_label}")
        print(f"  Session B: {session_b_label}")

        # ── Load raw legislator CSVs ──
        print("\n── Loading legislator data ──")
        leg_a = pl.read_csv(ks_a.data_dir / f"{ks_a.output_name}_legislators.csv")
        leg_b = pl.read_csv(ks_b.data_dir / f"{ks_b.output_name}_legislators.csv")
        print(f"  Session A: {leg_a.height} legislators")
        print(f"  Session B: {leg_b.height} legislators")

        # ── Match legislators ──
        print("\n── Matching legislators ──")
        matched = match_legislators(leg_a, leg_b)
        turnover = classify_turnover(leg_a, leg_b, matched)
        n_departing = turnover["departing"].height
        n_new = turnover["new"].height
        n_chamber_switch = int(matched["is_chamber_switch"].sum())
        n_party_switch = int(matched["is_party_switch"].sum())
        print(f"  Matched: {matched.height}")
        print(f"  Departing: {n_departing}")
        print(f"  New: {n_new}")
        if n_chamber_switch:
            print(f"  Chamber switches: {n_chamber_switch}")
        if n_party_switch:
            print(f"  Party switches: {n_party_switch}")

        # ── Load upstream results ──
        print("\n── Loading upstream analysis results ──")
        upstream_a = load_all_upstream(ks_a.results_dir)
        upstream_b = load_all_upstream(ks_b.results_dir)

        # ── Per-chamber analysis ──
        all_results: dict = {
            "matched": matched,
            "n_departing": n_departing,
            "n_new": n_new,
            "n_matched": matched.height,
            "chambers": chambers,
            "alignment_coefficients": {},
        }

        for chamber in chambers:
            chamber_cap = chamber.capitalize()
            print_header(f"{chamber_cap} Analysis")

            # Build legislator DataFrames
            leg_df_a = build_legislator_df(upstream_a, chamber)
            leg_df_b = build_legislator_df(upstream_b, chamber)
            print(f"  Legislator DF A: {leg_df_a.height} rows, {leg_df_a.width} cols")
            print(f"  Legislator DF B: {leg_df_b.height} rows, {leg_df_b.width} cols")

            # IRT ideal points
            irt_a = upstream_a[chamber].get("irt")
            irt_b = upstream_b[chamber].get("irt")
            if irt_a is None or irt_b is None:
                print(f"  WARNING: Missing IRT data for {chamber}, skipping")
                continue

            # ── Align IRT scales ──
            print("\n  Aligning IRT scales...")
            a_coef, b_coef, aligned = align_irt_scales(irt_a, irt_b, matched)
            print(f"    A = {a_coef:.4f}, B = {b_coef:.4f}")
            all_results["alignment_coefficients"][chamber_cap] = {"A": a_coef, "B": b_coef}

            # ── Ideology shift ──
            print("  Computing ideology shift...")
            shifted = compute_ideology_shift(aligned)
            n_movers = int(shifted["is_significant_mover"].sum())
            print(f"    {n_movers} significant movers out of {shifted.height}")

            # Correlation check
            from scipy import stats as sp_stats

            xi_a_arr = shifted["xi_a_aligned"].to_numpy()
            xi_b_arr = shifted["xi_b"].to_numpy()
            r_val, _ = sp_stats.pearsonr(xi_a_arr, xi_b_arr)
            print(f"    Cross-session ideology correlation: r = {r_val:.3f}")
            if r_val < CORRELATION_WARN:
                print(f"    WARNING: r < {CORRELATION_WARN} — alignment may be unreliable")

            # ── Metric stability ──
            print("  Computing metric stability...")
            stability = compute_metric_stability(leg_df_a, leg_df_b, matched)
            for row in stability.iter_rows(named=True):
                flag = " ⚠" if row["pearson_r"] < CORRELATION_WARN else ""
                m, pr, sr = row["metric"], row["pearson_r"], row["spearman_rho"]
                print(f"    {m:20s}  r={pr:.3f}  ρ={sr:.3f}{flag}")

            # ── Turnover impact ──
            print("  Computing turnover impact...")
            chamber_matched = matched.filter(pl.col("chamber_b").str.to_lowercase() == chamber)
            dep_slugs = set(
                turnover["departing"]
                .filter(pl.col("chamber").str.to_lowercase() == chamber)["legislator_slug"]
                .to_list()
            )
            new_slugs = set(
                turnover["new"]
                .filter(pl.col("chamber").str.to_lowercase() == chamber)["legislator_slug"]
                .to_list()
            )

            # Get xi values for each cohort from the raw IRT DataFrames
            ret_slugs = set(chamber_matched["slug_b"].to_list())
            xi_ret = irt_b.filter(pl.col("legislator_slug").is_in(ret_slugs))["xi_mean"].to_numpy()
            xi_dep = irt_a.filter(pl.col("legislator_slug").is_in(dep_slugs))["xi_mean"].to_numpy()
            xi_new = irt_b.filter(pl.col("legislator_slug").is_in(new_slugs))["xi_mean"].to_numpy()

            ti = compute_turnover_impact(xi_ret, xi_dep, xi_new)
            turnover_impact = ti
            print(f"    Returning: n={ti['returning_n']}, mean={ti['returning_mean']:.2f}")
            if ti["departing_n"] > 0 and ti["departing_mean"] is not None:
                print(f"    Departing: n={ti['departing_n']}, mean={ti['departing_mean']:.2f}")
            if ti["new_n"] > 0 and ti["new_mean"] is not None:
                print(f"    New:       n={ti['new_n']}, mean={ti['new_mean']:.2f}")

            # ── Detection validation ──
            print("  Validating detection thresholds...")
            detection = validate_detection(leg_df_a, leg_df_b, chamber)
            for role in ["maverick", "bridge", "paradox"]:
                na = detection.get(f"{role}_a", "—")
                nb = detection.get(f"{role}_b", "—")
                same = "✓" if (na and nb and na == nb) else ""
                print(f"    {role:8s}  A: {na or '—':25s}  B: {nb or '—':25s}  {same}")

            # ── Plots ──
            print("  Generating plots...")
            plot_ideology_shift_scatter(
                shifted,
                chamber_cap,
                ctx.plots_dir,
                session_a_label,
                session_b_label,
            )
            plot_biggest_movers(shifted, chamber_cap, ctx.plots_dir)
            plot_shift_distribution(shifted, chamber_cap, ctx.plots_dir)
            plot_turnover_impact(
                xi_ret,
                xi_dep,
                xi_new,
                chamber_cap,
                ctx.plots_dir,
                session_a_label,
                session_b_label,
            )

            # ── Save data ──
            shifted.write_parquet(ctx.data_dir / f"ideology_shift_{chamber}.parquet")
            stability.write_parquet(ctx.data_dir / f"metric_stability_{chamber}.parquet")
            with open(ctx.data_dir / f"turnover_impact_{chamber}.json", "w") as f:
                json.dump(turnover_impact, f, indent=2)

            all_results[chamber] = {
                "shifted": shifted,
                "stability": stability,
                "turnover": turnover_impact,
                "detection": detection,
                "r_value": r_val,
            }

        # ── Save detection results ──
        detection_summary = {
            ch: all_results[ch]["detection"]
            for ch in chambers
            if ch in all_results and "detection" in all_results[ch]
        }
        with open(ctx.data_dir / "detection_validation.json", "w") as f:
            json.dump(detection_summary, f, indent=2)

        # ── Build report ──
        print_header("Building Report")
        build_cross_session_report(
            ctx.report,
            results=all_results,
            plots_dir=ctx.plots_dir,
            session_a_label=session_a_label,
            session_b_label=session_b_label,
        )

        print("\n  Done.")


if __name__ == "__main__":
    main()
