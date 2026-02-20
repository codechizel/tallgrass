"""Clustering-specific HTML report builder.

Builds ~19 sections (tables + figures) for the clustering analysis report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from clustering.py):
    from analysis.clustering_report import build_clustering_report
    build_clustering_report(ctx.report, results=results, ...)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

try:
    from analysis.report import FigureSection, ReportBuilder, TableSection, make_gt
except ModuleNotFoundError:
    from report import FigureSection, ReportBuilder, TableSection, make_gt


def build_clustering_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    plots_dir: Path,
    skip_gmm: bool = False,
    skip_sensitivity: bool = False,
) -> None:
    """Build the full clustering HTML report by adding ~19 sections to the ReportBuilder."""
    _add_data_summary(report, results)

    for chamber, result in results.items():
        _add_party_loyalty_table(report, result, chamber)

    for chamber in results:
        _add_dendrogram_figure(report, plots_dir, chamber)

    for chamber in results:
        _add_model_selection_figure(report, plots_dir, chamber)

    if not skip_gmm:
        for chamber in results:
            _add_gmm_model_selection_figure(report, plots_dir, chamber)

    for chamber, result in results.items():
        _add_cluster_assignments_table(report, result, chamber)

    for chamber in results:
        _add_irt_clusters_figure(report, plots_dir, chamber)

    for chamber in results:
        _add_irt_loyalty_figure(report, plots_dir, chamber)

    for chamber, result in results.items():
        _add_cluster_composition_table(report, result, chamber)

    _add_cross_method_agreement(report, results)

    _add_flagged_legislators(report, results)

    for chamber in results:
        _add_cluster_box_figure(report, plots_dir, chamber)

    _add_veto_override_table(report, results)

    if not skip_sensitivity:
        _add_sensitivity_table(report, results)

    _add_analysis_parameters(report)
    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_data_summary(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Data dimensions and upstream sources per chamber."""
    rows = []
    for chamber, result in results.items():
        ip = result["ideal_points"]
        vm = result["vote_matrix"]
        rows.append({
            "Chamber": chamber,
            "N Legislators": ip.height,
            "N Votes (filtered)": len(vm.columns) - 1,
            "IRT Source": "irt/latest",
            "Kappa Source": "eda/latest",
        })

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Clustering Data Summary",
        subtitle="Upstream data dimensions per chamber",
    )
    report.add(TableSection(id="data-summary", title="Data Summary", html=html))


def _add_party_loyalty_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Top and bottom 15 legislators by party loyalty rate."""
    loyalty = result.get("loyalty")
    if loyalty is None or loyalty.height == 0:
        return

    sorted_loy = loyalty.sort("loyalty_rate")

    # Bottom 15 (least loyal)
    bottom = sorted_loy.head(15)
    # Top 15 (most loyal)
    top = sorted_loy.tail(15).sort("loyalty_rate", descending=True)
    combined = pl.concat([bottom, top])

    display_cols = ["full_name", "party", "loyalty_rate", "n_contested_votes", "n_agree"]
    available = [c for c in display_cols if c in combined.columns]
    df = combined.select(available)

    html = make_gt(
        df,
        title=f"{chamber} — Party Loyalty (Top/Bottom 15)",
        subtitle="Fraction of contested votes agreeing with party median",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "loyalty_rate": "Loyalty Rate",
            "n_contested_votes": "Contested Votes",
            "n_agree": "Agreed",
        },
        number_formats={"loyalty_rate": ".3f"},
        source_note=f"Contested: >= {10}% of party dissents on that vote.",
    )
    report.add(
        TableSection(
            id=f"party-loyalty-{chamber.lower()}",
            title=f"{chamber} Party Loyalty",
            html=html,
        )
    )


def _add_dendrogram_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"dendrogram_{chamber.lower()}.png"
    if path.exists():
        truncated = " (truncated)" if chamber == "House" else ""
        report.add(
            FigureSection.from_file(
                f"fig-dendrogram-{chamber.lower()}",
                f"{chamber} Dendrogram{truncated}",
                path,
                caption=(
                    f"Hierarchical clustering dendrogram ({chamber}) using Ward linkage "
                    "on Kappa distance. Leaf labels colored by party."
                ),
            )
        )


def _add_model_selection_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"model_selection_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-model-sel-{chamber.lower()}",
                f"{chamber} K-Means Model Selection",
                path,
                caption=(
                    f"K-Means model selection ({chamber}): inertia (elbow) and "
                    "silhouette score vs number of clusters."
                ),
            )
        )


def _add_gmm_model_selection_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"bic_aic_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-gmm-sel-{chamber.lower()}",
                f"{chamber} GMM Model Selection",
                path,
                caption=(
                    f"GMM model selection ({chamber}): BIC and AIC vs number of components. "
                    "Lower BIC = better model."
                ),
            )
        )


def _add_cluster_assignments_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: All legislators with cluster assignment, IRT, and loyalty."""
    ip = result["ideal_points"]
    km = result.get("kmeans", {})
    labels = km.get("labels")
    if labels is None:
        return

    loyalty = result.get("loyalty")

    df = ip.select("full_name", "party", "district", "xi_mean", "xi_sd").with_columns(
        pl.Series("cluster", labels.tolist())
    )

    if loyalty is not None and loyalty.height > 0:
        df = df.with_columns(
            pl.Series(
                "legislator_slug",
                ip["legislator_slug"].to_list(),
            )
        ).join(
            loyalty.select("legislator_slug", "loyalty_rate"),
            on="legislator_slug",
            how="left",
        ).drop("legislator_slug")

    df = df.sort("cluster", "xi_mean", descending=[False, True])

    labels_dict: dict[str, str] = {
        "full_name": "Legislator",
        "party": "Party",
        "district": "District",
        "xi_mean": "Ideal Point",
        "xi_sd": "Std Dev",
        "cluster": "Cluster",
    }
    formats: dict[str, str] = {
        "xi_mean": ".3f",
        "xi_sd": ".3f",
    }
    if "loyalty_rate" in df.columns:
        labels_dict["loyalty_rate"] = "Loyalty"
        formats["loyalty_rate"] = ".3f"

    html = make_gt(
        df,
        title=f"{chamber} — Cluster Assignments (K-Means)",
        subtitle=f"{df.height} legislators, k={km.get('optimal_k', '?')}",
        column_labels=labels_dict,
        number_formats=formats,
    )
    report.add(
        TableSection(
            id=f"cluster-assign-{chamber.lower()}",
            title=f"{chamber} Cluster Assignments",
            html=html,
        )
    )


def _add_irt_clusters_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"irt_clusters_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-irt-clusters-{chamber.lower()}",
                f"{chamber} IRT Clusters",
                path,
                caption=(
                    f"IRT ideal points colored by K-Means cluster ({chamber}). "
                    "Circles = Republican, Squares = Democrat."
                ),
            )
        )


def _add_irt_loyalty_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"irt_loyalty_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-irt-loyalty-{chamber.lower()}",
                f"{chamber} Ideology vs Party Loyalty",
                path,
                caption=(
                    f"2D view ({chamber}): IRT ideal point (x) vs party loyalty (y). "
                    "Mavericks appear in the low-loyalty region with extreme ideology."
                ),
            )
        )


def _add_cluster_composition_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Party breakdown per cluster."""
    summary = result.get("cluster_summary")
    if summary is None:
        return

    display_cols = [
        "cluster", "label", "n_legislators", "n_republican", "n_democrat",
        "pct_republican", "xi_mean", "xi_median", "avg_xi_sd",
    ]
    if "avg_loyalty" in summary.columns:
        display_cols.append("avg_loyalty")
    available = [c for c in display_cols if c in summary.columns]

    labels_dict: dict[str, str] = {
        "cluster": "Cluster",
        "label": "Label",
        "n_legislators": "N",
        "n_republican": "Republican",
        "n_democrat": "Democrat",
        "pct_republican": "% Republican",
        "xi_mean": "Mean Ideal Pt",
        "xi_median": "Median Ideal Pt",
        "avg_xi_sd": "Avg Std Dev",
        "avg_loyalty": "Avg Loyalty",
    }
    formats: dict[str, str] = {
        "pct_republican": ".1f",
        "xi_mean": ".3f",
        "xi_median": ".3f",
        "avg_xi_sd": ".3f",
        "avg_loyalty": ".3f",
    }

    html = make_gt(
        summary.select(available),
        title=f"{chamber} — Cluster Composition",
        subtitle="Party composition and IRT statistics per cluster",
        column_labels={k: v for k, v in labels_dict.items() if k in available},
        number_formats={k: v for k, v in formats.items() if k in available},
    )
    report.add(
        TableSection(
            id=f"cluster-comp-{chamber.lower()}",
            title=f"{chamber} Cluster Composition",
            html=html,
        )
    )


def _add_cross_method_agreement(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: ARI matrix between clustering methods."""
    rows = []
    for chamber, result in results.items():
        comparison = result.get("comparison", {})
        ari_matrix = comparison.get("ari_matrix", {})
        for pair, ari_val in ari_matrix.items():
            methods = pair.split("_vs_")
            rows.append({
                "Chamber": chamber,
                "Method A": methods[0] if len(methods) > 0 else pair,
                "Method B": methods[1] if len(methods) > 1 else "",
                "ARI": ari_val,
            })
        if comparison.get("mean_ari") is not None:
            rows.append({
                "Chamber": chamber,
                "Method A": "all methods",
                "Method B": "mean ARI",
                "ARI": comparison["mean_ari"],
            })

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Cross-Method Agreement",
        subtitle="Adjusted Rand Index between clustering methods (1.0 = identical)",
        column_labels={
            "Chamber": "Chamber",
            "Method A": "Method A",
            "Method B": "Method B",
            "ARI": "ARI / Rate",
        },
        number_formats={"ARI": ".4f"},
        source_note=(
            "ARI > 0.7 = strong agreement. "
            "Stability = fraction of legislators in same cluster across all methods."
        ),
    )
    report.add(
        TableSection(
            id="cross-method-ari",
            title="Cross-Method Agreement",
            html=html,
        )
    )


def _add_flagged_legislators(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Flagged legislators with assignments and notes."""
    rows = []
    for chamber, result in results.items():
        for entry in result.get("flagged_legislators", []):
            note = ""
            slug = entry["legislator_slug"]
            if "tyson" in slug:
                note = "Extreme IRT from contrarian pattern; check loyalty rate"
            elif "thompson" in slug:
                note = "Milder version of Tyson contrarian pattern"
            elif "miller" in slug:
                note = "Sparse data (30/194 votes); low-confidence cluster"
            elif "hill" in slug:
                note = "Widest HDI in Senate; lowest-confidence cluster"

            loy_val = entry.get("loyalty_rate")
            rows.append({
                "Chamber": chamber,
                "Legislator": entry["full_name"],
                "Party": entry["party"],
                "Ideal Point": entry["xi_mean"],
                "Std Dev": entry["xi_sd"],
                "Cluster": entry["cluster"],
                "Loyalty": loy_val,
                "Note": note,
            })

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Flagged Legislators — Cluster Assignments",
        subtitle="Legislators flagged in prior phases with their clustering results",
        column_labels={
            "Chamber": "Chamber",
            "Legislator": "Legislator",
            "Party": "Party",
            "Ideal Point": "Ideal Pt",
            "Std Dev": "SD",
            "Cluster": "Cluster",
            "Loyalty": "Loyalty",
            "Note": "Note",
        },
        number_formats={
            "Ideal Point": ".3f",
            "Std Dev": ".3f",
            "Loyalty": ".3f",
        },
    )
    report.add(
        TableSection(
            id="flagged-legislators",
            title="Flagged Legislators",
            html=html,
        )
    )


def _add_cluster_box_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"cluster_box_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-cluster-box-{chamber.lower()}",
                f"{chamber} Cluster Boxplot",
                path,
                caption=(
                    f"Distribution of IRT ideal points per cluster ({chamber}). "
                    "Box = IQR, whiskers = 1.5x IQR, dots = outliers."
                ),
            )
        )


def _add_veto_override_table(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Veto override cluster statistics."""
    rows = []
    for chamber, result in results.items():
        veto = result.get("veto_overrides", {})
        if veto.get("skipped"):
            rows.append({
                "Chamber": chamber,
                "N Override Votes": veto.get("n_override_votes", 0),
                "Cluster": "N/A",
                "Mean Override Yea": None,
                "N Legislators": None,
                "Note": "Insufficient override votes for analysis",
            })
            continue

        cluster_stats = veto.get("cluster_stats")
        if cluster_stats is None:
            continue

        for row in cluster_stats.iter_rows(named=True):
            rows.append({
                "Chamber": chamber,
                "N Override Votes": veto["n_override_votes"],
                "Cluster": row["full_cluster"],
                "Mean Override Yea": row["mean_override_yea_rate"],
                "N Legislators": row["n_legislators"],
                "Note": "",
            })

        rows.append({
            "Chamber": chamber,
            "N Override Votes": veto["n_override_votes"],
            "Cluster": "High Yea (>70%)",
            "Mean Override Yea": None,
            "N Legislators": veto.get("n_high_yea_r", 0) + veto.get("n_high_yea_d", 0),
            "Note": f"{veto.get('n_high_yea_r', 0)}R, {veto.get('n_high_yea_d', 0)}D",
        })

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Veto Override Voting Patterns by Cluster",
        subtitle="Mean Yea rate on veto override votes per full-dataset cluster",
        column_labels={
            "Chamber": "Chamber",
            "N Override Votes": "Override Votes",
            "Cluster": "Cluster",
            "Mean Override Yea": "Mean Yea Rate",
            "N Legislators": "N Legislators",
            "Note": "Note",
        },
        number_formats={"Mean Override Yea": ".3f"},
        source_note="Veto overrides require 2/3 supermajority, revealing cross-party coalitions.",
    )
    report.add(
        TableSection(
            id="veto-overrides",
            title="Veto Override Clusters",
            html=html,
        )
    )


def _add_sensitivity_table(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Sensitivity analysis — ARI across k variations."""
    rows = []
    for chamber, result in results.items():
        sensitivity = result.get("sensitivity", {})
        for key, data in sensitivity.items():
            if not isinstance(data, dict):
                continue
            rows.append({
                "Chamber": chamber,
                "Comparison": key,
                "ARI": data.get("ari", data.get("ari", None)),
            })

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Sensitivity Analysis — Cluster Stability",
        subtitle="ARI between default and alternative k values / methods",
        column_labels={
            "Chamber": "Chamber",
            "Comparison": "Comparison",
            "ARI": "ARI",
        },
        number_formats={"ARI": ".4f"},
        source_note="ARI > 0.7 indicates robust cluster structure across k choices.",
    )
    report.add(
        TableSection(
            id="sensitivity",
            title="Sensitivity Analysis",
            html=html,
        )
    )


def _add_analysis_parameters(report: ReportBuilder) -> None:
    """Table: All constants and settings used in this run."""
    try:
        from analysis.clustering import (
            CLUSTER_CMAP,
            CONTESTED_PARTY_THRESHOLD,
            COPHENETIC_THRESHOLD,
            DEFAULT_K,
            GMM_COVARIANCE,
            GMM_N_INIT,
            K_RANGE,
            LINKAGE_METHOD,
            MIN_VOTES,
            MINORITY_THRESHOLD,
            RANDOM_SEED,
            SENSITIVITY_THRESHOLD,
            SILHOUETTE_GOOD,
        )
    except ModuleNotFoundError:
        from clustering import (  # type: ignore[no-redef]
            CLUSTER_CMAP,
            CONTESTED_PARTY_THRESHOLD,
            COPHENETIC_THRESHOLD,
            DEFAULT_K,
            GMM_COVARIANCE,
            GMM_N_INIT,
            K_RANGE,
            LINKAGE_METHOD,
            MIN_VOTES,
            MINORITY_THRESHOLD,
            RANDOM_SEED,
            SENSITIVITY_THRESHOLD,
            SILHOUETTE_GOOD,
        )

    df = pl.DataFrame({
        "Parameter": [
            "Random Seed",
            "K Range",
            "Default K",
            "Linkage Method",
            "Cophenetic Threshold",
            "Silhouette 'Good' Threshold",
            "GMM Covariance Type",
            "GMM N Initializations",
            "Cluster Colormap",
            "Minority Threshold (Default)",
            "Minority Threshold (Sensitivity)",
            "Min Substantive Votes",
            "Contested Party Threshold",
        ],
        "Value": [
            str(RANDOM_SEED),
            str(list(K_RANGE)),
            str(DEFAULT_K),
            LINKAGE_METHOD,
            str(COPHENETIC_THRESHOLD),
            str(SILHOUETTE_GOOD),
            GMM_COVARIANCE,
            str(GMM_N_INIT),
            CLUSTER_CMAP,
            f"{MINORITY_THRESHOLD:.3f} ({MINORITY_THRESHOLD * 100:.1f}%)",
            f"{SENSITIVITY_THRESHOLD:.2f} ({SENSITIVITY_THRESHOLD * 100:.0f}%)",
            str(MIN_VOTES),
            f"{CONTESTED_PARTY_THRESHOLD:.2f} ({CONTESTED_PARTY_THRESHOLD * 100:.0f}%)",
        ],
        "Description": [
            "For reproducible k-means/GMM initialization",
            "Range of k values evaluated for model selection",
            "Expected optimal k (conservative R, moderate R, Democrat)",
            "Ward minimizes within-cluster variance",
            "Minimum cophenetic correlation for valid dendrogram",
            "Silhouette > this indicates good cluster structure",
            "Full covariance allows elliptical clusters",
            "Multiple GMM restarts for stability",
            "Matplotlib colormap for cluster visualization",
            "Inherited from EDA; votes with minority < this are filtered",
            "Alternative threshold for sensitivity analysis",
            "Inherited from EDA; legislators with < this filtered",
            "A vote is contested for a party if >= this fraction dissents",
        ],
    })
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="See analysis/design/clustering.md for justification.",
    )
    report.add(
        TableSection(
            id="analysis-params",
            title="Analysis Parameters",
            html=html,
        )
    )
