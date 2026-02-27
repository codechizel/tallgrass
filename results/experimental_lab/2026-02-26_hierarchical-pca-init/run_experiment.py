"""Experiment: PCA-informed init for hierarchical IRT.

Compares hierarchical IRT convergence with and without PCA-informed
initialization of xi_offset. Runs per-chamber models on the 91st biennium.

Usage:
    uv run python results/experiments/.../run_experiment.py --skip-joint
    uv run python results/experiments/.../run_experiment.py --skip-joint --pca-init
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import arviz as az
import numpy as np
import polars as pl

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from analysis.hierarchical import (
    build_per_chamber_model,
    check_hierarchical_convergence,
    compute_flat_hier_correlation,
    compute_variance_decomposition,
    extract_group_params,
    extract_hierarchical_ideal_points,
    plot_dispersion,
    plot_icc,
    plot_party_posteriors,
    plot_shrinkage_scatter,
    prepare_hierarchical_data,
)
from analysis.hierarchical_report import build_hierarchical_report
from analysis.irt import (
    RANDOM_SEED,
    load_eda_matrices,
    load_metadata,
    load_pca_scores,
    plot_forest,
)

from analysis.report import ReportBuilder

EXPERIMENT_DIR = Path(__file__).parent
SESSION = "2025-26"


def compute_pca_initvals(
    pca_scores: pl.DataFrame,
    data: dict,
) -> np.ndarray:
    """Compute xi_offset initvals from PCA PC1 scores.

    The hierarchical model uses non-centered parameterization:
        xi = mu_party[party_idx] + sigma_within[party_idx] * xi_offset
    where xi_offset ~ N(0, 1).

    PCA PC1 scores are a good proxy for xi. To convert to xi_offset scale,
    we standardize to mean=0, sd=1 — matching the N(0,1) prior.

    This is a rough approximation because the true offset depends on the
    (unknown) party means, but it places chains in the correct orientation.
    """
    slug_order = {s: i for i, s in enumerate(data["leg_slugs"])}
    pc1_vals = (
        pca_scores.filter(pl.col("legislator_slug").is_in(data["leg_slugs"]))
        .sort(pl.col("legislator_slug").replace_strict(slug_order))["PC1"]
        .to_numpy()
    )

    # Standardize to N(0,1) — the xi_offset prior scale
    pc1_std = (pc1_vals - pc1_vals.mean()) / (pc1_vals.std() + 1e-8)
    return pc1_std.astype(np.float64)


def build_per_chamber_model_with_init(
    data: dict,
    n_samples: int,
    n_tune: int,
    n_chains: int,
    cores: int | None = None,
    target_accept: float = 0.95,
    xi_offset_initvals: np.ndarray | None = None,
) -> tuple[az.InferenceData, float]:
    """Wrapper around build_per_chamber_model that injects initvals.

    Since we can't modify the production code for an experiment, we
    replicate the model-building logic with the initvals parameter.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from analysis.hierarchical import (
        SMALL_GROUP_SIGMA_SCALE,
        SMALL_GROUP_THRESHOLD,
    )

    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y = data["y"]
    n_leg = data["n_legislators"]
    n_votes = data["n_votes"]
    party_idx = data["party_idx"]
    n_parties = data["n_parties"]

    party_counts = np.array([int((party_idx == p).sum()) for p in range(n_parties)])
    sigma_scale = np.array(
        [
            SMALL_GROUP_SIGMA_SCALE if party_counts[p] < SMALL_GROUP_THRESHOLD else 1.0
            for p in range(n_parties)
        ]
    )
    for p in range(n_parties):
        if party_counts[p] < SMALL_GROUP_THRESHOLD:
            print(
                f"  Adaptive prior: {data['party_names'][p]} ({party_counts[p]} members) -> "
                f"sigma_within ~ HalfNormal({SMALL_GROUP_SIGMA_SCALE})"
            )

    coords = {
        "legislator": data["leg_slugs"],
        "vote": data["vote_ids"],
        "party": data["party_names"],
        "obs_id": np.arange(data["n_obs"]),
    }

    with pm.Model(coords=coords):
        mu_party_raw = pm.Normal("mu_party_raw", mu=0, sigma=2, shape=n_parties)
        mu_party = pm.Deterministic("mu_party", pt.sort(mu_party_raw), dims="party")

        sigma_within = pm.HalfNormal(
            "sigma_within", sigma=sigma_scale, shape=n_parties, dims="party"
        )

        xi_offset = pm.Normal("xi_offset", mu=0, sigma=1, shape=n_leg, dims="legislator")
        xi = pm.Deterministic(
            "xi",
            mu_party[party_idx] + sigma_within[party_idx] * xi_offset,
            dims="legislator",
        )

        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")
        beta = pm.Normal("beta", mu=0, sigma=1, shape=n_votes, dims="vote")

        eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

        print(f"  Sampling: {n_samples} draws, {n_tune} tune, {n_chains} chains")
        print(f"  target_accept={target_accept}, seed={RANDOM_SEED}")

        sample_kwargs: dict = {}
        if xi_offset_initvals is not None:
            sample_kwargs["initvals"] = {"xi_offset": xi_offset_initvals}
            print(
                f"  PCA-informed initvals: {len(xi_offset_initvals)} params, "
                f"range [{xi_offset_initvals.min():.2f}, {xi_offset_initvals.max():.2f}]"
            )
        else:
            print("  Default init (jitter+adapt_diag)")

        t0 = time.time()
        idata = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            cores=cores if cores is not None else n_chains,
            target_accept=target_accept,
            random_seed=RANDOM_SEED,
            progressbar=True,
            **sample_kwargs,
        )
        sampling_time = time.time() - t0

    print(f"  Sampling complete in {sampling_time:.1f}s")
    return idata, sampling_time


def run_chamber(
    chamber: str,
    matrix: pl.DataFrame,
    pca_scores: pl.DataFrame,
    legislators: pl.DataFrame,
    flat_ip: pl.DataFrame | None,
    use_pca_init: bool,
    out_dir: Path,
    plots_dir: Path,
) -> tuple[dict, dict]:
    """Run one chamber and return (metrics, chamber_result)."""
    ch = chamber.lower()
    data = prepare_hierarchical_data(matrix, legislators, chamber)

    print(f"\n{'=' * 72}")
    print(f"  {chamber} — {'PCA init' if use_pca_init else 'Default init'}")
    print(f"{'=' * 72}")
    print(f"  {data['n_legislators']} legislators x {data['n_votes']} votes")
    print(f"  {data['n_obs']} observations")

    xi_init = None
    if use_pca_init:
        xi_init = compute_pca_initvals(pca_scores, data)

    if use_pca_init:
        idata, sampling_time = build_per_chamber_model_with_init(
            data,
            n_samples=2000,
            n_tune=1500,
            n_chains=2,
            xi_offset_initvals=xi_init,
        )
    else:
        idata, sampling_time = build_per_chamber_model(
            data,
            n_samples=2000,
            n_tune=1500,
            n_chains=2,
        )

    convergence = check_hierarchical_convergence(idata, chamber)
    ideal_points = extract_hierarchical_ideal_points(idata, data, legislators, flat_ip=flat_ip)
    group_params = extract_group_params(idata, data)
    icc_df = compute_variance_decomposition(idata, data)

    flat_corr = float("nan")
    if flat_ip is not None:
        flat_corr = compute_flat_hier_correlation(ideal_points, flat_ip, chamber)

    # Plots
    plot_party_posteriors(idata, data, chamber, plots_dir)
    plot_icc(icc_df, chamber, plots_dir)
    plot_shrinkage_scatter(ideal_points, chamber, plots_dir)
    plot_forest(ideal_points, chamber, plots_dir)
    plot_dispersion(idata, data, chamber, plots_dir)

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    ideal_points.write_parquet(out_dir / f"ideal_points_{ch}.parquet")
    idata.to_netcdf(str(out_dir / f"idata_{ch}.nc"))

    metrics = {
        "chamber": chamber,
        "pca_init": use_pca_init,
        "rhat_xi_max": convergence.get("xi_rhat_max", float("nan")),
        "ess_xi_min": convergence.get("xi_ess_min", float("nan")),
        "ess_mu_party_min": convergence.get("mu_party_ess_min", float("nan")),
        "ess_sigma_within_min": convergence.get("sigma_within_ess_min", float("nan")),
        "divergences": convergence.get("divergences", -1),
        "sampling_time_s": round(sampling_time, 1),
        "flat_corr": round(flat_corr, 4),
        "all_ok": convergence["all_ok"],
    }

    chamber_result = {
        "data": data,
        "idata": idata,
        "ideal_points": ideal_points,
        "group_params": group_params,
        "icc_df": icc_df,
        "convergence": convergence,
        "sampling_time": sampling_time,
        "flat_corr": flat_corr,
    }

    print(f"\n  Metrics: {json.dumps(metrics, indent=2)}")
    return metrics, chamber_result


def main():
    parser = argparse.ArgumentParser(description="Hierarchical PCA-init experiment")
    parser.add_argument("--pca-init", action="store_true", help="Use PCA-informed initialization")
    parser.add_argument("--skip-joint", action="store_true", help="Skip joint model (recommended)")
    args = parser.parse_args()

    from tallgrass.session import KSSession

    ks = KSSession.from_session_string(SESSION)
    eda_dir = ks.results_dir / "eda" / "latest"
    pca_dir = ks.results_dir / "pca" / "latest"
    irt_dir = ks.results_dir / "irt" / "latest"

    run_name = "run_02_pca_init" if args.pca_init else "run_01_baseline"
    out_dir = EXPERIMENT_DIR / run_name

    print("Experiment: Hierarchical PCA-Init")
    print(f"Session: {SESSION}")
    print(f"Mode: {'PCA-informed' if args.pca_init else 'Default (baseline)'}")
    print(f"Output: {out_dir}")
    print()

    # Load data
    house_matrix, senate_matrix, _ = load_eda_matrices(eda_dir)
    house_pca, senate_pca = load_pca_scores(pca_dir)
    _, legislators = load_metadata(ks.data_dir)

    flat_ip: dict[str, pl.DataFrame | None] = {}
    for ch in ("house", "senate"):
        flat_path = irt_dir / "data" / f"ideal_points_{ch}.parquet"
        if flat_path.exists():
            flat_ip[ch] = pl.read_parquet(flat_path)
        else:
            flat_ip[ch] = None

    init_label_display = "PCA Init" if args.pca_init else "Baseline"
    report = ReportBuilder(
        title=f"Hierarchical PCA Init Experiment — {init_label_display}",
        session=SESSION,
    )

    plots_dir = EXPERIMENT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    per_chamber_results: dict[str, dict] = {}
    t_total = time.time()

    for chamber, matrix, pca_scores in [
        ("House", house_matrix, house_pca),
        ("Senate", senate_matrix, senate_pca),
    ]:
        ch = chamber.lower()
        metrics, chamber_result = run_chamber(
            chamber,
            matrix,
            pca_scores,
            legislators,
            flat_ip[ch],
            args.pca_init,
            out_dir,
            plots_dir,
        )
        all_metrics.append(metrics)
        per_chamber_results[chamber] = chamber_result

    elapsed = time.time() - t_total
    print(f"\n{'=' * 72}")
    print(f"  EXPERIMENT COMPLETE — {elapsed:.1f}s total")
    print(f"{'=' * 72}")

    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        payload = {
            "run": run_name,
            "elapsed_s": round(elapsed, 1),
            "chambers": all_metrics,
        }
        json.dump(payload, f, indent=2)

    print(f"\nMetrics saved to {out_dir / 'metrics.json'}")

    # Build full hierarchical report
    report.elapsed_display = f"{elapsed:.1f}s"
    build_hierarchical_report(
        report,
        chamber_results=per_chamber_results,
        joint_results=None,
        plots_dir=plots_dir,
    )

    init_label = "pca_init" if args.pca_init else "baseline"
    report_path = EXPERIMENT_DIR / f"experiment_report_{init_label}.html"
    report.write(report_path)
    print(f"  HTML Report: {report_path}")


if __name__ == "__main__":
    main()
