"""Experiment: 4-chain hierarchical IRT.

Compares 2-chain vs 4-chain hierarchical IRT on the 91st biennium (2025-26).
Tests whether 4 chains roughly double ESS, improve R-hat reliability, and
match the standard configuration the diagnostic literature was calibrated
against (Vehtari et al. 2021), with minimal wall-time increase on 6 P-cores.

All runs use PCA-informed xi_offset initialization (the proven fix from the
PCA-init experiment).

Usage:
    # Run 1: Baseline 2 chains
    uv run python results/experiments/2026-02-26_hierarchical-4-chains/run_experiment.py \
        --n-chains 2 --skip-joint

    # Run 2: 4 chains
    uv run python results/experiments/2026-02-26_hierarchical-4-chains/run_experiment.py \
        --n-chains 4 --skip-joint

    # Run 3: 4 chains + 2500 draws (if Run 2 still has marginal ESS)
    uv run python results/experiments/2026-02-26_hierarchical-4-chains/run_experiment.py \
        --n-chains 4 --n-samples 2500 --skip-joint
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
    SMALL_GROUP_SIGMA_SCALE,
    SMALL_GROUP_THRESHOLD,
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

# Run directory names keyed by (n_chains, n_samples)
RUN_NAMES = {
    (2, 2000): "run_01_2chains",
    (4, 2000): "run_02_4chains",
    (4, 2500): "run_03_4chains_2500draws",
}


def compute_pca_initvals(
    pca_scores: pl.DataFrame,
    data: dict,
) -> np.ndarray:
    """Compute xi_offset initvals from PCA PC1 scores.

    The hierarchical model uses non-centered parameterization:
        xi = mu_party[party_idx] + sigma_within[party_idx] * xi_offset
    where xi_offset ~ N(0, 1).

    PCA PC1 scores are a good proxy for xi. To convert to xi_offset scale,
    we standardize to mean=0, sd=1 -- matching the N(0,1) prior.
    """
    slug_order = {s: i for i, s in enumerate(data["leg_slugs"])}
    pc1_vals = (
        pca_scores.filter(pl.col("legislator_slug").is_in(data["leg_slugs"]))
        .sort(pl.col("legislator_slug").replace_strict(slug_order))["PC1"]
        .to_numpy()
    )

    # Standardize to N(0,1) -- the xi_offset prior scale
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
    """Build and sample the per-chamber hierarchical IRT model with PCA init.

    Replicates the production model-building logic from
    analysis.hierarchical.build_per_chamber_model, adding the initvals
    parameter for PCA-informed initialization.
    """
    import pymc as pm
    import pytensor.tensor as pt

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
            # Use adapt_diag (no jitter) when PCA initvals provided.
            # jitter+adapt_diag adds random perturbation that can push chains
            # past the mode boundary, causing reflection mode-splitting with
            # 4+ chains. PCA init already provides a good starting region.
            sample_kwargs["init"] = "adapt_diag"
            print(
                f"  PCA-informed initvals: {len(xi_offset_initvals)} params, "
                f"range [{xi_offset_initvals.min():.2f}, {xi_offset_initvals.max():.2f}]"
            )
            print("  init='adapt_diag' (no jitter — PCA initvals provide orientation)")

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


def compute_per_chain_ess(idata: az.InferenceData, var: str) -> list[float]:
    """Compute per-chain minimum ESS for a variable.

    ArviZ's az.ess() computes cross-chain ESS. For per-chain diagnostics,
    we split the InferenceData into single-chain subsets and compute ESS
    on each independently.

    Returns a list of min-ESS values, one per chain.
    """
    n_chains = idata.posterior.sizes["chain"]
    per_chain_min_ess = []
    for c in range(n_chains):
        # Select single chain, keeping the chain dimension for ArviZ
        chain_data = idata.posterior[var].isel(chain=[c])
        # az.ess() returns a Dataset; extract the variable's DataArray for .values
        chain_ess_ds = az.ess(chain_data)
        chain_ess_vals = chain_ess_ds[var].values
        per_chain_min_ess.append(float(np.min(chain_ess_vals)))
    return per_chain_min_ess


def run_chamber(
    chamber: str,
    matrix: pl.DataFrame,
    pca_scores: pl.DataFrame,
    legislators: pl.DataFrame,
    flat_ip: pl.DataFrame | None,
    n_chains: int,
    n_samples: int,
    out_dir: Path,
    plots_dir: Path,
) -> tuple[dict, dict]:
    """Run one chamber and return (metrics, chamber_result).

    metrics: experiment-specific dict with per-chain ESS etc.
    chamber_result: dict compatible with build_hierarchical_report().
    """
    ch = chamber.lower()
    data = prepare_hierarchical_data(matrix, legislators, chamber)

    print(f"\n{'=' * 72}")
    print(f"  {chamber} -- {n_chains} chains, {n_samples} draws, PCA init")
    print(f"{'=' * 72}")
    print(f"  {data['n_legislators']} legislators x {data['n_votes']} votes")
    print(f"  {data['n_obs']} observations")

    xi_init = compute_pca_initvals(pca_scores, data)

    idata, sampling_time = build_per_chamber_model_with_init(
        data,
        n_samples=n_samples,
        n_tune=1500,
        n_chains=n_chains,
        xi_offset_initvals=xi_init,
    )

    # Standard convergence checks (uses production thresholds)
    convergence = check_hierarchical_convergence(idata, chamber)

    # Per-chain ESS -- the key metric for this experiment
    per_chain_ess_xi = compute_per_chain_ess(idata, "xi")
    per_chain_ess_mu = compute_per_chain_ess(idata, "mu_party")
    per_chain_ess_sigma = compute_per_chain_ess(idata, "sigma_within")

    print(f"\n  Per-chain ESS (xi min): {[f'{v:.0f}' for v in per_chain_ess_xi]}")
    print(f"  Per-chain ESS (mu_party min): {[f'{v:.0f}' for v in per_chain_ess_mu]}")
    print(f"  Per-chain ESS (sigma_within min): {[f'{v:.0f}' for v in per_chain_ess_sigma]}")

    # Extract ideal points and group-level results
    ideal_points = extract_hierarchical_ideal_points(idata, data, legislators, flat_ip=flat_ip)
    group_params = extract_group_params(idata, data)
    icc_df = compute_variance_decomposition(idata, data)

    # E-BFMI per chain
    bfmi_values = az.bfmi(idata)

    flat_corr = float("nan")
    if flat_ip is not None:
        flat_corr = compute_flat_hier_correlation(ideal_points, flat_ip, chamber)

    # Generate plots
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
        "n_chains": n_chains,
        "n_samples": n_samples,
        "pca_init": True,
        "rhat_xi_max": convergence.get("xi_rhat_max", float("nan")),
        "rhat_mu_party_max": convergence.get("mu_party_rhat_max", float("nan")),
        "rhat_sigma_within_max": convergence.get("sigma_within_rhat_max", float("nan")),
        "ess_xi_min": convergence.get("xi_ess_min", float("nan")),
        "ess_xi_per_chain_min": per_chain_ess_xi,
        "ess_mu_party_min": convergence.get("mu_party_ess_min", float("nan")),
        "ess_mu_party_per_chain_min": per_chain_ess_mu,
        "ess_sigma_within_min": convergence.get("sigma_within_ess_min", float("nan")),
        "ess_sigma_within_per_chain_min": per_chain_ess_sigma,
        "divergences": convergence.get("divergences", -1),
        "ebfmi": [float(v) for v in bfmi_values],
        "ebfmi_min": float(min(bfmi_values)),
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

    print(f"\n  Metrics: {json.dumps(metrics, indent=2, default=str)}")
    return metrics, chamber_result


def compute_cross_run_correlation(run_dir: Path, baseline_dir: Path, chamber: str) -> float | None:
    """Compute Spearman correlation of ideal points between two runs."""
    from scipy.stats import spearmanr

    ch = chamber.lower()
    run_path = run_dir / f"ideal_points_{ch}.parquet"
    base_path = baseline_dir / f"ideal_points_{ch}.parquet"

    if not run_path.exists() or not base_path.exists():
        return None

    run_ip = pl.read_parquet(run_path)
    base_ip = pl.read_parquet(base_path)

    # Join on legislator_slug to align
    merged = run_ip.select(pl.col("legislator_slug"), pl.col("xi_mean").alias("xi_run")).join(
        base_ip.select(pl.col("legislator_slug"), pl.col("xi_mean").alias("xi_base")),
        on="legislator_slug",
        how="inner",
    )

    if len(merged) == 0:
        return None

    corr, _ = spearmanr(merged["xi_run"].to_numpy(), merged["xi_base"].to_numpy())
    return float(corr)


def main():
    parser = argparse.ArgumentParser(description="4-chain hierarchical IRT experiment")
    parser.add_argument(
        "--n-chains",
        type=int,
        default=2,
        help="Number of MCMC chains (default: 2)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of posterior draws per chain (default: 2000)",
    )
    parser.add_argument(
        "--skip-joint",
        action="store_true",
        help="Skip joint model (recommended for this experiment)",
    )
    args = parser.parse_args()

    from tallgrass.session import KSSession

    ks = KSSession.from_session_string(SESSION)
    eda_dir = ks.results_dir / "eda" / "latest"
    pca_dir = ks.results_dir / "pca" / "latest"
    irt_dir = ks.results_dir / "irt" / "latest"

    # Determine run name from parameters
    run_key = (args.n_chains, args.n_samples)
    if run_key in RUN_NAMES:
        run_name = RUN_NAMES[run_key]
    else:
        run_name = f"run_custom_{args.n_chains}chains_{args.n_samples}draws"

    out_dir = EXPERIMENT_DIR / run_name

    print("Experiment: 4-Chain Hierarchical IRT")
    print(f"Session: {SESSION}")
    print(f"Chains: {args.n_chains}, Draws: {args.n_samples}, PCA init: always")
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

    all_metrics = []
    per_chamber_results: dict[str, dict] = {}
    t_total = time.time()

    plots_dir = EXPERIMENT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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
            args.n_chains,
            args.n_samples,
            out_dir,
            plots_dir,
        )
        all_metrics.append(metrics)
        per_chamber_results[chamber] = chamber_result

    elapsed = time.time() - t_total

    # Compute cross-run correlation against baseline if this isn't Run 1
    baseline_dir = EXPERIMENT_DIR / "run_01_2chains"
    if out_dir != baseline_dir and baseline_dir.exists():
        print(f"\n{'=' * 72}")
        print("  CROSS-RUN CORRELATION vs baseline (run_01_2chains)")
        print(f"{'=' * 72}")
        for chamber in ("House", "Senate"):
            corr = compute_cross_run_correlation(out_dir, baseline_dir, chamber)
            if corr is not None:
                print(f"  {chamber} xi Spearman correlation: {corr:.4f}")
                # Add to the relevant chamber metrics
                for m in all_metrics:
                    if m["chamber"] == chamber:
                        m["xi_corr_vs_baseline"] = round(corr, 4)

    print(f"\n{'=' * 72}")
    print(f"  EXPERIMENT COMPLETE -- {elapsed:.1f}s total")
    print(f"{'=' * 72}")

    # Save metrics
    summary = {
        "run": run_name,
        "n_chains": args.n_chains,
        "n_samples": args.n_samples,
        "elapsed_s": round(elapsed, 1),
        "chambers": all_metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMetrics saved to {out_dir / 'metrics.json'}")

    # HTML report using production hierarchical report builder
    report = ReportBuilder(
        title=f"Hierarchical IRT — {args.n_chains} Chains, {args.n_samples} Draws",
        session=SESSION,
        elapsed_display=f"{elapsed:.1f}s",
    )
    build_hierarchical_report(
        report,
        chamber_results=per_chamber_results,
        joint_results=None,
        plots_dir=plots_dir,
    )
    report_path = EXPERIMENT_DIR / f"experiment_report_{run_name}.html"
    report.write(report_path)
    print(f"  HTML Report: {report_path}")


if __name__ == "__main__":
    main()
