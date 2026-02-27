"""Experiment: Positive beta constraint for hierarchical IRT convergence.

Tests whether constraining bill discrimination (beta) to be positive eliminates
the reflection-mode multimodality that causes House convergence failures.

Three per-chamber variants:
  - baseline: beta ~ Normal(0, 1)        [current production model]
  - lognormal: beta ~ LogNormal(0, 0.5)  [soft positive, prior median=1.0]
  - halfnormal: beta ~ HalfNormal(1)     [hard zero floor, wider spread]

Plus an optional joint cross-chamber run using the best-performing variant.

Each run produces full HTML reports, parquet outputs, and plots — identical to
the production hierarchical pipeline — so downstream impacts can be visually
inspected.

Usage:
    # Baseline (current model)
    uv run python results/experiments/2026-02-27_positive-beta/run_experiment.py \
        --variant baseline

    # LogNormal positive constraint
    uv run python results/experiments/2026-02-27_positive-beta/run_experiment.py \
        --variant lognormal

    # HalfNormal positive constraint
    uv run python results/experiments/2026-02-27_positive-beta/run_experiment.py \
        --variant halfnormal

    # Best variant + joint model
    uv run python results/experiments/2026-02-27_positive-beta/run_experiment.py \
        --variant lognormal --include-joint
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt
from scipy import stats

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.hierarchical import (
    SMALL_GROUP_SIGMA_SCALE,
    SMALL_GROUP_THRESHOLD,
    _match_bills_across_chambers,
    build_joint_model,
    check_hierarchical_convergence,
    compute_flat_hier_correlation,
    compute_variance_decomposition,
    extract_group_params,
    extract_hierarchical_ideal_points,
    fix_joint_sign_convention,
    plot_dispersion,
    plot_icc,
    plot_joint_party_spread,
    plot_party_posteriors,
    plot_shrinkage_scatter,
    prepare_hierarchical_data,
)
from analysis.hierarchical_report import build_hierarchical_report
from analysis.irt import (
    ESS_THRESHOLD,
    MAX_DIVERGENCES,
    PARTY_COLORS,
    RANDOM_SEED,
    RHAT_THRESHOLD,
    load_eda_matrices,
    load_metadata,
    load_pca_scores,
    plot_forest,
)
from analysis.report import ReportBuilder

EXPERIMENT_DIR = Path(__file__).parent
SESSION = "2025-26"

VARIANTS = {
    "baseline": {
        "name": "Baseline (Normal)",
        "dir": "run_01_baseline",
        "description": "beta ~ Normal(0, 1) — current production model",
    },
    "lognormal": {
        "name": "LogNormal",
        "dir": "run_02_lognormal",
        "description": "beta ~ LogNormal(0, 0.5) — soft positive, prior median=1.0",
    },
    "halfnormal": {
        "name": "HalfNormal",
        "dir": "run_03_halfnormal",
        "description": "beta ~ HalfNormal(1) — hard zero floor, wider spread",
    },
}

# MCMC settings (match production)
N_SAMPLES = 2000
N_TUNE = 1500
N_CHAINS = 4
TARGET_ACCEPT = 0.95


def print_header(title: str) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def compute_pca_initvals(
    pca_scores: pl.DataFrame,
    data: dict,
) -> np.ndarray:
    """Compute xi_offset initvals from PCA PC1 scores."""
    slug_order = {s: i for i, s in enumerate(data["leg_slugs"])}
    pc1_vals = (
        pca_scores.filter(pl.col("legislator_slug").is_in(data["leg_slugs"]))
        .sort(pl.col("legislator_slug").replace_strict(slug_order))["PC1"]
        .to_numpy()
    )
    pc1_std = (pc1_vals - pc1_vals.mean()) / (pc1_vals.std() + 1e-8)
    return pc1_std.astype(np.float64)


def build_model_with_variant(
    data: dict,
    variant: str,
    *,
    n_samples: int = N_SAMPLES,
    n_tune: int = N_TUNE,
    n_chains: int = N_CHAINS,
    target_accept: float = TARGET_ACCEPT,
    xi_offset_initvals: np.ndarray | None = None,
) -> tuple[az.InferenceData, float]:
    """Build and sample the per-chamber model with a specific beta prior variant.

    This replicates build_per_chamber_model from production but swaps the beta
    prior based on the variant parameter.
    """
    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y = data["y"]
    n_leg = data["n_legislators"]
    n_votes = data["n_votes"]
    party_idx = data["party_idx"]
    n_parties = data["n_parties"]

    # Adaptive priors for small groups (production behavior)
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
                f"  Adaptive prior: {data['party_names'][p]} ({party_counts[p]} members) "
                f"-> sigma_within ~ HalfNormal({SMALL_GROUP_SIGMA_SCALE})"
            )

    coords = {
        "legislator": data["leg_slugs"],
        "vote": data["vote_ids"],
        "party": data["party_names"],
        "obs_id": np.arange(data["n_obs"]),
    }

    with pm.Model(coords=coords):
        # --- Party-level parameters (identical across variants) ---
        mu_party_raw = pm.Normal("mu_party_raw", mu=0, sigma=2, shape=n_parties)
        mu_party = pm.Deterministic("mu_party", pt.sort(mu_party_raw), dims="party")

        sigma_within = pm.HalfNormal(
            "sigma_within", sigma=sigma_scale, shape=n_parties, dims="party"
        )

        # --- Non-centered legislator ideal points (identical) ---
        xi_offset = pm.Normal("xi_offset", mu=0, sigma=1, shape=n_leg, dims="legislator")
        xi = pm.Deterministic(
            "xi",
            mu_party[party_idx] + sigma_within[party_idx] * xi_offset,
            dims="legislator",
        )

        # --- Bill parameters: alpha is identical, beta varies by variant ---
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")

        if variant == "baseline":
            beta = pm.Normal("beta", mu=0, sigma=1, shape=n_votes, dims="vote")
            print("  Beta prior: Normal(0, 1) [baseline]")
        elif variant == "lognormal":
            beta = pm.LogNormal("beta", mu=0, sigma=0.5, shape=n_votes, dims="vote")
            print("  Beta prior: LogNormal(0, 0.5) [positive constraint]")
        elif variant == "halfnormal":
            beta = pm.HalfNormal("beta", sigma=1, shape=n_votes, dims="vote")
            print("  Beta prior: HalfNormal(1) [positive constraint]")
        else:
            msg = f"Unknown variant: {variant}"
            raise ValueError(msg)

        # --- Likelihood (identical) ---
        eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

        # --- Sample ---
        print(f"  Sampling: {n_samples} draws, {n_tune} tune, {n_chains} chains")
        print(f"  target_accept={target_accept}, seed={RANDOM_SEED}")

        sample_kwargs: dict = {}
        if xi_offset_initvals is not None:
            sample_kwargs["initvals"] = {"xi_offset": xi_offset_initvals}
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
            cores=n_chains,
            target_accept=target_accept,
            random_seed=RANDOM_SEED,
            progressbar=True,
            **sample_kwargs,
        )
        sampling_time = time.time() - t0

    print(f"  Sampling complete in {sampling_time:.1f}s")
    return idata, sampling_time


def build_joint_model_with_variant(
    house_data: dict,
    senate_data: dict,
    variant: str,
    *,
    n_samples: int = N_SAMPLES,
    n_tune: int = N_TUNE,
    n_chains: int = N_CHAINS,
    target_accept: float = TARGET_ACCEPT,
    rollcalls: pl.DataFrame,
) -> tuple[az.InferenceData, dict, float]:
    """Build and sample the joint cross-chamber model with a specific beta variant.

    Replicates build_joint_model from production but swaps the beta prior.
    Data preparation mirrors hierarchical.py build_joint_model lines 541-648.
    """
    # --- Combined legislator data ---
    n_house = house_data["n_legislators"]
    n_senate = senate_data["n_legislators"]
    n_leg = n_house + n_senate
    all_slugs = house_data["leg_slugs"] + senate_data["leg_slugs"]

    # --- Bill matching ---
    house_vote_ids = house_data["vote_ids"]
    senate_vote_ids = senate_data["vote_ids"]

    matched_bills, house_only_vids, senate_only_vids = _match_bills_across_chambers(
        house_vote_ids, senate_vote_ids, rollcalls,
    )
    n_shared = len(matched_bills)

    all_vote_ids: list[str] = []
    for m in matched_bills:
        all_vote_ids.append(f"matched_{m['bill_number']}")
    all_vote_ids.extend(house_only_vids)
    all_vote_ids.extend(senate_only_vids)
    n_votes = len(all_vote_ids)
    vote_to_idx = {v: i for i, v in enumerate(all_vote_ids)}

    original_vid_to_idx: dict[str, int] = {}
    for i, m in enumerate(matched_bills):
        original_vid_to_idx[m["house_vote_id"]] = i
        original_vid_to_idx[m["senate_vote_id"]] = i
    for vid in house_only_vids:
        original_vid_to_idx[vid] = vote_to_idx[vid]
    for vid in senate_only_vids:
        original_vid_to_idx[vid] = vote_to_idx[vid]

    print(
        f"  Joint bill matching: {n_shared} shared bills, "
        f"{len(house_only_vids)} house-only, {len(senate_only_vids)} senate-only"
    )

    # --- Remap indices ---
    senate_leg_offset = n_house
    combined_leg_idx = np.concatenate([
        house_data["leg_idx"],
        senate_data["leg_idx"] + senate_leg_offset,
    ])
    combined_vote_idx = np.concatenate([
        np.array([original_vid_to_idx[house_vote_ids[v]] for v in house_data["vote_idx"]]),
        np.array([original_vid_to_idx[senate_vote_ids[v]] for v in senate_data["vote_idx"]]),
    ])
    combined_y = np.concatenate([house_data["y"], senate_data["y"]])

    # --- Group indices ---
    group_names = ["House Democrat", "House Republican", "Senate Democrat", "Senate Republican"]
    n_groups = len(group_names)
    group_idx = np.concatenate([
        house_data["party_idx"],
        senate_data["party_idx"] + 2,
    ])
    group_chamber = np.array([0, 0, 1, 1], dtype=np.int64)

    # --- Adaptive priors ---
    group_counts = np.array([int((group_idx == g).sum()) for g in range(n_groups)])
    sigma_scale = np.array([
        SMALL_GROUP_SIGMA_SCALE if group_counts[g] < SMALL_GROUP_THRESHOLD else 1.0
        for g in range(n_groups)
    ])
    for g in range(n_groups):
        if group_counts[g] < SMALL_GROUP_THRESHOLD:
            print(
                f"  Adaptive prior: {group_names[g]} ({group_counts[g]} members) "
                f"-> sigma_within ~ HalfNormal({SMALL_GROUP_SIGMA_SCALE})"
            )

    n_obs = len(combined_y)

    coords = {
        "legislator": all_slugs,
        "vote": all_vote_ids,
        "group": group_names,
        "chamber": ["House", "Senate"],
        "obs_id": np.arange(n_obs),
    }

    with pm.Model(coords=coords):
        # --- Chamber-level ---
        mu_global = pm.Normal("mu_global", mu=0, sigma=2)
        sigma_chamber = pm.HalfNormal("sigma_chamber", sigma=1)
        chamber_offset = pm.Normal("chamber_offset", mu=0, sigma=1, shape=2, dims="chamber")
        mu_chamber = pm.Deterministic(
            "mu_chamber", mu_global + sigma_chamber * chamber_offset, dims="chamber"
        )

        # --- Group-level (4 groups: House-D, House-R, Senate-D, Senate-R) ---
        sigma_party = pm.HalfNormal("sigma_party", sigma=1)
        group_offset_raw = pm.Normal(
            "group_offset_raw", mu=0, sigma=1, shape=n_groups, dims="group"
        )
        house_pair = pt.sort(group_offset_raw[:2])
        senate_pair = pt.sort(group_offset_raw[2:])
        group_offset_sorted = pt.concatenate([house_pair, senate_pair])
        mu_group = pm.Deterministic(
            "mu_group",
            mu_chamber[group_chamber] + sigma_party * group_offset_sorted,
            dims="group",
        )

        # --- Within-group ---
        sigma_within = pm.HalfNormal(
            "sigma_within", sigma=sigma_scale, shape=n_groups, dims="group"
        )

        # --- Non-centered legislator ideal points ---
        xi_offset = pm.Normal("xi_offset", mu=0, sigma=1, shape=n_leg, dims="legislator")
        xi = pm.Deterministic(
            "xi",
            mu_group[group_idx] + sigma_within[group_idx] * xi_offset,
            dims="legislator",
        )

        # --- Bill parameters: alpha identical, beta varies ---
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")

        if variant == "lognormal":
            beta = pm.LogNormal("beta", mu=0, sigma=0.5, shape=n_votes, dims="vote")
            print("  Beta prior: LogNormal(0, 0.5) [positive constraint]")
        elif variant == "halfnormal":
            beta = pm.HalfNormal("beta", sigma=1, shape=n_votes, dims="vote")
            print("  Beta prior: HalfNormal(1) [positive constraint]")
        else:
            beta = pm.Normal("beta", mu=0, sigma=1, shape=n_votes, dims="vote")
            print("  Beta prior: Normal(0, 1) [baseline]")

        # --- Likelihood ---
        eta = beta[combined_vote_idx] * xi[combined_leg_idx] - alpha[combined_vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=combined_y, dims="obs_id")

        # --- Sample ---
        print(f"  Joint: {n_samples} draws, {n_tune} tune, {n_chains} chains")
        print(f"  {n_leg} legislators, {n_votes} votes ({n_shared} shared), {n_obs} observations")
        print(f"  target_accept={target_accept}, seed={RANDOM_SEED}")

        t0 = time.time()
        idata = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            cores=n_chains,
            target_accept=target_accept,
            random_seed=RANDOM_SEED,
            progressbar=True,
        )
        sampling_time = time.time() - t0

    combined_data = {
        "leg_slugs": all_slugs,
        "vote_ids": all_vote_ids,
        "n_legislators": n_leg,
        "n_votes": n_votes,
        "n_shared_bills": n_shared,
        "n_obs": n_obs,
        "group_idx": group_idx,
        "group_names": group_names,
        "n_groups": n_groups,
        "group_chamber": group_chamber,
        "n_house": n_house,
        "n_senate": n_senate,
        "matched_bills": matched_bills,
    }

    print(f"  Joint sampling complete in {sampling_time:.1f}s")
    return idata, combined_data, sampling_time


def run_per_chamber(
    chamber: str,
    variant: str,
    matrix: pl.DataFrame,
    pca_scores: pl.DataFrame,
    legislators: pl.DataFrame,
    flat_ip: pl.DataFrame | None,
    out_dir: Path,
) -> dict:
    """Run one chamber with the specified beta variant. Full production output."""
    ch = chamber.lower()
    data = prepare_hierarchical_data(matrix, legislators, chamber)

    print_header(f"HIERARCHICAL IRT — {chamber}")
    print(f"  {data['n_legislators']} legislators x {data['n_votes']} votes")
    print(f"  Observed cells: {data['n_obs']:,} / {data['n_legislators'] * data['n_votes']:,} "
          f"({data['n_obs'] / (data['n_legislators'] * data['n_votes']):.1%})")
    print(f"  Yea rate: {data['y'].mean():.3f}")
    for p in range(data["n_parties"]):
        n = int((data["party_idx"] == p).sum())
        print(f"  {data['party_names'][p]}: {n} legislators")

    # PCA init
    xi_init = compute_pca_initvals(pca_scores, data)
    print(f"  PCA init: {len(xi_init)} params, range [{xi_init.min():.2f}, {xi_init.max():.2f}]")

    # Sample
    print_header(f"SAMPLING — {chamber}")
    idata, sampling_time = build_model_with_variant(
        data, variant, xi_offset_initvals=xi_init,
    )

    # Convergence
    print_header(f"CONVERGENCE — {chamber}")
    convergence = check_hierarchical_convergence(idata, chamber)

    # Extract results
    print_header(f"EXTRACTING RESULTS — {chamber}")
    ideal_points = extract_hierarchical_ideal_points(
        idata, data, legislators, flat_ip=flat_ip,
    )
    group_params = extract_group_params(idata, data)
    icc_df = compute_variance_decomposition(idata, data)

    flat_corr = float("nan")
    if flat_ip is not None:
        flat_corr = compute_flat_hier_correlation(ideal_points, flat_ip, chamber)

    # Print group params
    print("\n  Group-level parameters:")
    for row in group_params.iter_rows(named=True):
        print(
            f"    {row['party']}: mu={row['mu_mean']:+.3f} "
            f"[{row['mu_hdi_2.5']:+.3f}, {row['mu_hdi_97.5']:+.3f}], "
            f"sigma={row['sigma_within_mean']:.3f}"
        )

    # Save data
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ideal_points.write_parquet(data_dir / f"hierarchical_ideal_points_{ch}.parquet")
    group_params.write_parquet(data_dir / f"group_params_{ch}.parquet")
    icc_df.write_parquet(data_dir / f"variance_decomposition_{ch}.parquet")
    idata.to_netcdf(str(data_dir / f"idata_{ch}.nc"))

    # Plots
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print_header(f"PLOTS — {chamber}")
    plot_party_posteriors(idata, data, chamber, plots_dir)
    plot_icc(icc_df, chamber, plots_dir)
    plot_shrinkage_scatter(ideal_points, chamber, plots_dir)
    plot_forest(ideal_points, chamber, plots_dir)
    plot_dispersion(idata, data, chamber, plots_dir)

    return {
        "data": data,
        "idata": idata,
        "ideal_points": ideal_points,
        "group_params": group_params,
        "icc_df": icc_df,
        "convergence": convergence,
        "sampling_time": sampling_time,
        "flat_corr": flat_corr,
    }


def run_joint(
    variant: str,
    per_chamber_results: dict,
    legislators: pl.DataFrame,
    rollcalls: pl.DataFrame,
    out_dir: Path,
) -> dict | None:
    """Run the joint cross-chamber model with the specified beta variant."""
    print_header("JOINT CROSS-CHAMBER MODEL")

    house_data = per_chamber_results["House"]["data"]
    senate_data = per_chamber_results["Senate"]["data"]

    joint_idata, combined_data, joint_time = build_joint_model_with_variant(
        house_data, senate_data, variant, rollcalls=rollcalls,
    )

    print_header("CONVERGENCE — Joint")
    joint_convergence = check_hierarchical_convergence(joint_idata, "Joint")

    # Fix sign indeterminacy
    joint_idata, flipped_chambers = fix_joint_sign_convention(
        joint_idata, combined_data, per_chamber_results,
    )

    # Extract
    joint_ip = extract_hierarchical_ideal_points(
        joint_idata, combined_data, legislators, flipped_chambers=flipped_chambers,
    )

    # Save
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    joint_ip.write_parquet(data_dir / "hierarchical_ideal_points_joint.parquet")
    joint_idata.to_netcdf(str(data_dir / "idata_joint.nc"))

    # Joint plots
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print_header("JOINT PLOTS")
    plot_joint_party_spread(joint_idata, combined_data, plots_dir)
    plot_forest(joint_ip, "Joint", plots_dir)

    return {
        "idata": joint_idata,
        "combined_data": combined_data,
        "ideal_points": joint_ip,
        "convergence": joint_convergence,
        "sampling_time": joint_time,
    }


def compute_cross_variant_correlation(
    run_dir: Path, baseline_dir: Path, chamber: str,
) -> float | None:
    """Pearson correlation of ideal points between a variant and baseline."""
    ch = chamber.lower()
    run_path = run_dir / "data" / f"hierarchical_ideal_points_{ch}.parquet"
    base_path = baseline_dir / "data" / f"hierarchical_ideal_points_{ch}.parquet"

    if not run_path.exists() or not base_path.exists():
        return None

    run_ip = pl.read_parquet(run_path)
    base_ip = pl.read_parquet(base_path)

    merged = run_ip.select(
        pl.col("legislator_slug"), pl.col("xi_mean").alias("xi_run"),
    ).join(
        base_ip.select(pl.col("legislator_slug"), pl.col("xi_mean").alias("xi_base")),
        on="legislator_slug",
        how="inner",
    )

    if len(merged) == 0:
        return None

    r, _ = stats.pearsonr(merged["xi_run"].to_numpy(), merged["xi_base"].to_numpy())
    return float(r)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Positive beta constraint experiment for hierarchical IRT"
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        required=True,
        help="Beta prior variant to test",
    )
    parser.add_argument(
        "--include-joint",
        action="store_true",
        help="Also run the joint cross-chamber model",
    )
    parser.add_argument(
        "--session",
        default=SESSION,
        help=f"Session to run (default: {SESSION})",
    )
    args = parser.parse_args()

    variant_info = VARIANTS[args.variant]
    out_dir = EXPERIMENT_DIR / variant_info["dir"]

    from tallgrass.session import KSSession

    ks = KSSession.from_session_string(args.session)
    eda_dir = ks.results_dir / "eda" / "latest"
    pca_dir = ks.results_dir / "pca" / "latest"
    irt_dir = ks.results_dir / "irt" / "latest"

    print(f"Experiment: Positive Beta Constraint")
    print(f"Variant: {variant_info['name']} — {variant_info['description']}")
    print(f"Session: {args.session}")
    print(f"Output: {out_dir}")
    print()

    # Load data
    house_matrix, senate_matrix, _ = load_eda_matrices(eda_dir)
    house_pca, senate_pca = load_pca_scores(pca_dir)
    rollcalls, legislators = load_metadata(ks.data_dir)

    flat_ip: dict[str, pl.DataFrame | None] = {}
    for ch in ("house", "senate"):
        flat_path = irt_dir / "data" / f"ideal_points_{ch}.parquet"
        flat_ip[ch] = pl.read_parquet(flat_path) if flat_path.exists() else None

    t_total = time.time()
    per_chamber_results: dict[str, dict] = {}

    # Per-chamber runs
    for chamber, matrix, pca_scores in [
        ("House", house_matrix, house_pca),
        ("Senate", senate_matrix, senate_pca),
    ]:
        ch = chamber.lower()
        per_chamber_results[chamber] = run_per_chamber(
            chamber, args.variant, matrix, pca_scores,
            legislators, flat_ip[ch], out_dir,
        )

    # Joint model (optional)
    joint_results = None
    if args.include_joint:
        try:
            joint_results = run_joint(
                args.variant, per_chamber_results, legislators, rollcalls, out_dir,
            )
        except Exception as e:
            print(f"\n  WARNING: Joint model failed: {e}")
            print("  Continuing with per-chamber results only.")

    elapsed = time.time() - t_total

    # Format elapsed time for the report header
    def _fmt_elapsed(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, secs = divmod(int(seconds), 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours, mins = divmod(minutes, 60)
        return f"{hours}h {mins}m {secs}s"

    # HTML Report
    print_header("HTML REPORT")
    report = ReportBuilder(
        title=f"Kansas Legislature {args.session} — Hierarchical Bayesian IRT "
              f"[{variant_info['name']}]",
        session=args.session,
        git_hash="experiment",
        elapsed_display=_fmt_elapsed(elapsed),
    )
    plots_dir = out_dir / "plots"
    build_hierarchical_report(
        report,
        chamber_results=per_chamber_results,
        joint_results=joint_results,
        plots_dir=plots_dir,
    )
    report_path = out_dir / "hierarchical_report.html"
    report.write(report_path)
    print(f"  Saved: {report_path}")

    # Cross-variant correlation
    baseline_dir = EXPERIMENT_DIR / VARIANTS["baseline"]["dir"]
    if out_dir != baseline_dir and baseline_dir.exists():
        print_header("CROSS-VARIANT CORRELATION vs Baseline")
        for chamber in ("House", "Senate"):
            r = compute_cross_variant_correlation(out_dir, baseline_dir, chamber)
            if r is not None:
                print(f"  {chamber} xi Pearson r vs baseline: {r:.4f}")

    # Metrics summary
    print_header("EXPERIMENT SUMMARY")
    metrics: dict = {
        "variant": args.variant,
        "variant_description": variant_info["description"],
        "session": args.session,
        "elapsed_s": round(elapsed, 1),
        "include_joint": args.include_joint,
        "chambers": {},
    }

    for chamber in ("House", "Senate"):
        if chamber not in per_chamber_results:
            continue
        res = per_chamber_results[chamber]
        ch_metrics = {
            "n_legislators": res["ideal_points"].height,
            "sampling_time_s": round(res["sampling_time"], 1),
            "convergence_ok": res["convergence"]["all_ok"],
            "rhat_xi_max": res["convergence"].get("xi_rhat_max"),
            "rhat_mu_party_max": res["convergence"].get("mu_party_rhat_max"),
            "rhat_sigma_within_max": res["convergence"].get("sigma_within_rhat_max"),
            "ess_xi_min": res["convergence"].get("xi_ess_min"),
            "ess_mu_party_min": res["convergence"].get("mu_party_ess_min"),
            "divergences": res["convergence"].get("divergences", -1),
            "flat_corr": round(res["flat_corr"], 4),
            "icc_mean": round(float(res["icc_df"]["icc_mean"][0]), 4),
        }

        # Cross-variant correlation
        if out_dir != baseline_dir and baseline_dir.exists():
            r = compute_cross_variant_correlation(out_dir, baseline_dir, chamber)
            if r is not None:
                ch_metrics["pearson_r_vs_baseline"] = round(r, 4)

        metrics["chambers"][chamber] = ch_metrics
        print(f"\n  {chamber}:")
        print(f"    Convergence: {'PASS' if ch_metrics['convergence_ok'] else 'FAIL'}")
        print(f"    R-hat(xi) max: {ch_metrics['rhat_xi_max']:.4f}")
        print(f"    ESS(xi) min: {ch_metrics['ess_xi_min']:.0f}")
        print(f"    R-hat(mu_party) max: {ch_metrics['rhat_mu_party_max']:.4f}")
        print(f"    ESS(mu_party) min: {ch_metrics['ess_mu_party_min']:.0f}")
        print(f"    Divergences: {ch_metrics['divergences']}")
        print(f"    Flat IRT r: {ch_metrics['flat_corr']:.4f}")
        print(f"    ICC: {ch_metrics['icc_mean']:.4f}")
        print(f"    Time: {ch_metrics['sampling_time_s']:.0f}s")

    if joint_results is not None:
        jm = {
            "n_legislators": joint_results["ideal_points"].height,
            "sampling_time_s": round(joint_results["sampling_time"], 1),
            "convergence_ok": joint_results["convergence"]["all_ok"],
            "rhat_xi_max": joint_results["convergence"].get("xi_rhat_max"),
            "ess_xi_min": joint_results["convergence"].get("xi_ess_min"),
            "divergences": joint_results["convergence"].get("divergences", -1),
            "n_shared_bills": joint_results["combined_data"]["n_shared_bills"],
        }
        metrics["joint"] = jm
        print(f"\n  Joint:")
        print(f"    Convergence: {'PASS' if jm['convergence_ok'] else 'FAIL'}")
        print(f"    R-hat(xi) max: {jm['rhat_xi_max']:.4f}")
        print(f"    ESS(xi) min: {jm['ess_xi_min']:.0f}")
        print(f"    Divergences: {jm['divergences']}")
        print(f"    Time: {jm['sampling_time_s']:.0f}s")

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"  DONE — {variant_info['name']} in {elapsed:.0f}s")
    print(f"  Metrics: {metrics_path}")
    print(f"  Report: {report_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
