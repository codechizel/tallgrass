"""Experiment 2b: nutpie hierarchical per-chamber IRT with PCA initialization.

Experiment 2 showed that nutpie resolves the House convergence failure but Senate
fails due to reflection mode-splitting without PCA initialization. This follow-up
tests whether PCA-informed xi_offset initialization fixes Senate while maintaining
House convergence.

Key insight: nutpie's PCA init mechanism differs from PyMC's. Instead of
`pm.sample(initvals=...)`, nutpie uses `compile_pymc_model(initial_points=...)`.
The `jitter_rvs=set()` parameter disables jitter — matching production's
`adapt_diag` (no jitter) strategy from ADR-0045.

Two runs:
  1. House (should still pass — PCA init may improve ESS)
  2. Senate (should now pass — PCA init prevents mode-splitting)

Comparison baselines:
  - Experiment 2 (nutpie, no PCA init)
  - PyMC production hierarchical
  - Flat IRT

Usage:
    uv run python results/experimental_lab/2026-02-27_nutpie-hierarchical/run_experiment_pca_init.py
"""

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
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.hierarchical import (
    HIER_CONVERGENCE_VARS,
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
from analysis.model_spec import PRODUCTION_BETA

from analysis.experiment_monitor import ExperimentLifecycle, PlatformCheck
from analysis.report import ReportBuilder
from tallgrass.session import KSSession

EXPERIMENT_DIR = Path(__file__).parent
SESSION = "2025-26"

# Match production hierarchical sampling settings
N_SAMPLES = 2000
N_TUNE = 1500
N_CHAINS = 4


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def build_hierarchical_model(data: dict, beta_prior=PRODUCTION_BETA):
    """Build hierarchical per-chamber IRT model and return it WITHOUT sampling.

    Extracts the model-building logic from build_per_chamber_model() so nutpie
    can compile the model graph separately from sampling.

    Model structure (identical to production):
        mu_party_raw ~ Normal(0, 2)  →  mu_party = sort(mu_party_raw)
        sigma_within ~ HalfNormal(sigma_scale)  (adaptive for small groups)
        xi_offset ~ Normal(0, 1)  →  xi = mu_party[p] + sigma_within[p] * xi_offset
        alpha ~ Normal(0, 5)
        beta ~ beta_prior (default Normal(0, 1))
        P(Yea) = logit^-1(beta * xi - alpha)
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

    # Group-size-adaptive priors for sigma_within (Gelman 2015)
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
                f"  Adaptive prior: {data['party_names'][p]} ({party_counts[p]} members) → "
                f"sigma_within ~ HalfNormal({SMALL_GROUP_SIGMA_SCALE})"
            )

    coords = {
        "legislator": data["leg_slugs"],
        "vote": data["vote_ids"],
        "party": data["party_names"],
        "obs_id": np.arange(data["n_obs"]),
    }

    with pm.Model(coords=coords) as model:
        # --- Party-level parameters ---
        mu_party_raw = pm.Normal("mu_party_raw", mu=0, sigma=2, shape=n_parties)
        mu_party = pm.Deterministic("mu_party", pt.sort(mu_party_raw), dims="party")

        sigma_within = pm.HalfNormal(
            "sigma_within", sigma=sigma_scale, shape=n_parties, dims="party"
        )

        # --- Non-centered legislator ideal points ---
        xi_offset = pm.Normal("xi_offset", mu=0, sigma=1, shape=n_leg, dims="legislator")
        xi = pm.Deterministic(
            "xi",
            mu_party[party_idx] + sigma_within[party_idx] * xi_offset,
            dims="legislator",
        )

        # --- Bill parameters ---
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")
        beta = beta_prior.build(n_votes)
        print(f"  Beta prior: {beta_prior.describe()}")

        # --- Likelihood ---
        eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

    return model


def compute_pca_initvals(pca_scores: pl.DataFrame, leg_slugs: list[str]) -> np.ndarray:
    """Compute standardized PC1 scores as xi_offset initvals.

    Matches the production code in hierarchical.py:1570-1579:
    - Filter PCA scores to legislators in this chamber
    - Sort to match leg_slugs ordering
    - Standardize to mean=0, std=1 (matches N(0,1) prior on xi_offset)
    """
    slug_order = {s: i for i, s in enumerate(leg_slugs)}
    pc1_vals = (
        pca_scores.filter(pl.col("legislator_slug").is_in(leg_slugs))
        .sort(pl.col("legislator_slug").replace_strict(slug_order))["PC1"]
        .to_numpy()
    )
    pc1_std = (pc1_vals - pc1_vals.mean()) / (pc1_vals.std() + 1e-8)
    return pc1_std.astype(np.float64)


def run_chamber(
    chamber: str,
    run_name: str,
    eda_dir: Path,
    pca_dir: Path,
    hier_dir: Path,
    irt_dir: Path,
    data_dir: Path,
    exp2_dir: Path | None,
    plots_dir: Path,
) -> tuple[dict, dict]:
    """Run nutpie experiment with PCA init for a single chamber.

    Returns (metrics, chamber_result) where chamber_result has keys matching
    build_hierarchical_report expectations: data, idata, ideal_points,
    group_params, icc_df, convergence, sampling_time, flat_corr.
    """
    out_dir = EXPERIMENT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print_header(f"CHAMBER: {chamber}")
    print(f"  Output: {out_dir}")

    # ── Load data ────────────────────────────────────────────────────────
    print_header(f"LOADING DATA — {chamber}")
    house_matrix, senate_matrix, _ = load_eda_matrices(eda_dir)
    _, legislators = load_metadata(data_dir)

    matrix = house_matrix if chamber == "House" else senate_matrix

    # Load PCA scores for initialization
    house_pca, senate_pca = load_pca_scores(pca_dir)
    pca_scores = house_pca if chamber == "House" else senate_pca
    print(f"  PCA scores loaded: {len(pca_scores)} legislators")

    # ── Prepare hierarchical data ────────────────────────────────────────
    print_header(f"PREPARING HIERARCHICAL DATA — {chamber}")
    data = prepare_hierarchical_data(matrix, legislators, chamber)
    print(f"  {data['n_legislators']} legislators x {data['n_votes']} votes")
    print(f"  {data['n_obs']:,} observations")
    print(f"  Parties: {data['party_names']}")

    # ── Compute PCA initvals ──────────────────────────────────────────────
    print_header(f"COMPUTING PCA INITVALS — {chamber}")
    xi_offset_initvals = compute_pca_initvals(pca_scores, data["leg_slugs"])
    print(
        f"  PCA-informed initvals: {len(xi_offset_initvals)} params, "
        f"range [{xi_offset_initvals.min():.2f}, {xi_offset_initvals.max():.2f}]"
    )

    # ── Build model ──────────────────────────────────────────────────────
    print_header(f"BUILDING PYMC MODEL — {chamber}")
    model = build_hierarchical_model(data, PRODUCTION_BETA)
    n_free_params = sum(v.size for v in model.initial_point().values())
    print(f"  Model built: {len(model.free_RVs)} free RVs")
    print(f"  Total free parameters: {n_free_params}")
    print(f"  Free RVs: {[v.name for v in model.free_RVs]}")

    # ── Compile with nutpie (PCA init) ────────────────────────────────────
    print_header(f"COMPILING WITH NUTPIE (Numba + PCA init) — {chamber}")
    import nutpie

    t_compile = time.time()
    try:
        compiled = nutpie.compile_pymc_model(
            model,
            initial_points={"xi_offset": xi_offset_initvals},
            jitter_rvs=set(),  # No jitter — PCA initvals provide orientation (ADR-0045)
        )
        compile_time = time.time() - t_compile
        print(f"  Compilation SUCCESS in {compile_time:.1f}s")
        print("  initial_points: xi_offset from PCA PC1 (standardized)")
        print("  jitter_rvs: disabled (matching production adapt_diag, no jitter)")
    except Exception as e:
        compile_time = time.time() - t_compile
        print(f"  Compilation FAILED after {compile_time:.1f}s")
        print(f"  Error: {e}")
        metrics = {
            "experiment": "nutpie-hierarchical-pca-init",
            "chamber": chamber,
            "phase": "compilation",
            "status": "FAILED",
            "error": str(e),
            "compile_time_s": round(compile_time, 1),
            "n_free_params": n_free_params,
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics, {}

    # ── Sample with nutpie ───────────────────────────────────────────────
    print_header(f"SAMPLING WITH NUTPIE — {chamber}")
    print(f"  {N_SAMPLES} draws, {N_TUNE} tune, {N_CHAINS} chains")
    print(f"  seed={RANDOM_SEED}")
    print("  PCA-informed xi_offset initialization (jitter disabled)")

    t_sample = time.time()
    idata = nutpie.sample(
        compiled,
        draws=N_SAMPLES,
        tune=N_TUNE,
        chains=N_CHAINS,
        seed=RANDOM_SEED,
        progress_bar=True,
        store_divergences=True,
    )
    sample_time = time.time() - t_sample
    print(f"  Sampling complete in {sample_time:.1f}s")

    # ── Convergence diagnostics ──────────────────────────────────────────
    # Production convergence check (populates console output + returns dict)
    convergence = check_hierarchical_convergence(idata, chamber)
    all_ok = convergence.get("all_ok", False)
    divergences = convergence.get("divergences", 0)

    # Also compute manual diagnostics for console detail
    rhat = az.rhat(idata)
    ess = az.ess(idata)
    diag: dict = {}
    available_vars = [v for v in HIER_CONVERGENCE_VARS if v in idata.posterior]
    for var in available_vars:
        if var in rhat:
            diag[f"{var}_rhat_max"] = float(rhat[var].max())
        if var in ess:
            min_ess = float(ess[var].min())
            diag[f"{var}_ess_min"] = min_ess
            diag[f"{var}_ess_per_chain"] = min_ess / N_CHAINS

    bfmi_values = az.bfmi(idata)
    diag["divergences"] = divergences
    diag["ebfmi"] = [round(float(v), 3) for v in bfmi_values]
    for i, v in enumerate(bfmi_values):
        status = "OK" if v > 0.3 else "WARNING"
        print(f"  E-BFMI chain {i}: {v:.3f}  {status}")

    # ── Extract ideal points + production results ──────────────────────────
    print_header(f"EXTRACTING IDEAL POINTS — {chamber}")

    flat_path = irt_dir / "data" / f"ideal_points_{chamber.lower()}.parquet"
    flat_ip = pl.read_parquet(flat_path) if flat_path.exists() else None
    if flat_ip is not None:
        print(f"  Flat IRT baseline loaded: {len(flat_ip)} legislators")
    else:
        print(f"  No flat IRT baseline at {flat_path}")

    ip_df = extract_hierarchical_ideal_points(idata, data, legislators, flat_ip)
    print(f"  Extracted {len(ip_df)} ideal points")

    # Production extraction: group params, ICC, flat correlation
    group_params = extract_group_params(idata, data)
    icc_df = compute_variance_decomposition(idata, data)

    flat_corr = float("nan")
    if flat_ip is not None:
        flat_corr = compute_flat_hier_correlation(ip_df, flat_ip, chamber)

    # Print group params
    print("\n  Group-level parameters:")
    for row in group_params.iter_rows(named=True):
        print(
            f"    {row['party']}: mu={row['mu_mean']:+.3f} "
            f"[{row['mu_hdi_2.5']:+.3f}, {row['mu_hdi_97.5']:+.3f}], "
            f"sigma={row['sigma_within_mean']:.3f}"
        )

    # ── Production plots ─────────────────────────────────────────────────
    print_header(f"PLOTS — {chamber}")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_party_posteriors(idata, data, chamber, plots_dir)
    plot_icc(icc_df, chamber, plots_dir)
    plot_shrinkage_scatter(ip_df, chamber, plots_dir)
    plot_forest(ip_df, chamber, plots_dir)
    plot_dispersion(idata, data, chamber, plots_dir)

    # ── Compare vs experiment 2 (no PCA init) ─────────────────────────────
    r_exp2 = float("nan")
    exp2_path = None
    if exp2_dir is not None:
        run_name_exp2 = "run_01_house" if chamber == "House" else "run_02_senate"
        ip_file = f"hierarchical_ideal_points_{chamber.lower()}.parquet"
        exp2_path = exp2_dir / run_name_exp2 / "data" / ip_file

    if exp2_path is not None and exp2_path.exists():
        print_header(f"COMPARISON VS EXPERIMENT 2 (no PCA init) — {chamber}")
        exp2_baseline = pl.read_parquet(exp2_path)
        xi_pca = ip_df.select("legislator_slug", pl.col("xi_mean").alias("xi_pca"))
        xi_nopca = exp2_baseline.select("legislator_slug", pl.col("xi_mean").alias("xi_nopca"))

        merged_exp2 = xi_pca.join(xi_nopca, on="legislator_slug", how="inner")
        if len(merged_exp2) > 2:
            r_exp2, p_exp2 = stats.pearsonr(
                merged_exp2["xi_pca"].to_numpy(),
                merged_exp2["xi_nopca"].to_numpy(),
            )
            abs_r_exp2 = abs(r_exp2)
            print(f"  Legislators matched: {len(merged_exp2)}")
            print(f"  Pearson r vs exp 2 (no PCA): {r_exp2:.6f} (p={p_exp2:.2e})")
            print(f"  |r| = {abs_r_exp2:.4f}")

            # Scatter plot
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(
                merged_exp2["xi_nopca"].to_numpy(),
                merged_exp2["xi_pca"].to_numpy(),
                alpha=0.6,
                s=20,
            )
            lim = (
                max(
                    abs(merged_exp2["xi_nopca"].to_numpy()).max(),
                    abs(merged_exp2["xi_pca"].to_numpy()).max(),
                )
                * 1.1
            )
            ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3)
            ax.set_xlabel("nutpie (no PCA init) — Experiment 2")
            ax.set_ylabel("nutpie (PCA init) — Experiment 2b")
            ax.set_title(f"PCA Init vs No PCA Init — {chamber} (|r|={abs_r_exp2:.4f})")
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(out_dir / "scatter_pca_vs_nopca.png", dpi=150)
            plt.close(fig)
            print("  Saved: scatter_pca_vs_nopca.png")
    else:
        print("  No experiment 2 baseline available")

    # ── Compare vs PyMC hierarchical baseline ────────────────────────────
    print_header(f"COMPARISON VS PYMC HIERARCHICAL — {chamber}")

    hier_path = hier_dir / "data" / f"hierarchical_ideal_points_{chamber.lower()}.parquet"
    r_hier = float("nan")
    if hier_path.exists():
        hier_baseline = pl.read_parquet(hier_path)
        xi_nutpie = ip_df.select("legislator_slug", pl.col("xi_mean").alias("xi_nutpie"))
        xi_pymc = hier_baseline.select("legislator_slug", pl.col("xi_mean").alias("xi_pymc_hier"))

        merged_hier = xi_nutpie.join(xi_pymc, on="legislator_slug", how="inner")
        if len(merged_hier) > 2:
            r_hier, p_hier = stats.pearsonr(
                merged_hier["xi_nutpie"].to_numpy(),
                merged_hier["xi_pymc_hier"].to_numpy(),
            )
            abs_r_hier = abs(r_hier)
            sign_flip_hier = r_hier < 0
            print(f"  Legislators matched: {len(merged_hier)}")
            print(f"  Pearson r vs PyMC hierarchical: {r_hier:.6f} (p={p_hier:.2e})")
            if sign_flip_hier:
                print("  Sign flip detected (expected — IRT reflection invariance)")
            print(f"  |r| = {abs_r_hier:.4f} (target > 0.95)")

            # Scatter plot
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(
                merged_hier["xi_pymc_hier"].to_numpy(),
                merged_hier["xi_nutpie"].to_numpy(),
                alpha=0.6,
                s=20,
            )
            lim = (
                max(
                    abs(merged_hier["xi_pymc_hier"].to_numpy()).max(),
                    abs(merged_hier["xi_nutpie"].to_numpy()).max(),
                )
                * 1.1
            )
            ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3)
            ax.set_xlabel("PyMC Hierarchical Ideal Points (xi)")
            ax.set_ylabel("nutpie + PCA Init Ideal Points (xi)")
            ax.set_title(f"nutpie+PCA vs PyMC Hierarchical — {chamber} (|r|={abs_r_hier:.4f})")
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(out_dir / "scatter_nutpie_pca_vs_pymc_hier.png", dpi=150)
            plt.close(fig)
            print("  Saved: scatter_nutpie_pca_vs_pymc_hier.png")
        else:
            print(f"  Only {len(merged_hier)} matched — skipping correlation")
    else:
        print(f"  No PyMC hierarchical baseline at {hier_path}")

    # ── Compare vs flat IRT baseline ─────────────────────────────────────
    print_header(f"COMPARISON VS FLAT IRT — {chamber}")

    r_flat = float("nan")
    if flat_ip is not None:
        xi_nutpie = ip_df.select("legislator_slug", pl.col("xi_mean").alias("xi_nutpie"))
        xi_flat = flat_ip.select("legislator_slug", pl.col("xi_mean").alias("xi_flat"))

        merged_flat = xi_nutpie.join(xi_flat, on="legislator_slug", how="inner")
        if len(merged_flat) > 2:
            r_flat, p_flat = stats.pearsonr(
                merged_flat["xi_nutpie"].to_numpy(),
                merged_flat["xi_flat"].to_numpy(),
            )
            abs_r_flat = abs(r_flat)
            sign_flip_flat = r_flat < 0
            print(f"  Legislators matched: {len(merged_flat)}")
            print(f"  Pearson r vs flat IRT: {r_flat:.6f} (p={p_flat:.2e})")
            if sign_flip_flat:
                print("  Sign flip detected (expected — different identification)")
            print(f"  |r| = {abs_r_flat:.4f} (target > 0.90)")

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(
                merged_flat["xi_flat"].to_numpy(),
                merged_flat["xi_nutpie"].to_numpy(),
                alpha=0.6,
                s=20,
            )
            lim = (
                max(
                    abs(merged_flat["xi_flat"].to_numpy()).max(),
                    abs(merged_flat["xi_nutpie"].to_numpy()).max(),
                )
                * 1.1
            )
            ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3)
            ax.set_xlabel("Flat IRT Ideal Points (xi)")
            ax.set_ylabel("nutpie + PCA Init Ideal Points (xi)")
            ax.set_title(f"nutpie+PCA Hierarchical vs Flat IRT — {chamber} (|r|={abs_r_flat:.4f})")
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(out_dir / "scatter_nutpie_pca_vs_flat.png", dpi=150)
            plt.close(fig)
            print("  Saved: scatter_nutpie_pca_vs_flat.png")
        else:
            print(f"  Only {len(merged_flat)} matched — skipping correlation")
    else:
        print("  No flat IRT baseline — skipping comparison")

    # ── Save outputs ─────────────────────────────────────────────────────
    print_header(f"SAVING OUTPUTS — {chamber}")

    data_out = out_dir / "data"
    data_out.mkdir(parents=True, exist_ok=True)
    ip_df.write_parquet(data_out / f"hierarchical_ideal_points_{chamber.lower()}.parquet")
    group_params.write_parquet(data_out / f"group_params_{chamber.lower()}.parquet")
    icc_df.write_parquet(data_out / f"variance_decomposition_{chamber.lower()}.parquet")
    print(f"  Saved: hierarchical_ideal_points_{chamber.lower()}.parquet")

    idata.to_netcdf(str(data_out / f"idata_{chamber.lower()}.nc"))
    print(f"  Saved: idata_{chamber.lower()}.nc")

    # ── Build metrics ────────────────────────────────────────────────────
    abs_r_hier = abs(r_hier) if not np.isnan(r_hier) else None
    abs_r_flat = abs(r_flat) if not np.isnan(r_flat) else None
    abs_r_exp2_val = abs(r_exp2) if not np.isnan(r_exp2) else None

    metrics = {
        "experiment": "nutpie-hierarchical-pca-init",
        "session": SESSION,
        "chamber": chamber,
        "sampler": "nutpie",
        "backend": "numba",
        "pca_init": True,
        "jitter": False,
        "n_legislators": data["n_legislators"],
        "n_votes": data["n_votes"],
        "n_obs": data["n_obs"],
        "n_free_params": n_free_params,
        "n_samples": N_SAMPLES,
        "n_tune": N_TUNE,
        "n_chains": N_CHAINS,
        "seed": RANDOM_SEED,
        "compile_time_s": round(compile_time, 1),
        "sample_time_s": round(sample_time, 1),
        "total_time_s": round(compile_time + sample_time, 1),
        "convergence": {
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in diag.items()},
            "all_ok": all_ok,
        },
        "comparison": {
            "vs_exp2_no_pca": {
                "pearson_r": round(r_exp2, 6) if not np.isnan(r_exp2) else None,
                "abs_r": round(abs_r_exp2_val, 6) if abs_r_exp2_val is not None else None,
            },
            "vs_pymc_hierarchical": {
                "pearson_r": round(r_hier, 6) if not np.isnan(r_hier) else None,
                "abs_r": round(abs_r_hier, 6) if abs_r_hier is not None else None,
                "sign_flip": bool(r_hier < 0) if not np.isnan(r_hier) else None,
                "pass_criteria": "|r| > 0.95",
                "pass": bool(abs_r_hier > 0.95) if abs_r_hier is not None else None,
            },
            "vs_flat_irt": {
                "pearson_r": round(r_flat, 6) if not np.isnan(r_flat) else None,
                "abs_r": round(abs_r_flat, 6) if abs_r_flat is not None else None,
                "sign_flip": bool(r_flat < 0) if not np.isnan(r_flat) else None,
                "pass_criteria": "|r| > 0.90",
                "pass": bool(abs_r_flat > 0.90) if abs_r_flat is not None else None,
            },
        },
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  Saved: metrics.json")

    # ── Summary ──────────────────────────────────────────────────────────
    print_header(f"SUMMARY — {chamber}")
    print(f"  Compile: {compile_time:.1f}s")
    print(f"  Sample:  {sample_time:.1f}s")
    print(f"  Total:   {compile_time + sample_time:.1f}s")
    for var in available_vars:
        rk = f"{var}_rhat_max"
        ek = f"{var}_ess_min"
        if rk in diag:
            print(f"  R-hat({var}) max: {diag[rk]:.4f}")
        if ek in diag:
            print(f"  ESS({var}) min: {diag[ek]:.0f}")
    print(f"  Divergences: {divergences}")
    print(f"  Convergence: {'PASS' if all_ok else 'FAIL'}")
    if abs_r_exp2_val is not None:
        print(f"  |r| vs exp 2 (no PCA): {abs_r_exp2_val:.4f}")
    if abs_r_hier is not None:
        print(f"  |r| vs PyMC hierarchical: {abs_r_hier:.4f}")
    if abs_r_flat is not None:
        print(f"  |r| vs flat IRT: {abs_r_flat:.4f}")

    # ── Build chamber_result dict for build_hierarchical_report ──────────
    chamber_result = {
        "data": data,
        "idata": idata,
        "ideal_points": ip_df,
        "group_params": group_params,
        "icc_df": icc_df,
        "convergence": convergence,
        "sampling_time": sample_time,
        "flat_corr": flat_corr,
    }

    return metrics, chamber_result


def _resolve_upstream(results_root: Path, new_name: str, old_name: str) -> Path:
    """Resolve upstream results directory, trying new numbered name first."""
    new = results_root / new_name / "latest"
    if new.exists():
        return new
    return results_root / old_name / "latest"


def main() -> None:
    ks = KSSession.from_session_string(SESSION)
    eda_dir = _resolve_upstream(ks.results_dir, "01_eda", "eda")
    pca_dir = _resolve_upstream(ks.results_dir, "02_pca", "pca")
    hier_dir = _resolve_upstream(ks.results_dir, "10_hierarchical", "hierarchical")
    irt_dir = _resolve_upstream(ks.results_dir, "04_irt", "irt")

    # Experiment 2 results (for comparison)
    exp2_house = EXPERIMENT_DIR / "run_01_house"
    exp2_dir = EXPERIMENT_DIR if exp2_house.exists() else None

    print("Experiment 2b: nutpie Hierarchical Per-Chamber IRT + PCA Init (Numba)")
    print(f"  Session: {SESSION}")
    print(f"  Sampling: {N_SAMPLES} draws, {N_TUNE} tune, {N_CHAINS} chains")
    print("  PCA-informed xi_offset initialization (jitter disabled)")
    print()

    # Platform check
    platform = PlatformCheck.current()
    warnings = platform.validate(N_CHAINS)
    if warnings:
        for w in warnings:
            print(f"  PLATFORM WARNING: {w}")
        fatal = [w for w in warnings if w.startswith("FATAL")]
        if fatal:
            print("\n  Aborting due to FATAL platform warnings.")
            sys.exit(1)
    else:
        print("  Platform checks: OK")
    print()

    t_total = time.time()
    plots_dir = EXPERIMENT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with ExperimentLifecycle("nutpie-hierarchical-pca-init"):
        all_metrics = {}
        per_chamber_results: dict[str, dict] = {}

        for chamber, run_name in [("House", "run_03_house_pca"), ("Senate", "run_04_senate_pca")]:
            metrics, chamber_result = run_chamber(
                chamber=chamber,
                run_name=run_name,
                eda_dir=eda_dir,
                pca_dir=pca_dir,
                hier_dir=hier_dir,
                irt_dir=irt_dir,
                data_dir=ks.data_dir,
                exp2_dir=exp2_dir,
                plots_dir=plots_dir,
            )
            all_metrics[chamber] = metrics
            if chamber_result:
                per_chamber_results[chamber] = chamber_result

        elapsed = time.time() - t_total

        # ── Final summary ────────────────────────────────────────────────
        print_header("EXPERIMENT 2b COMPLETE")

        for chamber, m in all_metrics.items():
            status = m.get("status", "")
            if status == "FAILED":
                print(f"  {chamber}: COMPILATION FAILED — {m.get('error', 'unknown')}")
                continue

            conv = m.get("convergence", {})
            comp = m.get("comparison", {})
            print(f"  {chamber}:")
            print(f"    Convergence: {'PASS' if conv.get('all_ok') else 'FAIL'}")
            print(f"    Time: {m.get('total_time_s', '?')}s")

            exp2_r = comp.get("vs_exp2_no_pca", {}).get("abs_r")
            hier_r = comp.get("vs_pymc_hierarchical", {}).get("abs_r")
            flat_r = comp.get("vs_flat_irt", {}).get("abs_r")
            if exp2_r is not None:
                print(f"    |r| vs exp 2 (no PCA): {exp2_r:.4f}")
            if hier_r is not None:
                print(f"    |r| vs PyMC hierarchical: {hier_r:.4f}")
            if flat_r is not None:
                print(f"    |r| vs flat IRT: {flat_r:.4f}")

        # Format elapsed time for the report header
        def _fmt_elapsed(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.1f}s"
            minutes, secs = divmod(int(seconds), 60)
            if minutes < 60:
                return f"{minutes}m {secs}s"
            hours, mins = divmod(minutes, 60)
            return f"{hours}h {mins}m {secs}s"

        # ── HTML Report (production build_hierarchical_report) ────────────
        print_header("HTML REPORT")
        report = ReportBuilder(
            title="nutpie Hierarchical IRT + PCA Init — Experiment 2b",
            session=SESSION,
            elapsed_display=_fmt_elapsed(elapsed),
        )
        build_hierarchical_report(
            report,
            chamber_results=per_chamber_results,
            joint_results=None,
            plots_dir=plots_dir,
        )
        report_path = EXPERIMENT_DIR / "experiment_2b_report.html"
        report.write(report_path)
        print(f"\n  HTML Report: {report_path}")

        # Key question
        house_ok = all_metrics.get("House", {}).get("convergence", {}).get("all_ok", False)
        senate_ok = all_metrics.get("Senate", {}).get("convergence", {}).get("all_ok", False)
        if house_ok and senate_ok:
            print("\n  KEY FINDING: nutpie + PCA init RESOLVES convergence for BOTH chambers!")
            print("  -> nutpie is ready for production use with PCA init.")
            print("  -> Experiment 3 (NF) is no longer needed for per-chamber models.")
        elif senate_ok and not house_ok:
            print("\n  UNEXPECTED: PCA init broke House convergence.")
        elif house_ok and not senate_ok:
            print("\n  PCA init did NOT fix Senate convergence.")
            print("  -> Experiment 3 (NF) becomes critical.")
        else:
            print("\n  BOTH chambers failed. Investigate.")


if __name__ == "__main__":
    main()
