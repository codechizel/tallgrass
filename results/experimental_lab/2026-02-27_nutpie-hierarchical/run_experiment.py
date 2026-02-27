"""Experiment 2: nutpie hierarchical per-chamber IRT (Numba).

Tests whether nutpie's Rust NUTS implementation resolves the House convergence
failure that plagues the hierarchical per-chamber IRT model under PyMC.

Two runs:
  1. House (130 legislators, ~694 free params — the problematic chamber)
  2. Senate (40 legislators, ~524 free params — the easy chamber, for comparison)

Each run:
  - Builds the hierarchical model (identical to production, minus sampling)
  - Compiles with nutpie Numba backend
  - Samples with nutpie (no PCA init — let nutpie find the mode from zeros)
  - Checks convergence diagnostics (R-hat, ESS, divergences, E-BFMI)
  - Compares ideal points vs PyMC hierarchical and flat IRT baselines

Usage:
    uv run python results/experiments/2026-02-27_nutpie-hierarchical/run_experiment.py
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
    extract_hierarchical_ideal_points,
    prepare_hierarchical_data,
)
from analysis.irt import (
    ESS_THRESHOLD,
    RANDOM_SEED,
    RHAT_THRESHOLD,
    load_eda_matrices,
    load_metadata,
)
from analysis.model_spec import PRODUCTION_BETA

from analysis.experiment_monitor import ExperimentLifecycle, PlatformCheck
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


def run_chamber(
    chamber: str,
    run_name: str,
    eda_dir: Path,
    hier_dir: Path,
    irt_dir: Path,
    data_dir: Path,
) -> dict:
    """Run nutpie experiment for a single chamber. Returns metrics dict."""
    out_dir = EXPERIMENT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print_header(f"CHAMBER: {chamber}")
    print(f"  Output: {out_dir}")

    # ── Load data ────────────────────────────────────────────────────────
    print_header(f"LOADING DATA — {chamber}")
    house_matrix, senate_matrix, _ = load_eda_matrices(eda_dir)
    _, legislators = load_metadata(data_dir)

    matrix = house_matrix if chamber == "House" else senate_matrix

    # ── Prepare hierarchical data ────────────────────────────────────────
    print_header(f"PREPARING HIERARCHICAL DATA — {chamber}")
    data = prepare_hierarchical_data(matrix, legislators, chamber)
    print(f"  {data['n_legislators']} legislators x {data['n_votes']} votes")
    print(f"  {data['n_obs']:,} observations")
    print(f"  Parties: {data['party_names']}")

    # ── Build model ──────────────────────────────────────────────────────
    print_header(f"BUILDING PYMC MODEL — {chamber}")
    model = build_hierarchical_model(data, PRODUCTION_BETA)
    n_free_params = sum(v.size for v in model.initial_point().values())
    print(f"  Model built: {len(model.free_RVs)} free RVs")
    print(f"  Total free parameters: {n_free_params}")
    print(f"  Free RVs: {[v.name for v in model.free_RVs]}")

    # ── Compile with nutpie ──────────────────────────────────────────────
    print_header(f"COMPILING WITH NUTPIE (Numba) — {chamber}")
    import nutpie

    t_compile = time.time()
    try:
        compiled = nutpie.compile_pymc_model(model)
        compile_time = time.time() - t_compile
        print(f"  Compilation SUCCESS in {compile_time:.1f}s")
    except Exception as e:
        compile_time = time.time() - t_compile
        print(f"  Compilation FAILED after {compile_time:.1f}s")
        print(f"  Error: {e}")
        metrics = {
            "experiment": "nutpie-hierarchical",
            "chamber": chamber,
            "phase": "compilation",
            "status": "FAILED",
            "error": str(e),
            "compile_time_s": round(compile_time, 1),
            "n_free_params": n_free_params,
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    # ── Sample with nutpie ───────────────────────────────────────────────
    print_header(f"SAMPLING WITH NUTPIE — {chamber}")
    print(f"  {N_SAMPLES} draws, {N_TUNE} tune, {N_CHAINS} chains")
    print(f"  seed={RANDOM_SEED}")
    print("  No PCA init — letting nutpie find the mode from zeros")

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
    print_header(f"CONVERGENCE DIAGNOSTICS — {chamber}")

    rhat = az.rhat(idata)
    ess = az.ess(idata)
    diag: dict = {}

    available_vars = [v for v in HIER_CONVERGENCE_VARS if v in idata.posterior]
    for var in available_vars:
        if var in rhat:
            max_rhat = float(rhat[var].max())
            diag[f"{var}_rhat_max"] = max_rhat
            status = "OK" if max_rhat < RHAT_THRESHOLD else "WARNING"
            print(f"  R-hat ({var}): max = {max_rhat:.4f}  {status}")

    for var in available_vars:
        if var in ess:
            min_ess = float(ess[var].min())
            per_chain = min_ess / N_CHAINS
            diag[f"{var}_ess_min"] = min_ess
            diag[f"{var}_ess_per_chain"] = per_chain
            status = "OK" if min_ess > ESS_THRESHOLD else "WARNING"
            print(f"  ESS ({var}): min = {min_ess:.0f}  {status}  (per-chain: {per_chain:.0f})")

    # Divergences
    if "diverging" in idata.sample_stats:
        divergences = int(idata.sample_stats["diverging"].sum().values)
    else:
        divergences = 0
    diag["divergences"] = divergences
    print(f"  Divergences: {divergences}")

    # E-BFMI
    bfmi_values = az.bfmi(idata)
    diag["ebfmi"] = [round(float(v), 3) for v in bfmi_values]
    for i, v in enumerate(bfmi_values):
        status = "OK" if v > 0.3 else "WARNING"
        print(f"  E-BFMI chain {i}: {v:.3f}  {status}")

    # Overall convergence assessment
    rhat_ok = all(
        diag.get(f"{v}_rhat_max", 0) < RHAT_THRESHOLD
        for v in available_vars
        if f"{v}_rhat_max" in diag
    )
    ess_ok = all(
        diag.get(f"{v}_ess_min", float("inf")) > ESS_THRESHOLD
        for v in available_vars
        if f"{v}_ess_min" in diag
    )
    div_ok = divergences == 0
    bfmi_ok = all(v > 0.3 for v in bfmi_values)
    all_ok = rhat_ok and ess_ok and div_ok and bfmi_ok

    if all_ok:
        print("  CONVERGENCE: ALL CHECKS PASSED")
    else:
        print("  CONVERGENCE: SOME CHECKS FAILED — inspect diagnostics")

    # ── Extract ideal points ─────────────────────────────────────────────
    print_header(f"EXTRACTING IDEAL POINTS — {chamber}")

    # Load flat IRT baseline for shrinkage comparison
    flat_path = irt_dir / "data" / f"ideal_points_{chamber.lower()}.parquet"
    flat_ip = pl.read_parquet(flat_path) if flat_path.exists() else None
    if flat_ip is not None:
        print(f"  Flat IRT baseline loaded: {len(flat_ip)} legislators")
    else:
        print(f"  No flat IRT baseline at {flat_path}")

    ip_df = extract_hierarchical_ideal_points(idata, data, legislators, flat_ip)
    print(f"  Extracted {len(ip_df)} ideal points")

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
            ax.set_ylabel("nutpie Hierarchical Ideal Points (xi)")
            ax.set_title(f"nutpie vs PyMC Hierarchical IRT — {chamber} (|r|={abs_r_hier:.4f})")
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(out_dir / "scatter_nutpie_vs_pymc_hier.png", dpi=150)
            plt.close(fig)
            print("  Saved: scatter_nutpie_vs_pymc_hier.png")
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

            # Scatter plot
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
            ax.set_ylabel("nutpie Hierarchical Ideal Points (xi)")
            ax.set_title(f"nutpie Hierarchical vs Flat IRT — {chamber} (|r|={abs_r_flat:.4f})")
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(out_dir / "scatter_nutpie_vs_flat.png", dpi=150)
            plt.close(fig)
            print("  Saved: scatter_nutpie_vs_flat.png")
        else:
            print(f"  Only {len(merged_flat)} matched — skipping correlation")
    else:
        print("  No flat IRT baseline — skipping comparison")

    # ── Save outputs ─────────────────────────────────────────────────────
    print_header(f"SAVING OUTPUTS — {chamber}")

    # Save ideal points
    data_out = out_dir / "data"
    data_out.mkdir(parents=True, exist_ok=True)
    ip_df.write_parquet(data_out / f"hierarchical_ideal_points_{chamber.lower()}.parquet")
    print(f"  Saved: hierarchical_ideal_points_{chamber.lower()}.parquet")

    # Save NetCDF trace
    idata.to_netcdf(str(data_out / f"idata_{chamber.lower()}.nc"))
    print(f"  Saved: idata_{chamber.lower()}.nc")

    # ── Build metrics ────────────────────────────────────────────────────
    abs_r_hier = abs(r_hier) if not np.isnan(r_hier) else None
    abs_r_flat = abs(r_flat) if not np.isnan(r_flat) else None

    metrics = {
        "experiment": "nutpie-hierarchical",
        "session": SESSION,
        "chamber": chamber,
        "sampler": "nutpie",
        "backend": "numba",
        "n_legislators": data["n_legislators"],
        "n_votes": data["n_votes"],
        "n_obs": data["n_obs"],
        "n_free_params": n_free_params,
        "n_samples": N_SAMPLES,
        "n_tune": N_TUNE,
        "n_chains": N_CHAINS,
        "seed": RANDOM_SEED,
        "pca_init": False,
        "compile_time_s": round(compile_time, 1),
        "sample_time_s": round(sample_time, 1),
        "total_time_s": round(compile_time + sample_time, 1),
        "convergence": {
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in diag.items()},
            "all_ok": all_ok,
        },
        "comparison": {
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
    if abs_r_hier is not None:
        print(f"  |r| vs PyMC hierarchical: {abs_r_hier:.4f}")
    if abs_r_flat is not None:
        print(f"  |r| vs flat IRT: {abs_r_flat:.4f}")

    return metrics


def _resolve_upstream(results_root: Path, new_name: str, old_name: str) -> Path:
    """Resolve upstream results directory, trying new numbered name first."""
    new = results_root / new_name / "latest"
    if new.exists():
        return new
    return results_root / old_name / "latest"


def main() -> None:
    ks = KSSession.from_session_string(SESSION)
    eda_dir = _resolve_upstream(ks.results_dir, "01_eda", "eda")
    hier_dir = _resolve_upstream(ks.results_dir, "10_hierarchical", "hierarchical")
    irt_dir = _resolve_upstream(ks.results_dir, "04_irt", "irt")

    print("Experiment 2: nutpie Hierarchical Per-Chamber IRT (Numba)")
    print(f"  Session: {SESSION}")
    print(f"  Sampling: {N_SAMPLES} draws, {N_TUNE} tune, {N_CHAINS} chains")
    print("  No PCA init — nutpie finds the mode from zeros")
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

    with ExperimentLifecycle("nutpie-hierarchical"):
        all_metrics = {}

        for chamber, run_name in [("House", "run_01_house"), ("Senate", "run_02_senate")]:
            metrics = run_chamber(
                chamber=chamber,
                run_name=run_name,
                eda_dir=eda_dir,
                hier_dir=hier_dir,
                irt_dir=irt_dir,
                data_dir=ks.data_dir,
            )
            all_metrics[chamber] = metrics

        # ── Final summary ────────────────────────────────────────────────
        print_header("EXPERIMENT 2 COMPLETE")
        for chamber, m in all_metrics.items():
            status = m.get("status", "")
            if status == "FAILED":
                print(f"  {chamber}: COMPILATION FAILED — {m.get('error', 'unknown')}")
            else:
                conv = m.get("convergence", {})
                print(f"  {chamber}:")
                print(f"    Convergence: {'PASS' if conv.get('all_ok') else 'FAIL'}")
                print(f"    Time: {m.get('total_time_s', '?')}s")
                comp = m.get("comparison", {})
                hier_r = comp.get("vs_pymc_hierarchical", {}).get("abs_r")
                flat_r = comp.get("vs_flat_irt", {}).get("abs_r")
                if hier_r is not None:
                    print(f"    |r| vs PyMC hierarchical: {hier_r:.4f}")
                if flat_r is not None:
                    print(f"    |r| vs flat IRT: {flat_r:.4f}")

        # Key question
        house_conv = all_metrics.get("House", {}).get("convergence", {})
        if house_conv.get("all_ok"):
            print("\n  KEY FINDING: nutpie RESOLVES the House convergence failure!")
            print("  → Experiment 3 (NF) may not be needed for per-chamber.")
        else:
            print("\n  KEY FINDING: nutpie does NOT resolve House convergence with standard NUTS.")
            print("  → Experiment 3 (NF adaptation) becomes critical.")


if __name__ == "__main__":
    main()
