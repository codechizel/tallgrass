"""Experiment 1: nutpie flat IRT baseline.

Tests whether nutpie compiles the flat 2PL IRT model cleanly and produces
ideal points that correlate r > 0.99 with the PyMC baseline.

Three phases:
  1. Build the PyMC model (identical to production)
  2. Compile with nutpie (Numba backend)
  3. Sample with nutpie and compare with PyMC baseline

Uses the 91st House (largest chamber, most parameters) as the test case.

Usage:
    uv run python results/experiments/2026-02-27_nutpie-flat-irt/run_experiment.py
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

from analysis.irt import (
    RANDOM_SEED,
    load_eda_matrices,
    load_pca_scores,
    prepare_irt_data,
    select_anchors,
)

from analysis.experiment_monitor import ExperimentLifecycle, PlatformCheck
from tallgrass.session import KSSession

EXPERIMENT_DIR = Path(__file__).parent
SESSION = "2025-26"

# Match production sampling settings from irt.py
N_SAMPLES = 2000
N_TUNE = 1000
N_CHAINS = 2
TARGET_ACCEPT = 0.95


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def build_flat_irt_model(data: dict, anchors: list[tuple[int, float]]):
    """Build the flat 2PL IRT model (identical to production) and return it.

    Does NOT sample — returns the model context for nutpie compilation.
    """
    import pymc as pm
    import pytensor.tensor as pt

    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y = data["y"]
    n_leg = data["n_legislators"]
    n_votes = data["n_votes"]
    n_anchors = len(anchors)
    anchor_indices = {idx for idx, _ in anchors}

    coords = {
        "legislator": data["leg_slugs"],
        "vote": data["vote_ids"],
        "obs_id": np.arange(data["n_obs"]),
    }

    with pm.Model(coords=coords) as model:
        # --- Legislator ideal points with anchors ---
        xi_free = pm.Normal("xi_free", mu=0, sigma=1, shape=n_leg - n_anchors)

        xi_raw = pt.zeros(n_leg)
        for anchor_idx, anchor_val in anchors:
            xi_raw = pt.set_subtensor(xi_raw[anchor_idx], anchor_val)

        free_positions = [i for i in range(n_leg) if i not in anchor_indices]
        for k, pos in enumerate(free_positions):
            xi_raw = pt.set_subtensor(xi_raw[pos], xi_free[k])

        xi = pm.Deterministic("xi", xi_raw, dims="legislator")

        # --- Roll call parameters ---
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")
        beta = pm.Normal("beta", mu=0, sigma=1, shape=n_votes, dims="vote")

        # --- Likelihood ---
        eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

    return model


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
    irt_dir = _resolve_upstream(ks.results_dir, "04_irt", "irt")
    out_dir = EXPERIMENT_DIR / "run_01_house"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Experiment 1: nutpie Flat IRT Baseline")
    print(f"  Session: {SESSION}")
    print(f"  Output: {out_dir}")
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

    with ExperimentLifecycle("nutpie-flat-irt"):
        # Load data
        print_header("LOADING DATA")
        house_matrix, _, _ = load_eda_matrices(eda_dir)
        house_pca, _ = load_pca_scores(pca_dir)

        # Prepare IRT data
        data = prepare_irt_data(house_matrix, "House")
        print(f"  {data['n_legislators']} legislators x {data['n_votes']} votes")
        print(f"  {data['n_obs']:,} observations")

        # Select anchors
        cons_idx, cons_slug, lib_idx, lib_slug = select_anchors(house_pca, house_matrix, "House")
        anchors = [(cons_idx, 1.0), (lib_idx, -1.0)]

        # ── Phase 1: Build Model ────────────────────────────────────────
        print_header("BUILDING PYMC MODEL")
        model = build_flat_irt_model(data, anchors)
        n_free_params = sum(v.size for v in model.initial_point().values())
        print(f"  Model built: {len(model.free_RVs)} free RVs")
        print(f"  Total free parameters: {n_free_params}")
        print(f"  Free RVs: {[v.name for v in model.free_RVs]}")

        # ── Phase 2: Compile with nutpie ────────────────────────────────
        print_header("COMPILING WITH NUTPIE (Numba backend)")
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
            # Save failure metrics
            metrics = {
                "experiment": "nutpie-flat-irt",
                "phase": "compilation",
                "status": "FAILED",
                "error": str(e),
                "compile_time_s": round(compile_time, 1),
                "n_free_params": n_free_params,
            }
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            sys.exit(1)

        # ── Phase 3: Sample with nutpie ─────────────────────────────────
        print_header("SAMPLING WITH NUTPIE")
        print(f"  {N_SAMPLES} draws, {N_TUNE} tune, {N_CHAINS} chains")
        print(f"  seed={RANDOM_SEED}")

        t_sample = time.time()
        idata_nutpie = nutpie.sample(
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

        # ── Phase 4: Diagnostics ────────────────────────────────────────
        print_header("CONVERGENCE DIAGNOSTICS")

        rhat = az.rhat(idata_nutpie)
        ess = az.ess(idata_nutpie)

        xi_rhat_max = float(rhat["xi"].max())
        xi_ess_min = float(ess["xi"].min())
        alpha_rhat_max = float(rhat["alpha"].max())
        beta_rhat_max = float(rhat["beta"].max())

        print(f"  R-hat (xi):    max = {xi_rhat_max:.4f}")
        print(f"  R-hat (alpha): max = {alpha_rhat_max:.4f}")
        print(f"  R-hat (beta):  max = {beta_rhat_max:.4f}")
        print(f"  ESS (xi):      min = {xi_ess_min:.0f}")

        # Divergences
        if "diverging" in idata_nutpie.sample_stats:
            divergences = int(idata_nutpie.sample_stats["diverging"].sum().values)
        else:
            divergences = 0
        print(f"  Divergences: {divergences}")

        # E-BFMI
        bfmi_values = az.bfmi(idata_nutpie)
        for i, v in enumerate(bfmi_values):
            print(f"  E-BFMI chain {i}: {v:.3f}")

        # ── Phase 5: Compare with PyMC baseline ────────────────────────
        print_header("COMPARISON WITH PYMC BASELINE")

        baseline_path = irt_dir / "data" / "ideal_points_house.parquet"
        if baseline_path.exists():
            baseline_ip = pl.read_parquet(baseline_path)

            # Extract nutpie ideal points
            xi_nutpie = idata_nutpie.posterior["xi"].mean(dim=["chain", "draw"]).values
            nutpie_df = pl.DataFrame(
                {
                    "legislator_slug": data["leg_slugs"],
                    "xi_nutpie": xi_nutpie,
                }
            )

            merged = nutpie_df.join(
                baseline_ip.select("legislator_slug", pl.col("xi_mean").alias("xi_pymc")),
                on="legislator_slug",
                how="inner",
            )

            r, p = stats.pearsonr(
                merged["xi_nutpie"].to_numpy(),
                merged["xi_pymc"].to_numpy(),
            )
            abs_r = abs(r)
            sign_flip = r < 0
            print(f"  Legislators matched: {len(merged)}")
            print(f"  Pearson r vs PyMC baseline: {r:.6f} (p={p:.2e})")
            if sign_flip:
                print("  Sign flip detected (expected — IRT reflection invariance)")
            print(f"  Pass criteria (|r| > 0.99): {'PASS' if abs_r > 0.99 else 'FAIL'}")

            # Scatter plot
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(
                merged["xi_pymc"].to_numpy(),
                merged["xi_nutpie"].to_numpy(),
                alpha=0.6,
                s=20,
            )
            ax.plot([-3, 3], [-3, 3], "k--", alpha=0.3)
            ax.set_xlabel("PyMC Ideal Points (xi)")
            ax.set_ylabel("nutpie Ideal Points (xi)")
            ax.set_title(f"nutpie vs PyMC Flat IRT — House (|r|={abs_r:.4f})")
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(out_dir / "scatter_nutpie_vs_pymc.png", dpi=150)
            plt.close(fig)
            print("  Saved: scatter_nutpie_vs_pymc.png")
        else:
            r = float("nan")
            abs_r = float("nan")
            sign_flip = False
            print(f"  No PyMC baseline found at {baseline_path}")

        # ── Save metrics ────────────────────────────────────────────────
        print_header("SUMMARY")
        metrics = {
            "experiment": "nutpie-flat-irt",
            "session": SESSION,
            "chamber": "House",
            "sampler": "nutpie",
            "backend": "numba",
            "n_free_params": n_free_params,
            "n_samples": N_SAMPLES,
            "n_tune": N_TUNE,
            "n_chains": N_CHAINS,
            "seed": RANDOM_SEED,
            "compile_time_s": round(compile_time, 1),
            "sample_time_s": round(sample_time, 1),
            "total_time_s": round(compile_time + sample_time, 1),
            "convergence": {
                "xi_rhat_max": round(xi_rhat_max, 6),
                "alpha_rhat_max": round(alpha_rhat_max, 6),
                "beta_rhat_max": round(beta_rhat_max, 6),
                "xi_ess_min": round(xi_ess_min, 1),
                "divergences": divergences,
                "ebfmi": [round(float(v), 3) for v in bfmi_values],
            },
            "comparison": {
                "pearson_r_vs_pymc": round(r, 6) if not np.isnan(r) else None,
                "abs_r": round(abs_r, 6) if not np.isnan(r) else None,
                "sign_flip": sign_flip if not np.isnan(r) else None,
                "pass_criteria": "|r| > 0.99",
                "pass": bool(abs_r > 0.99) if not np.isnan(r) else None,
            },
        }

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save nutpie trace
        data_dir = out_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        idata_nutpie.to_netcdf(str(data_dir / "idata_nutpie_house.nc"))

        print(f"  Compile: {compile_time:.1f}s")
        print(f"  Sample:  {sample_time:.1f}s")
        print(f"  Total:   {compile_time + sample_time:.1f}s")
        print(f"  R-hat(xi) max: {xi_rhat_max:.4f}")
        print(f"  ESS(xi) min: {xi_ess_min:.0f}")
        if not np.isnan(r):
            print(f"  Pearson r vs PyMC: {r:.6f} (|r|={abs_r:.6f})")
            if sign_flip:
                print("  Sign flip: yes (IRT reflection invariance)")
        print(f"\n  Metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
