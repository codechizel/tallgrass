"""
Kansas Legislature — Hierarchical Bayesian IRT Ideal Point Estimation

Extends the flat IRT model (Phase 3) with partial pooling by party. Legislators
with sparse voting records are shrunk toward their party mean. Variance
decomposition quantifies how much party explains. Shrinkage comparison shows what
changes vs. the flat model.

Two models:
- Per-chamber (primary): 2-level model, run separately for House and Senate
- Joint cross-chamber (secondary): 3-level model with both chambers, skipped with --skip-joint

Uses method 16 from Analytic_Methods/16_BAY_hierarchical_legislator_model.md.

Usage:
  uv run python analysis/hierarchical.py [--session 2025-26] [--skip-joint]
      [--n-samples 2000] [--n-tune 1500] [--n-chains 2]

Outputs (in results/<session>/hierarchical/<date>/):
  - data/:   Parquet files (ideal points, group params, variance decomp) + NetCDF
  - plots/:  PNG visualizations (party posteriors, ICC, shrinkage, forest, dispersion)
  - filtering_manifest.json, run_info.json, run_log.txt
  - hierarchical_report.html
"""

from __future__ import annotations

import argparse
import json
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

try:
    from analysis.run_context import RunContext
except ModuleNotFoundError:
    from run_context import RunContext

try:
    from analysis.hierarchical_report import build_hierarchical_report
except ModuleNotFoundError:
    from hierarchical_report import build_hierarchical_report  # type: ignore[no-redef]

try:
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
        prepare_irt_data,
    )
except ModuleNotFoundError:
    from irt import (  # type: ignore[no-redef]
        ESS_THRESHOLD,
        MAX_DIVERGENCES,
        PARTY_COLORS,
        RANDOM_SEED,
        RHAT_THRESHOLD,
        load_eda_matrices,
        load_metadata,
        load_pca_scores,
        plot_forest,
        prepare_irt_data,
    )

# ── Primer ───────────────────────────────────────────────────────────────────

HIERARCHICAL_PRIMER = """\
# Hierarchical Bayesian IRT Ideal Point Estimation

## Purpose

Extends the flat IRT model with **partial pooling by party**. Instead of
treating each legislator as an independent draw from Normal(0, 1), the
hierarchical model nests legislators within parties: each party has its own
mean ideal point, and individual legislators are drawn from their party's
distribution.

This matters because:
- Legislators with few votes get **shrunk toward their party mean**, producing
  more reliable estimates
- The model quantifies **how much party explains** (variance decomposition)
- Comparing with the flat IRT shows **who moved and by how much**

Covers analytic method 16 from `Analytic_Methods/`.

## Method

### 2-Level Hierarchical IRT (Per-Chamber)

```
mu_party_raw ~ Normal(0, 2)           -- party mean ideal points (2 parties)
mu_party     = sort(mu_party_raw)     -- ordering: D < R
sigma_within ~ HalfNormal(1)          -- within-party SD (per party)

xi_offset_i  ~ Normal(0, 1)           -- non-centered offset
xi_i         = mu_party[p_i] + sigma_within[p_i] * xi_offset_i

alpha_j ~ Normal(0, 5)                -- bill difficulty
beta_j  ~ Normal(0, 1)                -- bill discrimination

P(Yea) = logit^-1(beta_j * xi_i - alpha_j)
```

**Identification:** Ordering constraint via `sort(mu_party)` ensures
Democrat < Republican on the ideological scale.

**Non-centered parameterization** avoids the "funnel of hell" — a geometry
problem that makes hierarchical models hard to sample.

### 3-Level Joint Model (Secondary)

When both chambers are combined:
```
mu_global     ~ Normal(0, 2)
sigma_chamber ~ HalfNormal(1)
mu_chamber    = mu_global + sigma_chamber * offset_chamber

mu_group      = mu_chamber[c] + sigma_party * offset_party
sigma_within  ~ HalfNormal(1)
xi            = mu_group[g_i] + sigma_within[g_i] * xi_offset_i
```

## Inputs

Reads from upstream phases:
- `results/<session>/eda/latest/data/` — filtered vote matrices
- `results/<session>/pca/latest/data/` — PCA scores for anchor selection
- `results/<session>/irt/latest/data/` — flat IRT ideal points (for shrinkage comparison)
- `data/<session>/` — rollcalls and legislators CSVs

## Outputs

### `data/` — Parquet intermediates + NetCDF posteriors

| File | Description |
|------|-------------|
| `hierarchical_ideal_points_{chamber}.parquet` | Ideal points with shrinkage vs flat |
| `group_params_{chamber}.parquet` | Party-level mu, sigma posteriors |
| `variance_decomposition_{chamber}.parquet` | ICC with uncertainty |
| `idata_{chamber}.nc` | Full posterior (ArviZ NetCDF) |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `party_posteriors_{chamber}.png` | "Where Do the Parties Stand?" |
| `icc_{chamber}.png` | "How Much Does Party Explain?" |
| `shrinkage_scatter_{chamber}.png` | "How Does Accounting for Party Change Estimates?" |
| `forest_{chamber}.png` | Forest plot with hierarchical ideal points |
| `dispersion_{chamber}.png` | "Which Party Has More Internal Disagreement?" |

## Interpretation Guide

- **Party posterior KDEs**: Separation = polarization. Overlap = moderate overlap.
- **ICC**: 0.7 means party explains 70% of ideological variance.
- **Shrinkage scatter**: Points off the diagonal moved. Arrows show direction.
- **Forest plot**: Same as flat IRT but with hierarchical ideal points.
- **Dispersion**: Wider curve = more internal disagreement within that party.

## Caveats

- Non-centered parameterization may be slightly less efficient than centered for
  large, well-identified groups. The safety margin is worth the minor cost.
- Joint model may not converge — use `--skip-joint` if it fails.
- Small Senate-D group (~10 legislators) produces wide credible intervals on the
  Democratic party mean in the Senate.
"""

# ── Constants ────────────────────────────────────────────────────────────────

HIER_N_SAMPLES = 2000
HIER_N_TUNE = 1500
HIER_N_CHAINS = 2
HIER_TARGET_ACCEPT = 0.95

# Party index convention: 0 = Democrat, 1 = Republican (after sorting)
PARTY_NAMES = ["Democrat", "Republican"]
PARTY_IDX_MAP = {"Democrat": 0, "Republican": 1}


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Hierarchical Bayesian IRT")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override flat IRT results directory")
    parser.add_argument(
        "--n-samples", type=int, default=HIER_N_SAMPLES, help="MCMC samples per chain"
    )
    parser.add_argument(
        "--n-tune", type=int, default=HIER_N_TUNE, help="MCMC tuning samples (discarded)"
    )
    parser.add_argument("--n-chains", type=int, default=HIER_N_CHAINS, help="Number of MCMC chains")
    parser.add_argument("--skip-joint", action="store_true", help="Skip joint cross-chamber model")
    return parser.parse_args()


def print_header(title: str) -> None:
    width = 80
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Phase 1: Prepare Hierarchical Data ──────────────────────────────────────


def prepare_hierarchical_data(
    matrix: pl.DataFrame,
    legislators: pl.DataFrame,
    chamber: str,
) -> dict:
    """Extend flat IRT data with party indices for hierarchical model.

    Independent legislators are excluded from the hierarchical model because
    partial pooling by party requires party membership. They still appear in
    the flat IRT results.

    Returns the same dict as prepare_irt_data() plus:
    - party_idx: array mapping each legislator to their party index (0=D, 1=R)
    - party_names: list of party names in index order
    - n_parties: number of parties (2)
    - n_excluded: number of legislators excluded (non-major-party)
    """
    # Filter out non-major-party legislators (e.g. Independent) before IRT prep
    major_party_slugs = set(
        legislators.filter(pl.col("party").is_in(PARTY_NAMES))["slug"].to_list()
    )
    all_slugs = set(matrix["legislator_slug"].to_list())
    non_major = all_slugs - major_party_slugs
    if non_major:
        print(f"  Excluding {len(non_major)} non-major-party legislators: {sorted(non_major)}")
        matrix = matrix.filter(~pl.col("legislator_slug").is_in(non_major))

    data = prepare_irt_data(matrix, chamber)
    data["n_excluded"] = len(non_major)

    # Map legislator slugs to parties
    meta = legislators.select("slug", "party").unique(subset=["slug"])
    slug_to_party = dict(zip(meta["slug"].to_list(), meta["party"].to_list()))

    party_idx = np.array(
        [PARTY_IDX_MAP[slug_to_party[s]] for s in data["leg_slugs"]],
        dtype=np.int64,
    )

    data["party_idx"] = party_idx
    data["party_names"] = PARTY_NAMES
    data["n_parties"] = len(PARTY_NAMES)

    # Print party composition
    for i, name in enumerate(PARTY_NAMES):
        count = int((party_idx == i).sum())
        print(f"  {name}: {count} legislators")

    return data


# ── Phase 2: Build and Sample Models ────────────────────────────────────────


def build_per_chamber_model(
    data: dict,
    n_samples: int,
    n_tune: int,
    n_chains: int,
    target_accept: float = HIER_TARGET_ACCEPT,
) -> tuple[az.InferenceData, float]:
    """Build 2-level hierarchical IRT and sample with NUTS.

    Model structure:
        mu_party (sorted) → xi (non-centered) → likelihood
        sigma_within (per party) controls within-party spread

    Returns (InferenceData, sampling_time_seconds).
    """
    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y = data["y"]
    n_leg = data["n_legislators"]
    n_votes = data["n_votes"]
    party_idx = data["party_idx"]
    n_parties = data["n_parties"]

    coords = {
        "legislator": data["leg_slugs"],
        "vote": data["vote_ids"],
        "party": data["party_names"],
        "obs_id": np.arange(data["n_obs"]),
    }

    with pm.Model(coords=coords):
        # --- Party-level parameters ---
        # Raw party means, then sort for identification (D < R)
        mu_party_raw = pm.Normal("mu_party_raw", mu=0, sigma=2, shape=n_parties)
        mu_party = pm.Deterministic("mu_party", pt.sort(mu_party_raw), dims="party")

        # Per-party within-group standard deviation
        sigma_within = pm.HalfNormal("sigma_within", sigma=1, shape=n_parties, dims="party")

        # --- Non-centered legislator ideal points ---
        xi_offset = pm.Normal("xi_offset", mu=0, sigma=1, shape=n_leg, dims="legislator")
        xi = pm.Deterministic(
            "xi",
            mu_party[party_idx] + sigma_within[party_idx] * xi_offset,
            dims="legislator",
        )

        # --- Bill parameters ---
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")
        beta = pm.Normal("beta", mu=0, sigma=1, shape=n_votes, dims="vote")

        # --- Likelihood ---
        eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

        # --- Sample ---
        print(f"  Sampling: {n_samples} draws, {n_tune} tune, {n_chains} chains")
        print(f"  target_accept={target_accept}, seed={RANDOM_SEED}")
        t0 = time.time()
        idata = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=RANDOM_SEED,
            progressbar=True,
        )
        sampling_time = time.time() - t0

    print(f"  Sampling complete in {sampling_time:.1f}s")
    return idata, sampling_time


def build_joint_model(
    house_data: dict,
    senate_data: dict,
    n_samples: int,
    n_tune: int,
    n_chains: int,
    target_accept: float = HIER_TARGET_ACCEPT,
) -> tuple[az.InferenceData, dict, float]:
    """Build 3-level joint cross-chamber hierarchical IRT model.

    Model structure:
        mu_global → mu_group (4 groups: HD, HR, SD, SR) → xi → likelihood

    Combines House and Senate data into a single model with shared bill parameters
    where bills appear in both chambers.

    Returns (InferenceData, combined_data_dict, sampling_time_seconds).
    """
    # Combine legislator data
    n_house = house_data["n_legislators"]
    n_senate = senate_data["n_legislators"]
    n_leg = n_house + n_senate

    all_slugs = house_data["leg_slugs"] + senate_data["leg_slugs"]
    all_vote_ids = list(dict.fromkeys(house_data["vote_ids"] + senate_data["vote_ids"]))
    n_votes = len(all_vote_ids)
    vote_to_idx = {v: i for i, v in enumerate(all_vote_ids)}

    # Remap indices for combined model
    senate_leg_offset = n_house

    combined_leg_idx = np.concatenate(
        [
            house_data["leg_idx"],
            senate_data["leg_idx"] + senate_leg_offset,
        ]
    )
    combined_vote_idx = np.concatenate(
        [
            np.array([vote_to_idx[house_data["vote_ids"][v]] for v in house_data["vote_idx"]]),
            np.array([vote_to_idx[senate_data["vote_ids"][v]] for v in senate_data["vote_idx"]]),
        ]
    )
    combined_y = np.concatenate([house_data["y"], senate_data["y"]])

    # Group indices: 0=House-D, 1=House-R, 2=Senate-D, 3=Senate-R
    group_names = ["House Democrat", "House Republican", "Senate Democrat", "Senate Republican"]
    n_groups = len(group_names)
    group_idx = np.concatenate(
        [
            house_data["party_idx"],  # 0=D, 1=R for House
            senate_data["party_idx"] + 2,  # 2=D, 3=R for Senate
        ]
    )

    # Chamber index for each group (0=House, 1=Senate)
    group_chamber = np.array([0, 0, 1, 1], dtype=np.int64)

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
        # Use ordering constraint within each chamber (D < R) for identification,
        # mirroring the per-chamber model's pt.sort(mu_party_raw) approach.
        sigma_party = pm.HalfNormal("sigma_party", sigma=1)
        group_offset_raw = pm.Normal(
            "group_offset_raw", mu=0, sigma=1, shape=n_groups, dims="group"
        )
        # Sort each chamber's pair so D < R: indices [0,1] for House, [2,3] for Senate
        house_pair = pt.sort(group_offset_raw[:2])
        senate_pair = pt.sort(group_offset_raw[2:])
        group_offset_sorted = pt.concatenate([house_pair, senate_pair])
        mu_group = pm.Deterministic(
            "mu_group",
            mu_chamber[group_chamber] + sigma_party * group_offset_sorted,
            dims="group",
        )

        # --- Within-group ---
        sigma_within = pm.HalfNormal("sigma_within", sigma=1, shape=n_groups, dims="group")

        # --- Non-centered legislator ideal points ---
        xi_offset = pm.Normal("xi_offset", mu=0, sigma=1, shape=n_leg, dims="legislator")
        xi = pm.Deterministic(
            "xi",
            mu_group[group_idx] + sigma_within[group_idx] * xi_offset,
            dims="legislator",
        )

        # --- Bill parameters ---
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")
        beta = pm.Normal("beta", mu=0, sigma=1, shape=n_votes, dims="vote")

        # --- Likelihood ---
        eta = beta[combined_vote_idx] * xi[combined_leg_idx] - alpha[combined_vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=combined_y, dims="obs_id")

        # --- Sample ---
        print(f"  Joint: {n_samples} draws, {n_tune} tune, {n_chains} chains")
        print(f"  {n_leg} legislators, {n_votes} votes, {n_obs} observations")
        print(f"  target_accept={target_accept}, seed={RANDOM_SEED}")
        t0 = time.time()
        idata = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
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
        "n_obs": n_obs,
        "group_idx": group_idx,
        "group_names": group_names,
        "n_groups": n_groups,
        "group_chamber": group_chamber,
        "n_house": n_house,
        "n_senate": n_senate,
    }

    print(f"  Joint sampling complete in {sampling_time:.1f}s")
    return idata, combined_data, sampling_time


# ── Phase 3: Convergence Diagnostics ────────────────────────────────────────


def check_hierarchical_convergence(idata: az.InferenceData, chamber: str) -> dict:
    """Check convergence for hierarchical model (xi + mu_party + sigma_within).

    Returns dict with all diagnostic metrics.
    """
    print_header(f"CONVERGENCE — {chamber}")

    diag: dict = {}

    # Variables to check
    var_names = ["xi", "mu_party", "sigma_within", "alpha", "beta"]
    available_vars = [v for v in var_names if v in idata.posterior]

    # Check also joint-specific variables
    for extra in ["mu_group", "mu_chamber", "sigma_chamber", "sigma_party"]:
        if extra in idata.posterior:
            available_vars.append(extra)

    # R-hat
    rhat = az.rhat(idata)
    for var in available_vars:
        if var in rhat:
            max_rhat = float(rhat[var].max())
            diag[f"{var}_rhat_max"] = max_rhat
            status = "OK" if max_rhat < RHAT_THRESHOLD else "WARNING"
            print(f"  R-hat ({var}): max = {max_rhat:.4f}  {status}")

    # ESS
    ess = az.ess(idata)
    for var in available_vars:
        if var in ess:
            min_ess = float(ess[var].min())
            diag[f"{var}_ess_min"] = min_ess
            status = "OK" if min_ess > ESS_THRESHOLD else "WARNING"
            print(f"  ESS ({var}): min = {min_ess:.0f}  {status}")

    # Divergences
    divergences = int(idata.sample_stats["diverging"].sum().values)
    diag["divergences"] = divergences
    div_ok = divergences < MAX_DIVERGENCES
    print(f"  Divergences: {divergences}  {'OK' if div_ok else 'WARNING'}")

    # E-BFMI
    bfmi_values = az.bfmi(idata)
    diag["ebfmi"] = [float(v) for v in bfmi_values]
    bfmi_ok = all(v > 0.3 for v in bfmi_values)
    for i, v in enumerate(bfmi_values):
        print(f"  E-BFMI chain {i}: {v:.3f}  {'OK' if v > 0.3 else 'WARNING'}")

    # Overall assessment
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
    diag["all_ok"] = rhat_ok and ess_ok and div_ok and bfmi_ok
    if diag["all_ok"]:
        print("  CONVERGENCE: ALL CHECKS PASSED")
    else:
        print("  CONVERGENCE: SOME CHECKS FAILED — inspect diagnostics")

    return diag


# ── Phase 4: Extract Results ────────────────────────────────────────────────


def extract_hierarchical_ideal_points(
    idata: az.InferenceData,
    data: dict,
    legislators: pl.DataFrame,
    flat_ip: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Extract posterior summaries with shrinkage comparison to flat IRT.

    Returns DataFrame with legislator_slug, xi_mean, xi_sd, xi_hdi_*,
    party_mean, shrinkage_pct, delta_from_flat, plus metadata.
    """
    xi_posterior = idata.posterior["xi"]
    xi_mean = xi_posterior.mean(dim=["chain", "draw"]).values
    xi_sd = xi_posterior.std(dim=["chain", "draw"]).values
    xi_hdi = az.hdi(idata, var_names=["xi"], hdi_prob=0.95)["xi"].values

    # Group/party means — joint model uses mu_group, per-chamber uses mu_party
    if "mu_group" in idata.posterior:
        mu_mean = idata.posterior["mu_group"].mean(dim=["chain", "draw"]).values
        group_idx = data.get("group_idx", data.get("party_idx"))
    else:
        mu_mean = idata.posterior["mu_party"].mean(dim=["chain", "draw"]).values
        group_idx = data["party_idx"]

    slugs = data["leg_slugs"]

    rows = []
    for i, slug in enumerate(slugs):
        party_mean = float(mu_mean[group_idx[i]])
        rows.append(
            {
                "legislator_slug": slug,
                "xi_mean": float(xi_mean[i]),
                "xi_sd": float(xi_sd[i]),
                "xi_hdi_2.5": float(xi_hdi[i, 0]),
                "xi_hdi_97.5": float(xi_hdi[i, 1]),
                "party_mean": party_mean,
            }
        )

    df = pl.DataFrame(rows)

    # Join legislator metadata
    meta = legislators.select("slug", "full_name", "party", "district", "chamber")
    df = df.join(meta, left_on="legislator_slug", right_on="slug", how="left")

    # Shrinkage comparison with flat IRT
    if flat_ip is not None:
        flat_cols = flat_ip.select(
            pl.col("legislator_slug"),
            pl.col("xi_mean").alias("flat_xi_mean"),
            pl.col("xi_sd").alias("flat_xi_sd"),
        )
        df = df.join(flat_cols, on="legislator_slug", how="left")

        # Rescale flat estimates to hierarchical scale via linear regression.
        # The two models produce ideal points on different scales (flat ≈ [-4,3],
        # hier ≈ [-11,9]) because their identification constraints differ.
        # A linear transform (slope, intercept) maps flat → hier for comparison.
        matched = df.filter(pl.col("flat_xi_mean").is_not_null() & pl.col("xi_mean").is_not_null())
        if len(matched) > 2:
            flat_vals = matched["flat_xi_mean"].to_numpy()
            hier_vals = matched["xi_mean"].to_numpy()
            slope, intercept = np.polyfit(flat_vals, hier_vals, 1)
            df = df.with_columns(
                (pl.col("flat_xi_mean") * slope + intercept).alias("flat_xi_rescaled"),
            )
        else:
            slope = 1.0
            df = df.with_columns(
                pl.col("flat_xi_mean").alias("flat_xi_rescaled"),
            )

        df = df.with_columns(
            [
                # Delta in hierarchical scale (flat rescaled to match)
                (pl.col("xi_mean") - pl.col("flat_xi_rescaled")).alias("delta_from_flat"),
            ]
        )

        # Determine if shrinkage is toward party mean (using rescaled flat)
        flat_dist = (pl.col("flat_xi_rescaled") - pl.col("party_mean")).abs()
        hier_dist = (pl.col("xi_mean") - pl.col("party_mean")).abs()
        df = df.with_columns(
            [
                (hier_dist < flat_dist).alias("toward_party_mean"),
                # Shrinkage = fraction of flat's distance to party mean absorbed by pooling.
                # 100% = fully pooled to party mean. 0% = no change. Negative = moved away.
                # Null when flat_dist < 0.5 (ratio unstable near party mean) or anchored.
                pl.when((pl.col("flat_xi_sd") > 0.01) & (flat_dist > 0.5))
                .then((1 - hier_dist / flat_dist) * 100)
                .otherwise(None)
                .alias("shrinkage_pct"),
            ]
        )

        # Drop the intermediate rescaled column
        df = df.drop("flat_xi_rescaled")
    else:
        df = df.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("flat_xi_mean"),
                pl.lit(None).cast(pl.Float64).alias("flat_xi_sd"),
                pl.lit(None).cast(pl.Float64).alias("delta_from_flat"),
                pl.lit(None).cast(pl.Float64).alias("shrinkage_pct"),
                pl.lit(None).cast(pl.Boolean).alias("toward_party_mean"),
            ]
        )

    return df.sort("xi_mean", descending=True)


def extract_group_params(
    idata: az.InferenceData,
    data: dict,
) -> pl.DataFrame:
    """Extract party-level mu and sigma posteriors.

    Returns DataFrame with party, mu_mean, mu_sd, mu_hdi_*, sigma_within_mean, etc.
    """
    mu_post = idata.posterior["mu_party"]
    mu_mean = mu_post.mean(dim=["chain", "draw"]).values
    mu_sd = mu_post.std(dim=["chain", "draw"]).values
    mu_hdi = az.hdi(idata, var_names=["mu_party"], hdi_prob=0.95)["mu_party"].values

    sigma_post = idata.posterior["sigma_within"]
    sigma_mean = sigma_post.mean(dim=["chain", "draw"]).values
    sigma_sd = sigma_post.std(dim=["chain", "draw"]).values
    sigma_hdi = az.hdi(idata, var_names=["sigma_within"], hdi_prob=0.95)["sigma_within"].values

    party_names = data["party_names"]
    rows = []
    for i, name in enumerate(party_names):
        count = int((data["party_idx"] == i).sum())
        rows.append(
            {
                "party": name,
                "n_legislators": count,
                "mu_mean": float(mu_mean[i]),
                "mu_sd": float(mu_sd[i]),
                "mu_hdi_2.5": float(mu_hdi[i, 0]),
                "mu_hdi_97.5": float(mu_hdi[i, 1]),
                "sigma_within_mean": float(sigma_mean[i]),
                "sigma_within_sd": float(sigma_sd[i]),
                "sigma_within_hdi_2.5": float(sigma_hdi[i, 0]),
                "sigma_within_hdi_97.5": float(sigma_hdi[i, 1]),
            }
        )

    return pl.DataFrame(rows)


def compute_variance_decomposition(
    idata: az.InferenceData,
    data: dict,
) -> pl.DataFrame:
    """Compute ICC (intraclass correlation) from posterior samples.

    ICC = sigma_between² / (sigma_between² + sigma_within_pooled²)

    sigma_between is computed from the party means (var of mu_party).
    sigma_within_pooled is the mean of the per-party sigma_within.

    Returns single-row DataFrame with icc_mean, icc_sd, icc_hdi_*.
    """
    mu_post = idata.posterior["mu_party"].values  # (chain, draw, party)
    sigma_post = idata.posterior["sigma_within"].values  # (chain, draw, party)

    n_chain, n_draw, n_party = mu_post.shape

    # Compute ICC per posterior draw (vectorized)
    # Between-party variance: var of party means across parties per draw
    # mu_post shape: (chain, draw, party) → var across party axis
    sigma_between_sq = np.var(mu_post, axis=2)  # (chain, draw)

    # Within-party variance: pooled (weighted by group size)
    party_counts = np.array([(data["party_idx"] == p).sum() for p in range(n_party)])
    total = party_counts.sum()
    # sigma_post shape: (chain, draw, party)
    sigma_within_pooled_sq = (
        np.sum(party_counts[np.newaxis, np.newaxis, :] * sigma_post**2, axis=2) / total
    )  # (chain, draw)

    total_var = sigma_between_sq + sigma_within_pooled_sq
    icc_flat = np.where(total_var > 0, sigma_between_sq / total_var, 0.0)
    icc_samples = icc_flat.ravel()  # (chain * draw,)

    icc_mean = float(np.mean(icc_samples))
    icc_sd = float(np.std(icc_samples))
    icc_hdi = np.percentile(icc_samples, [2.5, 97.5])

    print(f"  ICC: {icc_mean:.3f} ± {icc_sd:.3f} [{icc_hdi[0]:.3f}, {icc_hdi[1]:.3f}]")
    print(f"  → Party explains {icc_mean:.0%} of ideological variance")

    return pl.DataFrame(
        {
            "icc_mean": [icc_mean],
            "icc_sd": [icc_sd],
            "icc_hdi_2.5": [float(icc_hdi[0])],
            "icc_hdi_97.5": [float(icc_hdi[1])],
        }
    )


def compute_flat_hier_correlation(
    hier_ip: pl.DataFrame,
    flat_ip: pl.DataFrame,
    chamber: str,
) -> float:
    """Compute Pearson r between hierarchical and flat IRT ideal points."""
    merged = hier_ip.select("legislator_slug", "xi_mean").join(
        flat_ip.select("legislator_slug", pl.col("xi_mean").alias("flat_xi")),
        on="legislator_slug",
        how="inner",
    )
    if merged.height < 3:
        return float("nan")

    hier_arr = merged["xi_mean"].to_numpy()
    flat_arr = merged["flat_xi"].to_numpy()
    r = float(np.corrcoef(hier_arr, flat_arr)[0, 1])
    print(f"  {chamber}: Hierarchical vs flat Pearson r = {r:.4f}")
    return r


# ── Phase 5: Plots ──────────────────────────────────────────────────────────


def plot_party_posteriors(
    idata: az.InferenceData,
    data: dict,
    chamber: str,
    out_dir: Path,
) -> None:
    """KDE of party mean posteriors — 'Where Do the Parties Stand?'"""
    mu_post = idata.posterior["mu_party"].values  # (chain, draw, party)
    party_names = data["party_names"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, name in enumerate(party_names):
        samples = mu_post[:, :, i].flatten()
        color = PARTY_COLORS.get(name, "#888888")

        # KDE
        kde = stats.gaussian_kde(samples)
        x = np.linspace(samples.min() - 0.5, samples.max() + 0.5, 300)
        ax.plot(x, kde(x), color=color, linewidth=2.5, label=name)
        ax.fill_between(x, kde(x), alpha=0.15, color=color)

        # Posterior mean marker
        mean_val = float(np.mean(samples))
        ax.axvline(mean_val, color=color, linestyle="--", alpha=0.6, linewidth=1)

    ax.set_xlabel("Party Mean Ideal Point (Liberal ← → Conservative)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"{chamber} — Where Do the Parties Stand?\n"
        "Posterior distributions of party-level ideal points",
        fontsize=13,
        fontweight="bold",
    )

    ax.text(
        0.02,
        0.98,
        "Separation = polarization\nOverlap = ideological common ground",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.9},
    )

    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"party_posteriors_{chamber.lower()}.png")


def plot_icc(
    icc_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Bar chart of ICC — 'How Much Does Party Explain?'"""
    icc_mean = float(icc_df["icc_mean"][0])
    icc_lo = float(icc_df["icc_hdi_2.5"][0])
    icc_hi = float(icc_df["icc_hdi_97.5"][0])

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_colors = ["#4C72B0", "#C0C0C0"]
    ax.bar(
        ["Party", "Individual"],
        [icc_mean, 1 - icc_mean],
        color=bar_colors,
        edgecolor="white",
        linewidth=2,
        width=0.5,
    )

    # Error bar on party portion
    ax.errorbar(
        0,
        icc_mean,
        yerr=[[icc_mean - icc_lo], [icc_hi - icc_mean]],
        color="black",
        capsize=8,
        capthick=2,
        linewidth=2,
    )

    # Percentage labels
    ax.text(
        0,
        icc_mean / 2,
        f"{icc_mean:.0%}",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="white",
    )
    ax.text(
        1,
        icc_mean + (1 - icc_mean) / 2,
        f"{1 - icc_mean:.0%}",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#333333",
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of Ideological Variance", fontsize=12)
    ax.set_title(
        f"{chamber} — How Much Does Party Explain?\n"
        f"ICC = {icc_mean:.0%} [{icc_lo:.0%}, {icc_hi:.0%}]",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Party", "Individual"], fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=8)

    fig.text(
        0.5,
        0.01,
        f"Party membership explains {icc_mean:.0%} of the variation in "
        f"how {chamber} members vote. "
        f"The remaining {1 - icc_mean:.0%} reflects individual differences within parties.",
        ha="center",
        fontsize=10,
        fontstyle="italic",
        color="#555555",
        wrap=True,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, out_dir / f"icc_{chamber.lower()}.png")


def plot_shrinkage_scatter(
    hier_ip: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Flat vs hierarchical scatter — 'How Does Accounting for Party Change Estimates?'"""
    if "flat_xi_mean" not in hier_ip.columns:
        return

    df = hier_ip.drop_nulls(subset=["flat_xi_mean"])
    if df.height == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    for party, color in PARTY_COLORS.items():
        sub = df.filter(pl.col("party") == party)
        if sub.height == 0:
            continue
        flat = sub["flat_xi_mean"].to_numpy()
        hier = sub["xi_mean"].to_numpy()
        ax.scatter(
            flat,
            hier,
            c=color,
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label=party,
        )

    # Identity line
    all_flat = df["flat_xi_mean"].to_numpy()
    all_hier = df["xi_mean"].to_numpy()
    lims = [
        min(all_flat.min(), all_hier.min()) - 0.3,
        max(all_flat.max(), all_hier.max()) + 0.3,
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, label="No change")

    # Annotate biggest movers (use delta_from_flat which is scale-corrected)
    if "delta_from_flat" in df.columns:
        df_with_delta = df.with_columns(pl.col("delta_from_flat").abs().alias("abs_delta")).sort(
            "abs_delta", descending=True
        )
    else:
        df_with_delta = df.with_columns(
            (pl.col("xi_mean") - pl.col("flat_xi_mean")).abs().alias("abs_delta")
        ).sort("abs_delta", descending=True)

    for row in df_with_delta.head(5).iter_rows(named=True):
        name = row.get("full_name", row["legislator_slug"])
        last_name = name.split()[-1] if name else "?"
        ax.annotate(
            last_name,
            (row["flat_xi_mean"], row["xi_mean"]),
            fontsize=8,
            fontweight="bold",
            xytext=(8, 8),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.8},
        )

    pearson_r = float(np.corrcoef(all_flat, all_hier)[0, 1])
    ax.set_xlabel("Flat IRT Ideal Point", fontsize=12)
    ax.set_ylabel("Hierarchical IRT Ideal Point", fontsize=12)
    ax.set_title(
        f"{chamber} — How Does Accounting for Party Change Estimates?\n"
        f"Pearson r = {pearson_r:.4f}. Points off the diagonal moved due to party pooling.",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.98,
        "Labels show the 5 legislators\nwhose estimates changed the most",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.9},
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"shrinkage_scatter_{chamber.lower()}.png")


def plot_dispersion(
    idata: az.InferenceData,
    data: dict,
    chamber: str,
    out_dir: Path,
) -> None:
    """KDE of sigma_within per party — 'Which Party Has More Internal Disagreement?'"""
    sigma_post = idata.posterior["sigma_within"].values  # (chain, draw, party)
    party_names = data["party_names"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, name in enumerate(party_names):
        samples = sigma_post[:, :, i].flatten()
        color = PARTY_COLORS.get(name, "#888888")
        kde = stats.gaussian_kde(samples)
        x = np.linspace(max(0, samples.min() - 0.1), samples.max() + 0.3, 300)
        ax.plot(x, kde(x), color=color, linewidth=2.5, label=name)
        ax.fill_between(x, kde(x), alpha=0.15, color=color)

        mean_val = float(np.mean(samples))
        ax.axvline(mean_val, color=color, linestyle="--", alpha=0.6, linewidth=1)

    ax.set_xlabel("Within-Party Standard Deviation (higher = more internal disagreement)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"{chamber} — Which Party Has More Internal Disagreement?",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.98,
        "Wider curves to the right = that party's\nmembers disagree more with each other",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.9},
    )
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"dispersion_{chamber.lower()}.png")


def plot_joint_party_spread(
    idata: az.InferenceData,
    combined_data: dict,
    out_dir: Path,
) -> None:
    """Joint model: party spread posteriors per chamber."""
    if "mu_group" not in idata.posterior:
        return

    mu_group_post = idata.posterior["mu_group"].values  # (chain, draw, group)
    group_names = combined_data["group_names"]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        "House Democrat": "#0015BC",
        "House Republican": "#E81B23",
        "Senate Democrat": "#6080D0",
        "Senate Republican": "#F07080",
    }
    styles = {
        "House Democrat": "-",
        "House Republican": "-",
        "Senate Democrat": "--",
        "Senate Republican": "--",
    }

    for i, name in enumerate(group_names):
        samples = mu_group_post[:, :, i].flatten()
        color = colors.get(name, "#888888")
        style = styles.get(name, "-")
        kde = stats.gaussian_kde(samples)
        x = np.linspace(samples.min() - 0.5, samples.max() + 0.5, 300)
        ax.plot(x, kde(x), color=color, linewidth=2, linestyle=style, label=name)
        ax.fill_between(x, kde(x), alpha=0.1, color=color)

    ax.set_xlabel("Group Mean Ideal Point", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Are Parties More Polarized in the House or Senate?\nSolid = House, Dashed = Senate",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / "joint_party_spread.png")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from ks_vote_scraper.session import KSSession

    ks = KSSession.from_session_string(args.session)
    results_root = ks.results_dir

    data_dir = Path(args.data_dir) if args.data_dir else ks.data_dir
    eda_dir = Path(args.eda_dir) if args.eda_dir else results_root / "eda" / "latest"
    pca_dir = Path(args.pca_dir) if args.pca_dir else results_root / "pca" / "latest"
    irt_dir = Path(args.irt_dir) if args.irt_dir else results_root / "irt" / "latest"

    with RunContext(
        session=args.session,
        analysis_name="hierarchical",
        params=vars(args),
        primer=HIERARCHICAL_PRIMER,
    ) as ctx:
        print(f"KS Legislature Hierarchical Bayesian IRT — Session {args.session}")
        print(f"Data:     {data_dir}")
        print(f"EDA:      {eda_dir}")
        print(f"PCA:      {pca_dir}")
        print(f"Flat IRT: {irt_dir}")
        print(f"Output:   {ctx.run_dir}")

        # ── Load data ──
        print_header("LOADING DATA")
        house_matrix, senate_matrix, full_matrix = load_eda_matrices(eda_dir)
        house_pca, senate_pca = load_pca_scores(pca_dir)
        rollcalls, legislators = load_metadata(data_dir)

        # Load flat IRT ideal points for comparison
        flat_ip: dict[str, pl.DataFrame | None] = {}
        for ch in ("house", "senate"):
            flat_path = irt_dir / "data" / f"ideal_points_{ch}.parquet"
            if flat_path.exists():
                flat_ip[ch] = pl.read_parquet(flat_path)
                print(f"  Flat IRT ({ch}): {flat_ip[ch].height} legislators loaded")
            else:
                flat_ip[ch] = None
                print(f"  Flat IRT ({ch}): not found at {flat_path}")

        # ── Per-chamber models ──
        per_chamber_results: dict[str, dict] = {}

        for chamber, matrix, pca_scores in [
            ("House", house_matrix, house_pca),
            ("Senate", senate_matrix, senate_pca),
        ]:
            ch = chamber.lower()
            print_header(f"HIERARCHICAL IRT — {chamber}")

            # Prepare data with party indices
            data = prepare_hierarchical_data(matrix, legislators, chamber)

            # Build and sample
            print_header(f"SAMPLING — {chamber}")
            idata, sampling_time = build_per_chamber_model(
                data,
                n_samples=args.n_samples,
                n_tune=args.n_tune,
                n_chains=args.n_chains,
            )

            # Convergence
            convergence = check_hierarchical_convergence(idata, chamber)

            # Extract results
            print_header(f"EXTRACTING RESULTS — {chamber}")
            ideal_points = extract_hierarchical_ideal_points(
                idata, data, legislators, flat_ip=flat_ip[ch]
            )
            group_params = extract_group_params(idata, data)
            icc_df = compute_variance_decomposition(idata, data)

            # Correlation with flat IRT
            flat_corr = float("nan")
            if flat_ip[ch] is not None:
                flat_corr = compute_flat_hier_correlation(ideal_points, flat_ip[ch], chamber)

            # Print group params
            print("\n  Group-level parameters:")
            for row in group_params.iter_rows(named=True):
                print(
                    f"    {row['party']}: mu={row['mu_mean']:+.3f} "
                    f"[{row['mu_hdi_2.5']:+.3f}, {row['mu_hdi_97.5']:+.3f}], "
                    f"sigma={row['sigma_within_mean']:.3f}"
                )

            # Save parquets + NetCDF
            ideal_points.write_parquet(ctx.data_dir / f"hierarchical_ideal_points_{ch}.parquet")
            group_params.write_parquet(ctx.data_dir / f"group_params_{ch}.parquet")
            icc_df.write_parquet(ctx.data_dir / f"variance_decomposition_{ch}.parquet")
            idata.to_netcdf(str(ctx.data_dir / f"idata_{ch}.nc"))

            # Plots
            print_header(f"PLOTS — {chamber}")
            plot_party_posteriors(idata, data, chamber, ctx.plots_dir)
            plot_icc(icc_df, chamber, ctx.plots_dir)
            plot_shrinkage_scatter(ideal_points, chamber, ctx.plots_dir)
            plot_forest(ideal_points, chamber, ctx.plots_dir)
            plot_dispersion(idata, data, chamber, ctx.plots_dir)

            per_chamber_results[chamber] = {
                "data": data,
                "idata": idata,
                "ideal_points": ideal_points,
                "group_params": group_params,
                "icc_df": icc_df,
                "convergence": convergence,
                "sampling_time": sampling_time,
                "flat_corr": flat_corr,
            }

        # ── Joint model (optional) ──
        joint_results: dict | None = None
        if not args.skip_joint:
            print_header("JOINT CROSS-CHAMBER MODEL")
            try:
                house_data = per_chamber_results["House"]["data"]
                senate_data = per_chamber_results["Senate"]["data"]

                joint_idata, combined_data, joint_time = build_joint_model(
                    house_data,
                    senate_data,
                    n_samples=args.n_samples,
                    n_tune=args.n_tune,
                    n_chains=args.n_chains,
                )

                joint_convergence = check_hierarchical_convergence(joint_idata, "Joint")

                # Extract joint ideal points
                joint_ip = extract_hierarchical_ideal_points(
                    joint_idata,
                    combined_data,
                    legislators,
                )

                # Save
                joint_ip.write_parquet(ctx.data_dir / "hierarchical_ideal_points_joint.parquet")
                joint_idata.to_netcdf(str(ctx.data_dir / "idata_joint.nc"))

                # Joint plots
                print_header("JOINT PLOTS")
                plot_joint_party_spread(joint_idata, combined_data, ctx.plots_dir)
                plot_forest(joint_ip, "Joint", ctx.plots_dir)

                joint_results = {
                    "idata": joint_idata,
                    "combined_data": combined_data,
                    "ideal_points": joint_ip,
                    "convergence": joint_convergence,
                    "sampling_time": joint_time,
                }

            except Exception as e:
                print(f"\n  WARNING: Joint model failed: {e}")
                print("  Continuing with per-chamber results only.")
                joint_results = None
        else:
            print("\n  Skipping joint model (--skip-joint)")

        # ── HTML Report ──
        print_header("HTML REPORT")
        ctx.report.title = f"Kansas Legislature {ctx.session} — Hierarchical Bayesian IRT"
        build_hierarchical_report(
            ctx.report,
            chamber_results=per_chamber_results,
            joint_results=joint_results,
            plots_dir=ctx.plots_dir,
        )

        # ── Filtering manifest ──
        print_header("FILTERING MANIFEST")
        manifest: dict = {
            "analysis": "hierarchical_irt",
            "session": args.session,
            "constants": {
                "HIER_N_SAMPLES": args.n_samples,
                "HIER_N_TUNE": args.n_tune,
                "HIER_N_CHAINS": args.n_chains,
                "HIER_TARGET_ACCEPT": HIER_TARGET_ACCEPT,
                "RHAT_THRESHOLD": RHAT_THRESHOLD,
                "ESS_THRESHOLD": ESS_THRESHOLD,
                "MAX_DIVERGENCES": MAX_DIVERGENCES,
            },
        }

        for chamber in ("House", "Senate"):
            if chamber not in per_chamber_results:
                continue
            ch = chamber.lower()
            res = per_chamber_results[chamber]
            manifest[f"{ch}_n_legislators"] = res["ideal_points"].height
            manifest[f"{ch}_sampling_time_s"] = round(res["sampling_time"], 1)
            manifest[f"{ch}_convergence_ok"] = res["convergence"]["all_ok"]
            manifest[f"{ch}_divergences"] = res["convergence"]["divergences"]
            manifest[f"{ch}_flat_corr"] = round(res["flat_corr"], 4)
            manifest[f"{ch}_icc_mean"] = round(float(res["icc_df"]["icc_mean"][0]), 4)

            for row in res["group_params"].iter_rows(named=True):
                party_key = row["party"].lower()
                manifest[f"{ch}_{party_key}_mu_mean"] = round(row["mu_mean"], 4)
                manifest[f"{ch}_{party_key}_sigma_within"] = round(row["sigma_within_mean"], 4)

        if joint_results is not None:
            manifest["joint_n_legislators"] = joint_results["ideal_points"].height
            manifest["joint_sampling_time_s"] = round(joint_results["sampling_time"], 1)
            manifest["joint_convergence_ok"] = joint_results["convergence"]["all_ok"]
        else:
            manifest["joint_skipped"] = True

        manifest_path = ctx.run_dir / "filtering_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  Saved: {manifest_path.name}")

        # ── Summary ──
        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")
        print(f"  NetCDF files:   {len(list(ctx.data_dir.glob('*.nc')))}")


if __name__ == "__main__":
    main()
