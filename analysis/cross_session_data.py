"""Pure data logic for cross-session validation.

Matching, IRT scale alignment, ideology shift metrics, and metric stability
comparison. No I/O, no plotting — all functions take DataFrames in, return
DataFrames or dicts out.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
from scipy import stats

# ── Constants ────────────────────────────────────────────────────────────────

MIN_OVERLAP: int = 20
"""Minimum returning legislators for meaningful comparison."""

SHIFT_THRESHOLD_SD: float = 1.0
"""Flag legislators who moved > this many SDs as significant movers."""

ALIGNMENT_TRIM_PCT: int = 10
"""Trim this % of extreme residuals from affine fit for robustness."""

CORRELATION_WARN: float = 0.70
"""Warn if cross-session Pearson r falls below this value."""

FEATURE_IMPORTANCE_TOP_K: int = 10
"""Compare top K SHAP features across sessions."""

PREDICTION_META_COLS: list[str] = ["legislator_slug", "vote_id", "vote_binary"]
"""Columns to exclude from feature sets during prediction."""

STABILITY_METRICS: list[str] = [
    "unity_score",
    "maverick_rate",
    "weighted_maverick",
    "betweenness",
    "eigenvector",
    "pagerank",
    "loyalty_rate",
    "PC1",
]
"""Metrics to compare across sessions in the stability analysis."""

_SUFFIX_RE = re.compile(r"\s*-\s+.*$")
"""Matches leadership suffixes like ' - House Minority Caucus Chair'."""


# ── Legislator Matching ──────────────────────────────────────────────────────


def normalize_name(name: str) -> str:
    """Normalize a legislator name for cross-session matching.

    Lowercases, strips whitespace, and removes leadership suffixes
    (e.g., ``' - House Minority Caucus Chair'``).
    """
    name = name.strip().lower()
    name = _SUFFIX_RE.sub("", name)
    return name


def _normalize_slug_col(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure the slug column is named ``legislator_slug``."""
    if "slug" in df.columns and "legislator_slug" not in df.columns:
        return df.rename({"slug": "legislator_slug"})
    return df


def _add_name_norm(df: pl.DataFrame) -> pl.DataFrame:
    """Add a ``name_norm`` column by applying :func:`normalize_name`."""
    return df.with_columns(
        pl.col("full_name").map_elements(normalize_name, return_dtype=pl.Utf8).alias("name_norm")
    )


def match_legislators(
    leg_a: pl.DataFrame,
    leg_b: pl.DataFrame,
) -> pl.DataFrame:
    """Match legislators across sessions by normalized full_name.

    Args:
        leg_a: Legislators from session A (needs ``full_name``,
            ``slug`` or ``legislator_slug``, ``party``, ``chamber``,
            ``district``).
        leg_b: Legislators from session B (same columns).

    Returns:
        DataFrame with columns: ``name_norm``, ``full_name_a``,
        ``full_name_b``, ``slug_a``, ``slug_b``, ``party_a``, ``party_b``,
        ``chamber_a``, ``chamber_b``, ``district_a``, ``district_b``,
        ``is_chamber_switch``, ``is_party_switch``.

    Raises:
        ValueError: If fewer than :data:`MIN_OVERLAP` legislators match.
    """
    a = _add_name_norm(_normalize_slug_col(leg_a))
    b = _add_name_norm(_normalize_slug_col(leg_b))

    matched = (
        a.select(
            "name_norm",
            pl.col("full_name").alias("full_name_a"),
            pl.col("legislator_slug").alias("slug_a"),
            pl.col("party").alias("party_a"),
            pl.col("chamber").alias("chamber_a"),
            pl.col("district").alias("district_a"),
        )
        .join(
            b.select(
                "name_norm",
                pl.col("full_name").alias("full_name_b"),
                pl.col("legislator_slug").alias("slug_b"),
                pl.col("party").alias("party_b"),
                pl.col("chamber").alias("chamber_b"),
                pl.col("district").alias("district_b"),
            ),
            on="name_norm",
            how="inner",
        )
        .with_columns(
            (pl.col("chamber_a") != pl.col("chamber_b")).alias("is_chamber_switch"),
            (pl.col("party_a") != pl.col("party_b")).alias("is_party_switch"),
        )
        .sort("name_norm")
    )

    if matched.height < MIN_OVERLAP:
        msg = (
            f"Only {matched.height} legislators matched across sessions "
            f"(minimum {MIN_OVERLAP}). Check data quality."
        )
        raise ValueError(msg)

    return matched


def classify_turnover(
    leg_a: pl.DataFrame,
    leg_b: pl.DataFrame,
    matched: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Classify legislators into returning, departing, and new cohorts.

    Args:
        leg_a: All legislators from session A.
        leg_b: All legislators from session B.
        matched: Output of :func:`match_legislators`.

    Returns:
        ``{"returning": matched, "departing": in A not B, "new": in B not A}``
    """
    a = _normalize_slug_col(leg_a)
    b = _normalize_slug_col(leg_b)

    matched_slugs_a = set(matched["slug_a"].to_list())
    matched_slugs_b = set(matched["slug_b"].to_list())

    departing = a.filter(~pl.col("legislator_slug").is_in(matched_slugs_a))
    new = b.filter(~pl.col("legislator_slug").is_in(matched_slugs_b))

    return {"returning": matched, "departing": departing, "new": new}


# ── IRT Scale Alignment ─────────────────────────────────────────────────────


def align_irt_scales(
    xi_a: pl.DataFrame,
    xi_b: pl.DataFrame,
    matched: pl.DataFrame,
) -> tuple[float, float, pl.DataFrame]:
    """Robust affine alignment of IRT ideal points across sessions.

    Transforms session A onto session B's scale:
    ``xi_a_aligned = A * xi_a + B``.

    Uses overlapping legislators as anchors.  Trims the
    ``ALIGNMENT_TRIM_PCT`` most extreme residuals (genuine movers) before
    the final fit for robustness.

    Args:
        xi_a: IRT ideal points from session A (needs ``legislator_slug``,
            ``xi_mean``, ``full_name``).
        xi_b: IRT ideal points from session B (same columns).
        matched: Output of :func:`match_legislators`.

    Returns:
        ``(A, B, aligned_df)`` where *aligned_df* has columns:
        ``name_norm``, ``slug_a``, ``slug_b``, ``full_name``, ``party``,
        ``chamber``, ``xi_a``, ``xi_b``, ``xi_a_aligned``, ``delta_xi``,
        ``abs_delta_xi``.

    Raises:
        ValueError: If fewer than :data:`MIN_OVERLAP` legislators have IRT
            scores in both sessions.
    """
    pairs = (
        matched.select("name_norm", "slug_a", "slug_b", "party_b", "chamber_b")
        .join(
            xi_a.select(
                pl.col("legislator_slug").alias("slug_a"),
                pl.col("xi_mean").alias("xi_a"),
            ),
            on="slug_a",
            how="inner",
        )
        .join(
            xi_b.select(
                pl.col("legislator_slug").alias("slug_b"),
                pl.col("xi_mean").alias("xi_b"),
                pl.col("full_name"),
            ),
            on="slug_b",
            how="inner",
        )
    )

    if pairs.height < MIN_OVERLAP:
        msg = f"Only {pairs.height} legislators have IRT scores in both sessions"
        raise ValueError(msg)

    x = pairs["xi_a"].to_numpy().astype(np.float64)
    y = pairs["xi_b"].to_numpy().astype(np.float64)

    # Initial OLS fit
    result = stats.linregress(x, y)
    a_init, b_init = float(result.slope), float(result.intercept)

    # Trim extreme residuals (genuine movers distort alignment)
    residuals = y - (a_init * x + b_init)
    abs_residuals = np.abs(residuals)
    cutoff = np.percentile(abs_residuals, 100 - ALIGNMENT_TRIM_PCT)
    keep_mask = abs_residuals <= cutoff

    if np.sum(keep_mask) >= MIN_OVERLAP:
        result_trimmed = stats.linregress(x[keep_mask], y[keep_mask])
        a_final = float(result_trimmed.slope)
        b_final = float(result_trimmed.intercept)
    else:
        a_final, b_final = a_init, b_init

    aligned = (
        pairs.with_columns(
            (pl.col("xi_a") * a_final + b_final).alias("xi_a_aligned"),
            # Strip leadership suffixes (" - President of the Senate" etc.)
            pl.col("full_name").str.replace(r"\s*-\s+.*$", "").alias("full_name"),
        )
        .with_columns(
            (pl.col("xi_b") - pl.col("xi_a_aligned")).alias("delta_xi"),
        )
        .with_columns(
            pl.col("delta_xi").abs().alias("abs_delta_xi"),
        )
        .rename({"party_b": "party", "chamber_b": "chamber"})
    )

    return a_final, b_final, aligned


# ── Shift Analysis ───────────────────────────────────────────────────────────


def compute_ideology_shift(aligned: pl.DataFrame) -> pl.DataFrame:
    """Add shift classification to an aligned DataFrame.

    New columns:
        ``rank_a`` / ``rank_b``: within-group ordinal rank by ideology.
        ``rank_shift``: ``rank_b - rank_a`` (positive = moved rightward in ranking).
        ``is_significant_mover``: ``|delta_xi| > SHIFT_THRESHOLD_SD * std(delta_xi)``.
        ``shift_direction``: ``"leftward"`` / ``"rightward"`` / ``"stable"``.
    """
    delta_std = aligned["delta_xi"].std()
    threshold = (
        SHIFT_THRESHOLD_SD * delta_std if delta_std is not None and delta_std > 0 else float("inf")
    )

    return aligned.with_columns(
        pl.col("xi_a_aligned").rank("ordinal").cast(pl.Int32).alias("rank_a"),
        pl.col("xi_b").rank("ordinal").cast(pl.Int32).alias("rank_b"),
    ).with_columns(
        (pl.col("rank_b") - pl.col("rank_a")).alias("rank_shift"),
        (pl.col("abs_delta_xi") > threshold).alias("is_significant_mover"),
        pl.when(pl.col("delta_xi") > threshold)
        .then(pl.lit("rightward"))
        .when(pl.col("delta_xi") < -threshold)
        .then(pl.lit("leftward"))
        .otherwise(pl.lit("stable"))
        .alias("shift_direction"),
    )


# ── Metric Stability ────────────────────────────────────────────────────────


def _empty_stability_df() -> pl.DataFrame:
    """Return an empty DataFrame with the metric stability schema."""
    return pl.DataFrame(
        schema={
            "metric": pl.Utf8,
            "pearson_r": pl.Float64,
            "spearman_rho": pl.Float64,
            "n_legislators": pl.Int64,
        }
    )


def compute_metric_stability(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    matched: pl.DataFrame,
    metrics: list[str] | None = None,
) -> pl.DataFrame:
    """Compute correlation of metrics across sessions for returning legislators.

    Args:
        df_a: Legislator DataFrame from session A (from ``build_legislator_df``).
        df_b: Legislator DataFrame from session B.
        matched: Output of :func:`match_legislators`.
        metrics: Column names to compare.  Defaults to :data:`STABILITY_METRICS`.

    Returns:
        DataFrame with columns: ``metric``, ``pearson_r``, ``spearman_rho``,
        ``n_legislators``.  Metrics missing from either session are skipped.
    """
    if metrics is None:
        metrics = STABILITY_METRICS

    rows: list[dict] = []

    for metric in metrics:
        if metric not in df_a.columns or metric not in df_b.columns:
            continue

        pairs = (
            matched.select("slug_a", "slug_b")
            .join(
                df_a.select(
                    pl.col("legislator_slug").alias("slug_a"),
                    pl.col(metric).alias("val_a"),
                ),
                on="slug_a",
                how="inner",
            )
            .join(
                df_b.select(
                    pl.col("legislator_slug").alias("slug_b"),
                    pl.col(metric).alias("val_b"),
                ),
                on="slug_b",
                how="inner",
            )
            .drop_nulls(subset=["val_a", "val_b"])
        )

        if pairs.height < 3:
            continue

        va = pairs["val_a"].to_numpy()
        vb = pairs["val_b"].to_numpy()

        pearson_r, _ = stats.pearsonr(va, vb)
        spearman_rho, _ = stats.spearmanr(va, vb)

        rows.append(
            {
                "metric": metric,
                "pearson_r": round(float(pearson_r), 4),
                "spearman_rho": round(float(spearman_rho), 4),
                "n_legislators": pairs.height,
            }
        )

    if not rows:
        return _empty_stability_df()

    return pl.DataFrame(rows)


def compute_turnover_impact(
    xi_returning: np.ndarray,
    xi_departing: np.ndarray,
    xi_new: np.ndarray,
) -> dict:
    """Compare ideology distributions across turnover cohorts.

    Args:
        xi_returning: IRT ideal points for returning legislators.
        xi_departing: IRT ideal points for departing legislators.
        xi_new: IRT ideal points for new legislators.

    Returns:
        Dict with per-cohort stats (``{cohort}_mean``, ``{cohort}_std``,
        ``{cohort}_n``) and KS test results for departing-vs-returning and
        new-vs-returning comparisons.
    """
    result: dict = {}

    for label, arr in [
        ("returning", xi_returning),
        ("departing", xi_departing),
        ("new", xi_new),
    ]:
        result[f"{label}_mean"] = float(np.mean(arr)) if len(arr) > 0 else None
        result[f"{label}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else None
        result[f"{label}_n"] = len(arr)

    if len(xi_departing) >= 2 and len(xi_returning) >= 2:
        ks_dep, p_dep = stats.ks_2samp(xi_departing, xi_returning)
        result["ks_departing_vs_returning"] = float(ks_dep)
        result["p_departing_vs_returning"] = float(p_dep)

    if len(xi_new) >= 2 and len(xi_returning) >= 2:
        ks_new, p_new = stats.ks_2samp(xi_new, xi_returning)
        result["ks_new_vs_returning"] = float(ks_new)
        result["p_new_vs_returning"] = float(p_new)

    return result


# ── Cross-Session Prediction Helpers ────────────────────────────────────────


def align_feature_columns(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Align feature columns across two vote-feature DataFrames.

    Takes the intersection of feature columns (excluding metadata).
    One-hot vote_type columns may differ between sessions — only shared
    columns are kept.

    Args:
        df_a: Vote features from session A.
        df_b: Vote features from session B.

    Returns:
        ``(df_a_aligned, df_b_aligned, feature_cols)`` where both
        DataFrames have exactly the same columns in the same order.
    """
    meta = set(PREDICTION_META_COLS)
    cols_a = set(df_a.columns) - meta
    cols_b = set(df_b.columns) - meta
    shared = sorted(cols_a & cols_b)

    select_cols = list(PREDICTION_META_COLS) + shared
    # Only select columns that actually exist (some meta cols may be absent)
    select_a = [c for c in select_cols if c in df_a.columns]
    select_b = [c for c in select_cols if c in df_b.columns]

    return df_a.select(select_a), df_b.select(select_b), shared


def standardize_features(
    df: pl.DataFrame,
    numeric_cols: list[str],
) -> pl.DataFrame:
    """Z-score standardize numeric feature columns in-place.

    Binary and one-hot columns (detected by having only 0/1 values)
    are left untouched. Only truly continuous columns are standardized.

    Args:
        df: Vote features DataFrame.
        numeric_cols: Columns to consider for standardization.

    Returns:
        DataFrame with continuous numeric columns z-scored.
    """
    binary_cols = set()
    for col in numeric_cols:
        vals = df[col].drop_nulls()
        if vals.n_unique() <= 2:
            binary_cols.add(col)

    z_exprs = []
    for col in numeric_cols:
        if col in binary_cols:
            continue
        mean = df[col].mean()
        std = df[col].std()
        if std is not None and std > 0:
            z_exprs.append(((pl.col(col) - mean) / std).alias(col))

    if z_exprs:
        return df.with_columns(z_exprs)
    return df


def compare_feature_importance(
    shap_a: np.ndarray,
    shap_b: np.ndarray,
    feature_names: list[str],
    top_k: int | None = None,
) -> tuple[pl.DataFrame, float]:
    """Compare SHAP importance rankings across sessions.

    Args:
        shap_a: SHAP values from session A model (n_samples x n_features).
        shap_b: SHAP values from session B model.
        feature_names: Feature names matching the columns.
        top_k: Compare top K features. Defaults to
            :data:`FEATURE_IMPORTANCE_TOP_K`.

    Returns:
        ``(comparison_df, kendall_tau)`` where *comparison_df* has columns
        ``feature``, ``importance_a``, ``importance_b``, ``rank_a``,
        ``rank_b``; and *kendall_tau* is Kendall's tau on the top-K
        rankings.
    """
    if top_k is None:
        top_k = FEATURE_IMPORTANCE_TOP_K

    imp_a = np.abs(shap_a).mean(axis=0)
    imp_b = np.abs(shap_b).mean(axis=0)

    # argsort gives indices that sort the array; we need ranks (position of each element)
    rank_a = np.empty_like(np.argsort(-imp_a))
    rank_a[np.argsort(-imp_a)] = np.arange(1, len(imp_a) + 1)
    rank_b = np.empty_like(np.argsort(-imp_b))
    rank_b[np.argsort(-imp_b)] = np.arange(1, len(imp_b) + 1)

    df = pl.DataFrame(
        {
            "feature": feature_names,
            "importance_a": imp_a.tolist(),
            "importance_b": imp_b.tolist(),
            "rank_a": rank_a.tolist(),
            "rank_b": rank_b.tolist(),
        }
    ).sort("rank_a")

    # Kendall's tau on top-K features (by session A ranking)
    top_features = df.head(min(top_k, len(feature_names)))
    if top_features.height >= 2:
        tau, _ = stats.kendalltau(
            top_features["rank_a"].to_numpy(),
            top_features["rank_b"].to_numpy(),
        )
    else:
        tau = float("nan")

    return df, float(tau)
