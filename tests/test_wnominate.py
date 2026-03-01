"""Tests for W-NOMINATE + OC validation pure data logic.

Tests only the Python functions in wnominate_data.py — never calls R.
All fixtures use synthetic data with known vote patterns.
"""

import polars as pl
import pytest
from analysis.wnominate_data import (
    ROLLCALL_MISSING,
    ROLLCALL_NAY,
    ROLLCALL_YEA,
    build_comparison_table,
    compute_three_way_correlations,
    compute_within_party_correlations,
    convert_vote_matrix_to_rollcall_csv,
    parse_eigenvalues,
    parse_fit_statistics,
    parse_oc_results,
    parse_wnominate_results,
    select_polarity_legislator,
    sign_align_scores,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def vote_matrix() -> pl.DataFrame:
    """Synthetic vote matrix: 5 legislators, 4 votes."""
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_a", "rep_b", "rep_c", "sen_d", "sen_e"],
            "vote_001": [1.0, 1.0, 0.0, 0.0, None],
            "vote_002": [1.0, 0.0, 0.0, 1.0, 1.0],
            "vote_003": [0.0, 1.0, None, 0.0, 1.0],
            "vote_004": [1.0, 1.0, 1.0, 0.0, 0.0],
        }
    )


@pytest.fixture
def pca_scores() -> pl.DataFrame:
    """PCA scores matched to vote_matrix fixture."""
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_a", "rep_b", "rep_c", "sen_d", "sen_e"],
            "PC1": [2.5, 1.8, 0.5, -1.2, -2.0],
        }
    )


@pytest.fixture
def irt_df() -> pl.DataFrame:
    """IRT ideal points matched to vote_matrix fixture."""
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_a", "rep_b", "rep_c", "sen_d", "sen_e"],
            "xi_mean": [1.5, 1.0, 0.3, -0.8, -1.5],
            "full_name": ["Alice A", "Bob B", "Charlie C", "Dana D", "Eve E"],
            "party": ["Republican", "Republican", "Republican", "Democrat", "Democrat"],
        }
    )


@pytest.fixture
def wnom_df() -> pl.DataFrame:
    """W-NOMINATE results matched to vote_matrix fixture."""
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_a", "rep_b", "rep_c", "sen_d", "sen_e"],
            "wnom_dim1": [0.8, 0.6, 0.1, -0.5, -0.9],
            "wnom_dim2": [0.1, -0.2, 0.3, 0.0, -0.1],
            "wnom_se1": [0.05, 0.06, 0.08, 0.07, 0.05],
            "wnom_se2": [0.04, 0.05, 0.07, 0.06, 0.04],
        }
    )


@pytest.fixture
def oc_df() -> pl.DataFrame:
    """OC results matched to vote_matrix fixture."""
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_a", "rep_b", "rep_c", "sen_d", "sen_e"],
            "oc_dim1": [0.9, 0.5, 0.2, -0.4, -0.8],
            "oc_dim2": [0.0, -0.1, 0.2, 0.1, -0.2],
            "oc_correct_class": [0.92, 0.88, 0.85, 0.90, 0.87],
        }
    )


# ── TestConvertVoteMatrixToRollcall ──────────────────────────────────────────


class TestConvertVoteMatrixToRollcall:
    def test_yea_coded_as_1(self, vote_matrix: pl.DataFrame) -> None:
        coded, slugs = convert_vote_matrix_to_rollcall_csv(vote_matrix)
        # rep_a vote_001 = 1.0 -> ROLLCALL_YEA = 1
        assert coded.filter(pl.col("legislator_slug") == "rep_a")["vote_001"][0] == ROLLCALL_YEA

    def test_nay_coded_as_6(self, vote_matrix: pl.DataFrame) -> None:
        coded, slugs = convert_vote_matrix_to_rollcall_csv(vote_matrix)
        # rep_c vote_001 = 0.0 -> ROLLCALL_NAY = 6
        assert coded.filter(pl.col("legislator_slug") == "rep_c")["vote_001"][0] == ROLLCALL_NAY

    def test_missing_coded_as_9(self, vote_matrix: pl.DataFrame) -> None:
        coded, slugs = convert_vote_matrix_to_rollcall_csv(vote_matrix)
        # sen_e vote_001 = None -> ROLLCALL_MISSING = 9
        assert coded.filter(pl.col("legislator_slug") == "sen_e")["vote_001"][0] == ROLLCALL_MISSING

    def test_preserves_legislator_order(self, vote_matrix: pl.DataFrame) -> None:
        coded, slugs = convert_vote_matrix_to_rollcall_csv(vote_matrix)
        assert slugs == ["rep_a", "rep_b", "rep_c", "sen_d", "sen_e"]

    def test_preserves_vote_columns(self, vote_matrix: pl.DataFrame) -> None:
        coded, slugs = convert_vote_matrix_to_rollcall_csv(vote_matrix)
        expected_vote_cols = ["vote_001", "vote_002", "vote_003", "vote_004"]
        assert coded.columns[1:] == expected_vote_cols


# ── TestSelectPolarityLegislator ─────────────────────────────────────────────


class TestSelectPolarityLegislator:
    def test_returns_highest_pc1(self, pca_scores: pl.DataFrame, vote_matrix: pl.DataFrame) -> None:
        idx = select_polarity_legislator(pca_scores, vote_matrix)
        # rep_a has PC1=2.5 (highest) and 3/4 votes = 75% participation
        assert idx == 1  # 1-based

    def test_returns_1_based_index(
        self, pca_scores: pl.DataFrame, vote_matrix: pl.DataFrame
    ) -> None:
        idx = select_polarity_legislator(pca_scores, vote_matrix)
        assert idx >= 1

    def test_respects_participation_threshold(self) -> None:
        # Legislator with highest PC1 has low participation
        vm = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_b"],
                "v1": [None, 1.0],
                "v2": [None, 0.0],
                "v3": [None, 1.0],
                "v4": [1.0, 0.0],
            }
        )
        pca = pl.DataFrame({"legislator_slug": ["rep_a", "rep_b"], "PC1": [3.0, 1.0]})
        # rep_a: 1/4 = 25% < 50% threshold; rep_b: 4/4 = 100%
        idx = select_polarity_legislator(pca, vm, min_participation=0.50)
        assert idx == 2  # rep_b (1-based)


# ── TestParseWnominateResults ────────────────────────────────────────────────


class TestParseWnominateResults:
    def test_maps_slugs_to_coordinates(self) -> None:
        coords = pl.DataFrame(
            {
                "coord1D": [0.5, -0.3],
                "coord2D": [0.1, -0.2],
                "se1": [0.05, 0.06],
                "se2": [0.04, 0.05],
            }
        )
        result = parse_wnominate_results(coords, ["rep_a", "rep_b"])
        assert result["legislator_slug"].to_list() == ["rep_a", "rep_b"]
        assert "wnom_dim1" in result.columns
        assert "wnom_dim2" in result.columns

    def test_handles_na_values(self) -> None:
        coords = pl.DataFrame(
            {
                "coord1D": [0.5, None],
                "coord2D": [None, -0.2],
                "se1": [0.05, None],
                "se2": [None, 0.05],
            }
        )
        result = parse_wnominate_results(coords, ["rep_a", "rep_b"])
        assert result.height == 2
        assert result["wnom_dim1"][0] == 0.5
        assert result["wnom_dim1"][1] is None

    def test_expected_columns(self) -> None:
        coords = pl.DataFrame({"coord1D": [0.5], "coord2D": [0.1], "se1": [0.05], "se2": [0.04]})
        result = parse_wnominate_results(coords, ["rep_a"])
        assert result.columns == [
            "legislator_slug",
            "wnom_dim1",
            "wnom_dim2",
            "wnom_se1",
            "wnom_se2",
        ]


# ── TestParseOcResults ───────────────────────────────────────────────────────


class TestParseOcResults:
    def test_maps_slugs_to_coordinates(self) -> None:
        coords = pl.DataFrame(
            {"coord1D": [0.5, -0.3], "coord2D": [0.1, -0.2], "correctClassification": [0.9, 0.85]}
        )
        result = parse_oc_results(coords, ["rep_a", "rep_b"])
        assert result["legislator_slug"].to_list() == ["rep_a", "rep_b"]
        assert "oc_dim1" in result.columns
        assert "oc_correct_class" in result.columns

    def test_expected_columns(self) -> None:
        coords = pl.DataFrame({"coord1D": [0.5], "coord2D": [0.1], "correctClassification": [0.9]})
        result = parse_oc_results(coords, ["rep_a"])
        assert result.columns == [
            "legislator_slug",
            "oc_dim1",
            "oc_dim2",
            "oc_correct_class",
        ]


# ── TestParseFitStatistics ───────────────────────────────────────────────────


class TestParseFitStatistics:
    def test_extracts_wnominate_stats(self) -> None:
        fit = {
            "wnominate": {"correctClassification": 0.88, "APRE": 0.65, "GMP": 0.72},
            "oc": {"correctClassification": 0.91, "APRE": 0.70, "GMP": 0.78},
        }
        result = parse_fit_statistics(fit)
        assert result["wnominate_correctClassification"] == 0.88
        assert result["wnominate_APRE"] == 0.65
        assert result["oc_GMP"] == 0.78

    def test_handles_missing_oc(self) -> None:
        fit = {"wnominate": {"correctClassification": 0.88, "APRE": 0.65, "GMP": 0.72}}
        result = parse_fit_statistics(fit)
        assert "wnominate_correctClassification" in result
        assert "oc_correctClassification" not in result


# ── TestSignAlignScores ──────────────────────────────────────────────────────


class TestSignAlignScores:
    def test_no_flip_when_positive_r(self, wnom_df: pl.DataFrame, irt_df: pl.DataFrame) -> None:
        # WNOM and IRT are already positively correlated
        result = sign_align_scores(wnom_df, "wnom_dim1", irt_df, "xi_mean")
        # Should remain unchanged
        assert result["wnom_dim1"][0] == pytest.approx(0.8)

    def test_flip_when_negative_r(self, irt_df: pl.DataFrame) -> None:
        # Create reversed scores
        reversed_wnom = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_b", "rep_c", "sen_d", "sen_e"],
                "wnom_dim1": [-0.8, -0.6, -0.1, 0.5, 0.9],
            }
        )
        result = sign_align_scores(reversed_wnom, "wnom_dim1", irt_df, "xi_mean")
        # Should be flipped to positive correlation
        assert result["wnom_dim1"][0] == pytest.approx(0.8)


# ── TestThreeWayCorrelations ─────────────────────────────────────────────────


class TestThreeWayCorrelations:
    def test_all_pairs_present(
        self, irt_df: pl.DataFrame, wnom_df: pl.DataFrame, oc_df: pl.DataFrame
    ) -> None:
        result = compute_three_way_correlations(irt_df, wnom_df, oc_df)
        assert "irt_wnom" in result
        assert "irt_oc" in result
        assert "wnom_oc" in result

    def test_perfect_correlation(self) -> None:
        df = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c", "d"],
                "xi_mean": [1.0, 2.0, 3.0, 4.0],
                "wnom_dim1": [1.0, 2.0, 3.0, 4.0],
            }
        )
        result = compute_three_way_correlations(
            df, df.rename({"xi_mean": "wnom_dim1_x"}), None, wnom_col="wnom_dim1"
        )
        assert result["irt_wnom"]["pearson_r"] == pytest.approx(1.0)

    def test_missing_oc_graceful(self, irt_df: pl.DataFrame, wnom_df: pl.DataFrame) -> None:
        result = compute_three_way_correlations(irt_df, wnom_df, oc_df=None)
        assert result["irt_oc"]["n"] == 0
        assert result["wnom_oc"]["n"] == 0
        assert result["irt_wnom"]["n"] > 0

    def test_within_party(
        self, irt_df: pl.DataFrame, wnom_df: pl.DataFrame, oc_df: pl.DataFrame
    ) -> None:
        # Add party to wnom for the join
        wnom_with_party = wnom_df.join(
            irt_df.select("legislator_slug", "party"), on="legislator_slug"
        )
        result = compute_within_party_correlations(irt_df, wnom_with_party, oc_df)
        assert "Republican" in result
        assert "Democrat" in result
        assert "irt_wnom" in result["Republican"]


# ── TestBuildComparisonTable ─────────────────────────────────────────────────


class TestBuildComparisonTable:
    def test_all_columns_present(
        self, irt_df: pl.DataFrame, wnom_df: pl.DataFrame, oc_df: pl.DataFrame
    ) -> None:
        table = build_comparison_table(irt_df, wnom_df, oc_df)
        expected = {
            "legislator_slug",
            "full_name",
            "party",
            "irt_score",
            "irt_rank",
            "wnom_score",
            "wnom_rank",
            "oc_score",
            "oc_rank",
            "max_rank_diff",
        }
        assert expected.issubset(set(table.columns))

    def test_ranks_computed(self, irt_df: pl.DataFrame, wnom_df: pl.DataFrame) -> None:
        table = build_comparison_table(irt_df, wnom_df)
        # rep_a has highest IRT score -> rank 1
        rep_a = table.filter(pl.col("legislator_slug") == "rep_a")
        assert rep_a["irt_rank"][0] == 1

    def test_max_rank_diff_without_oc(self, irt_df: pl.DataFrame, wnom_df: pl.DataFrame) -> None:
        table = build_comparison_table(irt_df, wnom_df, oc_df=None)
        assert "max_rank_diff" in table.columns
        # Should be |irt_rank - wnom_rank| since no OC
        for row in table.iter_rows(named=True):
            assert row["max_rank_diff"] == abs(row["irt_rank"] - row["wnom_rank"])


# ── TestParseEigenvalues ─────────────────────────────────────────────────────


class TestParseEigenvalues:
    def test_computes_pct_variance(self) -> None:
        eigen = pl.DataFrame({"dimension": [1, 2, 3], "eigenvalue": [5.0, 3.0, 2.0]})
        result = parse_eigenvalues(eigen)
        assert "pct_variance" in result.columns
        assert result["pct_variance"][0] == pytest.approx(50.0)

    def test_handles_unnamed_columns(self) -> None:
        eigen = pl.DataFrame({"col0": [1, 2], "col1": [4.0, 1.0]})
        result = parse_eigenvalues(eigen)
        assert "dimension" in result.columns
        assert "eigenvalue" in result.columns
