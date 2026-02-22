"""Tests for network analysis module.

Uses a synthetic 8-legislator fixture: 4 Republicans with high intra-agreement,
4 Democrats with high intra-agreement, low cross-party agreement.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import polars as pl
import pytest

from analysis.network import (
    KAPPA_THRESHOLD_DEFAULT,
    _community_label,
    build_kappa_network,
    build_within_party_network,
    compare_communities_to_party,
    compute_centralities,
    compute_network_summary,
    detect_communities_multi_resolution,
    find_cross_party_bridge,
    run_threshold_sweep,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_slugs() -> list[str]:
    return [
        "rep_r1_1",
        "rep_r2_1",
        "rep_r3_1",
        "rep_r4_1",
        "rep_d1_1",
        "rep_d2_1",
        "rep_d3_1",
        "rep_d4_1",
    ]


@pytest.fixture
def synthetic_kappa(synthetic_slugs: list[str]) -> pl.DataFrame:
    """8x8 Kappa matrix: high within-party (0.7-0.9), low cross-party (0.1-0.2)."""
    n = len(synthetic_slugs)
    mat = np.eye(n)

    # R-R block (indices 0-3): high agreement
    for i in range(4):
        for j in range(i + 1, 4):
            val = 0.75 + 0.05 * (i + j) / 5  # 0.75-0.82
            mat[i, j] = val
            mat[j, i] = val

    # D-D block (indices 4-7): high agreement
    for i in range(4, 8):
        for j in range(i + 1, 8):
            val = 0.70 + 0.05 * (i + j) / 10  # 0.70-0.77
            mat[i, j] = val
            mat[j, i] = val

    # Cross-party: low agreement
    for i in range(4):
        for j in range(4, 8):
            val = 0.15 + 0.02 * (i + j) / 10  # 0.15-0.17
            mat[i, j] = val
            mat[j, i] = val

    # Build polars DataFrame matching expected format
    data: dict[str, list] = {"legislator_slug": synthetic_slugs}
    for col_idx, slug in enumerate(synthetic_slugs):
        data[slug] = mat[:, col_idx].tolist()

    return pl.DataFrame(data)


@pytest.fixture
def synthetic_ideal_points(synthetic_slugs: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "legislator_slug": synthetic_slugs,
            "xi_mean": [2.0, 1.5, 1.0, 0.5, -0.5, -1.0, -1.5, -2.0],
            "xi_sd": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            "xi_hdi_2.5": [1.4, 0.9, 0.4, -0.1, -1.1, -1.6, -2.1, -2.6],
            "xi_hdi_97.5": [2.6, 2.1, 1.6, 1.1, 0.1, -0.4, -0.9, -1.4],
            "full_name": [
                "Rep R1",
                "Rep R2",
                "Rep R3",
                "Rep R4",
                "Rep D1",
                "Rep D2",
                "Rep D3",
                "Rep D4",
            ],
            "party": ["Republican"] * 4 + ["Democrat"] * 4,
            "district": [f"Dist {i}" for i in range(1, 9)],
            "chamber": ["House"] * 8,
        }
    )


@pytest.fixture
def synthetic_clusters(synthetic_slugs: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "legislator_slug": synthetic_slugs,
            "cluster_k2": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def synthetic_loyalty(synthetic_slugs: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "legislator_slug": synthetic_slugs,
            "party": ["Republican"] * 4 + ["Democrat"] * 4,
            "n_agree": [80, 75, 70, 65, 60, 55, 50, 45],
            "n_contested_votes": [100] * 8,
            "loyalty_rate": [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45],
            "full_name": [
                "Rep R1",
                "Rep R2",
                "Rep R3",
                "Rep R4",
                "Rep D1",
                "Rep D2",
                "Rep D3",
                "Rep D4",
            ],
        }
    )


@pytest.fixture
def synthetic_graph(
    synthetic_kappa: pl.DataFrame,
    synthetic_ideal_points: pl.DataFrame,
    synthetic_clusters: pl.DataFrame,
    synthetic_loyalty: pl.DataFrame,
) -> tuple:
    """Build a graph and return (G, kappa_df, ideal_points, clusters, loyalty)."""
    G = build_kappa_network(
        synthetic_kappa,
        synthetic_ideal_points,
        synthetic_clusters,
        synthetic_loyalty,
        threshold=KAPPA_THRESHOLD_DEFAULT,
    )
    return G, synthetic_kappa, synthetic_ideal_points, synthetic_clusters, synthetic_loyalty


# ── TestBuildKappaNetwork ────────────────────────────────────────────────────


class TestBuildKappaNetwork:
    def test_correct_node_count(self, synthetic_graph: tuple) -> None:
        G = synthetic_graph[0]
        assert G.number_of_nodes() == 8

    def test_threshold_filtering(self, synthetic_graph: tuple) -> None:
        G = synthetic_graph[0]
        # Cross-party Kappa ~0.15 < 0.40 threshold, so no cross-party edges
        for u, v in G.edges():
            pu = G.nodes[u]["party"]
            pv = G.nodes[v]["party"]
            assert pu == pv, f"Cross-party edge found: {u} ({pu}) - {v} ({pv})"

    def test_within_party_edges_exist(self, synthetic_graph: tuple) -> None:
        G = synthetic_graph[0]
        # R-R Kappa ~0.75-0.82 > 0.40, so within-R edges should exist
        r_nodes = [n for n in G.nodes() if G.nodes[n]["party"] == "Republican"]
        r_edges = [(u, v) for u, v in G.edges() if u in r_nodes and v in r_nodes]
        assert len(r_edges) > 0

    def test_nan_handling(
        self,
        synthetic_ideal_points: pl.DataFrame,
    ) -> None:
        """NaN Kappa should result in no edge."""
        slugs = ["rep_a_1", "rep_b_1"]
        kappa_df = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "rep_a_1": [1.0, float("nan")],
                "rep_b_1": [float("nan"), 1.0],
            }
        )
        ip = synthetic_ideal_points.head(2).with_columns(pl.Series("legislator_slug", slugs))
        G = build_kappa_network(kappa_df, ip, threshold=0.0)
        assert G.number_of_edges() == 0

    def test_node_attributes(self, synthetic_graph: tuple) -> None:
        G = synthetic_graph[0]
        node = G.nodes["rep_r1_1"]
        assert node["party"] == "Republican"
        assert node["chamber"] == "House"
        assert abs(node["xi_mean"] - 2.0) < 0.01
        assert node["cluster"] == 0
        assert abs(node["loyalty_rate"] - 0.80) < 0.01
        assert node["full_name"] == "Rep R1"

    def test_no_self_loops(self, synthetic_graph: tuple) -> None:
        G = synthetic_graph[0]
        for u, v in G.edges():
            assert u != v

    def test_edge_weight_and_distance(self, synthetic_graph: tuple) -> None:
        G = synthetic_graph[0]
        for u, v, d in G.edges(data=True):
            assert "weight" in d
            assert "distance" in d
            assert d["weight"] > KAPPA_THRESHOLD_DEFAULT
            assert abs(d["distance"] - 1.0 / d["weight"]) < 1e-10


# ── TestComputeCentralities ──────────────────────────────────────────────────


class TestComputeCentralities:
    def test_all_columns_present(self, synthetic_graph: tuple) -> None:
        G = synthetic_graph[0]
        centralities = compute_centralities(G)
        expected_cols = {
            "legislator_slug",
            "full_name",
            "party",
            "xi_mean",
            "degree",
            "weighted_degree",
            "betweenness",
            "eigenvector",
            "closeness",
            "pagerank",
        }
        assert expected_cols.issubset(set(centralities.columns))

    def test_correct_row_count(self, synthetic_graph: tuple) -> None:
        G = synthetic_graph[0]
        centralities = compute_centralities(G)
        assert centralities.height == G.number_of_nodes()

    def test_range_check(self, synthetic_graph: tuple) -> None:
        G = synthetic_graph[0]
        centralities = compute_centralities(G)
        # Degree centrality should be in [0, 1]
        assert centralities["degree"].min() >= 0.0
        assert centralities["degree"].max() <= 1.0
        # Betweenness should be in [0, 1] (normalized)
        assert centralities["betweenness"].min() >= 0.0
        assert centralities["betweenness"].max() <= 1.0

    def test_star_graph_center_highest_betweenness(self) -> None:
        """In a star graph, the center should have the highest betweenness."""
        G = nx.star_graph(5)
        # Add required attributes
        for n in G.nodes():
            G.nodes[n]["party"] = "Republican"
            G.nodes[n]["xi_mean"] = 0.0
            G.nodes[n]["full_name"] = f"Node {n}"
        # Add weight/distance to edges
        for u, v in G.edges():
            G[u][v]["weight"] = 0.8
            G[u][v]["distance"] = 1.25
        centralities = compute_centralities(G)
        top = centralities.sort("betweenness", descending=True)
        # Node 0 is the center of a star_graph
        assert top["legislator_slug"][0] == 0


# ── TestCommunityDetection ───────────────────────────────────────────────────


class TestCommunityDetection:
    def test_all_resolutions_produce_results(self, synthetic_graph: tuple) -> None:
        G = synthetic_graph[0]
        resolutions = [0.5, 1.0, 2.0]
        partitions, res_df = detect_communities_multi_resolution(G, resolutions)
        assert len(partitions) == 3
        assert res_df.height == 3

    def test_modularity_range(self, synthetic_graph: tuple) -> None:
        G = synthetic_graph[0]
        _, res_df = detect_communities_multi_resolution(G, [1.0])
        mod = res_df["modularity"][0]
        assert -0.5 <= mod <= 1.0

    def test_two_clique_recovery(self, synthetic_graph: tuple) -> None:
        """With two clear cliques (R and D), community detection should find ~2 groups."""
        G = synthetic_graph[0]
        partitions, _ = detect_communities_multi_resolution(G, [1.0])
        partition = partitions[1.0]
        n_communities = len(set(partition.values()))
        # Should find 2 communities (one per party) since no cross-party edges
        assert n_communities == 2


# ── TestCompareCommunitiesToParty ────────────────────────────────────────────


class TestCompareCommunitiesToParty:
    def test_perfect_alignment(self, synthetic_graph: tuple) -> None:
        """When communities exactly match party, NMI should be 1.0."""
        G = synthetic_graph[0]
        # Manually create perfect-alignment partition
        partition = {}
        for n in G.nodes():
            partition[n] = 0 if G.nodes[n]["party"] == "Republican" else 1
        result = compare_communities_to_party(partition, G)
        assert result["nmi"] == pytest.approx(1.0, abs=0.01)
        assert result["ari"] == pytest.approx(1.0, abs=0.01)
        assert len(result["misclassified"]) == 0

    def test_random_has_low_nmi(self, synthetic_graph: tuple) -> None:
        """Random partition should have low NMI."""
        G = synthetic_graph[0]
        rng = np.random.default_rng(42)
        partition = {n: int(rng.integers(0, 4)) for n in G.nodes()}
        result = compare_communities_to_party(partition, G)
        # NMI should be low (< 0.5) for random assignment
        assert result["nmi"] < 0.5


# ── TestWithinPartyNetwork ───────────────────────────────────────────────────


class TestWithinPartyNetwork:
    def test_correct_party_filtering(
        self,
        synthetic_kappa: pl.DataFrame,
        synthetic_ideal_points: pl.DataFrame,
    ) -> None:
        G = build_within_party_network(
            synthetic_kappa, synthetic_ideal_points, "Republican", threshold=0.40
        )
        assert G.number_of_nodes() == 4
        for n in G.nodes():
            assert G.nodes[n]["party"] == "Republican"

    def test_democrat_filtering(
        self,
        synthetic_kappa: pl.DataFrame,
        synthetic_ideal_points: pl.DataFrame,
    ) -> None:
        G = build_within_party_network(
            synthetic_kappa, synthetic_ideal_points, "Democrat", threshold=0.40
        )
        assert G.number_of_nodes() == 4
        for n in G.nodes():
            assert G.nodes[n]["party"] == "Democrat"


# ── TestNetworkSummary ───────────────────────────────────────────────────────


class TestNetworkSummary:
    def test_summary_keys(self, synthetic_graph: tuple) -> None:
        G = synthetic_graph[0]
        summary = compute_network_summary(G)
        expected_keys = {
            "n_nodes",
            "n_edges",
            "density",
            "avg_clustering",
            "transitivity",
            "n_components",
            "assortativity_party",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_two_components(self, synthetic_graph: tuple) -> None:
        """With no cross-party edges, should have 2 connected components."""
        G = synthetic_graph[0]
        summary = compute_network_summary(G)
        assert summary["n_components"] == 2


# ── TestThresholdSweep ───────────────────────────────────────────────────────


class TestThresholdSweep:
    def test_sweep_returns_all_thresholds(
        self,
        synthetic_kappa: pl.DataFrame,
        synthetic_ideal_points: pl.DataFrame,
    ) -> None:
        thresholds = [0.10, 0.50, 0.90]
        sweep = run_threshold_sweep(synthetic_kappa, synthetic_ideal_points, thresholds)
        assert sweep.height == 3
        assert sweep["threshold"].to_list() == thresholds

    def test_higher_threshold_fewer_edges(
        self,
        synthetic_kappa: pl.DataFrame,
        synthetic_ideal_points: pl.DataFrame,
    ) -> None:
        sweep = run_threshold_sweep(synthetic_kappa, synthetic_ideal_points, [0.10, 0.90])
        edges = sweep["n_edges"].to_list()
        assert edges[0] >= edges[1]


# ── Fixture: Kappa with cross-party bridge ─────────────────────────────────


@pytest.fixture
def synthetic_kappa_with_bridge(synthetic_slugs: list[str]) -> pl.DataFrame:
    """8x8 Kappa matrix like synthetic_kappa but with one R-D pair at 0.35.

    rep_r4_1 (index 3) ↔ rep_d1_1 (index 4) has κ=0.35, making rep_r4_1
    the cross-party bridge at threshold 0.30.
    """
    n = len(synthetic_slugs)
    mat = np.eye(n)

    # R-R block (indices 0-3): high agreement
    for i in range(4):
        for j in range(i + 1, 4):
            val = 0.75 + 0.05 * (i + j) / 5
            mat[i, j] = val
            mat[j, i] = val

    # D-D block (indices 4-7): high agreement
    for i in range(4, 8):
        for j in range(i + 1, 8):
            val = 0.70 + 0.05 * (i + j) / 10
            mat[i, j] = val
            mat[j, i] = val

    # Cross-party: low agreement (below 0.30)
    for i in range(4):
        for j in range(4, 8):
            val = 0.15 + 0.02 * (i + j) / 10
            mat[i, j] = val
            mat[j, i] = val

    # Override one cross-party pair: rep_r4_1 (3) ↔ rep_d1_1 (4) = 0.35
    mat[3, 4] = 0.35
    mat[4, 3] = 0.35

    data: dict[str, list] = {"legislator_slug": synthetic_slugs}
    for col_idx, slug in enumerate(synthetic_slugs):
        data[slug] = mat[:, col_idx].tolist()

    return pl.DataFrame(data)


# ── TestFindCrossPartyBridge ───────────────────────────────────────────────


class TestFindCrossPartyBridge:
    """Tests for find_cross_party_bridge().

    Run: uv run pytest tests/test_network.py::TestFindCrossPartyBridge -v
    """

    def test_finds_bridge_at_low_threshold(
        self,
        synthetic_kappa_with_bridge: pl.DataFrame,
        synthetic_ideal_points: pl.DataFrame,
    ) -> None:
        """At κ=0.30, rep_r4_1 should be the cross-party bridge."""
        G, bridge_slug = find_cross_party_bridge(
            synthetic_kappa_with_bridge, synthetic_ideal_points, threshold=0.30
        )
        assert bridge_slug == "rep_r4_1"

    def test_no_bridge_at_high_threshold(
        self,
        synthetic_kappa_with_bridge: pl.DataFrame,
        synthetic_ideal_points: pl.DataFrame,
    ) -> None:
        """At κ=0.40, the cross-party edge (0.35) is below threshold → no bridge."""
        G, bridge_slug = find_cross_party_bridge(
            synthetic_kappa_with_bridge, synthetic_ideal_points, threshold=0.40
        )
        assert bridge_slug is None

    def test_removal_changes_components(
        self,
        synthetic_kappa_with_bridge: pl.DataFrame,
        synthetic_ideal_points: pl.DataFrame,
    ) -> None:
        """Removing the bridge should increase the component count."""
        G, bridge_slug = find_cross_party_bridge(
            synthetic_kappa_with_bridge, synthetic_ideal_points, threshold=0.30
        )
        assert bridge_slug is not None
        n_before = nx.number_connected_components(G)
        G_removed = G.copy()
        G_removed.remove_node(bridge_slug)
        n_after = nx.number_connected_components(G_removed)
        assert n_after > n_before


# ── TestCommunityLabel ─────────────────────────────────────────────────────


class TestCommunityLabel:
    """Tests for _community_label() helper.

    Run: uv run pytest tests/test_network.py::TestCommunityLabel -v
    """

    def test_high_republican_label(self) -> None:
        """>=80% Republican → 'Mostly Republican'."""
        comp = pl.DataFrame({"community": [0], "pct_republican": [96.0], "n_legislators": [87]})
        label = _community_label(0, comp)
        assert "Mostly Republican" in label
        assert "96%" in label
        assert "n=87" in label

    def test_high_democrat_label(self) -> None:
        """<=20% Republican → 'Mostly Democrat'."""
        comp = pl.DataFrame({"community": [0], "pct_republican": [5.0], "n_legislators": [25]})
        label = _community_label(0, comp)
        assert "Mostly Democrat" in label
        assert "95%" in label
        assert "n=25" in label

    def test_mixed_label(self) -> None:
        """50% Republican → 'Mixed'."""
        comp = pl.DataFrame({"community": [0], "pct_republican": [50.0], "n_legislators": [28]})
        label = _community_label(0, comp)
        assert "Mixed" in label
        assert "50%" in label

    def test_missing_community_fallback(self) -> None:
        """Unknown community ID → 'Community N'."""
        comp = pl.DataFrame({"community": [0], "pct_republican": [96.0], "n_legislators": [87]})
        label = _community_label(5, comp)
        assert label == "Community 5"

    def test_none_composition_fallback(self) -> None:
        """None composition → 'Community N'."""
        label = _community_label(3, None)
        assert label == "Community 3"
