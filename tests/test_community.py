"""Tests for iai_mcp.community (bootstrap, stable UUIDs, /04)."""
from __future__ import annotations

import random
from uuid import uuid4

from iai_mcp.community import (
    CommunityAssignment,
    MAX_TOP_COMMUNITIES,
    MID_N_LEIDEN,
    MODULARITY_FLOOR,
    REFRESH_DELTA,
    SMALL_N_FLAT,
    UUID_ROTATE_COSINE,
    detect_communities,
    needs_refresh,
)
from iai_mcp.graph import MemoryGraph


def _random_emb(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(384)]


def test_small_n_flat_single_community() -> None:
    """N < SMALL_N_FLAT -> flat, single community."""
    g = MemoryGraph()
    for i in range(50):
        g.add_node(uuid4(), community_id=None, embedding=_random_emb(i))
    a = detect_communities(g, prior=None)
    assert a.backend == "flat"
    assert len(set(a.node_to_community.values())) == 1
    assert a.modularity == 0.0


def test_two_cliques_produce_multiple_communities() -> None:
    """2 dense cliques of 150 nodes -> N=300, Leiden should find Q >= 0.2."""
    g = MemoryGraph()
    clique_a = [uuid4() for _ in range(150)]
    clique_b = [uuid4() for _ in range(150)]
    for i, n in enumerate(clique_a):
        g.add_node(n, community_id=None, embedding=_random_emb(i))
    for i, n in enumerate(clique_b):
        g.add_node(n, community_id=None, embedding=_random_emb(10_000 + i))
    for i in range(150):
        for j in range(i + 1, 150):
            g.add_edge(clique_a[i], clique_a[j])
            g.add_edge(clique_b[i], clique_b[j])
    a = detect_communities(g, prior=None)
    assert a.backend.startswith("leiden")
    assert a.modularity >= MODULARITY_FLOOR
    assert len(set(a.node_to_community.values())) >= 2


def test_stable_uuids_on_identical_rerun() -> None:
    """identical graphs rerun with prior -> zero UUID churn."""
    g = MemoryGraph()
    clique_a = [uuid4() for _ in range(150)]
    clique_b = [uuid4() for _ in range(150)]
    for i, n in enumerate(clique_a):
        g.add_node(n, community_id=None, embedding=_random_emb(i))
    for i, n in enumerate(clique_b):
        g.add_node(n, community_id=None, embedding=_random_emb(10_000 + i))
    for i in range(150):
        for j in range(i + 1, 150):
            g.add_edge(clique_a[i], clique_a[j])
            g.add_edge(clique_b[i], clique_b[j])
    first = detect_communities(g, prior=None)
    second = detect_communities(g, prior=first)
    for node, comm_first in first.node_to_community.items():
        assert second.node_to_community[node] == comm_first


def test_top_communities_capped_at_seven() -> None:
    """MAX_TOP_COMMUNITIES = 7 enforced on level 1 output."""
    g = MemoryGraph()
    for i in range(SMALL_N_FLAT + 10):
        g.add_node(uuid4(), community_id=None, embedding=_random_emb(i))
    nodes = list(g.iter_nodes())
    for k in range(0, len(nodes) - 1, 20):
        for j in range(k, min(k + 20, len(nodes) - 1)):
            g.add_edge(nodes[j], nodes[j + 1])
    a = detect_communities(g, prior=None)
    assert len(a.top_communities) <= MAX_TOP_COMMUNITIES


def test_mid_regions_exposes_community_members() -> None:
    """level 2: mid_regions maps community UUID -> member UUIDs."""
    g = MemoryGraph()
    nodes = [uuid4() for _ in range(50)]
    for i, n in enumerate(nodes):
        g.add_node(n, community_id=None, embedding=_random_emb(i))
    a = detect_communities(g, prior=None)
    total_members = sum(len(members) for members in a.mid_regions.values())
    assert total_members == 50


def test_needs_refresh_threshold() -> None:
    """|Δ Q| > 0.05 -> refresh, else stable."""
    prior = CommunityAssignment(modularity=0.30)
    assert needs_refresh(prior, 0.36) is True  # Δ = 0.06 > 0.05
    assert needs_refresh(prior, 0.31) is False  # Δ = 0.01 < 0.05
    assert needs_refresh(prior, 0.24) is True  # Δ = 0.06 > 0.05 (negative side)
    # Boundary: Δ == 0.05 is NOT > 0.05 -> False (strict inequality).
    assert needs_refresh(prior, 0.35) is False


def test_empty_graph_returns_empty_assignment() -> None:
    g = MemoryGraph()
    a = detect_communities(g, prior=None)
    assert a.backend == "flat"
    assert a.node_to_community == {}
    assert a.community_centroids == {}


def test_constants_exposed() -> None:
    """Named constants are importable (verifies the grep acceptance criteria)."""
    assert SMALL_N_FLAT == 200
    assert MID_N_LEIDEN == 500
    assert MODULARITY_FLOOR == 0.2
    assert REFRESH_DELTA == 0.05
    assert UUID_ROTATE_COSINE == 0.7
    assert MAX_TOP_COMMUNITIES == 7


def test_mid_n_non_modular_falls_back_to_flat() -> None:
    """SMALL_N_FLAT <= N < MID_N_LEIDEN with Q < 0.2 -> flat fallback."""
    g = MemoryGraph()
    # 250 nodes fully connected -> a clique, Leiden will produce Q ~ 0.0
    nodes = [uuid4() for _ in range(250)]
    for i, n in enumerate(nodes):
        g.add_node(n, community_id=None, embedding=_random_emb(i))
    for i in range(250):
        for j in range(i + 1, 250):
            g.add_edge(nodes[i], nodes[j])
    a = detect_communities(g, prior=None)
    # Fully-connected graph has no community structure -> fall back to flat.
    assert a.backend == "flat"


def test_mid_regions_count_matches_community_count() -> None:
    """mid_regions has exactly one entry per distinct community."""
    g = MemoryGraph()
    clique_a = [uuid4() for _ in range(150)]
    clique_b = [uuid4() for _ in range(150)]
    for i, n in enumerate(clique_a + clique_b):
        g.add_node(n, community_id=None, embedding=_random_emb(i))
    for i in range(150):
        for j in range(i + 1, 150):
            g.add_edge(clique_a[i], clique_a[j])
            g.add_edge(clique_b[i], clique_b[j])
    a = detect_communities(g, prior=None)
    assert len(a.mid_regions) == len(set(a.node_to_community.values()))
