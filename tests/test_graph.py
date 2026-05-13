"""Tests for iai_mcp.graph ( dual-library wrapper, 2-hop spread)."""
from __future__ import annotations

from uuid import uuid4

import pytest

from iai_mcp.graph import IGRAPH_THRESHOLD, MemoryGraph, _HAS_IGRAPH


def test_small_graph_uses_networkx() -> None:
    g = MemoryGraph()
    for _ in range(10):
        g.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    assert g.backend == "networkx"


@pytest.mark.skipif(not _HAS_IGRAPH, reason="igraph optional on some boxes")
def test_large_graph_switches_to_igraph() -> None:
    g = MemoryGraph()
    for _ in range(IGRAPH_THRESHOLD + 1):
        g.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    assert g.backend == "igraph"


def test_backend_stays_networkx_just_below_threshold() -> None:
    g = MemoryGraph()
    for _ in range(IGRAPH_THRESHOLD - 1):
        g.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    assert g.backend == "networkx"


def test_two_hop_reaches_exactly_two_hops() -> None:
    """: linear chain A-B-C-D seeded at A returns {B, C} -- D is 3 hops."""
    g = MemoryGraph()
    a, b, c, d = uuid4(), uuid4(), uuid4(), uuid4()
    for n in (a, b, c, d):
        g.add_node(n, community_id=None, embedding=[0.0] * 384)
    g.add_edge(a, b)
    g.add_edge(b, c)
    g.add_edge(c, d)

    reached = set(g.two_hop_neighborhood([a], top_k=5))
    assert b in reached
    assert c in reached
    assert d not in reached  # 3 hops away
    assert a not in reached  # seed excluded


def test_two_hop_multiple_seeds_deduped() -> None:
    g = MemoryGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    for n in (a, b, c):
        g.add_node(n, community_id=None, embedding=[0.0] * 384)
    g.add_edge(a, b)
    g.add_edge(b, c)
    # Both a and c as seeds: 2-hop from a reaches {b,c}, from c reaches {b,a};
    # union minus seeds should be {b}.
    reached = set(g.two_hop_neighborhood([a, c], top_k=5))
    assert reached == {b}


def test_two_hop_empty_seeds_returns_empty_list() -> None:
    g = MemoryGraph()
    assert g.two_hop_neighborhood([], top_k=5) == []


def test_centrality_hub_beats_leaves() -> None:
    """5-node star: hub's betweenness strictly greater than any leaf's."""
    g = MemoryGraph()
    hub = uuid4()
    leaves = [uuid4() for _ in range(4)]
    g.add_node(hub, community_id=None, embedding=[0.0] * 384)
    for leaf in leaves:
        g.add_node(leaf, community_id=None, embedding=[0.0] * 384)
        g.add_edge(hub, leaf)
    c = g.centrality()
    for leaf in leaves:
        assert c[hub] > c[leaf]


def test_centrality_no_edges_all_zero() -> None:
    g = MemoryGraph()
    for _ in range(5):
        g.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    c = g.centrality()
    assert all(v == 0.0 for v in c.values())
    assert len(c) == 5


def test_get_embedding_returns_stored_vector() -> None:
    g = MemoryGraph()
    nid = uuid4()
    emb = [1.0] + [0.0] * 383
    g.add_node(nid, community_id=None, embedding=emb)
    assert g.get_embedding(nid) == emb
    assert g.get_embedding(uuid4()) is None


def test_rich_club_coefficient_on_star_graph() -> None:
    """Star has hub with degree 4; coefficient well-defined."""
    g = MemoryGraph()
    hub = uuid4()
    leaves = [uuid4() for _ in range(4)]
    g.add_node(hub, community_id=None, embedding=[0.0] * 384)
    for leaf in leaves:
        g.add_node(leaf, community_id=None, embedding=[0.0] * 384)
        g.add_edge(hub, leaf)
    # Should not raise; returns a float.
    coef = g.rich_club_coefficient()
    assert isinstance(coef, float)
    assert coef >= 0.0
