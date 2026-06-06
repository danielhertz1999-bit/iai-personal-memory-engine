"""Tests for iai_mcp.graph (adjacency-dict backend, 2-hop spread)."""
from __future__ import annotations

from uuid import uuid4

from iai_mcp.graph import MemoryGraph


def test_small_graph_constructs() -> None:
    g = MemoryGraph()
    for _ in range(10):
        g.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    assert g.node_count() == 10


def test_large_graph_constructs() -> None:
    g = MemoryGraph()
    for _ in range(501):
        g.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    assert g.node_count() == 501


def test_n_just_below_500_constructs() -> None:
    g = MemoryGraph()
    for _ in range(499):
        g.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    assert g.node_count() == 499


def test_two_hop_reaches_exactly_two_hops() -> None:
    """linear chain A-B-C-D seeded at A returns {B, C} -- D is 3 hops."""
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


def test_edge_attr_symmetric() -> None:
    """add_edge stores a SHARED dict between _adj[u][v] and _adj[v][u].

    Verifies the symmetry invariant: writing the weight in one direction
    propagates atomically to the other. Without this, Hebbian-strength
    updates would silently asymmetrize over long-running sessions.
    """
    g = MemoryGraph()
    u, v = uuid4(), uuid4()
    g.add_node(u, None, [0.0] * 384)
    g.add_node(v, None, [0.0] * 384)
    g.add_edge(u, v, weight=0.5)
    assert g._adj[str(u)][str(v)] is g._adj[str(v)][str(u)]
    g._adj[str(u)][str(v)]["weight"] = 9.9
    assert g._adj[str(v)][str(u)]["weight"] == 9.9


def test_iter_edges_once_only() -> None:
    """iter_edges_with_weight emits each undirected edge exactly once.

    Adjacency-list storage double-emits each (u, v) pair (once from u's
    neighbor list, once from v's). The canonical-pair ``u <= v`` filter
    inside iter_edges_with_weight collapses the two visits into one
    so consumers see the same once-only semantic a hash-mapped graph
    would emit.
    """
    g = MemoryGraph()
    u, v, w = uuid4(), uuid4(), uuid4()
    for n in (u, v, w):
        g.add_node(n, None, [0.0] * 384)
    g.add_edge(u, v)
    g.add_edge(v, w)
    g.add_edge(u, w)
    edges = list(g.iter_edges_with_weight())
    assert len(edges) == 3


def test_self_loop_preserved_in_storage() -> None:
    """Self-loops are stored once in _adj, emitted once by the iterator,
    and stripped from CSR.
    """
    g = MemoryGraph()
    u = uuid4()
    g.add_node(u, None, [0.0] * 384)
    g.add_edge(u, u, weight=0.5)
    assert str(u) in g._adj[str(u)]
    edges = list(g.iter_edges_with_weight())
    assert sum(1 for src, dst, _ in edges if src == dst == u) == 1
    _indptr, indices, _data = g.to_csr_arrays()
    # Self-loop stripped at CSR construction.
    assert len(indices) == 0
