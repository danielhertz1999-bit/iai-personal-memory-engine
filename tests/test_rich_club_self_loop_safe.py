from __future__ import annotations

from uuid import uuid4

import networkx as nx

from iai_mcp.graph import MemoryGraph

def _make_graph_with_edges(edges: list[tuple]) -> tuple[MemoryGraph, list]:
    node_count = max(max(u, v) for u, v in edges) + 1
    uuids = [uuid4() for _ in range(node_count)]
    g = MemoryGraph()
    for uid in uuids:
        g.add_node(uid, community_id=None, embedding=[0.0] * 384)
    for u, v in edges:
        g.add_edge(uuids[u], uuids[v])
    return g, uuids

def test_self_loop_graph_does_not_raise() -> None:
    g, _ = _make_graph_with_edges([(0, 0), (0, 1), (1, 2), (2, 0)])
    result = g.rich_club_coefficient()
    assert isinstance(result, float)
    assert result >= 0.0

def test_filter_idempotent_on_no_loop_graph() -> None:
    g, _ = _make_graph_with_edges([(0, 1), (1, 2), (2, 3), (3, 0)])

    r1 = g.rich_club_coefficient(k_threshold=1)
    r2 = g.rich_club_coefficient(k_threshold=1)
    assert r1 == r2, f"non-deterministic: {r1} vs {r2}"

    oracle = nx.Graph()
    for u, v, w in g.iter_edges_with_weight():
        oracle.add_edge(str(u), str(v), weight=w)
    reference = nx.rich_club_coefficient(oracle, normalized=False).get(1, 0.0)
    assert r1 == float(reference), (
        f"helper diverged from direct NetworkX: {r1} vs {reference}"
    )

def test_all_self_loop_returns_zero() -> None:
    g, _ = _make_graph_with_edges([(0, 0), (1, 1), (2, 2)])
    result = g.rich_club_coefficient()
    assert result == 0.0, f"expected exact 0.0, got {result!r}"

def test_mixed_graph_equals_filtered_graph_result() -> None:
    g_a, _ = _make_graph_with_edges([(0, 0), (0, 1), (1, 2), (2, 0), (1, 1)])
    g_b, _ = _make_graph_with_edges([(0, 1), (1, 2), (2, 0)])
    coef_a = g_a.rich_club_coefficient(k_threshold=1)
    coef_b = g_b.rich_club_coefficient(k_threshold=1)
    assert coef_a == coef_b, (
        f"with-self-loops {coef_a} != without-self-loops {coef_b}"
    )
