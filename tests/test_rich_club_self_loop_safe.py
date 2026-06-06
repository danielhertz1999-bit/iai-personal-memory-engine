"""RED-witness coverage for graph.rich_club_coefficient self-loop safety.

NetworkX 3.6.1 raises bare ``Exception`` (NOT
``NetworkXError`` or ``NotImplementedError`` -- empirically verified) with
message ``"rich_club_coefficient is not implemented for graphs with self
loops."`` when the input graph contains self-loops. The existing wrapper at
``src/iai_mcp/graph.py`` catches only ``(ZeroDivisionError,
NetworkXError)``, so the exception leaks through ``EssentialVariableTracker``
and pollutes the bench run.log (6 traces per run).

The fix in ``graph.py`` filters self-loops on a graph copy before the
NetworkX call. These tests cover:

- ``test_self_loop_graph_does_not_raise``: must complete without raising
  any exception class.
- ``test_filter_idempotent_on_no_loop_graph``: deterministic across two
  calls and matches direct NetworkX result.
- ``test_all_self_loop_returns_zero``: degenerate all-self-loop graph
  returns 0.0 (post-filter is edgeless).
- ``test_mixed_graph_equals_filtered_graph_result``: filter is
  semantically equivalent to removing self-loops by hand.
"""
from __future__ import annotations

from uuid import uuid4

import networkx as nx

from iai_mcp.graph import MemoryGraph


def _make_graph_with_edges(edges: list[tuple]) -> tuple[MemoryGraph, list]:
    """Build a MemoryGraph with the given edge list. Edges are indices into a
    fresh list of UUIDs; (i, i) becomes a self-loop on node i.

    Returns (graph, uuid_list) so tests can address individual nodes by index.
    """
    node_count = max(max(u, v) for u, v in edges) + 1
    uuids = [uuid4() for _ in range(node_count)]
    g = MemoryGraph()
    for uid in uuids:
        g.add_node(uid, community_id=None, embedding=[0.0] * 384)
    for u, v in edges:
        g.add_edge(uuids[u], uuids[v])
    return g, uuids


def test_self_loop_graph_does_not_raise() -> None:
    """Graph with self-loops must NOT leak any exception.

    Pre-fix: NetworkX raises bare ``Exception`` (not in the catch tuple) ->
    propagates -> ``EssentialVariableTracker hook failed`` warning in
    bench run.log.

    Post-fix: filter on a copy removes the self-loops; computation succeeds
    and returns a float.
    """
    g, _ = _make_graph_with_edges([(0, 0), (0, 1), (1, 2), (2, 0)])
    # Must not raise any exception (RED-witness: pre-fix this leaks
    # builtins.Exception via NetworkX).
    result = g.rich_club_coefficient()
    assert isinstance(result, float)
    assert result >= 0.0


def test_filter_idempotent_on_no_loop_graph() -> None:
    """No-self-loop graph: two calls return same value; matches NX oracle.

    Comparing two calls catches any in-place mutation that would change
    the graph topology across invocations. Comparing against a NetworkX
    oracle built fresh from the public iter_edges_with_weight surface is
    the differential parity check.
    """
    # 4-node cycle: 0-1-2-3-0 -- no self-loops.
    g, _ = _make_graph_with_edges([(0, 1), (1, 2), (2, 3), (3, 0)])

    r1 = g.rich_club_coefficient(k_threshold=1)
    r2 = g.rich_club_coefficient(k_threshold=1)
    assert r1 == r2, f"non-deterministic: {r1} vs {r2}"

    # Build a NetworkX oracle from the public read surface.
    oracle = nx.Graph()
    for u, v, w in g.iter_edges_with_weight():
        oracle.add_edge(str(u), str(v), weight=w)
    reference = nx.rich_club_coefficient(oracle, normalized=False).get(1, 0.0)
    assert r1 == float(reference), (
        f"helper diverged from direct NetworkX: {r1} vs {reference}"
    )


def test_all_self_loop_returns_zero() -> None:
    """All-self-loop graph: post-filter has zero edges -> 0.0 returned.

    A graph with N nodes and only self-loops has ``number_of_edges() == N``,
    so the existing short-circuit at line 200-204 does NOT fire on the
    original. The new ``G_for_rc.number_of_edges() == 0`` guard after the
    filter is what catches the degenerate case.
    """
    g, _ = _make_graph_with_edges([(0, 0), (1, 1), (2, 2)])
    result = g.rich_club_coefficient()
    assert result == 0.0, f"expected exact 0.0, got {result!r}"


def test_mixed_graph_equals_filtered_graph_result() -> None:
    """Filter is semantically equivalent to removing self-loops by hand.

    Graph A: edges = [(0,0), (0,1), (1,2), (2,0), (1,1)] (with self-loops)
    Graph B: edges = [(0,1), (1,2), (2,0)] (same edges minus self-loops)

    Both graphs must produce the same rich_club_coefficient at k=1, which
    proves the helper's filter logic matches the explicit hand-filtered
    case.

    NOTE: ``k_threshold=1`` is passed explicitly. The default branch
    computes ``k_threshold`` from the post-strip degree distribution
    which counts self-loops differently than the pre-strip distribution
    would -- graphs A and B would pick different thresholds even though
    the filtered graph is identical.
    """
    g_a, _ = _make_graph_with_edges([(0, 0), (0, 1), (1, 2), (2, 0), (1, 1)])
    g_b, _ = _make_graph_with_edges([(0, 1), (1, 2), (2, 0)])
    coef_a = g_a.rich_club_coefficient(k_threshold=1)
    coef_b = g_b.rich_club_coefficient(k_threshold=1)
    assert coef_a == coef_b, (
        f"with-self-loops {coef_a} != without-self-loops {coef_b}"
    )
