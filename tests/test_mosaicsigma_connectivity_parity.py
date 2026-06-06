"""Differential parity gate for the native connectivity primitives.

Three callables under test (PyO3 wrappers around `rustworkx_core::connectivity`):

  * `iai_mcp_native.graph.connected_components(indptr, indices, n_nodes) -> list[list[int]]`
  * `iai_mcp_native.graph.is_connected(indptr, indices, n_nodes) -> bool`
  * `iai_mcp_native.graph.selfloop_edges(indptr, indices, n_nodes) -> list[tuple[int, int]]`

Oracle: `networkx` (the dev-pin version captured below). Inputs follow the
canonical CSR triple used across the native graph layer:

  * `indptr` — int64 array of length n_nodes + 1, cumulative neighbour offsets
  * `indices` — int64 array of length 2*m for undirected graphs (each edge in
    both directions)
  * `n_nodes` — total node count

Empty-graph semantics: networkx 3.6.1 raises `NetworkXPointlessConcept` for
`is_connected(nx.Graph())`. The native wrapper mirrors this by raising
`ValueError` on n_nodes == 0; the test below pins the contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("networkx")
pytest.importorskip("numpy")
import networkx as nx


def _native_available() -> bool:
    try:
        from iai_mcp_native import graph  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(),
    reason="iai_mcp_native wheel not installed — run maturin build first",
)


FIXTURE_FILE = Path(__file__).parent / "fixtures" / "sigma_baseline.json"


@pytest.fixture(scope="module")
def fixtures() -> dict[str, dict]:
    raw = json.loads(FIXTURE_FILE.read_text())
    # Drop placeholder fixtures with no nodes (live_n2000 holds n=0 in the
    # frozen baseline — unusable for connectivity parity).
    return {
        name: payload
        for name, payload in raw["fixtures"].items()
        if payload.get("n", 0) > 0
    }


def _edges_to_csr(
    edges: list[tuple[int, int]],
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert an undirected edge list to (indptr, indices, n_nodes).

    Every edge appears in both endpoints' neighbour lists (undirected
    convention). Self-loops appear once: when u == v we add the single entry
    that the input pair represents to match `nx.selfloop_edges` semantics.
    """
    neighbours: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in edges:
        if u == v:
            neighbours[u].append(v)
        else:
            neighbours[u].append(v)
            neighbours[v].append(u)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    for i, nbrs in enumerate(neighbours):
        indptr[i + 1] = indptr[i] + len(nbrs)
    indices = np.fromiter(
        (v for row in neighbours for v in row),
        dtype=np.int64,
        count=int(indptr[-1]),
    )
    return indptr, indices, n_nodes


def _fixture_to_csr(payload: dict) -> tuple[np.ndarray, np.ndarray, int, "nx.Graph"]:
    n = int(payload["n"])
    edges = [(int(u), int(v)) for u, v in payload["edges"]]
    indptr, indices, n_nodes = _edges_to_csr(edges, n)
    g_nx = nx.Graph()
    g_nx.add_nodes_from(range(n))
    g_nx.add_edges_from(edges)
    return indptr, indices, n_nodes, g_nx


def test_connected_components_set_equality_on_fixtures(fixtures):
    from iai_mcp_native import graph as native_graph

    failures: list[tuple[str, set, set]] = []
    for name, payload in fixtures.items():
        indptr, indices, n_nodes, g_nx = _fixture_to_csr(payload)
        oracle = {frozenset(c) for c in nx.connected_components(g_nx)}
        ours_raw = native_graph.connected_components(indptr, indices, n_nodes)
        ours = {frozenset(c) for c in ours_raw}
        if ours != oracle:
            failures.append((name, ours, oracle))
    assert not failures, (
        f"{len(failures)} fixtures had divergent components:\n"
        + "\n".join(
            f"  fixture={n} ours={sorted(map(sorted, o))[:3]}... "
            f"oracle={sorted(map(sorted, x))[:3]}..."
            for n, o, x in failures[:5]
        )
    )


def test_is_connected_boolean_matches_on_fixtures(fixtures):
    from iai_mcp_native import graph as native_graph

    failures: list[tuple[str, bool, bool]] = []
    for name, payload in fixtures.items():
        indptr, indices, n_nodes, g_nx = _fixture_to_csr(payload)
        oracle = bool(nx.is_connected(g_nx))
        ours = bool(native_graph.is_connected(indptr, indices, n_nodes))
        if ours != oracle:
            failures.append((name, ours, oracle))
    assert not failures, (
        f"{len(failures)} fixtures had divergent is_connected:\n"
        + "\n".join(f"  fixture={n} ours={o} oracle={x}" for n, o, x in failures)
    )


def test_selfloop_edges_set_equality_on_synthetic_graph():
    """Synthetic graph carries explicit self-loops on nodes 0 and 1."""
    from iai_mcp_native import graph as native_graph

    edges = [(0, 0), (0, 1), (1, 1), (1, 2)]
    n_nodes = 3
    indptr, indices, _ = _edges_to_csr(edges, n_nodes)
    g_nx = nx.Graph()
    g_nx.add_nodes_from(range(n_nodes))
    g_nx.add_edges_from(edges)

    oracle = {tuple(sorted(pair)) for pair in nx.selfloop_edges(g_nx)}
    ours = {tuple(sorted(pair)) for pair in native_graph.selfloop_edges(indptr, indices, n_nodes)}
    assert ours == oracle, f"selfloop_edges mismatch: ours={ours} oracle={oracle}"
    # And the loop set is non-empty (sanity check that the synthetic input
    # actually exercised the self-loop branch).
    assert ours == {(0, 0), (1, 1)}


def test_karate_hand_trace_single_component(fixtures):
    """Karate club is a single 34-node connected component.

    Hand-trace: Zachary's karate club (Zachary 1977) has 34 members with 78
    interaction edges. The graph is well-known to be connected; both factions
    that emerged after the split remained linked by mutual friendship ties.
    """
    from iai_mcp_native import graph as native_graph

    payload = fixtures["karate"]
    indptr, indices, n_nodes, _ = _fixture_to_csr(payload)
    components = native_graph.connected_components(indptr, indices, n_nodes)
    assert len(components) == 1, f"expected 1 component, got {len(components)}"
    assert sorted(components[0]) == list(range(34))


def test_disconnected_graph_returns_multiple_components():
    """Two disjoint K3 cliques: 6 nodes, 2 components of size 3."""
    from iai_mcp_native import graph as native_graph

    # K3 on {0,1,2} and K3 on {3,4,5}
    edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    n_nodes = 6
    indptr, indices, _ = _edges_to_csr(edges, n_nodes)

    components = native_graph.connected_components(indptr, indices, n_nodes)
    assert len(components) == 2
    sizes = sorted(len(c) for c in components)
    assert sizes == [3, 3]
    sets = {frozenset(c) for c in components}
    assert sets == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}

    assert native_graph.is_connected(indptr, indices, n_nodes) is False


def test_is_connected_empty_graph_semantics():
    """Empty-graph contract pinned to networkx 3.6.1 behaviour.

    networkx 3.6.1 raises ``nx.NetworkXPointlessConcept`` on
    ``nx.is_connected(nx.Graph())`` ("Connectivity is undefined for the null
    graph."). The native wrapper mirrors this fail-loud posture by raising
    ``ValueError`` for n_nodes == 0 — the test asserts both libraries refuse
    to answer.
    """
    from iai_mcp_native import graph as native_graph

    g_nx = nx.Graph()
    with pytest.raises(nx.NetworkXPointlessConcept):
        nx.is_connected(g_nx)

    empty_indptr = np.array([0], dtype=np.int64)
    empty_indices = np.array([], dtype=np.int64)
    with pytest.raises(ValueError):
        native_graph.is_connected(empty_indptr, empty_indices, 0)
