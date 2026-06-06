"""Differential parity for iai_mcp_native.graph.average_shortest_path_length.

Rust composer over rustworkx_core::shortest_path::distance_matrix with a
largest-connected-component guard matching the Python pattern in
src/iai_mcp/sigma.py. This module is the gatekeeper: every
supported fixture must match the networkx oracle within |delta| <= 1e-9
on the largest connected component, and the small hand-built graphs
(single node, two-node path, five-node path, disjoint K3 pair) must
match the hand-computed value to floating-point exactness.

Inputs to the Rust function are a pair of CSR-style numpy arrays
(``indptr`` and ``indices``) plus the node count. The CSR encodes the
undirected adjacency with both directions materialised — the helper
:func:`_to_csr_undirected` builds this layout from a list of edges.
"""
from __future__ import annotations

import json
import pathlib

import pytest

# Skip the whole module when the native wheel or networkx/numpy is missing.
pytest.importorskip("networkx")
pytest.importorskip("numpy")
pytest.importorskip("iai_mcp_native")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from iai_mcp_native import graph as native_graph  # noqa: E402


FIXTURE_PATH = (
    pathlib.Path(__file__).parent / "fixtures" / "sigma_baseline.json"
)


def _load_fixtures() -> dict:
    """Return the fixtures dict from the locked sigma_baseline.json."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]


# Fixtures that ship with an `edges` array. live_n2000 carries empty edges
# (verified at σ assembly) and is intentionally excluded; the
# remaining 8 cover empirical small graphs (karate, les_miserables),
# Erdos-Renyi random baselines (er_200, er_500, er_1000),
# tiny ring lattices (tiny_10_ws_k4, tiny_20_ws_p010), and one larger
# regular ring (ws_2500_k4_p0).
APSL_FIXTURE_KEYS = [
    "karate",
    "les_miserables",
    "er_200",
    "er_500",
    "er_1000",
    "tiny_10_ws_k4",
    "tiny_20_ws_p010",
    "ws_2500_k4_p0",
]


def _to_csr_undirected(
    n_nodes: int, edges: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise a CSR-style adjacency for undirected graphs.

    Every edge ``(u, v)`` produces both ``u -> v`` and ``v -> u`` entries
    so the resulting CSR represents an undirected graph; self-loops are
    listed once, double-listed edges collapse via the consuming
    ``UnGraphMap::add_edge`` (idempotent) on the Rust side.

    Returns ``(indptr, indices)`` as ``np.int64`` arrays sized ``(n+1)``
    and ``(2 * E_non_selfloop + E_selfloop)`` respectively.
    """
    # Bucket neighbours per source.
    neighbours: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in edges:
        neighbours[u].append(v)
        if u != v:
            neighbours[v].append(u)

    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    flat: list[int] = []
    for i, nbrs in enumerate(neighbours):
        flat.extend(nbrs)
        indptr[i + 1] = len(flat)
    indices = np.asarray(flat, dtype=np.int64)
    return indptr, indices


def _networkx_apsl_on_largest_cc(
    n_nodes: int, edges: list[tuple[int, int]]
) -> float:
    """Oracle: networkx APSL on the largest connected component.

    Mirrors sigma.py's _largest_cc — networkx raises on disconnected
    input to ``average_shortest_path_length``; the wrapper extracts the
    largest connected component first.
    """
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(edges)
    if g.number_of_nodes() == 0:
        return 0.0
    if nx.is_connected(g):
        return float(nx.average_shortest_path_length(g))
    largest = max(nx.connected_components(g), key=len)
    sub = g.subgraph(largest).copy()
    if sub.number_of_nodes() <= 1:
        return 0.0
    return float(nx.average_shortest_path_length(sub))


# --------------------------------------------------------- parity assertions


def test_apsl_matches_networkx_on_connected_fixtures() -> None:
    """Differential parity vs networkx on every shipped fixture.

    Failure-list idiom: collect drifts per fixture, raise at the end with
    the full diagnostic. Tolerance |delta| <= 1e-9 — every fixture should
    match to floating-point exactness since both implementations evaluate
    BFS hop counts on the same edge set and average over the same
    denominator.
    """
    fixtures = _load_fixtures()
    drifts: list[str] = []
    for key in APSL_FIXTURE_KEYS:
        fx = fixtures[key]
        n = int(fx["n"])
        edges = [tuple(e) for e in fx["edges"]]
        oracle = _networkx_apsl_on_largest_cc(n, edges)
        indptr, indices = _to_csr_undirected(n, edges)
        ours = native_graph.average_shortest_path_length(indptr, indices, n)
        delta = abs(oracle - ours)
        if delta > 1e-9:
            drifts.append(
                f"{key}: oracle={oracle:.12f} ours={ours:.12f} "
                f"|delta|={delta:.2e}"
            )
    assert not drifts, (
        "APSL parity drifts vs networkx oracle:\n  "
        + "\n  ".join(drifts)
    )


def test_apsl_largest_cc_guard_on_disconnected_input() -> None:
    """Disjoint K3 pair — APSL on the largest CC must match networkx.

    Two disjoint triangles (6 nodes, 6 edges). ``nx.is_connected`` is
    False; the oracle is computed by extracting one of the two K3
    components (both have size 3, networkx picks the lexicographically
    smaller) and calling ``average_shortest_path_length`` on the
    subgraph. The Rust composer's internal largest-CC guard must produce
    the same value.
    """
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    n = 6
    oracle = _networkx_apsl_on_largest_cc(n, edges)
    indptr, indices = _to_csr_undirected(n, edges)
    ours = native_graph.average_shortest_path_length(indptr, indices, n)
    delta = abs(oracle - ours)
    assert delta <= 1e-9, (
        f"disjoint-K3 drift: oracle={oracle!r} ours={ours!r} "
        f"|delta|={delta:.2e}"
    )
    # Sanity: on a K3 every pair is at distance 1, so APSL == 1.0.
    assert ours == pytest.approx(1.0, abs=1e-12), (
        f"K3 APSL should be exactly 1.0; got {ours!r}"
    )


def test_apsl_single_node_returns_zero() -> None:
    """Single-node graph: APSL == 0.0 (matches networkx convention)."""
    indptr = np.array([0, 0], dtype=np.int64)
    indices = np.zeros(0, dtype=np.int64)
    ours = native_graph.average_shortest_path_length(indptr, indices, 1)
    assert ours == 0.0


def test_apsl_two_node_path() -> None:
    """Two-node path graph: d(0,1) = 1; APSL == 1.0 exactly."""
    edges = [(0, 1)]
    indptr, indices = _to_csr_undirected(2, edges)
    ours = native_graph.average_shortest_path_length(indptr, indices, 2)
    assert ours == pytest.approx(1.0, abs=1e-12)


def test_apsl_path_graph_5nodes() -> None:
    """5-node path 0-1-2-3-4: APSL == 2.0 by hand-trace.

    Unordered pairwise distances::

        d(0,1)=1, d(0,2)=2, d(0,3)=3, d(0,4)=4,
        d(1,2)=1, d(1,3)=2, d(1,4)=3,
        d(2,3)=1, d(2,4)=2,
        d(3,4)=1
        ----
        sum = 1+2+3+4 + 1+2+3 + 1+2 + 1 = 20

    Over 10 unordered pairs that gives APSL = 20/10 = 2.0. Equivalently
    in ordered-pair form (the kernel iterates ordered i!=j): 2·20 = 40
    over N·(N-1) = 5·4 = 20 gives 40/20 = 2.0. FP-exact equality is the
    correct gate — every term is an integer hop count, the divisor is an
    integer count of pairs, and IEEE 754 represents 2.0 exactly.

    A common mistake is to mix the unordered-pair sum (20) with the
    ordered-pair denominator (5·4 = 20), which would falsely yield 1.0.
    This test is the gate that catches that off-by-2 algebra.
    """
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    indptr, indices = _to_csr_undirected(5, edges)
    ours = native_graph.average_shortest_path_length(indptr, indices, 5)
    assert ours == pytest.approx(2.0, abs=1e-12), (
        f"5-node path APSL should be exactly 2.0; got {ours!r} "
        f"(an off-by-2 denominator would yield 1.0 or 4.0 instead)"
    )
