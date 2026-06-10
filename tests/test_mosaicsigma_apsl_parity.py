from __future__ import annotations

import json
import pathlib

import pytest

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
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]


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


def test_apsl_matches_networkx_on_connected_fixtures() -> None:
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
    assert ours == pytest.approx(1.0, abs=1e-12), (
        f"K3 APSL should be exactly 1.0; got {ours!r}"
    )


def test_apsl_single_node_returns_zero() -> None:
    indptr = np.array([0, 0], dtype=np.int64)
    indices = np.zeros(0, dtype=np.int64)
    ours = native_graph.average_shortest_path_length(indptr, indices, 1)
    assert ours == 0.0


def test_apsl_two_node_path() -> None:
    edges = [(0, 1)]
    indptr, indices = _to_csr_undirected(2, edges)
    ours = native_graph.average_shortest_path_length(indptr, indices, 2)
    assert ours == pytest.approx(1.0, abs=1e-12)


def test_apsl_path_graph_5nodes() -> None:
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    indptr, indices = _to_csr_undirected(5, edges)
    ours = native_graph.average_shortest_path_length(indptr, indices, 5)
    assert ours == pytest.approx(2.0, abs=1e-12), (
        f"5-node path APSL should be exactly 2.0; got {ours!r} "
        f"(an off-by-2 denominator would yield 1.0 or 4.0 instead)"
    )
