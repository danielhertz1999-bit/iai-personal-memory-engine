
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

sys.path.insert(0, str(Path(__file__).parent))
from memorygraph_adj_spike import _AdjacencyBackend  # noqa: E402

from iai_mcp.graph import MemoryGraph  # noqa: E402  # current nx-backed

FIXTURES: tuple[str, ...] = (
    "karate",
    "les_miserables",
    "er_200",
    "er_500",
    "er_1000",
    "ws_2500_k4_p0",
)
FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "sigma_baseline.json"
)


def deterministic_uuid(node_index: int) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, str(node_index))


def build_both(
    edges: list[list[int]],
) -> tuple[MemoryGraph, _AdjacencyBackend]:
    nx_graph = MemoryGraph()
    adj_graph = _AdjacencyBackend()
    nodes = sorted({n for edge in edges for n in edge[:2]})
    for n in nodes:
        uid = deterministic_uuid(n)
        nx_graph.add_node(uid, community_id=None, embedding=[0.0] * 384)
        adj_graph.add_node(uid, community_id=None, embedding=[0.0] * 384)
    for edge in edges:
        u_int, v_int = edge[0], edge[1]
        w = float(edge[2]) if len(edge) > 2 else 1.0
        u_uuid = deterministic_uuid(u_int)
        v_uuid = deterministic_uuid(v_int)
        nx_graph.add_edge(u_uuid, v_uuid, weight=w)
        adj_graph.add_edge(u_uuid, v_uuid, weight=w)
    return nx_graph, adj_graph


def diff_csr_triple(
    name: str,
    nx_triple: tuple[np.ndarray, np.ndarray, np.ndarray],
    adj_triple: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> str | None:
    n_indptr, n_indices, n_data = nx_triple
    a_indptr, a_indices, a_data = adj_triple
    for arr_name, n_arr, a_arr in (
        ("indptr", n_indptr, a_indptr),
        ("indices", n_indices, a_indices),
        ("data", n_data, a_data),
    ):
        if n_arr.shape != a_arr.shape:
            return (
                f"FAIL: {name} {arr_name} shape diverges: "
                f"nx={n_arr.shape} adj={a_arr.shape}"
            )
        if n_arr.dtype != a_arr.dtype:
            return (
                f"FAIL: {name} {arr_name} dtype diverges: "
                f"nx={n_arr.dtype} adj={a_arr.dtype}"
            )
        if not np.array_equal(n_arr, a_arr):
            diff_indices = np.where(n_arr != a_arr)[0]
            i = int(diff_indices[0])
            return (
                f"FAIL: {name} {arr_name} diverges at index {i}: "
                f"nx={n_arr[i]!r} adj={a_arr[i]!r}"
            )
    return None


def main() -> int:
    payload: dict[str, Any] = json.loads(FIXTURE_PATH.read_text())
    fixtures: dict[str, Any] = payload["fixtures"]
    for name in FIXTURES:
        if name not in fixtures:
            print(
                f"MISSING: {name} not in {FIXTURE_PATH.name}",
                file=sys.stderr,
            )
            return 1
        fx = fixtures[name]
        edges = fx.get("edges")
        if not edges:
            print(
                f"NO EDGES for {name}; fixture shape unexpected",
                file=sys.stderr,
            )
            return 1
        nx_graph, adj_graph = build_both(edges)
        nx_triple = nx_graph.to_csr_arrays()
        adj_triple = adj_graph.to_csr_arrays()
        diag = diff_csr_triple(name, nx_triple, adj_triple)
        if diag is not None:
            print(diag, file=sys.stderr)
            return 1
        nnz = int(nx_triple[0][-1])
        n_nodes = len(nx_triple[0]) - 1
        print(f"PASS: {name} (N={n_nodes}, nnz={nnz})")
    print("OK: CSR byte-parity on all 6 sigma fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
