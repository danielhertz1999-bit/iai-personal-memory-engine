"""CSR byte-exact parity gate against the sigma fixture set.

Builds the same graph twice — once via the current networkx-backed
``MemoryGraph``, once via the standalone ``_AdjacencyBackend`` prototype —
and asserts byte-exact equality on the ``(indptr, indices, data)`` triple
returned by ``to_csr_arrays``. The triple is what the downstream native
graph kernels consume; if the two backends diverge on this triple, the
storage swap would silently regress centrality / clustering / shortest-path
math at the downstream parity gate.

Gate. Exit 0 → byte-parity proven on all 6 mandatory
sigma fixtures (karate, les_miserables, er_200, er_500, er_1000,
ws_2500_k4_p0). Any byte-divergence prints the fixture name + first
divergent array + index + values on each side BEFORE exiting non-zero.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

# Resolve ``iai_mcp.*`` to this worktree's ``src/`` rather than an editable
# install elsewhere on the path. Idempotent: each ``sys.path.insert`` is
# guarded by an "if not already present" check.
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

# Allow the spike sibling to be imported by absolute name when this script
# is executed via ``python bench/memorygraph_csr_parity.py``.
sys.path.insert(0, str(Path(__file__).parent))
from memorygraph_adj_spike import _AdjacencyBackend  # noqa: E402

from iai_mcp.graph import MemoryGraph  # noqa: E402 # current nx-backed

# 6 mandatory sigma fixtures (baseline set, status=mandatory).
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
    """Map a fixture integer node-id to a stable UUID.

    Both backends see identical labels — uuid5 is referentially transparent
    so the two build paths produce the same sort order and CSR rows.
    """
    return uuid.uuid5(uuid.NAMESPACE_DNS, str(node_index))


def build_both(
    edges: list[list[int]],
) -> tuple[MemoryGraph, _AdjacencyBackend]:
    """Build a networkx-backed MemoryGraph and an adjacency-dict prototype
    from the same edge list, with identical UUIDs on each side.
    """
    nx_graph = MemoryGraph()
    adj_graph = _AdjacencyBackend()
    # Collect node set first so add_node is called before add_edge for both
    # backends. Sorted for deterministic insertion order (matches the
    # iteration shape both Wave 2 will see at production callsites).
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
    """Return None on byte-exact match, or a one-line diagnostic string.

    Surfaces the first divergent index on whichever array (indptr /
    indices / data) breaks parity first — gives the root-cause hook
    if any fixture fails.
    """
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
