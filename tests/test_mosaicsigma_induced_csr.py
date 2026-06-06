"""Unit tests for the module-scope CSR helpers in ``src/iai_mcp/sigma.py``.

The σ assembly rewire introduces three private helpers:

  - ``_build_csr_from_edge_lists(u_list, v_list, n_nodes)`` -- wrap a
    pair of edge-list arrays (the output of the native
    ``gnm_random_graph`` generator) into a symmetric CSR triple.
  - ``_induced_csr_from_component(indptr, indices, data,
    component_nodes)`` -- extract the induced subgraph CSR for a
    component-node list, returning the new ``(indptr, indices, data,
    sub_node_count)`` tuple.
  - ``_largest_component_csr(indptr, indices, data, n_nodes)`` --
    return the CSR of the largest connected component, falling back to
    the input unchanged on a single-CC graph.

These helpers replace the legacy ``g.subgraph(largest_cc).copy()``
pattern. The unit tests exercise each helper independently so the
σ-pipeline regressions stay attributable to the helper boundary.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_build_csr_from_edge_lists_round_trip() -> None:
    """Symmetric CSR build: every undirected edge appears in both rows."""
    from iai_mcp.sigma import _build_csr_from_edge_lists

    # Triangle: (0-1), (1-2), (0-2)
    u_list = [0, 1, 0]
    v_list = [1, 2, 2]
    indptr, indices, data = _build_csr_from_edge_lists(u_list, v_list, 3)
    # 3 nodes, 3 undirected edges -> 6 directed entries.
    assert indptr.tolist() == [0, 2, 4, 6]
    # row 0: neighbors {1, 2}
    assert sorted(indices[indptr[0]:indptr[1]].tolist()) == [1, 2]
    # row 1: neighbors {0, 2}
    assert sorted(indices[indptr[1]:indptr[2]].tolist()) == [0, 2]
    # row 2: neighbors {0, 1}
    assert sorted(indices[indptr[2]:indptr[3]].tolist()) == [0, 1]
    # All data values are 1.0 (unweighted gnm output).
    assert np.allclose(data, 1.0)


def test_build_csr_from_empty_edge_lists() -> None:
    """Empty edge-list output: CSR has only the indptr header."""
    from iai_mcp.sigma import _build_csr_from_edge_lists

    indptr, indices, data = _build_csr_from_edge_lists([], [], 5)
    assert indptr.tolist() == [0, 0, 0, 0, 0, 0]
    assert indices.tolist() == []
    assert data.tolist() == []


def test_induced_csr_extracts_largest_component_correctly() -> None:
    """Disconnected graph: extract the 5-node component, leave the 3-node behind.

    Builds a CSR for two disjoint cliques (sizes 5 and 3). The helper
    must return a CSR over the 5-node component with the new node
    indices ``0..4`` corresponding to the original component-node
    ordering.
    """
    from iai_mcp.sigma import _build_csr_from_edge_lists, _induced_csr_from_component
    from iai_mcp_native import graph as lilli_graph

    # K_5 on nodes 0..4, K_3 on nodes 5..7.
    edges_k5 = [(u, v) for u in range(5) for v in range(u + 1, 5)]
    edges_k3 = [(u, v) for u in range(5, 8) for v in range(u + 1, 8)]
    all_edges = edges_k5 + edges_k3
    u_list = [u for u, _v in all_edges]
    v_list = [v for _u, v in all_edges]
    indptr, indices, data = _build_csr_from_edge_lists(u_list, v_list, 8)

    components = lilli_graph.connected_components(indptr, indices, 8)
    # Two components; pick the larger (5 nodes).
    largest = max(components, key=len)
    assert sorted(largest) == [0, 1, 2, 3, 4]

    sub_indptr, sub_indices, sub_data, sub_n = _induced_csr_from_component(
        indptr, indices, data, list(largest)
    )
    assert sub_n == 5
    # K_5 has 10 undirected edges -> 20 directed entries in symmetric CSR.
    assert sub_indptr[-1] == 20
    # Each row of K_5 has degree 4.
    for u in range(5):
        row_len = sub_indptr[u + 1] - sub_indptr[u]
        assert row_len == 4, f"K5 row {u} degree should be 4, got {row_len}"


def test_largest_component_csr_on_connected_graph_is_identity() -> None:
    """Single-CC graph: helper returns the input unchanged.

    No subgraph build, no reindex -- the optimisation matters for the σ
    assembly's reference-graph loop where most random graphs are
    connected.
    """
    from iai_mcp.sigma import (
        _build_csr_from_edge_lists,
        _largest_component_csr,
    )

    # K_4: complete graph, single CC.
    edges = [(u, v) for u in range(4) for v in range(u + 1, 4)]
    u_list = [u for u, _v in edges]
    v_list = [v for _u, v in edges]
    indptr, indices, data = _build_csr_from_edge_lists(u_list, v_list, 4)

    sub_indptr, sub_indices, sub_data, sub_n = _largest_component_csr(
        indptr, indices, data, 4
    )
    assert sub_n == 4
    assert np.array_equal(sub_indptr, indptr)
    assert np.array_equal(sub_indices, indices)
    assert np.array_equal(sub_data, data)
