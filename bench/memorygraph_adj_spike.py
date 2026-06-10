
from __future__ import annotations

from typing import Any, Iterator
from uuid import UUID

import numpy as np


class _AdjacencyBackend:

    def __init__(self) -> None:
        self._adj: dict[str, dict[str, dict[str, Any]]] = {}
        self._attrs: dict[UUID, dict[str, Any]] = {}
        self._node_payload: dict[str, dict[str, Any]] = {}
        self._centrality_cache: dict[UUID, float] | None = None
        self._dirty_since_centrality: bool = True


    def node_count(self) -> int:
        return len(self._adj)

    def has_node(self, node_id: UUID | str) -> bool:
        return str(node_id) in self._adj


    def add_node(
        self,
        node_id: UUID,
        community_id: UUID | None,
        embedding: list[float],
    ) -> None:
        label = str(node_id)
        self._adj.setdefault(label, {})
        self._attrs[node_id] = {"community_id": community_id}
        self._node_payload[label] = {"embedding": list(embedding)}
        self._dirty_since_centrality = True

    def add_edge(
        self,
        src: UUID,
        dst: UUID,
        weight: float = 1.0,
        edge_type: str = "hebbian",
    ) -> None:
        u, v = str(src), str(dst)
        self._adj.setdefault(u, {})
        self._adj.setdefault(v, {})
        attrs = {"weight": float(weight), "edge_type": str(edge_type)}
        self._adj[u][v] = attrs
        if u != v:
            self._adj[v][u] = attrs
        self._dirty_since_centrality = True

    def remove_node(self, node_id: UUID | str) -> None:
        label = str(node_id)
        if label in self._adj:
            for neighbor_label in list(self._adj[label].keys()):
                if neighbor_label == label:
                    continue
                self._adj[neighbor_label].pop(label, None)
            del self._adj[label]
        if isinstance(node_id, UUID):
            self._attrs.pop(node_id, None)
        else:
            try:
                self._attrs.pop(UUID(label), None)
            except (TypeError, ValueError):
                pass
        self._node_payload.pop(label, None)
        self._dirty_since_centrality = True


    def iter_nodes(self) -> Iterator[UUID]:
        for label in self._adj:
            yield UUID(label)

    def iter_edges_with_weight(self) -> Iterator[tuple[UUID, UUID, float]]:
        for u_label, neighbors in self._adj.items():
            for v_label, attrs in neighbors.items():
                if u_label <= v_label:
                    try:
                        weight = float(attrs.get("weight", 1.0))
                    except (TypeError, ValueError):
                        weight = 1.0
                    yield UUID(u_label), UUID(v_label), weight

    def degrees(self) -> Iterator[tuple[UUID, int]]:
        for label, neighbors in self._adj.items():
            yield UUID(label), len(neighbors)


    def to_csr_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels: list[str] = sorted(self._adj)
        n = len(labels)
        if n == 0:
            return (
                np.zeros(1, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float64),
            )
        idx_map: dict[str, int] = {label: i for i, label in enumerate(labels)}
        rows: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for u_label, neighbors in self._adj.items():
            a = idx_map[u_label]
            for v_label, attrs in neighbors.items():
                if u_label == v_label:
                    continue
                try:
                    w = float(attrs.get("weight", 1.0))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(w) or w < 0.0:
                    continue
                b = idx_map.get(v_label)
                if b is None:
                    continue
                rows[a].append((b, w))
        for i in range(n):
            rows[i].sort(key=lambda pair: pair[0])
        indptr = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            indptr[i + 1] = indptr[i] + len(rows[i])
        nnz = int(indptr[-1])
        indices = np.empty(nnz, dtype=np.int64)
        data_arr = np.empty(nnz, dtype=np.float64)
        cursor = 0
        for i in range(n):
            for col, w in rows[i]:
                indices[cursor] = col
                data_arr[cursor] = w
                cursor += 1
        return indptr, indices, data_arr


    def two_hop_neighborhood(
        self, seeds: list[UUID], top_k: int = 5
    ) -> list[UUID]:
        visited: set[str] = {str(s) for s in seeds}
        frontier: set[str] = {str(s) for s in seeds if str(s) in self._adj}
        collected: set[str] = set()
        for _ in range(2):
            next_frontier: set[str] = set()
            for node in frontier:
                if node not in self._adj:
                    continue
                neighbours = [
                    (n, float(attrs.get("weight", 1.0)))
                    for n, attrs in self._adj[node].items()
                ]
                neighbours.sort(key=lambda x: x[1], reverse=True)
                for n, _ in neighbours[:top_k]:
                    if n not in visited:
                        next_frontier.add(n)
                        collected.add(n)
                        visited.add(n)
            frontier = next_frontier
            if not frontier:
                break
        return [UUID(n) for n in collected]


if __name__ == "__main__":
    import uuid as _uuid

    def _u(i: int) -> _uuid.UUID:
        return _uuid.uuid5(_uuid.NAMESPACE_DNS, f"node-{i}")

    g = _AdjacencyBackend()
    assert g.node_count() == 0
    g.add_node(_u(0), community_id=None, embedding=[0.0])
    g.add_node(_u(1), community_id=None, embedding=[0.0])
    g.add_node(_u(2), community_id=None, embedding=[0.0])
    assert g.node_count() == 3, f"expected 3, got {g.node_count()}"
    g.add_node(_u(0), community_id=None, embedding=[0.0])
    assert g.node_count() == 3, "add_node should be idempotent"

    g = _AdjacencyBackend()
    u, v = _u(0), _u(1)
    g.add_node(u, None, [0.0])
    g.add_node(v, None, [0.0])
    g.add_edge(u, v, weight=0.7, edge_type="hebbian")
    assert g._adj[str(u)][str(v)] is g._adj[str(v)][str(u)], (
        "add_edge must share the SAME attr dict between both directional views"
    )

    g._adj[str(u)][str(v)]["weight"] = 9.9
    assert g._adj[str(v)][str(u)]["weight"] == 9.9, (
        "shared-dict invariant must propagate weight mutations across views"
    )

    g = _AdjacencyBackend()
    u_, v_, w_ = _u(0), _u(1), _u(2)
    for n in (u_, v_, w_):
        g.add_node(n, None, [0.0])
    g.add_edge(u_, v_)
    g.add_edge(v_, w_)
    g.add_edge(u_, w_)
    edges = list(g.iter_edges_with_weight())
    assert len(edges) == 3, (
        f"expected 3 edges from once-only iter, got {len(edges)}: {edges}"
    )

    g = _AdjacencyBackend()
    s = _u(99)
    g.add_node(s, None, [0.0])
    g.add_edge(s, s, weight=0.5)
    edges_sl = list(g.iter_edges_with_weight())
    assert len(edges_sl) == 1, (
        f"self-loop must be emitted once, got {len(edges_sl)}: {edges_sl}"
    )
    indptr, indices, data = g.to_csr_arrays()
    assert len(indices) == 0, (
        f"self-loop must be stripped from CSR, got indices len {len(indices)}"
    )
    assert len(data) == 0, (
        f"self-loop must be stripped from CSR data, got len {len(data)}"
    )

    g = _AdjacencyBackend()
    a, b = _u(10), _u(11)
    g.add_node(a, None, [0.0])
    g.add_node(b, None, [0.0])
    g.add_edge(a, b)
    assert str(a) in g._adj and str(b) in g._adj[str(a)]
    g.remove_node(a)
    assert str(a) not in g._adj, "removed node must drop from _adj"
    assert str(a) not in g._adj[str(b)], (
        "removed node's back-ref must be scrubbed from neighbour's adj list"
    )

    g = _AdjacencyBackend()
    indptr_e, indices_e, data_e = g.to_csr_arrays()
    assert indptr_e.dtype == np.int64 and indptr_e.shape == (1,) and indptr_e[0] == 0
    assert indices_e.dtype == np.int64 and indices_e.shape == (0,)
    assert data_e.dtype == np.float64 and data_e.shape == (0,)

    g = _AdjacencyBackend()
    n_a = _uuid.uuid5(_uuid.NAMESPACE_DNS, "alpha")
    n_c = _uuid.uuid5(_uuid.NAMESPACE_DNS, "cc")
    n_b = _uuid.uuid5(_uuid.NAMESPACE_DNS, "second")
    insertion_labels = [str(n_a), str(n_c), str(n_b)]
    sorted_labels = sorted(insertion_labels)
    assert sorted_labels != insertion_labels, (
        "test design error: chosen uuids happen to be in sorted order; "
        "pick different seeds"
    )
    g.add_node(n_a, None, [0.0])
    g.add_node(n_c, None, [0.0])
    g.add_node(n_b, None, [0.0])
    iter_order = list(g.iter_nodes())
    assert iter_order == [n_a, n_c, n_b], (
        f"iter_nodes must preserve insertion order, got {iter_order}"
    )
    assert sorted(g._adj) == sorted_labels

    print("OK: _AdjacencyBackend spike passes")
