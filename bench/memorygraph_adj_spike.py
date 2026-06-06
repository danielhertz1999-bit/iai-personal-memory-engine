"""Standalone adjacency-dict prototype for MemoryGraph storage.

Pure-Python adjacency-list backend with the public method signatures targeted
for a future runtime storage swap. Ships nothing into ``src/``: this file lives
in ``bench/`` and exists to prove the storage shape can reproduce a byte-exact
CSR triple before the production refactor.

Storage shape:
    ``_adj[str(uuid)][str(neighbor_uuid)] -> {"weight": float, "edge_type": str}``

Shared edge-attr dict invariant:
    ``_adj[u][v]`` and ``_adj[v][u]`` resolve to the SAME dict object (identity,
    not equality). Mutations to the edge attribute dict propagate atomically
    across the two directional views — this is the foundation that prevents
    asymmetric Hebbian-strength drift on long-running sessions.

Once-only edge iteration:
    Adjacency-list storage naturally double-emits each undirected edge (once
    from ``u``'s neighbor list, once from ``v``'s). ``iter_edges_with_weight``
    applies a canonical-pair ``u <= v`` filter so consumers see the same
    once-only semantic that a hash-mapped Graph would emit. Self-loops
    (``u == v``) are emitted once. ``to_csr_arrays`` strips self-loops
    downstream.

Insertion order:
    Python 3.7+ ``dict`` preserves insertion order (PEP 468). ``iter_nodes``
    yields in insertion order; ``to_csr_arrays`` re-sorts the labels lexically
    (``str`` comparison) so CSR rows align with the canonical neighbor-index
    ordering downstream Rust kernels expect.

Empty-graph contract:
    ``to_csr_arrays`` returns
    ``(np.zeros(1, int64), np.zeros(0, int64), np.zeros(0, float64))``.
"""

from __future__ import annotations

from typing import Any, Iterator
from uuid import UUID

import numpy as np


class _AdjacencyBackend:
    """Pure-Python adjacency-list storage prototype.

    Public method signatures mirror MemoryGraph's CSR-relevant surface
    (``add_node``, ``add_edge``, ``remove_node``, ``node_count``,
    ``has_node``, ``iter_nodes``, ``iter_edges_with_weight``, ``degrees``,
    ``to_csr_arrays``, ``two_hop_neighborhood``).
    """

    def __init__(self) -> None:
        # adjacency-list storage: outer key is str(uuid), inner key is
        # str(neighbor_uuid), value is the shared edge-attr dict.
        self._adj: dict[str, dict[str, dict[str, Any]]] = {}
        # community_id sidecar (matches MemoryGraph._attrs after the untangle
        # wave: this dict carries community_id only, not record payload).
        self._attrs: dict[UUID, dict[str, Any]] = {}
        # node payload sidecar keyed by str(uuid) — embedding + future
        # per-node scalars. Decoupled from topology so the storage swap
        # never touches consumer-payload callsites.
        self._node_payload: dict[str, dict[str, Any]] = {}
        # centrality cache (set by future native-kernel callers).
        self._centrality_cache: dict[UUID, float] | None = None
        self._dirty_since_centrality: bool = True

    # ---------------------------------------------------------------- accessors

    def node_count(self) -> int:
        return len(self._adj)

    def has_node(self, node_id: UUID | str) -> bool:
        return str(node_id) in self._adj

    # -------------------------------------------------------------------- writes

    def add_node(
        self,
        node_id: UUID,
        community_id: UUID | None,
        embedding: list[float],
    ) -> None:
        label = str(node_id)
        # setdefault is idempotent — repeated add_node on an existing label
        # leaves the neighbor dict intact (matches hash-Graph add_node semantic).
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
        # Materialise endpoints — handle the case where add_edge is called
        # without a prior add_node (canonical hash-Graph add_edge semantic).
        self._adj.setdefault(u, {})
        self._adj.setdefault(v, {})
        # Shared edge-attr dict: ONE object, two adjacency-list pointers.
        # Mutating attrs["weight"] propagates atomically to both directional
        # views — this is the core invariant of the swap.
        attrs = {"weight": float(weight), "edge_type": str(edge_type)}
        self._adj[u][v] = attrs
        if u != v:
            self._adj[v][u] = attrs
        self._dirty_since_centrality = True

    def remove_node(self, node_id: UUID | str) -> None:
        label = str(node_id)
        if label in self._adj:
            # Scrub backrefs from every neighbor's adjacency list before
            # dropping the node's own entry. Atomic remove vs. orphan
            # back-pointer.
            for neighbor_label in list(self._adj[label].keys()):
                if neighbor_label == label:
                    # self-loop entry — only one place to remove.
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

    # ----------------------------------------------------------------- iterators

    def iter_nodes(self) -> Iterator[UUID]:
        """Yield each node UUID in insertion order."""
        for label in self._adj:
            yield UUID(label)

    def iter_edges_with_weight(self) -> Iterator[tuple[UUID, UUID, float]]:
        """Yield ``(u_uuid, v_uuid, weight_float)`` once per undirected edge.

        Canonical-pair dedup: emits only when ``u_label <= v_label`` in str
        comparison. Self-loops (``u_label == v_label``) are emitted once.
        Weight defaults to 1.0 when an edge has no explicit weight attribute.
        """
        for u_label, neighbors in self._adj.items():
            for v_label, attrs in neighbors.items():
                if u_label <= v_label:
                    try:
                        weight = float(attrs.get("weight", 1.0))
                    except (TypeError, ValueError):
                        weight = 1.0
                    yield UUID(u_label), UUID(v_label), weight

    def degrees(self) -> Iterator[tuple[UUID, int]]:
        """Yield ``(node_uuid, degree_int)`` for each node."""
        for label, neighbors in self._adj.items():
            yield UUID(label), len(neighbors)

    # --------------------------------------------------------------------- CSR

    def to_csr_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (indptr, indices, data) at (int64, int64, float64).

        Canonical sort order: ``sorted(self._adj)`` (str-lexical). Self-loops
        stripped. Non-finite or negative weights dropped. Each
        row's neighbor column indices are sorted ascending (precondition for
        the native LOCAL clustering kernel's binary-search neighbor lookup).
        """
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
        # adjacency-list storage already symmetric — iterate _adj.items()
        # and emit one row entry per (u, v) visit. The outer loop walks every
        # u; the inner loop walks every neighbor v of u. Each undirected
        # edge (u, v) is therefore visited twice (once from u's side, once
        # from v's), which yields the symmetric CSR contract naturally.
        for u_label, neighbors in self._adj.items():
            a = idx_map[u_label]
            for v_label, attrs in neighbors.items():
                if u_label == v_label:
                    continue  # R14: self-loops stripped
                try:
                    w = float(attrs.get("weight", 1.0))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(w) or w < 0.0:
                    continue  # R12: non-finite / negative dropped
                b = idx_map.get(v_label)
                if b is None:
                    continue
                rows[a].append((b, w))
        # Sort each row ascending by column index. Native clustering kernel
        # binary-searches neighbor slices; other consumers are sort-agnostic.
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

    # --------------------------------------------------------- graph algorithms

    def two_hop_neighborhood(
        self, seeds: list[UUID], top_k: int = 5
    ) -> list[UUID]:
        """2-hop greedy spread (top-k highest-weight neighbours per hop)."""
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


# --------------------------------------------------------------- inline tests
if __name__ == "__main__":
    import uuid as _uuid

    def _u(i: int) -> _uuid.UUID:
        # Deterministic test UUIDs — distinct labels, str-comparable.
        return _uuid.uuid5(_uuid.NAMESPACE_DNS, f"node-{i}")

    # ------------------------------------------------------------------- Test 1
    # add_node + node_count: idempotent on repeat.
    g = _AdjacencyBackend()
    assert g.node_count() == 0
    g.add_node(_u(0), community_id=None, embedding=[0.0])
    g.add_node(_u(1), community_id=None, embedding=[0.0])
    g.add_node(_u(2), community_id=None, embedding=[0.0])
    assert g.node_count() == 3, f"expected 3, got {g.node_count()}"
    g.add_node(_u(0), community_id=None, embedding=[0.0])  # idempotent
    assert g.node_count() == 3, "add_node should be idempotent"

    # ------------------------------------------------------------------- Test 2
    # add_edge symmetry: _adj[u][v] is _adj[v][u] (object identity).
    g = _AdjacencyBackend()
    u, v = _u(0), _u(1)
    g.add_node(u, None, [0.0])
    g.add_node(v, None, [0.0])
    g.add_edge(u, v, weight=0.7, edge_type="hebbian")
    assert g._adj[str(u)][str(v)] is g._adj[str(v)][str(u)], (
        "add_edge must share the SAME attr dict between both directional views"
    )

    # ------------------------------------------------------------------- Test 3
    # Mutation propagation: writing to one side propagates to the other.
    g._adj[str(u)][str(v)]["weight"] = 9.9
    assert g._adj[str(v)][str(u)]["weight"] == 9.9, (
        "shared-dict invariant must propagate weight mutations across views"
    )

    # ------------------------------------------------------------------- Test 4
    # iter_edges_with_weight once-only: 3 edges yields exactly 3 tuples.
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

    # ------------------------------------------------------------------- Test 5
    # Self-loop handling: emitted once by iter_edges_with_weight, stripped
    # from to_csr_arrays indices/data.
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

    # ------------------------------------------------------------------- Test 6
    # remove_node scrubs back-references.
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

    # ------------------------------------------------------------------- Test 7
    # Empty graph CSR: (np.zeros(1, int64), np.zeros(0, int64), np.zeros(0, float64))
    g = _AdjacencyBackend()
    indptr_e, indices_e, data_e = g.to_csr_arrays()
    assert indptr_e.dtype == np.int64 and indptr_e.shape == (1,) and indptr_e[0] == 0
    assert indices_e.dtype == np.int64 and indices_e.shape == (0,)
    assert data_e.dtype == np.float64 and data_e.shape == (0,)

    # ------------------------------------------------------------------- Test 8
    # Insertion order vs sort order: iter_nodes preserves insertion order;
    # to_csr_arrays sorts labels lexically (column 0 = lex-smallest label).
    g = _AdjacencyBackend()
    # Choose seeds whose insertion order disagrees with the str-lexical
    # sort order so the divergence is observable.
    n_a = _uuid.uuid5(_uuid.NAMESPACE_DNS, "alpha")
    n_c = _uuid.uuid5(_uuid.NAMESPACE_DNS, "cc")
    n_b = _uuid.uuid5(_uuid.NAMESPACE_DNS, "second")
    # Sanity-check the seed choice — insertion order [A, C, B] must differ
    # from the str-lexical sort of the three labels.
    insertion_labels = [str(n_a), str(n_c), str(n_b)]
    sorted_labels = sorted(insertion_labels)
    assert sorted_labels != insertion_labels, (
        "test design error: chosen uuids happen to be in sorted order; "
        "pick different seeds"
    )
    # Insert in [A, C, B] order — NOT lexically sorted.
    g.add_node(n_a, None, [0.0])
    g.add_node(n_c, None, [0.0])
    g.add_node(n_b, None, [0.0])
    iter_order = list(g.iter_nodes())
    assert iter_order == [n_a, n_c, n_b], (
        f"iter_nodes must preserve insertion order, got {iter_order}"
    )
    # to_csr_arrays internally sorts by str(uuid). Verify _adj-sort yields
    # the str-lexical order (CSR column 0 = lex-smallest label).
    assert sorted(g._adj) == sorted_labels

    print("OK: _AdjacencyBackend spike passes")
