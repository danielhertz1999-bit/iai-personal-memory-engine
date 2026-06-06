"""Adjacency-dict graph wrapper for MemoryGraph.

Pure-Python adjacency-list backend. ``self._adj[label][neighbor_label]``
resolves to the shared edge-attr dict (``{"weight": float, "edge_type": str}``)
so writing the weight in one direction propagates atomically to the other.
Python 3.7+ ``dict`` preserves insertion order (PEP 468), which the public
``iter_nodes`` API yields verbatim; ``to_csr_arrays`` re-sorts labels
lexically (``str`` comparison) for the canonical CSR row order downstream
Rust kernels consume.

Node-payload sidecar:
``self._node_payload`` is the single source of truth for record-payload
fields (embedding, surface, centrality, tier, pinned, tags, language).
``self._attrs[uuid]`` carries ONLY ``community_id``. The sidecar is keyed
by ``str(uuid)`` so a future native graph backend can swap in without
rewriting key types at every callsite.

Exposed surface (consumed by community.py, richclub.py, pipeline.py,
retrieve.py, mosaic.py, mosaic_lineage.py, memory_bank.py):
- add_node, add_edge, set_node_payload, remove_node
- node_count, has_node
- centrality() -> dict[UUID, float] # betweenness
- iter_nodes() -> Iterable[UUID] # node UUID iterator
- iter_edges_with_weight() -> Iterable[(UUID, UUID, float)]
- to_csr_arrays() -> (indptr, indices, data) int64/int64/float64
- degrees() -> Iterable[(UUID, int)]
- two_hop_neighborhood(seeds, top_k) # greedy spread
- rich_club_coefficient() # van den Heuvel & Sporns 2011
- get_embedding(node_id) # reads from sidecar
- get_centrality(node_id) -> float # reads from sidecar
- get_payload(node_id) -> dict # reads from sidecar
"""
from __future__ import annotations

import os
from typing import Any, Iterator
from uuid import UUID

import numpy as np


# Default cache mode resolved by the centrality recompute measurement bench
# (bench/mosaicsigma_centrality_perf.py). The env var
# ``IAI_MCP_CENTRALITY_CACHE`` (values: "on" / "off" / "auto"; default
# "auto") resolves to this constant when set to "auto" or unset. The bench
# writes either "off" (sub-threshold recompute) or "on" (super-threshold
# or synthetic-fallback) into this slot when run by the orchestrator;
# operators on slower hardware can override at runtime via the env var.
AUTO_CACHE_DEFAULT = "on"


class MemoryGraph:
    """Adjacency-dict graph. Pure-Python single-backend storage.

    Storage model:
    - ``self._adj[str(uuid)][str(neighbor_uuid)]`` resolves to the shared
      edge-attr dict. ``self._adj[u][v] is self._adj[v][u]`` (object
      identity) so mutating ``attrs["weight"]`` on either side propagates
      atomically to the other directional view.
    - ``self._attrs[uuid]`` carries ONLY ``{"community_id": UUID|None}``.
      Record payload no longer lives here.
    - ``self._node_payload[str(uuid)]`` is the sidecar dict holding
      embedding, surface, centrality, tier, pinned, tags, language.
      Keyed by stringified UUID so a future native (Rust) backend can
      swap in without key-type rewrites at every callsite.
    """

    def __init__(self) -> None:
        """Adjacency-dict storage. No external graph library required.

        ``_adj[str(uuid)][str(neighbor_uuid)] -> {"weight": float, "edge_type": str}``
        with shared dict object between ``_adj[u][v]`` and ``_adj[v][u]`` so edge
        attribute mutations propagate atomically across the two directional views.
        Python 3.7+ guarantees insertion-order preservation on ``dict``.
        """
        self._adj: dict[str, dict[str, dict[str, Any]]] = {}
        self._attrs: dict[UUID, dict[str, Any]] = {}
        self._node_payload: dict[str, dict[str, Any]] = {}
        # Centrality cache + dirty flag. Default-dirty (True) so the
        # first ``centrality()`` call always computes — never reads a
        # stale cache that some other code path might have wired
        # outside ``__init__`` via attribute assignment.
        self._centrality_cache: dict[UUID, float] | None = None
        self._dirty_since_centrality: bool = True

    # ---------------------------------------------------------------- accessors

    def node_count(self) -> int:
        return len(self._adj)

    def has_node(self, node_id: UUID | str) -> bool:
        """Return True iff a node with this id is in the graph."""
        return str(node_id) in self._adj

    # ----------------------------------------------------------------- writes

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
        # _attrs holds ONLY community_id. Record-payload fields (embedding,
        # surface, centrality, tier, pinned, tags, language) live in the
        # _node_payload sidecar instead.
        self._attrs[node_id] = {
            "community_id": community_id,
        }
        # Sidecar key is str(uuid) so a future native backend can use
        # interned strings without UUID-object hashing.
        self._node_payload[label] = {
            "embedding": list(embedding),
        }
        # Topology mutated — invalidate the centrality cache so the
        # next ``centrality()`` call under auto/on mode recomputes.
        self._dirty_since_centrality = True

    def set_node_payload(
        self, node_id: UUID | str, payload: dict[str, Any]
    ) -> None:
        """Write the record-payload fields into the sidecar (merge semantics).

        Replaces the legacy direct-attribute write pattern. ``node_id``
        accepts either a ``UUID`` object or its string form — both
        normalise to ``str(uuid)`` before lookup so the silent
        UUID-vs-str key mismatch cannot recur.

        Idempotent merge: calling ``set_node_payload`` twice with the same
        payload is a no-op on disk shape (single dict, fields overwritten by
        the latest write).
        """
        key = str(node_id)
        existing = self._node_payload.get(key, {})
        merged = dict(existing)
        for k, v in payload.items():
            merged[k] = v
        self._node_payload[key] = merged

    def set_node_centrality(self, node_id: UUID | str, value: float) -> None:
        """Convenience helper for the centrality back-write hot path.

        Writes ``{"centrality": float(value)}`` into the sidecar. Used by
        retrieve.build_runtime_graph's centrality-attach phase and by any
        future Hebbian / activation-spreading update path that needs to
        refresh a single per-node scalar without round-tripping the full
        payload dict.
        """
        self.set_node_payload(node_id, {"centrality": float(value)})

    def remove_node(self, node_id: UUID | str) -> None:
        """Drop a node and all sidecar / attrs state for it.

        Used by the store -> graph sync hook on delete. Topology-and-sidecar
        atomic removal keeps the two stores in lockstep at the WRITE boundary.
        """
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
        # _attrs is keyed by UUID; coerce when caller passes the str form.
        if isinstance(node_id, UUID):
            self._attrs.pop(node_id, None)
        else:
            try:
                self._attrs.pop(UUID(label), None)
            except (TypeError, ValueError):
                pass
        self._node_payload.pop(label, None)
        # Topology mutated — invalidate the centrality cache.
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
        # views. Self-loops where u == v collapse to one assignment (the
        # second write overwrites the first with the same object) — that
        # is correct semantics; no special case needed.
        attrs = {"weight": float(weight), "edge_type": str(edge_type)}
        self._adj[u][v] = attrs
        if u != v:
            self._adj[v][u] = attrs
        # Topology mutated — invalidate the centrality cache.
        self._dirty_since_centrality = True

    # ---------------------------------------------------------- graph metrics

    def centrality(self) -> dict[UUID, float]:
        """UNWEIGHTED BFS-Brandes betweenness via the native Rust path.

        Behaviour change vs the pre-cutover dual-library implementation:
        the upstream Rust kernel (rustworkx-core 0.17) has no weight-map
        parameter, so the Hebbian-strength weighted-Brandes semantic is
        intentionally dropped. Returns ``{}`` for an empty graph;
        isolated nodes get ``0.0`` (matches the unweighted-Brandes
        convention).

        Cache flag wiring:
          - env var ``IAI_MCP_CENTRALITY_CACHE`` accepts ``on``, ``off``,
            or ``auto`` (the default when unset).
          - ``auto`` resolves to the module-level ``AUTO_CACHE_DEFAULT``
            constant (committed by the perf-gate bench at plan
            execution time).
          - ``on`` caches the dict and re-returns the same object on
            subsequent calls while the dirty flag is clear.
          - ``off`` always recomputes — never writes the cache, never
            reads from it. The escape hatch for operators on slower
            hardware where the cached read is preferable to a recompute
            despite the staleness window.
        """
        env_mode = os.environ.get("IAI_MCP_CENTRALITY_CACHE", "auto").lower()
        effective_mode = (
            AUTO_CACHE_DEFAULT if env_mode == "auto" else env_mode
        )

        if (
            effective_mode == "on"
            and self._centrality_cache is not None
            and not self._dirty_since_centrality
        ):
            return self._centrality_cache

        # Import inside the method to keep module-import cheap on hosts
        # that never compute centrality (mcp-wrapper tooling, etc).
        from iai_mcp_native import graph as _native

        # to_csr_arrays() emits an int64/int64/float64 triple. We discard
        # the float64 weight slice — rustworkx-core 0.17's
        # ``betweenness_centrality`` has no weight-map parameter, so the
        # Hebbian-strength weighted-Brandes semantic is dropped here.
        indptr, indices, _data_discarded = self.to_csr_arrays()
        n_nodes = len(indptr) - 1

        # CSR row order matches the ``to_csr_arrays`` internal sort
        # over the str-form node labels. The Rust kernel returns a
        # ``node_arr`` of CSR-row indices, and we map each scalar back
        # to its UUID via this sorted-label table — NOT via the raw
        # insertion order from ``iter_nodes()``, because the two
        # orderings drift on every non-trivial node-id distribution.
        self._node_ids_csr_order: list[UUID] = sorted(
            self.iter_nodes(), key=str
        )

        centrality_arr, node_arr = _native.betweenness_centrality(
            indptr, indices, n_nodes, normalized=True
        )
        result: dict[UUID, float] = {
            self._node_ids_csr_order[int(idx)]: float(val)
            for idx, val in zip(node_arr, centrality_arr)
        }
        if effective_mode != "off":
            self._centrality_cache = result
            self._dirty_since_centrality = False
        return result

    def two_hop_neighborhood(
        self, seeds: list[UUID], top_k: int = 5
    ) -> list[UUID]:
        """2-hop greedy spread.

        At each hop, for each frontier node, take the top_k highest-weight
        neighbours (Seguin 2018 local-information reconstruction). Dedup
        across seeds and hops; exclude seeds themselves.
        """
        visited: set[str] = {str(s) for s in seeds}
        frontier: set[str] = {str(s) for s in seeds if str(s) in self._adj}
        collected: set[str] = set()

        for _ in range(2):  # 2 hops
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

    def rich_club_coefficient(self, k_threshold: int | None = None) -> float:
        """Rich-club coefficient (van den Heuvel & Sporns 2011, J Neurosci
        31:15775-15786).

        phi(k) = 2 * E_{>k} / (N_{>k} * (N_{>k} - 1))

        where E_{>k} is the count of edges between nodes whose degree exceeds
        k, and N_{>k} is the count of nodes with degree > k.

        Self-loops are stripped from the edge iterator before degree
        counting. Self-loops are valid in the project edge schema (Hebbian
        self-loops on records) but not meaningful for a metric that measures
        connectivity between hubs.

        Default ``k_threshold`` is the 90th percentile of the post-strip
        degree distribution (the 10%-rich-club convention from the connectome
        literature). Degrees are rebuilt from the post-strip edge list
        seeded with every node returned by:meth:`iter_nodes` at zero, so
        isolated nodes still contribute degree 0 to the distribution. The
        result is bounded in [0.0, 1.0] by construction and is exactly 0.0
        whenever (a) the post-strip edge list is empty, (b) fewer than two
        nodes have degree > k, or (c) no edges connect two such nodes.
        """
        # Step 1: strip self-loops on the edge iterator.
        edges_no_selfloop: list[tuple[UUID, UUID]] = [
            (u, v) for u, v, _w in self.iter_edges_with_weight() if u != v
        ]
        if not edges_no_selfloop:
            return 0.0
        # Step 2: rebuild degrees from post-strip edges, seeded with every
        # node so isolates contribute degree 0 to the distribution. Without
        # this seed the 90th-percentile threshold and N_{>k} would drift
        # vs the published formula's intent (which counts every node in V).
        degrees: dict[UUID, int] = {nid: 0 for nid in self.iter_nodes()}
        for u, v in edges_no_selfloop:
            degrees[u] = degrees.get(u, 0) + 1
            degrees[v] = degrees.get(v, 0) + 1
        if k_threshold is None:
            deg_values = list(degrees.values())
            if not deg_values:
                return 0.0
            k_threshold = int(np.percentile(deg_values, 90))
        n_gt_k = sum(1 for d in degrees.values() if d > k_threshold)
        if n_gt_k < 2:
            return 0.0
        e_gt_k = sum(
            1
            for u, v in edges_no_selfloop
            if degrees.get(u, 0) > k_threshold
            and degrees.get(v, 0) > k_threshold
        )
        return 2.0 * e_gt_k / (n_gt_k * (n_gt_k - 1))

    # ---------------------------------------------- public read-API surface

    def iter_nodes(self) -> Iterator[UUID]:
        """Yield each node's UUID in insertion order.

        Backed by ``self._adj`` (Python 3.7+ dict preserves insertion
        order per PEP 468). Callers route through this method instead of
        touching ``_adj`` directly so a future native backend can swap
        in without rewriting consumer sites.
        """
        for label in self._adj:
            yield UUID(label)

    def nodes(self) -> Iterator[UUID]:
        """Alias for iter_nodes(); yields each node UUID.

        Provided so callers that use the standard graph ``nodes()``
        convention work without change.
        """
        return self.iter_nodes()

    def iter_edges_with_weight(
        self,
    ) -> Iterator[tuple[UUID, UUID, float]]:
        """Yield ``(u_uuid, v_uuid, weight_float)`` once per undirected edge.

        Canonical-pair dedup: emits only when ``u_label <= v_label`` in
        str comparison. Self-loops (``u_label == v_label``) are emitted
        once. Weight defaults to 1.0 when an edge has no explicit weight
        attribute (the same default the legacy consumers used).
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

    def to_csr_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the graph adjacency as CSR-style arrays.

        Returns (indptr, indices, data) with dtypes (int64, int64, float64).
        Row/column indexing is canonical-sort order (``sorted by str(uuid)``)
        to match the mosaic.py CSR build invariant — self-loops stripped,
        symmetric, non-negative weights. NaN / negative / non-numeric weights
        are dropped (the existing mosaic.py policy).

        A future native (Rust) backend will produce these arrays directly;
        the Python implementation here mirrors the canonical mosaic.py path
        verbatim so the invariant tests exercising downstream consumers keep
        passing during the backend swap.
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
        # Adjacency-list storage already symmetric — iterate _adj.items()
        # and emit one row entry per (u, v) visit. The outer loop walks
        # every u; the inner loop walks every neighbor v of u. Each
        # undirected edge (u, v) is therefore visited twice (once from
        # u's side, once from v's), which yields the symmetric CSR
        # contract naturally.
        for u_label, neighbors in self._adj.items():
            a = idx_map[u_label]
            for v_label, attrs in neighbors.items():
                if u_label == v_label:
                    continue  # self-loops stripped.
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
        # Sort each row's neighbor list ascending by column index. The native
        # `average_clustering` kernel (rust/iai_mcp_graph_core/src/clustering.rs)
        # binary-searches neighbor slices, which requires sorted-ascending
        # order. Other consumers (centrality / connectivity) are
        # sort-agnostic so the sort is a no-op for them.
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

    # ---------------------------------------------- payload read-API surface

    def get_embedding(self, node_id: UUID | str) -> list[float] | None:
        """Return the embedding attached at add_node() time, or None.

        Reads from the ``_node_payload`` sidecar. Accepts either ``UUID``
        object or stringified UUID — both normalise to ``str(uuid)``
        before lookup so the silent UUID-vs-str key mismatch surfaced in
        the threat register cannot return ``None`` on a present node.
        """
        payload = self._node_payload.get(str(node_id))
        if not payload:
            return None
        emb = payload.get("embedding")
        return emb if emb else None

    def get_centrality(self, node_id: UUID | str) -> float:
        """Return the per-node centrality scalar, or 0.0 when absent.

        Reads from the sidecar.
        """
        payload = self._node_payload.get(str(node_id))
        if not payload:
            return 0.0
        try:
            return float(payload.get("centrality", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def get_payload(self, node_id: UUID | str) -> dict[str, Any]:
        """Return a shallow copy of the full sidecar dict for this node.

        Returns an empty dict when the node is unknown. The copy is shallow
        so list-valued fields (embedding, tags) are shared with the sidecar
        — callers MUST NOT mutate them in place; use ``set_node_payload``
        for writes.
        """
        payload = self._node_payload.get(str(node_id))
        if not payload:
            return {}
        return dict(payload)
