"""Dual-library graph wrapper.

NetworkX for dev ergonomics at small N; igraph (C-backed) for hot-path at
N >= IGRAPH_THRESHOLD. Backend switches automatically in add_node when the
node count crosses the threshold, so callers don't have to care.

Exposed surface (consumed by community.py, richclub.py, pipeline.py):
- add_node, add_edge
- node_count, backend (property)
- centrality() -> dict[UUID, float]       # betweenness
- two_hop_neighborhood(seeds, top_k) # greedy spread
- rich_club_coefficient()                  # van den Heuvel & Sporns 2011
- get_embedding(node_id)
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

import networkx as nx

# switch to C-backed igraph at N >= 500 (centrality + Leiden hot path).
IGRAPH_THRESHOLD = 500

try:
    import igraph as ig  # type: ignore
    _HAS_IGRAPH = True
except ImportError:  # pragma: no cover -- igraph is a hard dep in pyproject
    _HAS_IGRAPH = False


class MemoryGraph:
    """Dual-library graph. NetworkX is the source of truth for topology; igraph
    is rebuilt on demand when backend flips.

    Storage model:
    - `self._nx` holds the authoritative NetworkX graph (str(UUID) node labels).
    - `self._attrs` maps UUID -> {"community_id": UUID|None, "embedding": list[float]}.
    - `self._ig` holds a cached igraph mirror once the backend switches.
    """

    def __init__(self) -> None:
        self._nx: nx.Graph = nx.Graph()
        self._ig: "ig.Graph | None" = None
        self._attrs: dict[UUID, dict[str, Any]] = {}
        self._backend: str = "networkx"

    # -------------------------------------------------------------- properties

    @property
    def backend(self) -> str:
        return self._backend

    def node_count(self) -> int:
        return self._nx.number_of_nodes()

    # ----------------------------------------------------------------- writes

    def add_node(
        self,
        node_id: UUID,
        community_id: UUID | None,
        embedding: list[float],
    ) -> None:
        self._nx.add_node(str(node_id))
        self._attrs[node_id] = {
            "community_id": community_id,
            "embedding": embedding,
        }
        self._maybe_switch_backend()

    def add_edge(
        self,
        src: UUID,
        dst: UUID,
        weight: float = 1.0,
        edge_type: str = "hebbian",
    ) -> None:
        self._nx.add_edge(
            str(src), str(dst), weight=weight, edge_type=edge_type
        )
        if self._ig is not None:
            # igraph mirror is immutable by topology; rebuild after each edge
            # write while in igraph backend. Cheap enough at Phase-1 scale.
            self._rebuild_igraph()

    # ------------------------------------------------------ backend switching

    def _maybe_switch_backend(self) -> None:
        n = self.node_count()
        if (
            n >= IGRAPH_THRESHOLD
            and self._backend == "networkx"
            and _HAS_IGRAPH
        ):
            self._rebuild_igraph()
            self._backend = "igraph"

    def _rebuild_igraph(self) -> None:
        if not _HAS_IGRAPH:
            return
        nodes = list(self._nx.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for u, v in self._nx.edges()]
        weights = [
            float(self._nx[u][v].get("weight", 1.0)) for u, v in self._nx.edges()
        ]
        g = ig.Graph(n=len(nodes), edges=edges, directed=False)
        g.vs["name"] = nodes
        if weights:
            g.es["weight"] = weights
        self._ig = g

    # ---------------------------------------------------------- graph metrics

    def centrality(self) -> dict[UUID, float]:
        """Betweenness centrality. NetworkX for small N, igraph at scale.

        Empty-edge graphs return all-zero centrality (betweenness undefined).
        """
        if self._backend == "networkx":
            if self._nx.number_of_edges() == 0:
                return {UUID(n): 0.0 for n in self._nx.nodes()}
            bc = nx.betweenness_centrality(self._nx, weight="weight")
            return {UUID(n): float(c) for n, c in bc.items()}
        # igraph path
        assert self._ig is not None
        has_weight = "weight" in self._ig.es.attributes()
        raw = self._ig.betweenness(weights="weight" if has_weight else None)
        names = self._ig.vs["name"]
        return {UUID(name): float(c) for name, c in zip(names, raw)}

    def two_hop_neighborhood(
        self, seeds: list[UUID], top_k: int = 5
    ) -> list[UUID]:
        """: 2-hop greedy spread.

        At each hop, for each frontier node, take the top_k highest-weight
        neighbours (Seguin 2018 local-information reconstruction). Dedup
        across seeds and hops; exclude seeds themselves.
        """
        visited: set[str] = {str(s) for s in seeds}
        frontier: set[str] = {str(s) for s in seeds if str(s) in self._nx}
        collected: set[str] = set()

        for _ in range(2):  # 2 hops
            next_frontier: set[str] = set()
            for node in frontier:
                if node not in self._nx:
                    continue
                neighbours = [
                    (n, float(self._nx[node][n].get("weight", 1.0)))
                    for n in self._nx.neighbors(node)
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
        """van den Heuvel & Sporns 2011 -- rich-club coefficient.

        Defaults to using the degree at the 90th percentile as the threshold,
        matching the 10% rich-club convention used in the connectome literature.
        Returns 0.0 on graphs smaller than 2 nodes or without any edges.
        """
        if (
            self._nx.number_of_nodes() < 2
            or self._nx.number_of_edges() == 0
        ):
            return 0.0
        if k_threshold is None:
            degrees = [d for _, d in self._nx.degree()]
            if not degrees:
                return 0.0
            sorted_deg = sorted(degrees)
            # 90th percentile ~ top 10% threshold. len//10 is conservative rounding.
            k_threshold = int(max(1, sorted_deg[-max(1, len(degrees) // 10)]))
        try:
            coeffs = nx.rich_club_coefficient(self._nx, normalized=False)
        except (ZeroDivisionError, nx.NetworkXError):
            # Rich-club is undefined for disconnected or very small graphs.
            return 0.0
        return float(coeffs.get(k_threshold, 0.0))

    # ---------------------------------------------------------------- helpers

    def get_embedding(self, node_id: UUID) -> list[float] | None:
        """Return the embedding attached at add_node() time, or None."""
        attrs = self._attrs.get(node_id)
        return attrs.get("embedding") if attrs else None
