from __future__ import annotations

from math import ceil
from uuid import UUID

from iai_mcp.graph import MemoryGraph


def rich_club_nodes(graph: MemoryGraph, percent: float = 0.10) -> list[UUID]:
    if graph.node_count() == 0:
        return []
    centrality = graph.centrality()
    if not centrality:
        return []
    k = max(1, ceil(len(centrality) * percent))
    ranked = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)
    return [node_id for node_id, _ in ranked[:k]]
