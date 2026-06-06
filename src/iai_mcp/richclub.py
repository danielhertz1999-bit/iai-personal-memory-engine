"""Rich-club pre-fetch.

Top 10% of nodes by centrality. Used by pipeline.pipeline_recall at stage 4
(union with 2-hop spread) and by the session-start assembler to pre-warm
the prompt cache with a stable global-hub set.

The top ~10% of hub nodes handle the large majority of the network's
shortest-path traffic, so that percentile is used as the pre-fetch size.
"""
from __future__ import annotations

from math import ceil
from uuid import UUID

from iai_mcp.graph import MemoryGraph


def rich_club_nodes(graph: MemoryGraph, percent: float = 0.10) -> list[UUID]:
    """Return the top `percent` fraction of nodes by centrality.

    - Empty graph -> [].
    - Non-empty graph -> at least 1 node (ceil) even if percent rounds to 0.
      A rich club of zero is useless at the pipeline's Stage 4 union step.
    - Deterministic tie-break: dict.items() preserves insertion order; sort
      is stable, so equal-centrality nodes keep their insertion ordering.
    """
    if graph.node_count() == 0:
        return []
    centrality = graph.centrality()
    if not centrality:
        return []
    k = max(1, ceil(len(centrality) * percent))
    ranked = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)
    return [node_id for node_id, _ in ranked[:k]]
