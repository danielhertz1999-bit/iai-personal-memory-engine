from __future__ import annotations

from math import ceil
from uuid import UUID

from iai_mcp.graph import MemoryGraph


def rich_club_nodes(
    graph: MemoryGraph,
    percent: float = 0.10,
    centrality: "dict[UUID, float] | None" = None,
) -> list[UUID]:
    """Top-``percent`` nodes by betweenness centrality.

    When ``centrality`` is supplied the ranking reuses it directly — the caller
    has already computed (or loaded) the betweenness map and passes it in so this
    function never triggers a second exact betweenness pass. The long-lived
    recall process always supplies it: an exact in-parent Brandes pass on a large
    graph spikes the resident set toward the watchdog cap, so the parent must
    reuse the child-computed / cached / neutral map rather than recompute here.

    When ``centrality`` is None the map is computed on the graph in-process. That
    is reserved for callers that run in a short-lived child process (which
    reclaims its own arenas on exit) or operate on a graph small enough that the
    in-process pass is genuinely bounded.
    """
    if graph.node_count() == 0:
        return []
    if centrality is None:
        centrality = graph.centrality()
    if not centrality:
        return []
    k = max(1, ceil(len(centrality) * percent))
    ranked = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)
    return [node_id for node_id, _ in ranked[:k]]
