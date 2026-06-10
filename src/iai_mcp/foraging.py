from __future__ import annotations

import logging
from uuid import UUID, uuid4

from iai_mcp.store import MemoryStore

logger = logging.getLogger(__name__)


def detect_weak_bridges(
    store: MemoryStore,
    min_communities: int = 3,
    max_bridges: int = 5,
) -> list[dict]:
    from iai_mcp import retrieve
    from iai_mcp.store import EDGES_TABLE

    try:
        graph, assignment, _ = retrieve.build_runtime_graph(store)
    except (OSError, RuntimeError, ValueError):
        return []

    if len(assignment.community_centroids) < min_communities:
        return []

    community_ids = list(assignment.community_centroids.keys())
    inter_community_edges: dict[tuple[UUID, UUID], float] = {}

    try:
        tbl = store.db.open_table(EDGES_TABLE)
        df = tbl.to_pandas()
    except (OSError, RuntimeError, ValueError):
        return []

    if df.empty:
        return []

    node_to_comm = assignment.node_to_community
    for _, row in df.iterrows():
        try:
            src = UUID(str(row["src"]))
            dst = UUID(str(row["dst"]))
        except (ValueError, TypeError):
            continue
        src_comm = node_to_comm.get(src)
        dst_comm = node_to_comm.get(dst)
        if src_comm and dst_comm and src_comm != dst_comm:
            pair = (min(src_comm, dst_comm), max(src_comm, dst_comm))
            inter_community_edges[pair] = inter_community_edges.get(pair, 0.0) + float(
                row.get("weight", 1.0)
            )

    if not inter_community_edges:
        all_pairs = [
            (community_ids[i], community_ids[j])
            for i in range(len(community_ids))
            for j in range(i + 1, len(community_ids))
        ]
        return [
            {"community_a": str(a), "community_b": str(b), "bridge_strength": 0.0}
            for a, b in all_pairs[:max_bridges]
        ]

    sorted_bridges = sorted(inter_community_edges.items(), key=lambda x: x[1])
    return [
        {
            "community_a": str(pair[0]),
            "community_b": str(pair[1]),
            "bridge_strength": float(weight),
        }
        for pair, weight in sorted_bridges[:max_bridges]
    ]


def forage_for_connections(store: MemoryStore, max_edges: int = 3) -> int:
    import numpy as np

    from iai_mcp import retrieve

    weak = detect_weak_bridges(store, max_bridges=max_edges)
    if not weak:
        return 0

    try:
        _, assignment, _ = retrieve.build_runtime_graph(store)
    except (OSError, RuntimeError, ValueError):
        return 0

    edges_created = 0
    for bridge in weak:
        if bridge["bridge_strength"] > 0.5:
            continue
        comm_a = UUID(bridge["community_a"])
        comm_b = UUID(bridge["community_b"])
        members_a = assignment.mid_regions.get(comm_a, [])
        members_b = assignment.mid_regions.get(comm_b, [])
        if not members_a or not members_b:
            continue
        centroid_a = assignment.community_centroids.get(comm_a)
        centroid_b = assignment.community_centroids.get(comm_b)
        if not centroid_a or not centroid_b:
            continue
        cos_sim = float(
            np.dot(centroid_a, centroid_b)
            / (np.linalg.norm(centroid_a) * np.linalg.norm(centroid_b) + 1e-9)
        )
        if cos_sim < 0.2:
            continue
        src_node = members_a[0] if isinstance(members_a[0], UUID) else UUID(str(members_a[0]))
        dst_node = members_b[0] if isinstance(members_b[0], UUID) else UUID(str(members_b[0]))
        try:
            store.boost_edges(
                [(src_node, dst_node)],
                edge_type="self_foraging",
                delta=cos_sim * 0.5,
            )
            edges_created += 1
        except (OSError, RuntimeError, ValueError):
            pass

    return edges_created
