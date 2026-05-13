"""Hierarchical community detection ( bootstrap + stable UUIDs + /04).

Policy:
- N < SMALL_N_FLAT (200): single flat community. Rich-club coefficient is too noisy
  below this per van den Heuvel & Sporns 2011; Leiden output is unstable too.
- SMALL_N_FLAT <= N < MID_N_LEIDEN (500): run Leiden; accept only if Q >= 0.2
  (MODULARITY_FLOOR), else fall back to flat. Protects against Leiden producing
  visible but unjustified communities in sparse graphs.
- N >= MID_N_LEIDEN: always run Leiden; accept result regardless of Q
  (graph is big enough that any modular structure is meaningful).

Stable UUIDs:
- Every community gets a persistent UUID at creation.
- On re-run, each new community's centroid is matched against prior centroids;
  the highest cosine >= UUID_ROTATE_COSINE (0.7) reuses the prior UUID.
  If no prior centroid passes the 0.7 bar, a fresh UUID is allocated.
- This prevents ID churn on re-runs where Leiden re-orders labels but the
  cluster membership is essentially the same.

 three-level parcellation (approximation):
- Level 1: top_communities -- top 7 (Yeo-like) by member count.
- Level 2: mid_regions -- community UUID -> member node UUIDs
           (Schaefer-scale 200-400 sub-parcellation is a Phase-2 refinement;
            for we expose the community -> members mapping).
- Level 3: node_to_community -- every leaf record's community assignment.

 refresh threshold:
- needs_refresh(prior, current_Q) returns True iff |prior.Q - current_Q| > 0.05.
  The pipeline or session-start assembler decides when to re-run detect_communities
  based on this signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import numpy as np

from iai_mcp.graph import _HAS_IGRAPH, IGRAPH_THRESHOLD, MemoryGraph

# bootstrap thresholds
SMALL_N_FLAT = 200
MID_N_LEIDEN = 500
MODULARITY_FLOOR = 0.2

# refresh trigger
REFRESH_DELTA = 0.05

# stable-UUID cosine floor
UUID_ROTATE_COSINE = 0.7

# level-1 cap (Yeo-like 7 networks)
MAX_TOP_COMMUNITIES = 7


@dataclass
class CommunityAssignment:
    """Output of detect_communities -- consumed by pipeline.pipeline_recall.

    - node_to_community: leaf UUID -> community UUID
    - community_centroids: community UUID -> mean of member embeddings
    - modularity: Leiden Q (0.0 for flat)
    - backend: "flat" | "leiden-networkx" | "leiden-igraph"
    - top_communities: up to MAX_TOP_COMMUNITIES by member count ( L1)
    - mid_regions: community UUID -> list of member leaf UUIDs ( L2)
    """

    node_to_community: dict[UUID, UUID] = field(default_factory=dict)
    community_centroids: dict[UUID, list[float]] = field(default_factory=dict)
    modularity: float = 0.0
    backend: str = "flat"
    top_communities: list[UUID] = field(default_factory=list)
    mid_regions: dict[UUID, list[UUID]] = field(default_factory=dict)


# ---------------------------------------------------------------- math helpers


def _cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def _compute_centroid(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    arr = np.asarray(embeddings, dtype=np.float32)
    centroid = arr.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    return centroid.tolist()


def _map_to_stable_uuids(
    raw_partition: dict[UUID, int],
    graph: MemoryGraph,
    prior: CommunityAssignment | None,
) -> tuple[dict[UUID, UUID], dict[UUID, list[float]]]:
    """assign UUIDs to raw integer community labels, reusing prior UUIDs
    when a new centroid matches a prior centroid with cosine >= UUID_ROTATE_COSINE.

    Matching is greedy (descending best-match-first) and one-to-one: each prior
    UUID is claimed by at most one new community.
    """
    # Group nodes by raw integer label.
    groups: dict[int, list[UUID]] = {}
    for node, grp in raw_partition.items():
        groups.setdefault(grp, []).append(node)

    # Compute new centroids per group. Filter out nodes with no embedding
    # (e.g. sentinel UUIDs like PROFILE_SENTINEL) and zero-pad the remaining
    # members to the *current* store dim rather than a hardcoded 384d, so the
    # centroid input stays homogeneous after a 384d -> 1024d re-embed migration.
    new_centroids: dict[int, list[float]] = {}
    for grp, nodes in groups.items():
        valid = [e for n in nodes if (e := graph.get_embedding(n))]
        if not valid:
            continue
        dim = len(valid[0])
        embs = [graph.get_embedding(n) or [0.0] * dim for n in nodes]
        new_centroids[grp] = _compute_centroid(embs)

    # Greedy one-to-one assignment: for each new group, pick the best unused
    # prior UUID with cosine >= UUID_ROTATE_COSINE.
    uuid_for_group: dict[int, UUID] = {}
    used_prior: set[UUID] = set()
    if prior:
        # Stable ordering: by group id ascending so tie-breaks are deterministic.
        for grp in sorted(new_centroids.keys()):
            cent = new_centroids[grp]
            best_prior: UUID | None = None
            best_sim: float = -1.0
            for prior_uuid, prior_cent in prior.community_centroids.items():
                if prior_uuid in used_prior:
                    continue
                s = _cosine(cent, prior_cent)
                if s > best_sim:
                    best_sim = s
                    best_prior = prior_uuid
            if best_prior is not None and best_sim >= UUID_ROTATE_COSINE:
                uuid_for_group[grp] = best_prior
                used_prior.add(best_prior)

    # Allocate fresh UUIDs for groups that didn't match any prior.
    for grp in groups:
        if grp not in uuid_for_group:
            uuid_for_group[grp] = uuid4()

    # Build final maps.
    node_to_community: dict[UUID, UUID] = {}
    community_centroids: dict[UUID, list[float]] = {}
    for grp, nodes in groups.items():
        u = uuid_for_group[grp]
        community_centroids[u] = new_centroids[grp]
        for n in nodes:
            node_to_community[n] = u

    return node_to_community, community_centroids


# ------------------------------------------------------------- flat assignment


def _flat_assignment(
    graph: MemoryGraph, prior: CommunityAssignment | None
) -> CommunityAssignment:
    """Single flat community covering every node."""
    nodes: list[UUID] = []
    valid_embs: list[list[float]] = []
    for node in graph._nx.nodes():
        u = UUID(node)
        nodes.append(u)
        emb = graph.get_embedding(u)
        if emb:
            valid_embs.append(emb)
    if not nodes:
        return CommunityAssignment(backend="flat")

    # Zero-pad any sentinel nodes to the detected store dim so centroid math
    # stays homogeneous post-re-embed (was hardcoded 384d before 1024d support).
    dim = len(valid_embs[0]) if valid_embs else 0
    embs: list[list[float]] = []
    for node in graph._nx.nodes():
        u = UUID(node)
        emb = graph.get_embedding(u)
        embs.append(emb if emb else [0.0] * dim)
    centroid = _compute_centroid(embs) if dim else []

    # Stable UUID across flat runs: reuse prior's single UUID if centroid matches.
    flat_uuid: UUID | None = None
    if prior and len(prior.community_centroids) == 1:
        prior_uuid, prior_cent = next(iter(prior.community_centroids.items()))
        if _cosine(centroid, prior_cent) >= UUID_ROTATE_COSINE:
            flat_uuid = prior_uuid
    if flat_uuid is None:
        flat_uuid = uuid4()

    node_to_community = {n: flat_uuid for n in nodes}
    community_centroids = {flat_uuid: centroid}
    return CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids=community_centroids,
        modularity=0.0,
        backend="flat",
        top_communities=[flat_uuid],
        mid_regions={flat_uuid: nodes},
    )


# ------------------------------------------------------------------ leiden run


def _run_leiden(graph: MemoryGraph) -> tuple[dict[UUID, int], float, str]:
    """Run leidenalg on a NetworkX graph via an igraph mirror.

    Returns (node_uuid -> int label, modularity Q, backend_label).
    Backend label reflects which library owns the hot path per :
    "leiden-igraph" for N >= IGRAPH_THRESHOLD, "leiden-networkx" for smaller graphs
    (both internally use leidenalg since python-louvain is Louvain, not Leiden).
    Seed=42 for determinism across calls.
    """
    import igraph as ig  # local import so leiden dep is lazy
    import leidenalg

    g = graph._nx
    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in g.edges()]
    weights = [float(g[u][v].get("weight", 1.0)) for u, v in g.edges()]

    ih = ig.Graph(n=len(nodes), edges=edges, directed=False)
    if weights:
        ih.es["weight"] = weights

    part = leidenalg.find_partition(
        ih,
        leidenalg.ModularityVertexPartition,
        seed=42,
        weights="weight" if weights else None,
    )
    q = float(part.modularity)
    mapping = {
        UUID(nodes[i]): int(part.membership[i]) for i in range(len(nodes))
    }

    # Backend label matches split even though both paths use leidenalg.
    if _HAS_IGRAPH and graph.node_count() >= IGRAPH_THRESHOLD:
        return mapping, q, "leiden-igraph"
    return mapping, q, "leiden-networkx"


# ------------------------------------------------------------------ public API


def detect_communities(
    graph: MemoryGraph,
    prior: CommunityAssignment | None = None,
) -> CommunityAssignment:
    """ bootstrap + stable UUIDs + three-level parcellation.

    Empty graph -> empty CommunityAssignment(backend="flat").
    """
    n = graph.node_count()
    if n == 0:
        return CommunityAssignment(backend="flat")
    if n < SMALL_N_FLAT:
        return _flat_assignment(graph, prior)

    try:
        raw_partition, q, backend = _run_leiden(graph)
    except Exception:
        # Leiden unavailable or graph pathological -> degrade gracefully.
        return _flat_assignment(graph, prior)

    # Mid-N guard: Leiden output only acceptable if Q >= 0.2.
    if n < MID_N_LEIDEN and q < MODULARITY_FLOOR:
        return _flat_assignment(graph, prior)

    node_to_community, community_centroids = _map_to_stable_uuids(
        raw_partition, graph, prior
    )

    # level 1: top 7 communities by member count.
    counts: dict[UUID, int] = {}
    for c in node_to_community.values():
        counts[c] = counts.get(c, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[
        :MAX_TOP_COMMUNITIES
    ]
    top_communities = [u for u, _ in top]

    # level 2 (mid-regions): community UUID -> member node UUIDs.
    mid_regions: dict[UUID, list[UUID]] = {}
    for node, comm in node_to_community.items():
        mid_regions.setdefault(comm, []).append(node)

    return CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids=community_centroids,
        modularity=q,
        backend=backend,
        top_communities=top_communities,
        mid_regions=mid_regions,
    )


def needs_refresh(
    prior: CommunityAssignment, current_modularity: float
) -> bool:
    """: refresh signal when |Δ modularity| > REFRESH_DELTA (0.05).

    Consumer (session-start assembler / maintenance job) calls this on each
    new Leiden run; a True return triggers a re-assignment + cache invalidation.
    """
    return abs(prior.modularity - current_modularity) > REFRESH_DELTA
