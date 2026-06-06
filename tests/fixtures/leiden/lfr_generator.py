"""Deterministic LFR-like benchmark generator.

LFR multi-mu exercises the algorithm across the realistic difficulty spectrum
(mu=0.1 clean separation -> mu=0.5 mixed clusters).

DEVIATION FROM CANONICAL LFR:

  Canonical Lancichinetti-Fortunato-Radicchi 2008 LFR (DOI 10.1103/PhysRevE.78.046110)
  requires `python-igraph` for its reference implementation.
  The MIT-clean objective avoids that native dependency, so this generator
  substitutes a planted-partition with degree heterogeneity:

    1. Community sizes sampled from a clipped power-law-light distribution
       between [min_community, max_community], normalised to sum to N.
    2. Per-node target degree sampled from a clipped power-law (gamma=2.5,
       min=avg_degree/4, max=max_degree).
    3. Edge placement: fraction (1 - mu) of each node's edges go to
       intra-community neighbours, mu goes to inter-community.
    4. Rejection sampling avoids duplicates and self-loops.

  The substitute is calibrated to deliver comparable difficulty to canonical
  LFR at the same mu. NMI thresholds in `lfr_seeds.json` are calibrated to
  the substitute (0.90/0.80/0.65), NOT canonical LFR's 0.95-0.98.

DETERMINISM CONTRACT:

  Same `(n, avg_degree, max_degree, mu, n_communities, min_community,
  max_community, seed)` always produces byte-identical edge list AND
  byte-identical `planted_labels`. Verified by
  `test_generator_deterministic` in `test_custom_leiden_lfr_gauntlet.py`.

  Determinism mechanism:
    - `random.Random(seed)` for community sampling + degree sampling.
    - UUIDs allocated via `UUID(int=seed * 10**12 + i)` so the canonical
      `build_csr_sanitized` UUID-sort order is `[0, 1,..., n-1]` -- making
      `planted_labels[i]` align with the canonical node order without any
      post-hoc re-mapping.
    - Edges sorted before insertion.
"""
from __future__ import annotations

import math
import random
from uuid import UUID

from iai_mcp.graph import MemoryGraph


# ----------------------------------------------------------------- helpers


def _emb(seed: int, dim: int = 384) -> list[float]:
    """Deterministic 384-d embedding for a fixture node.

    LFR-gauntlet tests exercise the community-detection kernel, not the
    embedding path; the centroids are computed but never compared.
    Embeddings are filled with a deterministic uniform draw to match the
    shape `build_csr_sanitized` expects.
    """
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def _sample_community_sizes(
    n: int,
    n_communities: int,
    min_size: int,
    max_size: int,
    rng: random.Random,
) -> list[int]:
    """Sample `n_communities` community sizes summing to exactly `n`.

    Sizes are drawn from a power-law-light distribution: each size has
    probability proportional to `size ** -1.5` over `[min_size, max_size]`.
    The raw draws are normalised to sum to `n` and rounded to integers,
    with the residual placed on the largest community.

    Falls back to uniform sizes if `n_communities * min_size > n` (cannot
    satisfy the minimum constraint).
    """
    if n_communities <= 0:
        raise ValueError(f"n_communities must be positive, got {n_communities}")
    if n_communities * min_size > n:
        # Cannot satisfy the floor; relax to uniform sizes.
        base = n // n_communities
        sizes = [base] * n_communities
        sizes[-1] += n - sum(sizes)
        return sizes

    # Draw raw sizes from the power-law-light range.
    raw = []
    for _ in range(n_communities):
        # Inverse-CDF on a power-law-light: u in (0, 1] -> size.
        u = rng.random()
        # gamma = 1.5 (light tail compared to LFR's 1-2 range).
        # size = min_size * (1 - u + u * (max_size / min_size) ** (1 - gamma)) ** (1 / (1 - gamma))
        # Simplified clipped Pareto draw.
        size = min_size * ((1.0 - u + u * (max_size / min_size) ** -0.5) ** -2.0)
        raw.append(max(min_size, min(max_size, size)))

    # Normalise to sum to n.
    total = sum(raw)
    sizes = [max(min_size, int(round(s * n / total))) for s in raw]
    # Adjust residual on the largest community.
    diff = n - sum(sizes)
    if diff != 0:
        idx = max(range(n_communities), key=lambda i: sizes[i])
        sizes[idx] = max(min_size, sizes[idx] + diff)
    # Final residual after the clamp.
    diff = n - sum(sizes)
    if diff != 0:
        # Distribute remainder one-by-one over the largest communities.
        order = sorted(range(n_communities), key=lambda i: -sizes[i])
        i = 0
        step = 1 if diff > 0 else -1
        while diff != 0:
            target = order[i % n_communities]
            if step < 0 and sizes[target] <= min_size:
                i += 1
                continue
            sizes[target] += step
            diff -= step
            i += 1
    assert sum(sizes) == n, f"size sum {sum(sizes)} != n {n}"
    return sizes


def _sample_target_degrees(
    n: int,
    avg_degree: float,
    max_degree: int,
    rng: random.Random,
) -> list[int]:
    """Sample `n` target degrees from a clipped power-law with gamma=2.5.

    Range: `[max(1, avg_degree // 4), max_degree]`. The mean of the
    raw draws is rescaled to match `avg_degree`.
    """
    min_deg = max(1, int(avg_degree // 4))
    if min_deg >= max_degree:
        return [int(avg_degree)] * n

    gamma = 2.5
    raw = []
    for _ in range(n):
        u = rng.random()
        # Inverse-CDF on a power-law with exponent gamma between [a, b]:
        # x = (u * (b ** (1 - gamma) - a ** (1 - gamma)) + a ** (1 - gamma)) ** (1 / (1 - gamma))
        a_pow = min_deg ** (1.0 - gamma)
        b_pow = max_degree ** (1.0 - gamma)
        d_raw = (u * (b_pow - a_pow) + a_pow) ** (1.0 / (1.0 - gamma))
        raw.append(max(min_deg, min(max_degree, d_raw)))

    # Rescale to target average.
    actual_mean = sum(raw) / n
    if actual_mean > 0:
        scale = avg_degree / actual_mean
        raw = [r * scale for r in raw]

    degrees = [max(1, min(max_degree, int(round(d)))) for d in raw]
    return degrees


# --------------------------------------------------------------- entry point


def generate_lfr_like(
    n: int,
    avg_degree: float,
    max_degree: int,
    mu: float,
    n_communities: int,
    min_community: int = 10,
    max_community: int = 100,
    seed: int = 42,
) -> tuple[MemoryGraph, list[int]]:
    """Generate a planted-partition LFR-like graph with `n_communities`.

    Returns:
      (graph, planted_labels) where `planted_labels[i]` is the community id
      of node i in canonical UUID-sorted order. UUIDs are allocated
      deterministically via `UUID(int=seed * 10**12 + i)`, so the canonical
      sort order is `[0, 1,..., n-1]` and `planted_labels` aligns with
      `build_csr_sanitized`'s canonical node order WITHOUT any post-hoc
      re-mapping.

    Construction (planted-partition, NOT canonical Lancichinetti LFR -- see
    module docstring for the deviation disclosure):
      1. Sample `n_communities` community sizes via `_sample_community_sizes`.
      2. Sample per-node target degrees via `_sample_target_degrees`.
      3. For each node i with target k_i: place `(1 - mu) * k_i` edges to
         intra-community neighbours, `mu * k_i` edges to inter-community.
         Rejection sampling avoids duplicates + self-loops.
      4. Insert each (i, j) edge once (deduped).

    Determinism: `random.Random(seed)` for all sampling; deterministic
    UUID allocation. Same params + seed -> byte-equal edges + labels.

    Args:
      n: total node count.
      avg_degree: mean target degree.
      max_degree: degree cap (clipped power-law).
      mu: mixing parameter in [0, 1]; fraction of edges going to
        inter-community neighbours.
      n_communities: number of planted communities.
      min_community: minimum community size.
      max_community: maximum community size.
      seed: PRNG seed (also used for UUID allocation).

    Raises:
      ValueError: if `mu not in [0, 1]`, `n <= 0`, or
        `n_communities * min_community > n` AND uniform-size fallback
        cannot be satisfied either.
    """
    if not 0.0 <= mu <= 1.0:
        raise ValueError(f"mu must be in [0, 1], got {mu}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if avg_degree < 1:
        raise ValueError(f"avg_degree must be >= 1, got {avg_degree}")
    if max_degree < avg_degree:
        raise ValueError(
            f"max_degree {max_degree} must be >= avg_degree {avg_degree}"
        )

    rng = random.Random(seed)

    # (a) Allocate deterministic UUIDs. seed * 10**12 + i is a stable hash
    # guaranteeing that canonical UUID-sort order is identity [0..n-1].
    uuids: list[UUID] = [UUID(int=seed * 10**12 + i) for i in range(n)]

    # (b) Sample community sizes; assign nodes to communities in block order.
    sizes = _sample_community_sizes(
        n, n_communities, min_community, max_community, rng
    )
    planted_labels: list[int] = []
    for c_id, size in enumerate(sizes):
        planted_labels.extend([c_id] * size)
    assert len(planted_labels) == n

    # Group node indices by community.
    community_members: dict[int, list[int]] = {}
    for i, c_id in enumerate(planted_labels):
        community_members.setdefault(c_id, []).append(i)

    # (c) Per-node target degrees.
    target_degrees = _sample_target_degrees(n, avg_degree, max_degree, rng)

    # (d) Edge placement.
    edges: set[tuple[int, int]] = set()
    # Edges-per-node placed so far -- helps stop when k_i reached.
    placed_per_node = [0] * n
    # Cap rejection retries to avoid pathological loops on adversarial inputs.
    MAX_RETRIES_PER_EDGE = 30

    # Iterate nodes in a deterministic order for reproducibility.
    for i in range(n):
        c_id = planted_labels[i]
        target_k = target_degrees[i]
        # Number of intra/inter edges to attempt placing.
        target_intra = int(round((1.0 - mu) * target_k))
        target_inter = target_k - target_intra
        intra_pool = [j for j in community_members[c_id] if j != i]
        inter_pool: list[int] = []
        for other_c, members in community_members.items():
            if other_c != c_id:
                inter_pool.extend(members)

        # Skip if pools are empty (singleton-community degenerate case).
        if not intra_pool and not inter_pool:
            continue
        if not intra_pool:
            target_inter += target_intra
            target_intra = 0
        if not inter_pool:
            target_intra += target_inter
            target_inter = 0

        # Place intra-community edges.
        for _ in range(target_intra):
            if placed_per_node[i] >= target_k:
                break
            for retry in range(MAX_RETRIES_PER_EDGE):
                j = rng.choice(intra_pool)
                if j == i:
                    continue
                key = (min(i, j), max(i, j))
                if key in edges:
                    continue
                edges.add(key)
                placed_per_node[i] += 1
                placed_per_node[j] += 1
                break

        # Place inter-community edges.
        for _ in range(target_inter):
            if placed_per_node[i] >= target_k:
                break
            for retry in range(MAX_RETRIES_PER_EDGE):
                j = rng.choice(inter_pool)
                if j == i:
                    continue
                key = (min(i, j), max(i, j))
                if key in edges:
                    continue
                edges.add(key)
                placed_per_node[i] += 1
                placed_per_node[j] += 1
                break

    # (e) Build MemoryGraph. Adjacency-dict add_edge is O(1) so the loop
    # below has no per-edge rebuild cost — the historical bulk
    # workaround that pre-dated the storage swap is no longer needed.
    g = MemoryGraph()
    for i, u in enumerate(uuids):
        g.add_node(u, community_id=None, embedding=_emb(seed * 1000 + i))

    sorted_edges = sorted(edges)
    for u, v in sorted_edges:
        g.add_edge(uuids[u], uuids[v], weight=1.0, edge_type="hebbian")

    return g, planted_labels


__all__ = ["generate_lfr_like"]
