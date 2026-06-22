"""Bounded, deterministic approximate centrality for the warm graph.

Exact Brandes betweenness is ``O(V*E)``: on a tens-of-thousands-node corpus it
climbs toward the daemon memory cap, never completes, and is retried every warm
cycle. The seed score that consumes it is ``0.6*cos + 0.4*centrality``, so the
centrality term only needs to preserve the *bridge / rich-club ranking* of the
exact map, not its absolute scalars. Two ``O(E)``-class approximations recover
that ranking cheaply:

``sampled_betweenness`` (Brandes-Pich)
    Run the exact Brandes single-source dependency accumulation, but only from a
    deterministically-chosen set of ``k`` source nodes, and scale the result by
    ``n / k``. Sources are the ``k`` highest-degree nodes (ties broken by CSR row
    index), so the same graph yields the same sources -- and therefore the same
    map -- every cycle. High-degree pivots concentrate the shortest-path tree
    coverage on the structural bridges betweenness rewards, which is why a small
    ``k`` already tracks the exact top-K seed set. Cost ``O(k*E)``.

``harmonic``
    Harmonic centrality: for every node, ``sum(1/d)`` over all reachable others,
    computed by an unweighted BFS from each source. Naturally deterministic (no
    pivots), ``O(V*E)`` in the worst case but with a far smaller constant than
    Brandes (no dependency back-propagation, no predecessor bookkeeping). It is
    the fallback when sampled betweenness is too slow.

Both kernels read the CSR adjacency buffers that ``MemoryGraph.to_csr_arrays``
already produces and return a dense per-row score array. The scores are
**percentile-normalized per cycle** before they reach the seed blend: the raw
magnitudes of an approximation drift with ``k`` and with corpus size, and the
seed term is scale-sensitive, so feeding a rank-normalized score keeps the
0.4 weight meaning the same thing every cycle regardless of the absolute values
the approximation happened to produce.

References:
    Brandes, U. (2001). A faster algorithm for betweenness centrality.
        Journal of Mathematical Sociology, 25(2), 163-177.
    Brandes, U., & Pich, C. (2007). Centrality estimation in large networks.
        International Journal of Bifurcation and Chaos, 17(7), 2303-2318.
"""
from __future__ import annotations

import os
from typing import Any
from uuid import UUID

import numpy as np


# Candidate identifiers exposed through the single ``approximate_centrality``
# interface. The runtime default is resolved from the bench winner via the
# ``IAI_MCP_RGC_CENTRALITY_METHOD`` env override (see ``runtime_method``).
METHOD_SAMPLED_BETWEENNESS = "sampled_betweenness"
METHOD_HARMONIC = "harmonic"

# Default pivot count for sampled betweenness. The bench picks the smallest ``k``
# that clears the fidelity gate; the operator can scale it for very large corpora
# via ``IAI_MCP_RGC_CENTRALITY_K``.
DEFAULT_K = 512

# Node count at or below which the warm path computes EXACT betweenness instead
# of approximating: the exact pass is genuinely bounded on a small graph, and
# the approximation buys nothing there. One continuous policy, no behavioural
# cliff -- the only thing that changes at the boundary is exact-vs-approx, and
# both preserve the same bridge ranking by construction.
DEFAULT_EXACT_BELOW = 4096


try:  # numba is a hard runtime dependency of the mosaic kernels; reuse its JIT.
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba absent is not a supported config
    _HAVE_NUMBA = False

    def njit(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def _wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return _wrap


@njit(cache=True)
def _sampled_betweenness_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_nodes: int,
    sources: np.ndarray,
) -> np.ndarray:
    """Brandes single-source dependency accumulation from ``sources`` only.

    A faithful transcription of Brandes 2001 for unweighted graphs (BFS shortest
    paths + dependency back-propagation), restricted to the supplied source set.
    The per-source contributions are summed into ``delta_total``; the caller
    scales by ``n / len(sources)`` to estimate the full-source betweenness.
    """
    betweenness = np.zeros(n_nodes, dtype=np.float64)

    # Scratch buffers reused across sources to keep the per-source allocation at
    # zero (the warm child's footprint is the whole point of this work).
    sigma = np.zeros(n_nodes, dtype=np.float64)
    dist = np.empty(n_nodes, dtype=np.int64)
    delta = np.zeros(n_nodes, dtype=np.float64)
    queue = np.empty(n_nodes, dtype=np.int64)
    stack = np.empty(n_nodes, dtype=np.int64)
    # Predecessor lists stored flat: pred_flat holds predecessors back-to-back,
    # pred_count[v] is how many v currently has. Capacity is the edge count
    # (each directed BFS-tree edge contributes one predecessor entry per layer).
    pred_count = np.zeros(n_nodes, dtype=np.int64)
    pred_head = np.empty(n_nodes, dtype=np.int64)
    pred_cap = indices.shape[0] + n_nodes
    pred_flat = np.empty(pred_cap, dtype=np.int64)

    for s_i in range(sources.shape[0]):
        s = sources[s_i]

        for v in range(n_nodes):
            sigma[v] = 0.0
            dist[v] = -1
            delta[v] = 0.0
            pred_count[v] = 0
        sigma[s] = 1.0
        dist[s] = 0

        # Assign each node a contiguous predecessor region. The region size is
        # the in-degree upper bound (the node's neighbour count), summed as a
        # prefix so regions never overlap.
        offset = 0
        for v in range(n_nodes):
            pred_head[v] = offset
            offset += indptr[v + 1] - indptr[v]

        q_head = 0
        q_tail = 0
        s_top = 0
        queue[q_tail] = s
        q_tail += 1

        while q_head < q_tail:
            v = queue[q_head]
            q_head += 1
            stack[s_top] = v
            s_top += 1
            row_start = indptr[v]
            row_end = indptr[v + 1]
            for e in range(row_start, row_end):
                w = indices[e]
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue[q_tail] = w
                    q_tail += 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred_flat[pred_head[w] + pred_count[w]] = v
                    pred_count[w] += 1

        # Back-propagation in reverse BFS order.
        for idx in range(s_top - 1, -1, -1):
            w = stack[idx]
            coeff = (1.0 + delta[w]) / sigma[w]
            base = pred_head[w]
            for p_i in range(pred_count[w]):
                v = pred_flat[base + p_i]
                delta[v] += sigma[v] * coeff
            if w != s:
                betweenness[w] += delta[w]

    return betweenness


@njit(cache=True)
def _harmonic_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Unweighted harmonic centrality: ``sum(1/d)`` over reachable nodes.

    One BFS per node accumulates the inverse shortest-path distance from every
    other reachable node. Deterministic by construction (no source sampling).
    """
    harmonic = np.zeros(n_nodes, dtype=np.float64)
    dist = np.empty(n_nodes, dtype=np.int64)
    queue = np.empty(n_nodes, dtype=np.int64)

    for s in range(n_nodes):
        for v in range(n_nodes):
            dist[v] = -1
        dist[s] = 0
        q_head = 0
        q_tail = 0
        queue[q_tail] = s
        q_tail += 1
        acc = 0.0
        while q_head < q_tail:
            v = queue[q_head]
            q_head += 1
            d_next = dist[v] + 1
            row_start = indptr[v]
            row_end = indptr[v + 1]
            for e in range(row_start, row_end):
                w = indices[e]
                if dist[w] < 0:
                    dist[w] = d_next
                    queue[q_tail] = w
                    q_tail += 1
                    acc += 1.0 / d_next
        harmonic[s] = acc

    return harmonic


def _deterministic_sources(
    indptr: np.ndarray, n_nodes: int, k: int
) -> np.ndarray:
    """The ``k`` highest-degree nodes, ties broken by ascending CSR row index.

    Degree is ``indptr[v+1] - indptr[v]``. Sorting by ``(-degree, row)`` is a
    pure function of the CSR, so the source set -- and therefore the whole
    sampled map -- is identical on every call for a given graph. No RNG, no
    per-cycle flicker.
    """
    if k >= n_nodes:
        return np.arange(n_nodes, dtype=np.int64)
    degrees = indptr[1:] - indptr[:-1]
    # argsort is stable; negating degree gives descending degree with ascending
    # row index as the deterministic tie-break.
    order = np.argsort(-degrees, kind="stable")
    return order[:k].astype(np.int64)


def _percentile_normalize(values: np.ndarray) -> np.ndarray:
    """Map scores to their per-cycle percentile rank in ``[0, 1]``.

    The seed blend's 0.4 weight is scale-sensitive; an approximation's raw
    magnitude drifts with ``k`` and corpus size, so the absolute values are not
    comparable across cycles. Ranking strips that drift: the relative ordering --
    the only thing the seed term needs from centrality -- is preserved exactly,
    while the magnitude becomes a stable ``[0, 1]`` percentile every cycle.
    Ties share the average rank so equal scores stay equal.
    """
    n = values.shape[0]
    if n == 0:
        return values
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # Average rank for the tie block [i, j].
        avg_rank = (i + j) / 2.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg_rank
        i = j + 1
    return ranks / (n - 1)


def runtime_method() -> str:
    """The approximate method the warm path uses, resolved from the env.

    Defaults to the bench winner; an operator override lets the fallback be
    selected without a code change.
    """
    raw = os.environ.get("IAI_MCP_RGC_CENTRALITY_METHOD", "").strip().lower()
    if raw in (METHOD_SAMPLED_BETWEENNESS, METHOD_HARMONIC):
        return raw
    return METHOD_SAMPLED_BETWEENNESS


def runtime_k() -> int:
    """Pivot count for sampled betweenness, env-overridable for huge corpora."""
    raw = os.environ.get("IAI_MCP_RGC_CENTRALITY_K", "").strip()
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return DEFAULT_K


def exact_below() -> int:
    """Node count at/below which the warm path stays exact, env-overridable."""
    raw = os.environ.get("IAI_MCP_RGC_CENTRALITY_EXACT_BELOW", "").strip()
    if raw:
        try:
            parsed = int(raw)
            if parsed >= 0:
                return parsed
        except ValueError:
            pass
    return DEFAULT_EXACT_BELOW


def approximate_centrality(
    graph: Any,
    method: str = METHOD_SAMPLED_BETWEENNESS,
    *,
    k: int = DEFAULT_K,
    seed: int = 0,
    normalize: bool = True,
) -> dict[UUID, float]:
    """Bounded, deterministic approximate centrality as ``{node_id: score}``.

    ``method`` selects the candidate (``sampled_betweenness`` or ``harmonic``).
    ``k`` is the pivot count for sampled betweenness (ignored by harmonic).
    ``seed`` is accepted for interface symmetry; both candidates are deterministic
    by construction (degree-ordered pivots / full BFS), so the result does not
    depend on it -- it exists so a future stochastic-pivot variant can slot in
    without changing callers. ``normalize`` applies the per-cycle percentile
    normalization the seed blend expects; pass ``False`` to inspect raw scores
    (the fidelity bench compares raw scores against exact).
    """
    indptr, indices, _data = graph.to_csr_arrays()
    n_nodes = len(indptr) - 1
    if n_nodes == 0:
        return {}

    node_ids: list[UUID] = sorted(graph.iter_nodes(), key=str)

    if method == METHOD_HARMONIC:
        scores = _harmonic_kernel(
            np.ascontiguousarray(indptr),
            np.ascontiguousarray(indices),
            n_nodes,
        )
    elif method == METHOD_SAMPLED_BETWEENNESS:
        sources = _deterministic_sources(indptr, n_nodes, max(1, int(k)))
        raw = _sampled_betweenness_kernel(
            np.ascontiguousarray(indptr),
            np.ascontiguousarray(indices),
            n_nodes,
            sources,
        )
        # Scale the k-source estimate up to the full-source magnitude. The
        # normalization below makes the constant factor irrelevant to ranking,
        # but the scaled value is the honest betweenness estimate when raw
        # scores are inspected (the bench reads them un-normalized).
        scale = float(n_nodes) / float(sources.shape[0])
        scores = raw * scale
    else:
        raise ValueError(f"unknown centrality method: {method!r}")

    if normalize:
        scores = _percentile_normalize(scores)

    return {
        node_ids[i]: float(scores[i]) for i in range(n_nodes)
    }


def centrality_for_runtime(graph: Any) -> dict[UUID, float]:
    """The centrality the warm-graph child computes: exact below the cutoff,
    bounded approximate above it.

    This is the single dispatch point the warm path routes through. Below
    ``exact_below()`` nodes the exact Brandes pass is genuinely bounded, so it is
    returned verbatim -- byte-identical to ``graph.centrality()``, which the
    child-parity contract still pins. Above the cutoff the env-selected
    approximate method runs with the env-configured ``k``, returning a
    raw-betweenness-scale estimate (the k-source sum scaled by ``n/k``), so the
    cached map carries the same kind of value the exact map would on either side
    of the boundary -- one continuous quantity, no cliff.

    Both branches are deterministic, so consecutive cycles on an unchanged graph
    produce an identical map. The per-cycle percentile normalization the seed
    blend needs happens at the consumption point (``_pick_seeds``), over the
    candidate slice, NOT here: normalizing here would diverge the cached/streamed
    map from the exact map the parity tests compare against, and the seed term is
    what actually needs the scale-stable rank, so that is where the rank is taken.
    """
    node_count = graph.node_count()
    if node_count == 0:
        return {}

    if node_count <= exact_below():
        return graph.centrality()

    return approximate_centrality(
        graph,
        method=runtime_method(),
        k=runtime_k(),
        normalize=False,
    )
