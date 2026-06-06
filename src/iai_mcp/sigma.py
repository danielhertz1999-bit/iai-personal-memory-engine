"""Small-world sigma as diagnostic.

Ground-truth reference: Humphries MD, Gurney K (2008) "Network 'small-world-ness':
a quantitative method for determining canonical network equivalence."

Invariants:
- sigma is a DIAGNOSTIC, not a "RAG fallback".
- Cold-start sigma<1 at N<500 is a DEVELOPMENTAL phase, not pathological.
  Emit kind=sigma_observation phase=developmental + boost Hebbian rate.
- Mid-life drift sigma<1 at N>=500 emits kind=sigma_drift as an S4 event.
- sigma trajectory is published as a deep-time metric, NEVER a routing
  decision. No code path in this module switches retrieval modes on sigma.

Design discipline:
- DO NOT use an external library's reference small-worldness routine;
  reference implementations with ``niter=100, nrand=10`` are empirically
  unusable at N>=200 (timed out at 60s+ in benchmarking).
- Custom ``fast_sigma`` follows Humphries-Gurney 2008 directly with a small
  number of single-reference Erdos-Renyi random graphs (G(n, m), same edge
  count). Validated 0.05s @ N=200, 0.34s @ N=500, 1.28s @ N=1000.

Backend discipline (post-eviction):
- All graph algorithms route through the native Rust
  ``iai_mcp_native.graph`` namespace (``lilli_graph`` alias).
- ``fast_sigma`` accepts a ``MemoryGraph`` instance as its canonical input
  contract. Tests that build oracle fixtures via the optional dev-only
  library wrap the result via the adapter exposed in
  ``tests/conftest.py``.

Module-level constants:
- ``SIGMA_N_FLOOR = 200`` -- floor below which sigma is undefined (imports
  semantically from community.SMALL_N_FLAT -- same Humphries-Gurney 2008 floor).
- ``SIGMA_MID_LIFE_THRESHOLD = 500`` -- mid-life regime threshold
  (imports semantically from community.MID_N_LEIDEN).

Public API:
- ``compute_sigma(graph, *, seed=42)`` -> Optional[float]
- ``fast_sigma(graph, *, n_random=3, seed=42)`` -> tuple[float, float, float, float, float]
- ``classify_regime(N, sigma)`` -> str
- ``compute_topology_snapshot(graph)`` -> dict
- ``compute_and_emit(store)`` -> dict
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

import numpy as np

from iai_mcp.events import write_event
from iai_mcp_native import graph as lilli_graph

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.store import MemoryStore


# sigma is undefined below N=200 (Humphries-Gurney 2008 floor).
# Aliased from community.SMALL_N_FLAT.
SIGMA_N_FLOOR: int = 200

# mid-life vs developmental boundary (community.MID_N_LEIDEN).
SIGMA_MID_LIFE_THRESHOLD: int = 500

# Event kinds emitted by this module. Naming follows the snake_case
# noun_verb shape established in s4.py / s5.py.
SIGMA_OBSERVATION_KIND: str = "sigma_observation"
SIGMA_DRIFT_KIND: str = "sigma_drift"

# Hebbian rate boost applied during developmental phase.
HEBBIAN_DEVELOPMENTAL_BOOST_FACTOR: float = 1.3
HEBBIAN_DEVELOPMENTAL_BOOST_TTL_SESSIONS: int = 5

# Knob name we tag in profile_updated events when boosting the Hebbian rate
# during developmental phase. The 11-knob registry is NOT modified -- this is
# a transient operational tag, not an AUTIST kernel knob.
HEBBIAN_RATE_KNOB: str = "hebbian_rate"


# ---------------------------------------------------------------- CSR helpers


def _build_csr_from_edge_lists(
    u_list, v_list, n_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert symmetric edge lists to a CSR triple.

    Used to wrap the output of ``lilli_graph.gnm_random_graph`` (which
    returns a pair of ``(u_list, v_list)`` parallel arrays) into the
    canonical CSR triple that the remaining native graph algorithms
    consume. Each undirected edge ``(u, v)`` is materialized symmetrically
    (both directions) so the CSR rows match the unweighted adjacency
    semantics expected by ``is_connected`` / ``connected_components`` /
    ``average_clustering`` / ``average_shortest_path_length``.

    Returns ``(indptr int64, indices int64, data float64)``. ``data`` is
    all-ones — the gnm generator yields unweighted edges.
    """
    import scipy.sparse

    m = len(u_list)
    if m == 0 or n_nodes == 0:
        return (
            np.zeros(max(n_nodes + 1, 1), dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    # Symmetric edges: emit both u->v AND v->u so the resulting CSR is
    # the standard undirected adjacency matrix.
    rows = np.concatenate(
        [
            np.asarray(u_list, dtype=np.int64),
            np.asarray(v_list, dtype=np.int64),
        ]
    )
    cols = np.concatenate(
        [
            np.asarray(v_list, dtype=np.int64),
            np.asarray(u_list, dtype=np.int64),
        ]
    )
    data = np.ones(2 * m, dtype=np.float64)
    coo = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    csr = coo.tocsr()
    return (
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        csr.data.astype(np.float64),
    )


def _induced_csr_from_component(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    component_nodes: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract the induced subgraph CSR for ``component_nodes``.

    Replaces the legacy ``g.subgraph(largest_cc).copy()`` chain with an
    explicit CSR-domain helper. Returns ``(sub_indptr, sub_indices,
    sub_data, sub_node_count)`` where the new node indices are
    ``0..sub_node_count-1`` in the order of ``component_nodes``.
    """
    old_to_new = {old: new for new, old in enumerate(component_nodes)}
    node_set = set(component_nodes)
    sub_n = len(component_nodes)
    if sub_n == 0:
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            0,
        )
    sub_rows: list[int] = []
    sub_cols: list[int] = []
    sub_data: list[float] = []
    for new_u, old_u in enumerate(component_nodes):
        start = int(indptr[old_u])
        end = int(indptr[old_u + 1])
        for idx in range(start, end):
            old_v = int(indices[idx])
            if old_v in node_set:
                new_v = old_to_new[old_v]
                sub_rows.append(new_u)
                sub_cols.append(new_v)
                sub_data.append(
                    float(data[idx]) if data is not None and len(data) > idx else 1.0
                )
    if not sub_rows:
        return (
            np.zeros(sub_n + 1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            sub_n,
        )
    import scipy.sparse

    coo = scipy.sparse.coo_matrix(
        (sub_data, (sub_rows, sub_cols)),
        shape=(sub_n, sub_n),
    )
    csr = coo.tocsr()
    return (
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        csr.data.astype(np.float64),
        sub_n,
    )


def _largest_component_csr(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return the CSR of the largest connected component.

    If the graph is already connected, returns the input unchanged with
    ``sub_node_count == n_nodes``. Mirrors the legacy ``_largest_cc``
    semantic but in the CSR domain.
    """
    if n_nodes == 0:
        return indptr, indices, data, 0
    components = lilli_graph.connected_components(indptr, indices, n_nodes)
    if not components:
        return indptr, indices, data, n_nodes
    if len(components) == 1:
        return indptr, indices, data, n_nodes
    largest = max(components, key=len)
    return _induced_csr_from_component(indptr, indices, data, list(largest))


# ---------------------------------------------------------------- sigma core


def fast_sigma(
    graph: "MemoryGraph",
    *,
    n_random: int = 3,
    seed: int = 42,
) -> tuple[float, float, float, float, float]:
    """Humphries-Gurney 2008 sigma via single-reference random graph(s).

    Returns ``(sigma, C, L, Cr, Lr)`` where:
    - sigma = (C / Cr) / (L / Lr)
    - C / L: clustering / characteristic path length on the input graph
    - Cr / Lr: same metrics averaged over ``n_random`` Erdos-Renyi G(n, m)
      reference graphs.

    Input contract: ``graph`` is a ``MemoryGraph`` instance. ``MemoryGraph``
    exposes ``to_csr_arrays()`` which yields the canonical ``(indptr,
    indices, data)`` triple consumed by every native algorithm. Tests that
    build oracle fixtures via the optional dev-only library wrap the result
    via the adapter exposed in ``tests/conftest.py``.

    Pre-processing: when the input graph is disconnected, the largest
    connected component is taken first. The native shortest-path kernel
    raises on disconnected inputs (mirrors the published semantic) so
    we strip the rest of the graph here.

    Notes
    -----
    - Returns NaN sigma when Cr or Lr collapses to zero (degenerate
      reference; shouldn't happen at our N>=200 floor but defensive).
    - Deterministic per ``seed`` -- the n_random reference graphs use
      ``seed, seed+1,..., seed+n_random-1``.
    """
    indptr, indices, data = graph.to_csr_arrays()
    n_nodes = len(indptr) - 1
    if n_nodes < 2 or len(indices) == 0:
        return (float("nan"), 0.0, 0.0, 0.0, 0.0)

    # Restrict the source graph to its largest connected component so the
    # shortest-path kernel never raises on disconnected input.
    sub_indptr, sub_indices, sub_data, n = _largest_component_csr(
        indptr, indices, data, n_nodes
    )
    # Edge count on the (undirected) largest CC. The CSR is symmetric so
    # the directed edge count is twice the undirected count.
    m = int(len(sub_indices) // 2)
    if n < 2 or m == 0:
        return (float("nan"), 0.0, 0.0, 0.0, 0.0)

    C = float(lilli_graph.average_clustering(sub_indptr, sub_indices, n))
    L = float(lilli_graph.average_shortest_path_length(sub_indptr, sub_indices, n))

    Cs: list[float] = []
    Ls: list[float] = []
    for k in range(max(1, n_random)):
        u_list, v_list = lilli_graph.gnm_random_graph(n, m, seed=seed + k)
        ref_indptr, ref_indices, ref_data = _build_csr_from_edge_lists(
            u_list, v_list, n
        )
        # Same disconnected-graph guard for the reference.
        ref_n = n
        try:
            ref_connected = lilli_graph.is_connected(ref_indptr, ref_indices, ref_n)
        except ValueError:
            # Empty graph case: skip this reference.
            continue
        if not ref_connected:
            ref_indptr, ref_indices, ref_data, ref_n = _largest_component_csr(
                ref_indptr, ref_indices, ref_data, ref_n
            )
        if ref_n < 2 or len(ref_indices) == 0:
            continue
        Cs.append(
            float(lilli_graph.average_clustering(ref_indptr, ref_indices, ref_n))
        )
        Ls.append(
            float(
                lilli_graph.average_shortest_path_length(
                    ref_indptr, ref_indices, ref_n
                )
            )
        )

    if not Cs or not Ls:
        return (float("nan"), C, L, 0.0, 0.0)

    Cr = sum(Cs) / len(Cs)
    Lr = sum(Ls) / len(Ls)
    if Cr <= 0 or Lr <= 0 or L <= 0:
        return (float("nan"), C, L, Cr, Lr)

    sigma_val = (C / Cr) / (L / Lr)
    return (sigma_val, C, L, Cr, Lr)


def compute_sigma(graph: "MemoryGraph", *, seed: int = 42) -> Optional[float]:
    """Compute sigma at N>=SIGMA_N_FLOOR; returns None for smaller graphs.

    Returns None for graphs with fewer than SIGMA_N_FLOOR nodes -- below
    that threshold, the random-graph baselines are too noisy to interpret
    (Humphries-Gurney 2008).
    """
    if graph.node_count() < SIGMA_N_FLOOR:
        return None
    sigma_val, *_ = fast_sigma(graph, seed=seed)
    if isinstance(sigma_val, float) and math.isnan(sigma_val):
        return None
    return float(sigma_val)


def classify_regime(N: int, sigma: Optional[float]) -> str:
    """Four-cell regime truth table.

    Returns one of:
    - "insufficient_data": sigma is None (N < SIGMA_N_FLOOR)
    - "developmental": N < SIGMA_MID_LIFE_THRESHOLD AND sigma < 1
    - "mid_life_drift": N >= SIGMA_MID_LIFE_THRESHOLD AND sigma < 1
    - "healthy": sigma >= 1 (any N >= floor)
    """
    if sigma is None:
        return "insufficient_data"
    if isinstance(sigma, float) and math.isnan(sigma):
        return "insufficient_data"
    if sigma < 1.0:
        if N < SIGMA_MID_LIFE_THRESHOLD:
            return "developmental"
        return "mid_life_drift"
    return "healthy"


def compute_topology_snapshot(graph) -> dict:
    """Snapshot dict consumed by the topology CLI subcommand.

    Returns: ``{C, L, sigma, community_count, rich_club_ratio, N, regime}``.

    - C: average clustering on the largest connected component.
    - L: average shortest path length on the largest CC.
    - sigma: compute_sigma(graph) (None if N < SIGMA_N_FLOOR).
    - community_count: Leiden community count.
    - rich_club_ratio: len(rich_club_nodes) / N.
    - N: node count.
    - regime: classify_regime(N, sigma).
    """
    from iai_mcp.graph import MemoryGraph

    if not isinstance(graph, MemoryGraph):
        raise TypeError(
            f"compute_topology_snapshot expects MemoryGraph, got "
            f"{type(graph).__name__}"
        )

    N = int(graph.node_count())
    if N == 0:
        return {
            "C": 0.0,
            "L": 0.0,
            "sigma": None,
            "community_count": 0,
            "rich_club_ratio": 0.0,
            "N": 0,
            "regime": "insufficient_data",
        }

    indptr, indices, data = graph.to_csr_arrays()
    n_nodes = len(indptr) - 1

    # Largest connected component for C / L (matches the legacy semantic
    # where average_shortest_path_length raised on disconnected inputs).
    sub_indptr, sub_indices, sub_data, sub_n = _largest_component_csr(
        indptr, indices, data, n_nodes
    )

    C = 0.0
    if sub_n >= 1:
        try:
            C = float(
                lilli_graph.average_clustering(sub_indptr, sub_indices, sub_n)
            )
        except (RuntimeError, ValueError):
            C = 0.0

    L = 0.0
    if sub_n >= 2 and len(sub_indices) > 0:
        try:
            L = float(
                lilli_graph.average_shortest_path_length(
                    sub_indptr, sub_indices, sub_n
                )
            )
        except (RuntimeError, ValueError):
            L = 0.0

    sigma_val = compute_sigma(graph)

    community_count = 0
    rich_club_ratio = 0.0
    try:
        from iai_mcp.community import detect_communities
        from iai_mcp.richclub import rich_club_nodes

        try:
            assignment = detect_communities(
                graph, prior=None, prior_mode="seeded"
            )
            community_count = int(len(assignment.community_centroids))
        except (RuntimeError, ValueError, TypeError):
            community_count = 0
        try:
            rc = rich_club_nodes(graph, percent=0.10)
            rich_club_ratio = (len(rc) / N) if N > 0 else 0.0
        except (RuntimeError, ValueError, TypeError):
            rich_club_ratio = 0.0
    except (ImportError, RuntimeError, TypeError):
        pass

    regime = classify_regime(N, sigma_val)
    return {
        "C": C,
        "L": L,
        "sigma": sigma_val,
        "community_count": community_count,
        "rich_club_ratio": rich_club_ratio,
        "N": N,
        "regime": regime,
    }


def _bump_hebbian_rate_developmental(store: "MemoryStore", N: int) -> None:
    """Emit a profile_updated event marking the Hebbian-rate boost.

    The developmental phase warrants a temporary Hebbian-rate boost.
    Rather than mutating the 10-knob AUTIST profile
    registry (which would violate len(PROFILE_KNOBS)==11), we record the
    intent as a profile_updated event with knob='hebbian_rate'. Downstream
    Hebbian write paths can read the most recent value and apply it.
    """
    write_event(
        store,
        kind="profile_updated",
        data={
            "knob": HEBBIAN_RATE_KNOB,
            "old": 1.0,
            "new": HEBBIAN_DEVELOPMENTAL_BOOST_FACTOR,
            "ttl_sessions": HEBBIAN_DEVELOPMENTAL_BOOST_TTL_SESSIONS,
            "reason": "sigma_developmental_phase",
            "N": N,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        severity="info",
    )


def compute_and_emit(store: "MemoryStore") -> dict:
    """S4 offline-pass entry point: build runtime graph, snapshot, emit event.

    Routes to the correct event kind based on the regime classification:
    - "developmental" -> kind=sigma_observation, data.phase="developmental",
                             AND a profile_updated event for hebbian_rate boost.
    - "mid_life_drift" -> kind=sigma_drift, data with full snapshot.
    - "healthy" -> kind=sigma_observation, data.phase="healthy".
    - "insufficient_data" -> kind=sigma_observation, data.phase="insufficient_data".

    NEVER toggles retrieval modes (invariant).
    """
    from iai_mcp import retrieve

    graph_bundle = retrieve.build_runtime_graph(store)
    # build_runtime_graph returns (graph, assignment, rich_club).
    if isinstance(graph_bundle, tuple):
        graph = graph_bundle[0]
    else:
        graph = graph_bundle

    snap = compute_topology_snapshot(graph)
    regime = snap.get("regime", "insufficient_data")

    base_data = {
        "sigma": snap.get("sigma"),
        "N": snap.get("N", 0),
        "C": snap.get("C", 0.0),
        "L": snap.get("L", 0.0),
        "community_count": snap.get("community_count", 0),
        "rich_club_ratio": snap.get("rich_club_ratio", 0.0),
        "regime": regime,
    }

    if regime == "mid_life_drift":
        write_event(
            store,
            kind=SIGMA_DRIFT_KIND,
            data={**base_data, "phase": "mid_life_drift"},
            severity="warning",
        )
    elif regime == "developmental":
        write_event(
            store,
            kind=SIGMA_OBSERVATION_KIND,
            data={**base_data, "phase": "developmental"},
            severity="info",
        )
        try:
            _bump_hebbian_rate_developmental(store, int(snap.get("N", 0)))
        except (OSError, RuntimeError, ValueError) as exc:
            # Diagnostic only: never block the sigma observation on the
            # follow-up Hebbian boost.
            logger.debug("hebbian_rate_boost_failed", extra={"err": str(exc)[:80]})
            pass
    elif regime == "healthy":
        write_event(
            store,
            kind=SIGMA_OBSERVATION_KIND,
            data={**base_data, "phase": "healthy"},
            severity="info",
        )
    else:  # insufficient_data
        write_event(
            store,
            kind=SIGMA_OBSERVATION_KIND,
            data={**base_data, "phase": "insufficient_data"},
            severity="info",
        )

    return snap
