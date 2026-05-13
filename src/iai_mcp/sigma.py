""": small-world sigma as Ashby ultrastability diagnostic.

Ground-truth reference: Humphries MD, Gurney K (2008) "Network 'small-world-ness':
a quantitative method for determining canonical network equivalence."

Constitutional anchor:
- sigma is a CYBERNETIC DIAGNOSTIC (Ashby ultrastability), not a "RAG fallback".
- Cold-start sigma<1 at N<500 is a DEVELOPMENTAL phase, not pathological.
  Emit kind=sigma_observation phase=developmental + boost Hebbian rate.
- Mid-life drift sigma<1 at N>=500 emits kind=sigma_drift as an S4 event.
- sigma trajectory is published as a deep-time metric, NEVER a routing
  decision. No code path in this module switches retrieval modes on sigma.

Design discipline:
- DO NOT use NetworkX's built-in small-worldness function. NetworkX 3.6.1's
  built-in (niter=100, nrand=10) is empirically unusable at N>=200 (timed out
  at 60s+ during research session).
- Custom `fast_sigma` follows Humphries-Gurney 2008 directly with a small
  number of single-reference Erdos-Renyi random graphs (G(n, m), same edge
  count). Validated 0.05s @ N=200, 0.34s @ N=500, 1.28s @ N=1000.

Module-level constants:
- SIGMA_N_FLOOR = 200 -- D-SIGMA-01 floor (imports semantically from
  community.SMALL_N_FLAT -- same Humphries-Gurney 2008 floor).
- SIGMA_MID_LIFE_THRESHOLD = 500 -- D-SIGMA-03 mid-life regime threshold
  (imports semantically from community.MID_N_LEIDEN).

Public API:
- compute_sigma(graph, *, seed=42)            -> Optional[float]
- fast_sigma(graph, *, n_random=3, seed=42)   -> tuple[float, float, float, float, float]
- classify_regime(N, sigma)                   -> str
- compute_topology_snapshot(graph)            -> dict
- compute_and_emit(store)                     -> dict
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

import networkx as nx

from iai_mcp.events import write_event

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore


# D-SIGMA-01: sigma is undefined below N=200 (Humphries-Gurney 2008 floor).
# Aliased semantically from community.SMALL_N_FLAT -- same constitutional floor.
SIGMA_N_FLOOR: int = 200

# D-SIGMA-03: mid-life vs developmental boundary (community.MID_N_LEIDEN).
SIGMA_MID_LIFE_THRESHOLD: int = 500

# Event kinds emitted by this module. Naming follows the snake_case
# noun_verb shape established in s4.py / s5.py.
SIGMA_OBSERVATION_KIND: str = "sigma_observation"
SIGMA_DRIFT_KIND: str = "sigma_drift"

# Hebbian rate boost applied during developmental phase (D-SIGMA-02).
HEBBIAN_DEVELOPMENTAL_BOOST_FACTOR: float = 1.3
HEBBIAN_DEVELOPMENTAL_BOOST_TTL_SESSIONS: int = 5

# Knob name we tag in profile_updated events when boosting the Hebbian rate
# during developmental phase. The 11-knob registry is NOT modified -- this is
# a transient operational tag, not an AUTIST kernel knob.
HEBBIAN_RATE_KNOB: str = "hebbian_rate"


def _largest_cc(graph: "nx.Graph") -> "nx.Graph":
    """Return the largest connected component as a copy.

    NetworkX raises on disconnected inputs to ``average_shortest_path_length``;
    take the largest CC up front so the rest of fast_sigma can stay simple.
    """
    if graph.number_of_nodes() == 0:
        return graph
    if nx.is_connected(graph):
        return graph
    largest = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest).copy()


def fast_sigma(
    graph: "nx.Graph",
    *,
    n_random: int = 3,
    seed: int = 42,
) -> tuple[float, float, float, float, float]:
    """Humphries-Gurney 2008 sigma via single-reference random graph(s).

    Returns ``(sigma, C, L, Cr, Lr)`` where:
    - sigma = (C / Cr) / (L / Lr)
    - C / L : clustering / characteristic path length on the input graph
    - Cr / Lr : same metrics averaged over ``n_random`` Erdos-Renyi G(n, m)
      reference graphs.

    DO NOT use NetworkX's built-in small-worldness function -- it is
    empirically unusable at N>=200 (>60s timeout).
    This implementation builds ONE G(n, m) reference per seed and averages
    the C and L values, NOT the library's full edge-rewiring loop.

    Pre-processing: when the input graph is disconnected, the largest
    connected component is taken first. NetworkX raises on disconnected
    inputs to ``average_shortest_path_length``.

    Notes
    -----
    - Returns NaN sigma when Cr or Lr collapses to zero (degenerate reference;
      shouldn't happen at our N>=200 floor but defensive).
    - Deterministic per ``seed`` -- the n_random reference graphs use
      ``seed, seed+1, ..., seed+n_random-1``.
    """
    g = _largest_cc(graph)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    if n < 2 or m == 0:
        return (float("nan"), 0.0, 0.0, 0.0, 0.0)

    C = float(nx.average_clustering(g))
    L = float(nx.average_shortest_path_length(g))

    Cs: list[float] = []
    Ls: list[float] = []
    for k in range(max(1, n_random)):
        gr_full = nx.gnm_random_graph(n, m, seed=seed + k)
        # Same disconnected-graph guard for the reference.
        if not nx.is_connected(gr_full):
            largest = max(nx.connected_components(gr_full), key=len)
            gr = gr_full.subgraph(largest).copy()
        else:
            gr = gr_full
        if gr.number_of_nodes() < 2 or gr.number_of_edges() == 0:
            continue
        Cs.append(float(nx.average_clustering(gr)))
        Ls.append(float(nx.average_shortest_path_length(gr)))

    if not Cs or not Ls:
        return (float("nan"), C, L, 0.0, 0.0)

    Cr = sum(Cs) / len(Cs)
    Lr = sum(Ls) / len(Ls)
    if Cr <= 0 or Lr <= 0 or L <= 0:
        return (float("nan"), C, L, Cr, Lr)

    sigma_val = (C / Cr) / (L / Lr)
    return (sigma_val, C, L, Cr, Lr)


def compute_sigma(graph: "nx.Graph", *, seed: int = 42) -> Optional[float]:
    """D-SIGMA-01: sigma at N>=SIGMA_N_FLOOR; otherwise None.

    Returns None for graphs with fewer than SIGMA_N_FLOOR nodes -- below
    that threshold, the random-graph baselines are too noisy to interpret
    (Humphries-Gurney 2008).
    """
    if graph.number_of_nodes() < SIGMA_N_FLOOR:
        return None
    sigma_val, *_ = fast_sigma(graph, seed=seed)
    if isinstance(sigma_val, float) and math.isnan(sigma_val):
        return None
    return float(sigma_val)


def classify_regime(N: int, sigma: Optional[float]) -> str:
    """Four-cell regime truth table (D-SIGMA-02 / D-SIGMA-03).

    Returns one of:
    - "insufficient_data" : sigma is None (N < SIGMA_N_FLOOR)
    - "developmental"     : N < SIGMA_MID_LIFE_THRESHOLD AND sigma < 1
    - "mid_life_drift"    : N >= SIGMA_MID_LIFE_THRESHOLD AND sigma < 1
    - "healthy"           : sigma >= 1 (any N >= floor)
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


def _coerce_to_nx_graph(graph_or_wrapper) -> "nx.Graph":
    """Accept either a raw nx.Graph or our MemoryGraph wrapper.

    MemoryGraph (src/iai_mcp/graph.py) carries the underlying nx.Graph as
    ``_nx``. The CLI passes a MemoryGraph; tests / fast_sigma also accept raw
    nx.Graph for portability.
    """
    if isinstance(graph_or_wrapper, nx.Graph):
        return graph_or_wrapper
    underlying = getattr(graph_or_wrapper, "_nx", None)
    if isinstance(underlying, nx.Graph):
        return underlying
    raise TypeError(
        f"expected nx.Graph or MemoryGraph wrapper, got {type(graph_or_wrapper).__name__}"
    )


def compute_topology_snapshot(graph) -> dict:
    """Snapshot dict consumed by the topology CLI subcommand.

    Returns: ``{C, L, sigma, community_count, rich_club_ratio, N, regime}``.

    - C : average clustering on the largest connected component.
    - L : average shortest path length on the largest CC.
    - sigma : compute_sigma(graph) (None if N < SIGMA_N_FLOOR).
    - community_count : Leiden community count ( reuse via
      community.detect_communities); uses an isolated MemoryGraph wrapper.
    - rich_club_ratio : len(rich_club_nodes) / N ( reuse).
    - N : node count.
    - regime : classify_regime(N, sigma).
    """
    nx_g = _coerce_to_nx_graph(graph)
    N = int(nx_g.number_of_nodes())

    if N == 0:
        return {
            "C": 0.0, "L": 0.0, "sigma": None,
            "community_count": 0, "rich_club_ratio": 0.0,
            "N": 0, "regime": "insufficient_data",
        }

    g_cc = _largest_cc(nx_g)
    try:
        C = float(nx.average_clustering(g_cc)) if g_cc.number_of_nodes() else 0.0
    except Exception:
        C = 0.0
    try:
        L = (
            float(nx.average_shortest_path_length(g_cc))
            if g_cc.number_of_nodes() >= 2 and g_cc.number_of_edges() > 0
            else 0.0
        )
    except Exception:
        L = 0.0

    sigma_val = compute_sigma(nx_g)

    # community_count + rich_club_ratio require the MemoryGraph wrapper.
    community_count = 0
    rich_club_ratio = 0.0
    try:
        from iai_mcp.community import detect_communities
        from iai_mcp.graph import MemoryGraph
        from iai_mcp.richclub import rich_club_nodes
        if isinstance(graph, MemoryGraph):
            mg = graph
        else:
            mg = None
        if mg is not None:
            try:
                assignment = detect_communities(mg, prior=None)
                community_count = int(len(assignment.community_centroids))
            except Exception:
                community_count = 0
            try:
                rc = rich_club_nodes(mg, percent=0.10)
                rich_club_ratio = (len(rc) / N) if N > 0 else 0.0
            except Exception:
                rich_club_ratio = 0.0
    except Exception:
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

    Per D-SIGMA-02 the developmental phase warrants a temporary
    Hebbian-rate boost. Rather than mutating the 10-knob AUTIST profile
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
    - "developmental"     -> kind=sigma_observation, data.phase="developmental",
                             AND a profile_updated event for hebbian_rate boost.
    - "mid_life_drift"    -> kind=sigma_drift, data with full snapshot.
    - "healthy"           -> kind=sigma_observation, data.phase="healthy".
    - "insufficient_data" -> kind=sigma_observation, data.phase="insufficient_data".

    NEVER toggles retrieval modes (constitutional guard).
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
        except Exception:
            # Diagnostic only: never block the sigma observation on the
            # follow-up Hebbian boost.
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
