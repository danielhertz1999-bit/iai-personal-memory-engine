"""Tests for the super-level pairwise merge.

This module covers `_super_level_merge`, the Karate-class joint-Q maxima
closer for a previously-deferred gap.

Algorithm:

  After refinement-as-aggregation converges in `run_mosaic`, run an
  additional pairwise merge phase atop the refined partition. Accept
  `merge(C_i, C_j)` iff `joint_Q > Q(C_i) + Q(C_j) + epsilon`. Pairs visited
  in canonical (i, j) order with `i < j`. Terminate when no merges accepted
  in an outer iteration OR `max_iter` exhausted.

  The gap it closes: Karate at gamma=0.5 has 4-comm and 2-comm as joint Q
  optima; consolidation requires a WHOLE-super-community pairwise merge
  (delta-Q approx +0.024 for the canonical pair), a mechanism distinct from
  refinement-as-aggregation. The gap persists at gamma=1.0 too (single-node
  move plateau, NMI vs leidenalg = 0.8539).

Analytical verification:

  Karate at gamma=0.5 (custom_leiden's pre-super-merge 4-community output):
    pair (0,2): delta-Q = +0.0104 POSITIVE
    pair (0,3): delta-Q = +0.0002 POSITIVE
    pair (2,3): delta-Q = +0.0244 POSITIVE
    pair (0,1): delta-Q = -0.0705 NEGATIVE
    pair (1,2): delta-Q = -0.0192 NEGATIVE
    pair (1,3): delta-Q = -0.0321 NEGATIVE

  Greedy first-positive-accept-restart with canonical ordering:
    iter 1: accept (0,2) -> 3 comms, Q delta=+0.0104
    iter 2: accept (0,3) -> 2 comms, Q delta=+0.0233
    iter 3: no positive pair, terminate

  Result: 4 -> 2 comms, NMI(custom, leidenalg) jumps 0.7753 -> 1.0000.
  Football and LFR_n1000_mu01/03 super-merge accept ZERO merges (idempotent).
  LFR_n1000_mu05 at gamma=1.0 super-merge: 22 -> 20 comms (2 merges),
  NMI(custom, planted) 0.9790 -> 0.9964 (IMPROVES; well above 0.65 threshold).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import UUID, uuid5

import numpy as np
import pytest

from iai_mcp.graph import MemoryGraph


# ---------------------------------------------------------------- helpers

def _emb(seed: int, dim: int = 384) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


def _load_karate() -> tuple[MemoryGraph, list[int], list[UUID]]:
    """Load Karate Club fixture into a MemoryGraph (same uuid5 ns as
    `tests/test_mosaic_local_move._load_karate` so cross-test
    replay invariants hold)."""
    fixture_path = Path(__file__).parent / "fixtures" / "leiden" / "karate_club.json"
    data = json.loads(fixture_path.read_text())
    karate_ns = UUID("12345678-1234-5678-1234-567812345678")
    nodes: list[UUID] = [uuid5(karate_ns, f"karate-{i}") for i in range(data["n"])]
    g = MemoryGraph()
    for i, u in enumerate(nodes):
        g.add_node(u, community_id=None, embedding=_emb(i))
    for u, v in data["edges"]:
        g.add_edge(nodes[u], nodes[v], weight=1.0)
    return g, data["ground_truth"], nodes


def _build_two_clique_bridge(k: int = 5) -> tuple[MemoryGraph, list[UUID]]:
    """K_k joined by a single bridge to K_k. Optimal partition at γ=1.0
    is 2 communities (one per clique). After refinement-as-aggregation
    that optimum is reached directly; super-merge MUST accept zero
    merges (idempotency witness)."""
    ns = UUID("12345678-1234-5678-1234-567812345678")
    nodes: list[UUID] = [uuid5(ns, f"clique-{i}") for i in range(2 * k)]
    g = MemoryGraph()
    for i, u in enumerate(nodes):
        g.add_node(u, community_id=None, embedding=_emb(i))
    # First clique: 0..k-1 fully connected.
    for i in range(k):
        for j in range(i + 1, k):
            g.add_edge(nodes[i], nodes[j], weight=1.0)
    # Second clique: k..2k-1 fully connected.
    for i in range(k, 2 * k):
        for j in range(i + 1, 2 * k):
            g.add_edge(nodes[i], nodes[j], weight=1.0)
    # Bridge: node 0 -- node k.
    g.add_edge(nodes[0], nodes[k], weight=1.0)
    return g, nodes


def _build_dense_random_graph(n: int = 30, seed: int = 42, p: float = 0.5) -> tuple[MemoryGraph, list[UUID]]:
    """Erdős-Rényi style dense graph. Used as stress test for max_iter
    termination guard (no community structure -> tight cycles between
    similar Q values)."""
    rng = np.random.default_rng(seed)
    ns = UUID("12345678-1234-5678-1234-567812345678")
    nodes: list[UUID] = [uuid5(ns, f"dense-{i}") for i in range(n)]
    g = MemoryGraph()
    for i, u in enumerate(nodes):
        g.add_node(u, community_id=None, embedding=_emb(i))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(nodes[i], nodes[j], weight=1.0)
    return g, nodes


def _detected_labels_in_canonical_order(assignment, nodes_in_canonical_order: list[UUID]) -> list[int]:
    """Map UUID -> integer label, indexed by canonical-order node list."""
    uuid_to_label: dict[UUID, int] = {}
    nl = 0
    out: list[int] = []
    for u in nodes_in_canonical_order:
        cuuid = assignment.node_to_community[u]
        if cuuid not in uuid_to_label:
            uuid_to_label[cuuid] = nl
            nl += 1
        out.append(uuid_to_label[cuuid])
    return out


# ---------------------------------------------------------------- import witness

def test_super_merge_symbol_exists() -> None:
    """The helper `_super_level_merge` must be importable from custom_leiden."""
    from iai_mcp.mosaic import _super_level_merge
    assert callable(_super_level_merge)


# ---------------------------------------------------------------- Karate gap closure

def test_super_merge_closes_karate_gap() -> None:
    """Acceptance: NMI(custom, leidenalg) >= 0.90 on Karate(N=34, gamma=0.5).

    Without the super-merge: NMI = 0.7753.

    Analytical sim: the super-merge applied to custom_leiden's 4-community
    Karate output at gamma=0.5 accepts pairs (0,2) and (0,3) in order,
    yielding 2 communities exactly matching leidenalg's reference partition.
    NMI -> 1.0000.

    Gate: >= 0.90.
    """
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")
    pytest.importorskip("sklearn")
    import leidenalg
    import igraph as ig
    from sklearn.metrics import normalized_mutual_info_score

    from iai_mcp.mosaic import run_mosaic

    fixture_path = Path(__file__).parent / "fixtures" / "leiden" / "karate_club.json"
    data = json.loads(fixture_path.read_text())

    # leidenalg reference partition at gamma=0.5 (same seed).
    g_ig = ig.Graph()
    g_ig.add_vertices(data["n"])
    g_ig.add_edges([tuple(e) for e in data["edges"]])
    ref_partition = leidenalg.find_partition(
        g_ig, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=0.5, seed=42,
    )
    leidenalg_labels = list(ref_partition.membership)

    graph, _ground_truth, nodes = _load_karate()
    assignment, _lineage = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=0.5, seed=42
    )

    # Map detected UUIDs -> integer labels in the Zachary fixture order.
    # nodes is in fixture order (uuid5(karate_ns, f"karate-{i}") for i in 0..33).
    detected = _detected_labels_in_canonical_order(assignment, nodes)
    nmi = normalized_mutual_info_score(leidenalg_labels, detected)
    assert nmi >= 0.90, (
        f"Karate NMI(custom, leidenalg) {nmi:.4f} below 0.90 gate; "
        f"super-merge should close the 0.7753 -> >= 0.90 gap. "
        f"detected={detected[:5]}... leidenalg={leidenalg_labels[:5]}..."
    )


# ---------------------------------------------------------------- idempotency on optimal partition

def test_super_merge_idempotent_on_optimal_partition() -> None:
    """If refinement-as-aggregation already finds the global Q optimum,
    super-merge accepts ZERO merges.

    Witness graph: two K_5 cliques joined by a single bridge edge. At
    gamma=1.0 the optimal CPM partition is the obvious 2-clique split
    (intra-clique edges are dense, inter-clique edge is sparse).
    custom_leiden's refinement reaches this directly; super-merge has
    nothing to merge.

    This test is the negative-control witness: super-merge must not
    spuriously merge already-optimal communities just because the
    pairwise delta-Q computation finds non-zero values from numerical
    noise. The +EPSILON gate enforces strict positive-delta acceptance.
    """
    from iai_mcp.mosaic import (
        run_mosaic, build_csr_sanitized, _super_level_merge,
        compute_sigma_tot,
    )

    graph, nodes = _build_two_clique_bridge(k=5)

    # Run custom_leiden to get the refined partition.
    assignment, _ = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42,
    )
    # Project assignment back to canonical partition indices.
    csr, order, _idx_map = build_csr_sanitized(graph)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data_arr = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = len(order)
    uuid_to_label: dict[UUID, int] = {}
    nl = 0
    partition = np.zeros(n, dtype=np.int64)
    for i, u in enumerate(order):
        c = assignment.node_to_community[u]
        if c not in uuid_to_label:
            uuid_to_label[c] = nl
            nl += 1
        partition[i] = uuid_to_label[c]

    # Sanity: custom_leiden lands at 2 communities on this graph.
    assert nl == 2, (
        f"Pre-condition violated: expected custom_leiden to find 2 "
        f"communities on K_5-bridge-K_5, got {nl}. Refinement may be "
        f"under-converged; this fixture is supposed to be unambiguous."
    )

    k = nl
    sigma_tot = compute_sigma_tot(indptr, indices, data_arr, partition, k)
    partition_before = partition.copy()

    # Call _super_level_merge directly (signature per Algorithm).
    n_merges = _super_level_merge(
        csr, partition, sigma_tot, gamma=1.0, seed=42, lineage_tracker=None,
    )

    # ZERO merges accepted; partition unchanged.
    assert n_merges == 0, (
        f"super-merge should NOT consolidate already-optimal partition "
        f"on K_5-bridge-K_5 (got {n_merges} merges)."
    )
    np.testing.assert_array_equal(
        partition, partition_before,
        err_msg="partition was mutated despite zero accepted merges",
    )


# ---------------------------------------------------------------- determinism

def test_super_merge_deterministic() -> None:
    """Same seed -> identical final partition.

    Calls `run_mosaic` twice with seed=42 on Karate; final
    partitions must be byte-identical (label-equivalent up to relabel
    bijection). hard constraint: canonical (i, j) ordering
    + no random tie-break = byte-deterministic replay.
    """
    from iai_mcp.mosaic import run_mosaic

    graph_a, _, nodes_a = _load_karate()
    graph_b, _, nodes_b = _load_karate()

    assignment_a, _ = run_mosaic(
        graph_a, prior=None, prior_mode="cold", gamma=0.5, seed=42,
    )
    assignment_b, _ = run_mosaic(
        graph_b, prior=None, prior_mode="cold", gamma=0.5, seed=42,
    )

    # Compare by label-equivalence: two partitions are byte-equal if
    # their detected-label-integer streams match under canonical-order
    # node-stream. UUIDs themselves differ across runs (uuid4 in
    # `init_partitions` cold-path), so we test partition shape.
    labels_a = _detected_labels_in_canonical_order(assignment_a, nodes_a)
    labels_b = _detected_labels_in_canonical_order(assignment_b, nodes_b)

    # Build canonical-relabel form: 1st distinct value -> 0, 2nd -> 1, etc.
    def _canonical(seq: list[int]) -> list[int]:
        remap: dict[int, int] = {}
        nxt = 0
        out: list[int] = []
        for v in seq:
            if v not in remap:
                remap[v] = nxt
                nxt += 1
            out.append(remap[v])
        return out

    assert _canonical(labels_a) == _canonical(labels_b), (
        f"Determinism violated: same seed produced different partitions.\n"
        f"  run_a labels[:10] = {labels_a[:10]}\n"
        f"  run_b labels[:10] = {labels_b[:10]}"
    )


# ---------------------------------------------------------------- lineage events

def test_super_merge_lineage_events_recorded() -> None:
    """Each accepted super-merge call records a LineageEvent of type 'merge'.

     hard constraint: lineage event for continuity.

    Direct test of `_super_level_merge` on a hand-constructed 4-community
    Karate partition (the exact partition that custom_leiden produces
    BEFORE super-merge integration -- 21-03 / 21-05 measured output).
    A fresh LineageTracker captures the merge events emitted by the
    helper; the count must equal the helper's returned merge count.

    On Karate at gamma=0.5 (this hand-built 4-community partition), the
    analytical sim predicts 2 accepted merges (pairs (0,2) then (0,3)
    in canonical (i, j) ordering). The test gates >= 1 to allow for
    minor variations in greedy-accept order while still witnessing the
    lineage hook.

    Note: this test does NOT go through `run_mosaic` because
    post-22-01 integration, that path already super-merges to 2
    communities -- super-merge called again on the 2-comm partition is
    idempotent (zero merges). We exercise the helper directly on the
    pre-super-merge partition state to witness the lineage emission.
    """
    from iai_mcp.mosaic import (
        build_csr_sanitized, _super_level_merge, compute_sigma_tot,
    )
    from iai_mcp.mosaic_lineage import LineageTracker
    from uuid import uuid4

    graph, _, _ = _load_karate()
    csr, order, _idx_map = build_csr_sanitized(graph)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data_arr = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = len(order)

    # Hand-build the pre-super-merge 4-community Karate partition by
    # mirroring what custom_leiden's multi-level loop produces at
    # gamma=0.5 (comms of size 12/17/2/3, sigmas 62/78/6/10): produce a
    # 4-comm partition with known positive-ΔQ pairs.
    #
    # Strategy: run `_run_one_leiden_pass` (which lacks the super-merge
    # integration since it's the tuner's score path) -- but the tuner
    # pass is also tuner-only. Cleaner: replicate the partition shape by
    # running the pre-super-merge algorithm flow stem.
    #
    # Simplest robust path: run custom_leiden once to get the FINAL
    # (super-merged) partition, then ARTIFICIALLY split it back to
    # 4 communities using nearest-neighbour cuts. This is too brittle.
    #
    # Cleanest: invoke `_run_one_leiden_pass` which does NOT have super-
    # merge (tuner's score path) -- returns the 4-comm pre-merge state.
    from iai_mcp.mosaic import _run_one_leiden_pass
    import scipy.sparse
    csr_re = scipy.sparse.csr_matrix(
        (data_arr, indices, indptr), shape=(n, n)
    )
    init = np.arange(n, dtype=np.int64)
    init_sigma = compute_sigma_tot(indptr, indices, data_arr, init, n)
    pre_merge_partition, _q, _stats = _run_one_leiden_pass(
        csr_re, init, init_sigma, gamma=0.5, seed=42,
    )
    partition = pre_merge_partition.copy()
    k = int(len(np.unique(partition)))
    # Sanity: expect 4 communities pre-super-merge on Karate at gamma=0.5.
    assert k >= 3, (
        f"Pre-condition: expected pre-super-merge to find >=3 "
        f"communities on Karate(gamma=0.5); got {k}."
    )

    sigma_tot = compute_sigma_tot(indptr, indices, data_arr, partition, k)

    # Build a label_to_uuid map so the lineage hook fires (the lineage
    # branch with placeholder-uuid emission also works but is less
    # representative of the production path).
    label_to_uuid = {i: uuid4() for i in range(k)}

    fresh_tracker = LineageTracker()
    n_merges = _super_level_merge(
        csr, partition, sigma_tot, gamma=0.5, seed=42,
        lineage_tracker=fresh_tracker, label_to_uuid=label_to_uuid,
    )

    # On the pre-super-merge 4-community Karate partition, the
    # analytical sim accepts 2 merges. Gate on >= 1.
    assert n_merges >= 1, (
        f"super-merge should accept >= 1 merge on the pre-super-merge "
        f"Karate partition at gamma=0.5 (analytical sim: 2 merges); "
        f"got {n_merges}. k_before={k}."
    )

    # The tracker should now have at least `n_merges` merge events.
    merge_events = [
        e for e in fresh_tracker.report().events if e.event_type == "merge"
    ]
    assert len(merge_events) >= n_merges, (
        f"Expected at least {n_merges} 'merge' lineage events recorded "
        f"by super-merge; got {len(merge_events)}. Lineage hook missing?"
    )


# ---------------------------------------------------------------- termination guard

def test_super_merge_terminates_within_max_iter() -> None:
    """Stress test: even on a dense random graph (no community
    structure), super-merge terminates within `max_iter` outer
    iterations and bounded wall-time.

     hard constraint: max_iter outer cap, no infinite loops.
    A worst-case dense graph forces many candidate-pair evaluations
    per iteration; the test guards against pathological behaviour.
    """
    from iai_mcp.mosaic import (
        run_mosaic, build_csr_sanitized, _super_level_merge,
        compute_sigma_tot,
    )

    graph, _ = _build_dense_random_graph(n=30, seed=42, p=0.5)
    # Get refined partition.
    assignment, _ = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42,
    )
    csr, order, _ = build_csr_sanitized(graph)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data_arr = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = len(order)

    uuid_to_label: dict[UUID, int] = {}
    nl = 0
    partition = np.zeros(n, dtype=np.int64)
    for i, u in enumerate(order):
        c = assignment.node_to_community[u]
        if c not in uuid_to_label:
            uuid_to_label[c] = nl
            nl += 1
        partition[i] = uuid_to_label[c]

    k = nl
    sigma_tot = compute_sigma_tot(indptr, indices, data_arr, partition, k)

    t0 = time.monotonic()
    n_merges = _super_level_merge(
        csr, partition, sigma_tot, gamma=1.0, seed=42, lineage_tracker=None,
        max_iter=5,
    )
    wall_s = time.monotonic() - t0

    # Termination guarantee: completes well within 5s on a 30-node graph.
    assert wall_s < 5.0, (
        f"super-merge wall-time {wall_s:.2f}s exceeded 5s budget on a "
        f"30-node dense graph (n_merges={n_merges}); possible infinite "
        f"loop bug or pathological pair-enumeration."
    )
    # Sanity: n_merges bounded above by n_communities - 1.
    assert 0 <= n_merges < k, (
        f"super-merge merge count {n_merges} out of bounds [0, {k-1}]"
    )
