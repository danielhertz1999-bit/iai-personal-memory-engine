"""CLUSTER_SUMMARY must stay bounded in memory and write batched, not per-cluster.

Two pathologies are guarded here:

1. Pairing blow-up — ``_process_cluster_summaries`` pairs the nodes of each
   connected component with ``combinations``. On a single large component this is
   O(k^2) and exhausts memory. The per-cluster pairing must be capped.

2. Per-cluster serial writes — issuing one ``boost_edges`` per cluster
   re-materializes and re-scans the whole edges table once per cluster, grinding
   for minutes at scale. The intra-cluster reinforcement must be written in a
   small constant number of writes, not N-per-cluster.

The reinforcement signal must stay correct: the intra-cluster hebbian edges that
exist still get potentiated by ``HEAVY_LTP_DELTA``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from uuid import UUID, uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


def _record(text: str, i: int) -> MemoryRecord:
    """A record with a distinct, low-cosine vector so the pattern-separation
    near-duplicate gate keeps every insert (degenerate identical vectors would
    collapse the corpus and mask the cluster topology)."""
    now = datetime.now(timezone.utc)
    emb = [0.0] * EMBED_DIM
    emb[i % EMBED_DIM] = 1.0
    emb[(i * 31 + 7) % EMBED_DIM] = 0.5
    emb[(i * 53 + 13) % EMBED_DIM] = 0.25
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=emb,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def _hebbian_weight(store, a: UUID, b: UUID) -> float | None:
    from iai_mcp.store import EDGES_TABLE

    key = sorted([str(a), str(b)])
    df = store.db.open_table(EDGES_TABLE).to_pandas()
    if df.empty:
        return None
    mask = (
        (df["src"] == key[0])
        & (df["dst"] == key[1])
        & (df["edge_type"] == "hebbian")
    )
    if not mask.any():
        return None
    return float(df.loc[mask, "weight"].iloc[0])


def _persisted_ids(store) -> list[UUID]:
    lock = getattr(store.db, "_conn_lock", None)
    if lock is not None:
        with lock:
            rows = store.db._conn.execute(
                "SELECT id FROM records ORDER BY rowid"
            ).fetchall()
    else:
        rows = store.db._conn.execute(
            "SELECT id FROM records ORDER BY rowid"
        ).fetchall()
    return [UUID(str(r[0])) for r in rows]


def _seed(store, n: int) -> list[UUID]:
    from iai_mcp.store._buffers import flush_record_buffer

    for i in range(n):
        store.insert(_record(f"r{i}", i))
    flush_record_buffer(store)
    return _persisted_ids(store)


def _add_hebbian(store, pairs) -> None:
    from iai_mcp.store._buffers import flush_edge_buffer

    for s in range(0, len(pairs), 2000):
        store.boost_edges(pairs[s:s + 2000], delta=0.1, edge_type="hebbian")
    flush_edge_buffer(store)


def _edge_type_of(args, kwargs) -> str | None:
    """Resolve the edge_type a boost_edges call used (keyword or positional)."""
    edge_type = kwargs.get("edge_type")
    if edge_type is None and len(args) >= 2:
        edge_type = args[1]
    return edge_type


class _BoostSpy:
    """Records boost_edges calls, partitioned by edge_type, with batch sizes.

    The cluster-summary path issues two kinds of boost: the O(k^2) intra-cluster
    ``hebbian`` potentiation (the one that explodes and must be capped) and the
    O(k) ``consolidated_from`` summary links (one per cluster member, linear and
    bounded by construction). The cap guard targets the hebbian batch.
    """

    def __init__(self, store) -> None:
        self._store = store
        self._real = store.boost_edges
        self.calls = 0
        self.max_pairs_in_a_call = 0
        self.max_hebbian_pairs_in_a_call = 0

    def __call__(self, pairs, *args, **kwargs):
        self.calls += 1
        self.max_pairs_in_a_call = max(self.max_pairs_in_a_call, len(pairs))
        if _edge_type_of(args, kwargs) == "hebbian":
            self.max_hebbian_pairs_in_a_call = max(
                self.max_hebbian_pairs_in_a_call, len(pairs)
            )
        return self._real(pairs, *args, **kwargs)


def test_giant_cluster_pairing_is_bounded(tmp_path):
    """A single giant connected component must NOT expand into O(k^2) pairs.

    Without a cap a 200-node component is combinations(200, 2) = 19_900 pairs in
    one boost; the cap holds the per-cluster contribution to MAX_PAIRS_PER_CLUSTER.
    """
    from iai_mcp.lilli.cycle.sleep_pipeline import MAX_PAIRS_PER_CLUSTER
    from iai_mcp.sleep import _process_cluster_summaries
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, 200)
    # One giant component: a similarity chain r0 ~ r1 ~ ... ~ rN.
    chain = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    _add_hebbian(store, chain)

    # Sanity: this really is ONE big component, not many small ones.
    from iai_mcp.sleep import _build_hebbian_clusters

    clusters = _build_hebbian_clusters(store)
    assert len(clusters) == 1, f"expected one giant component, got {len(clusters)}"
    assert len(clusters[0]) == len(ids)
    uncapped = len(list(combinations(clusters[0], 2)))
    assert uncapped > 10 * MAX_PAIRS_PER_CLUSTER, (
        "fixture must make the uncapped pairing far exceed the cap"
    )

    spy = _BoostSpy(store)
    store.boost_edges = spy  # type: ignore[method-assign]
    _process_cluster_summaries(store)

    # The intra-cluster hebbian potentiation must be capped well below the
    # uncapped k(k-1)/2 explosion. (The O(k) consolidated_from summary links are
    # linear by construction and are not the quadratic pathology.)
    assert spy.max_hebbian_pairs_in_a_call > 0, (
        "no hebbian potentiation issued — the test exercised nothing"
    )
    assert spy.max_hebbian_pairs_in_a_call <= MAX_PAIRS_PER_CLUSTER, (
        f"largest hebbian boost_edges batch {spy.max_hebbian_pairs_in_a_call} "
        f"exceeded the cap {MAX_PAIRS_PER_CLUSTER}; the giant-cluster pairing "
        f"was not bounded (uncapped would be {uncapped} pairs)"
    )


def test_edge_boosts_are_batched_not_per_cluster(tmp_path):
    """Intra-cluster hebbian potentiation must be ONE boost_edges call.

    With M realistic clusters the old code issued M hebbian boosts; the batched
    version issues exactly one regardless of cluster count.
    """
    from iai_mcp.sleep import _process_cluster_summaries
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, 120)
    # Twelve bounded clusters of 10, intra-cluster ring edges only.
    cs = 10
    ring = []
    for start in range(0, len(ids), cs):
        cl = ids[start:start + cs]
        for i in range(len(cl)):
            ring.append((cl[i], cl[(i + 1) % len(cl)]))
    _add_hebbian(store, ring)

    from iai_mcp.sleep import _build_hebbian_clusters

    n_clusters = len(_build_hebbian_clusters(store))
    assert n_clusters >= 8, f"fixture must yield many clusters, got {n_clusters}"

    # Spy only on hebbian boosts (the cluster-summary potentiation); the
    # consolidated_from boosts from _create_semantic_summary use a different
    # edge_type and are one-per-summary by design.
    real = store.boost_edges
    hebbian_calls = {"n": 0}

    def _spy(pairs, *args, **kwargs):
        # boost_edges signature is (pairs, delta=..., edge_type=...); all callers
        # in the cluster-summary path pass edge_type as a keyword.
        edge_type = kwargs.get("edge_type")
        if edge_type is None and len(args) >= 2:
            edge_type = args[1]
        if edge_type == "hebbian":
            hebbian_calls["n"] += 1
        return real(pairs, *args, **kwargs)

    store.boost_edges = _spy  # type: ignore[method-assign]
    _process_cluster_summaries(store)

    assert hebbian_calls["n"] <= 1, (
        f"expected the intra-cluster hebbian potentiation to be a single batched "
        f"boost_edges call, got {hebbian_calls['n']} (one-per-cluster regression "
        f"with {n_clusters} clusters)"
    )


def test_reinforcement_signal_is_correct(tmp_path):
    """The right intra-cluster edges still get potentiated by HEAVY_LTP_DELTA,
    and a non-cluster edge is left untouched."""
    from iai_mcp.sleep import HEAVY_LTP_DELTA, _process_cluster_summaries
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, 8)

    # A 4-node fully-connected cluster (6 intra-cluster pairs, under the cap).
    cluster = ids[:4]
    cluster_pairs = list(combinations(cluster, 2))
    for a, b in cluster_pairs:
        store.boost_edges([(a, b)], delta=0.3, edge_type="hebbian")

    # A lone non-cluster edge between two records that are NOT in any
    # >= CLUSTER_MIN_SIZE component on their own.
    lone_a, lone_b = ids[6], ids[7]
    store.boost_edges([(lone_a, lone_b)], delta=0.4, edge_type="hebbian")

    from iai_mcp.store._buffers import flush_edge_buffer

    flush_edge_buffer(store)

    before = {p: _hebbian_weight(store, *p) for p in cluster_pairs}
    lone_before = _hebbian_weight(store, lone_a, lone_b)
    assert all(w == pytest.approx(0.3, abs=1e-3) for w in before.values())
    assert lone_before == pytest.approx(0.4, abs=1e-3)

    _process_cluster_summaries(store)
    flush_edge_buffer(store)

    for a, b in cluster_pairs:
        w = _hebbian_weight(store, a, b)
        assert w is not None, f"cluster edge {a}/{b} vanished"
        assert w == pytest.approx(0.3 + HEAVY_LTP_DELTA, abs=1e-3), (
            f"cluster edge {a}/{b} not potentiated by HEAVY_LTP_DELTA: got {w}"
        )

    # The lone non-cluster edge is its own 2-node component (< CLUSTER_MIN_SIZE),
    # so it must be left exactly as it was.
    lone_after = _hebbian_weight(store, lone_a, lone_b)
    assert lone_after == pytest.approx(0.4, abs=1e-3), (
        f"non-cluster edge changed: {lone_before} -> {lone_after}"
    )
