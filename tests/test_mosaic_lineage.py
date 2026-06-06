"""RED-witness suite for the lineage
contract + prior-aware initialisation.

It exercises:

  - `LineageTracker._birth_ts` bookkeeping.
  - `LineageTracker.register_prior_birth` -- bootstraps a known-prior UUID's
    birth timestamp WITHOUT emitting an event.
  - `LineageTracker.pick_merge_survivor` -- survival policy:
    oldest by birth timestamp; tie-break = `min(uuid, key=str)`; unknown UUID
    loses via `datetime.max` sentinel.
  - `LineageTracker.known_uuids` -- inspection accessor for tests.
  - Event-shape invariants for birth / split / merge / death.
  - `LineageReport` frozen-snapshot semantics (mutating the tracker after
    `report()` does not leak into the returned tuple).
  - `init_partitions(graph, prior, prior_mode)`:
      * `prior_mode="cold"` (or `prior is None`): each node is its own
        singleton with a fresh `uuid4()` community.
      * `prior_mode="seeded"`: filter stale prior UUIDs, reuse prior community
        UUIDs for surviving members, new nodes get fresh singletons with
        a recorded `birth` event.
      * Canonical UUID ordering.
      * `partition.dtype == np.int64`.
      * Invalid `prior_mode` raises `ValueError`.
      * `Literal["seeded", "cold"]` annotation visible on the signature.
"""
from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from typing import get_args
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph


def _emb(seed: int, dim: int = 384) -> list[float]:
    """Deterministic embedding for test nodes (init does not look at
    embeddings, but `MemoryGraph.add_node` requires the argument)."""
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


def _build_graph(n_nodes: int) -> tuple[MemoryGraph, list[UUID]]:
    """Build a graph with n nodes and no edges. The init-partitions tests do
    not care about topology -- partition assignment is structural."""
    g = MemoryGraph()
    uuids = [uuid4() for _ in range(n_nodes)]
    for i, u in enumerate(uuids):
        g.add_node(u, community_id=None, embedding=_emb(i))
    return g, uuids


# ============================================================================
# LineageTracker._birth_ts + register_prior_birth + pick_merge_survivor + known_uuids
# ============================================================================


def test_register_prior_birth_seeds_timestamp_without_event() -> None:
    """Bootstrap a known-prior UUID's birth_ts
    WITHOUT emitting a LineageEvent. The recorder is event-driven; only true
    births during the run emit `birth` events."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u = uuid4()
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    tracker.register_prior_birth(u, ts)

    # Birth timestamp bookkept...
    assert tracker._birth_ts[u] == ts
    #...but NO event emitted.
    assert tracker.report().events == ()


def test_register_prior_birth_setdefault_semantics() -> None:
    """Second call with the same UUID must NOT overwrite the original ts."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u = uuid4()
    first = datetime(2024, 1, 1, tzinfo=timezone.utc)
    second = datetime(2026, 1, 1, tzinfo=timezone.utc)

    tracker.register_prior_birth(u, first)
    tracker.register_prior_birth(u, second)

    assert tracker._birth_ts[u] == first


def test_pick_merge_survivor_oldest_wins() -> None:
    """Oldest birth_ts survives a merge."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u_old = uuid4()
    u_new = uuid4()
    tracker.register_prior_birth(u_old, datetime(2024, 1, 1, tzinfo=timezone.utc))
    tracker.register_prior_birth(u_new, datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert tracker.pick_merge_survivor([u_old, u_new]) == u_old
    assert tracker.pick_merge_survivor([u_new, u_old]) == u_old


def test_pick_merge_survivor_tie_break_lex() -> None:
    """When both candidates
    share the same birth timestamp (the first-migration degenerate case
    where every surviving prior UUID has the same `now - 1µs` ts), tie-break
    is `min(uuid, key=str)`."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u1 = UUID("00000000-0000-0000-0000-000000000001")
    u2 = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    same_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tracker.register_prior_birth(u1, same_ts)
    tracker.register_prior_birth(u2, same_ts)

    survivor = tracker.pick_merge_survivor([u1, u2])
    assert survivor == min([u1, u2], key=str)
    assert survivor == u1  # 0x00...01 sorts before ffff...ffff


def test_pick_merge_survivor_unknown_uuid_loses() -> None:
    """A UUID with no registered birth_ts loses
    every tie (sentinel = `datetime.max` so the registered UUID wins)."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u_known = uuid4()
    u_unknown = uuid4()
    tracker.register_prior_birth(
        u_known, datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    # u_unknown has no birth_ts entry.

    assert tracker.pick_merge_survivor([u_known, u_unknown]) == u_known
    assert tracker.pick_merge_survivor([u_unknown, u_known]) == u_known


def test_pick_merge_survivor_all_unknown_lex_only() -> None:
    """When NO candidate has a birth_ts, every candidate gets the same
    `datetime.max` sentinel and the result is pure lex-by-str. Warning #3
    edge case: first migration where prior had zero birth_ts records."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u_a = UUID("00000000-0000-0000-0000-00000000000a")
    u_b = UUID("00000000-0000-0000-0000-00000000000b")

    survivor = tracker.pick_merge_survivor([u_b, u_a])
    assert survivor == min([u_a, u_b], key=str)
    assert survivor == u_a


def test_known_uuids_returns_birth_ts_keys() -> None:
    """Inspection accessor for tests."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u1 = uuid4()
    u2 = uuid4()
    tracker.register_prior_birth(u1, datetime(2026, 1, 1, tzinfo=timezone.utc))
    tracker.record_birth(u2, member_count=1)

    known = tracker.known_uuids()
    assert isinstance(known, set)
    assert known == {u1, u2}


# ============================================================================
# Event-shape invariants
# ============================================================================


def test_record_birth_event_shape() -> None:
    """Birth event records the new UUID as the
    single `child_uuids` entry with `parent_uuid=None`."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    new = uuid4()
    before = datetime.now(timezone.utc) - timedelta(seconds=1)
    tracker.record_birth(new, member_count=5)
    after = datetime.now(timezone.utc) + timedelta(seconds=1)

    events = tracker.report().events
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "birth"
    assert ev.parent_uuid is None
    assert ev.child_uuids == (new,)
    assert ev.member_count == 5
    assert before <= ev.timestamp <= after
    # `_birth_ts` is also stamped (so a later split/merge survivor pick works).
    assert new in tracker._birth_ts


def test_record_split_event_shape() -> None:
    """Split event records the parent UUID and
    the full tuple of child UUIDs; children get fresh birth timestamps."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    parent = uuid4()
    c1, c2 = uuid4(), uuid4()
    # Register parent first so the recorder can see it.
    tracker.register_prior_birth(
        parent, datetime(2024, 1, 1, tzinfo=timezone.utc)
    )

    tracker.record_split(parent, [c1, c2], member_count=7)

    events = tracker.report().events
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "split"
    assert ev.parent_uuid == parent
    assert ev.child_uuids == (c1, c2)
    assert ev.member_count == 7
    # Both children get birth timestamps; parent ts untouched.
    assert c1 in tracker._birth_ts
    assert c2 in tracker._birth_ts
    assert tracker._birth_ts[parent] == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_record_merge_event_shape() -> None:
    """Merge event puts the surviving UUID as
    `parent_uuid`; the retired UUIDs go into `child_uuids` (legible direction:
    survivor = parent, retired = children)."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u1, u2 = uuid4(), uuid4()
    tracker.register_prior_birth(u1, datetime(2024, 1, 1, tzinfo=timezone.utc))
    tracker.register_prior_birth(u2, datetime(2026, 1, 1, tzinfo=timezone.utc))

    surviving = tracker.pick_merge_survivor([u1, u2])
    tracker.record_merge([u1, u2], surviving, member_count=10)

    events = tracker.report().events
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "merge"
    assert ev.parent_uuid == surviving
    # The non-survivor goes into `child_uuids` (singleton tuple).
    others = tuple(p for p in [u1, u2] if p != surviving)
    assert ev.child_uuids == others
    assert ev.member_count == 10


def test_record_death_event_shape() -> None:
    """Death event retires a UUID; `child_uuids`
    is empty (the retired UUID itself sits in `parent_uuid`)."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u = uuid4()

    tracker.record_death(u, member_count=0)

    events = tracker.report().events
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "death"
    assert ev.parent_uuid == u
    assert ev.child_uuids == ()
    assert ev.member_count == 0


def test_lineage_report_returns_frozen_snapshot() -> None:
    """`report()` returns an immutable tuple. Mutating
    the tracker after taking a report does NOT leak into the snapshot."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u1 = uuid4()
    tracker.record_birth(u1, member_count=3)
    snapshot = tracker.report()
    assert len(snapshot.events) == 1

    # Now mutate the tracker after the snapshot.
    u2 = uuid4()
    tracker.record_birth(u2, member_count=2)

    # Snapshot is unchanged.
    assert len(snapshot.events) == 1
    assert snapshot.events[0].child_uuids == (u1,)
    # The live tracker reports the new state.
    assert len(tracker.report().events) == 2


def test_lineage_report_preserves_event_order() -> None:
    """Events appear in the exact order they
    were recorded; consumers may walk by index."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u_a = uuid4()
    u_b = uuid4()
    u_c = uuid4()
    u_d = uuid4()

    tracker.record_birth(u_a, member_count=1)
    tracker.record_split(u_a, [u_b, u_c], member_count=2)
    tracker.record_merge([u_b, u_c], u_b, member_count=2)
    tracker.record_death(u_d, member_count=0)

    events = tracker.report().events
    assert [e.event_type for e in events] == ["birth", "split", "merge", "death"]


def test_birth_timestamp_set_only_once_via_setdefault() -> None:
    """`_birth_ts.setdefault(uuid, ts)` semantics: the FIRST
    `record_birth` for a UUID stamps the birth_ts. A subsequent
    `record_birth(u,...)` (e.g. defensive re-emission by a buggy caller)
    must NOT overwrite. Two events are still recorded; only one ts kept."""
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u = uuid4()

    tracker.record_birth(u, member_count=1)
    first_ts = tracker._birth_ts[u]

    # Force-call again -- defensively, the recorder should NOT overwrite.
    tracker.record_birth(u, member_count=1)
    second_ts = tracker._birth_ts[u]

    assert first_ts == second_ts
    # Both events ARE recorded -- bookkeeping is per-UUID, not per-event.
    assert len(tracker.report().events) == 2


# ============================================================================
# init_partitions
# ============================================================================


def test_init_partitions_returns_three_tuple() -> None:
    """Entrypoint returns
    `(partition: np.ndarray, int_to_uuid: dict[int, UUID], lineage: LineageTracker)`."""
    from iai_mcp.mosaic_lineage import LineageTracker, init_partitions

    g, _ = _build_graph(5)

    partition, int_to_uuid, lineage = init_partitions(g, None, "seeded")
    assert isinstance(partition, np.ndarray)
    assert isinstance(int_to_uuid, dict)
    assert isinstance(lineage, LineageTracker)


def test_init_partitions_dtype_int64() -> None:
    """Partition dtype is strict int64."""
    from iai_mcp.mosaic_lineage import init_partitions

    g, _ = _build_graph(3)
    partition, _, _ = init_partitions(g, None, "cold")
    assert partition.dtype == np.int64


def test_init_partitions_empty_graph_returns_empty() -> None:
    """0-node graph short-circuits: empty partition, empty map, empty lineage."""
    from iai_mcp.mosaic_lineage import init_partitions

    g = MemoryGraph()
    partition, int_to_uuid, lineage = init_partitions(g, None, "seeded")
    assert partition.shape == (0,)
    assert partition.dtype == np.int64
    assert int_to_uuid == {}
    assert lineage.report().events == ()


def test_init_partitions_cold_with_none_prior() -> None:
    """`prior is None` collapses to all-singletons in
    BOTH modes (the `seeded` branch falls through to cold when prior is None)."""
    from iai_mcp.mosaic_lineage import init_partitions

    g, _ = _build_graph(10)
    for mode in ("seeded", "cold"):
        partition, int_to_uuid, lineage = init_partitions(g, None, mode)
        # All singletons -- 10 distinct ints, 10 distinct UUIDs.
        assert sorted(partition.tolist()) == list(range(10))
        assert len(int_to_uuid) == 10
        assert len(set(int_to_uuid.values())) == 10


def test_init_partitions_cold_explicit_discards_prior() -> None:
    """Invariant:
    `prior_mode='cold'` REGENERATES every UUID even when a prior is supplied
    (crisis_recluster path; the prior is intentionally discarded)."""
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(10)
    u_old = uuid4()
    prior = CommunityAssignment(
        node_to_community={leaf: u_old for leaf in uuids},
        community_centroids={u_old: [0.0] * 384},
    )

    partition, int_to_uuid, _ = init_partitions(g, prior, "cold")
    # All singletons.
    assert sorted(partition.tolist()) == list(range(10))
    # Prior UUID does NOT leak through.
    assert u_old not in int_to_uuid.values()
    # 10 fresh UUIDs.
    assert len(set(int_to_uuid.values())) == 10


def test_init_partitions_seeded_preserves_prior_uuids() -> None:
    """Seeded mode reuses prior community UUIDs
    when their nodes are still in the graph."""
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(5)
    u_a = uuid4()
    u_b = uuid4()
    # nodes [0,1,2] -> U_a; nodes [3,4] -> U_b
    prior = CommunityAssignment(
        node_to_community={
            uuids[0]: u_a, uuids[1]: u_a, uuids[2]: u_a,
            uuids[3]: u_b, uuids[4]: u_b,
        },
        community_centroids={u_a: [0.1] * 384, u_b: [0.2] * 384},
    )

    partition, int_to_uuid, _ = init_partitions(g, prior, "seeded")
    # Exactly 2 distinct community UUIDs; both from the prior.
    assert set(int_to_uuid.values()) == {u_a, u_b}
    assert len(set(partition.tolist())) == 2


def test_init_partitions_seeded_stale_uuids_dropped() -> None:
    """Stale prior leaf UUIDs (UUIDs no
    longer in `graph._nx.nodes()`) are dropped silently. Their community
    UUIDs do NOT appear in the resulting int_to_uuid."""
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(3)
    n_stale = uuid4()  # NOT in the graph
    u_stale = uuid4()
    u_active = uuid4()
    prior = CommunityAssignment(
        node_to_community={
            n_stale: u_stale,  # stale entry
            uuids[0]: u_active,
            uuids[1]: u_active,
            uuids[2]: u_active,
        },
        community_centroids={u_stale: [0.0] * 384, u_active: [0.5] * 384},
    )

    # MUST NOT raise.
    partition, int_to_uuid, _ = init_partitions(g, prior, "seeded")
    # u_stale does NOT leak into the result.
    assert u_stale not in int_to_uuid.values()
    # u_active survives.
    assert u_active in int_to_uuid.values()


def test_init_partitions_seeded_new_nodes_get_fresh_uuids_and_birth_event() -> None:
    """Nodes in the graph but NOT in
    `prior.node_to_community` get fresh `uuid4()` community UUIDs AND a
    `birth` event recorded in the lineage tracker."""
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(5)
    u_prior = uuid4()
    # Only the first 3 nodes have a prior entry.
    prior = CommunityAssignment(
        node_to_community={
            uuids[0]: u_prior, uuids[1]: u_prior, uuids[2]: u_prior,
        },
        community_centroids={u_prior: [0.3] * 384},
    )

    _partition, int_to_uuid, lineage = init_partitions(g, prior, "seeded")
    # u_prior survives.
    assert u_prior in int_to_uuid.values()
    # 2 new UUIDs (one per new node).
    new_uuids = {u for u in int_to_uuid.values() if u != u_prior}
    assert len(new_uuids) == 2

    # The lineage tracker has exactly 2 birth events -- one per new node.
    birth_events = [
        e for e in lineage.report().events if e.event_type == "birth"
    ]
    assert len(birth_events) == 2
    # The prior UUID is NOT in birth_events (it was seeded via
    # register_prior_birth, not born this run).
    for ev in birth_events:
        assert ev.parent_uuid is None
        assert ev.child_uuids[0] != u_prior


def test_init_partitions_canonical_ordering() -> None:
    """The partition array is indexed by
    `sorted(uuids, key=str)`. Same set of UUIDs inserted in different orders
    must produce the same partition."""
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(6)
    sorted_uuids = sorted(uuids, key=str)

    partition, int_to_uuid, _ = init_partitions(g, None, "cold")
    # All singletons -- partition[i] is the int label for sorted_uuids[i].
    # Verify each sorted UUID maps to a distinct int label in 0..5.
    int_labels = [int(partition[i]) for i in range(len(sorted_uuids))]
    assert sorted(int_labels) == list(range(6))
    # The int->UUID map covers all 6 ints.
    assert set(int_to_uuid.keys()) == set(range(6))


def test_init_partitions_invalid_mode_raises_value_error() -> None:
    """Unknown `prior_mode` is a programming error,
    not a silent fallback. Raise `ValueError` with a clear message."""
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(3)
    prior = CommunityAssignment(
        node_to_community={uuids[0]: uuid4()},
        community_centroids={},
    )

    with pytest.raises(ValueError, match=r"prior_mode"):
        init_partitions(g, prior, "warm")  # type: ignore[arg-type]


def test_init_partitions_seeded_register_prior_birth_called_per_unique_uuid() -> None:
    """Every surviving prior community UUID
    appears in `tracker._birth_ts` after init (so a later merge can score
    them by age). New-node UUIDs also appear (via `record_birth`)."""
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(5)
    u_a = uuid4()
    u_b = uuid4()
    prior = CommunityAssignment(
        node_to_community={
            uuids[0]: u_a, uuids[1]: u_a,
            uuids[2]: u_b,
        },
        community_centroids={u_a: [0.1] * 384, u_b: [0.2] * 384},
    )

    _, int_to_uuid, lineage = init_partitions(g, prior, "seeded")
    # All UUIDs in the result must be in `_birth_ts` (so survivor picks work).
    for u in int_to_uuid.values():
        assert u in lineage.known_uuids()
    # Prior UUIDs survived.
    assert u_a in lineage.known_uuids()
    assert u_b in lineage.known_uuids()


def test_init_partitions_typed_prior_mode_literal() -> None:
    """`prior_mode` annotation visible on the
    signature as `Literal["seeded", "cold"]` (caller-side IDE / mypy check).

    `from __future__ import annotations` strings the annotation; check the
    string form contains both literal members.
    """
    from iai_mcp.mosaic_lineage import init_partitions

    sig = inspect.signature(init_partitions)
    ann = sig.parameters["prior_mode"].annotation
    # Under PEP 563 the annotation is a string; under runtime evaluation
    # it's a typing.Literal generic.
    if isinstance(ann, str):
        assert "seeded" in ann
        assert "cold" in ann
        assert "Literal" in ann
    else:
        args = get_args(ann)
        assert "seeded" in args
        assert "cold" in args


def test_record_methods_deterministic_modulo_timestamps() -> None:
    """Same inputs -> same event shape sequence
    (only timestamps may differ across calls)."""
    from iai_mcp.mosaic_lineage import LineageTracker

    def replay() -> list[tuple]:
        t = LineageTracker()
        u_a = UUID("00000000-0000-0000-0000-00000000000a")
        u_b = UUID("00000000-0000-0000-0000-00000000000b")
        u_c = UUID("00000000-0000-0000-0000-00000000000c")
        t.record_birth(u_a, member_count=1)
        t.record_split(u_a, [u_b, u_c], member_count=2)
        t.record_merge([u_b, u_c], u_b, member_count=2)
        t.record_death(u_c, member_count=0)
        return [
            (e.event_type, e.parent_uuid, e.child_uuids, e.member_count)
            for e in t.report().events
        ]

    a = replay()
    b = replay()
    assert a == b


def test_lineage_replay_determinism_seeded_init() -> None:
    """Same `(graph, prior)` -> same int_to_uuid mapping
    keys ordering (canonical sort by str(uuid)). NEW-NODE uuids vary because
    they're fresh `uuid4()` per call; the prior-survival uuids and the int
    labels are stable."""
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(5)
    u_a = uuid4()
    prior = CommunityAssignment(
        node_to_community={uuids[0]: u_a, uuids[1]: u_a, uuids[2]: u_a},
        community_centroids={u_a: [0.0] * 384},
    )

    p1, m1, _ = init_partitions(g, prior, "seeded")
    p2, m2, _ = init_partitions(g, prior, "seeded")

    # The prior UUID lands at the SAME int label both times (canonical order).
    int_for_ua_1 = [k for k, v in m1.items() if v == u_a][0]
    int_for_ua_2 = [k for k, v in m2.items() if v == u_a][0]
    assert int_for_ua_1 == int_for_ua_2
    # Partition entries for nodes in U_a match across the two runs.
    sorted_uuids = sorted(uuids, key=str)
    for i, u in enumerate(sorted_uuids):
        if u in {uuids[0], uuids[1], uuids[2]}:
            assert p1[i] == p2[i] == int_for_ua_1
