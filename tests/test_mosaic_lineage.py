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
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


def _build_graph(n_nodes: int) -> tuple[MemoryGraph, list[UUID]]:
    g = MemoryGraph()
    uuids = [uuid4() for _ in range(n_nodes)]
    for i, u in enumerate(uuids):
        g.add_node(u, community_id=None, embedding=_emb(i))
    return g, uuids


def test_register_prior_birth_seeds_timestamp_without_event() -> None:
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u = uuid4()
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    tracker.register_prior_birth(u, ts)

    assert tracker._birth_ts[u] == ts
    assert tracker.report().events == ()


def test_register_prior_birth_setdefault_semantics() -> None:
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u = uuid4()
    first = datetime(2024, 1, 1, tzinfo=timezone.utc)
    second = datetime(2026, 1, 1, tzinfo=timezone.utc)

    tracker.register_prior_birth(u, first)
    tracker.register_prior_birth(u, second)

    assert tracker._birth_ts[u] == first


def test_pick_merge_survivor_oldest_wins() -> None:
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u_old = uuid4()
    u_new = uuid4()
    tracker.register_prior_birth(u_old, datetime(2024, 1, 1, tzinfo=timezone.utc))
    tracker.register_prior_birth(u_new, datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert tracker.pick_merge_survivor([u_old, u_new]) == u_old
    assert tracker.pick_merge_survivor([u_new, u_old]) == u_old


def test_pick_merge_survivor_tie_break_lex() -> None:
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u1 = UUID("00000000-0000-0000-0000-000000000001")
    u2 = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    same_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tracker.register_prior_birth(u1, same_ts)
    tracker.register_prior_birth(u2, same_ts)

    survivor = tracker.pick_merge_survivor([u1, u2])
    assert survivor == min([u1, u2], key=str)
    assert survivor == u1


def test_pick_merge_survivor_unknown_uuid_loses() -> None:
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u_known = uuid4()
    u_unknown = uuid4()
    tracker.register_prior_birth(
        u_known, datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    assert tracker.pick_merge_survivor([u_known, u_unknown]) == u_known
    assert tracker.pick_merge_survivor([u_unknown, u_known]) == u_known


def test_pick_merge_survivor_all_unknown_lex_only() -> None:
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u_a = UUID("00000000-0000-0000-0000-00000000000a")
    u_b = UUID("00000000-0000-0000-0000-00000000000b")

    survivor = tracker.pick_merge_survivor([u_b, u_a])
    assert survivor == min([u_a, u_b], key=str)
    assert survivor == u_a


def test_known_uuids_returns_birth_ts_keys() -> None:
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u1 = uuid4()
    u2 = uuid4()
    tracker.register_prior_birth(u1, datetime(2026, 1, 1, tzinfo=timezone.utc))
    tracker.record_birth(u2, member_count=1)

    known = tracker.known_uuids()
    assert isinstance(known, set)
    assert known == {u1, u2}


def test_record_birth_event_shape() -> None:
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
    assert new in tracker._birth_ts


def test_record_split_event_shape() -> None:
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    parent = uuid4()
    c1, c2 = uuid4(), uuid4()
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
    assert c1 in tracker._birth_ts
    assert c2 in tracker._birth_ts
    assert tracker._birth_ts[parent] == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_record_merge_event_shape() -> None:
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
    others = tuple(p for p in [u1, u2] if p != surviving)
    assert ev.child_uuids == others
    assert ev.member_count == 10


def test_record_death_event_shape() -> None:
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
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u1 = uuid4()
    tracker.record_birth(u1, member_count=3)
    snapshot = tracker.report()
    assert len(snapshot.events) == 1

    u2 = uuid4()
    tracker.record_birth(u2, member_count=2)

    assert len(snapshot.events) == 1
    assert snapshot.events[0].child_uuids == (u1,)
    assert len(tracker.report().events) == 2


def test_lineage_report_preserves_event_order() -> None:
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
    from iai_mcp.mosaic_lineage import LineageTracker

    tracker = LineageTracker()
    u = uuid4()

    tracker.record_birth(u, member_count=1)
    first_ts = tracker._birth_ts[u]

    tracker.record_birth(u, member_count=1)
    second_ts = tracker._birth_ts[u]

    assert first_ts == second_ts
    assert len(tracker.report().events) == 2


def test_init_partitions_returns_three_tuple() -> None:
    from iai_mcp.mosaic_lineage import LineageTracker, init_partitions

    g, _ = _build_graph(5)

    partition, int_to_uuid, lineage = init_partitions(g, None, "seeded")
    assert isinstance(partition, np.ndarray)
    assert isinstance(int_to_uuid, dict)
    assert isinstance(lineage, LineageTracker)


def test_init_partitions_dtype_int64() -> None:
    from iai_mcp.mosaic_lineage import init_partitions

    g, _ = _build_graph(3)
    partition, _, _ = init_partitions(g, None, "cold")
    assert partition.dtype == np.int64


def test_init_partitions_empty_graph_returns_empty() -> None:
    from iai_mcp.mosaic_lineage import init_partitions

    g = MemoryGraph()
    partition, int_to_uuid, lineage = init_partitions(g, None, "seeded")
    assert partition.shape == (0,)
    assert partition.dtype == np.int64
    assert int_to_uuid == {}
    assert lineage.report().events == ()


def test_init_partitions_cold_with_none_prior() -> None:
    from iai_mcp.mosaic_lineage import init_partitions

    g, _ = _build_graph(10)
    for mode in ("seeded", "cold"):
        partition, int_to_uuid, lineage = init_partitions(g, None, mode)
        assert sorted(partition.tolist()) == list(range(10))
        assert len(int_to_uuid) == 10
        assert len(set(int_to_uuid.values())) == 10


def test_init_partitions_cold_explicit_discards_prior() -> None:
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(10)
    u_old = uuid4()
    prior = CommunityAssignment(
        node_to_community={leaf: u_old for leaf in uuids},
        community_centroids={u_old: [0.0] * 384},
    )

    partition, int_to_uuid, _ = init_partitions(g, prior, "cold")
    assert sorted(partition.tolist()) == list(range(10))
    assert u_old not in int_to_uuid.values()
    assert len(set(int_to_uuid.values())) == 10


def test_init_partitions_seeded_preserves_prior_uuids() -> None:
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(5)
    u_a = uuid4()
    u_b = uuid4()
    prior = CommunityAssignment(
        node_to_community={
            uuids[0]: u_a, uuids[1]: u_a, uuids[2]: u_a,
            uuids[3]: u_b, uuids[4]: u_b,
        },
        community_centroids={u_a: [0.1] * 384, u_b: [0.2] * 384},
    )

    partition, int_to_uuid, _ = init_partitions(g, prior, "seeded")
    assert set(int_to_uuid.values()) == {u_a, u_b}
    assert len(set(partition.tolist())) == 2


def test_init_partitions_seeded_stale_uuids_dropped() -> None:
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(3)
    n_stale = uuid4()
    u_stale = uuid4()
    u_active = uuid4()
    prior = CommunityAssignment(
        node_to_community={
            n_stale: u_stale,
            uuids[0]: u_active,
            uuids[1]: u_active,
            uuids[2]: u_active,
        },
        community_centroids={u_stale: [0.0] * 384, u_active: [0.5] * 384},
    )

    partition, int_to_uuid, _ = init_partitions(g, prior, "seeded")
    assert u_stale not in int_to_uuid.values()
    assert u_active in int_to_uuid.values()


def test_init_partitions_seeded_new_nodes_get_fresh_uuids_and_birth_event() -> None:
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(5)
    u_prior = uuid4()
    prior = CommunityAssignment(
        node_to_community={
            uuids[0]: u_prior, uuids[1]: u_prior, uuids[2]: u_prior,
        },
        community_centroids={u_prior: [0.3] * 384},
    )

    _partition, int_to_uuid, lineage = init_partitions(g, prior, "seeded")
    assert u_prior in int_to_uuid.values()
    new_uuids = {u for u in int_to_uuid.values() if u != u_prior}
    assert len(new_uuids) == 2

    birth_events = [
        e for e in lineage.report().events if e.event_type == "birth"
    ]
    assert len(birth_events) == 2
    for ev in birth_events:
        assert ev.parent_uuid is None
        assert ev.child_uuids[0] != u_prior


def test_init_partitions_canonical_ordering() -> None:
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(6)
    sorted_uuids = sorted(uuids, key=str)

    partition, int_to_uuid, _ = init_partitions(g, None, "cold")
    int_labels = [int(partition[i]) for i in range(len(sorted_uuids))]
    assert sorted(int_labels) == list(range(6))
    assert set(int_to_uuid.keys()) == set(range(6))


def test_init_partitions_invalid_mode_raises_value_error() -> None:
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(3)
    prior = CommunityAssignment(
        node_to_community={uuids[0]: uuid4()},
        community_centroids={},
    )

    with pytest.raises(ValueError, match=r"prior_mode"):
        init_partitions(g, prior, "warm")  # type: ignore[arg-type]


def test_init_partitions_seeded_register_prior_birth_called_per_unique_uuid() -> None:
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
    for u in int_to_uuid.values():
        assert u in lineage.known_uuids()
    assert u_a in lineage.known_uuids()
    assert u_b in lineage.known_uuids()


def test_init_partitions_typed_prior_mode_literal() -> None:
    from iai_mcp.mosaic_lineage import init_partitions

    sig = inspect.signature(init_partitions)
    ann = sig.parameters["prior_mode"].annotation
    if isinstance(ann, str):
        assert "seeded" in ann
        assert "cold" in ann
        assert "Literal" in ann
    else:
        args = get_args(ann)
        assert "seeded" in args
        assert "cold" in args


def test_record_methods_deterministic_modulo_timestamps() -> None:
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
    from iai_mcp.mosaic_lineage import init_partitions

    g, uuids = _build_graph(5)
    u_a = uuid4()
    prior = CommunityAssignment(
        node_to_community={uuids[0]: u_a, uuids[1]: u_a, uuids[2]: u_a},
        community_centroids={u_a: [0.0] * 384},
    )

    p1, m1, _ = init_partitions(g, prior, "seeded")
    p2, m2, _ = init_partitions(g, prior, "seeded")

    int_for_ua_1 = [k for k, v in m1.items() if v == u_a][0]
    int_for_ua_2 = [k for k, v in m2.items() if v == u_a][0]
    assert int_for_ua_1 == int_for_ua_2
    sorted_uuids = sorted(uuids, key=str)
    for i, u in enumerate(sorted_uuids):
        if u in {uuids[0], uuids[1], uuids[2]}:
            assert p1[i] == p2[i] == int_for_ua_1
