from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from iai_mcp import hippea_cascade
from iai_mcp.community import CommunityAssignment
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password",
        lambda s, u, p: fake.__setitem__((s, u), p),
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None),
    )
    yield fake


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "db")


def _rec(*, literal: str, centrality: float, store: MemoryStore) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    r = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=literal,
        aaak_index="",
        embedding=[0.0] * store.embed_dim,
        community_id=None,
        centrality=centrality,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )
    store.insert(r)
    return r


def test_centrality_for_ids_no_decrypt(store: MemoryStore) -> None:
    r_high = _rec(literal="high", centrality=0.8, store=store)
    r_low = _rec(literal="low", centrality=0.3, store=store)
    r_zero = _rec(literal="zero", centrality=0.0, store=store)
    absent_id = uuid4()

    ids = [r_high.id, r_low.id, r_zero.id, absent_id]

    with patch.object(store, "get", side_effect=AssertionError("store.get must not be called")):
        result = store.centrality_for_ids(ids)

    assert isinstance(result, dict)
    assert result[r_high.id] == pytest.approx(0.8)
    assert result[r_low.id] == pytest.approx(0.3)
    assert result[r_zero.id] == pytest.approx(0.0)
    assert absent_id not in result
    assert set(result.keys()) == {r_high.id, r_low.id, r_zero.id}


def test_centrality_for_ids_empty_input(store: MemoryStore) -> None:
    with patch.object(store, "get", side_effect=AssertionError("store.get must not be called")):
        result = store.centrality_for_ids([])
    assert result == {}


def test_centrality_for_ids_all_absent(store: MemoryStore) -> None:
    ids = [uuid4(), uuid4()]
    with patch.object(store, "get", side_effect=AssertionError("store.get must not be called")):
        result = store.centrality_for_ids(ids)
    assert result == {}


def _legacy_top_n(
    store_centrality_map: dict[UUID, float],
    member_ids: list[UUID],
    n: int,
) -> list[UUID]:
    scored: list[tuple[float, UUID]] = []
    for rid in member_ids:
        if rid not in store_centrality_map:
            continue
        try:
            centrality = float(store_centrality_map[rid] or 0.0)
        except (TypeError, ValueError):
            centrality = 0.0
        scored.append((centrality, rid))
    scored.sort(key=lambda kv: (-kv[0], str(kv[1])))
    return [rid for _c, rid in scored[:n]]


def test_top_n_matches_legacy(store: MemoryStore) -> None:
    cid = uuid4()
    r_a = _rec(literal="a", centrality=0.9, store=store)
    r_b = _rec(literal="b", centrality=0.5, store=store)
    r_c = _rec(literal="c", centrality=0.5, store=store)
    r_d = _rec(literal="d", centrality=0.1, store=store)
    absent_id = uuid4()

    member_ids = [r_a.id, r_b.id, r_c.id, r_d.id, absent_id]

    assignment = CommunityAssignment(
        top_communities=[cid],
        community_centroids={cid: [0.0] * store.embed_dim},
        node_to_community={r_a.id: cid, r_b.id: cid, r_c.id: cid, r_d.id: cid},
        mid_regions={cid: member_ids},
    )

    ref_map = {
        r_a.id: 0.9,
        r_b.id: 0.5,
        r_c.id: 0.5,
        r_d.id: 0.1,
    }
    expected = _legacy_top_n(ref_map, member_ids, n=3)
    assert len(expected) == 3

    get_calls: list[UUID] = []

    def _spy_get(rid: UUID):
        get_calls.append(rid)
        raise AssertionError(f"store.get called for {rid} — must not be called")

    with patch.object(store, "get", side_effect=_spy_get):
        result = hippea_cascade._top_n_records_by_centrality(
            store, assignment, cid, n=3
        )

    assert get_calls == [], f"store.get was called for: {get_calls}"
    assert result == expected, f"result {result} != expected {expected}"


def test_top_n_empty_community(store: MemoryStore) -> None:
    cid = uuid4()
    assignment = CommunityAssignment(
        top_communities=[cid],
        community_centroids={cid: [0.0] * store.embed_dim},
        mid_regions={cid: []},
    )
    with patch.object(store, "get", side_effect=AssertionError("store.get must not be called")):
        result = hippea_cascade._top_n_records_by_centrality(store, assignment, cid, n=5)
    assert result == []


def test_top_n_all_zero_centrality_stable_sort(store: MemoryStore) -> None:
    cid = uuid4()
    recs = [_rec(literal=f"r{i}", centrality=0.0, store=store) for i in range(5)]
    member_ids = [r.id for r in recs]

    assignment = CommunityAssignment(
        top_communities=[cid],
        community_centroids={cid: [0.0] * store.embed_dim},
        node_to_community={r.id: cid for r in recs},
        mid_regions={cid: member_ids},
    )

    ref_map = {r.id: 0.0 for r in recs}
    expected = _legacy_top_n(ref_map, member_ids, n=3)

    with patch.object(store, "get", side_effect=AssertionError("store.get must not be called")):
        result = hippea_cascade._top_n_records_by_centrality(
            store, assignment, cid, n=3
        )

    assert result == expected


def test_top_n_n_larger_than_members(store: MemoryStore) -> None:
    cid = uuid4()
    recs = [_rec(literal=f"r{i}", centrality=float(i) / 10, store=store) for i in range(3)]
    member_ids = [r.id for r in recs]

    assignment = CommunityAssignment(
        top_communities=[cid],
        community_centroids={cid: [0.0] * store.embed_dim},
        node_to_community={r.id: cid for r in recs},
        mid_regions={cid: member_ids},
    )

    ref_map = {r.id: float(i) / 10 for i, r in enumerate(recs)}
    expected = _legacy_top_n(ref_map, member_ids, n=100)

    with patch.object(store, "get", side_effect=AssertionError("store.get must not be called")):
        result = hippea_cascade._top_n_records_by_centrality(
            store, assignment, cid, n=100
        )

    assert result == expected
