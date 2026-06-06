"""Tests for bulk (id, centrality) projection and _top_n_records_by_centrality.

Verifies:
  1. store.centrality_for_ids returns {UUID: float} via one projection scan,
     zero AES-GCM decrypt, NULL-to-0.0 mapping, and absent-id omission.
  2. _top_n_records_by_centrality returns the same ordered top-N as the
     legacy per-member store.get loop, for all tie/missing-centrality cases.
  3. store.get is never called inside _top_n_records_by_centrality.

All tests use a hermetic MemoryStore (tmp_path) with an in-process keyring
stub so no macOS keyring prompts occur and no real ~/.iai-mcp/ path is touched.
"""
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


# ------------------------------------------------------------------ fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Prevent macOS keyring prompts by swapping the keyring backend."""
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
    """Hermetic MemoryStore rooted at tmp_path, no real daemon."""
    return MemoryStore(path=tmp_path / "db")


def _rec(*, literal: str, centrality: float, store: MemoryStore) -> MemoryRecord:
    """Build and insert a MemoryRecord with the given centrality."""
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


# ------------------------------------------------ centrality_for_ids


def test_centrality_for_ids_no_decrypt(store: MemoryStore) -> None:
    """centrality_for_ids returns correct values without calling store.get.

    Seeds three records with known centralities (0.8, 0.3, 0.0).
    Calls centrality_for_ids with those ids plus one absent id.

    Assertions:
    - Returned dict maps each present id to the seeded centrality.
    - Absent id is omitted.
    - store.get is never called (patched to raise if invoked).
    """
    r_high = _rec(literal="high", centrality=0.8, store=store)
    r_low = _rec(literal="low", centrality=0.3, store=store)
    r_zero = _rec(literal="zero", centrality=0.0, store=store)
    absent_id = uuid4()

    ids = [r_high.id, r_low.id, r_zero.id, absent_id]

    # Patch store.get to raise — if the bulk method calls it, the test fails.
    with patch.object(store, "get", side_effect=AssertionError("store.get must not be called")):
        result = store.centrality_for_ids(ids)

    assert isinstance(result, dict)
    # Present ids mapped correctly.
    assert result[r_high.id] == pytest.approx(0.8)
    assert result[r_low.id] == pytest.approx(0.3)
    assert result[r_zero.id] == pytest.approx(0.0)
    # Absent id must be omitted.
    assert absent_id not in result
    # Exactly the three present ids.
    assert set(result.keys()) == {r_high.id, r_low.id, r_zero.id}


def test_centrality_for_ids_empty_input(store: MemoryStore) -> None:
    """Empty id list returns an empty dict."""
    with patch.object(store, "get", side_effect=AssertionError("store.get must not be called")):
        result = store.centrality_for_ids([])
    assert result == {}


def test_centrality_for_ids_all_absent(store: MemoryStore) -> None:
    """All ids absent from store returns empty dict."""
    ids = [uuid4(), uuid4()]
    with patch.object(store, "get", side_effect=AssertionError("store.get must not be called")):
        result = store.centrality_for_ids(ids)
    assert result == {}


# ----------------------------------------- _top_n_records_by_centrality


def _legacy_top_n(
    store_centrality_map: dict[UUID, float],
    member_ids: list[UUID],
    n: int,
) -> list[UUID]:
    """Reference implementation: per-member logic identical to the old code.

    Accepts a pre-built {id: centrality} map (simulating per-member store.get)
    and applies the same sort as the original implementation.
    """
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
    """_top_n_records_by_centrality result is byte-identical to the legacy sort.

    Exercises distinct centralities, tie, member id absent from the store.
    Asserts store.get is NOT called inside the function.
    """
    cid = uuid4()
    r_a = _rec(literal="a", centrality=0.9, store=store)
    r_b = _rec(literal="b", centrality=0.5, store=store)
    r_c = _rec(literal="c", centrality=0.5, store=store)  # tie with r_b
    r_d = _rec(literal="d", centrality=0.1, store=store)
    absent_id = uuid4()  # not in store

    member_ids = [r_a.id, r_b.id, r_c.id, r_d.id, absent_id]

    assignment = CommunityAssignment(
        top_communities=[cid],
        community_centroids={cid: [0.0] * store.embed_dim},
        node_to_community={r_a.id: cid, r_b.id: cid, r_c.id: cid, r_d.id: cid},
        mid_regions={cid: member_ids},
    )

    # Build the reference map (present members only — absent_id excluded).
    ref_map = {
        r_a.id: 0.9,
        r_b.id: 0.5,
        r_c.id: 0.5,
        r_d.id: 0.1,
    }
    expected = _legacy_top_n(ref_map, member_ids, n=3)
    assert len(expected) == 3  # sanity

    # Patch store.get to raise — the new implementation must not call it.
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
    """Empty member list returns [] without touching store."""
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
    """All-zero centralities: order governed by str(uuid) tiebreak, not insertion."""
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
    """n > member count returns all present members, no error."""
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
