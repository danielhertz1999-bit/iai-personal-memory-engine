"""Minimal cache-key staleness-window tests.

Scope:
  (a) Single-write try_load HIT — one record inserted between two recalls
      keeps the same windowed cache key, so the second recall HITs.
  (b) Window-crossing MISS — inserting >=_STALENESS_WINDOW records forces
      a key change (the window bucket increments) → try_load returns None.
  (c) CACHE_VERSION / schema_version / embed_dim change → try_load MISS.
  (d) CACHE_VERSION != "07-09-v3" (the prior value has been bumped).
  (e) load_last_good_structural: returns persisted (assignment, rich_club)
      when only the count components have drifted.
  (f) load_last_good_structural: returns None when no file, or
      cache_version / embed_dim / schema_version mismatches.
  (g) Structural-correctness: the assignment served across a single write
      is the same object that was saved (no structural regression).

All tests are hermetic: HOME + IAI_MCP_STORE + IAI_DAEMON_SOCKET_PATH
are monkeypatched to tmp_path. The live ~/.iai-mcp store and daemon
are never touched.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.community import CommunityAssignment
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord
import iai_mcp.runtime_graph_cache as rgc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(seed: int, text: str = "rec") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=_random_vec(seed),
        community_id=None,
        centrality=0.0,
        detail_level=2,
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


def _make_store(tmp_path: Path, monkeypatch) -> MemoryStore:
    store_root = tmp_path / "store"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    return MemoryStore(path=store_root)


def _flat_assignment(recs: list[MemoryRecord]) -> CommunityAssignment:
    cid = uuid4()
    centroid = [1.0] + [0.0] * (EMBED_DIM - 1)
    return CommunityAssignment(
        node_to_community={r.id: cid for r in recs},
        community_centroids={cid: centroid},
        modularity=0.5,
        backend="flat-test",
        top_communities=[cid],
        mid_regions={cid: [r.id for r in recs]},
    )


# ---------------------------------------------------------------------------
# (d) CACHE_VERSION bumped from the prior value
# ---------------------------------------------------------------------------


def test_cache_version_bumped():
    """CACHE_VERSION must have been bumped from the old '07-09-v3' value."""
    assert rgc.CACHE_VERSION != "07-09-v3", (
        f"CACHE_VERSION is still '07-09-v3'; it must be bumped for the "
        f"staleness-window decouple."
    )
    # Should be a non-empty string with at least a version suffix.
    assert isinstance(rgc.CACHE_VERSION, str) and len(rgc.CACHE_VERSION) > 3


# ---------------------------------------------------------------------------
# (a) Single-write try_load HIT
# ---------------------------------------------------------------------------


def test_single_write_try_load_hit(tmp_path, monkeypatch):
    """One record inserted between save and try_load → same windowed key → HIT."""
    store = _make_store(tmp_path, monkeypatch)

    # Insert base records so we start mid-window (not at a boundary).
    # Choose a base count that is mid-window: window is _STALENESS_WINDOW,
    # so a count of _STALENESS_WINDOW + 1 is at index 1 inside the window.
    window = rgc._STALENESS_WINDOW
    base = window + 1  # e.g. 11 for window=10
    recs = [_make_rec(i) for i in range(base)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    rich_club = [recs[0].id]

    # Save the cache at this point.
    ok = rgc.save(store, assignment, rich_club, max_degree=1)
    assert ok, "save() returned False — cache write failed"

    # Insert ONE more record (still within the same window bucket).
    extra = _make_rec(seed=base + 100)
    store.insert(extra)

    # The windowed key should still match → HIT.
    result = rgc.try_load(store)
    assert result is not None, (
        "try_load returned None after a single-write — the windowed cache key "
        "should have kept the same bucket (HIT expected)."
    )
    loaded_assignment, loaded_rich_club, _node_payload, _max_degree = result
    assert loaded_assignment is not None
    assert loaded_rich_club == [recs[0].id]


# ---------------------------------------------------------------------------
# (b) Window-crossing MISS
# ---------------------------------------------------------------------------


def test_window_crossing_miss(tmp_path, monkeypatch):
    """Inserting >= _STALENESS_WINDOW records crosses the window → try_load MISS."""
    store = _make_store(tmp_path, monkeypatch)

    window = rgc._STALENESS_WINDOW
    base = window + 1
    recs = [_make_rec(i) for i in range(base)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    ok = rgc.save(store, assignment, [], max_degree=0)
    assert ok

    # Insert enough records to cross the next window boundary.
    for i in range(window):
        store.insert(_make_rec(seed=base + i + 200))

    # Now the windowed count has ticked over at least once → MISS.
    result = rgc.try_load(store)
    assert result is None, (
        "try_load returned a HIT after crossing a window boundary — "
        "the cache key should have changed (MISS expected)."
    )


# ---------------------------------------------------------------------------
# (c) Schema / embed_dim / CACHE_VERSION change → MISS
# ---------------------------------------------------------------------------


def test_cache_version_change_misses(tmp_path, monkeypatch):
    """A tampered CACHE_VERSION in the saved file → try_load MISS."""
    store = _make_store(tmp_path, monkeypatch)

    recs = [_make_rec(i) for i in range(5)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    ok = rgc.save(store, assignment, [], max_degree=0)
    assert ok

    # Corrupt the cache by decrypting it, changing cache_version, and
    # re-saving under the SAME path but with a wrong version.
    import json
    from iai_mcp.crypto import decrypt_field, encrypt_field, is_encrypted

    cache_path = rgc._cache_path(store)
    raw = cache_path.read_text(encoding="utf-8")
    key = rgc._cache_encryption_key(store)
    plaintext = decrypt_field(raw, key, rgc._CACHE_AAD)
    data = json.loads(plaintext)
    data["cache_version"] = "corrupted-version-xyz"
    # Update key tuple's last element too.
    old_key = list(data.get("key", []))
    if old_key:
        old_key[-1] = "corrupted-version-xyz"
        data["key"] = old_key
    new_plaintext = json.dumps(data)
    new_cipher = encrypt_field(new_plaintext, key, rgc._CACHE_AAD)
    cache_path.write_text(new_cipher, encoding="ascii")

    result = rgc.try_load(store)
    assert result is None, "Expected MISS after CACHE_VERSION tampered"


# ---------------------------------------------------------------------------
# (e) load_last_good_structural: returns last-good on count-only drift
# ---------------------------------------------------------------------------


def test_load_last_good_structural_count_drift_hit(tmp_path, monkeypatch):
    """load_last_good_structural returns cached (assignment, rich_club) even
    when try_load would MISS due to a window-crossing count change."""
    store = _make_store(tmp_path, monkeypatch)

    window = rgc._STALENESS_WINDOW
    base = window + 1
    recs = [_make_rec(i) for i in range(base)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    rich_club = [recs[0].id, recs[1].id]
    ok = rgc.save(store, assignment, rich_club, max_degree=2)
    assert ok

    # Cross the window so try_load returns None.
    for i in range(window):
        store.insert(_make_rec(seed=base + i + 300))

    assert rgc.try_load(store) is None, "Precondition: try_load should MISS"

    # load_last_good_structural must still return the persisted assignment.
    result = rgc.load_last_good_structural(store)
    assert result is not None, (
        "load_last_good_structural returned None after a count-only drift — "
        "should return the last-good (assignment, rich_club)."
    )
    loaded_assignment, loaded_rich_club = result
    assert loaded_assignment is not None
    assert set(loaded_rich_club) == {recs[0].id, recs[1].id}
    # The loaded assignment should have nodes from the original save.
    assert len(loaded_assignment.node_to_community) == base


# ---------------------------------------------------------------------------
# (f) load_last_good_structural: returns None on no-file / parity mismatch
# ---------------------------------------------------------------------------


def test_load_last_good_structural_no_file(tmp_path, monkeypatch):
    """load_last_good_structural returns None when no cache file exists."""
    store = _make_store(tmp_path, monkeypatch)
    # Don't save — no file present.
    assert rgc.load_last_good_structural(store) is None


def test_load_last_good_structural_cache_version_mismatch(tmp_path, monkeypatch):
    """load_last_good_structural returns None when the saved cache_version
    differs from the current CACHE_VERSION (e.g. after a bump)."""
    store = _make_store(tmp_path, monkeypatch)

    recs = [_make_rec(i) for i in range(5)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    ok = rgc.save(store, assignment, [], max_degree=0)
    assert ok

    # Tamper: change saved cache_version to simulate an old-format file.
    import json
    from iai_mcp.crypto import decrypt_field, encrypt_field

    cache_path = rgc._cache_path(store)
    raw = cache_path.read_text(encoding="utf-8")
    key = rgc._cache_encryption_key(store)
    plaintext = decrypt_field(raw, key, rgc._CACHE_AAD)
    data = json.loads(plaintext)
    data["cache_version"] = "old-format-v1"
    old_key = list(data.get("key", []))
    if old_key:
        old_key[-1] = "old-format-v1"
        data["key"] = old_key
    new_plaintext = json.dumps(data)
    new_cipher = encrypt_field(new_plaintext, key, rgc._CACHE_AAD)
    cache_path.write_text(new_cipher, encoding="ascii")

    result = rgc.load_last_good_structural(store)
    assert result is None, (
        "load_last_good_structural should return None when cache_version "
        "differs — a stale-format snapshot must never reach the hot path."
    )


def test_load_last_good_structural_embed_dim_mismatch(tmp_path, monkeypatch):
    """load_last_good_structural returns None when the saved embed_dim
    differs from the current store's embed_dim."""
    store = _make_store(tmp_path, monkeypatch)

    recs = [_make_rec(i) for i in range(5)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    ok = rgc.save(store, assignment, [], max_degree=0)
    assert ok

    # Tamper: corrupt the embed_dim in the saved key.
    import json
    from iai_mcp.crypto import decrypt_field, encrypt_field

    cache_path = rgc._cache_path(store)
    raw = cache_path.read_text(encoding="utf-8")
    key = rgc._cache_encryption_key(store)
    plaintext = decrypt_field(raw, key, rgc._CACHE_AAD)
    data = json.loads(plaintext)
    old_key = list(data.get("key", []))
    if len(old_key) >= 4:
        old_key[3] = 999  # wrong embed_dim
        data["key"] = old_key
    new_plaintext = json.dumps(data)
    new_cipher = encrypt_field(new_plaintext, key, rgc._CACHE_AAD)
    cache_path.write_text(new_cipher, encoding="ascii")

    result = rgc.load_last_good_structural(store)
    assert result is None, (
        "load_last_good_structural should return None when embed_dim differs."
    )


# ---------------------------------------------------------------------------
# (g) Served-across-write structural correctness
# ---------------------------------------------------------------------------


def test_assignment_served_across_single_write_correctness(tmp_path, monkeypatch):
    """The assignment served after a single write (via HIT) is structurally
    identical to the one that was saved — no node map corruption."""
    store = _make_store(tmp_path, monkeypatch)

    window = rgc._STALENESS_WINDOW
    base = window + 1
    recs = [_make_rec(i) for i in range(base)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    rich_club = [recs[0].id]
    ok = rgc.save(store, assignment, rich_club, max_degree=1)
    assert ok

    # Insert ONE record — should still be a HIT.
    store.insert(_make_rec(seed=base + 50))

    result = rgc.try_load(store)
    assert result is not None, "Expected HIT after single write"

    loaded_assignment, _loaded_rich_club, _node_payload, _max_degree = result
    # Node-to-community map should cover the same records as the original save.
    original_ids = {r.id for r in recs}
    loaded_ids = set(loaded_assignment.node_to_community.keys())
    # The loaded assignment was saved with the original recs — the sets match.
    assert loaded_ids == original_ids, (
        f"Loaded assignment IDs differ from saved: "
        f"{len(loaded_ids)} loaded vs {len(original_ids)} saved"
    )
    # Community assignment modularity should be preserved.
    assert loaded_assignment.modularity == assignment.modularity
    assert loaded_assignment.backend == assignment.backend


# ---------------------------------------------------------------------------
# load_recall_structural: 3-case smoke test
# ---------------------------------------------------------------------------


def test_load_recall_structural_case1_hit(tmp_path, monkeypatch):
    """Case 1: try_load HIT → structural_source = 'normal'."""
    store = _make_store(tmp_path, monkeypatch)

    window = rgc._STALENESS_WINDOW
    recs = [_make_rec(i) for i in range(window + 2)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    ok = rgc.save(store, assignment, [recs[0].id], max_degree=1)
    assert ok

    a, rc, max_deg, src = rgc.load_recall_structural(store)
    assert src == "normal"
    assert a is not None
    assert recs[0].id in rc
    assert max_deg == 1


def test_load_recall_structural_case2_last_good(tmp_path, monkeypatch):
    """Case 2: count drift → try_load MISS but file present → 'last_good'."""
    store = _make_store(tmp_path, monkeypatch)

    window = rgc._STALENESS_WINDOW
    recs = [_make_rec(i) for i in range(window + 2)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    ok = rgc.save(store, assignment, [recs[0].id], max_degree=1)
    assert ok

    # Cross the window.
    for i in range(window):
        store.insert(_make_rec(seed=300 + i))

    a, rc, max_deg, src = rgc.load_recall_structural(store)
    assert src == "last_good"
    assert a is not None
    assert recs[0].id in rc
    assert max_deg == 0  # last_good path does not expose max_degree (returns 0)


def test_load_recall_structural_case3_cold(tmp_path, monkeypatch):
    """Case 3: no cache file → 'cold_degrade', empty assignment, empty rich_club."""
    store = _make_store(tmp_path, monkeypatch)

    a, rc, max_deg, src = rgc.load_recall_structural(store)
    assert src == "cold_degrade"
    assert len(a.node_to_community) == 0
    assert rc == []
    assert max_deg == 0
