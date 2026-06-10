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

def test_cache_version_bumped():
    assert rgc.CACHE_VERSION != "07-09-v3", (
        f"CACHE_VERSION is still '07-09-v3'; it must be bumped for the "
        f"staleness-window decouple."
    )
    assert isinstance(rgc.CACHE_VERSION, str) and len(rgc.CACHE_VERSION) > 3

def test_single_write_try_load_hit(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)

    window = rgc._STALENESS_WINDOW
    base = window + 1
    recs = [_make_rec(i) for i in range(base)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    rich_club = [recs[0].id]

    ok = rgc.save(store, assignment, rich_club, max_degree=1)
    assert ok, "save() returned False — cache write failed"

    extra = _make_rec(seed=base + 100)
    store.insert(extra)

    result = rgc.try_load(store)
    assert result is not None, (
        "try_load returned None after a single-write — the windowed cache key "
        "should have kept the same bucket (HIT expected)."
    )
    loaded_assignment, loaded_rich_club, _node_payload, _max_degree = result
    assert loaded_assignment is not None
    assert loaded_rich_club == [recs[0].id]

def test_window_crossing_miss(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)

    window = rgc._STALENESS_WINDOW
    base = window + 1
    recs = [_make_rec(i) for i in range(base)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    ok = rgc.save(store, assignment, [], max_degree=0)
    assert ok

    for i in range(window):
        store.insert(_make_rec(seed=base + i + 200))

    result = rgc.try_load(store)
    assert result is None, (
        "try_load returned a HIT after crossing a window boundary — "
        "the cache key should have changed (MISS expected)."
    )

def test_cache_version_change_misses(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)

    recs = [_make_rec(i) for i in range(5)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    ok = rgc.save(store, assignment, [], max_degree=0)
    assert ok

    import json
    from iai_mcp.crypto import decrypt_field, encrypt_field, is_encrypted

    cache_path = rgc._cache_path(store)
    raw = cache_path.read_text(encoding="utf-8")
    key = rgc._cache_encryption_key(store)
    plaintext = decrypt_field(raw, key, rgc._CACHE_AAD)
    data = json.loads(plaintext)
    data["cache_version"] = "corrupted-version-xyz"
    old_key = list(data.get("key", []))
    if old_key:
        old_key[-1] = "corrupted-version-xyz"
        data["key"] = old_key
    new_plaintext = json.dumps(data)
    new_cipher = encrypt_field(new_plaintext, key, rgc._CACHE_AAD)
    cache_path.write_text(new_cipher, encoding="ascii")

    result = rgc.try_load(store)
    assert result is None, "Expected MISS after CACHE_VERSION tampered"

def test_load_last_good_structural_count_drift_hit(tmp_path, monkeypatch):
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

    for i in range(window):
        store.insert(_make_rec(seed=base + i + 300))

    assert rgc.try_load(store) is None, "Precondition: try_load should MISS"

    result = rgc.load_last_good_structural(store)
    assert result is not None, (
        "load_last_good_structural returned None after a count-only drift — "
        "should return the last-good (assignment, rich_club)."
    )
    loaded_assignment, loaded_rich_club = result
    assert loaded_assignment is not None
    assert set(loaded_rich_club) == {recs[0].id, recs[1].id}
    assert len(loaded_assignment.node_to_community) == base

def test_load_last_good_structural_no_file(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    assert rgc.load_last_good_structural(store) is None

def test_load_last_good_structural_cache_version_mismatch(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)

    recs = [_make_rec(i) for i in range(5)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    ok = rgc.save(store, assignment, [], max_degree=0)
    assert ok

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
    store = _make_store(tmp_path, monkeypatch)

    recs = [_make_rec(i) for i in range(5)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    ok = rgc.save(store, assignment, [], max_degree=0)
    assert ok

    import json
    from iai_mcp.crypto import decrypt_field, encrypt_field

    cache_path = rgc._cache_path(store)
    raw = cache_path.read_text(encoding="utf-8")
    key = rgc._cache_encryption_key(store)
    plaintext = decrypt_field(raw, key, rgc._CACHE_AAD)
    data = json.loads(plaintext)
    old_key = list(data.get("key", []))
    if len(old_key) >= 4:
        old_key[3] = 999
        data["key"] = old_key
    new_plaintext = json.dumps(data)
    new_cipher = encrypt_field(new_plaintext, key, rgc._CACHE_AAD)
    cache_path.write_text(new_cipher, encoding="ascii")

    result = rgc.load_last_good_structural(store)
    assert result is None, (
        "load_last_good_structural should return None when embed_dim differs."
    )

def test_assignment_served_across_single_write_correctness(tmp_path, monkeypatch):
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

    store.insert(_make_rec(seed=base + 50))

    result = rgc.try_load(store)
    assert result is not None, "Expected HIT after single write"

    loaded_assignment, _loaded_rich_club, _node_payload, _max_degree = result
    original_ids = {r.id for r in recs}
    loaded_ids = set(loaded_assignment.node_to_community.keys())
    assert loaded_ids == original_ids, (
        f"Loaded assignment IDs differ from saved: "
        f"{len(loaded_ids)} loaded vs {len(original_ids)} saved"
    )
    assert loaded_assignment.modularity == assignment.modularity
    assert loaded_assignment.backend == assignment.backend

def test_load_recall_structural_case1_hit(tmp_path, monkeypatch):
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
    store = _make_store(tmp_path, monkeypatch)

    window = rgc._STALENESS_WINDOW
    recs = [_make_rec(i) for i in range(window + 2)]
    for r in recs:
        store.insert(r)

    assignment = _flat_assignment(recs)
    ok = rgc.save(store, assignment, [recs[0].id], max_degree=1)
    assert ok

    for i in range(window):
        store.insert(_make_rec(seed=300 + i))

    a, rc, max_deg, src = rgc.load_recall_structural(store)
    assert src == "last_good"
    assert a is not None
    assert recs[0].id in rc
    assert max_deg == 0

def test_load_recall_structural_case3_cold(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)

    a, rc, max_deg, src = rgc.load_recall_structural(store)
    assert src == "cold_degrade"
    assert len(a.node_to_community) == 0
    assert rc == []
    assert max_deg == 0
