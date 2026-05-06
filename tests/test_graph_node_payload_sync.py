"""Plan 05-12 — store <-> graph write-sync hook tests (RED scaffold).

``build_runtime_graph`` registers a ``_graph_sync_hook`` on the store so
every ``insert`` / ``update`` / ``delete`` mutates the in-RAM graph's
node payload. Hook exceptions are logged to stderr as structured events
but NEVER break the underlying store write — the store is authoritative.

Covered contracts:

  B1 — ``store.insert`` with registered hook adds the graph node + payload.
  B2 — ``store.update`` mutates the node's embedding / surface payload.
  B3 — ``store.delete`` removes the node from the graph.
  B4 — hook that raises does not break ``store.insert`` — write
        completes, stderr carries a structured ``graph_sync_failed`` event.
  B5 — cold start: after save/try_load round-trip the node payload blob
        restores every node attribute from cache.
  B6 — CACHE_VERSION bump from "05-09-v1" -> "05-12-v1" invalidates the
        old cache cleanly (forward-compat fence).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp import retrieve, runtime_graph_cache
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(path=tmp_path / "lancedb")
    s.root = tmp_path
    return s


def _make_record(
    store: MemoryStore,
    text: str = "hello",
    vec_seed: float = 0.1,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[vec_seed] * store.embed_dim,
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
        tags=["t"],
        language="en",
    )


# ---------------------------------------------------------------- B1: insert


def test_B1_insert_updates_graph_node(store):
    """B1: store.insert while a hook is registered adds node + payload."""
    # Seed one record so build_runtime_graph has something to register with.
    seed = _make_record(store, "seed", 0.5)
    store.insert(seed)

    graph, _a, _rc = retrieve.build_runtime_graph(store)
    assert str(seed.id) in graph._nx.nodes
    # Now insert a second record; the hook should mirror it to the graph.
    new_rec = _make_record(store, "freshly-inserted", 0.3)
    store.insert(new_rec)

    assert str(new_rec.id) in graph._nx.nodes
    node = graph._nx.nodes[str(new_rec.id)]
    assert node.get("surface") == "freshly-inserted"
    assert "embedding" in node


# ---------------------------------------------------------------- B2: update


def test_B2_update_mutates_node_payload(store):
    """B2: store.update rewrites the node's embedding + surface on the graph."""
    rec = _make_record(store, "before-update", 0.2)
    store.insert(rec)
    graph, _a, _rc = retrieve.build_runtime_graph(store)

    node_before = graph._nx.nodes[str(rec.id)]
    assert node_before["surface"] == "before-update"

    # Mutate surface and embedding.
    rec.literal_surface = "after-update"
    rec.embedding = [0.9] * store.embed_dim
    store.update(rec)

    node_after = graph._nx.nodes[str(rec.id)]
    assert node_after["surface"] == "after-update"
    # embedding replaced (first element is 0.9 now)
    assert list(node_after["embedding"])[0] == pytest.approx(0.9)


# ---------------------------------------------------------------- B3: delete


def test_B3_delete_removes_node(store):
    """B3: store.delete drops the node from the graph."""
    rec = _make_record(store, "to-be-deleted", 0.4)
    store.insert(rec)
    graph, _a, _rc = retrieve.build_runtime_graph(store)
    assert str(rec.id) in graph._nx.nodes

    store.delete(rec.id)
    assert str(rec.id) not in graph._nx.nodes


# ---------------------------------------------------------------- B4: hook robustness


def test_B4_hook_exception_does_not_break_store_insert(store, capsys):
    """B4: a raising hook must never break store.insert; stderr logs a
    structured ``graph_sync_failed`` event."""
    def _bad_hook(op, record):
        raise RuntimeError("hook is sad")

    store.register_graph_sync_hook(_bad_hook)

    rec = _make_record(store, "store-write-is-authoritative", 0.15)
    store.insert(rec)  # must not raise

    # Verify the record actually landed in LanceDB.
    roundtrip = store.get(rec.id)
    assert roundtrip is not None
    assert roundtrip.literal_surface == "store-write-is-authoritative"

    # Structured stderr event logged.
    captured = capsys.readouterr()
    assert "graph_sync_failed" in captured.err
    # JSON parseability of at least one stderr line.
    found = False
    for line in captured.err.splitlines():
        try:
            payload = json.loads(line)
            if payload.get("event") == "graph_sync_failed":
                assert payload.get("op") == "insert"
                found = True
                break
        except (ValueError, TypeError):
            continue
    assert found, "expected a JSON graph_sync_failed event on stderr"


# ---------------------------------------------------------------- B5: cold start


def test_B5_cold_start_restores_node_payload_from_cache(store):
    """B5: after save/try_load, build_runtime_graph rehydrates node
    attrs from the cache without re-reading all records."""
    rec = _make_record(store, "cached-payload", 0.25)
    store.insert(rec)

    # First build — writes the v2 cache with node_payload blob.
    graph1, _a, _rc = retrieve.build_runtime_graph(store)
    node1 = graph1._nx.nodes[str(rec.id)]
    expected_surface = node1["surface"]
    expected_emb = list(node1["embedding"])

    # Inspect via try_load (cache is encrypted under v3 sidecar per Phase 07.9
    # W3 / D-03; raw file is ciphertext, so json.load on it would fail).
    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None, "cache must be loadable"
    _assignment, _rich_club, node_payload, _max_degree = loaded
    assert node_payload is not None, "cache is missing node_payload blob"
    assert str(rec.id) in node_payload

    # Rebuild — cache HIT must rehydrate payload without scanning store.all_records.
    graph2, _a, _rc = retrieve.build_runtime_graph(store)
    node2 = graph2._nx.nodes[str(rec.id)]
    assert node2["surface"] == expected_surface
    assert list(node2["embedding"]) == expected_emb


# ---------------------------------------------------------------- B6: version bump


def test_B6_cache_version_bump_invalidates_old_cache(store):
    """B6: CACHE_VERSION is "05-12-v1" — old "05-09-v1" caches invalidate
    cleanly on try_load.
    """
    # Plant an old-format cache file manually.
    cache_path = runtime_graph_cache._cache_path(store)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(
            {
                "cache_version": "05-09-v1",  # legacy
                "key": [0, 0, 4, store.embed_dim, "05-09-v1"],
                "assignment": {
                    "node_to_community": {},
                    "community_centroids": {},
                    "modularity": 0.0,
                    "backend": "flat",
                    "top_communities": [],
                    "mid_regions": {},
                },
                "rich_club": [],
                "saved_at": "2026-01-01T00:00:00+00:00",
            },
            f,
        )

    # CACHE_VERSION constant is the current one (Phase 07.9 W3 / bump
    # to "07-09-v3" with AES-256-GCM sidecar). Legacy 05-09 / 05-12 / 05-13
    # / 06-02 cache files are rejected.
    assert runtime_graph_cache.CACHE_VERSION == "07-09-v3"

    # try_load on the old cache returns None (mismatch).
    assert runtime_graph_cache.try_load(store) is None
