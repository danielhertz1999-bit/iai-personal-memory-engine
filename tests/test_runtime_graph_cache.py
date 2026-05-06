"""Plan 05-09 (P4.A) — runtime_graph_cache tests.

The cache persists the Leiden community assignment + rich-club node
list next to the LanceDB store. On subsequent ``build_runtime_graph``
calls, a valid cache skips the Leiden + rich-club computations;
invalid cache falls through to a clean rebuild.

Covered contracts:

    1. save() creates the JSON file
    2. try_load on unchanged store returns the cached pair
    3. adding a record invalidates the cache (key mismatch)
    4. CACHE_VERSION mismatch triggers rebuild (forward-compat fence)
    5. corrupt JSON falls through to a clean None
    6. absent file returns None cleanly
    7. build_runtime_graph on second call avoids detect_communities
    8. build_runtime_graph on store change triggers detect_communities
    9. atomic write: interrupted save leaves old cache intact
   10. invalidate() deletes the cache file
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
from uuid import UUID, uuid4

import pytest

from iai_mcp import retrieve, runtime_graph_cache
from iai_mcp.community import CommunityAssignment
from iai_mcp.store import MemoryStore


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
    """Fresh MemoryStore in tmp_path/lancedb with the cache file path set
    to tmp_path/runtime_graph_cache.json."""
    s = MemoryStore(path=tmp_path / "lancedb")
    # Override root so the cache file lives in tmp_path, not the real store
    # root (which would be tmp_path/lancedb — still fine, but explicit).
    s.root = tmp_path
    return s


def _read_decrypted_cache(store: MemoryStore, path: Path) -> dict:
    """Phase 07.9 W3: decrypt the v3 ciphertext sidecar and return the
    underlying JSON dict. Tests that pre-07.9 read the cache via
    ``json.load(f)`` go through this helper instead.
    """
    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.startswith("iai:enc:v1:"):
        # Legacy / hand-written plaintext (test 4 simulating v2).
        return json.loads(raw_text)
    from iai_mcp.crypto import decrypt_field
    plaintext = decrypt_field(
        raw_text,
        store._key(),
        runtime_graph_cache._CACHE_AAD,
    )
    return json.loads(plaintext)


def _write_encrypted_cache(store: MemoryStore, path: Path, data: dict) -> None:
    """Inverse of _read_decrypted_cache: encrypt and write a hand-modified
    JSON dict back to the sidecar in v3 ciphertext format.
    """
    from iai_mcp.crypto import encrypt_field
    plaintext = json.dumps(data)
    ciphertext = encrypt_field(
        plaintext,
        store._key(),
        runtime_graph_cache._CACHE_AAD,
    )
    path.write_text(ciphertext, encoding="ascii")


def _make_assignment(n_communities: int = 2) -> CommunityAssignment:
    comms = [uuid4() for _ in range(n_communities)]
    nodes = [uuid4() for _ in range(n_communities * 3)]
    return CommunityAssignment(
        node_to_community={
            nodes[i]: comms[i // 3] for i in range(len(nodes))
        },
        community_centroids={c: [0.1, 0.2, 0.3] for c in comms},
        modularity=0.42,
        backend="leiden-networkx",
        top_communities=comms,
        mid_regions={c: nodes[i * 3:(i + 1) * 3] for i, c in enumerate(comms)},
    )


# --------------------------------------------------------------------------- Test 1


def test_save_creates_json_file(store):
    assignment = _make_assignment()
    rich_club = [uuid4() for _ in range(5)]
    ok = runtime_graph_cache.save(store, assignment, rich_club)
    assert ok is True
    path = runtime_graph_cache._cache_path(store)
    assert path.exists()
    # W3: file is now AES-256-GCM-wrapped — decrypt before
    # inspecting the JSON shape.
    raw = path.read_text(encoding="utf-8")
    assert raw.startswith("iai:enc:v1:"), (
        "Phase 07.9 W3: cache must be v3 ciphertext on disk"
    )
    data = _read_decrypted_cache(store, path)
    assert data["cache_version"] == runtime_graph_cache.CACHE_VERSION
    assert "assignment" in data
    assert "rich_club" in data
    assert "key" in data


# --------------------------------------------------------------------------- Test 2


def test_try_load_round_trip_on_unchanged_store(store):
    assignment = _make_assignment()
    rich_club = [uuid4() for _ in range(3)]
    runtime_graph_cache.save(store, assignment, rich_club)

    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None
    # try_load returns (assignment, rich_club, node_payload, ...).
    # node_payload is the v2 blob — None (or empty) when legacy 2-arg
    # save() shape was used.
    # 4th element is max_degree (int >= 0).
    loaded_assignment, loaded_rich_club, _node_payload, _max_degree = loaded
    assert loaded_assignment.backend == assignment.backend
    assert loaded_assignment.modularity == pytest.approx(assignment.modularity)
    assert set(loaded_assignment.top_communities) == set(assignment.top_communities)
    assert set(loaded_rich_club) == set(rich_club)


# --------------------------------------------------------------------------- Test 3


def test_key_mismatch_invalidates_cache(store):
    # Save with original key.
    runtime_graph_cache.save(store, _make_assignment(), [uuid4()])
    path = runtime_graph_cache._cache_path(store)
    assert path.exists()

    # Simulate a store change by forging a wrong key in the saved file.
    # W3: decrypt → mutate → re-encrypt round-trip.
    data = _read_decrypted_cache(store, path)
    data["key"][0] = 999  # bogus records_count
    _write_encrypted_cache(store, path, data)

    assert runtime_graph_cache.try_load(store) is None


# --------------------------------------------------------------------------- Test 4


def test_cache_version_mismatch_triggers_rebuild(store):
    runtime_graph_cache.save(store, _make_assignment(), [uuid4()])
    path = runtime_graph_cache._cache_path(store)
    # W3: decrypt → mutate version → re-encrypt round-trip.
    data = _read_decrypted_cache(store, path)
    data["cache_version"] = "old-format-v0"
    _write_encrypted_cache(store, path, data)

    assert runtime_graph_cache.try_load(store) is None


# --------------------------------------------------------------------------- Test 5


def test_corrupt_json_returns_none(store):
    path = runtime_graph_cache._cache_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json at all")
    assert runtime_graph_cache.try_load(store) is None


# --------------------------------------------------------------------------- Test 6


def test_absent_cache_returns_none(store):
    path = runtime_graph_cache._cache_path(store)
    assert not path.exists()
    assert runtime_graph_cache.try_load(store) is None


# --------------------------------------------------------------------------- Test 7


def test_build_runtime_graph_uses_cache_on_second_call(store):
    # First call: detect_communities runs, cache is written.
    with mock.patch(
        "iai_mcp.community.detect_communities",
        wraps=__import__("iai_mcp.community", fromlist=["detect_communities"]).detect_communities,
    ) as detect_spy:
        retrieve.build_runtime_graph(store)
        assert detect_spy.call_count == 1

    # Second call: cache is valid, detect_communities must NOT re-run.
    with mock.patch(
        "iai_mcp.community.detect_communities",
    ) as detect_spy:
        retrieve.build_runtime_graph(store)
        assert detect_spy.call_count == 0


# --------------------------------------------------------------------------- Test 8


def test_build_runtime_graph_invalidates_on_record_added(store, tmp_path):
    """Adding a record bumps records_count -> cache key changes -> rebuild."""
    # First call seeds the cache on an empty store.
    retrieve.build_runtime_graph(store)
    assert runtime_graph_cache._cache_path(store).exists()

    # Insert a real record via the store so records_count increments.
    from datetime import datetime, timezone
    from iai_mcp.types import MemoryRecord

    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="x",
        aaak_index="",
        embedding=[0.0] * store.embed_dim,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["t"],
        language="en",
    )
    store.insert(rec)

    # Second build sees records_count changed -> cache invalid -> rebuild.
    with mock.patch(
        "iai_mcp.community.detect_communities",
        wraps=__import__("iai_mcp.community", fromlist=["detect_communities"]).detect_communities,
    ) as detect_spy:
        retrieve.build_runtime_graph(store)
        assert detect_spy.call_count == 1


# --------------------------------------------------------------------------- Test 9


def test_save_is_atomic_leaves_old_file_on_error(store, monkeypatch):
    """If os.replace raises mid-save the .tmp is cleaned up and the old
    cache file (if any) is untouched."""
    # Seed a valid cache first.
    original_assignment = _make_assignment()
    runtime_graph_cache.save(store, original_assignment, [uuid4()])
    path = runtime_graph_cache._cache_path(store)
    original_text = path.read_text()

    # Now break os.replace and try to save again.
    monkeypatch.setattr(
        "iai_mcp.runtime_graph_cache.os.replace",
        mock.Mock(side_effect=OSError("rename failed")),
    )
    ok = runtime_graph_cache.save(store, _make_assignment(), [uuid4()])
    assert ok is False
    # Original cache still intact.
    assert path.read_text() == original_text
    # Temp file was cleaned up.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    assert not tmp_path.exists()


# --------------------------------------------------------------------------- Test 10


def test_invalidate_removes_cache_file(store):
    runtime_graph_cache.save(store, _make_assignment(), [uuid4()])
    path = runtime_graph_cache._cache_path(store)
    assert path.exists()

    runtime_graph_cache.invalidate(store)
    assert not path.exists()

    # Idempotent — second invalidate on missing file is a no-op.
    runtime_graph_cache.invalidate(store)


# --------------------------------------------------------------------------- Test 11 (bonus)


def test_embed_dim_change_invalidates(store):
    """swapping embedders (1024d -> 384d) must force a
    rebuild. The cache_key includes store.embed_dim so this is
    automatic — no separate signal needed."""
    runtime_graph_cache.save(store, _make_assignment(), [uuid4()])
    assert runtime_graph_cache.try_load(store) is not None

    # Simulate an embedder swap by monkeypatching store.embed_dim.
    store._embed_dim = 1024  # underlying attr
    assert runtime_graph_cache.try_load(store) is None


# --------------------------------------------------------------------------- tests


def test_save_drops_oversize_community_centroids(store):
    """when ``assignment.community_centroids`` alone overflows
    the 10 MiB cap (the scenario on an all-isolated graph where Leiden
    gives one community per node), the iterative drop path must prune
    centroids too — not just node_payload — and produce a file ≤ cap.

    Pre-F-09 behaviour: the single-shot drop cleared node_payload,
    re-serialised, saw the payload still > cap, and shipped it anyway.
    Post-node_payload drops first, centroids drop next,
    node_to_community + modularity + top_communities survive.
    """
    # 2000 communities × 1024-dim float vectors ≈ 40 MiB JSON-encoded,
    # well over the 10 MiB cap. This shape mirrors the live bug.
    big_centroids = {uuid4(): [0.123456] * 1024 for _ in range(2000)}
    big_node_to_community = {uuid4(): uuid4() for _ in range(50)}
    assignment = CommunityAssignment(
        node_to_community=big_node_to_community,
        community_centroids=big_centroids,
        modularity=0.37,
        backend="leiden-networkx",
        top_communities=list(big_centroids.keys())[:5],
        mid_regions={c: [] for c in list(big_centroids.keys())[:5]},
    )
    rich_club = [uuid4() for _ in range(10)]

    # Non-empty node_payload so the first drop step has something to remove.
    node_payload = {
        str(uuid4()): {
            "embedding": [0.0] * 384,
            "surface": "probe",
            "centrality": 0.1,
            "tier": "episodic",
            "pinned": False,
            "tags": [],
            "language": "en",
        }
    }

    ok = runtime_graph_cache.save(store, assignment, rich_club, node_payload=node_payload)
    assert ok is True

    path = runtime_graph_cache._cache_path(store)
    assert path.exists()

    # File respects the cap.
    size = path.stat().st_size
    assert size <= runtime_graph_cache.MAX_CACHE_BYTES, (
        f"cache file {size} bytes exceeds cap "
        f"{runtime_graph_cache.MAX_CACHE_BYTES}"
    )

    data = _read_decrypted_cache(store, path)

    # Both pre-F-09 drop candidates emptied in order.
    assert data["node_payload"] == {}
    assert data["assignment"]["community_centroids"] == {}

    # Authoritative fields survived.
    assert data["assignment"]["modularity"] == pytest.approx(0.37)
    assert data["assignment"]["backend"] == "leiden-networkx"
    assert len(data["assignment"]["node_to_community"]) == len(big_node_to_community)
    assert len(data["assignment"]["top_communities"]) == 5
    # rich_club preserved.
    assert len(data["rich_club"]) == len(rich_club)


def test_save_small_payload_survives_unchanged(store):
    """Negative case: when the payload is comfortably under the cap,
    no drops fire — centroids, mid_regions, and node_payload round-trip
    intact. Guards against an over-eager iterative-drop path.
    """
    assignment = _make_assignment(n_communities=2)
    rich_club = [uuid4() for _ in range(3)]
    node_payload = {
        str(uuid4()): {
            "embedding": [0.1] * 384,
            "surface": "hello",
            "centrality": 0.2,
            "tier": "episodic",
            "pinned": False,
            "tags": ["t"],
            "language": "en",
        }
        for _ in range(5)
    }

    ok = runtime_graph_cache.save(store, assignment, rich_club, node_payload=node_payload)
    assert ok is True

    path = runtime_graph_cache._cache_path(store)
    data = _read_decrypted_cache(store, path)

    # Well under the cap.
    assert path.stat().st_size < runtime_graph_cache.MAX_CACHE_BYTES

    # Nothing pruned.
    assert data["node_payload"] != {}
    assert len(data["node_payload"]) == 5
    assert data["assignment"]["community_centroids"] != {}
    assert len(data["assignment"]["community_centroids"]) == 2
    assert data["assignment"]["mid_regions"] != {}
    assert len(data["assignment"]["mid_regions"]) == 2


# --------------------------------------------------------------------------- W3 / tests


def test_save_writes_ciphertext_no_plaintext_surface(store):
    """W3 / saved sidecar must NOT contain plaintext surface bytes
    anywhere on disk. The whole JSON payload is wrapped in AES-256-GCM."""
    canary = "PLAINTEXT_CANARY_4d7f_07_9_W3"
    rid = uuid4()
    node_payload = {
        str(rid): {
            "embedding": [0.1] * 384,
            "surface": canary,
            "centrality": 0.5,
            "tier": "episodic",
            "pinned": False,
            "tags": ["t"],
            "language": "en",
        },
    }
    ok = runtime_graph_cache.save(store, _make_assignment(), [uuid4()],
                                  node_payload=node_payload, max_degree=3)
    assert ok is True
    path = runtime_graph_cache._cache_path(store)
    raw_bytes = path.read_bytes()

    # Plaintext canary must not appear anywhere in the on-disk bytes.
    assert canary.encode("utf-8") not in raw_bytes, (
        "plaintext surface canary leaked into the on-disk sidecar"
    )
    # Ciphertext envelope present.
    assert raw_bytes.startswith(b"iai:enc:v1:"), (
        f"expected v3 ciphertext envelope; got prefix {raw_bytes[:32]!r}"
    )


def test_save_then_try_load_preserves_surface_byte_for_byte(store):
    """W3 / surface round-trips through encrypt → decrypt cleanly,
    including non-ASCII (byte-for-byte across encryption)."""
    rid = uuid4()
    surface = "user сказал важное — please remember this 重要"
    node_payload = {
        str(rid): {
            "embedding": [0.42] * 384,
            "surface": surface,
            "centrality": 0.42,
            "tier": "episodic",
            "pinned": True,
            "tags": ["t1", "t2"],
            "language": "ru",
        },
    }
    runtime_graph_cache.save(store, _make_assignment(), [rid],
                             node_payload=node_payload, max_degree=7)

    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None
    _, _, payload, max_deg = loaded
    assert payload is not None
    assert payload[str(rid)]["surface"] == surface
    assert payload[str(rid)]["centrality"] == pytest.approx(0.42)
    assert payload[str(rid)]["tags"] == ["t1", "t2"]
    assert payload[str(rid)]["language"] == "ru"
    assert max_deg == 7


def test_v2_plaintext_lazy_migrates_to_v3(store):
    """W3 / a hand-written legacy v2 plaintext file is read once,
    then re-saved under the v3 ciphertext format on the same call.
    Subsequent reads see only ciphertext on disk."""
    path = runtime_graph_cache._cache_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Hand-craft a v2 plaintext file in the legacy shape. Use the
    # current store key so the cache_key matches and try_load
    # accepts the contents.
    rid = uuid4()
    legacy_data = {
        "cache_version": runtime_graph_cache.LEGACY_CACHE_VERSION_PLAINTEXT,
        "key": list(runtime_graph_cache._cache_key(store)),
        "assignment": runtime_graph_cache._encode_assignment(_make_assignment()),
        "rich_club": [str(uuid4())],
        "node_payload": {
            str(rid): {
                "embedding": [0.0] * 384,
                "surface": "legacy_plain_canary",
                "centrality": 0.0,
                "tier": "episodic",
                "pinned": False,
                "tags": [],
                "language": "en",
            }
        },
        "max_degree": 1,
        "saved_at": "2026-04-29T00:00:00Z",
    }
    # Force the legacy cache_version into the saved key so try_load's
    # current_key check matches. (cache_version is the 5th element.)
    if len(legacy_data["key"]) >= 5:
        legacy_data["key"][4] = runtime_graph_cache.LEGACY_CACHE_VERSION_PLAINTEXT
    path.write_text(json.dumps(legacy_data), encoding="utf-8")

    # First try_load reads the v2 plaintext, decodes, and re-saves
    # under the v3 ciphertext format.
    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None
    _, _, payload, _ = loaded
    assert payload is not None
    assert payload[str(rid)]["surface"] == "legacy_plain_canary"

    # On-disk file is now ciphertext.
    raw_after = path.read_bytes()
    assert raw_after.startswith(b"iai:enc:v1:"), (
        "v2 plaintext file must be lazily migrated to v3 ciphertext"
    )
    assert b"legacy_plain_canary" not in raw_after, (
        "post-migration on-disk bytes must not contain the legacy plaintext"
    )
