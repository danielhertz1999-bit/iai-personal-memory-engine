"""W2 / tests for the dry-size-estimator drop loop in
`runtime_graph_cache.save`. Defends against the regression where save()
materialised a multi-GB intermediate Python string by calling json.dumps
up to 4 times per save (CONTEXT.md py-spy 2026-04-29 PID 7959
RSS 7.6GB). After Task 1, json.dumps runs AT MOST once per
save invocation, regardless of how many fields are dropped.

W3 / cache file is now AES-256-GCM-wrapped. These
tests use ``_decrypt_cache_for_inspection`` to peek at the JSON shape.
"""
from __future__ import annotations

import json as real_json
import os
import pathlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

import iai_mcp.runtime_graph_cache as rgc
from iai_mcp.community import CommunityAssignment

# W3: encryption key resolves via CryptoKey on a fake-store
# fallback path; ensure a deterministic passphrase is set for the
# whole module so save() can encrypt without needing a keyring entry.
os.environ.setdefault("IAI_MCP_CRYPTO_PASSPHRASE", "test-rgc-size-guard-passphrase")


def _make_fake_store(tmp_path):
    # save() reads store.root for the cache file path and store.db /
    # store.embed_dim for _cache_key. _cache_key is robust against
    # exceptions and falls back to a placeholder tuple, so a minimal
    # SimpleNamespace with .root suffices for these tests.
    return SimpleNamespace(root=tmp_path)


def _decrypt_cache_for_inspection(store, path: pathlib.Path) -> dict:
    """W3: read the encrypted sidecar and decode the inner
    JSON. Used by size-guard tests that previously did
    ``real_json.load(f)`` directly on the cache file.
    """
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("iai:enc:v1:"):
        return real_json.loads(raw)
    from iai_mcp.crypto import decrypt_field
    plaintext = decrypt_field(
        raw,
        rgc._cache_encryption_key(store),
        rgc._CACHE_AAD,
    )
    return real_json.loads(plaintext)


def _make_assignment(centroids_count=5, mid_regions_count=5, embed_dim=384):
    """Build a real CommunityAssignment dataclass — `_encode_assignment`
    in runtime_graph_cache.py uses `getattr(assignment, "<field>", default)`,
    which silently returns the default on plain dicts. Only attribute access
    works (dataclass / SimpleNamespace). We use the real dataclass to mirror
    the existing test-style in tests/test_runtime_graph_cache.py.
    """
    comm_uuids = [uuid4() for _ in range(centroids_count)]
    node_uuids = [uuid4() for _ in range(centroids_count)]
    member_uuids_per_comm = [
        [uuid4() for _ in range(mid_regions_count)] for _ in range(centroids_count)
    ]
    return CommunityAssignment(
        node_to_community={node_uuids[i]: comm_uuids[i] for i in range(centroids_count)},
        community_centroids={c: [0.123456789] * embed_dim for c in comm_uuids},
        modularity=0.42,
        backend="leiden",
        top_communities=comm_uuids[: min(centroids_count, 8)],
        mid_regions={comm_uuids[i]: member_uuids_per_comm[i] for i in range(centroids_count)},
    )


def _make_node_payload(count=10, embed_dim=384):
    return {
        f"u{i}": {
            "embedding": [0.123456789] * embed_dim,
            "surface": "hello world",
            "centrality": 0.1,
            "tier": "episodic",
            "pinned": False,
            "tags": ["t1", "t2"],
            "language": "en",
        }
        for i in range(count)
    }


@pytest.fixture
def dumps_counter(monkeypatch):
    """Wrap json.dumps inside the runtime_graph_cache module so we can
    count invocations. Returns a list whose len(.) == call count.
    """
    calls = []
    original = real_json.dumps

    def _counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    # The module imports `import json` at module load -- patch the bound
    # name on the module itself so the patched version is used by save().
    monkeypatch.setattr(rgc.json, "dumps", _counted)
    return calls


def test_no_drop_path_calls_dumps_once(tmp_path, dumps_counter):
    store = _make_fake_store(tmp_path)
    assignment = _make_assignment(centroids_count=5, mid_regions_count=3)
    node_payload = _make_node_payload(count=10)
    ok = rgc.save(store, assignment, [], node_payload=node_payload, max_degree=4)
    assert ok is True
    assert len(dumps_counter) == 1, f"json.dumps called {len(dumps_counter)} times (expected 1)"
    # Saved file exists + parses + node_payload survived
    cache_path = pathlib.Path(tmp_path) / rgc.CACHE_FILENAME
    assert cache_path.exists()
    written = _decrypt_cache_for_inspection(_make_fake_store(tmp_path), cache_path)
    assert written["node_payload"], "node_payload should not have been dropped on the no-drop fast path"
    assert "community_centroids" in written["assignment"]


def test_oversize_drops_node_payload_first(tmp_path, dumps_counter, monkeypatch):
    # Force a tiny cap so node_payload alone overflows.
    monkeypatch.setattr(rgc, "MAX_CACHE_BYTES", 50_000)
    store = _make_fake_store(tmp_path)
    assignment = _make_assignment(centroids_count=2, mid_regions_count=2)
    node_payload = _make_node_payload(count=20)  # 20 * 10240 ~ 200k > 50k
    ok = rgc.save(store, assignment, [], node_payload=node_payload, max_degree=4)
    assert ok is True
    assert len(dumps_counter) == 1, "json.dumps must be called exactly once across the drop loop"
    cache_path = pathlib.Path(tmp_path) / rgc.CACHE_FILENAME
    written = _decrypt_cache_for_inspection(_make_fake_store(tmp_path), cache_path)
    assert written["node_payload"] == {}, "node_payload should have been dropped"
    # community_centroids should still be intact -- only the first drop fired
    assert written["assignment"]["community_centroids"], "community_centroids must survive when node_payload drop alone is sufficient"


def test_oversize_drops_centroids_when_node_payload_drop_insufficient(tmp_path, dumps_counter, monkeypatch):
    # Cap small enough that even after node_payload drop, centroids alone overflow.
    monkeypatch.setattr(rgc, "MAX_CACHE_BYTES", 50_000)
    store = _make_fake_store(tmp_path)
    # 20 centroids * 9472 = ~190k > 50k cap -- forces step-2 drop
    assignment = _make_assignment(centroids_count=20, mid_regions_count=2)
    ok = rgc.save(store, assignment, [], node_payload=None, max_degree=4)
    assert ok is True
    assert len(dumps_counter) == 1, "json.dumps must be called exactly once"
    cache_path = pathlib.Path(tmp_path) / rgc.CACHE_FILENAME
    written = _decrypt_cache_for_inspection(_make_fake_store(tmp_path), cache_path)
    assert written["assignment"]["community_centroids"] == {}
    # mid_regions should still be intact -- step-3 drop should NOT have fired
    assert written["assignment"]["mid_regions"]


def test_returns_false_when_all_drops_insufficient(tmp_path, dumps_counter, monkeypatch):
    # Cap so small that even an empty assignment + empty node_payload would fail
    monkeypatch.setattr(rgc, "MAX_CACHE_BYTES", 100)  # well below _BASE_SCAFFOLD_BYTES = 4096
    store = _make_fake_store(tmp_path)
    assignment = _make_assignment(centroids_count=10, mid_regions_count=10)
    node_payload = _make_node_payload(count=5)
    ok = rgc.save(store, assignment, [], node_payload=node_payload, max_degree=4)
    assert ok is False
    assert len(dumps_counter) == 0, f"json.dumps called {len(dumps_counter)} times when all drops insufficient (expected 0)"
    cache_path = pathlib.Path(tmp_path) / rgc.CACHE_FILENAME
    assert not cache_path.exists(), "no cache file should be written on give-up path"


def test_estimator_overshoots_actual_dumps_size():
    # Test 5 builds `data` directly (no save() round-trip) and json.dumps it.
    # CommunityAssignment is not JSON-serialisable, so we encode it via the
    # same path save() takes (`_encode_assignment`) before placing it under
    # `data["assignment"]` — this is exactly the dict shape `save()` ends up
    # encoding at the end of its drop loop.
    encoded_assignment = rgc._encode_assignment(_make_assignment(centroids_count=5, mid_regions_count=5))
    data = {
        "cache_version": rgc.CACHE_VERSION,
        "key": [10, 5, 4, 384, rgc.CACHE_VERSION],
        "assignment": encoded_assignment,
        "rich_club": [f"u{i}" for i in range(10)],
        "node_payload": _make_node_payload(count=10),
        "max_degree": 6,
        "saved_at": "2026-04-29T13:00:00+00:00",
    }
    actual = len(real_json.dumps(data).encode("utf-8"))
    estimate = rgc._estimate_serialised_bytes(data)
    assert estimate >= actual, f"estimator must overshoot: estimate={estimate} actual={actual}"


def test_d11_stale_comment_fixed():
    src = pathlib.Path("src/iai_mcp/runtime_graph_cache.py").read_text()
    assert "1024-dim" not in src, "stale 1024-dim comment must be removed"
    assert "384-dim" in src, "384-dim must replace stale 1024-dim comment"
