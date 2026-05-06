"""Plan 05-13 RED scaffold — cached centrality on graph nodes.

``build_runtime_graph`` must compute betweenness centrality ONCE and
attach it as the ``centrality`` NetworkX node attribute so the rank
stage can read it O(1) instead of recomputing ``graph.centrality()``
on every recall. The cache file must round-trip the per-node
centrality alongside the rest of the node payload so a cold-start
rebuild hits the cache and the pipeline-hot-path stays allocation-free.

Contracts:
    C1 — every graph node has a ``centrality`` float attribute after
         ``build_runtime_graph`` returns.
    C2 — runtime_graph_cache round-trips the ``centrality`` value per node
         (save + try_load preserves the exact float).
    C3 — when a node is missing ``centrality`` (pre-05-13 graph / race),
         recall_for_response falls back to inline computation without crashing.
    C4 — CACHE_VERSION bumped from "05-12-v1" to "05-13-v1"; legacy cache
         files are invalidated cleanly.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp import retrieve, runtime_graph_cache
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


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


def _make_record(store: MemoryStore, text: str, seed: int) -> MemoryRecord:
    import numpy as np
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(store.embed_dim).astype(np.float32)
    v /= float(np.linalg.norm(v)) or 1.0
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=v.tolist(),
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


@pytest.fixture
def seeded_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(path=tmp_path / "lancedb")
    store.root = tmp_path
    # Seed enough records to produce a non-trivial graph so betweenness > 0
    # on at least some nodes.
    for i in range(15):
        store.insert(_make_record(store, f"fact-{i}", i + 1))
    # Create some edges so betweenness has something to measure.
    records = list(store.all_records())
    ids = [r.id for r in records]
    pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    pairs += [(ids[0], ids[5]), (ids[2], ids[10])]
    store.boost_edges(pairs, delta=0.5)
    return store


# --------------------------------------------------------------- C1


def test_C1_every_node_has_centrality_attr(seeded_store):
    """After build_runtime_graph, every node carries a 'centrality' float attr."""
    graph, _a, _rc = retrieve.build_runtime_graph(seeded_store)
    assert len(graph._nx.nodes) > 0
    for nid in graph._nx.nodes:
        node = graph._nx.nodes[nid]
        assert "centrality" in node, f"node {nid} missing centrality attr"
        assert isinstance(node["centrality"], float), (
            f"centrality on {nid} must be float, got {type(node['centrality'])}"
        )


# --------------------------------------------------------------- C2


def test_C2_cache_round_trips_centrality(seeded_store):
    """save + try_load preserves per-node centrality exactly."""
    graph, assignment, rich_club = retrieve.build_runtime_graph(seeded_store)

    # Snapshot centrality from the live graph.
    live_cent = {
        nid: float(graph._nx.nodes[nid]["centrality"])
        for nid in graph._nx.nodes
    }

    # Force a fresh save by invalidating then re-running build.
    runtime_graph_cache.invalidate(seeded_store)
    graph2, _a2, _rc2 = retrieve.build_runtime_graph(seeded_store)

    # Now cache should be populated. try_load should give us node_payload
    # with centrality baked in.
    cached = runtime_graph_cache.try_load(seeded_store)
    assert cached is not None, "cache should be populated after build"
    # try_load returns 4-tuple (max_degree appended).
    _assignment, _rich_club, node_payload, _max_degree = cached
    assert node_payload is not None and len(node_payload) > 0

    for nid, live in live_cent.items():
        payload = node_payload.get(nid)
        assert payload is not None, f"missing payload for {nid}"
        assert "centrality" in payload, f"payload {nid} missing centrality"
        # Exact-float equality — JSON round-trip preserves float64.
        assert abs(payload["centrality"] - live) < 1e-9, (
            f"centrality drift on {nid}: cache={payload['centrality']} "
            f"live={live}"
        )


# --------------------------------------------------------------- C3


def test_C3_missing_centrality_fallback_inline(seeded_store):
    """Graph with missing 'centrality' on nodes must not crash rank stage."""
    from iai_mcp import pipeline

    class _E:
        DIM = seeded_store.embed_dim
        DEFAULT_DIM = seeded_store.embed_dim
        DEFAULT_MODEL_KEY = "t"

        def embed(self, t):
            import numpy as np
            import hashlib
            rng = np.random.default_rng(
                int(hashlib.sha256(t.encode()).hexdigest()[:16], 16)
            )
            v = rng.standard_normal(self.DIM).astype(np.float32)
            v /= float(np.linalg.norm(v)) or 1.0
            return v.tolist()

    graph, assignment, rich_club = retrieve.build_runtime_graph(seeded_store)
    # Strip centrality from all nodes — simulates a pre-05-13 graph shape
    # or a race in _graph_sync_hook.
    for nid in list(graph._nx.nodes):
        graph._nx.nodes[nid].pop("centrality", None)

    resp = pipeline.recall_for_response(
        store=seeded_store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=_E(), cue="fact-3",
        session_id="t-C3", budget_tokens=4000,
    )
    # No crash; still returns hits.
    assert resp is not None
    assert isinstance(resp.hits, list)


# --------------------------------------------------------------- C4


def test_C4_cache_version_bumped_to_05_13_v1():
    """CACHE_VERSION moved forward over the cache-shape evolution (05-12-v1
    -> 05-13-v1 -> 06-02-v1 -> 07-09-v3, with W3 / wrapping
    the file in AES-256-GCM). Legacy files invalidate cleanly on version
    mismatch (and the legacy plaintext-shape "06-02-v1" lazy-migrates to
    the encrypted shape on first warm-start under 07.9).
    """
    assert runtime_graph_cache.CACHE_VERSION == "07-09-v3"


def test_C4_legacy_cache_invalidated(seeded_store, tmp_path: Path):
    """A cache file written with CACHE_VERSION=05-12-v1 must NOT load.

    W3: the on-disk format is now AES-256-GCM-wrapped. Decrypt
    the file, mutate cache_version, re-encrypt, then assert try_load
    rejects the stale version cleanly.
    """
    from iai_mcp.crypto import decrypt_field, encrypt_field

    # First build the graph so we know the path.
    graph, assignment, rich_club = retrieve.build_runtime_graph(seeded_store)
    cache_path = tmp_path / "runtime_graph_cache.json"
    assert cache_path.exists(), "cache not created by build_runtime_graph"

    # Decrypt → mutate version → re-encrypt round-trip.
    key = runtime_graph_cache._cache_encryption_key(seeded_store)
    raw_text = cache_path.read_text(encoding="utf-8")
    plaintext = decrypt_field(raw_text, key, runtime_graph_cache._CACHE_AAD)
    raw = json.loads(plaintext)
    raw["cache_version"] = "05-12-v1"
    new_ct = encrypt_field(json.dumps(raw), key, runtime_graph_cache._CACHE_AAD)
    cache_path.write_text(new_ct, encoding="ascii")

    # try_load must reject it (legacy version stamp).
    assert runtime_graph_cache.try_load(seeded_store) is None
