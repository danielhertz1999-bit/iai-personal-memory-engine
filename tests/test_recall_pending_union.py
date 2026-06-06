"""QUAL-02: just-written embedding_pending record surfaces via the bounded
recency union (recent_pending_markers, NOT all_records).

Hermetic: monkeypatched HOME/IAI_MCP_STORE/IAI_DAEMON_SOCKET_PATH to tmp.
No live ~/.iai-mcp, no daemon stop/restart.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _monkeypatch_env(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    monkeypatch.setenv("IAI_MCP_RECALL_SAMPLE_RATE", "1.0")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_pending_marker_surfaces_via_bounded_recency_union(tmp_path, monkeypatch):
    """A just-written embedding_pending record surfaces in recall via the
    ANN-union-recency path (recent_pending_markers), even though it is
    excluded from query_similar by the WHERE clause.

    Assertions:
    - The pending record appears in the recall hit set.
    - store.all_records() is NOT called during recall (assert via monkeypatch
      that replaces all_records with a function that raises, so if it fires
      the test fails).
    - store.recent_pending_markers IS called (spy confirms).
    """
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "pending-union-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    # Insert a few normal records so the store is non-trivial.
    for i in range(20):
        rec = _make(text=f"User filler record {i}", vec=_random_vec(1000 + i))
        store.insert(rec)

    # Insert the pending record: embedding_pending=1, no real embedding.
    # Use hippo.insert_pending_row() which sets embedding_pending=1 directly
    # in SQLite (normal store.insert() does not set this flag).
    pending_text = "User pending marker just written test content"
    pending_id = UUID(int=999_001)
    now = datetime.now(timezone.utc)
    store.db.insert_pending_row(
        record_id=str(pending_id),
        tier="episodic",
        literal_surface=pending_text,
        provenance_json="[]",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        tags_json="[]",
    )

    # Verify it was stored as pending via direct SQL (sanity check).
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT embedding_pending FROM records WHERE id = ?",
            (str(pending_id),),
        ).fetchone()
    assert row is not None, "pending record not stored"
    assert row[0] == 1, f"record not stored as pending: embedding_pending={row[0]}"

    # Verify ANN excludes it (cosine ~0 to any cue → not in top-K).
    ann_results = store.query_similar([0.1] * EMBED_DIM, k=200)
    ann_ids = {r.id for r, _ in ann_results}
    assert UUID(int=999_001) not in ann_ids, (
        "pending record should be excluded from query_similar"
    )

    # Now assert all_records is NOT called during recall.
    all_records_call_count = {"n": 0}
    original_all_records = store.all_records

    def _spy_all_records(*args, **kwargs):
        all_records_call_count["n"] += 1
        raise AssertionError(
            "all_records() called during recall — QUAL-02 requires recent_pending_markers only"
        )

    monkeypatch.setattr(store, "all_records", _spy_all_records)

    # Spy on recent_pending_markers.
    pending_markers_call_count = {"n": 0}
    original_rpm = store.recent_pending_markers

    def _spy_rpm(n=50):
        pending_markers_call_count["n"] += 1
        return original_rpm(n=n)

    monkeypatch.setattr(store, "recent_pending_markers", _spy_rpm)

    # Warm cache prime (off-path, untimed) so load_recall_structural is a HIT.
    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc

    _graph, _assignment, _rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, _assignment, _rc)

    # Run recall via recall_for_response (same path as production).
    from iai_mcp.embed import Embedder
    from iai_mcp.pipeline import recall_for_response
    import iai_mcp.pipeline as _pipeline_mod

    _pipeline_mod._last_recall_latency_ms = 0.0
    embedder = Embedder()

    response = recall_for_response(
        store=store,
        graph=_graph,
        assignment=_assignment,
        rich_club=_rc,
        embedder=embedder,
        cue="User pending marker just written",
        session_id="test-session",
        budget_tokens=2000,
        mode="concept",
    )

    # Assert the pending record appears in the hit set.
    hit_ids = {h.record_id for h in response.hits}
    assert UUID(int=999_001) in hit_ids, (
        f"pending record not surfaced via recency union; hit_ids={hit_ids}"
    )

    # Assert all_records was NOT called.
    assert all_records_call_count["n"] == 0, (
        f"all_records() was called {all_records_call_count['n']} time(s) — "
        "QUAL-02 recency union must use recent_pending_markers only"
    )

    # Assert recent_pending_markers WAS called.
    assert pending_markers_call_count["n"] >= 1, (
        "recent_pending_markers() was not called — "
        "QUAL-02 union did not fire"
    )


def test_pending_union_does_not_double_count_ranked_pending(tmp_path, monkeypatch):
    """If a record is both in the ANN result set AND pending (edge-case),
    it must appear exactly once in the final hit set.
    """
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "dedup-test-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    # Insert normal records.
    for i in range(10):
        rec = _make(text=f"User dedup test filler {i}", vec=_random_vec(2000 + i))
        store.insert(rec)

    # Insert a pending record via the pending-row path.
    pending_id = UUID(int=888_001)
    now = datetime.now(timezone.utc)
    store.db.insert_pending_row(
        record_id=str(pending_id),
        tier="episodic",
        literal_surface="User dedup pending test",
        provenance_json="[]",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        tags_json="[]",
    )

    # Off-path cache prime.
    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc
    _g, _a, _rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, _a, _rc)

    from iai_mcp.embed import Embedder
    from iai_mcp.pipeline import recall_for_response
    import iai_mcp.pipeline as _pipeline_mod

    _pipeline_mod._last_recall_latency_ms = 0.0
    embedder = Embedder()

    response = recall_for_response(
        store=store,
        graph=_g,
        assignment=_a,
        rich_club=_rc,
        embedder=embedder,
        cue="User dedup pending test",
        session_id="test-session",
        budget_tokens=2000,
        mode="concept",
    )

    hit_ids = [h.record_id for h in response.hits]
    # No duplicate record ids.
    assert len(hit_ids) == len(set(hit_ids)), (
        f"duplicate record_ids in hits: {hit_ids}"
    )
