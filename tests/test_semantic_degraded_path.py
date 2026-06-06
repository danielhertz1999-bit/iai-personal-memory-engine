"""Tests for the semantic recall + embedder home and the warm-pollution guard.

Covers semantic hard-fail when the consolidation daemon is stopped:

  semantic memory_recall returns store-backed hits in ≤1.5 s when a warm
        embedder is reachable (daemon alive/WAKE, embed RPC available).
  when no warm embedder is available (daemon process dead), memory_recall
        returns a functional STORE-backed degraded result (recency/temporal direct),
        NOT empty, NOT a crash, NOT a bank substring scan.
  a pending row (embedding_pending=1, zero-vector BLOB) must NEVER
        appear as a semantic/cosine/graph candidate, but MUST appear in the recency
        path (all_records/recent_user_turns stay pending-inclusive).

embed_cue routing: embed_cue MUST be in CONTROL_MSG_TYPES so the socket layer
        forwards it to _dispatch_socket_request.
"""
from __future__ import annotations

import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_vector_blob(embed_dim: int) -> bytes:
    """Return an embed_dim zero-vector as a BLOB (float32 little-endian)."""
    return struct.pack(f"<{embed_dim}f", *([0.0] * embed_dim))


def _make_normal_record(text: str, seed: int = 42):
    """Return a MemoryRecord with a real (seeded random) embedding."""
    import numpy as np
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    rng = np.random.RandomState(seed=seed)
    vec = rng.randn(EMBED_DIM).tolist()
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": "test-session", "role": "user"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["role:user"],
        language="en",
    )


# ---------------------------------------------------------------------------
# Test 1: warm embedder, store-backed hits in ≤1.5 s
# ---------------------------------------------------------------------------


def test_semantic_warm_embedder_returns_store_hits(
    hermetic_store: Path, monkeypatch
) -> None:
    """Semantic recall returns store-backed hits in ≤1.5 s with a warm embedder.

    Seeds the store with a distinctive turn, then calls the semantic recall path
    with the embedder funnel stubbed (so the construct is deterministic + fast in
    the hermetic env — no real model load, no network) and asserts store-backed
    hits are returned within the 1.5 s SLO.

    The fake embedder returns the SAME vector as the seeded record so the
    daemon-independent ANN/structural path surfaces it.
    """
    import time
    import numpy as np
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM
    import iai_mcp.embed as _embed_mod

    from iai_mcp.semantic_recall import recall_semantic_warm

    # The seeded record's embedding (seed=10) — the fake embedder returns the
    # same vector so the cue lands on the seeded record.
    seed_vec = np.random.RandomState(seed=10).randn(EMBED_DIM).tolist()

    store = MemoryStore(hermetic_store)
    try:
        rec = _make_normal_record("warm semantic probe distinctive text", seed=10)
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    class _FakeEmbedder:
        DIM = EMBED_DIM

        def embed(self, text: str) -> list:
            return list(seed_vec)

    # Stub the single embedder funnel so the construct is fast + hermetic.
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: _FakeEmbedder())

    t0 = time.monotonic()
    hits = recall_semantic_warm(
        store_root=hermetic_store,
        cue="warm semantic probe distinctive text",
        n=5,
    )
    elapsed = time.monotonic() - t0

    assert elapsed <= 1.5, f"warm semantic recall took {elapsed:.3f} s (SLO ≤1.5 s)"
    assert hits, "warm semantic recall returned no hits (must be store-backed)"
    surfaces = [h.get("literal_surface", "") or "" for h in hits]
    assert any("warm semantic probe" in s for s in surfaces), (
        "distinctive seeded turn not in warm semantic hits"
    )


# ---------------------------------------------------------------------------
# Test 2: no warm embedder, STORE-backed degraded result (not bank scan)
# ---------------------------------------------------------------------------


def test_semantic_no_embedder_degrades_not_empty(hermetic_store: Path) -> None:
    """With no warm embedder, recall degrades to STORE-backed result (not bank, not empty).

    Seeds a distinctive turn ONLY in the tmp store (not in bank/live), then calls
    the semantic recall path with the daemon down and asserts:
    (1) no crash;
    (2) result is not empty;
    (3) result is store-backed — the distinctive turn from the tmp store appears
        (bank cannot produce it; the live layer cannot produce it).
    """
    from iai_mcp.store import MemoryStore, flush_record_buffer

    # Import the not-yet-existing degraded store path.
    from iai_mcp.semantic_recall import recall_semantic_degraded  # type: ignore[import]

    store = MemoryStore(hermetic_store)
    try:
        rec = _make_normal_record("degraded store backed probe text store only", seed=20)
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    # Daemon is down (hermetic_store fixture already sets a dead socket path).
    hits = recall_semantic_degraded(
        store_root=hermetic_store,
        cue="degraded store backed probe",
        n=5,
    )

    assert hits is not None, "degraded recall must return a result, not None"
    assert len(hits) > 0, (
        "degraded recall must return a non-empty result "
        "(recency/temporal direct from store), not empty"
    )
    surfaces = [h.get("literal_surface", "") or "" for h in hits]
    assert any("degraded store backed probe" in s for s in surfaces), (
        "degraded result must be STORE-backed "
        "(bank cannot produce this turn — it was seeded only in the tmp store)"
    )


# ---------------------------------------------------------------------------
# Test 3: pending row never pollutes warm semantic candidates
# ---------------------------------------------------------------------------


def test_pending_row_excluded_from_warm_semantic_until_reembed(
    hermetic_store: Path,
) -> None:
    """Pending row excluded from warm semantic, IS a recency hit.

    Constructs a store with:
    - one normal fully-embedded row (the 'normal' row)
    - one pending row (embedding_pending=1, zero-vector BLOB, text='pending probe text')

    Then asserts:
    (1) the pending row NEVER appears in the WARM semantic/cosine/graph candidate
        set — it must not be a graph node with a zero-vector, nor a query_similar hit;
    (2) the SAME pending row IS returned by the recency path (all_records /
        recent_user_turns) — recency is embedding-independent, pending-inclusive;
    (3) AFTER a simulated re-embed pass (BLOB filled, flag cleared), the row CAN
        appear as a warm semantic hit.

    The test drives the REAL retrieve.py build_runtime_graph MISS path (the
    load-bearing graph-candidate source) — NOT only query_similar — so a leak
    through the graph path is caught.
    """
    import sqlite3 as _sqlite3

    from iai_mcp.types import EMBED_DIM
    from iai_mcp.store import MemoryStore, flush_record_buffer

    # Seed: one normal row + inject one pending row.
    store = MemoryStore(hermetic_store)
    try:
        rec_normal = _make_normal_record("normal embedded row text", seed=30)
        store.insert(rec_normal)
        flush_record_buffer(store)
    finally:
        store.close()

    # Inject pending row directly into SQLite.
    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    pending_id = str(uuid.uuid4())
    zero_blob = _zero_vector_blob(EMBED_DIM)
    now = datetime.now(timezone.utc).isoformat()

    conn = _sqlite3.connect(str(db_path))
    conn.row_factory = _sqlite3.Row
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
        if "embedding_pending" not in cols:
            conn.execute(
                "ALTER TABLE records ADD COLUMN embedding_pending INTEGER NOT NULL DEFAULT 0"
            )
            conn.commit()
        conn.execute(
            "INSERT INTO records "
            "(id, tier, literal_surface, aaak_index, embedding, embedding_pending, "
            " created_at, updated_at, hv_tier, structure_hv_payload) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?, 'bsc', x'')",
            (pending_id, "episodic", "pending probe text", "", zero_blob, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    # (1) Pending row must NOT appear in warm semantic / graph candidates.
    # Drive the REAL build_runtime_graph MISS path.
    store2 = MemoryStore(hermetic_store)
    try:
        from iai_mcp.retrieve import build_runtime_graph  # type: ignore[import]
        graph, _assignment, _rich_club = build_runtime_graph(store2)

        # Assert: the pending record's UUID is NOT a graph node.
        from uuid import UUID as _UUID
        pending_uuid = _UUID(pending_id)
        # MemoryGraph.nodes returns the set of node UUIDs.
        graph_nodes = set(graph.nodes())
        assert pending_uuid not in graph_nodes, (
            "pending row (embedding_pending=1) must NOT be a graph node; "
            "its zero-vector would pollute cosine/graph candidates"
        )

        # Also assert query_similar does not return the pending row.
        import numpy as np
        rng = np.random.RandomState(seed=30)
        cue_vec = rng.randn(EMBED_DIM).tolist()  # same seed as normal row → maximally similar
        similar = store2.query_similar(cue_vec, n=10)
        similar_ids = {str(r.id) for r in similar}
        assert pending_id not in similar_ids, (
            "pending row must not appear in query_similar results "
            "(zero-vector must be excluded from ANN candidates)"
        )

        # (2) Pending row IS returned by recency path (pending-inclusive).
        turns = store2.all_records()
        all_ids = {str(r.id) for r in turns}
        assert pending_id in all_ids, (
            "pending row must still be in all_records() — "
            "recency is embedding-independent; pending-inclusive"
        )

        recent = store2.recent_user_turns(n=20)
        recent_ids = {str(r.id) for r in recent}
        assert pending_id in recent_ids, (
            "pending row must appear in recent_user_turns() — "
            "recency must not filter by embedding_pending"
        )

    finally:
        store2.close()

    # (3) After re-embed: pending row CAN appear as a warm semantic hit.
    # Simulate the daemon re-embed by writing a real embedding + clearing the flag.
    import numpy as np
    rng2 = np.random.RandomState(seed=31)
    real_vec = rng2.randn(EMBED_DIM).tolist()
    real_blob = struct.pack(f"<{EMBED_DIM}f", *real_vec)

    conn3 = _sqlite3.connect(str(db_path))
    try:
        conn3.execute(
            "UPDATE records SET embedding = ?, embedding_pending = 0 WHERE id = ?",
            (real_blob, pending_id),
        )
        conn3.commit()
    finally:
        conn3.close()

    # Reopen and confirm the row is now a graph node (post re-embed).
    store3 = MemoryStore(hermetic_store)
    try:
        from iai_mcp.retrieve import build_runtime_graph as _brg
        graph3, _, _ = _brg(store3)
        graph_nodes3 = set(graph3.nodes())
        from uuid import UUID as _UUID2
        assert _UUID2(pending_id) in graph_nodes3, (
            "after re-embed (embedding_pending=0), row must appear in "
            "the runtime graph as a warm semantic candidate"
        )
    finally:
        store3.close()


# ---------------------------------------------------------------------------
# Test 4: embed_cue routing — CONTROL_MSG_TYPES membership + dispatch
# ---------------------------------------------------------------------------


def test_embed_cue_in_control_msg_types() -> None:
    """embed_cue MUST be in CONTROL_MSG_TYPES.

    Without this membership, the socket layer never forwards embed_cue
    messages to _dispatch_socket_request and the RPC is silently dropped.
    """
    from iai_mcp.socket_server import SocketServer

    assert "embed_cue" in SocketServer.CONTROL_MSG_TYPES, (
        "embed_cue must be in SocketServer.CONTROL_MSG_TYPES for the socket "
        "layer to forward it to _dispatch_socket_request"
    )


def test_embed_cue_dispatch_warm_stub(hermetic_store: Path) -> None:
    """embed_cue dispatches to a 384-d embedding with a stubbed embedder.

    Tests the dispatch handler directly (bypasses the socket) with a
    monkeypatched Embedder that returns a deterministic 384-d vector.
    Asserts:
    - ok=True when the embedder is ready.
    - embedding has len == 384 (dim validation).
    - ok=False with reason=daemon_not_ready when the embedder raises.
    """
    import asyncio
    from unittest.mock import patch

    from iai_mcp.store import MemoryStore
    from iai_mcp.concurrency import _dispatch_socket_request

    store = MemoryStore(hermetic_store)
    try:
        # The embed_cue handler does NOT use state — it is just required by
        # the dispatch signature.
        state: dict = {}

        # Stub the embedder to return a deterministic 384-d vector.
        import numpy as np
        fake_vec = np.random.RandomState(42).randn(384).tolist()

        class _FakeEmbedder:
            DIM = 384
            def embed(self, text: str) -> list:
                return list(fake_vec)

        with patch("iai_mcp.embed.Embedder", return_value=_FakeEmbedder()):
            result = asyncio.run(
                _dispatch_socket_request(
                    {"type": "embed_cue", "cue": "test cue for embed"},
                    store,
                    state,
                )
            )

        assert result.get("ok") is True, f"embed_cue dispatch must return ok=True when warm; got {result}"
        embedding = result.get("embedding")
        assert isinstance(embedding, list), "embedding must be a list"
        assert len(embedding) == 384, f"embedding must be 384-d, got {len(embedding)}"

        # Test daemon_not_ready path: embedder raises.
        class _FailEmbedder:
            DIM = 384
            def embed(self, text: str) -> list:
                raise RuntimeError("embedder not ready")

        with patch("iai_mcp.embed.Embedder", return_value=_FailEmbedder()):
            result_fail = asyncio.run(
                _dispatch_socket_request(
                    {"type": "embed_cue", "cue": "test"},
                    store,
                    state,
                )
            )

        assert result_fail.get("ok") is False, "embed_cue must return ok=False when embedder fails"
        assert result_fail.get("reason") == "daemon_not_ready", (
            f"reason must be 'daemon_not_ready', got {result_fail.get('reason')!r}"
        )
    finally:
        store.close()
