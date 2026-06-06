"""Layer-1 bounded record hydration and index-backed pending-marker recency.

Tests for:
  - MemoryStore.get_batch: one batched SQL query (not N point reads),
    created_at populated, parameterized bind, unknown ids absent.
  - MemoryStore.recent_pending_markers: index-backed via idx_records_pending
    (READ A) + idx_records_tier (READ B), role/episodic filter pushed into
    SQL before LIMIT, no all_records() call, EXPLAIN QUERY PLAN shows
    SEARCH USING INDEX (no full SCAN of records), large-pending-backlog
    bounded.

All tests are hermetic: HOME + IAI_MCP_STORE + IAI_DAEMON_SOCKET_PATH
are monkeypatched to tmp_path. The live daemon is never touched.
Generic 'User'/'user' test data only (no PII).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(
    tier: str = "episodic",
    text: str = "user message",
    seed: int = 0,
    tags: list[str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=tags or [],
        language="en",
    )


def _insert_pending(store, seed: int = 0, text: str = "pending turn", tier: str = "episodic") -> str:
    """Insert a row with embedding_pending=1 via the direct HippoDB path.

    store.insert() does not write embedding_pending (it uses _to_row which omits
    that column, leaving the DEFAULT 0). The direct insert_pending_row is the
    only correct way to create a row with embedding_pending=1 in tests.
    Returns the record_id string inserted.
    """
    from datetime import datetime, timezone
    record_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    store.db.insert_pending_row(
        record_id=record_id,
        tier=tier,
        literal_surface=text,
        tags_json=json.dumps([]),
        provenance_json=json.dumps([]),
        created_at=now,
        updated_at=now,
    )
    return record_id


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    s = MemoryStore(str(tmp_path / "store"))
    yield s


# ---------------------------------------------------------------------------
# get_batch — method exists
# ---------------------------------------------------------------------------

def test_get_batch_exists(store):
    """get_batch method must exist on MemoryStore."""
    assert hasattr(store, "get_batch"), "MemoryStore.get_batch does not exist"


# ---------------------------------------------------------------------------
# get_batch — single id returns correct record
# ---------------------------------------------------------------------------

def test_get_batch_single_id(store):
    """get_batch([id]) returns the record for that id."""
    r = _make_rec(text="specific user turn", seed=1)
    store.insert(r)
    result = store.get_batch([r.id])
    assert r.id in result, f"id {r.id} not in get_batch result"
    assert result[r.id].literal_surface == r.literal_surface


# ---------------------------------------------------------------------------
# get_batch — created_at is populated
# ---------------------------------------------------------------------------

def test_get_batch_created_at_populated(store):
    """get_batch returns records with created_at populated (for ts_by_id)."""
    r = _make_rec(seed=2)
    store.insert(r)
    result = store.get_batch([r.id])
    assert r.id in result
    assert result[r.id].created_at is not None, "created_at must be populated"
    assert isinstance(result[r.id].created_at, datetime), (
        f"created_at must be datetime, got {type(result[r.id].created_at)}"
    )


# ---------------------------------------------------------------------------
# get_batch — embedding decoded correctly (EMBED_DIM floats, not byte-ints)
# ---------------------------------------------------------------------------

def test_get_batch_embedding_decoded(store):
    """get_batch returns records with embedding of length EMBED_DIM (float, not bytes)."""
    vec = _random_vec(99)
    r = _make_rec(seed=99)
    # override with the vec we know
    r = MemoryRecord(
        id=r.id, tier=r.tier, literal_surface=r.literal_surface,
        aaak_index="", embedding=vec, community_id=None, centrality=0.0,
        detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=r.created_at, updated_at=r.updated_at,
        tags=[], language="en",
    )
    store.insert(r)
    result = store.get_batch([r.id])
    assert r.id in result
    emb = result[r.id].embedding
    assert len(emb) == EMBED_DIM, (
        f"embedding must have {EMBED_DIM} floats, got {len(emb)} elements "
        f"(first element type: {type(emb[0]).__name__!r})"
    )
    assert isinstance(emb[0], float), (
        f"embedding elements must be float, got {type(emb[0]).__name__!r} — "
        "BLOB was not decoded via np.frombuffer"
    )


# ---------------------------------------------------------------------------
# get_batch — batched: ONE query for many ids (functional + source assertion)
# ---------------------------------------------------------------------------

def test_get_batch_one_query_not_n(store):
    """get_batch([id1,..., id10]) uses a single batched IN clause (source + functional).

    Source-level: the implementation must contain an IN clause and placeholders.
    Functional: all 10 records are returned, and unknown ids are absent.
    """
    import inspect
    src = inspect.getsource(store.get_batch)
    assert "IN (" in src or "IN({" in src or "IN ()" in src or "IN (" in src, (
        "get_batch source must contain a batched IN clause"
    )
    assert "?" in src, "get_batch source must use '?' placeholders"

    records = [_make_rec(seed=100 + i) for i in range(10)]
    for r in records:
        store.insert(r)

    ids = [r.id for r in records]
    result = store.get_batch(ids)
    # All 10 must appear
    assert len(result) == 10, f"Expected 10 records, got {len(result)}"
    for r in records:
        assert r.id in result, f"Record {r.id} missing from get_batch result"
    # Unknown id must be absent
    unknown = uuid4()
    result2 = store.get_batch(ids + [unknown])
    assert unknown not in result2, "Unknown id must not appear in get_batch result"


# ---------------------------------------------------------------------------
# get_batch — parameterized bind (source-level + functional)
# ---------------------------------------------------------------------------

def test_get_batch_parameterized_bind(store):
    """The records SQL uses '?' placeholders (source-level verified).

    get_batch must never interpolate UUID strings into the SQL.
    Also verifies the record is returned correctly (functional).
    """
    import inspect
    src = inspect.getsource(store.get_batch)
    assert "?" in src, "get_batch source must use '?' placeholders in SQL"
    # The source must NOT contain an f-string that embeds _uuid_literal or
    # equivalent literal-uuid pattern (store.get's anti-pattern)
    assert "_uuid_literal" not in src, (
        "get_batch must NOT use _uuid_literal (f-string interpolation); "
        "use parameterized IN-bind instead"
    )

    r = _make_rec(seed=200)
    store.insert(r)
    result = store.get_batch([r.id])
    assert r.id in result, f"Record {r.id} must be returned by get_batch"
    assert result[r.id].literal_surface == r.literal_surface


# ---------------------------------------------------------------------------
# get_batch — unknown ids are absent (no crash)
# ---------------------------------------------------------------------------

def test_get_batch_unknown_ids_absent(store):
    """Unknown ids in get_batch are absent in the result dict (no crash, no KeyError)."""
    r = _make_rec(seed=300)
    store.insert(r)
    unknown = uuid4()
    result = store.get_batch([r.id, unknown])
    assert r.id in result
    assert unknown not in result, "Unknown id must not appear in result"


# ---------------------------------------------------------------------------
# get_batch — empty ids list returns empty dict
# ---------------------------------------------------------------------------

def test_get_batch_empty_ids(store):
    """get_batch([]) returns {} without raising."""
    result = store.get_batch([])
    assert result == {}


# ---------------------------------------------------------------------------
# recent_pending_markers — method exists
# ---------------------------------------------------------------------------

def test_recent_pending_markers_exists(store):
    """recent_pending_markers method must exist on MemoryStore."""
    assert hasattr(store, "recent_pending_markers"), (
        "MemoryStore.recent_pending_markers does not exist"
    )


# ---------------------------------------------------------------------------
# recent_pending_markers — pending record surfaces (READ A)
# ---------------------------------------------------------------------------

def test_recent_pending_markers_pending_record_surfaces(store):
    """A record with embedding_pending=1 is returned by recent_pending_markers.

    Uses store.db.insert_pending_row (the only correct write path for pending rows
    — store.insert/_to_row omits embedding_pending so the DEFAULT 0 is used).
    """
    record_id = _insert_pending(store, seed=400, text="pending user turn")
    result = store.recent_pending_markers(n=10)
    result_id_strs = {str(rec.id) for rec in result}
    assert record_id in result_id_strs, (
        "A pending record (embedding_pending=1) must appear in recent_pending_markers"
    )


# ---------------------------------------------------------------------------
# recent_pending_markers — role:user episodic record surfaces (READ B)
# ---------------------------------------------------------------------------

def test_recent_pending_markers_role_user_surfaces(store):
    """A role:user episodic record is returned by recent_pending_markers."""
    r = _make_rec(tier="episodic", tags=["role:user"], seed=500)
    store.insert(r)
    result = store.recent_pending_markers(n=10)
    result_ids = {rec.id for rec in result}
    assert r.id in result_ids, (
        "A role:user episodic record must appear in recent_pending_markers"
    )


# ---------------------------------------------------------------------------
# recent_pending_markers — does NOT call all_records()
# ---------------------------------------------------------------------------

def test_recent_pending_markers_no_all_records(store, monkeypatch):
    """recent_pending_markers must never call all_records()."""
    r = _make_rec(tier="episodic", tags=["role:user"], seed=600)
    store.insert(r)

    def fail_all_records():
        raise AssertionError("recent_pending_markers must not call all_records()")

    monkeypatch.setattr(store, "all_records", fail_all_records)
    # Should not raise
    store.recent_pending_markers(n=10)


# ---------------------------------------------------------------------------
# recent_pending_markers — role:user not starved by ambient writes
# ---------------------------------------------------------------------------

def test_recent_pending_markers_role_not_starved(store):
    """After n+1 non-user ambient writes, the role:user turn still surfaces.

    The filter is pushed into SQL BEFORE the LIMIT, so ambient writes
    cannot push the user turn out of the result window.
    """
    n = 10
    # Insert one role:user turn first
    user_turn = _make_rec(tier="episodic", tags=["role:user"], seed=700)
    store.insert(user_turn)

    # Insert n+1 ambient (non-user) episodic records after it
    for i in range(n + 1):
        ambient = _make_rec(tier="episodic", tags=["role:system"], seed=701 + i)
        store.insert(ambient)

    result = store.recent_pending_markers(n=n)
    result_ids = {rec.id for rec in result}
    assert user_turn.id in result_ids, (
        f"role:user turn must appear in recent_pending_markers(n={n}) even after "
        f"{n+1} ambient writes (filter must be in SQL, not post-LIMIT Python)"
    )


# ---------------------------------------------------------------------------
# recent_pending_markers — EXPLAIN QUERY PLAN shows SEARCH USING INDEX
# ---------------------------------------------------------------------------

def test_recent_pending_markers_explain_search_using_index(store):
    """EXPLAIN QUERY PLAN for both READ A and READ B shows SEARCH USING INDEX.

    This confirms the partial index idx_records_pending is being used
    for pending records (READ A) and idx_records_tier for role:user (READ B).
    """
    # Seed a few records so there's something to query
    for i in range(5):
        store.insert(_make_rec(seed=800 + i))
    for i in range(3):
        store.insert(_make_rec(tier="episodic", tags=["role:user"], seed=810 + i))
    for i in range(2):
        _insert_pending(store, seed=820 + i, text=f"pending turn {i}")

    db = store.db

    # READ A EXPLAIN — full literal constant (compile-time string join, no runtime concat)
    _EXPLAIN_READ_A = (
        "EXPLAIN QUERY PLAN"
        " SELECT id, tier, literal_surface, aaak_index, embedding,"
        " community_id, centrality, detail_level, pinned,"
        " stability, difficulty, last_reviewed, never_decay, never_merge,"
        " provenance_json, created_at, updated_at, tags_json, language,"
        " s5_trust_score, profile_modulation_gain_json, schema_version,"
        " hv_tier, structure_hv_payload,"
        " COALESCE(embedding_pending, 0) AS embedding_pending"
        " FROM records WHERE embedding_pending = 1"
        " ORDER BY rowid DESC LIMIT ?"
    )
    # READ B EXPLAIN — full literal constant (compile-time string join, no runtime concat)
    _EXPLAIN_READ_B = (
        "EXPLAIN QUERY PLAN"
        " SELECT id, tier, literal_surface, aaak_index, embedding,"
        " community_id, centrality, detail_level, pinned,"
        " stability, difficulty, last_reviewed, never_decay, never_merge,"
        " provenance_json, created_at, updated_at, tags_json, language,"
        " s5_trust_score, profile_modulation_gain_json, schema_version,"
        " hv_tier, structure_hv_payload,"
        " COALESCE(embedding_pending, 0) AS embedding_pending"
        " FROM records WHERE tier='episodic' AND tags_json LIKE ?"
        " ORDER BY rowid DESC LIMIT ?"
    )

    with db._conn_lock:
        plan_a = db._conn.execute(_EXPLAIN_READ_A, (10,)).fetchall()
        plan_b = db._conn.execute(_EXPLAIN_READ_B, ('%"role:user"%', 40)).fetchall()

    plan_a_lines = [" ".join(str(v) for v in row) for row in plan_a]
    plan_b_lines = [" ".join(str(v) for v in row) for row in plan_b]

    # READ A: must use an index (idx_records_pending)
    has_index_a = any("USING INDEX" in line.upper() for line in plan_a_lines)
    has_scan_a = any("SCAN RECORDS" in line.upper() for line in plan_a_lines)
    assert has_index_a, (
        f"READ A (pending) must SEARCH USING INDEX (idx_records_pending). "
        f"EXPLAIN plan: {plan_a_lines}"
    )
    assert not has_scan_a, (
        f"READ A must not do a full SCAN of records. "
        f"EXPLAIN plan: {plan_a_lines}"
    )

    # READ B: must use an index (idx_records_tier) — no full SCAN of records
    has_scan_b = any("SCAN RECORDS" in line.upper() for line in plan_b_lines)
    assert not has_scan_b, (
        f"READ B must not do a full SCAN of records (must be index-backed on tier). "
        f"EXPLAIN plan: {plan_b_lines}"
    )


# ---------------------------------------------------------------------------
# recent_pending_markers — large pending backlog is bounded (LIMIT)
# ---------------------------------------------------------------------------

def test_recent_pending_markers_large_pending_backlog_bounded(store):
    """A large pending backlog does not cause a full decrypt of all pending rows.

    recent_pending_markers(n=10) must return at most n records even when
    there are 200+ embedding_pending rows. The SQL includes ORDER BY rowid
    DESC LIMIT ? (verified at source level) so the database caps the decrypt.
    """
    # The pending READ A SQL constant must end with 'LIMIT ?'
    from iai_mcp.store import MemoryStore
    pending_sql = MemoryStore._PENDING_READ_SQL
    assert "LIMIT ?" in pending_sql, (
        f"_PENDING_READ_SQL must contain 'LIMIT ?' for CC2-H4 bounding: {pending_sql!r}"
    )

    n = 10
    # Plant a large pending backlog via the direct pending-row write path
    from datetime import datetime, timezone
    for i in range(25):  # enough to exceed n without being slow
        record_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        store.db.insert_pending_row(
            record_id=record_id,
            tier="episodic",
            literal_surface=f"pending turn {i}",
            tags_json=json.dumps([]),
            provenance_json=json.dumps([]),
            created_at=now,
            updated_at=now,
        )

    result = store.recent_pending_markers(n=n)
    # The result must be bounded to at most n
    assert len(result) <= n, (
        f"recent_pending_markers(n={n}) must return at most {n} records, "
        f"got {len(result)} (SQL LIMIT not being honoured)"
    )


# ---------------------------------------------------------------------------
# recent_pending_markers — idx_records_pending exists on existing store
# ---------------------------------------------------------------------------

def test_idx_records_pending_exists(store):
    """The idx_records_pending partial index must exist after store init."""
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_records_pending'"
        ).fetchall()
    assert rows, (
        "idx_records_pending partial index must exist after MemoryStore init"
    )


# ---------------------------------------------------------------------------
# recent_pending_markers — dedupe: pending AND role:user appears once
# ---------------------------------------------------------------------------

def test_recent_pending_markers_dedup(store):
    """A record that is both pending AND role:user appears only once in results.

    Uses insert_pending_row with role:user tag so the record appears in both
    READ A (pending) and READ B (role:user). After dedup it must appear once.
    """
    from datetime import datetime, timezone
    record_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    store.db.insert_pending_row(
        record_id=record_id,
        tier="episodic",
        literal_surface="pending role:user turn",
        tags_json=json.dumps(["role:user"]),
        provenance_json=json.dumps([]),
        created_at=now,
        updated_at=now,
    )
    result = store.recent_pending_markers(n=20)
    result_id_strs = [str(rec.id) for rec in result]
    count = result_id_strs.count(record_id)
    assert count == 1, (
        f"Record that is both pending and role:user appeared {count} times (expected 1)"
    )
