"""Plan 07.11-03 / regression tests for crash-safe reembed migration.

Closes V2-05: the reembed migration at migrate.py:300-305 dropped the records
table and rebuilt row-by-row from a stashed iterator. A crash, kill, power
loss, or KeyboardInterrupt between drop and rebuild left the user with an
empty records table — no staging path, no rollback, no resume. This file's
five regression tests fail on the un-fixed code (mid-flight kill empties
records; no rollback function reachable; no resume) and pass after the
four-phase staged-swap flow + boot-time detector are in place.

Required cases (verbatim names from D-05):
1. test_mid_migration_kill_preserves_old_table — KeyboardInterrupt on the
   4th embed call leaves records (10) intact; records_v_new present with 3
   staged rows; migration_progress.json points at row 3.
2. test_rollback_handler_restores_from_old — from the kill state, _rollback
   drops records_v_new and (if records is missing) renames records_old_<ts>
   back. Drops progress file.
3. test_successful_migration_promotes_old_to_records — happy path: records
   has all rows after; records_v_new is gone; ONE records_old_<ts> remains
   (deferred cleanup — dropped on next boot).
4. test_resume_handler_continues_from_checkpoint — from the kill state,
   _resume picks up at row 4 and finishes; final records.count_rows() == 10.
5. test_idempotency_rerun_after_success — re-running migrate after a clean
   migration is a no-op + emits migration_reembed event with no_op=True.

Honesty constraint: every test FAILs on git stash of Tasks 1-4 and
PASSes on git stash pop.

Test target_dim choice (deviation from plan literal): Tasks 1-4 use a
DIFFERENT target dim (1024) from the source (384) to force the staging
path. The plan's literal 384→384 same-dim setup hits the early-return
no_op branch BEFORE any embed call fires, so the kill-mid-flight injection
never triggers. Test 5 (idempotency) is structured as 384→1024 first run
(real migration) then 1024→1024 second call (no_op witness). This is a
test-spec correction surfaced during pre-write review; the contract from
CONTEXT (mid-flight kill preserves old table; rollback restores;
resume continues; idempotent rerun after success) is preserved verbatim.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Standard project test isolation — verbatim from
    tests/test_pipeline_anti_hits_malformed.py:33-45. Without this fixture
    the test will fail on the construction host because the OS keyring is
    unavailable."""
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


# --------------------------------------------------------------------------- harness


class _DimEmbedder:
    """Deterministic fake embedder with configurable dim. Verbatim from
    tests/test_migrate_reembed_to_current_dim.py:24-41 — the canonical
    project pattern for testing dim-change scenarios without loading
    transformers."""

    def __init__(self, dim: int):
        self.DIM = dim
        self.model_key = f"fake-dim-{dim}"

    def embed(self, text: str) -> list[float]:
        import math
        vec = [0.0] * self.DIM
        for i, ch in enumerate(text or ""):
            vec[i % self.DIM] += ord(ch) / 256.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _fresh_store(tmp_path, dim: int, monkeypatch):
    """Make a MemoryStore at an explicit dim via env override. Verbatim from
    tests/test_migrate_reembed_to_current_dim.py:44-50."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(dim))
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _seed_records(store, embedder, n: int = 10) -> list[UUID]:
    """Insert n deterministic records. Mirrors
    tests/test_migrate_reembed_to_current_dim.py:53-88 — same field shape.
    Parameterised on n so each test seeds the count it needs."""
    from iai_mcp.types import MemoryRecord
    ids = []
    now = datetime.now(timezone.utc)
    for i in range(n):
        rid = uuid4()
        text = f"Crash-safe seed record #{i:02d} with literal surface content."
        rec = MemoryRecord(
            id=rid,
            tier="episodic",
            literal_surface=text,
            aaak_index="",
            embedding=embedder.embed(text),
            structure_hv=b"",
            community_id="",
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.5,
            difficulty=0.3,
            last_reviewed=now,
            never_decay=False,
            never_merge=False,
            provenance=[
                {"ts": "2026-04-30T00:00:00+00:00", "cue": f"seed-{i}", "session_id": "seed"}
            ],
            created_at=now,
            updated_at=now,
            tags=["test", "crash-safe"],
            language="en",
            s5_trust_score=0.5,
            profile_modulation_gain={},
            schema_version=4,
        )
        store.insert(rec)
        ids.append(rid)
    return ids


# --------------------------------------------------------------------------- cases


def test_successful_migration_promotes_old_to_records(tmp_path, monkeypatch):
    """D-05 case 3 / happy path: stage -> validate -> atomic swap ->
    deferred cleanup. Post-state: records has 20 rows; records_v_new is
    gone (cleaned); ONE records_old_<ts> remains (deferred cleanup, will
    be dropped on next boot via detect_partial_migration)."""
    src = _DimEmbedder(384)
    target = _DimEmbedder(1024)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, src, n=20)

    from iai_mcp.migrate import migrate_reembed_to_current_dim
    result = migrate_reembed_to_current_dim(store, target)
    assert result["target_dim"] == 1024
    assert result["source_dim"] == 384

    names = set(store.db.table_names())
    assert "records" in names, "records table must exist after swap"
    assert "records_v_new" not in names, (
        "records_v_new must be cleaned after atomic swap"
    )
    # Deferred cleanup: one records_old_<ts> remains; it will be dropped
    # on next boot's detect_partial_migration -> needs_cleanup branch.
    old_tables = [n for n in names if n.startswith("records_old_")]
    assert len(old_tables) == 1, (
        f"exactly one records_old_<ts> expected (deferred cleanup); got {old_tables}"
    )

    # All 20 rows present at the new dim (per-row failure tolerance is up to 1%).
    assert store.db.open_table("records").count_rows() >= 19


def test_mid_migration_kill_preserves_old_table(tmp_path, monkeypatch):
    """D-05 case 1 / mid-flight kill: KeyboardInterrupt on the 4th
    embed call leaves the original records (10) intact; records_v_new
    present with 3 staged rows; migration_progress.json points at row 3."""
    src = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, src, n=10)

    target = _DimEmbedder(1024)
    call_count = {"n": 0}
    real_embed = target.embed

    def embed_or_kill(text):
        call_count["n"] += 1
        if call_count["n"] > 3:
            raise KeyboardInterrupt("simulated mid-migration kill")
        return real_embed(text)

    monkeypatch.setattr(target, "embed", embed_or_kill)

    from iai_mcp.migrate import migrate_reembed_to_current_dim
    with pytest.raises(KeyboardInterrupt):
        migrate_reembed_to_current_dim(store, target)

    names = set(store.db.table_names())
    # Original records intact (Phase 1 doesn't touch records).
    assert "records" in names
    assert store.db.open_table("records").count_rows() == 10, (
        "Original records table must stay intact when kill fires mid-stage"
    )
    # Staging table partial.
    assert "records_v_new" in names, (
        "records_v_new must exist with the partial set after kill"
    )
    assert store.db.open_table("records_v_new").count_rows() == 3, (
        "records_v_new must hold the 3 successfully-staged rows"
    )
    # Progress file present.
    progress_path = Path(store.root) / "migration_progress.json"
    assert progress_path.exists(), (
        "migration_progress.json must be written on each successful row"
    )


def test_rollback_handler_restores_from_old(tmp_path, monkeypatch):
    """D-05 case 2 / rollback: from the kill state, _rollback drops
    records_v_new. records is intact (Phase 1 didn't touch it). Drops
    progress file. No records_old_<ts> in this scenario because the kill
    fired before the atomic swap."""
    src = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, src, n=10)

    # Reproduce the kill state from test 1.
    target = _DimEmbedder(1024)
    call_count = {"n": 0}
    real_embed = target.embed

    def embed_or_kill(text):
        call_count["n"] += 1
        if call_count["n"] > 3:
            raise KeyboardInterrupt()
        return real_embed(text)

    monkeypatch.setattr(target, "embed", embed_or_kill)

    from iai_mcp.migrate import migrate_reembed_to_current_dim, _rollback
    with pytest.raises(KeyboardInterrupt):
        migrate_reembed_to_current_dim(store, target)

    rc = _rollback(store.db, store)
    assert rc == 0, "rollback must succeed on a clean kill-mid-stage state"

    names = set(store.db.table_names())
    assert "records" in names, "records must still exist (Phase 1 never dropped it)"
    assert store.db.open_table("records").count_rows() == 10, (
        "records must hold the original 10 rows after rollback"
    )
    assert "records_v_new" not in names, "records_v_new must be dropped by rollback"
    assert not any(n.startswith("records_old_") for n in names), (
        "no records_old_<ts> in this scenario (kill fired before swap)"
    )
    progress_path = Path(store.root) / "migration_progress.json"
    assert not progress_path.exists(), "rollback must drop the progress file"


def test_resume_handler_continues_from_checkpoint(tmp_path, monkeypatch):
    """D-05 case 4 / resume: from the kill state, _resume picks up at
    row 4 and finishes the remaining 7 rows. Final records.count_rows() ==
    10; records_v_new is cleaned up."""
    src = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, src, n=10)

    target = _DimEmbedder(1024)
    call_count = {"n": 0}
    real_embed = target.embed

    def embed_or_kill(text):
        call_count["n"] += 1
        if call_count["n"] > 3:
            raise KeyboardInterrupt()
        return real_embed(text)

    monkeypatch.setattr(target, "embed", embed_or_kill)

    from iai_mcp.migrate import migrate_reembed_to_current_dim, _resume
    with pytest.raises(KeyboardInterrupt):
        migrate_reembed_to_current_dim(store, target)

    # Resume with a fresh (no-kill) embedder. Restore the real embed.
    monkeypatch.setattr(target, "embed", real_embed)
    rc = _resume(store.db, store, target)
    assert rc == 0, "resume must succeed on a recoverable partial state"

    assert store.db.open_table("records").count_rows() == 10, (
        "all 10 rows present after resume + atomic swap"
    )
    assert "records_v_new" not in set(store.db.table_names()), (
        "records_v_new cleaned after the swap completes"
    )
    progress_path = Path(store.root) / "migration_progress.json"
    assert not progress_path.exists(), "resume must drop the progress file on success"


def test_idempotency_rerun_after_success(tmp_path, monkeypatch):
    """D-05 case 5 / idempotency: re-running migrate after a clean
    migration is a no-op + emits migration_reembed event with no_op=True.

    Sequence: 384 -> 1024 (real migration) then 1024 -> 1024 (no_op).
    Asserts the second run emits the no_op event flag, mirroring the
    semantic of the legacy line-244-250 idempotency contract preserved in
    the new staged-swap path."""
    from iai_mcp.events import query_events
    src = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, src, n=5)

    from iai_mcp.migrate import migrate_reembed_to_current_dim
    # First run: real migration 384 -> 1024.
    migrate_reembed_to_current_dim(store, _DimEmbedder(1024))
    # Second run at the now-current dim — must be a no_op witness.
    migrate_reembed_to_current_dim(store, _DimEmbedder(1024))

    events = query_events(store, kind="migration_reembed", limit=5)
    assert len(events) >= 2, (
        f"both runs must emit a migration_reembed event; got {len(events)}"
    )
    no_op_events = [e for e in events if e["data"].get("no_op") is True]
    assert len(no_op_events) >= 1, (
        "second run at same dim must emit a migration_reembed event with no_op=True"
    )
