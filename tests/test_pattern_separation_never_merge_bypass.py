"""Regression tests: pattern_separation_gate must bypass SKIP-merge when either
party has never_merge=True.

Invariant: records with never_merge=True are sacrosanct. The
pattern_separation_gate SKIP-merge branch must not return GateAction.SKIP when
either the new candidate or the existing top-1 hit carries never_merge=True.

Three scenarios:
1. New candidate pinned (never_merge=True), existing record not pinned.
   Expected: INSERT action — the pinned record must exist as its own row.
2. Existing top-1 hit pinned (never_merge=True), new candidate not pinned.
   Expected: INSERT action — the pinned record must not absorb new content.
3. Neither party pinned (regression-control): standard SKIP-merge still fires.
   Expected: SKIP action (control — must stay green before AND after fix).

RED->GREEN discipline:
- Scenarios 1+2 FAIL on current source (gate returns SKIP unconditionally).
- Scenario 3 PASSES both before and after the fix.
- All 3 PASS after the store.py fix.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.events import query_events
from iai_mcp.store import (
    RECORDS_TABLE,
    GateAction,
    MemoryStore,
)
from iai_mcp.types import MemoryRecord


# ---------------------------------------------------------------------------
# Module-level constants — mirror test_phase11_1_pattern_separation.py
# ---------------------------------------------------------------------------

EMBED_DIM = 4
REFERENCE_EMBEDDING: list[float] = [1.0, 0.0, 0.0, 0.0]
NEAR_DUP_DEFAULT: float = 0.92


# ---------------------------------------------------------------------------
# Helpers — borrowed verbatim from test_phase11_1_pattern_separation.py
# ---------------------------------------------------------------------------


def _make_embedding_at_cosine(
    cos_target: float, embed_dim: int = EMBED_DIM,
) -> list[float]:
    if not (-1.0 <= cos_target <= 1.0):
        raise ValueError(
            f"cos_target must be in [-1.0, 1.0], got {cos_target}"
        )
    if embed_dim < 2:
        raise ValueError(f"embed_dim must be >= 2, got {embed_dim}")
    residual = math.sqrt(max(0.0, 1.0 - cos_target * cos_target))
    return [cos_target, residual] + [0.0] * (embed_dim - 2)


def _make_record(
    *,
    embedding: list[float],
    tier: str = "episodic",
    literal_surface: str = "alice prefers tea over coffee",
    **overrides,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    base = dict(
        id=uuid4(),
        tier=tier,
        literal_surface=literal_surface,
        aaak_index="",
        embedding=list(embedding),
        community_id=None,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
    )
    base.update(overrides)
    return MemoryRecord(**base)


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        path=str(tmp_path / "iai-mcp"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )


def _events_in_insert_order(store: MemoryStore) -> list[dict]:
    """Return pattern_separation_pass events in chronological (insert) order."""
    events = query_events(store, kind="pattern_separation_pass", limit=1000)
    return list(reversed(events))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_patsep_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clear IAI_MCP_PATSEP_* overrides + pin embed dim + pin store path."""
    for var in (
        "IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT",
        "IAI_MCP_PATSEP_TOP_K",
        "IAI_MCP_PATSEP_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(EMBED_DIM))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp"))


@pytest.fixture
def fresh_store(tmp_path: Path) -> MemoryStore:
    return _make_store(tmp_path)


# ---------------------------------------------------------------------------
# Scenario 1 — new candidate pinned, existing not pinned
# ---------------------------------------------------------------------------


def test_never_merge_new_candidate_bypasses_skip(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pattern-sep gate must INSERT when the NEW candidate has never_merge=True.

    Setup: insert an unpinned A at REFERENCE_EMBEDDING; then insert a
    never_merge=True B at cos=0.97 (above near_dup_threshold=0.92).

    Before fix: gate returns GateAction.SKIP — B's id is replaced with A's id
    and the records table has only 1 row.
    After fix: gate returns GateAction.INSERT — both rows exist independently.
    """
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    # Insert unpinned anchor A.
    rec_a = _make_record(embedding=REFERENCE_EMBEDDING, never_merge=False)
    fresh_store.insert(rec_a)
    a_id = rec_a.id

    # B is near-duplicate of A (cos=0.97) but is itself pinned.
    near_dup_embedding = _make_embedding_at_cosine(0.97)
    rec_b = _make_record(
        embedding=near_dup_embedding,
        never_merge=True,
        literal_surface="alice identity anchor — never merge",
    )
    b_id_before = rec_b.id
    fresh_store.insert(rec_b)

    # After insert: rec_b.id must NOT have been overwritten with a_id.
    assert rec_b.id == b_id_before, (
        f"never_merge=True new candidate must keep its own id; "
        f"got rec_b.id={rec_b.id}, expected {b_id_before}"
    )
    assert rec_b.id != a_id, (
        "never_merge=True new candidate must not be collapsed into existing record"
    )

    # Both records must exist as distinct rows.
    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 2, (
        f"never_merge=True candidate must INSERT as its own row; "
        f"records table has {tbl.count_rows()} rows (expected 2)"
    )

    # The event for B's insert must report action="insert", not action="skip".
    events = _events_in_insert_order(fresh_store)
    # events[0] = A insert, events[1] = B insert
    assert len(events) == 2, (
        f"expected 2 pattern_separation_pass events; got {len(events)}"
    )
    b_event_body = events[1]["data"]
    assert b_event_body["action"] == "insert", (
        f"event for never_merge=True candidate must report action=insert; "
        f"got body={b_event_body}"
    )


# ---------------------------------------------------------------------------
# Scenario 2 — existing top-1 hit pinned, new candidate not pinned
# ---------------------------------------------------------------------------


def test_never_merge_existing_record_bypasses_skip(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pattern-sep gate must INSERT when the EXISTING top-1 hit has never_merge=True.

    Setup: insert a pinned (never_merge=True) A at REFERENCE_EMBEDDING; then
    insert an unpinned B at cos=0.97 (above near_dup_threshold=0.92).

    Before fix: gate returns GateAction.SKIP — B's id is replaced with A's id.
    After fix: gate returns GateAction.INSERT — both rows exist independently.
    The pinned A is never absorbed into by a new record.
    """
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    # Insert pinned anchor A.
    rec_a = _make_record(
        embedding=REFERENCE_EMBEDDING,
        never_merge=True,
        literal_surface="alice identity anchor — must never absorb duplicates",
    )
    fresh_store.insert(rec_a)
    a_id = rec_a.id

    # B is near-duplicate of A (cos=0.97) and is itself NOT pinned.
    near_dup_embedding = _make_embedding_at_cosine(0.97)
    rec_b = _make_record(embedding=near_dup_embedding, never_merge=False)
    b_id_before = rec_b.id
    fresh_store.insert(rec_b)

    # After insert: rec_b.id must NOT have been overwritten with a_id.
    assert rec_b.id == b_id_before, (
        f"new candidate near existing never_merge=True record must keep its id; "
        f"got rec_b.id={rec_b.id}, expected {b_id_before}"
    )
    assert rec_b.id != a_id, (
        "new candidate must not be collapsed into a never_merge=True existing record"
    )

    # Both records must exist as distinct rows.
    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 2, (
        f"never_merge=True existing record must not absorb new inserts; "
        f"records table has {tbl.count_rows()} rows (expected 2)"
    )

    # The event for B's insert must report action="insert", not action="skip".
    events = _events_in_insert_order(fresh_store)
    assert len(events) == 2, (
        f"expected 2 pattern_separation_pass events; got {len(events)}"
    )
    b_event_body = events[1]["data"]
    assert b_event_body["action"] == "insert", (
        f"event for insert near never_merge=True existing record must be action=insert; "
        f"got body={b_event_body}"
    )


# ---------------------------------------------------------------------------
# Scenario 3 — neither party pinned (regression-control: SKIP still fires)
# ---------------------------------------------------------------------------


def test_neither_pinned_skip_merge_still_fires(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression-control: standard SKIP-merge behavior must be preserved when
    neither the new candidate nor the existing record has never_merge=True.

    Setup: insert an unpinned A at REFERENCE_EMBEDDING; then insert an unpinned B
    at cos=0.97 (above near_dup_threshold=0.92).

    Expected (before AND after fix): gate returns GateAction.SKIP — only 1 row
    in records table; B's id is overwritten with A's id.

    This scenario must PASS before the fix (control) and PASS after the fix
    (no regression to normal near-dup merging behavior).
    """
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING, never_merge=False)
    fresh_store.insert(rec_a)
    a_id = rec_a.id

    near_dup_embedding = _make_embedding_at_cosine(0.97)
    rec_b = _make_record(embedding=near_dup_embedding, never_merge=False)
    fresh_store.insert(rec_b)

    # SKIP must have fired: B's id becomes A's id and only 1 row exists.
    assert rec_b.id == a_id, (
        f"SKIP-merge must fire when neither party is pinned; "
        f"expected rec_b.id={a_id}, got {rec_b.id}"
    )

    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 1, (
        f"SKIP-merge must collapse near-duplicate to 1 row when neither pinned; "
        f"got {tbl.count_rows()} rows"
    )

    # Event must report action="skip".
    events = _events_in_insert_order(fresh_store)
    assert len(events) == 2, (
        f"expected 2 pattern_separation_pass events; got {len(events)}"
    )
    b_event_body = events[1]["data"]
    assert b_event_body["action"] == "skip", (
        f"event for standard near-dup must report action=skip; "
        f"got body={b_event_body}"
    )
    assert b_event_body["near_dup_hit_id"] == str(a_id), b_event_body
