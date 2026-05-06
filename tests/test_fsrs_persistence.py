"""Tests for 02-REVIEW.md H-01 (FSRS tick not persisted across restart).

Bug: `run_light_consolidation` calls `_apply_fsrs(r, now)` which mutates
record.stability and record.last_reviewed in-place on the in-memory
MemoryRecord object. The updated record was never written back to the store.
Every process restart reset all FSRS fields to their previous checkpoint.

Fix:
    - Add MemoryStore.update_record(record) that rewrites ONLY the FSRS
      columns (stability, difficulty, last_reviewed, updated_at) via
      _uuid_literal-safe WHERE predicate. No embedding / provenance /
      tags / community_id changes -- avoids clobbering concurrent
      boost_edges / append_provenance writers.
    - Call store.update_record(r) inside run_light_consolidation after
      _apply_fsrs mutates r.

Constitutional contract (MEM-07 FSRS biological fidelity + D-STORAGE):
    FSRS stability is the biological decay curve state. Losing it on every
    restart equivalates to wiping short-term memory at every session
    switch -- unacceptable for a system whose promise is "Claude remembers
    every word".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


# ---------------------------------------------------------------- helpers


def _record(
    *,
    text: str = "fsrs-target",
    stability: float = 0.1,
    prov_seconds_ago: int = 30,
) -> MemoryRecord:
    """Build a record with a fresh provenance entry so run_light_consolidation
    will actually tick it (the light pass only nudges records whose last
    provenance entry is < 1h old)."""
    now = datetime.now(timezone.utc)
    prov_ts = (now - timedelta(seconds=prov_seconds_ago)).isoformat()
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=stability,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": prov_ts, "cue": "recall", "session_id": "s1"}],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


# ============================================== update_record API unit tests


def test_update_record_writes_back_fsrs_columns(tmp_path):
    """MemoryStore.update_record persists stability/difficulty/last_reviewed."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _record(stability=0.1)
    store.insert(rec)

    # Mutate the in-memory copy then write it back
    rec.stability = 0.55
    rec.difficulty = 0.42
    new_reviewed = datetime.now(timezone.utc)
    rec.last_reviewed = new_reviewed

    store.update_record(rec)

    fresh = store.get(rec.id)
    assert fresh is not None
    assert fresh.stability == pytest.approx(0.55, abs=1e-3)
    assert fresh.difficulty == pytest.approx(0.42, abs=1e-3)
    assert fresh.last_reviewed is not None


def test_update_record_rejects_unknown_id(tmp_path):
    """Calling update_record on a record id that is not in the table must be
    a no-op (no exception, no table growth)."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    # No insert -- record never existed
    phantom = _record(stability=0.9)

    # Row count before
    before = store.db.open_table("records").count_rows()

    # Must not raise
    store.update_record(phantom)

    # Row count unchanged (no row was inserted)
    after = store.db.open_table("records").count_rows()
    assert after == before


def test_update_record_does_not_touch_untouched_columns(tmp_path):
    """update_record must only rewrite FSRS-relevant columns. Embedding,
    provenance, tags, community_id must survive unchanged."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _record(stability=0.1)
    rec.tags = ["important", "keep-me"]
    rec.provenance = [
        {"ts": "2026-04-16T00:00:00Z", "cue": "seed", "session_id": "s0"},
    ]
    store.insert(rec)

    # Only change FSRS fields in-memory; leave rec.tags / rec.provenance alone.
    rec.stability = 0.6
    rec.last_reviewed = datetime.now(timezone.utc)
    store.update_record(rec)

    fresh = store.get(rec.id)
    assert fresh is not None
    # FSRS columns updated
    assert fresh.stability == pytest.approx(0.6, abs=1e-3)
    # Unrelated columns preserved
    assert fresh.tags == ["important", "keep-me"]
    assert len(fresh.provenance) == 1
    assert fresh.provenance[0]["cue"] == "seed"


# ============================================== H-01 end-to-end persistence


def test_fsrs_state_persists_across_store_reopen(tmp_path):
    """H-01 end-to-end: after run_light_consolidation, a NEW MemoryStore
    instance at the same tmp_path must see updated stability + last_reviewed.

    Pre-fix: stability stayed at 0.1 because _apply_fsrs only mutated the
    in-memory object; nothing was written back.
    Post-fix: stability >= 0.1 + FSRS_STABILITY_BOOST (0.3 cap at 1.0).
    """
    from iai_mcp.sleep import FSRS_STABILITY_BOOST, run_light_consolidation
    from iai_mcp.store import MemoryStore

    # Phase A: create, insert with fresh provenance, run light cycle
    store = MemoryStore(path=tmp_path)
    rec = _record(stability=0.1, prov_seconds_ago=30)
    rec_id = rec.id
    store.insert(rec)

    result = run_light_consolidation(store, session_id="persist-test")
    assert result["fsrs_ticked"] >= 1

    # Phase B: close (via new instance on the same path) and re-read
    del store
    store2 = MemoryStore(path=tmp_path)
    fresh = store2.get(rec_id)
    assert fresh is not None

    # Stability boosted and persisted
    expected_min = 0.1 + FSRS_STABILITY_BOOST - 1e-3
    assert fresh.stability >= expected_min, (
        f"FSRS stability not persisted: expected >= {expected_min}, "
        f"got {fresh.stability}"
    )
    # last_reviewed populated
    assert fresh.last_reviewed is not None


def test_fsrs_persistence_only_fresh_provenance(tmp_path):
    """Records with STALE provenance (>1h old) must NOT be FSRS-ticked. This
    preserves the current sleep.py light-phase gating; our update_record fix
    must not widen that surface.
    """
    from iai_mcp.sleep import run_light_consolidation
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    # 2h-old provenance -- outside the 1h tick window
    rec = _record(stability=0.1, prov_seconds_ago=7200)
    store.insert(rec)

    run_light_consolidation(store, session_id="no-tick")
    fresh = store.get(rec.id)
    assert fresh is not None
    # Stability unchanged
    assert fresh.stability == pytest.approx(0.1, abs=1e-3)
