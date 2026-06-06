"""Tests for Task 4 LEARN-04 curiosity.

 trigger: entropy > 0.7 bits, 3-turn cooldown.
 tiered style:
- low entropy (0.4-0.7): silent log via events table (curiosity_silent_log)
- mid entropy (0.7-0.9): inline hint in next response
- high entropy (>0.9): direct clarifying question

compute_entropy operates in base-2 (bits) consistent with "0.7 bits".
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _rec(vec=None, tags=None):
    vec = vec or [1.0] + [0.0] * (EMBED_DIM - 1)
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="r",
        aaak_index="",
        embedding=vec,
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
        tags=list(tags or []),
        language="en",
    )


class _Hit:
    def __init__(self, rid: UUID, score: float):
        self.record_id = rid
        self.score = score


# ---------------------------------------------------------------- constants


def test_curiosity_thresholds():
    from iai_mcp import curiosity

    assert curiosity.ENTROPY_LOW == 0.4
    assert curiosity.ENTROPY_MID == 0.7
    assert curiosity.ENTROPY_HIGH == 0.9
    assert curiosity.COOLDOWN_TURNS == 3


# ---------------------------------------------------------------- compute_entropy


def test_compute_entropy_uniform():
    """Shannon entropy of [0.5, 0.5] = 1.0 bit."""
    from iai_mcp.curiosity import compute_entropy

    e = compute_entropy([0.5, 0.5])
    assert abs(e - 1.0) < 1e-6


def test_compute_entropy_skewed():
    from iai_mcp.curiosity import compute_entropy

    e = compute_entropy([0.9, 0.1])
    # H([0.9,0.1]) = -(0.9*log2(0.9) + 0.1*log2(0.1)) ~ 0.469
    assert e < 0.5


def test_compute_entropy_degenerate():
    from iai_mcp.curiosity import compute_entropy

    assert compute_entropy([1.0]) == 0.0


def test_compute_entropy_empty():
    from iai_mcp.curiosity import compute_entropy

    assert compute_entropy([]) == 0.0


def test_compute_entropy_zero_scores_handled():
    from iai_mcp.curiosity import compute_entropy

    # Negative scores shouldn't crash (max(0, s) normalisation).
    e = compute_entropy([-1.0, 0.5, 0.5])
    assert e >= 0.0


# ---------------------------------------------------------------- fire_curiosity


def test_fire_curiosity_below_threshold_silent(tmp_path):
    """Low entropy (0.5) -> silent log, returns None."""
    from iai_mcp.curiosity import fire_curiosity
    from iai_mcp.events import query_events

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)
    hits = [_Hit(r.id, 0.8)]
    q = fire_curiosity(
        store, hits, cue="ambiguous", entropy=0.5,
        session_id="s1", turn=1,
    )
    assert q is None
    silent = query_events(store, kind="curiosity_silent_log")
    assert len(silent) >= 1


def test_fire_curiosity_below_ENTROPY_LOW_returns_none(tmp_path):
    """Very low entropy (below ENTROPY_LOW=0.4) returns None without logging."""
    from iai_mcp.curiosity import fire_curiosity

    store = MemoryStore(path=tmp_path)
    q = fire_curiosity(
        store, [], cue="x", entropy=0.1,
        session_id="s-silent", turn=1,
    )
    assert q is None


def test_fire_curiosity_mid_entropy_inline_hint(tmp_path):
    """Entropy 0.8 -> CuriosityQuestion with tier='inline'."""
    from iai_mcp.curiosity import fire_curiosity

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)
    hits = [_Hit(r.id, 0.6)]
    q = fire_curiosity(
        store, hits, cue="maybe", entropy=0.8,
        session_id="s2", turn=1,
    )
    assert q is not None
    assert q.tier == "inline"


def test_fire_curiosity_high_entropy_direct_question(tmp_path):
    from iai_mcp.curiosity import fire_curiosity

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)
    hits = [_Hit(r.id, 0.5)]
    q = fire_curiosity(
        store, hits, cue="unknown", entropy=0.95,
        session_id="s3", turn=1,
    )
    assert q is not None
    assert q.tier == "question"


def test_fire_curiosity_cooldown_3_turns(tmp_path):
    """Fire turn 1 -> fires. Turn 2 -> None (cooldown). Turn 3 -> None."""
    from iai_mcp.curiosity import fire_curiosity

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)
    hits = [_Hit(r.id, 0.5)]
    q1 = fire_curiosity(store, hits, "x", 0.95, "s4", turn=1)
    assert q1 is not None
    q2 = fire_curiosity(store, hits, "x", 0.95, "s4", turn=2)
    assert q2 is None
    q3 = fire_curiosity(store, hits, "x", 0.95, "s4", turn=3)
    assert q3 is None


def test_fire_curiosity_cooldown_releases(tmp_path):
    """Turn 4 after turn 1 firing -> cooldown released."""
    from iai_mcp.curiosity import fire_curiosity

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)
    hits = [_Hit(r.id, 0.5)]
    q1 = fire_curiosity(store, hits, "x", 0.95, "s5", turn=1)
    assert q1 is not None
    q4 = fire_curiosity(store, hits, "x", 0.95, "s5", turn=4)
    assert q4 is not None


# ---------------------------------------------------------------- pending_questions


def test_pending_questions_empty(tmp_path):
    from iai_mcp.curiosity import pending_questions

    store = MemoryStore(path=tmp_path)
    assert pending_questions(store) == []


def test_pending_questions_filter_resolved(tmp_path):
    """5 fired, 3 resolved -> pending_questions returns 2."""
    from iai_mcp.curiosity import fire_curiosity, pending_questions
    from iai_mcp.events import write_event

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)
    hits = [_Hit(r.id, 0.5)]
    # Fire 5 questions across different sessions so cooldown doesn't block.
    q_ids: list = []
    for i in range(5):
        q = fire_curiosity(store, hits, f"cue{i}", 0.95, f"session-{i}", turn=1)
        assert q is not None
        q_ids.append(q.id)

    # Resolve 3 via curiosity_resolved event
    for qid in q_ids[:3]:
        write_event(
            store, kind="curiosity_resolved",
            data={"question_id": str(qid)},
            severity="info",
        )

    pending = pending_questions(store)
    assert len(pending) == 2


def test_pending_questions_by_session(tmp_path):
    from iai_mcp.curiosity import fire_curiosity, pending_questions

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)
    hits = [_Hit(r.id, 0.5)]
    fire_curiosity(store, hits, "c", 0.95, "sA", turn=1)
    fire_curiosity(store, hits, "c", 0.95, "sB", turn=1)

    onlyA = pending_questions(store, session_id="sA")
    onlyB = pending_questions(store, session_id="sB")
    assert len(onlyA) == 1
    assert len(onlyB) == 1
