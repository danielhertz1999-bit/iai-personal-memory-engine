"""Active curiosity (LEARN-04, , ) -- Task 4.

 trigger: prediction entropy > 0.7 bits AND 3-turn cooldown since last
curiosity question in this session.

 tiered style:
- entropy in [ENTROPY_LOW, ENTROPY_MID)  -> silent log event, no question
- entropy in [ENTROPY_MID, ENTROPY_HIGH) -> inline hint
- entropy >= ENTROPY_HIGH                -> direct clarifying question

Every question creates curiosity_bridge edges from each triggering record to
the question's UUID (used as a stable hub id). The question itself lives in
the events table (kind=curiosity_question); callers may insert a first-class
record if persistent text is desired, but keeps questions
event-sourced to minimise LanceDB write volume.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore


# ---------------------------------------------------------------- constants


ENTROPY_LOW: float = 0.4
ENTROPY_MID: float = 0.7
ENTROPY_HIGH: float = 0.9
COOLDOWN_TURNS: int = 3


# ---------------------------------------------------------------- types


@dataclass
class CuriosityQuestion:
    """One curiosity question surfaced by fire_curiosity."""

    id: UUID
    text: str
    triggered_by_record_ids: list[UUID] = field(default_factory=list)
    entropy: float = 0.0
    tier: str = "question"   # "silent" | "inline" | "question"
    resolved: bool = False


# ---------------------------------------------------------------- helpers


def compute_entropy(scores: list[float]) -> float:
    """Shannon entropy (base-2, bits) over a score distribution.

    Returns 0.0 for empty or degenerate inputs. Negative scores are clamped
    to 0 before normalisation so the probability vector is well-defined.
    """
    if not scores:
        return 0.0
    positive = [max(0.0, float(s)) for s in scores]
    total = sum(positive)
    if total <= 0:
        return 0.0
    probs = [p / total for p in positive]
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h


def _last_curiosity_turn(store: MemoryStore, session_id: str) -> int | None:
    """Return the turn of the most recent curiosity_question in this session."""
    events = query_events(store, kind="curiosity_question", limit=20)
    for e in events:
        if e.get("session_id") == session_id:
            try:
                return int(e["data"].get("turn", 0))
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------- fire_curiosity


def fire_curiosity(
    store: MemoryStore,
    hits: list,
    cue: str,
    entropy: float,
    session_id: str,
    turn: int,
) -> CuriosityQuestion | None:
    """ gate + tiering.

    Returns a CuriosityQuestion (or None) and, as a side effect:
    - emits a curiosity_silent_log event for low-entropy misses
    - emits a curiosity_question event for mid/high fires
    - creates curiosity_bridge edges from each triggering record -> question
    """
    if entropy < ENTROPY_LOW:
        return None

    # Low-mid band -> silent log, no question.
    if entropy < ENTROPY_MID:
        write_event(
            store,
            kind="curiosity_silent_log",
            data={
                "cue": cue[:200],
                "entropy": float(entropy),
                "source_ids": [str(h.record_id) for h in hits[:3]],
            },
            severity="info",
            session_id=session_id,
        )
        return None

    # Cooldown check.
    last = _last_curiosity_turn(store, session_id)
    if last is not None and (turn - last) < COOLDOWN_TURNS:
        return None

    q_id = uuid4()
    if entropy < ENTROPY_HIGH:
        tier = "inline"
        text = f"I'm not fully sure -- did you mean {cue!r}?"
    else:
        tier = "question"
        text = f"Could you clarify: {cue!r}?"

    trigger_ids: list[UUID] = [h.record_id for h in hits[:5]]
    question = CuriosityQuestion(
        id=q_id,
        text=text,
        triggered_by_record_ids=trigger_ids,
        entropy=float(entropy),
        tier=tier,
    )

    # curiosity_bridge edges. Delta proportional to entropy so higher-entropy
    # questions get stronger edges.
    # R3: batch all triggers into a single boost_edges call
    # (one merge_insert + one tbl.add at most). The diagnostic try/except
    # boundary is preserved at the SINGLE-call level — failure of the batched
    # write must never block the curiosity fire path.
    bridge_pairs = [(tid, q_id) for tid in trigger_ids]
    if bridge_pairs:
        try:
            store.boost_edges(
                bridge_pairs,
                edge_type="curiosity_bridge",
                delta=float(entropy),
            )
        except Exception:
            # Diagnostic; never block the curiosity fire on edge failure.
            pass

    write_event(
        store,
        kind="curiosity_question",
        data={
            "question_id": str(q_id),
            "text": text,
            "tier": tier,
            "entropy": float(entropy),
            "turn": int(turn),
            "triggered_by": [str(t) for t in trigger_ids],
        },
        severity="info",
        session_id=session_id,
        source_ids=trigger_ids,
    )
    return question


# ---------------------------------------------------------------- pending


def pending_questions(
    store: MemoryStore,
    session_id: str | None = None,
) -> list[CuriosityQuestion]:
    """Return unresolved curiosity questions, optionally scoped to a session."""
    events = query_events(store, kind="curiosity_question", limit=200)
    resolved_events = query_events(store, kind="curiosity_resolved", limit=500)
    resolved_ids = {
        r["data"].get("question_id")
        for r in resolved_events
        if r["data"].get("question_id")
    }
    out: list[CuriosityQuestion] = []
    for e in events:
        if session_id is not None and e.get("session_id") != session_id:
            continue
        data = e["data"]
        qid_raw = data.get("question_id")
        if not qid_raw:
            continue
        if qid_raw in resolved_ids:
            continue
        try:
            qid = UUID(qid_raw)
        except (TypeError, ValueError):
            continue
        triggered: list[UUID] = []
        for t in data.get("triggered_by", []):
            try:
                triggered.append(UUID(t))
            except (TypeError, ValueError):
                continue
        out.append(
            CuriosityQuestion(
                id=qid,
                text=data.get("text", ""),
                triggered_by_record_ids=triggered,
                entropy=float(data.get("entropy", 0.0)),
                tier=data.get("tier", "question"),
                resolved=False,
            )
        )
    return out
