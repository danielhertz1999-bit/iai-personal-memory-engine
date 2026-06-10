from __future__ import annotations

import math
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore


ENTROPY_LOW: float = 0.4
ENTROPY_MID: float = 0.7
ENTROPY_HIGH: float = 0.9
COOLDOWN_TURNS: int = 3


@dataclass
class CuriosityQuestion:

    id: UUID
    text: str
    triggered_by_record_ids: list[UUID] = field(default_factory=list)
    entropy: float = 0.0
    tier: str = "question"
    resolved: bool = False


def compute_entropy(scores: list[float]) -> float:
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
    events = query_events(store, kind="curiosity_question", limit=20)
    for e in events:
        if e.get("session_id") == session_id:
            try:
                return int(e["data"].get("turn", 0))
            except (TypeError, ValueError):
                return None
    return None


def fire_curiosity(
    store: MemoryStore,
    hits: list,
    cue: str,
    entropy: float,
    session_id: str,
    turn: int,
) -> CuriosityQuestion | None:
    if entropy < ENTROPY_LOW:
        return None

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

    bridge_pairs = [(tid, q_id) for tid in trigger_ids]
    if bridge_pairs:
        try:
            store.boost_edges(
                bridge_pairs,
                edge_type="curiosity_bridge",
                delta=float(entropy),
            )
        except (OSError, RuntimeError, ValueError):
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


def pending_questions(
    store: MemoryStore,
    session_id: str | None = None,
) -> list[CuriosityQuestion]:
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


def get_pending_questions(store: MemoryStore, limit: int = 2) -> list[dict]:
    qs = pending_questions(store)
    return [
        {"text": q.text, "entropy": q.entropy, "tier": q.tier}
        for q in qs[:limit]
    ]
