from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import numpy as np

from iai_mcp.aaak import enforce_language_tagged, generate_aaak_index
from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


IDENTITY_VIGILANCE_RHO: float = 0.99
S5_CONSENSUS_M: int = 3
S5_CONSENSUS_N: int = 5
COOLDOWN_HOURS: int = 48
TRUST_THRESHOLD_IDENTITY: float = 0.9
CONSENSUS_WINDOW_HOURS: int = 24


def _cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def _recent_proposals_for(
    store: MemoryStore, anchor_id: UUID,
) -> list[dict]:
    since = datetime.now(timezone.utc) - timedelta(hours=CONSENSUS_WINDOW_HOURS)
    events = query_events(store, kind="s5_invariant_proposal", since=since, limit=100)
    return [e for e in events if e["data"].get("anchor_id") == str(anchor_id)]


def _in_cooldown(store: MemoryStore, anchor_id: UUID) -> bool:
    since = datetime.now(timezone.utc) - timedelta(hours=COOLDOWN_HOURS)
    events = query_events(store, kind="s5_invariant_update", since=since, limit=10)
    for e in events:
        if e["data"].get("anchor_id") == str(anchor_id):
            return True
    return False


def propose_invariant_update(
    store: MemoryStore,
    anchor_id: UUID,
    new_fact: str,
    session_id: str,
) -> tuple[str, UUID | None]:
    if _in_cooldown(store, anchor_id):
        write_event(
            store,
            kind="s5_cooldown_block",
            data={"anchor_id": str(anchor_id), "session_id": session_id},
            severity="warning",
            session_id=session_id,
            source_ids=[anchor_id],
        )
        return "cooldown", None

    anchor = store.get(anchor_id)
    if anchor is None:
        return "rejected", None

    from iai_mcp.embed import embedder_for_store
    emb = embedder_for_store(store).embed(new_fact)
    sim = _cosine(anchor.embedding, emb)
    passes_vigilance = sim >= IDENTITY_VIGILANCE_RHO

    proposal_id = uuid4()
    write_event(
        store,
        kind="s5_invariant_proposal",
        data={
            "proposal_id": str(proposal_id),
            "anchor_id": str(anchor_id),
            "new_fact": new_fact[:200],
            "similarity": sim,
            "passes_vigilance": passes_vigilance,
        },
        severity="info",
        session_id=session_id,
        source_ids=[anchor_id],
    )

    recent = _recent_proposals_for(store, anchor_id)
    agree_count = sum(1 for r in recent if r["data"].get("passes_vigilance"))
    total = len(recent)

    if agree_count >= S5_CONSENSUS_M:
        now = datetime.now(timezone.utc)
        updated = MemoryRecord(
            id=uuid4(),
            tier=anchor.tier,
            literal_surface=new_fact,
            aaak_index="",
            embedding=emb,
            community_id=anchor.community_id,
            centrality=anchor.centrality,
            detail_level=anchor.detail_level,
            pinned=anchor.pinned,
            stability=anchor.stability,
            difficulty=anchor.difficulty,
            last_reviewed=now,
            never_decay=True,
            never_merge=True,
            provenance=[
                {
                    "ts": now.isoformat(),
                    "cue": "s5_consensus",
                    "session_id": session_id,
                }
            ],
            created_at=now,
            updated_at=now,
            tags=[*anchor.tags, "s5_consensus"],
            language=anchor.language or "en",
            s5_trust_score=min(1.0, anchor.s5_trust_score + 0.05),
            profile_modulation_gain=dict(anchor.profile_modulation_gain),
            schema_version=2,
        )
        enforce_language_tagged(updated)
        updated.aaak_index = generate_aaak_index(updated)
        store.insert(updated)
        store.boost_edges(
            [(anchor_id, updated.id)],
            edge_type="invariant_anchor",
            delta=1.0,
        )
        write_event(
            store,
            kind="s5_invariant_update",
            data={
                "anchor_id": str(anchor_id),
                "new_record_id": str(updated.id),
                "session_ids": [r["session_id"] for r in recent],
                "agree_count": agree_count,
                "total_proposals": total,
                "similarity": sim,
            },
            severity="info",
            session_id=session_id,
            source_ids=[anchor_id, updated.id],
        )
        return "committed", updated.id

    if total >= S5_CONSENSUS_N:
        return "rejected", None

    return "staged", proposal_id


def check_identity_anchor_on_write(
    store: MemoryStore,
    record: MemoryRecord,
    profile_state: dict,
) -> tuple[bool, str]:
    if record.s5_trust_score < TRUST_THRESHOLD_IDENTITY:
        return True, ""

    from iai_mcp.shield import ShieldTier, evaluate_injection_risk

    shield_verdict = evaluate_injection_risk(
        record.literal_surface or "",
        ShieldTier.HARD_BLOCK,
        target_language=record.language or None,
    )
    if shield_verdict.action == "reject":
        return (
            False,
            f"shield HARD_BLOCK: {shield_verdict.reason}",
        )

    if "s5_consensus" not in (record.tags or []):
        return (
            False,
            "identity-tier write (s5_trust_score >= 0.9) requires "
            "propose_invariant_update consensus; direct inserts forbidden.",
        )

    try:
        anchors_with_other_lang = [
            r for r in store.all_records()
            if r.pinned
            and r.s5_trust_score >= TRUST_THRESHOLD_IDENTITY
            and (r.language or "") != ""
            and (r.language or "") != (record.language or "")
        ]
    except (OSError, RuntimeError, ValueError):
        anchors_with_other_lang = []
    if anchors_with_other_lang:
        anchor_langs = sorted({
            r.language for r in anchors_with_other_lang if r.language
        })
        write_event(
            store,
            kind="identity_cross_lingual_warning",
            data={
                "record_id": str(record.id),
                "record_language": record.language,
                "existing_anchor_languages": anchor_langs,
            },
            severity="warning",
            session_id="-",
            source_ids=[record.id],
        )

    return True, ""


AUDIT_EVENT_KINDS: tuple[str, ...] = (
    "s5_invariant_update",
    "s5_invariant_proposal",
    "s5_cooldown_block",
    "s5_drift_alert",
    "shield_rejection",
    "shield_flag",
    "identity_cross_lingual_warning",
)


def detect_drift_anomaly(
    store: MemoryStore,
    window_sessions: int = 5,
) -> list[dict]:
    events = query_events(store, kind="trajectory_metric", limit=1000)
    m4: list[tuple] = []
    for e in events:
        data = e.get("data") or {}
        if data.get("metric") != "m4":
            continue
        try:
            v = float(data.get("value", 0.0))
        except (TypeError, ValueError):
            continue
        ts = e.get("ts")
        m4.append((ts, v))

    if len(m4) < window_sessions:
        return []

    try:
        m4.sort(key=lambda x: x[0])
    except TypeError:
        pass
    recent = m4[-window_sessions:]

    increases = 0
    for i in range(1, len(recent)):
        if recent[i][1] > recent[i - 1][1]:
            increases += 1

    threshold = max(1, window_sessions - 2)
    if increases < threshold:
        return []

    alert = {
        "kind": "s5_drift_alert",
        "severity": "warning",
        "window_sessions": window_sessions,
        "increases": increases,
        "first_value": float(recent[0][1]),
        "last_value": float(recent[-1][1]),
    }
    write_event(
        store,
        kind="s5_drift_alert",
        data={
            "window_sessions": window_sessions,
            "increases": increases,
            "first_value": alert["first_value"],
            "last_value": alert["last_value"],
        },
        severity="warning",
    )
    return [alert]


def audit_identity_events(
    store: MemoryStore,
    since: datetime | None = None,
    kinds: tuple[str, ...] = AUDIT_EVENT_KINDS,
) -> list[dict]:
    out: list[dict] = []
    for kind in kinds:
        out.extend(query_events(store, kind=kind, since=since, limit=500))
    try:
        out.sort(key=lambda e: e.get("ts"), reverse=True)
    except TypeError:
        pass
    return out
