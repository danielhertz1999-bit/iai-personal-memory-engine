from __future__ import annotations

from uuid import UUID

import numpy as np

from iai_mcp.types import MemoryRecord

VIGILANCE_RHO = 0.95


def cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def apply_art_gate(
    existing_records: list[MemoryRecord],
    new_record: MemoryRecord,
    rho: float = VIGILANCE_RHO,
) -> tuple[str, UUID]:
    best_sim: float = -1.0
    best_id: UUID | None = None
    for rec in existing_records:
        if rec.never_merge:
            continue
        sim = cosine(new_record.embedding, rec.embedding)
        if sim > best_sim:
            best_sim = sim
            best_id = rec.id
    if best_id is not None and best_sim >= rho:
        return ("merge", best_id)
    return ("create", new_record.id)


def _shield_tier_for_record(record: MemoryRecord):
    from iai_mcp.shield import ShieldTier

    if record.pinned or record.s5_trust_score >= 0.9:
        return ShieldTier.HARD_BLOCK
    if "profile" in (record.tags or []):
        return ShieldTier.FLAG_FOR_REVIEW
    return ShieldTier.LOG_ONLY


def guarded_insert(
    store,
    record: MemoryRecord,
    profile_state: dict,
    session_id: str = "-",
) -> tuple[bool, str]:
    from iai_mcp.s5 import check_identity_anchor_on_write
    from iai_mcp.shield import ShieldTier, apply_shield

    tier = _shield_tier_for_record(record)
    verdict = apply_shield(store, record, tier, session_id=session_id)
    if verdict.action == "reject":
        return False, f"shield: {verdict.reason}"
    flagged = verdict.action == "flag" and verdict.detected

    ok, reason = check_identity_anchor_on_write(store, record, profile_state)
    if not ok:
        return False, reason

    existing = store.all_records()
    gate_verdict, target = apply_art_gate(existing, record)
    if gate_verdict == "create":
        store.insert(record)
        return True, ("flagged" if flagged else "created")
    return True, f"merged_into:{target}"
