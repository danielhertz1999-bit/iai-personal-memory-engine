"""ART vigilance write gate (, ) + S5 identity guard (, )
+ prompt-injection shield (, , ).

Grossberg-style Adaptive Resonance Theory vigilance: on write, compare the new
record against existing records by cosine similarity. If the best match exceeds
vigilance ρ, merge; else create a new distinct record.

ρ is fixed at 0.95 for per (matches autistic-kernel literal_preservation=strong).
High ρ = prefer distinct record over merge = preserves fine detail.

adds `guarded_insert` which layers the S5 identity gate on top of
the ART decision. Identity-tier records (s5_trust_score >= 0.9) must carry
the `s5_consensus` tag -- direct writes are rejected to prevent prompt-
injection poisoning.

extends `guarded_insert` with a shield pre-check ( / ):
the tier is determined from record properties, and the shield is consulted
BEFORE the S5 gate. HARD_BLOCK rejects propagate as (False, "shield: ...");
FLAG and LOG tiers emit events but allow the write to proceed.
"""
from __future__ import annotations

from uuid import UUID

import numpy as np

from iai_mcp.types import MemoryRecord

# fixed ρ for (matches literal_preservation=strong in autistic kernel).
# DO NOT CHANGE without updating tests.
VIGILANCE_RHO = 0.95  # float constant -- plan acceptance criterion greps for exact literal


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if either vector is zero-norm."""
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
    """Return ('create', new_record.id) or ('merge', target_record_id).

    Skips any existing record with `never_merge=True` ( pinned-L0 guarantee):
    even if the input matches L0 perfectly, L0 is never overwritten.

    Args:
        existing_records: candidates to compare against.
        new_record: the write-candidate to admit.
        rho: vigilance threshold. Defaults to VIGILANCE_RHO (0.95).

    Returns:
        ("create", new_record.id) if novelty > (1 - rho), else ("merge", target_id).
    """
    best_sim: float = -1.0
    best_id: UUID | None = None
    for rec in existing_records:
        if rec.never_merge:
            continue  # L0 and other pinned-immutable records are skipped
        sim = cosine(new_record.embedding, rec.embedding)
        if sim > best_sim:
            best_sim = sim
            best_id = rec.id
    if best_id is not None and best_sim >= rho:
        return ("merge", best_id)
    return ("create", new_record.id)


def _shield_tier_for_record(record: MemoryRecord):
    """tier determination.

    HARD_BLOCK: pinned records OR s5_trust_score >= 0.9 (identity-tier)
    FLAG_FOR_REVIEW: records tagged "profile" (profile-knob updates)
    LOG_ONLY: everything else (content records)
    """
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
    """Central write gate combining shield pre-check + S5 identity check + ART gate.

    (, ): determine the shield tier from the record
    (HARD_BLOCK for pinned/identity-tier, FLAG for profile, LOG for content),
    evaluate the shield, then:
      - HARD_BLOCK + detection -> reject (shield_rejection event already logged)
      - FLAG + detection        -> proceed (shield_flag event already logged)
      - LOG + detection         -> proceed (shield_log event already logged)

    identity-tier records (s5_trust_score >= 0.9)
    must pass through propose_invariant_update. Direct writes -- via this
    function, the MCP surface, or any other write path -- are rejected unless
    they carry the `s5_consensus` marker tag.

    Below-identity writes (s5_trust_score < 0.9) fall through the ART gate.
    Currently we use the existing Phase-1 behaviour (create-or-merge) and
    report the outcome via the return tuple. Callers receive:
        (True, "created")          -- store.insert succeeded, distinct record
        (True, "merged_into:<id>") -- ART gate merged into an existing record
        (True, "flagged")          -- shield FLAG tier matched; write still proceeded
        (False, reason)            -- shield OR S5 blocked the write
    """
    # Lazy imports so write.py doesn't pull events/numpy into every read path.
    from iai_mcp.s5 import check_identity_anchor_on_write
    from iai_mcp.shield import ShieldTier, apply_shield

    # shield pre-check.
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
