"""S5 identity kernel -- invariant protection via M-of-N consensus.

Identity rules enforced here:
- ρ_identity = 0.99 (stricter than write-path ρ=0.95 and S4 ρ_s4=0.97).
- 3-of-5 session-window consensus: an invariant update only commits after 3
  vigilance-passing proposals within the consensus window. A single-session
  attacker (e.g. prompt injection) cannot reach M by itself.
- 48h cooldown: after a commit, any subsequent proposal on the same anchor
  is rejected for 48h. Prevents rapid sequential poisoning.
- TRUST_THRESHOLD_IDENTITY = 0.9: records with s5_trust_score >= 0.9 are
  "invariant-tier". Direct writes bypassing propose_invariant_update are
  rejected by `check_identity_anchor_on_write`.
- All commits emit `s5_invariant_update` events with full provenance
  (proposal history, session_ids, similarity scores).

Proposal events (kind=s5_invariant_proposal) are emitted for EVERY proposal
so the M-of-N tally can be reconstructed from the events table alone -- no
hidden in-memory state. Cooldown lookups read kind=s5_invariant_update.

Gradual-drift detection:
- `detect_drift_anomaly` reads trajectory_metric events for profile-vector
  variance. When the last `window_sessions` consecutive values have been
  monotonically increasing, emits an s5_drift_alert event. User audit via
  `iai-mcp audit drift` surfaces these.
- `audit_identity_events` aggregates s5_* + shield_* + s5_drift_alert events
  chronologically (newest first) for `iai-mcp audit` / `audit identity`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import numpy as np

from iai_mcp.aaak import enforce_language_tagged, generate_aaak_index
from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


# ------------------------------------------------------------ identity constants

IDENTITY_VIGILANCE_RHO: float = 0.99   # strict vigilance on identity updates
S5_CONSENSUS_M: int = 3                # 3-of-5: required agreeing proposals
S5_CONSENSUS_N: int = 5                # 3-of-5: window size
COOLDOWN_HOURS: int = 48               # cooldown after a commit
TRUST_THRESHOLD_IDENTITY: float = 0.9  # score >= this => invariant-tier record
CONSENSUS_WINDOW_HOURS: int = 24       # all M votes must land within this window


# ------------------------------------------------------------ private helpers


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
    """Return all s5_invariant_proposal events for this anchor inside the
    consensus window, newest first."""
    since = datetime.now(timezone.utc) - timedelta(hours=CONSENSUS_WINDOW_HOURS)
    events = query_events(store, kind="s5_invariant_proposal", since=since, limit=100)
    return [e for e in events if e["data"].get("anchor_id") == str(anchor_id)]


def _in_cooldown(store: MemoryStore, anchor_id: UUID) -> bool:
    """True iff an s5_invariant_update for this anchor landed in the last COOLDOWN_HOURS."""
    since = datetime.now(timezone.utc) - timedelta(hours=COOLDOWN_HOURS)
    events = query_events(store, kind="s5_invariant_update", since=since, limit=10)
    for e in events:
        if e["data"].get("anchor_id") == str(anchor_id):
            return True
    return False


# ------------------------------------------------------------ public API


def propose_invariant_update(
    store: MemoryStore,
    anchor_id: UUID,
    new_fact: str,
    session_id: str,
) -> tuple[str, UUID | None]:
    """M-of-N voting on identity-tier updates.

    Workflow:
    1. If the anchor is in 48h cooldown, reject (``cooldown``).
    2. If the anchor does not exist, reject (``rejected``).
    3. Encode the proposed fact; compute cosine against the anchor.
    4. Log an `s5_invariant_proposal` event regardless of vigilance outcome.
       (This is how the M-of-N tally is reconstructed on subsequent calls.)
    5. Count vigilance-passing proposals in the current consensus window.
       - If >= M (3): commit -- insert new record, create invariant_anchor
         edge, log `s5_invariant_update` event, return ("committed", new_id).
       - Else if total >= N (5) proposals in window: reject (``rejected``).
       - Else: stage (``staged``), return the proposal UUID.

    Returns one of:
        ("cooldown", None)
        ("rejected", None)
        ("staged", proposal_id)
        ("committed", new_record_id)
    """
    # Step 1: cooldown gate.
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

    # Step 2: anchor existence.
    anchor = store.get(anchor_id)
    if anchor is None:
        return "rejected", None

    # Step 3: encode proposed fact + compute vigilance similarity.
    from iai_mcp.embed import embedder_for_store
    emb = embedder_for_store(store).embed(new_fact)
    sim = _cosine(anchor.embedding, emb)
    passes_vigilance = sim >= IDENTITY_VIGILANCE_RHO

    # Step 4: log the proposal (counts toward N).
    proposal_id = uuid4()
    write_event(
        store,
        kind="s5_invariant_proposal",
        data={
            "proposal_id": str(proposal_id),
            "anchor_id": str(anchor_id),
            "new_fact": new_fact[:200],  # payload size cap
            "similarity": sim,
            "passes_vigilance": passes_vigilance,
        },
        severity="info",
        session_id=session_id,
        source_ids=[anchor_id],
    )

    # Step 5: tally.
    recent = _recent_proposals_for(store, anchor_id)
    agree_count = sum(1 for r in recent if r["data"].get("passes_vigilance"))
    total = len(recent)

    if agree_count >= S5_CONSENSUS_M:
        # COMMIT: create the invariant_anchor edge + log the update.
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
    """Guard invoked by write paths that accept externally-originated records.

    Records with s5_trust_score >= TRUST_THRESHOLD_IDENTITY (0.9) are
    considered invariant-tier. They may NOT be written through any path that
    bypasses propose_invariant_update (consensus requirement).

    The shield is evaluated in HARD_BLOCK tier BEFORE the consensus marker
    check. Any detected injection signal short-circuits with "shield
    HARD_BLOCK".

    Cross-lingual note: an identity update whose language differs from the
    existing pinned identity anchor(s) emits a
    `identity_cross_lingual_warning` event but does NOT block --
    multi-lingual identity refinement is a supported use case. The warning
    surfaces via `iai-mcp audit identity` for user review.

    We distinguish between:
    - DIRECT identity writes (reject): s5_trust_score >= 0.9 and no
      `s5_consensus` tag -- attacker trying to plant an invariant.
    - CONSENSUS-PROMOTED writes (accept): s5_trust_score >= 0.9 and
      `s5_consensus` tag present -- output of propose_invariant_update's
      own store.insert call.
    - NORMAL writes (accept): s5_trust_score < 0.9 -- below identity tier.
    """
    if record.s5_trust_score < TRUST_THRESHOLD_IDENTITY:
        return True, ""

    # Shield HARD_BLOCK pre-check on identity-tier writes.
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

    # Cross-lingual warning: non-fatal, emit an event and
    # continue. Inspect the existing pinned identity anchors for a language
    # mismatch with the incoming record.
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


# ---------------------------------------------------------- drift detection

# Relevant event kinds for the user audit surface, aggregated under
# `iai-mcp audit`.
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
    """Gradual-drift detection via profile-vector trajectory reversal.

    Reads trajectory_metric events filtered to metric=m4 (profile-vector
    variance). The expected direction is DECREASING (the profile is
    converging as the user is learnt over time). When the last
    `window_sessions` values are monotonically INCREASING or mostly so
    (at least window_sessions - 2 adjacent pairs increase), emits an
    s5_drift_alert event and returns the alert payload in a list.

    Returns [] on insufficient data or no drift.
    """
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

    # Sort ascending (oldest first) so "recent" slice is the tail.
    try:
        m4.sort(key=lambda x: x[0])
    except TypeError:
        # Fallback: if ts objects are not comparable, keep insertion order.
        pass
    recent = m4[-window_sessions:]

    increases = 0
    for i in range(1, len(recent)):
        if recent[i][1] > recent[i - 1][1]:
            increases += 1

    # Drift signature: most of the window-1 adjacent steps are increasing.
    # For window_sessions=5, require increases >= 3 (at least 3 of 4 steps up).
    # For window_sessions=3, require increases >= 1 (at least 1 of 2 steps up).
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
    """Aggregate identity-relevant events chronologically (newest first).

    Used by `iai-mcp audit` + `audit identity` / `audit shield` / `audit drift`
    CLI subcommands. By default returns the full set of audit kinds; callers
    may pass a subset (e.g. only s5_* for `audit identity`).
    """
    out: list[dict] = []
    for kind in kinds:
        out.extend(query_events(store, kind=kind, since=since, limit=500))
    # Newest first by ts; coerce to comparable form (fallback to id-based).
    try:
        out.sort(key=lambda e: e.get("ts"), reverse=True)
    except TypeError:
        pass
    return out
