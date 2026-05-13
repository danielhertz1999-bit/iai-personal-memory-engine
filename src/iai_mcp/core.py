"""JSON-RPC core for IAI-MCP.

Binds the Phase-1 MCP tools to the Python internals. The TypeScript MCP
wrapper spawns this module as a subprocess (`python -m iai_mcp.core`) and forwards
line-delimited JSON-RPC 2.0 requests over stdio.

Boot sequence:
1. Open MemoryStore at ~/.iai-mcp/lancedb (, )
2. Seed pinned L0 identity record if absent (, ), stamping its aaak_index
3. Loop: read JSON line from stdin, dispatch, write JSON-RPC response to stdout.

All writes are synchronous.

rewires the profile branches to read `iai_mcp.profile.PROFILE_KNOBS`
(the full 11-knob registry: 10 AUTIST + 1 wake_depth, ; -02
removed AUTIST-02/08/11/12 dead knobs), replacing the inline LIVE_KNOBS/
DEFERRED_KNOBS dict. The old names `LIVE_KNOBS` / `DEFERRED_KNOBS`
/ `L0_ID` are re-exported for backwards compatibility with test
suite -- they now point at the authoritative registry state rather than
local copies.

adds real CLS sleep cycle + S5 identity kernel
dispatch:
- `memory_consolidate`: real heavy consolidation (replaces stub)
- `session_exit`: light consolidation
- `s5_propose`: M-of-N voting on invariant updates
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import traceback
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from iai_mcp import profile, retrieve
from iai_mcp.aaak import enforce_english_raw, generate_aaak_index
from iai_mcp.concurrency import SOCKET_PATH
from iai_mcp.daemon_state import get_pending_digest, load_state
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# ----------------------------------------------------- -02 V3-03 fix
class UnknownMethodError(Exception):
    """Raised by ``core.dispatch`` when the requested method name is not
    in the dispatch chain.

    Trigger: the if/elif method == "..." chain falls through without
    matching. ``e.args[0]`` is the offending method name.

    Mapped by ``socket_server.handle`` to JSON-RPC error code -32601
    ERR_METHOD_NOT_FOUND with message ``"unknown method '<name>'"``.

    Subclasses ``Exception`` (not ``RuntimeError``) because an unknown
    method is a routine client error, not a "should be impossible"
    invariant violation. Compare ``crypto.CryptoKeyError(RuntimeError)``
    which IS an invariant-class failure.
    """


# --------------------------------------------------------- constants
# cooperative force-wake cap. Daemon completes at most one 15-min REM
# cycle before yielding; the JSON-RPC caller waits up to this long before
# giving up with a "timeout" response.
FORCE_WAKE_TIMEOUT_SEC: int = 15 * 60  # 900s


# ----------------------------------------------------------- cross-process LRU
#
# The sleep daemon owns its own HIPPEA cascade LRU (hippea_cascade._warm_lru).
# The MCP core runs in a different process; that LRU is invisible across the
# process boundary. ``snapshot_warm_ids()`` returns [] in core on every fresh
# boot, so ``_first_turn_recall_hook`` has no daemon-side warm-up to consult.
#
# Closure: core maintains its OWN, process-local LRU here. When
# ``_first_turn_recall_hook`` sees an empty daemon snapshot, it runs a
# synchronous cascade once per session and populates ``_CORE_WARM_LRU``.
# Subsequent recalls in the same session reuse the warmed records via the
# normal ``get_warm_record(rid)`` lookup path.
#
# C1 (read-only): compute_core_side_warm_snapshot touches store only via
# ``store.get`` -- no mutation.
# C3 (zero API): no paid-API calls; salience is pure-local.
# C6 (no writes): cascade produces record ids only; LRU writes are per-process
# RAM, not store-backed.
from cachetools import TTLCache as _CoreTTLCache

_CORE_WARM_LRU: _CoreTTLCache = _CoreTTLCache(maxsize=50, ttl=300)
_CORE_CASCADE_FIRED_PER_SESSION: set[str] = set()


# ----------------------------------------------------------------- knob state
# Per-process mutable profile state initialised from profile.default_state().
# profile_get / profile_set both read and write this dict.
_profile_state: dict[str, Any] = profile.default_state()

# LEARN-01 posterior state accumulator. Keyed by knob name,
# each entry carries conjugate-prior state (alpha/beta for bool, alphas for
# enum, weighted_sum/total_weight/mean for float/int, per_key for dict).
_posterior_state: dict[str, Any] = {}

# RESEARCH §1 Option B: serialize mutations to module-level state across
# concurrent socket-driven dispatch calls. Read-only paths do NOT acquire this
# lock — the GIL keeps individual dict ops atomic; only read-modify-write
# sequences (profile_set, profile_update_from_signal) need it. MUST be
# threading.RLock (re-entrant, sync) because Wave 2's socket_server invokes
# `dispatch` via `await asyncio.to_thread(...)`, so the lock is acquired from
# a thread-pool worker where asyncio primitives are unreachable. Re-entrancy
# means a guarded helper that calls another guarded helper in the same thread
# does not deadlock.
_profile_lock: threading.RLock = threading.RLock()

# exposed two module-level names that test_hebbian.py imports:
# `LIVE_KNOBS` (mutable dict) and `DEFERRED_KNOBS` (frozenset). Preserve them as
# aliases/derivations of the new registry so the tests keep working.
LIVE_KNOBS: dict[str, Any] = _profile_state  # mutating LIVE_KNOBS still mutates state
DEFERRED_KNOBS: frozenset[str] = frozenset(
    profile.PHASE_2_DEFERRED | profile.PHASE_3_DEFERRED
)
# flipped the 9 Phase-2 knobs to phase=1.
# FLIPS the final camouflaging_relaxation knob to phase=1.
# REMOVED 4 dead KnobSpec entries (AUTIST-02/08/11/12) — 10
# autistic-kernel knobs are now live and DEFERRED_KNOBS is empty.
assert len(DEFERRED_KNOBS) == 0, ": all 10 autistic-kernel knobs live"


# ----------------------------------------------------------------------- seed
# deterministic L0 UUID so seed idempotency check is cheap and cross-process
# stable. session-start assembler reads this record by UUID.
L0_ID = UUID("00000000-0000-0000-0000-000000000001")


def _seed_l0_identity(store: MemoryStore) -> None:
    """Seed the pinned L0 identity record (, continuity seed).

    Idempotent: returns immediately if L0_ID already exists. Called once at core
    boot. re-embeds this record with the configured embedder
    (bge-small-en-v1.5 by default per ); stamps its aaak_index
    via generate_aaak_index so the session-start manifest can reference the L0
    metadata without leaking literal_surface content.

    the seed carries language="en". made the
    English-Only Brain canonical; new records always default to "en".
    """
    existing = store.get(L0_ID)
    if existing is not None:
        return
    now = datetime.now(timezone.utc)
    # Resolve the store's current embedding dimension so the zero-vector matches.
    seed_dim = store.embed_dim
    seed = MemoryRecord(
        id=L0_ID,
        tier="semantic",
        literal_surface=(
            "User identity: not yet configured. "
            "IAI-MCP defaults: literal_preservation=strong, masking_off=true, "
            "task_support=cued_recognition, scene_construction_scaffold=on. "
            "The system will learn about the user from session transcripts."
        ),
        aaak_index="",
        embedding=[0.0] * seed_dim, # re-embeds via graph reconstruction
        community_id=None,
        centrality=1.0,               # treat as max-central pin
        detail_level=5,
        pinned=True,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,             # ART gate must never overwrite L0
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["identity", "l0", "pinned"],
        language="en",                # L0 identity text is English
    )
    # constitutional guard -- ASCII English identity passes cleanly.
    enforce_english_raw(seed)
    # metadata stamp so session-start assembler has a populated aaak_index.
    seed.aaak_index = generate_aaak_index(seed)
    store.insert(seed)


# ------------------------------------------------------------- JSON-RPC layer

def dispatch(store: MemoryStore, method: str, params: dict) -> dict:
    """Route a single JSON-RPC method to the corresponding internal function.

    Tool contract per . Profile knob split per .
    """
    if method == "memory_recall":
        # R4: classify the cue BEFORE choosing the recall
        # path so both the empty-store fallback and the full pipeline see
        # the same mode. The classifier reads only the cue text (regex on
        # surface signals — quoted phrases, EN word-markers, RU starts-with
        # triggers) and returns ('verbatim' | 'concept', triggered_pattern).
        # The triggered_pattern is for diagnostic logging only; only the
        # mode string flows downstream.
        from iai_mcp.cue_router import _classify_cue
        cue_mode, _triggered_pattern = _classify_cue(params.get("cue", ""))

        # -03 BLOCKER 3: seed the audit accumulator BEFORE recall
        # fires its gain branches. Threaded into recall_for_response and
        # mutated in place by profile.py:profile_modulation_for_record
        # (AUTIST-01/03/09 entries) and by apply_profile below (helper-keyed
        # AUTIST entries). Attached to the response so MCP callers can
        # audit which knobs actually consulted/mutated the recall.
        knobs_applied: dict[str, str] = {}
        # wake_depth seed: operator-facing knob; provenance points
        # into session.py:373 (assemble_session_start: wake_depth = state.get(...)).
        _wake_depth_value = (_profile_state or {}).get("wake_depth", "minimal")
        if _wake_depth_value not in ("minimal", "standard", "deep"):
            _wake_depth_value = "minimal"
        knobs_applied["MCP-12"] = (
            f"session.py:assemble_session_start:wake_depth={_wake_depth_value}"
        )

        # dispatch: non-empty store -> 5-stage pipeline;
        # empty store -> baseline cosine recall (fallback).
        records_count = store.db.open_table("records").count_rows()
        if records_count == 0:
            cue_embedding = params.get("cue_embedding") or [0.0] * EMBED_DIM
            resp = retrieve.recall(
                store=store,
                cue_embedding=cue_embedding,
                cue_text=params["cue"],
                session_id=params.get("session_id", "unknown"),
                budget_tokens=params.get("budget_tokens", 1500),
                # R4: thread classified mode into the baseline
                # fallback so the degraded path honours the same contract
                # (verbatim cue → episodic-only candidates regardless of
                # which route core dispatched to).
                mode=cue_mode,
            )
        else:
            from iai_mcp.embed import embedder_for_store
            from iai_mcp.pipeline import recall_for_response
            # R7: defensive try/except around the full-pipeline
            # branch so a graph-build failure (cache miss + corruption,
            # community detection error, OOM, etc.) routes to the baseline
            # fallback with the classified mode preserved. Pre-Plan-06-04
            # the exception propagated and crashed the JSON-RPC loop with
            # a -32000 error; 's North-Star ≥99% essential variable is
            # better defended by a degraded surface than by no response.
            try:
                graph, assignment, rc = retrieve.build_runtime_graph(store)
                embedder = embedder_for_store(store)
                # R3: thread the per-process profile state into
                # recall_for_response (entry-point split;
                # mode-dependent bias receives `mode=cue_mode` from cue-classifier
                # unchanged) so the rank stage can read literal_preservation
                # and any other knob-derived modulators. Pre-Plan-06-03
                # dispatch was silently dropping profile_state — the
                # literal_preservation knob was dead in production for the
                # entire history of the project.
                resp = recall_for_response(
                    store=store,
                    graph=graph,
                    assignment=assignment,
                    rich_club=rc,
                    embedder=embedder,
                    cue=params["cue"],
                    session_id=params.get("session_id", "unknown"),
                    budget_tokens=params.get("budget_tokens", 1500),
                    profile_state=_profile_state,
                    # R4: thread classified mode into recall_for_response
                    # so verbatim cues drive the verbatim mode behaviour
                    # (episodic-only candidates, zero W_DEGREE, no schema surface).
                    # the entry-point split preserves this mode
                    # plumbing verbatim — _recall_core receives `mode` unchanged
                    # for the mode-dependent gate bias.
                    mode=cue_mode,
                    # -03 BLOCKER 3: thread audit accumulator into
                    # the gains-application path so AUTIST-01/03/09 record
                    # provenance into the same dict attached to the response.
                    knobs_applied=knobs_applied,
                )
            except Exception:
                # R7 + graph-build / pipeline failure fallback.
                # Keep the classified mode — verbatim default protects the
                # North-Star essential variable on the degraded path.
                cue_embedding = params.get("cue_embedding") or [0.0] * EMBED_DIM
                resp = retrieve.recall(
                    store=store,
                    cue_embedding=cue_embedding,
                    cue_text=params["cue"],
                    session_id=params.get("session_id", "unknown"),
                    budget_tokens=params.get("budget_tokens", 1500),
                    mode=cue_mode,
                )
        response = {
            "hits": [_hit_to_json(h) for h in resp.hits],
            "anti_hits": [_hit_to_json(h) for h in resp.anti_hits],
            "activation_trace": [str(x) for x in resp.activation_trace],
            "budget_used": resp.budget_used,
            # surface the new RecallResponse fields on the
            # JSON-RPC response so MCP callers see the classified mode
            # (verbatim/concept) and any displaced concept-mode schema
            # records (patterns_observed[], max 3 entries).
            "cue_mode": resp.cue_mode,
            "patterns_observed": list(resp.patterns_observed or []),
            # -03 BLOCKER 3: attach the audit accumulator to the
            # response. Already populated by recall_for_response upstream
            # (AUTIST-01/03/09 from profile.py + wake_depth seed
            # above); apply_profile below extends the same dict in place
            # with helper-keyed AUTIST entries (CONTEXT ).
            "_knobs_applied": knobs_applied,
        }
        # inject sleep_suggestion when dual-gate passes.
        _inject_sleep_suggestion(
            response,
            cue=params.get("cue", ""),
            language=params.get("language", "en"),
        )
        # first memory_recall of the day
        # (>18h since last shown OR never shown) carries the overnight
        # digest. daemon_state.get_pending_digest clears the digest from
        # state so it appears exactly once per 18h window.
        _inject_overnight_digest(response, store=store)
        # / D5-03: first-turn auto-recall hook. Fires
        # exactly once per session; runs a scoped recall and injects
        # `first_turn_recall` field. Silent-fail.
        _first_turn_recall_hook(response, params=params, store=store)
        # / D5-04: server-side profile knob decorator.
        # Knob names never cross the MCP wire.
        try:
            from iai_mcp.response_decorator import apply_profile
            apply_profile(response, _profile_state)
        except Exception:
            pass  # decorator must not break the hot path
        return response

    # --- dispatch (TEM factorization) ---
    # memory_recall_structural: structural query enters the pipeline via
    # role->filler dict. Pure numpy + bytewise XOR -- ZERO LLM token cost,
    # no Embedder() instantiated, no anthropic client touched. Constitutional
    # contract: structural queries are first-class peers of cosine, NOT a
    # "VSA retrieval layer over cosine."
    if method == "memory_recall_structural":
        from iai_mcp import tem
        from iai_mcp.hebbian_structure import structural_similarity
        from iai_mcp.types import STRUCTURE_HV_BYTES

        structure_query: dict = params.get("structure_query") or {}
        budget_tokens = int(params.get("budget_tokens", 2000))
        max_records = int(params.get("max_records", 5000))
        if max_records < 1:
            max_records = 5000
        if max_records > 50_000:
            max_records = 50_000

        # Build query hypervector via tem.pack_pairs over (role, filler_hv).
        if structure_query:
            query_pairs = [
                (str(role), tem.filler_hv(str(value)))
                for role, value in structure_query.items()
            ]
            query_hv = tem.pack_pairs(query_pairs)
        else:
            query_hv = bytes(STRUCTURE_HV_BYTES)

        records = store.all_records()
        if len(records) > max_records:
            records = records[:max_records]
        scored: list[tuple[float, "object"]] = []
        for rec in records:
            if not rec.structure_hv:
                continue
            sim = structural_similarity(query_hv, rec.structure_hv)
            scored.append((sim, rec))
        scored.sort(key=lambda x: x[0], reverse=True)

        hits_out: list[dict] = []
        budget_used = 0
        for sim, rec in scored:
            tokens = max(1, len(rec.literal_surface) // 4)
            if budget_used + tokens > budget_tokens and hits_out:
                break
            hits_out.append({
                "record_id": str(rec.id),
                "score": float(sim),
                "reason": f"structural similarity {sim:.3f} (D=10000 BSC Hamming)",
                "literal_surface": rec.literal_surface,
                "adjacent_suggestions": [],
            })
            budget_used += tokens

        return {
            "hits": hits_out,
            "anti_hits": [],
            "activation_trace": [],
            "budget_used": budget_used,
            "structural_query_size": len(structure_query),
        }
    # --- / dispatch ---

    if method == "memory_reinforce":
        ids = [UUID(x) for x in params["ids"]]
        upd = retrieve.reinforce_edges(store, ids)
        return {
            "edges_boosted": upd.edges_boosted,
            "new_weights": upd.new_weights,
        }

    if method == "memory_contradict":
        cue_embedding = params.get("cue_embedding") or [0.0] * EMBED_DIM
        rec = retrieve.contradict(
            store, UUID(params["id"]), params["new_fact"], cue_embedding
        )
        return {
            "original_id": str(rec.original_id),
            "new_record_id": str(rec.new_record_id),
            "edge_type": rec.edge_type,
            "ts": rec.ts.isoformat(),
        }

    # --- WRITE-side ambient capture (conversation -> store) ---
    if method == "memory_capture":
        from iai_mcp.capture import capture_turn
        return capture_turn(
            store,
            cue=params.get("cue", ""),
            text=params["text"],
            tier=params.get("tier", "episodic"),
            session_id=params.get("session_id", "-"),
            role=params.get("role", "user"),
        )

    # --- dispatch ---
    # replaces 's memory_consolidate stub with real sleep
    # cycle dispatch. The tool signature stays compatible:
    # {"method":"memory_consolidate","params":{"session_id": "..."}}
    if method == "memory_consolidate":
        from iai_mcp.guard import BudgetLedger, RateLimitLedger
        from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

        cfg = SleepConfig()  # defaults are MANUAL-friendly; llm_enabled=False
        budget = BudgetLedger(store)
        rate = RateLimitLedger(store)
        result = run_heavy_consolidation(
            store,
            session_id=params.get("session_id", "-"),
            config=cfg,
            budget=budget,
            rate=rate,
            has_api_key=bool(os.environ.get("ANTHROPIC_API_KEY")),
        )
        # Normalise JSON-friendly output (no dataclasses).
        return {
            "mode": result["mode"],
            "tier": result["tier"],
            "summaries_created": int(result["summaries_created"]),
            "decay_result": dict(result["decay_result"]),
            "schema_candidates": list(result["schema_candidates"]),
        }

    # light consolidation entry point.
    # extends session_exit to also emit M1..M6 trajectory events.
    if method == "session_exit":
        from iai_mcp.sleep import run_light_consolidation
        from iai_mcp.trajectory import (
            compute_session_metrics_snapshot,
            record_session_metrics,
        )

        sid = params.get("session_id", "-")
        result = run_light_consolidation(store, session_id=sid)
        # trajectory emission.
        snapshot = compute_session_metrics_snapshot(store, sid)
        record_session_metrics(store, session_id=sid, metrics=snapshot)
        result["trajectory_metrics_emitted"] = len(snapshot)
        return result

    # S5 identity kernel. Internal method -- not
    # advertised on the MCP tools/list surface yet (adds that),
    # but the dispatch hook is live so tests and subagents can call it.
    if method == "s5_propose":
        from iai_mcp.s5 import propose_invariant_update

        verdict, pid = propose_invariant_update(
            store,
            UUID(params["anchor_id"]),
            params["new_fact"],
            params.get("session_id", "-"),
        )
        return {
            "verdict": verdict,
            "proposal_id": str(pid) if pid is not None else None,
        }
    # --- /dispatch ---

    # --- dispatch ---
    # adds four internal methods tied to the learning layer:
    #
    # - profile_update_from_signal: LEARN-01 Bayesian update; accepts
    #   {knob, signal, observed} and mutates _profile_state + _posterior_state.
    # - schema_induce: LEARN-03 manual trigger for Tier-0 fallback; returns
    #   the SchemaCandidate list without persisting.
    # - curiosity_pending: surface; returns unresolved curiosity
    #   questions optionally filtered by session_id.
    # - trajectory_record: LEARN-07 ; writes M1..M6 events for a session.
    if method == "profile_update_from_signal":
        from iai_mcp.profile import bayesian_update

        global _posterior_state
        knob = params["knob"]
        signal = params["signal"]
        observed = params["observed"]
        # serialize the read-modify-write of _profile_state (mutated
        # in-place by bayesian_update) and the rebind of _posterior_state.
        # See _profile_lock declaration above for the choice of threading.RLock
        # over an asyncio primitive (rationale lives in the lock docstring).
        with _profile_lock:
            new_val, new_post = bayesian_update(
                knob, signal, observed, _profile_state, _posterior_state,
            )
            _posterior_state = new_post
        return {"new_value": new_val, "knob": knob, "signal": signal}

    if method == "schema_induce":
        from iai_mcp.guard import BudgetLedger, RateLimitLedger
        from iai_mcp.schema import induce_schemas_tier1

        budget = BudgetLedger(store)
        rate = RateLimitLedger(store)
        candidates = induce_schemas_tier1(
            store, budget=budget, rate=rate, llm_enabled=False,
        )
        return {
            "candidates": [
                {
                    "pattern": c.pattern,
                    "confidence": c.confidence,
                    "evidence_count": c.evidence_count,
                    "status": c.status,
                }
                for c in candidates
            ],
            "count": len(candidates),
        }

    if method == "curiosity_pending":
        from iai_mcp.curiosity import pending_questions

        qs = pending_questions(store, params.get("session_id"))
        return {
            "questions": [
                {
                    "id": str(q.id),
                    "text": q.text,
                    "tier": q.tier,
                    "entropy": q.entropy,
                    "triggered_by_record_ids": [str(t) for t in q.triggered_by_record_ids],
                }
                for q in qs
            ],
            "count": len(qs),
        }

    if method == "trajectory_record":
        from iai_mcp.trajectory import record_session_metrics

        metrics = params.get("metrics", {})
        record_session_metrics(
            store, session_id=params.get("session_id", "-"), metrics=metrics,
        )
        return {"recorded": len(metrics), "session_id": params.get("session_id", "-")}
    # --- /dispatch ---

    # --- dispatch ---
    # adds user-facing MCP tool dispatches:
    #
    # - schema_list: surface. Walks all records tagged "schema",
    #   parses pattern / confidence / status from tags + literal_surface, and
    #   counts `schema_instance_of` inbound edges per schema for
    #   evidence_count + exceptions_count. Supports domain + confidence_min
    #   filters.
    # - events_query: surface with a strict whitelist of user-visible
    #   event kinds. Rejects identity-kernel kinds (s5_invariant_update etc)
    # to preserve 's trust boundary ( threat model).
    if method == "schema_list":
        return _schema_list_dispatch(store, params)

    if method == "events_query":
        return _events_query_dispatch(store, params)
    # --- /dispatch ---

    # --- dispatch ---
    # user-audit surface. Three dispatch entrypoints:
    #
    # - audit_query: delegates to s5.audit_identity_events; returns the same
    #   newest-first list of identity-relevant events the CLI renders. Caller
    #   may pass since_iso (ISO-8601 UTC) + kinds override; shield payloads
    #   are NOT redacted here (dispatch is trusted; CLI redacts for display).
    # - detect_drift: one-shot drift check; returns any s5_drift_alert payloads
    #   and side-effects the events table (same as CLI's `audit drift`).
    # - shield_check: exposed for test + subagent introspection. Does NOT mutate
    #   the store; pure evaluate_injection_risk wrapper.
    if method == "audit_query":
        from iai_mcp.s5 import AUDIT_EVENT_KINDS, audit_identity_events

        since_raw = params.get("since")
        since_dt = None
        if since_raw:
            try:
                since_dt = datetime.fromisoformat(
                    str(since_raw).replace("Z", "+00:00"),
                )
                if since_dt.tzinfo is None:
                    since_dt = since_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return {"error": f"since must be ISO-8601, got {since_raw!r}"}

        kinds_param = params.get("kinds")
        kinds = (
            tuple(kinds_param) if isinstance(kinds_param, (list, tuple))
            else AUDIT_EVENT_KINDS
        )
        events = audit_identity_events(store, since=since_dt, kinds=kinds)
        out_events: list[dict] = []
        for e in events:
            ts = e.get("ts")
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            out_events.append({
                "id": str(e.get("id")),
                "kind": e.get("kind"),
                "severity": e.get("severity"),
                "ts": ts_str,
                "data": e.get("data", {}),
                "session_id": e.get("session_id"),
            })
        return {"events": out_events, "count": len(out_events)}

    if method == "detect_drift":
        from iai_mcp.s5 import detect_drift_anomaly

        window = int(params.get("window_sessions", 5) or 5)
        alerts = detect_drift_anomaly(store, window_sessions=window)
        return {"alerts": alerts, "count": len(alerts)}

    if method == "shield_check":
        from iai_mcp.shield import ShieldTier, evaluate_injection_risk

        text = params.get("text", "") or ""
        tier_name = str(params.get("tier", "hard_block")).lower()
        try:
            tier = ShieldTier(tier_name)
        except ValueError:
            return {"error": f"unknown shield tier {tier_name!r}"}
        verdict = evaluate_injection_risk(
            text, tier, target_language=params.get("language"),
        )
        return {
            "tier": verdict.tier.value,
            "detected": verdict.detected,
            "matched_patterns": list(verdict.matched_patterns),
            "severity": verdict.severity,
            "action": verdict.action,
            "reason": verdict.reason,
            "confidence": verdict.confidence,
            "language": verdict.language,
        }
    # --- /dispatch ---

    # --- dispatch (Ashby sigma diagnostic) ---
    # topology: read-only snapshot of the current runtime graph
    # (N, C, L, sigma, community_count, rich_club_ratio, regime).
    # Purely diagnostic — retrieval modes NEVER toggle based on sigma
    # (constitutional guard: sigma is diagnostic, not a fallback).
    if method == "topology":
        from iai_mcp import sigma as sigma_mod

        records_count = store.db.open_table("records").count_rows()
        if records_count == 0:
            return {
                "N": 0, "C": 0.0, "L": 0.0, "sigma": None,
                "community_count": 0, "rich_club_ratio": 0.0,
                "regime": "insufficient_data",
            }
        graph_bundle = retrieve.build_runtime_graph(store)
        graph = graph_bundle[0] if isinstance(graph_bundle, tuple) else graph_bundle
        return sigma_mod.compute_topology_snapshot(graph)

    # --- dispatch (ecological self-regulation) ---
    # camouflaging_status: read-only detector report over the last weekly window.
    # NEVER models the user; observes surface formality trajectory only.
    # Calling this does NOT relax the register — that pathway runs on the weekly
    # pass (sigma.run_weekly_pass / camouflaging.run_weekly_pass) at S4 cadence.
    if method == "camouflaging_status":
        from iai_mcp import camouflaging

        window = int(params.get("window_size", 5) or 5)
        result = camouflaging.detect_camouflaging(store, window_size=window)
        # Include the current knob value so the caller can see OUR register state
        # without a second profile_get round-trip.
        result["camouflaging_relaxation"] = float(
            _profile_state.get("camouflaging_relaxation", 0.0),
        )
        return result

    # --- / dispatch ---
    # initiate_sleep_mode: explicit user consent gate (, C2 invariant).
    # Consent=False returns immediately without touching the daemon socket.
    # Consent=True sends {"type":"user_initiated_sleep"} NDJSON over the
    # ~/.iai-mcp/.daemon.sock unix socket and returns the daemon's response.
    if method == "initiate_sleep_mode":
        return asyncio.run(handle_initiate_sleep_mode(params))

    # force_wake: cooperative wake. Sends {"type":"force_wake"} over
    # the socket and waits up to 15 min for daemon to complete current REM
    # cycle and yield. Graceful when daemon is unreachable.
    if method == "force_wake":
        return asyncio.run(handle_force_wake(params))
    # --- /dispatch ---

    if method == "profile_get":
        # full 11-knob registry via profile module (10 AUTIST + 1 wake_depth; -02 removed AUTIST-02/08/11/12).
        return profile.profile_get(params.get("knob"), _profile_state)

    if method == "profile_set":
        # M4 LIVE: pass store so a successful change emits
        # kind='profile_updated' for trajectory.m4_profile_variance_live.
        # profile.profile_set mutates _profile_state in-place; serialize
        # so two concurrent socket-driven dispatch threads cannot interleave a
        # read-modify-write on the same knob.
        with _profile_lock:
            return profile.profile_set(
                params["knob"], params["value"], _profile_state, store=store,
            )

    if method == "session_start_payload":
        # session-start assembly .
        # M6 LIVE: assemble_session_start now also emits
        # kind='session_started' for context-repeat-rate measurement.
        # : thread the per-process profile state so the
        # wake_depth knob reaches the assembler.
        from iai_mcp.session import assemble_session_start, SessionStartPayload
        sid = params.get("session_id", "-")
        records_count = store.db.open_table("records").count_rows()
        if records_count == 0:
            empty = SessionStartPayload(
                l0="",
                l1="",
                l2=[],
                rich_club="",
                total_cached_tokens=0,
                total_dynamic_tokens=1000,
            )
            return _payload_to_json(empty)
        _graph, assignment, rc = retrieve.build_runtime_graph(store)
        payload = assemble_session_start(
            store, assignment, rc,
            session_id=sid,
            profile_state=_profile_state,
        )
        return _payload_to_json(payload)

    raise UnknownMethodError(method)


def _hit_to_json(h) -> dict:
    # Derived temporal validity, computed at recall time from the contradicts-
    # edge graph. None when the record has no superseding contradiction
    # (valid_to) or when enrichment was not run on this code path (back-compat
    # default -- applies to recall_for_benchmark and any pre-temporal-validity
    # caller that constructs MemoryHit-shaped objects). getattr fallback
    # defends against any future MemoryHit-shaped object the serializer might
    # be handed without the new fields (partial mock in a test, etc.). The
    # _stale_downweighted sentinel from apply_stale_downweight is intentionally
    # NOT serialized -- only the public hit fields plus valid_from / valid_to
    # cross onto the JSON wire.
    _vf = getattr(h, "valid_from", None)
    _vt = getattr(h, "valid_to", None)
    return {
        "record_id": str(h.record_id),
        "score": float(h.score),
        "reason": h.reason,
        "literal_surface": h.literal_surface,
        "adjacent_suggestions": [str(x) for x in h.adjacent_suggestions],
        "valid_from": _vf.isoformat() if _vf is not None else None,
        "valid_to": _vt.isoformat() if _vt is not None else None,
    }


# ---------------------------------------------------------- helpers


# events_query whitelist. / 02-03 write many event kinds;
# only the user-introspection-safe subset is exposed via the MCP surface.
# s5_invariant_update / s5_invariant_proposal stay internal (identity kernel).
EVENTS_QUERY_WHITELIST: frozenset[str] = frozenset({
    "s4_contradiction",
    "trajectory_metric",
    "schema_induction_run",
    "llm_health",
    "curiosity_silent_log",
    "curiosity_question",
    "cls_consolidation_run",
    "crypto_key_rotated",
})


def _schema_list_dispatch(store: MemoryStore, params: dict) -> dict:
    """MCP-08 schema_list implementation.

    Walks all records tagged "schema" (created by schema.persist_schema).
    Parses pattern + confidence + status from record tags + literal_surface.
    Counts schema_instance_of inbound edges for evidence_count; uses weight<0
    marker for exceptions (future extension -- defaults to 0 in ).
    Filters:
      - confidence_min (float): only schemas whose parsed confidence >= this.
      - domain (str): only schemas tagged domain:<name>.
    """
    import pandas as pd

    confidence_min = float(params.get("confidence_min", 0.0) or 0.0)
    domain_filter = params.get("domain")

    records = store.all_records()
    schema_records = [r for r in records if "schema" in (r.tags or [])]

    edges_df = store.db.open_table("edges").to_pandas()
    if not edges_df.empty:
        schema_edges = edges_df[edges_df["edge_type"] == "schema_instance_of"]
    else:
        schema_edges = pd.DataFrame(columns=["src", "dst", "weight"])

    out: list[dict] = []
    for rec in schema_records:
        # Parse pattern from tags: "pattern:..." tag (persist_schema writes this).
        pattern = ""
        status = "auto"
        for t in (rec.tags or []):
            if t.startswith("pattern:"):
                pattern = t.split(":", 1)[1]
            elif t in ("auto", "pending_user_approval"):
                status = t
        if not pattern and rec.literal_surface.startswith("Schema: "):
            # Fall back to parsing the summary: "Schema: <pattern> (confidence=...)"
            rest = rec.literal_surface[len("Schema: "):]
            pattern = rest.split(" (confidence=")[0]

        # Parse confidence from the summary line: "...(confidence=0.90)".
        confidence = 0.0
        if "(confidence=" in rec.literal_surface:
            try:
                seg = rec.literal_surface.rsplit("(confidence=", 1)[1]
                num = seg.split(")")[0]
                confidence = float(num)
            except (ValueError, IndexError):
                confidence = 0.0

        # Domain filter (opt-in).
        if domain_filter is not None:
            domain_tag = f"domain:{domain_filter}"
            if domain_tag not in (rec.tags or []):
                continue

        # Confidence filter.
        if confidence < confidence_min:
            continue

        # Evidence count = schema_instance_of edges whose dst is this schema.
        sid = str(rec.id)
        if len(schema_edges) > 0:
            evidence = schema_edges[schema_edges["dst"] == sid]
            evidence_count = int(len(evidence))
            # Exceptions = negative-weight schema_instance_of edges (future use).
            exceptions_count = int(
                len(evidence[evidence["weight"] < 0])
            ) if "weight" in evidence.columns else 0
        else:
            evidence_count = 0
            exceptions_count = 0

        out.append({
            "id": str(rec.id),
            "pattern": pattern,
            "confidence": float(confidence),
            "evidence_count": evidence_count,
            "exceptions_count": exceptions_count,
            "status": status,
            "language": rec.language,
        })

    return {"schemas": out, "total": len(out)}


def _events_query_dispatch(store: MemoryStore, params: dict) -> dict:
    """MCP-05 events_query implementation.

    Whitelist-gated. Parses since as ISO-8601. Caps limit at 1000. Returns
    events with ISO-string timestamps (pandas Timestamps are not
    JSON-serialisable out of the box).
    """
    from iai_mcp.events import query_events

    kind = params.get("kind")
    if not kind:
        return {"error": "kind parameter is required"}
    if kind not in EVENTS_QUERY_WHITELIST:
        return {
            "error": (
                f"kind {kind!r} is not user-visible; "
                f"allowed: {sorted(EVENTS_QUERY_WHITELIST)}"
            )
        }

    severity = params.get("severity")
    since_raw = params.get("since")
    since_dt = None
    if since_raw:
        try:
            since_dt = datetime.fromisoformat(str(since_raw).replace("Z", "+00:00"))
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return {"error": f"since must be ISO-8601, got {since_raw!r}"}

    limit = int(params.get("limit", 100) or 100)
    limit = max(1, min(1000, limit))

    events = query_events(
        store,
        kind=kind,
        since=since_dt,
        severity=severity,
        limit=limit,
    )
    out_events: list[dict] = []
    for e in events:
        ts = e["ts"]
        if hasattr(ts, "isoformat"):
            try:
                ts_str = ts.isoformat()
            except Exception:
                ts_str = str(ts)
        else:
            ts_str = str(ts)
        out_events.append({
            "id": str(e["id"]),
            "kind": e["kind"],
            "severity": e.get("severity"),
            "domain": e.get("domain"),
            "ts": ts_str,
            "data": e["data"],
            "session_id": e.get("session_id"),
            "source_ids": e.get("source_ids", []),
        })
    return {"events": out_events, "count": len(out_events)}


# -------------------------------------------------------- helpers
# / wiring lives here. Three public entry points:
#   - _send_to_daemon: internal NDJSON helper over ~/.iai-mcp/.daemon.sock
#   - handle_initiate_sleep_mode: JSON-RPC method with C2 consent guard
#   - handle_force_wake: JSON-RPC method with 15-min cooperative cap
#   - _inject_sleep_suggestion: memory_recall dispatch hook
#
# Constitutional invariant C2: the socket WRITE in handle_initiate_sleep_mode
# is unreachable unless params["consent"] is literally True. Short-circuits
# on missing key, wrong type, or False. Grep guard: "consent is not True".


async def _send_to_daemon(
    message: dict,
    *,
    timeout: float = 30.0,
    socket_path=None,
) -> dict:
    """Send one NDJSON message over the daemon unix socket and read one reply.

    Returns a dict. Failure modes (always structured, never raised):
        - FileNotFoundError / ConnectionRefusedError -> daemon_not_running
        - read timeout                               -> timeout
        - empty read (daemon closed)                 -> empty_response
    Socket write errors propagate; callers should not catch broadly.
    """
    # Imported lazily so test monkeypatches of iai_mcp.core.SOCKET_PATH take
    # precedence over the module-level import symbol.
    path_used = socket_path if socket_path is not None else SOCKET_PATH
    try:
        reader, writer = await asyncio.open_unix_connection(str(path_used))
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        return {"ok": False, "reason": "daemon_not_running", "error": str(exc)}

    try:
        writer.write((json.dumps(message) + "\n").encode("utf-8"))
        await writer.drain()
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "reason": "timeout"}
        if not line:
            return {"ok": False, "reason": "empty_response"}
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            return {"ok": False, "reason": "invalid_json", "error": str(exc)}
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def handle_initiate_sleep_mode(params: dict) -> dict:
    """user-consent gate for daemon sleep mode.

    Strict schema validation per ASVS V5: raises ValueError for missing
    or wrong-typed params. Returns a dict in the normal path.

    C2 invariant: the socket write is unreachable unless
    `params["consent"] is True` -- False, missing, or non-bool values all
    return early with "consent_declined" BEFORE touching the socket.
    """
    if not isinstance(params, dict):
        raise ValueError("initiate_sleep_mode params must be an object")
    if "consent" not in params:
        raise ValueError("initiate_sleep_mode requires 'consent' (bool)")
    if "reason" not in params:
        raise ValueError("initiate_sleep_mode requires 'reason' (str)")
    if not isinstance(params["consent"], bool):
        raise ValueError("'consent' must be bool")
    if not isinstance(params["reason"], str):
        raise ValueError("'reason' must be str")

    # C2 guard: only `True` (literal bool) progresses to the daemon socket.
    if params["consent"] is not True:
        return {"ok": False, "reason": "consent_declined"}

    # Clip reason to a safe length for log payload (ASVS V5 output hardening).
    reason = params["reason"][:500]
    return await _send_to_daemon({
        "type": "user_initiated_sleep",
        "reason": reason,
        "ts": datetime.now(timezone.utc).isoformat(),
    })


async def handle_force_wake(params: dict) -> dict:
    """cooperative force-wake.

    Sends {"type":"force_wake"} NDJSON and waits up to
    FORCE_WAKE_TIMEOUT_SEC (15 min) for the daemon to complete its current
    REM cycle and reply. Never SIGTERM. Daemon-unreachable returns a
    structured {"ok": False, "reason": "daemon_not_running"} instead of
    crashing the JSON-RPC loop.
    """
    return await _send_to_daemon(
        {
            "type": "force_wake",
            "ts": datetime.now(timezone.utc).isoformat(),
        },
        timeout=float(FORCE_WAKE_TIMEOUT_SEC),
    )


def _inject_sleep_suggestion(
    response: dict,
    *,
    cue: str,
    language: str,
) -> None:
    """inject `sleep_suggestion` into a memory_recall response when the
    dual-gate wind-down detector fires.

    Silent-fail on any exception: detector failure must NEVER break the
    memory_recall path (daemon-state corruption, bedtime import error, tz
    lookup failure, etc. are all tolerated). The response simply goes out
    without a `sleep_suggestion` key -- the absence IS the signal.
    """
    try:
        from iai_mcp.bedtime import detect_wind_down
        from iai_mcp.daemon_state import load_state
        from iai_mcp.tz import load_user_tz

        state = load_state()
        now = datetime.now(timezone.utc)
        tz = load_user_tz()
        suggestion = detect_wind_down(cue, language, state, now, tz)
        if suggestion:
            response["sleep_suggestion"] = suggestion
    except Exception:
        # Silent fail -- memory_recall is the hot path and must not break.
        pass


# Deterministic overnight_digest contract.
# The key is ALWAYS present in memory_recall responses; this is the
# zeroed default when daemon_state has no pending digest (or the
# digest pipeline silent-fails). Field shape MUST match the rich-
# payload branch inside _inject_overnight_digest so consumers see
# one stable schema regardless of daemon REM-cycle state.
_EMPTY_OVERNIGHT_DIGEST: dict = {
    "rem_cycles_completed": 0,
    "episodes_processed": 0,
    "schemas_induced_tier0": 0,
    "claude_call_used": False,
    "quota_used_pct": 0.0,
    "main_insight_text": None,
    "sigma_observed": None,
    "s5_drift_alerts": [],
    "daemon_uptime_hours": 0,
    "timed_out_cycles": 0,
}


def _inject_overnight_digest(response: dict, store: MemoryStore | None = None) -> None:
    """Every memory_recall response carries an ``overnight_digest`` key.

    The digest lives inside ``.daemon-state.json``;
    ``daemon_state.get_pending_digest`` handles the 18h timing gate and
    CLEARS the digest from state on delivery, so the rich payload still
    surfaces exactly once per window (once-per-window invariant preserved).

    The ``overnight_digest`` key is ALWAYS present in the mutated response.
    When the daemon has a pending digest within the 18h once-per-window gate,
    the payload is the rich dict; otherwise it is ``_EMPTY_OVERNIGHT_DIGEST``
    (structured zeros). This guarantees byte-identical top-level shape across
    stdio and socket transports regardless of daemon timing.

    Silent-fail on any exception: corrupt state, disk failure, or schema drift
    must NEVER break the memory_recall hot path. On exception the zeroed
    default is still written first so determinism holds even on a daemon-
    state IO hiccup; when ``store`` is provided, a best-effort
    ``digest_inject_error`` warning event is emitted so operators can see
    that the digest pipeline failed once.
    """
    try:
        state = load_state()
        now = datetime.now(timezone.utc)
        digest = get_pending_digest(state, now)
        if not digest:
            # Deterministic contract -- key always present, zeroed default
            # when no digest is pending. Copy to avoid sharing the module-
            # level mutable default across responses.
            response["overnight_digest"] = dict(_EMPTY_OVERNIGHT_DIGEST)
            return
        response["overnight_digest"] = {
            "rem_cycles_completed": digest.get("rem_cycles_completed", 0),
            "episodes_processed": digest.get("episodes_processed", 0),
            "schemas_induced_tier0": digest.get("schemas_induced_tier0", 0),
            "claude_call_used": digest.get("claude_call_used", False),
            "quota_used_pct": digest.get("quota_used_pct", 0.0),
            "main_insight_text": digest.get("main_insight_text"),
            "sigma_observed": digest.get("sigma_observed"),
            "s5_drift_alerts": digest.get("s5_drift_alerts", []),
            "daemon_uptime_hours": digest.get("daemon_uptime_hours", 0),
            "timed_out_cycles": digest.get("timed_out_cycles", 0),
        }
    except Exception as exc:  # noqa: BLE001 -- hot path must never break
        # Set the zeroed default BEFORE the silent-fail event write so a
        # daemon-state IO hiccup cannot re-introduce non-determinism in
        # top-level response keys.
        response["overnight_digest"] = dict(_EMPTY_OVERNIGHT_DIGEST)
        if store is not None:
            try:
                from iai_mcp.events import write_event
                write_event(
                    store,
                    "digest_inject_error",
                    {"error": str(exc)[:500]},
                    severity="warning",
                )
            except Exception:
                pass


def _first_turn_recall_hook(
    response: dict,
    *,
    params: dict,
    store: MemoryStore,
) -> None:
    """ / D5-03: first-turn auto-recall hook.

    Fires exactly once per session. Runs a scoped ``retrieve.recall`` with
    a capped budget (400 tok) using the user's cue as-is, clamped to 2000
    chars per V5 security domain. Injects the result as ``first_turn_recall``
    in the response. Silent-fail on any exception: the hot recall path must
    not break if daemon_state is unreachable, recall raises, or the event
    table is full.

    Security:
    - V5 input-length clamp: `cue[:2000]` before handing to recall.
    - The hook never calls any paid API (C3 invariant).

    Idempotency:
    - `daemon_state.consume_first_turn` is a pop+save; a concurrent second
      dispatcher will see the flag already consumed and skip the hook.
    """
    try:
        from iai_mcp.daemon_state import consume_first_turn, load_state
        state = load_state()
        session_id = params.get("session_id", "unknown")
        if not consume_first_turn(state, session_id):
            return  # not the first turn; bail
        # V5 input length clamp.
        raw_cue = params.get("cue", "")
        cue = str(raw_cue)[:2000] if raw_cue is not None else ""
        if not cue:
            return
        # : consult the HIPPEA cascade warm LRU BEFORE going
        # cold. The LRU is populated by the daemon-side cascade on session_open
        # (D5-05). If empty (daemon down, core+daemon in separate processes,
        # or cascade hasn't fired yet) we fall through to the cold baseline.
        warm_hit_ids: list = []
        try:
            from iai_mcp.hippea_cascade import snapshot_warm_ids
            warm_hit_ids = snapshot_warm_ids()
        except Exception:
            warm_hit_ids = []

        # cross-process closure: when the daemon's LRU is not
        # visible to this process (which is always the case on fresh core
        # boot), fire a synchronous cascade once per session and populate
        # the core-local LRU. Duplicates daemon work; the cost is one-time
        # per session, amortised across subsequent recall calls.
        warm_lru_source = "daemon" if warm_hit_ids else "none"
        if not warm_hit_ids and str(session_id) not in _CORE_CASCADE_FIRED_PER_SESSION:
            try:
                from iai_mcp.hippea_cascade import compute_core_side_warm_snapshot
                from iai_mcp import retrieve as _retrieve
                _graph, assignment, _rc = _retrieve.build_runtime_graph(store)
                warm_ids = compute_core_side_warm_snapshot(
                    store, assignment, top_k=3, max_records=50,
                )
                for rid in warm_ids:
                    try:
                        rec = store.get(rid)
                        if rec is not None:
                            _CORE_WARM_LRU[rid] = rec
                    except Exception:
                        continue
                _CORE_CASCADE_FIRED_PER_SESSION.add(str(session_id))
                if _CORE_WARM_LRU:
                    warm_hit_ids = list(_CORE_WARM_LRU.keys())
                    warm_lru_source = "core_fallback"
            except Exception:
                # Cascade failed; cold path still runs below. Hot path
                # must never break.
                pass

        # Scoped recall: capped budget (400 tok per D5-03), modest k.
        # The warm LRU hint is surfaced in the response so observability can
        # measure whether the cascade is firing on this process -- but the
        # authoritative hit set stays the cold recall path so verbatim recall
        # correctness is unchanged by LRU population.
        cue_embedding = params.get("cue_embedding") or [0.0] * EMBED_DIM
        # retrieve.recall now defaults to mode='verbatim'
        # (conservative fallback default protects North-Star on the degraded
        # path). The first-turn hook is NOT a degraded-path call — it runs
        # alongside the main dispatch on every fresh session, regardless of
        # whether the cue is verbatim-flavoured. The contract is
        # "scoped recall over all tiers as a session-warm-up signal", which
        # is concept-mode semantics. Pin explicitly so the hook does not
        # silently flip to episodic-only filtering when ships.
        result = retrieve.recall(
            store=store,
            cue_embedding=cue_embedding,
            cue_text=cue,
            session_id=str(session_id),
            budget_tokens=400,
            k_hits=5,
            k_anti=2,
            mode="concept",
        )
        response["first_turn_recall"] = {
            "hits": [_hit_to_json(h) for h in result.hits],
            "budget_tokens": 400,
            "budget_used": result.budget_used,
            "warm_lru_size": len(warm_hit_ids),
            "warm_lru_source": warm_lru_source,
        }
        # Diagnostic-only event emit; never block the recall path.
        try:
            from iai_mcp.events import write_event
            write_event(
                store,
                "first_turn_recall",
                {"session_id": str(session_id), "cue_len": len(cue)},
                severity="info",
            )
        except Exception:
            pass
    except Exception:
        # Hot path must not break. The absence of `first_turn_recall`
        # in the response IS the signal that the hook did not fire.
        pass


def _payload_to_json(payload) -> dict:
    """Serialise SessionStartPayload for JSON-RPC transport .

    D5-02: new wake_depth-branched fields surfaced alongside
    legacy l0/l1/l2/rich_club so the TS wrapper can read either set.
    """
    return {
        "l0": payload.l0,
        "l1": payload.l1,
        "l2": list(payload.l2),
        "rich_club": payload.rich_club,
        "total_cached_tokens": int(payload.total_cached_tokens),
        "total_dynamic_tokens": int(payload.total_dynamic_tokens),
        "breakpoint_marker": payload.breakpoint_marker,
        # D5-02 lazy-session-start surface.
        "identity_pointer": getattr(payload, "identity_pointer", ""),
        "brain_handle": getattr(payload, "brain_handle", ""),
        "topic_cluster_hint": getattr(payload, "topic_cluster_hint", ""),
        # compact handle (<iai:{16-hex}>).
        "compact_handle": getattr(payload, "compact_handle", ""),
        "wake_depth": getattr(payload, "wake_depth", "minimal"),
    }


# --------------------------------------------------------------------- daemon

def main() -> None:
    """stdio JSON-RPC loop -- reads one JSON object per line, writes responses.

    announce the user's IANA timezone on boot so users can
    see at a glance how their sleep-cycle quiet_window and CLI timestamps are
    being interpreted. Quiet by default; logs to stderr to avoid polluting
    the stdin/stdout JSON-RPC channel.
    """
    store = MemoryStore()
    _seed_l0_identity(store)

    # timezone announcement (stderr, not stdout -- stdout is JSON-RPC).
    try:
        from iai_mcp.tz import load_user_tz
        tz = load_user_tz()
        sys.stderr.write(f"iai-mcp: timezone={tz.key}\n")
        sys.stderr.flush()
    except Exception as e:  # pragma: no cover -- boot diagnostics must not break the core
        sys.stderr.write(f"iai-mcp: timezone detection failed: {e}\n")
        sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id: Any = None
        try:
            req = json.loads(line)
            req_id = req.get("id") if isinstance(req, dict) else None
            method = req.get("method")
            params = req.get("params") or {}
            if not method:
                raise ValueError("missing method")
            result = dispatch(store, method, params)
            sys.stdout.write(
                json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n"
            )
        except Exception as e:
            err = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32000,
                    "message": str(e),
                    "trace": traceback.format_exc() if sys.flags.dev_mode else None,
                },
            }
            sys.stdout.write(json.dumps(err) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
