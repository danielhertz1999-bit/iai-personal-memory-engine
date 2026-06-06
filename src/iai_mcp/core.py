"""JSON-RPC core for IAI-MCP.

Binds the MCP tools to the Python internals. The TypeScript MCP wrapper spawns
this module as a subprocess (`python -m iai_mcp.core`) and forwards
line-delimited JSON-RPC 2.0 requests over stdio.

Boot sequence:
1. Open MemoryStore at ~/.iai-mcp
2. Seed pinned L0 identity record if absent, stamping its aaak_index
3. Loop: read JSON line from stdin, dispatch, write JSON-RPC response to stdout.

All writes are synchronous.

Exports `LIVE_KNOBS` / `DEFERRED_KNOBS` / `L0_ID` for backwards compatibility;
they now point at the authoritative profile registry state rather than local
copies.

Dispatch surfaces:
- `memory_consolidate`: real heavy consolidation
- `session_exit`: light consolidation
- `s5_propose`: M-of-N voting on invariant updates
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time as _time
import traceback
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from iai_mcp.exceptions import IAIMCPError, RetrievalError, EmbeddingError, StoreError, NativeError

from iai_mcp import profile, retrieve

logger = logging.getLogger(__name__)
from iai_mcp.aaak import enforce_english_raw, generate_aaak_index
from iai_mcp.concurrency import SOCKET_PATH
from iai_mcp.daemon_state import get_pending_digest, load_state
from iai_mcp.native_guard import _require_native
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


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


# Cooperative force-wake cap. Daemon completes at most one 15-min REM
# cycle before yielding; the JSON-RPC caller waits up to this long before
# giving up with a "timeout" response.
FORCE_WAKE_TIMEOUT_SEC: int = 15 * 60  # 900s


# Cross-process LRU
#
# The sleep daemon owns its own HIPPEA cascade LRU. The MCP core runs in a
# different process; that LRU is invisible across the process boundary.
# ``snapshot_warm_ids()`` returns [] in core on every fresh boot.
#
# Closure: core maintains its OWN, process-local LRU here. When
# ``_first_turn_recall_hook`` sees an empty daemon snapshot, it runs a
# synchronous cascade once per session and populates ``_CORE_WARM_LRU``.
# Subsequent recalls reuse the warmed records via the normal
# ``get_warm_record(rid)`` lookup path.
#
# Read-only: compute_core_side_warm_snapshot touches store only via
# ``store.get`` -- no mutation. No paid-API calls; salience is pure-local.
# Cascade produces record ids only; LRU writes are per-process RAM, not
# store-backed.
from cachetools import TTLCache as _CoreTTLCache

_CORE_WARM_LRU: _CoreTTLCache = _CoreTTLCache(maxsize=50, ttl=300)
_CORE_CASCADE_FIRED_PER_SESSION: set[str] = set()


# ----------------------------------------------------------------- knob state
# Per-process mutable profile state initialised from profile.default_state().
# profile_get / profile_set both read and write this dict.
_profile_state: dict[str, Any] = profile.default_state()

# LEARN-01 posterior state accumulator. Keyed by knob name, each entry carries
# conjugate-prior state (alpha/beta for bool, alphas for enum,
# weighted_sum/total_weight/mean for float/int, per_key for dict).
_posterior_state: dict[str, Any] = {}

# Arousal-budget lifecycle: per-process ArousalState updated on every recall.
# Drives budget_tokens dynamically (high stress → 800, low stress → 3000).
# Graceful degradation: if arousal_budget module is missing, falls back to
# the hardcoded budget_tokens=1500 default unchanged.
_arousal_state: object | None = None  # ArousalState instance or None

# Trajectory coupling: store last recall injection embedding for next-turn
# cosine comparison.
_last_injection_embedding: list[float] | None = None
_last_injection_ids: list[str] = []

# Serialize mutations to module-level state across concurrent socket-driven
# dispatch calls. Read-only paths do NOT acquire this lock — the GIL keeps
# individual dict ops atomic; only read-modify-write sequences (profile_set,
# profile_update_from_signal) need it. MUST be threading.RLock (re-entrant,
# sync) because socket_server invokes `dispatch` via
# `await asyncio.to_thread(...)`, so the lock is acquired from a thread-pool
# worker where asyncio primitives are unreachable.
_profile_lock: threading.RLock = threading.RLock()

# `LIVE_KNOBS` (mutable dict) and `DEFERRED_KNOBS` (frozenset) are exported
# for backwards compatibility; they now alias the authoritative registry.
LIVE_KNOBS: dict[str, Any] = _profile_state  # mutating LIVE_KNOBS still mutates state
DEFERRED_KNOBS: frozenset[str] = frozenset(
    profile.PHASE_2_DEFERRED | profile.PHASE_3_DEFERRED
)
# All 10 autistic-kernel knobs are now live; DEFERRED_KNOBS must be empty.
assert len(DEFERRED_KNOBS) == 0, "all 10 autistic-kernel knobs live"


# ----------------------------------------------------------------------- seed
# Deterministic L0 UUID so seed idempotency check is cheap and cross-process
# stable.
L0_ID = UUID("00000000-0000-0000-0000-000000000001")


_DEFAULT_L0_SEED = (
    "User identity not yet configured. "
    "Run `iai-mcp config identity` to set your name, language, and role."
)


def _load_l0_identity_seed() -> str:
    """Load L0 identity seed from user config or return default."""
    config_path = os.path.join(
        os.environ.get("IAI_MCP_STORE", os.path.expanduser("~/.iai-mcp")),
        "config.json",
    )
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            identity = cfg.get("identity", {})
            parts = []
            if identity.get("name"):
                parts.append(f"User: {identity['name']}.")
            if identity.get("languages"):
                parts.append(f"Primary languages: {identity['languages']}.")
            if identity.get("role"):
                parts.append(f"Role: {identity['role']}.")
            if identity.get("project"):
                parts.append(f"Active project: {identity['project']}.")
            if identity.get("extra"):
                parts.append(identity["extra"])
            if parts:
                return " ".join(parts)
        except (json.JSONDecodeError, OSError):
            pass
    return _DEFAULT_L0_SEED


def _seed_l0_identity(store: MemoryStore) -> None:
    """Seed the pinned L0 identity record.

    Idempotent: returns immediately if L0_ID already exists. Called once at
    core boot. The zero-vector embedding is re-embedded with the configured
    embedder on first graph reconstruction; aaak_index is stamped so the
    session-start manifest can reference the L0 metadata without leaking
    literal_surface content.
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
        literal_surface=_load_l0_identity_seed(),
        aaak_index="",
        embedding=[0.0] * seed_dim,   # re-embedded via graph reconstruction
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
    # Identity guard -- ASCII English identity passes cleanly.
    enforce_english_raw(seed)
    # Metadata stamp so session-start assembler has a populated aaak_index.
    seed.aaak_index = generate_aaak_index(seed)
    store.insert(seed)


# ------------------------------------------------------------- JSON-RPC layer

def dispatch(store: MemoryStore, method: str, params: dict) -> dict:
    """Route a single JSON-RPC method to the corresponding internal function."""
    global _last_injection_embedding, _last_injection_ids, _arousal_state
    if method == "memory_recall":
        # Capture at method-entry so _recall_latency_ms is emitted on all
        # dispatch paths (empty-store, cortex-fallback, normal, exception-fallback).
        _recall_t0 = _time.perf_counter()
        # Classify the cue BEFORE choosing the recall path so both the
        # empty-store fallback and the full pipeline see the same mode.
        # The classifier reads only the cue text (regex on surface signals —
        # quoted phrases, EN word-markers, RU starts-with triggers) and returns
        # ('verbatim' | 'concept', triggered_pattern).
        # The triggered_pattern is for diagnostic logging only; only the
        # mode string flows downstream.
        from iai_mcp.cue_router import _classify_cue
        # 3-tuple return (mode, intent, label). intent is consumed by
        # _recall_core directly (it re-calls _classify_cue inside
        # recall_for_response); preserved here for back-compat of the
        # cue_mode plumbing into RecallResponse.
        cue_mode, _cue_intent, _triggered_pattern = _classify_cue(params.get("cue", ""))

        # Seed the audit accumulator BEFORE recall fires its gain branches.
        # Threaded into recall_for_response and mutated in place by
        # profile_modulation_for_record; apply_profile below extends in place.
        # Attached to the response so MCP callers can audit which knobs fired.
        knobs_applied: dict[str, str] = {}
        # wake_depth seed: operator-facing knob; provenance points into
        # session.py:assemble_session_start (wake_depth = state.get(...)).
        _wake_depth_value = (_profile_state or {}).get("wake_depth", "minimal")
        if _wake_depth_value not in ("minimal", "standard", "deep"):
            _wake_depth_value = "minimal"
        knobs_applied["MCP-12"] = (
            f"session.py:assemble_session_start:wake_depth={_wake_depth_value}"
        )

        # Arousal-budget lifecycle: compute retrieval params from arousal state.
        # Full try/except so a missing/broken arousal_budget module falls back
        # to the existing hardcoded budget_tokens=1500 unchanged.
        _arousal_budget_tokens: int = 1500
        _arousal_retrieval_params = None
        _arousal_diag: dict | None = None
        try:
            from iai_mcp.arousal_budget import (
                ArousalState as _ArousalState,
                compute_retrieval_params as _compute_retrieval_params,
                update_arousal as _update_arousal,
            )
            global _arousal_state
            if _arousal_state is None:
                _arousal_state = _ArousalState()
            _arousal_retrieval_params = _compute_retrieval_params(_arousal_state)
            _arousal_budget_tokens = _arousal_retrieval_params.budget_tokens
            _arousal_diag = {
                "level": _arousal_state.level,
                "mode": _arousal_retrieval_params.mode,
            }
        except Exception as exc:  # noqa: BLE001 -- graceful degradation
            logger.debug("arousal_budget_init_failed: %s", exc)
            _arousal_budget_tokens = 1500
            _arousal_diag = None

        # Non-empty store -> 5-stage pipeline; empty store -> baseline cosine recall.
        _cortex_fallback = False
        _structural_source: str = ""  # populated by the ANN-first path
        records_count = store.db.open_table("records").count_rows()
        if records_count == 0:
            cue_embedding = params.get("cue_embedding") or [0.0] * EMBED_DIM
            resp = retrieve.recall(
                store=store,
                cue_embedding=cue_embedding,
                cue_text=params["cue"],
                session_id=params.get("session_id", "unknown"),
                budget_tokens=params.get("budget_tokens") or _arousal_budget_tokens,
                # Thread classified mode so the degraded path honours the same
                # contract (verbatim cue → episodic-only candidates).
                mode=cue_mode,
            )
        else:
            from iai_mcp.embed import embedder_for_store
            from iai_mcp.pipeline import recall_for_response
            # Defensive try/except around the full-pipeline branch so a
            # graph-build failure (cache miss + corruption, community
            # detection error, OOM, etc.) routes to the baseline fallback
            # with the classified mode preserved.
            # CQRS sleep detection: serve from baseline retrieve when
            # daemon is SLEEP/DREAMING to avoid blocking on graph build.
            try:
                from iai_mcp.daemon_state import load_state as _ds_load
                _ds = _ds_load()
                if _ds.get("current_state", "WAKE") in ("SLEEP", "DREAMING"):
                    cue_embedding = params.get("cue_embedding") or [0.0] * EMBED_DIM
                    resp = retrieve.recall(
                        store=store,
                        cue_embedding=cue_embedding,
                        cue_text=params["cue"],
                        session_id=params.get("session_id", "unknown"),
                        budget_tokens=params.get("budget_tokens") or _arousal_budget_tokens,
                        mode=cue_mode,
                    )
                    _cortex_fallback = True
            except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
                logger.debug("cqrs_sleep_detection_failed: %s", exc)
            if not _cortex_fallback:
                try:
                    from iai_mcp import runtime_graph_cache as _rgc
                    from iai_mcp.graph import MemoryGraph
                    from iai_mcp.pipeline import K_CANDIDATES

                    embedder = embedder_for_store(store)

                    # 3-case consume-only structural loader.
                    # NEVER calls build_runtime_graph on the recall hot path.
                    assignment, rc, _cached_max_degree, _structural_source = _rgc.load_recall_structural(store)

                    # Layer-1 ANN-first candidate pool.
                    # Step 1: embed the cue (NativeError propagates loud-stop).
                    # Measure encode_ms tightly around the clean embed call so
                    # the daemon-up recall tail is observable (telemetry below).
                    _encode_ms: "float | None" = None
                    _encode_t0 = _time.perf_counter()
                    try:
                        _cue_vec = embedder.embed(params["cue"])
                        _encode_ms = (_time.perf_counter() - _encode_t0) * 1000.0
                    except Exception as _emb_exc:
                        # Layer-2 store-backed event: emit BEFORE raise so the
                        # failure is observable even when the caller's handler
                        # swallows the exception.  Best-effort (never re-raises).
                        try:
                            from iai_mcp.events import write_event, TELEMETRY_EMBED_NATIVE_FAILURE
                            write_event(
                                store,
                                TELEMETRY_EMBED_NATIVE_FAILURE,
                                {"op_type": "recall_cue", "error": str(_emb_exc)},
                                severity="critical",
                                buffered=True,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        raise NativeError(f"recall cue encode failed: {_emb_exc}") from _emb_exc

                    # Step 2: ANN top-K candidates.
                    _ann_pairs = store.query_similar(_cue_vec, k=K_CANDIDATES)
                    # {UUID: MemoryRecord} for all candidates so far.
                    _candidate_recs: dict = {_r.id: _r for _r, _s in _ann_pairs}

                    # Step 3: bounded 2-HOP incident spread (CC-C: two hops,
                    # top_k=5 at each hop, mirrors two_hop_neighborhood).
                    _hop1_edges = store.incident_edges(list(_candidate_recs.keys()), top_k=5)
                    _hop1_new_ids = list({
                        _nbr
                        for _nbr_list in _hop1_edges.values()
                        for (_nbr, _et, _wt) in _nbr_list
                        if _nbr not in _candidate_recs
                    })
                    if _hop1_new_ids:
                        _candidate_recs.update(store.get_batch(_hop1_new_ids))

                    _hop2_edges = store.incident_edges(_hop1_new_ids, top_k=5) if _hop1_new_ids else {}
                    _hop2_new_ids = list({
                        _nbr
                        for _nbr_list in _hop2_edges.values()
                        for (_nbr, _et, _wt) in _nbr_list
                        if _nbr not in _candidate_recs
                    })
                    if _hop2_new_ids:
                        _candidate_recs.update(store.get_batch(_hop2_new_ids))

                    # Step 4: capped rich-club union (top-50 hubs).
                    _RC_CAP = 50
                    _rc_cap = (rc or [])[:_RC_CAP]
                    _rc_new_ids = [_rid for _rid in _rc_cap if _rid not in _candidate_recs]
                    if _rc_new_ids:
                        _candidate_recs.update(store.get_batch(_rc_new_ids))

                    # Step 5: build bounded MemoryGraph from the candidate pool.
                    # Nodes: all candidates. Edges: from hop-1 and hop-2 expansions
                    # so graph.two_hop_neighborhood returns the pre-expanded set.
                    graph = MemoryGraph()
                    for _rec in _candidate_recs.values():
                        graph.add_node(
                            _rec.id,
                            community_id=getattr(_rec, "community_id", None),
                            embedding=list(_rec.embedding or []),
                        )
                        graph.set_node_payload(_rec.id, {
                            "embedding": list(_rec.embedding or []),
                            "surface": _rec.literal_surface or "",
                            "centrality": float(getattr(_rec, "centrality", 0.0) or 0.0),
                            "tier": _rec.tier or "episodic",
                            "tags": list(_rec.tags or []),
                            "language": _rec.language or "en",
                        })
                    # Add hop-1 edges so two_hop_neighborhood spreads correctly.
                    for _qid, _nbr_list in _hop1_edges.items():
                        for (_nbr, _et, _wt) in _nbr_list:
                            if _nbr in _candidate_recs:
                                try:
                                    graph.add_edge(_qid, _nbr, weight=_wt, edge_type=_et)
                                except Exception:  # noqa: BLE001 — edge add fail-safe
                                    pass
                    # Add hop-2 edges.
                    for _qid2, _nbr_list2 in _hop2_edges.items():
                        for (_nbr2, _et2, _wt2) in _nbr_list2:
                            if _nbr2 in _candidate_recs:
                                try:
                                    graph.add_edge(_qid2, _nbr2, weight=_wt2, edge_type=_et2)
                                except Exception:  # noqa: BLE001 — edge add fail-safe
                                    pass

                    # Step 5b: global degree scoring (architectural correctness).
                    # The bounded MemoryGraph only contains in-pool edges, so
                    # graph.degrees() returns sub-graph degrees, not global degrees.
                    # The full-graph path uses global degrees because build_runtime_graph
                    # materialises ALL edge endpoints (incl. phantom nodes).  To match,
                    # we attach global degree counts to the bounded graph via two
                    # bounded SQL reads (O(candidates), NOT O(N)):
                    #   (a) One uncapped incident_edges call for ALL candidates →
                    #       count incident edges per candidate (global degree).
                    #   (b) _cached_max_degree from the persisted cache → global
                    #       max degree for log-normalisation.
                    # These are set as graph._global_degree and graph._max_degree so
                    # pipeline._recall_core picks them up without touching the
                    # full-graph path.
                    # NOTE: only count HEBBIAN edges for the degree signal, NOT
                    # contradicts edges. Contradicts edges are temporal-validity
                    # signals (not degree/centrality), and including them in the
                    # global degree count would mark the SOURCE of a contradicts
                    # edge as high-degree, inflating its score. The full-graph
                    # pipeline also uses graph.degrees() which counts edges in
                    # the adjacency dict — but build_runtime_graph loads ALL edge
                    # types via edges.to_pandas(). For parity, use all edge types
                    # here too, but exclude contradicts from degree so the scoring
                    # signal is consistent with the structural (not temporal)
                    # degree signal.
                    try:
                        _all_cand_ids = list(_candidate_recs.keys())
                        # Use hebbian edges only for degree signal to match the
                        # full-graph path's structural degree (centrality signal).
                        # Contradicts edges are temporal-validity metadata, not
                        # topological degree.
                        _global_edges_hebb = store.incident_edges(
                            _all_cand_ids,
                            edge_types=["hebbian"],
                            top_k=None,  # uncapped: count ALL hebbian edges
                        )
                        graph._global_degree = {
                            str(_cid): len(_nbrs)
                            for _cid, _nbrs in _global_edges_hebb.items()
                        }
                        # Use cached global max_degree; fall back to the maximum
                        # across the bounded candidates if the cache has 0.
                        if _cached_max_degree > 0:
                            graph._max_degree = int(_cached_max_degree)
                        else:
                            _local_max = max(graph._global_degree.values(), default=0)
                            if _local_max > 0:
                                graph._max_degree = _local_max
                    except Exception as _gd_exc:  # noqa: BLE001 — degrade gracefully
                        logger.debug("layer1_global_degree_failed: %s", _gd_exc)

                    # Step 6: candidate-scoped ts_by_id with UNCAPPED contradicts
                    # (a superseding contradicts edge outside top-5 must NOT
                    # be dropped; replaces the O(N) build_temporal_validity_maps scan).
                    _tv_outgoing_l1: dict[str, list[str]] = {}
                    _tv_ts_l1: dict = {}
                    try:
                        _all_candidate_ids = list(_candidate_recs.keys())
                        # Temporal ts from the candidates themselves (created_at populated
                        # by _from_row for ANN results and get_batch rows).
                        for _rec in _candidate_recs.values():
                            _ca = getattr(_rec, "created_at", None)
                            if _ca is not None:
                                _tv_ts_l1[str(_rec.id)] = _ca
                        # UNCAPPED contradicts-dst (top_k=None) for temporal validity.
                        _contr_edges = store.incident_edges(
                            _all_candidate_ids,
                            edge_types=["contradicts"],
                            top_k=None,
                        )
                        # Build outgoing contradicts map + hydrate dst records for ts.
                        _contr_dst_ids = []
                        for _src_id, _edges in _contr_edges.items():
                            for (_dst, _et, _wt) in _edges:
                                _src_s = str(_src_id)
                                _dst_s = str(_dst)
                                _tv_outgoing_l1.setdefault(_src_s, []).append(_dst_s)
                                if _dst not in _candidate_recs:
                                    _contr_dst_ids.append(_dst)
                        if _contr_dst_ids:
                            _contr_recs = store.get_batch(_contr_dst_ids)
                            for _cr in _contr_recs.values():
                                _ca = getattr(_cr, "created_at", None)
                                if _ca is not None:
                                    _tv_ts_l1[str(_cr.id)] = _ca
                    except Exception as _tv_exc:  # noqa: BLE001 — degrade gracefully
                        logger.debug("layer1_tv_build_failed: %s", _tv_exc)
                        _tv_outgoing_l1, _tv_ts_l1 = {}, {}

                    resp = recall_for_response(
                        store=store,
                        graph=graph,
                        assignment=assignment,
                        rich_club=rc,
                        embedder=embedder,
                        cue=params["cue"],
                        session_id=params.get("session_id", "unknown"),
                        budget_tokens=params.get("budget_tokens") or _arousal_budget_tokens,
                        profile_state=_profile_state,
                        mode=cue_mode,
                        knobs_applied=knobs_applied,
                        arousal_state=_arousal_diag,
                        # Pre-built candidate-scoped tv_maps (bypasses O(N) scan).
                        tv_maps=(_tv_outgoing_l1, _tv_ts_l1) if _tv_ts_l1 else None,
                    )
                    # Deterministic unsampled anti-masking marker.
                    resp.ann_path_used = True
                    # Recall-path observability (best-effort): the daemon-up
                    # in-process MCP path served this recall.  Emit encode_ms
                    # (measured tightly around the clean embed call above) so
                    # the daemon-up tail is observable and fallback_rate is
                    # derivable across sources.  This emit has its OWN complete
                    # try/except so it can NEVER fall through to the outer
                    # soft-availability fallback (which would re-run recall) —
                    # observability must not change recall logic. Payload carries
                    # cue-DERIVED metrics only (no cue text).
                    try:
                        from iai_mcp.events import emit_best_effort, TELEMETRY_RECALL_SOURCE
                        _du_data = {"source": "daemon"}
                        if _encode_ms is not None:
                            _du_data["encode_ms"] = round(_encode_ms, 2)
                        emit_best_effort(
                            store,
                            TELEMETRY_RECALL_SOURCE,
                            _du_data,
                            severity="info",
                            session_id=params.get("session_id", "unknown"),
                        )
                    except Exception:  # noqa: BLE001 -- telemetry must never break recall
                        pass
                except NativeError:
                    # A native-extension failure (encode or graph compute) must
                    # propagate — the soft availability fallback below does NOT
                    # apply.  Re-raise so the caller sees a loud stop.
                    raise
                except Exception as exc:  # noqa: BLE001 -- soft availability fallback
                    # Graph-build / pipeline soft failure (cache miss, community
                    # detection OOM, etc.): degrade to baseline cosine recall.
                    # Verbatim default protects the availability invariant on
                    # transient infrastructure errors.
                    logger.warning("recall_pipeline_fallback: %s", exc)
                    # Update arousal on pipeline error.
                    try:
                        _update_arousal(_arousal_state, "error")
                    except Exception:  # noqa: BLE001 -- arousal update fail-safe
                        pass
                    cue_embedding = params.get("cue_embedding") or [0.0] * EMBED_DIM
                    resp = retrieve.recall(
                        store=store,
                        cue_embedding=cue_embedding,
                        cue_text=params["cue"],
                        session_id=params.get("session_id", "unknown"),
                        budget_tokens=params.get("budget_tokens") or _arousal_budget_tokens,
                        mode=cue_mode,
                    )
        # Arousal-budget lifecycle: update arousal after recall result resolves.
        # success = hits found; failed = zero hits. Wrapped in try/except so
        # a broken arousal module never corrupts the recall response.
        try:
            _arousal_event = "recall_success" if resp.hits else "recall_failed"
            _update_arousal(_arousal_state, _arousal_event)
        except Exception:  # noqa: BLE001 -- arousal update fail-safe
            pass

        response = {
            "hits": [_hit_to_json(h) for h in resp.hits],
            "anti_hits": [_hit_to_json(h) for h in resp.anti_hits],
            "activation_trace": [str(x) for x in resp.activation_trace],
            "budget_used": resp.budget_used,
            # Surface classified mode and displaced concept-mode schema records.
            "cue_mode": resp.cue_mode,
            "patterns_observed": list(resp.patterns_observed or []),
            # Audit accumulator: populated by recall_for_response upstream;
            # apply_profile below extends in place with helper-keyed entries.
            "_knobs_applied": knobs_applied,
            # Deterministic unsampled marker: True when the bounded ANN-first
            # path was taken; False on the soft-fallback. Back-compat default
            # False via getattr so callers on the soft-fallback path are safe.
            "ann_path_used": getattr(resp, "ann_path_used", False),
        }
        # CQRS sleep detection: stamp cortex-fallback source marker.
        if _cortex_fallback:
            response["_source"] = "cortex-fallback"
        # Stamp cold-structural-degrade when the ANN path
        # served with empty structural bias (truly-cold never-consolidated
        # store). Observable, not silent.
        if not _cortex_fallback and _structural_source == "cold_degrade":
            response["_source"] = "cold-structural-degrade"
        # Record recall latency for self-regulation.
        try:
            _recall_ms = (_time.perf_counter() - _recall_t0) * 1000
            response["_recall_latency_ms"] = round(_recall_ms, 1)
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("recall_latency_measure_failed: %s", exc)
        # Surface unresolved curiosity signals for proactive clarification.
        try:
            from iai_mcp.curiosity import get_pending_questions
            _qs = get_pending_questions(store, limit=2)
            if _qs:
                response["curiosity_signals"] = _qs
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("curiosity_signals_failed: %s", exc)
        # Reconsolidation labile-write: stamp labile_until on every recalled
        # hit so the next REM reconsolidation pass can find them.
        # reinforce_record(is_retrieval=True) gates the labile-stamp on the
        # kwarg AND on the dry_run config. Wrapped in try/except so a
        # labile-write failure does NOT corrupt the memory_recall hot-path.
        try:
            for hit in resp.hits:
                store.reinforce_record(hit.record_id, is_retrieval=True)
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("labile_write_failed: %s", exc)
        # Inject sleep_suggestion when dual-gate passes.
        _inject_sleep_suggestion(
            response,
            cue=params.get("cue", ""),
            language=params.get("language", "en"),
        )
        # First memory_recall of the day (>18h since last shown OR never shown)
        # carries the overnight digest. daemon_state.get_pending_digest clears
        # the digest from state so it appears exactly once per 18h window.
        _inject_overnight_digest(response, store=store)
        # First-turn auto-recall hook. Fires exactly once per session; runs a
        # scoped recall and injects `first_turn_recall` field. Silent-fail.
        _first_turn_recall_hook(response, params=params, store=store)
        # Server-side profile knob decorator. Knob names never cross the MCP wire.
        try:
            from iai_mcp.response_decorator import apply_profile
            apply_profile(response, _profile_state)
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("apply_profile_failed: %s", exc)
        # Pask teachback: pre-injection contradiction verification.
        # CALL-ON-DEMAND _load_pask_config so env-var monkeypatch in tests is
        # visible. Try/except guards the whole block — non-critical and any
        # failure must NOT break the memory_recall hot path.
        try:
            from iai_mcp.daemon_config import _load_pask_config
            from iai_mcp.events import write_event
            from iai_mcp.pask_teachback import verify_hit_set
            pask_cfg = _load_pask_config()
            if pask_cfg.enabled:
                hit_ids = [
                    h.record_id if hasattr(h, "record_id") else h.get("record_id")
                    for h in resp.hits
                ]
                hit_ids = [h for h in hit_ids if h is not None]
                teachback = verify_hit_set(store, hit_ids)
                response["pask_teachback"] = teachback
                try:
                    write_event(
                        store,
                        "pask_teachback_pass",
                        {
                            "hit_count": teachback["hit_count"],
                            "has_contradictions": teachback["has_contradictions"],
                            "contradiction_count": len(teachback["contradiction_pairs"]),
                            "dry_run_mode": pask_cfg.dry_run,
                        },
                        severity="info",
                    )
                except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
                    logger.debug("pask_teachback_event_failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("pask_teachback_failed: %s", exc)
        # Trajectory coupling: store mean embedding of injected hits for
        # next-turn coupling measurement.
        try:
            if resp.hits:
                import numpy as _np
                embeddings = [h.embedding for h in resp.hits if hasattr(h, "embedding") and h.embedding]
                if not embeddings:
                    _emb_cache = {}
                    for h in resp.hits[:5]:
                        rec = store.get(h.record_id)
                        if rec and rec.embedding:
                            _emb_cache[h.record_id] = rec.embedding
                    embeddings = list(_emb_cache.values())
                if embeddings:
                    _last_injection_embedding = _np.mean(embeddings, axis=0).tolist()
                    _last_injection_ids = [str(h.record_id) for h in resp.hits[:5]]
                else:
                    _last_injection_embedding = None
                    _last_injection_ids = []
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("trajectory_coupling_store_failed: %s", exc)
            _last_injection_embedding = None
            _last_injection_ids = []
        return response

    # memory_recall_structural: structural query enters the pipeline via
    # role->filler dict. Pure numpy + bytewise XOR -- zero LLM token cost,
    # no Embedder() instantiated, no anthropic client touched. Structural
    # queries are first-class peers of cosine retrieval.
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
    # --- /memory_recall_structural dispatch ---

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
        # Trajectory coupling: measure cos(last_injection, this_capture).
        if _last_injection_embedding:
            try:
                import numpy as _np
                from iai_mcp.embed import embedder_for_store
                from iai_mcp.events import write_event
                _emb = embedder_for_store(store)
                _cap_vec = _emb.embed(params["text"])
                _inj_vec = _np.asarray(_last_injection_embedding, dtype=_np.float32)
                _cap_arr = _np.asarray(_cap_vec, dtype=_np.float32)
                _n1 = float(_np.linalg.norm(_inj_vec))
                _n2 = float(_np.linalg.norm(_cap_arr))
                _coupling = float(_np.dot(_inj_vec, _cap_arr) / (_n1 * _n2)) if _n1 > 0 and _n2 > 0 else 0.0
                write_event(
                    store,
                    kind="trajectory_coupling",
                    data={
                        "coupling_score": round(_coupling, 4),
                        "injected_ids": _last_injection_ids[:5],
                        "direction": "toward" if _coupling > 0.3 else "neutral",
                    },
                    severity="info",
                    session_id=params.get("session_id", "-"),
                )
            except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
                logger.debug("trajectory_coupling_measure_failed: %s", exc)
            _last_injection_embedding = None
            _last_injection_ids = []
        result = capture_turn(
            store,
            cue=params.get("cue", ""),
            text=params["text"],
            tier=params.get("tier", "episodic"),
            session_id=params.get("session_id", "-"),
            role=params.get("role", "user"),
        )
        # Flush the record buffer immediately so the captured record is
        # visible to same-session reads (episodes_recent, iai last) without
        # waiting for the next periodic tick (~30 s).  The periodic tick also
        # flushes, so the only consequence of this call failing is a slightly
        # delayed appearance — the tick is the safety net.
        try:
            from iai_mcp.store import flush_record_buffer
            flush_record_buffer(store)
        except Exception:  # noqa: BLE001
            pass
        return result

    # memory_consolidate: real sleep cycle dispatch.
    # The tool signature: {"method":"memory_consolidate","params":{"session_id": "..."}}
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
            # no paid-API surface remains; subscription
            # gating happens inside claude_cli, not via env-key probe.
            has_api_key=False,
        )
        # Normalise JSON-friendly output (no dataclasses).
        return {
            "mode": result["mode"],
            "tier": result["tier"],
            "summaries_created": int(result["summaries_created"]),
            "decay_result": dict(result["decay_result"]),
            "schema_candidates": list(result["schema_candidates"]),
        }

    # session_exit: light consolidation entry point; also emits M1..M6 trajectory events.
    if method == "session_exit":
        from iai_mcp.sleep import run_light_consolidation
        from iai_mcp.trajectory import (
            compute_session_metrics_snapshot,
            record_session_metrics,
        )

        sid = params.get("session_id", "-")
        result = run_light_consolidation(store, session_id=sid)
        # Trajectory emission.
        snapshot = compute_session_metrics_snapshot(store, sid)
        record_session_metrics(store, session_id=sid, metrics=snapshot)
        result["trajectory_metrics_emitted"] = len(snapshot)
        return result

    # s5_propose: S5 identity kernel. Internal method -- not advertised on
    # the MCP tools/list surface, but the dispatch hook is live so tests and
    # subagents can call it.
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
    # Learning-layer methods:
    #
    # - profile_update_from_signal: Bayesian update; accepts {knob, signal, observed}
    #   and mutates _profile_state + _posterior_state.
    # - schema_induce: manual trigger for Tier-0 fallback; returns the
    #   SchemaCandidate list without persisting.
    # - curiosity_pending: returns unresolved curiosity questions optionally
    #   filtered by session_id.
    # - trajectory_record: writes M1..M6 events for a session.
    if method == "profile_update_from_signal":
        from iai_mcp.profile import bayesian_update

        global _posterior_state
        knob = params["knob"]
        signal = params["signal"]
        observed = params["observed"]
        # Serialize the read-modify-write of _profile_state and the rebind of
        # _posterior_state (see _profile_lock declaration above).
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

    # schema_list: walks all records tagged "schema", parses pattern /
    # confidence / status from tags + literal_surface, and counts
    # `schema_instance_of` inbound edges per schema. Supports domain +
    # confidence_min filters.
    # events_query: strict whitelist of user-visible event kinds. Rejects
    # identity-kernel kinds (s5_invariant_update etc) to preserve trust boundary.
    if method == "schema_list":
        return _schema_list_dispatch(store, params)

    if method == "events_query":
        return _events_query_dispatch(store, params)

    # audit_query: returns newest-first identity-relevant events. Caller may
    # pass since_iso (ISO-8601 UTC) + kinds override; shield payloads are NOT
    # redacted here (dispatch is trusted; CLI redacts for display).
    # detect_drift: one-shot drift check; returns any s5_drift_alert payloads.
    # shield_check: pure evaluate_injection_risk wrapper; does NOT mutate the store.
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
    # topology: read-only snapshot of the current runtime graph
    # (N, C, L, sigma, community_count, rich_club_ratio, regime).
    # Purely diagnostic — retrieval modes NEVER toggle based on sigma.
    if method == "topology":
        from iai_mcp import sigma as sigma_mod
        from iai_mcp.events import write_event

        records_count = store.db.open_table("records").count_rows()
        if records_count == 0:
            return {
                "N": 0, "C": 0.0, "L": 0.0, "sigma": None,
                "community_count": 0, "rich_club_ratio": 0.0,
                "regime": "insufficient_data",
            }
        try:
            graph_bundle = retrieve.build_runtime_graph(store)
            graph = graph_bundle[0] if isinstance(graph_bundle, tuple) else graph_bundle
            return sigma_mod.compute_topology_snapshot(graph)
        except Exception as exc:
            write_event(
                store,
                "topology_native_failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise

    # camouflaging_status: read-only detector report over the last weekly window.
    # NEVER models the user; observes surface formality trajectory only.
    # Calling this does NOT relax the register — that pathway runs on the weekly pass.
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

    # initiate_sleep_mode: explicit user consent gate.
    # Consent=False returns immediately without touching the daemon socket.
    # Consent=True sends {"type":"user_initiated_sleep"} NDJSON over the
    # ~/.iai-mcp/.daemon.sock unix socket and returns the daemon's response.
    if method == "initiate_sleep_mode":
        return asyncio.run(handle_initiate_sleep_mode(params))

    # force_wake: cooperative wake. Sends {"type":"force_wake"} over the socket
    # and waits up to 15 min for daemon to complete current REM cycle and yield.
    # Graceful when daemon is unreachable.
    if method == "force_wake":
        return asyncio.run(handle_force_wake(params))

    if method == "profile_get":
        # Full 11-knob registry via profile module (10 AUTIST + 1 wake_depth).
        return profile.profile_get(params.get("knob"), _profile_state)

    if method == "profile_set":
        # Pass store so a successful change emits kind='profile_updated' for
        # trajectory tracking. Serialize to prevent concurrent interleave on
        # read-modify-write on the same knob.
        with _profile_lock:
            return profile.profile_set(
                params["knob"], params["value"], _profile_state, store=store,
            )

    if method == "session_start_payload":
        # Session-start assembly: assemble_session_start emits kind='session_started'
        # for context-repeat-rate measurement. Profile state is threaded so
        # wake_depth reaches the assembler.
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

        # User-model predictive prefetch: load the persisted UserModel (or default
        # on first run), run the prefetcher, prepend result-ids to payload.l2
        # dedup'd. Fail-safe: any exception here MUST NOT crash session_start_payload.
        try:
            from iai_mcp.user_model import (
                UserModelPrefetcher,
                load as _user_model_load,
            )
            from iai_mcp.daemon_config import _load_user_model_config
            _user_model_cfg = _load_user_model_config()
            _user_model = _user_model_load()
            _prefetched_ids = UserModelPrefetcher().prefetch(
                store, _user_model, top_k=_user_model_cfg.prefetch_top_k,
            )
            if _prefetched_ids:
                # Prepend prefetched ids; dedup against existing l2 entries.
                # Cap total l2 length at len(prior) + top_k so the prefetch
                # never explodes the payload.
                _existing = set(payload.l2)
                _new = [
                    rid for rid in _prefetched_ids if rid not in _existing
                ]
                payload.l2 = _new + list(payload.l2)
                _cap = len(_existing) + _user_model_cfg.prefetch_top_k
                if len(payload.l2) > _cap:
                    payload.l2 = payload.l2[:_cap]
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            import logging
            logging.getLogger(__name__).warning(
                "user_model_prefetch_failed",
                extra={
                    "err_type": type(exc).__name__,
                    "err": str(exc)[:120],
                },
            )

        return _payload_to_json(payload)

    if method == "session_refresh_if_stale":
        # Two-stage drain before re-reading MAX(created_at):
        #
        # 1. drain_deferred_captures: ended / renamed files (the normal path).
        # 2. drain_active_live_captures: partial offset-tracked drain of OTHER
        #    sessions' still-open .live.jsonl files (exclude the refreshing
        #    session's own file so it never self-triggers).
        #
        # Both drains run synchronously (dispatch is not async).  Failure in
        # either returns rendered="" rather than propagating into the RPC layer.
        #
        # Does NOT emit session_started. Uses _compose_session_start_payload
        # (the emit-free path) with forced wake_depth="standard" so the result
        # is non-empty even when the user's profile default is wake_depth="minimal".
        from iai_mcp.capture import drain_active_live_captures, drain_deferred_captures
        from iai_mcp.session import (
            SESSION_START_CACHE_MAX_CHARS,
            _compose_session_start_payload,
            format_payload_as_markdown,
            max_record_created_at,
        )

        caller_watermark = params.get("watermark") or ""
        refreshing_session_id = params.get("session_id", "-")

        try:
            drain_deferred_captures(store)
        except Exception as _drain_exc:  # noqa: BLE001
            logger.warning(
                "session_refresh_drain_failed",
                extra={"err": str(_drain_exc)[:120]},
            )
            return {"rendered": "", "new_max_ts": ""}

        try:
            drain_active_live_captures(store, exclude_session_id=refreshing_session_id)
        except Exception as _live_drain_exc:  # noqa: BLE001
            logger.warning(
                "session_refresh_live_drain_failed",
                extra={"err": str(_live_drain_exc)[:120]},
            )
            # Live-drain failure is non-fatal: ended-file drain already ran,
            # so we continue with whatever the store already contains.

        # Flush the sync record-write buffer to SQLite so the just-drained
        # live-file records are visible to max_record_created_at below.
        # drain_active_live_captures uses the same buffered store.insert()
        # path as all other capture routes; without an explicit flush here
        # the rows live in the in-memory buffer and the MAX(created_at) query
        # returns the pre-drain watermark, causing the rendered brief to omit
        # the new records.  Failure is non-fatal: the WAKE tick will flush
        # them on the next cycle.
        try:
            from iai_mcp.store import flush_record_buffer
            flush_record_buffer(store)
        except Exception as _flush_exc:  # noqa: BLE001
            logger.warning(
                "session_refresh_flush_failed",
                extra={"err": str(_flush_exc)[:120]},
            )

        new_max_ts = max_record_created_at(store)

        # Normalize both timestamps to UTC ISO (T-separator) before comparing
        # so that mixed SQLite-space-format vs watermark-T-format strings do
        # not produce false "nothing new" results via raw lexicographic order
        # (space 0x20 < T 0x54 would make any SQLite timestamp appear older
        # than the same instant stored in T-format by the watermark sidecar).
        def _norm(ts: str) -> str:
            try:
                from datetime import datetime, timezone as _tz
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00").replace(" ", "T"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_tz.utc)
                return dt.astimezone(_tz.utc).isoformat()
            except (TypeError, ValueError):
                return ts

        _new_max_norm = _norm(new_max_ts) if new_max_ts else ""
        _wm_norm = _norm(caller_watermark) if caller_watermark else ""

        if not new_max_ts or (caller_watermark and _new_max_norm <= _wm_norm):
            return {"rendered": "", "new_max_ts": new_max_ts or ""}

        _graph, assignment, rc = retrieve.build_runtime_graph(store)
        payload = _compose_session_start_payload(
            store,
            assignment,
            rc,
            session_id=params.get("session_id", "-"),
            profile_state={"wake_depth": "standard"},
        )
        rendered = format_payload_as_markdown(payload)
        if len(rendered) > SESSION_START_CACHE_MAX_CHARS:
            rendered = rendered[:SESSION_START_CACHE_MAX_CHARS]
        return {"rendered": rendered, "new_max_ts": new_max_ts}

    if method == "episodes_recent":
        # Return the N most-recent role:user episodic records, time-desc.
        # n is clamped to [0, 1000] as a DoS guard. session_id filtering
        # happens in Python after store decryption (provenance_json is
        # AES-GCM encrypted — SQL WHERE on it silently returns nothing).
        # Pending live events are merged for immediate (pre-drain) recall.
        from iai_mcp.capture import read_pending_live_events
        n = max(0, min(int(params.get("n", 10)), 1000))
        session_id = params.get("session_id")
        pending = read_pending_live_events(session_id=session_id)
        records = store.recent_user_turns(n, session_id=session_id, pending_live_events=pending)
        turns = []
        for r in records:
            if r.id is None:
                # Pending turn: build a non-None record_id.
                su = getattr(r, "_pending_source_uuid", None)
                idem = getattr(r, "_pending_idem_tag", "")
                if su:
                    rid = f"pending:{su}"
                else:
                    # Use the hex suffix of the idem tag (format: "idem:<hex>")
                    idem_hex = idem[5:] if idem.startswith("idem:") else idem
                    rid = f"pending:{idem_hex}" if idem_hex else f"pending:unknown"
            else:
                rid = str(r.id)
            turns.append({
                "record_id": rid,
                "literal_surface": r.literal_surface,
                "session_id": (r.provenance or [{}])[0].get("session_id"),
                "captured_at": (
                    r.created_at.isoformat() if r.created_at else None
                ),
            })
        return {"turns": turns, "count": len(turns)}

    if method == "drain_permanent_failed":
        # Recover terminal .permanent-failed-*.jsonl files via the daemon socket.
        # The daemon owns the HippoDB exclusive lock; routing here avoids a
        # second writer against the live store. Direct-open fallback in the CLI
        # activates only when the daemon is down (socket absent/unreachable).
        from iai_mcp.capture import drain_permanent_failed_files
        from pathlib import Path as _Path

        dry_run = bool(params.get("dry_run", False))
        # Resolve deferred_dir from store.root (the iai-mcp home root).
        # MemoryStore.root is e.g. ~/.iai-mcp; deferred captures live at
        # ~/.iai-mcp/.deferred-captures alongside it.
        try:
            deferred_dir = _Path(store.root) / ".deferred-captures"
        except Exception:  # noqa: BLE001 -- deferred_dir=None triggers default resolution
            deferred_dir = None
        result = drain_permanent_failed_files(store, deferred_dir=deferred_dir, dry_run=dry_run)
        return result

    raise UnknownMethodError(method)


def _hit_to_json(h) -> dict:
    # Derived temporal validity, computed
    # at recall time from the contradicts-edge graph. None when the record
    # has no superseding contradiction (valid_to) or when enrichment was
    # not run on this code path (back-compat default — applies to
    # recall_for_benchmark and any pre-U2 caller that constructs
    # MemoryHit-shaped objects). getattr fallback defends against any
    # future MemoryHit-shaped object the serializer might be handed
    # without the new fields (partial mock in a test, etc.). The
    # _stale_downweighted sentinel from apply_stale_downweight is
    # intentionally NOT serialized — only the public MCP-01 fields plus
    # valid_from / valid_to cross onto the JSON wire.
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
        "session_id": getattr(h, "session_id", None),
        "captured_at": getattr(h, "captured_at", None),
    }


# events_query whitelist. Only the user-introspection-safe subset is exposed
# via the MCP surface; identity-kernel kinds stay internal.
EVENTS_QUERY_WHITELIST: frozenset[str] = frozenset({
    "s4_contradiction",
    "trajectory_metric",
    "schema_induction_run",
    "llm_health",
    "curiosity_silent_log",
    "curiosity_question",
    "cls_consolidation_run",
    "crypto_key_rotated",
    "session_started",
    # Recall-path observability: source + construct/encode timing, so the
    # fallback_rate (recency-degrade / total) is DERIVABLE from the counts at
    # query time. Payloads carry cue-DERIVED metrics only (no cue text).
    # Must equal TELEMETRY_RECALL_SOURCE / TELEMETRY_EMBED_CONSTRUCT.
    "recall_source",
    "embed_construct",
})


def _schema_list_dispatch(store: MemoryStore, params: dict) -> dict:
    """schema_list: walk all records tagged "schema" (created by schema.persist_schema).

    Parses pattern + confidence + status from record tags + literal_surface.
    Counts schema_instance_of inbound edges for evidence_count.
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
    """events_query: whitelist-gated event lookup.

    Parses since as ISO-8601. Caps limit at 1000. Returns events with
    ISO-string timestamps (pandas Timestamps are not JSON-serialisable).
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
            except (ValueError, TypeError, AttributeError) as exc:
                logger.debug("ts_isoformat_failed: %s", exc)
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


# Daemon communication helpers. Entry points:
#   - _send_to_daemon: internal NDJSON helper over ~/.iai-mcp/.daemon.sock
#   - handle_initiate_sleep_mode: JSON-RPC method with user consent gate
#   - handle_force_wake: JSON-RPC method with 15-min cooperative cap
#   - _inject_sleep_suggestion: memory_recall dispatch hook
#
# Invariant: the socket WRITE in handle_initiate_sleep_mode
# is unreachable unless params["consent"] is literally True. Short-circuits
# on missing key, wrong type, or False.


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
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("socket_writer_close_failed: %s", exc)


async def handle_initiate_sleep_mode(params: dict) -> dict:
    """User-consent gate for daemon sleep mode.

    Strict schema validation: raises ValueError for missing or wrong-typed
    params. Returns a dict in the normal path.

    The socket write is unreachable unless
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
    """Cooperative force-wake.

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
    """Inject `sleep_suggestion` into a memory_recall response when the
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
    except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
        # Silent fail -- memory_recall is the hot path and must not break.
        logger.debug("sleep_suggestion_failed: %s", exc)


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
    """Inject ``overnight_digest`` key into every memory_recall response.

    The digest lives inside ``.daemon-state.json``;
    ``daemon_state.get_pending_digest`` handles the 18h timing gate and
    CLEARS the digest from state on delivery, so the rich payload still
    surfaces exactly once per window.

    The ``overnight_digest`` key is ALWAYS present in the mutated response.
    When the daemon has a pending digest within the 18h once-per-window gate,
    the payload is the rich dict; otherwise it is ``_EMPTY_OVERNIGHT_DIGEST``
    (structured zeros), guaranteeing a stable shape across stdio and socket
    transports regardless of daemon timing.

    Silent-fail on any exception: corrupt state, disk failure, or schema drift
    must NEVER break the memory_recall hot path. On exception the zeroed
    default is still written first so determinism holds even on a daemon-state
    IO hiccup; when ``store`` is provided, a best-effort ``digest_inject_error``
    warning event is emitted.
    """
    try:
        from iai_mcp.daemon_state import load_state as _load_state
        from iai_mcp.daemon_state import get_pending_digest as _get_pending_digest
        state = _load_state()
        now = datetime.now(timezone.utc)
        digest = _get_pending_digest(state, now)
        if not digest:
            # Deterministic contract -- key always present, zeroed default when
            # no digest is pending. Copy to avoid sharing the module-level
            # mutable default across responses.
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
            except Exception as exc2:  # noqa: BLE001 -- MCP boundary fail-safe
                logger.debug("digest_inject_error_event_failed: %s", exc2)


def _first_turn_recall_hook(
    response: dict,
    *,
    params: dict,
    store: MemoryStore,
) -> None:
    """First-turn auto-recall hook.

    Fires exactly once per session. Runs a scoped ``retrieve.recall`` with
    a capped budget (400 tok) using the user's cue as-is, clamped to 2000
    chars. Injects the result as ``first_turn_recall`` in the response.
    Silent-fail on any exception: the hot recall path must not break.

    Security:
    - Input-length clamp: `cue[:2000]` before handing to recall.
    - Never calls any paid API.

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
        # Consult the HIPPEA cascade warm LRU BEFORE going cold. The LRU is
        # populated by the daemon-side cascade on session_open. If empty
        # (daemon down or cascade hasn't fired yet) fall through to cold baseline.
        warm_hit_ids: list = []
        try:
            from iai_mcp.hippea_cascade import snapshot_warm_ids
            warm_hit_ids = snapshot_warm_ids()
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("snapshot_warm_ids_failed: %s", exc)
            warm_hit_ids = []

        # Cross-process closure: when the daemon's LRU is not visible to this
        # process, fire a synchronous cascade once per session and populate
        # the core-local LRU. Cost is one-time per session.
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
                    except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
                        logger.debug("warm_lru_store_get_failed rid=%s: %s", rid, exc)
                        continue
                _CORE_CASCADE_FIRED_PER_SESSION.add(str(session_id))
                if _CORE_WARM_LRU:
                    warm_hit_ids = list(_CORE_WARM_LRU.keys())
                    warm_lru_source = "core_fallback"
            except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
                # Cascade failed; cold path still runs below. Hot path
                # must never break.
                logger.debug("core_cascade_failed: %s", exc)

        # Scoped recall: capped budget (400 tok per D5-03), modest k.
        # The warm LRU hint is surfaced in the response so observability can
        # measure whether the cascade is firing on this process -- but the
        # authoritative hit set stays the cold recall path so verbatim recall
        # correctness (M-04) is unchanged by LRU population.
        cue_embedding = params.get("cue_embedding") or [0.0] * EMBED_DIM
        # The first-turn hook is a session-warm-up signal over all tiers —
        # concept-mode semantics. Pin explicitly to avoid the default verbatim
        # mode (which would restrict to episodic-only candidates).
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
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("first_turn_recall_event_failed: %s", exc)
    except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
        # Hot path must not break. The absence of `first_turn_recall`
        # in the response IS the signal that the hook did not fire.
        logger.debug("first_turn_recall_hook_failed: %s", exc)


def _payload_to_json(payload) -> dict:
    """Serialise SessionStartPayload for JSON-RPC transport."""
    return {
        "l0": payload.l0,
        "l1": payload.l1,
        "l2": list(payload.l2),
        "rich_club": payload.rich_club,
        "total_cached_tokens": int(payload.total_cached_tokens),
        "total_dynamic_tokens": int(payload.total_dynamic_tokens),
        "breakpoint_marker": payload.breakpoint_marker,
        "identity_pointer": getattr(payload, "identity_pointer", ""),
        "brain_handle": getattr(payload, "brain_handle", ""),
        "topic_cluster_hint": getattr(payload, "topic_cluster_hint", ""),
        "compact_handle": getattr(payload, "compact_handle", ""),
        "wake_depth": getattr(payload, "wake_depth", "minimal"),
    }


# --------------------------------------------------------------------- daemon

def main() -> None:
    """stdio JSON-RPC loop -- reads one JSON object per line, writes responses.

    Announces the user's IANA timezone on boot so users can see at a glance
    how their sleep-cycle quiet_window and CLI timestamps are interpreted.
    Quiet by default; logs to stderr to avoid polluting the JSON-RPC channel.
    """
    _require_native()

    store = MemoryStore()
    _seed_l0_identity(store)

    # Timezone announcement (stderr, not stdout -- stdout is JSON-RPC).
    try:
        from iai_mcp.tz import load_user_tz
        tz = load_user_tz()
        sys.stderr.write(f"iai-mcp: timezone={tz.key}\n")
        sys.stderr.flush()
    except Exception as e:  # noqa: BLE001 pragma: no cover -- boot diagnostics must not break
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
        except Exception as e:  # noqa: BLE001 -- MCP boundary fail-safe
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
