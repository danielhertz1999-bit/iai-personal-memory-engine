"""Retrieval + reinforcement + contradiction paths.

- `recall`: baseline cosine top-k -- kept as a fallback for the
  empty-store case and for regression tests.
- `build_runtime_graph`: reconstruct a MemoryGraph + CommunityAssignment +
  rich-club from store state; consumed by core.py to drive `pipeline_recall`.
- `reinforce_edges`, `contradict`: reinforce/contradicts-edge helpers.
- `link_temporal_next`: records a `record_inserted` event
  and creates a `temporal_next` edge from the previous same-session insertion
  to the new record if that event happened within the last 5 minutes.

Rules enforced here:
- Every recall appends a provenance entry to every returned record.
- reinforce boosts pairwise Hebbian edges among co-retrieved ids.
- contradict creates a linked record, preserves the original.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from itertools import combinations
from uuid import UUID, uuid4

from iai_mcp.aaak import enforce_english_raw, generate_aaak_index
from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import (
    EMBED_DIM,
    EdgeUpdate,
    MemoryHit,
    MemoryRecord,
    RecallResponse,
    ReconsolidationReceipt,
)


# Structured-log handle for the graph-build decrypt-failure path.
# Used by the `graph_build_decrypt_failed` event when AES-GCM decrypt
# of a record's literal_surface raises during build_runtime_graph.
log = logging.getLogger(__name__)

# Per-process rate limit for graph_build_decrypt_failed (rid -> monotonic ts).
_GRAPH_DECRYPT_WARN_LAST: dict[str, float] = {}
_GRAPH_DECRYPT_WARN_INTERVAL_SEC = 300.0


# Temporal-next window. Records inserted within this window
# in the same session are linked with a temporal_next edge.
TEMPORAL_NEXT_WINDOW = timedelta(minutes=5)


# Downweight factor applied to MemoryHit.score when the
# derived valid_to < now (record was superseded by a newer contradicting
# record). 0.5 chosen as: aggressive enough that a fresh lower-cosine record
# can outrank a stale high-cosine hit on typical score distributions, while
# leaving the stale hit visible in the response for audit.
# Design intent: "downweight, not hide, to preserve audit trail."
STALE_DOWNWEIGHT_FACTOR: float = 0.5

# Suffix appended to MemoryHit.reason when a record is downweighted as stale.
# Spaces around · are intentional — the reason field already carries
# "cosine X.XXX + …" segments separated by " + "; " · stale" stays readable.
_STALE_REASON_SUFFIX: str = " · stale"


def recall(
    store: MemoryStore,
    cue_embedding: list[float],
    cue_text: str,
    session_id: str,
    budget_tokens: int = 1500,
    k_hits: int = 5,
    k_anti: int = 3,
    mode: str = "verbatim",
) -> RecallResponse:
    """Baseline cosine-similarity retrieval.

    Fetches top (k_hits + k_anti) by cosine similarity; treats the top k_hits as
    excitatory hits and the bottom k_anti as anti-hit candidates.

    Every returned hit gets a provenance entry appended.

    `mode` kwarg defaults to 'verbatim'. The baseline is the conservative
    fallback path (used by core.dispatch when the runtime graph is unavailable /
    build fails / store is empty). Defaulting to verbatim protects the
    North-Star ≥99% essential variable on the degraded path — the user never
    silently lands on a schema-dominated surface even when the full pipeline is
    unreachable. Verbatim mode applies the same tier filter + schema exclusion
    as pipeline_recall verbatim mode so the contract on hits[] is identical
    regardless of which route core dispatched to. Concept mode preserves
    the pure-cosine baseline (no filter).
    """
    raw = store.query_similar(cue_embedding, k=k_hits + k_anti)

    # Verbatim mode candidate filter on the baseline path.
    # tier='episodic' AND no pattern:* tag — same exclusion contract as
    # pipeline_recall verbatim mode. Also excludes
    # tier='semantic_pruned' soft-deleted schemas naturally.
    if mode == "verbatim":
        raw = [
            (rec, score) for rec, score in raw
            if rec.tier == "episodic"
            and not any(t.startswith("pattern:") for t in (rec.tags or []))
        ]

    hits: list[MemoryHit] = []
    # Collect provenance entries during the hit-building loop, flush via ONE
    # store.append_provenance_batch call after the loop closes. Replaces the
    # per-hit `store.append_provenance(record.id, entry)` pattern that
    # produced a 64x wall-clock blow-up and rank perturbation under memory
    # pressure (measured on an 8 GB M1).
    provenance_pending: list[tuple[UUID, dict]] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for record, score in raw[:k_hits]:
        _prov = (record.provenance or [{}])[0]
        hits.append(
            MemoryHit(
                record_id=record.id,
                score=float(score),
                reason=f"cosine {score:.3f}",
                literal_surface=record.literal_surface,
                adjacent_suggestions=[],  # filled by the adjacency stage
                session_id=_prov.get("session_id"),
                captured_at=record.created_at.isoformat() if record.created_at else None,
            )
        )
        # Every recall appends a provenance entry; write is batched
        # end-of-loop to preserve rank stability.
        provenance_pending.append((
            record.id,
            {
                "ts": now_iso,
                "cue": cue_text,
                "session_id": session_id,
            },
        ))

    # Provenance flush: route through the store's non-blocking queue when
    # enabled, falling back to the synchronous batch write otherwise.  Never
    # block the recall hot path on a provenance-write failure.
    if provenance_pending:
        try:
            store.queue_provenance_batch(provenance_pending)
        except (OSError, ValueError, RuntimeError) as exc:
            log.warning("provenance_batch write failed: %s", exc)

    anti_hits: list[MemoryHit] = []
    # Anti-hit stub: bottom-k of the same query (low-similarity candidates).
    tail = raw[-k_anti:] if len(raw) >= k_anti else []
    for record, score in reversed(tail):
        anti_hits.append(
            MemoryHit(
                record_id=record.id,
                score=float(score),
                reason="low-similarity baseline anti-hit",
                literal_surface=record.literal_surface,
                adjacent_suggestions=[],
            )
        )

    # Derive valid_from / valid_to from contradicts edges,
    # then downweight stale, then re-sort hits. anti_hits left in their semantic
    # order: they're an inhibitory tail, not a ranked list. Downweighting them
    # lowers their weight in any downstream consumer that uses anti-hit score
    # (rank stage, schema-induction reader) — without this, a user who reverses
    # an opinion twice would see ghosts of the first reversal still actively
    # inhibiting current recall.
    derive_temporal_validity(store, hits)
    derive_temporal_validity(store, anti_hits)
    apply_stale_downweight(hits)
    apply_stale_downweight(anti_hits)
    hits.sort(key=lambda h: h.score, reverse=True)

    # On-read S4 viability check on the baseline recall
    # path too, so behaviour is consistent regardless of which recall route
    # core.py dispatches to.
    try:
        from iai_mcp.s4 import on_read_check
        s4_hints = on_read_check(store, hits, session_id=session_id)
    except (OSError, ValueError, RuntimeError) as exc:
        log.warning("s4 on_read_check failed: %s", exc)
        s4_hints = []

    response = RecallResponse(
        hits=hits,
        anti_hits=anti_hits,
        activation_trace=[h.record_id for h in hits],
        # ~4 chars per token heuristic.
        budget_used=sum(len(h.literal_surface) for h in hits) // 4,
        hints=s4_hints,
        # Surface mode on the baseline response too. The baseline does not
        # produce concept-mode patterns_observed (that's the full pipeline's
        # job — patterns_observed reflects displaced candidates the rank stage
        # would have surfaced; baseline has no rank stage). Default [] is
        # correct for both modes here.
        cue_mode=mode,
        patterns_observed=[],
    )

    # Emit kind='retrieval_used' so precision@5 can be computed live from
    # production emits. Diagnostic-only: never block the recall path on
    # emit failure.
    try:
        write_event(
            store,
            kind="retrieval_used",
            data={
                "hit_ids": [str(h.record_id) for h in hits],
                "query": cue_text,
                "used": len(hits) > 0,
                "budget_used": response.budget_used,
                "path": "baseline_recall",
            },
            severity="info",
            session_id=session_id,
            buffered=True,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        log.warning("retrieval_used event write failed: %s", exc)

    return response


def reinforce_edges(
    store: MemoryStore, ids: list[UUID], delta: float = 0.1
) -> EdgeUpdate:
    """Hebbian boost on all pairwise edges among co-retrieved ids.

    Pairwise = C(n, 2) combinations. Delta 0.1 is the default increment.
    """
    pairs: list[tuple[UUID, UUID]] = list(combinations(ids, 2))
    new_weights = store.boost_edges(pairs, delta=delta)
    # Canonical JSON-string keys (tuples are not JSON-serialisable).
    new_weights_str = {f"{a}|{b}": float(w) for (a, b), w in new_weights.items()}
    return EdgeUpdate(
        edges_boosted=len(pairs),
        pairs=pairs,
        new_weights=new_weights_str,
    )


def contradict(
    store: MemoryStore,
    original_id: UUID,
    new_fact: str,
    new_embedding: list[float],
) -> ReconsolidationReceipt:
    """Edge-based reconsolidation.

    Creates a new record with `new_fact` and adds a `contradicts` edge from
    original -> new. Does NOT rewrite the original record.
    """
    # Flush the record buffer before the point-read so a just-captured
    # original that is still in _record_buffer is visible to store.get().
    # Without this flush, contradicting a record that was captured in the
    # same session (before the 500-row threshold forced a drain) raises
    # "unknown record" because store.get() reads SQLite, not the buffer.
    # Contradicts are rare and load-bearing; the flush cost is negligible
    # and confined to the write path.
    flush_record_buffer(store)
    original = store.get(original_id)
    if original is None:
        raise ValueError(f"unknown record {original_id}")
    # Validate against the store's actual embedding dim, not a hardcoded
    # constant. Migrations and env overrides both rely on store.embed_dim
    # as source of truth.
    target_dim = store.embed_dim
    if len(new_embedding) != target_dim:
        raise ValueError(
            f"new_embedding must be {target_dim}d, got {len(new_embedding)}"
        )
    now = datetime.now(timezone.utc)
    new_rec = MemoryRecord(
        id=uuid4(),
        tier=original.tier,
        literal_surface=new_fact,
        aaak_index="",
        embedding=list(new_embedding),
        community_id=original.community_id,
        centrality=0.0,
        detail_level=original.detail_level,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=(original.detail_level >= 3),
        never_merge=False,
        provenance=[{"ts": now.isoformat(), "cue": "contradict", "session_id": "-"}],
        created_at=now,
        updated_at=now,
        tags=["contradict"],
        # Propagate the original record's language tag to the contradiction.
        # A contradiction is a linguistic amendment; it lives in the same
        # conversational register as the source.
        language=getattr(original, "language", "en") or "en",
    )
    # Language guard: must run on EVERY write path. A Cyrillic/CJK
    # `new_fact` without an explicit `raw:<lang>` tag would otherwise land in
    # literal_surface unguarded. Callers who intentionally store non-English
    # raw capture pre-tag the record via the MCP surface.
    enforce_english_raw(new_rec)
    new_rec.aaak_index = generate_aaak_index(new_rec)
    store.insert(new_rec)
    store.add_contradicts_edge(original_id, new_rec.id)
    invalidate_temporal_validity_cache(store)

    # Monotropic proactive check fires only in high-focus domains. Hints
    # aren't surfaced via contradict() (its signature is fixed to
    # ReconsolidationReceipt), but events land in the events table so the
    # user can inspect them via `iai-mcp contradictions`.
    try:
        from iai_mcp.s4 import monotropic_proactive_check
        # Deliberately empty profile_state: callers of contradict() don't pass
        # one; core.py can inject a fuller state via its own wrapper once the
        # profile is wired to pipeline_recall.
        monotropic_proactive_check(store, new_rec, {}, session_id="-")
    except (OSError, ValueError, RuntimeError) as exc:
        log.warning("monotropic_proactive_check failed: %s", exc)

    return ReconsolidationReceipt(
        original_id=original_id,
        new_record_id=new_rec.id,
        edge_type="contradicts",
        ts=now,
    )


_tv_cache: dict[int, tuple[dict[str, list[str]], dict[str, datetime]]] = {}
_tv_cache_dirty: dict[int, bool] = {}


def invalidate_temporal_validity_cache(store: "MemoryStore") -> None:
    _tv_cache_dirty[id(store)] = True


def build_temporal_validity_maps(
    store: MemoryStore,
) -> tuple[dict[str, list[str]], dict[str, datetime]] | None:
    """One-shot builder for the two lookup maps that
    derive_temporal_validity consumes.

    Returns:
        (outgoing, ts_by_id)
            outgoing  : src_id_str → [dst_id_str, ...] for edge_type='contradicts'
            ts_by_id  : record_id_str → created_at (datetime / pandas.Timestamp)

        Both empty if the store has no edges / records — callers should
        treat that as "nothing to derive". Returns None on a hard read
        failure so callers can short-circuit (recall hot path: never raise).

    Hoist this call to the entry-point that needs to enrich both hits AND
    anti_hits in one recall — passes the same two maps into both
    derive_temporal_validity calls so the records table is scanned once
    instead of twice.

    Why hoisting matters: each store.db.open_table('records').to_pandas()
    call costs ~28ms at N=100. Running it twice per recall (once for hits,
    once for anti_hits) doubled overhead to ~56ms; one shared scan halves that.

    Perf-tight: the records table has wide rows (384d/1024d embedding +
    encrypted literal_surface + 1250-byte structure_hv). A bare
    `to_pandas()` materializes ALL columns. We only need (id, created_at)
    here. Column-subset scans skip the heavy payload columns and measurably
    reduce per-recall overhead.

    Known perf gap: at N=300 the column-subset scan still adds ~45 ms
    to recall_for_response, exceeding the 75 ms / N=300 budget. The cheapest
    fix — plumb `created_at` into the graph node payload — is deferred.
    """
    store_id = id(store)
    if not _tv_cache_dirty.get(store_id, True) and store_id in _tv_cache:
        return _tv_cache[store_id]

    edges_tbl = store.db.open_table("edges")
    try:
        edges_count = int(edges_tbl.count_rows())
        if edges_count > 0:
            edges_df = (
                edges_tbl.search()
                .select(["src", "dst", "edge_type"])
                .limit(edges_count)
                .to_pandas()
            )
        else:
            edges_df = None
    except (OSError, ValueError, RuntimeError) as exc:
        log.warning("build_temporal_validity_maps edges read failed: %s", exc)
        return None

    outgoing: dict[str, list[str]] = {}
    if edges_df is not None and not edges_df.empty:
        try:
            ctr = edges_df[edges_df["edge_type"] == "contradicts"]
        except (KeyError, ValueError, RuntimeError) as exc:
            log.warning("build_temporal_validity_maps filter failed: %s", exc)
            return None
        if not ctr.empty:
            try:
                for src_s, dst_s in zip(
                    ctr["src"].tolist(), ctr["dst"].tolist(), strict=False
                ):
                    outgoing.setdefault(str(src_s), []).append(str(dst_s))
            except (KeyError, ValueError, RuntimeError) as exc:
                log.warning("build_temporal_validity_maps zip failed: %s", exc)
                return None

    # ts_by_id must be populated even on contradicts-free stores so valid_from
    # derives correctly (src_ts=None -> continue -> valid_from stays None).
    # The records-table scan below (now-unconditional) populates it.

    try:
        records_tbl = store.db.open_table("records")
        records_count = int(records_tbl.count_rows())
        if records_count > 0:
            records_df = (
                records_tbl.search()
                .select(["id", "created_at"])
                .limit(records_count)
                .to_pandas()
            )
            def _parse_ts(v: object) -> datetime:
                # SQLite stores created_at as ISO text; pandas returns str.
                # A pandas Timestamp is datetime-compatible via duck-typing.
                # Normalize to an aware datetime so comparisons and .isoformat() work.
                if isinstance(v, datetime):
                    return v if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)
                s = str(v)
                dt = datetime.fromisoformat(s)
                return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

            ts_by_id: dict[str, datetime] = {
                str(k): _parse_ts(v)
                for k, v in zip(
                    records_df["id"].tolist(),
                    records_df["created_at"].tolist(),
                    strict=False,
                )
            }
        else:
            ts_by_id = {}
    except (OSError, ValueError, RuntimeError) as exc:
        log.warning("build_temporal_validity_maps records read failed: %s", exc)
        return None
    _result_full: tuple[dict[str, list[str]], dict[str, datetime]] = (outgoing, ts_by_id)
    _tv_cache[store_id] = _result_full
    _tv_cache_dirty[store_id] = False
    return _result_full


def derive_temporal_validity(
    store: MemoryStore | None,
    hits: list[MemoryHit],
    records_cache: dict[UUID, MemoryRecord] | None = None,
    *,
    outgoing: dict[str, list[str]] | None = None,
    ts_by_id: dict[str, datetime] | None = None,
) -> list[MemoryHit]:
    """Derive valid_from / valid_to per hit from the contradicts-edge graph.

    For each hit:
        valid_from = record.created_at  (always set when record is loadable)
        valid_to   = oldest contradicting-record.created_at where
                     edge: src=record.id, dst=contradicting_id, edge_type='contradicts'
                     AND contradicting_id.created_at > record.created_at
                     (None if no such record exists — record still valid)

    MUTATES the hits in place (sets .valid_from and .valid_to on each
    MemoryHit) AND returns the same list for ergonomic chaining. Does NOT
    change .score — downweight is the caller's responsibility (see
    apply_stale_downweight below).

    Two call patterns:

      1. Pre-built maps (preferred in hot path):
         outgoing + ts_by_id from build_temporal_validity_maps(store).
         `store` may be None when both maps are pre-built — useful for
         tests and for the shared-scan pattern in recall_for_response.

      2. Lazy build (baseline path):
         pass `store` only; the helper calls build_temporal_validity_maps
         internally. Convenient for callers (baseline retrieve.recall())
         that enrich a single list per recall — one scan, one consumer.

    `records_cache` is RESERVED for future use when graph node attrs carry
    a real `created_at`. Today the helper bypasses the cache and reads
    (id, created_at) from the records table — see perf note in
    build_temporal_validity_maps above.

    PERF NOTE: per-hit `store.get(rid)` triggers the AES-GCM decrypt path
    on `literal_surface` (cost ~12ms per hit on M1). The to_pandas() scan
    returns RAW store rows (no decrypt until store._from_row), keeping the
    helper under the gate.
    """
    if not hits:
        return hits

    if outgoing is None or ts_by_id is None:
        if store is None:
            # No way to build the maps. Defensive: leave hits untouched.
            return hits
        built = build_temporal_validity_maps(store)
        if built is None:
            return hits
        outgoing, ts_by_id = built

    def _created_at(rid: UUID) -> datetime | None:
        return ts_by_id.get(str(rid))

    for hit in hits:
        src_ts = _created_at(hit.record_id)
        if src_ts is None:
            # Record not in the snapshot (raced delete / unknown id) —
            # leave valid_from / valid_to None.
            continue
        hit.valid_from = src_ts
        candidates = outgoing.get(str(hit.record_id), [])
        if not candidates:
            continue
        oldest_newer: datetime | None = None
        for dst_str in candidates:
            try:
                dst_id = UUID(dst_str)
            except (TypeError, ValueError):
                continue
            dst_ts = _created_at(dst_id)
            if dst_ts is None:
                continue
            # Strict ">": defensive against malformed older-pointing edges
            # (test 4). The dst must be NEWER than src to count as
            # "contradicted by a newer record".
            if dst_ts <= src_ts:
                continue
            if oldest_newer is None or dst_ts < oldest_newer:
                oldest_newer = dst_ts
        if oldest_newer is not None:
            hit.valid_to = oldest_newer
    return hits


def apply_stale_downweight(
    hits: list[MemoryHit],
    now: datetime | None = None,
    *,
    cue_intent: str | None = None,
) -> list[MemoryHit]:
    """Multiply MemoryHit.score by STALE_DOWNWEIGHT_FACTOR
    for hits whose derived valid_to < now. Append " · stale" to .reason
    for visibility.

    MUTATES hits in place. Returns the same list (NOT re-sorted — caller
    decides ranking semantics; anti_hits typically stay in their semantic
    order, ranked hits are re-sorted by the caller).

    Idempotent on both the reason-suffix append and the score multiplication:
    a second call on already-downweighted hits is a no-op. The score guard
    uses a private `_stale_downweighted` sentinel attribute that never
    crosses onto the JSON wire (core._hit_to_json emits only the public
    response fields plus valid_from/valid_to).

    `now` is parameterizable for deterministic tests; defaults to
    datetime.now(timezone.utc).

    When ``cue_intent == "historical_verbatim"`` the downweight is skipped
    entirely: the cue explicitly asks for the superseded (stale) wording, so
    staleness is the TARGET property, not a penalty. Halving stale records
    on such a cue is anti-aligned with the request and would re-bury the very
    record the recall path just surfaced by association with its corrector.
    """
    if cue_intent == "historical_verbatim":
        return hits
    now_value = now or datetime.now(timezone.utc)
    for hit in hits:
        if hit.valid_to is None or hit.valid_to >= now_value:
            continue
        if not getattr(hit, "_stale_downweighted", False):
            hit.score *= STALE_DOWNWEIGHT_FACTOR
            hit._stale_downweighted = True
        if not hit.reason.endswith(_STALE_REASON_SUFFIX):
            hit.reason = f"{hit.reason}{_STALE_REASON_SUFFIX}"
    return hits


def link_temporal_next(
    store: MemoryStore,
    new_record: MemoryRecord,
    session_id: str,
) -> UUID | None:
    """Create temporal_next edge + record_inserted event.

    Reads the most recent `record_inserted` event (any record) from the events
    table. If that event happened within TEMPORAL_NEXT_WINDOW AND in the same
    session, create a `temporal_next` edge from the previous record to the new
    record.

    Then write a fresh `record_inserted` event marking this insertion.

    Returns the previous record UUID (the edge source) or None if no edge was
    created (either no prior insert or stale / cross-session).
    """
    now = datetime.now(timezone.utc)
    # Look at the last ~20 record_inserted events to find the most recent match.
    prior_events = query_events(
        store, kind="record_inserted",
        since=now - TEMPORAL_NEXT_WINDOW, limit=20,
    )
    previous_id: UUID | None = None
    for ev in prior_events:
        if ev.get("session_id") != session_id:
            continue
        raw = ev["data"].get("record_id")
        if not raw:
            continue
        try:
            candidate = UUID(raw)
        except (TypeError, ValueError):
            continue
        if candidate == new_record.id:
            continue
        previous_id = candidate
        break  # events are newest-first

    if previous_id is not None:
        try:
            store.boost_edges(
                [(previous_id, new_record.id)],
                edge_type="temporal_next",
                delta=1.0,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            # Diagnostic only; don't block the write path on edge failure.
            log.warning("temporal_next edge creation failed: %s", exc)

    write_event(
        store,
        kind="record_inserted",
        data={
            "record_id": str(new_record.id),
            "tier": new_record.tier,
        },
        severity="info",
        session_id=session_id,
        source_ids=[new_record.id],
    )
    return previous_id


def _make_graph_sync_hook(graph):
    """Factory for the store -> graph mutation callback.

    Returned callable dispatches on ``op`` (insert|update|delete) and
    mutates ``graph`` (a ``MemoryGraph``) in-place via the public API:
    ``add_node`` + ``set_node_payload`` for insert / update, and direct
    node removal for delete. On unknown op or any payload
    shape error, the hook is a quiet no-op — the store's try/except
    surface turns exceptions into stderr events anyway, but we stay
    defensive here so hook-level bugs never reach the store.

    Dirty-counter increment: the RECORD-only hook also increments the
    Layer-2 overlay freshness-fuse dirty counter on every record
    insert/update/delete.  This is RECORD-only: recall-path boost_edges
    writes are invisible to the store's graph-sync hook and therefore do
    NOT increment the counter (boost_edges is an edge write, not a record
    write).

    Folding the increment INTO this factory (rather than wrapping at the
    register site) ensures every rebuild that calls
    store.register_graph_sync_hook(_make_graph_sync_hook(graph))
    automatically re-includes the increment, surviving daemon restarts
    and nightly rebuild re-registrations.
    """
    def _hook(op: str, record) -> None:
        nid = record.id
        nid_str = str(nid)
        if op in ("insert", "update"):
            payload = {
                "embedding": list(record.embedding),
                "surface": record.literal_surface,
                "centrality": float(record.centrality),
                "tier": record.tier,
                "pinned": bool(record.pinned),
                "tags": list(getattr(record, "tags", []) or []),
                "language": str(getattr(record, "language", "en") or "en"),
            }
            if nid_str not in graph._node_payload:
                graph.add_node(
                    nid,
                    community_id=None,
                    embedding=payload["embedding"],
                )
            graph.set_node_payload(nid, payload)
            # Increment the in-process record-mutation dirty counter.
            # Runs only on RECORD writes (insert/update) — NOT on recall-path
            # boost_edges (that is an edge write, not a record write).
            try:
                from iai_mcp import runtime_graph_cache as _rgc
                _rgc.increment_dirty_counter()
            except Exception:  # noqa: BLE001 -- never break a record write
                pass
        elif op == "delete":
            graph.remove_node(nid)
            # Deletions also count as record mutations for the dirty counter.
            try:
                from iai_mcp import runtime_graph_cache as _rgc
                _rgc.increment_dirty_counter()
            except Exception:  # noqa: BLE001 -- never break a record delete
                pass
        # Unknown op: silently ignore. The store writes are authoritative;
        # unknown ops will be picked up on the next full rebuild.
    return _hook


def build_runtime_graph(store: MemoryStore):
    """Reconstruct MemoryGraph + CommunityAssignment + rich-club from the store.

    Called by core.py's `memory_recall` dispatch when the store is non-empty.
    The expensive pieces — Leiden community detection + rich-club selection —
    are cached to disk in ``runtime_graph_cache.json`` keyed on the store's
    (records_count, edges_count, schema_version, embed_dim) tuple. Cache hit
    skips ~230 ms of Leiden + rich-club work. MemoryGraph itself is rebuilt
    on every call from the store rows because caching it would require a
    non-JSON format for the graph object.

    Every graph node carries the record's payload (embedding, surface,
    centrality, tier, pinned) as graph node attributes.
    ``pipeline._read_record_payload`` reads from these attributes at seed +
    spread stages, eliminating per-id ``store.get`` round-trips
    (737 ms -> target ~20-30 ms at N=1k). A ``_graph_sync_hook`` is registered
    on the store so insert/update/delete mirror their mutations to the in-RAM
    graph; hook failures are logged, never raised (write-path authoritative).
    On cache HIT the node_payload blob rehydrates the graph attributes
    directly; MISS rebuilds them from the fresh store.all_records() walk.

    Returns (graph, assignment, rich_club).

    Local imports keep the heavy graph/community modules out of the
    hot path (core.py module-load time stays small).
    """
    from iai_mcp.community import CommunityAssignment, detect_communities
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.richclub import rich_club_nodes
    from iai_mcp import runtime_graph_cache

    graph = MemoryGraph()

    # Try the on-disk cache before running Leiden + rich-club.
    # Cache-first so we can consult the v2 node_payload blob for free.
    cached = runtime_graph_cache.try_load(store)
    assignment = None
    rich_club = None
    cached_node_payload: dict[str, dict] | None = None
    # Cached max_degree rehydrates without re-walking the graph.
    # Used as a defensive fallback if the live degree walk below fails
    # for any reason.
    cached_max_degree: int = 0
    if cached is not None:
        assignment, rich_club, cached_node_payload, cached_max_degree = cached

    # Build nodes. If the cache gave us a node_payload blob AND the store
    # record count matches, reuse it — skips the encrypted store scan.
    # Otherwise fall through to the full row walk so node attrs stay
    # strictly derived from the authoritative store.
    records_tbl = store.db.open_table("records")
    # Use the non-pending count so pending rows do not force perpetual
    # cache miss / rebuild churn.  The MISS-path walk skips pending rows, so the
    # node-payload count IS the non-pending count; the gate must agree.
    records_count = store.active_records_count()
    use_cached_payload = (
        cached_node_payload is not None
        and len(cached_node_payload) == records_count
    )

    if use_cached_payload:
        # Fast path: graph nodes + attributes come from the cache JSON.
        for nid, payload in cached_node_payload.items():
            # MemoryGraph.add_node has a fixed signature; use it for
            # topology, then write the full payload via the public
            # sidecar setter (graph._node_payload, not the private _nx).
            graph.add_node(
                UUID(nid),
                community_id=None,
                embedding=list(payload.get("embedding") or []),
            )
            graph.set_node_payload(nid, {
                "embedding": list(payload.get("embedding") or []),
                "surface": payload.get("surface", ""),
                "centrality": float(payload.get("centrality") or 0.0),
                "tier": payload.get("tier", "episodic"),
                "pinned": bool(payload.get("pinned", False)),
                "tags": list(payload.get("tags") or []),
                "language": str(payload.get("language", "en") or "en"),
            })
        node_payload_for_cache = cached_node_payload
    else:
        # MISS path: walk the records table, attach payload at
        # graph.add_node time, and remember the payload so we can
        # persist it into the cache below.
        df = records_tbl.to_pandas()
        node_payload_for_cache = {}
        decrypt_fail_events = 0
        decrypt_fail_unique: set[str] = set()
        for _, row in df.iterrows():
            # Exclude pending rows from the warm semantic candidate construction
            # (load-bearing exclusion).  A pending row carries a zero-vector
            # which would be a degenerate cosine neighbor.
            # all_records() / recent_user_turns() stays pending-INCLUSIVE;
            # the exclusion lives HERE and in query_similar only.
            if int(row.get("embedding_pending") or 0) != 0:
                continue
            rid = UUID(row["id"])
            _comm_raw = row["community_id"]
            # Guard against pandas NaN (truthy but not a string).
            if _comm_raw is not None and not isinstance(_comm_raw, str):
                try:
                    import math as _math
                    if _math.isnan(float(_comm_raw)):
                        _comm_raw = None
                except (TypeError, ValueError):
                    _comm_raw = None
            community_id = UUID(_comm_raw) if _comm_raw else None
            embedding = (
                list(row["embedding"])
                if row["embedding"] is not None
                else [0.0] * EMBED_DIM
            )
            # literal_surface is AES-GCM encrypted at rest. Decrypt here
            # via the store's helper so the graph payload carries plaintext
            # the pipeline can use directly.
            literal_raw = row.get("literal_surface") or ""
            try:
                from iai_mcp.crypto import is_encrypted
                if is_encrypted(literal_raw):
                    literal_raw = store._decrypt_for_record(rid, literal_raw)
            except Exception:  # noqa: BLE001 -- InvalidTag / OSError / ValueError / RuntimeError
                # A decrypt failure here used to assign ``literal_raw = ""``
                # and then fall through
                # to update the live graph node + persist to
                # ``node_payload_for_cache``. That empty-surface payload
                # then poisoned the on-disk runtime_graph_cache, and on
                # warm-restart pipeline._read_record_payload happily
                # returned ``literal_surface=""`` claiming success —
                # silent corruption of verbatim recall.
                #
                # Skip-the-node approach (chosen over the _decrypt_failed
                # sentinel-flag because it produces the smallest disk
                # footprint and the simplest invariant: "the cache
                # contains only records whose surface successfully
                # decrypted"). The pipeline read path falls back to
                # store.get(rid) which has its own retry semantics in
                # crypto.py.
                #
                # Tail-end mandate: per-record ``graph_build_decrypt_failed``
                # warnings are rate-limited (default 300s) so wrong-key floods
                # do not spam launchd stderr; a per-build summary still fires.
                rid_s = str(rid)
                decrypt_fail_events += 1
                decrypt_fail_unique.add(rid_s)
                now_m = time.monotonic()
                last_m = _GRAPH_DECRYPT_WARN_LAST.get(rid_s, 0.0)
                if now_m - last_m >= _GRAPH_DECRYPT_WARN_INTERVAL_SEC:
                    _GRAPH_DECRYPT_WARN_LAST[rid_s] = now_m
                    log.warning(
                        "graph_build_decrypt_failed",
                        extra={"record_id": rid_s},
                    )
                continue

            tier = row.get("tier") or "episodic"
            centrality = float(row.get("centrality") or 0.0)
            pinned = bool(row.get("pinned") or False)
            # Tags travel on graph nodes so the rank stage's SimpleRecordView
            # carries tags for profile_modulation_for_record without needing
            # a store.get fallback in the hot path.
            tags_raw = row.get("tags_json") or "[]"
            try:
                import json as _json
                tags_list = _json.loads(tags_raw) if isinstance(tags_raw, str) else list(tags_raw)
                if not isinstance(tags_list, list):
                    tags_list = []
            except (ValueError, TypeError):
                tags_list = []
            language = str(row.get("language") or "en")

            graph.add_node(
                rid,
                community_id=community_id,
                embedding=embedding,
            )
            # Write record payload via the public sidecar setter; the legacy
            # _nx.nodes[*].update(...) write pattern was retired when the
            # graph backend moved off the in-memory adjacency store.
            graph.set_node_payload(rid, {
                "embedding": list(embedding),
                "surface": str(literal_raw),
                "centrality": centrality,
                "tier": str(tier),
                "pinned": pinned,
                "tags": list(tags_list),
                "language": language,
            })
            node_payload_for_cache[str(rid)] = {
                "embedding": list(embedding),
                "surface": str(literal_raw),
                "centrality": centrality,
                "tier": str(tier),
                "pinned": pinned,
                "tags": list(tags_list),
                "language": language,
            }

        if decrypt_fail_events > 0:
            log.warning(
                "graph_build_decrypt_failed_summary",
                extra={
                    "unique_records": len(decrypt_fail_unique),
                    "total_skip_events": decrypt_fail_events,
                },
            )

    edges_df = store.db.open_table("edges").to_pandas()
    for _, row in edges_df.iterrows():
        graph.add_edge(
            UUID(row["src"]),
            UUID(row["dst"]),
            weight=float(row["weight"]),
            edge_type=row["edge_type"],
        )

    # Cache the maximum graph degree so the rank stage can normalise
    # log(1+deg) into [0,1] (sample-rank-comparable to
    # cosine; W_DEGREE * deg_norm bounded by W_DEGREE itself instead of
    # by an unbounded log term that scales with hub connectivity).
    # Computed once per build; rehydrated from disk on warm starts via
    # the runtime_graph_cache.json payload. Defensive: fall back to the
    # cached value if the live degree() walk fails for any reason — and
    # never let a bare AttributeError reach the rank stage.
    try:
        deg_values = [d for _, d in graph.degrees()]
        max_degree = max(deg_values) if deg_values else 0
    except (ValueError, RuntimeError, AttributeError):
        max_degree = cached_max_degree
    if max_degree == 0 and cached_max_degree > 0:
        # Live walk produced 0 (no edges yet) but the cache held a real
        # value — prefer the cached value. Triggers when an upstream
        # path stripped edges before the rebuild reached us.
        max_degree = cached_max_degree
    graph._max_degree = int(max_degree)

    # Run (or reuse cached) Leiden + rich-club.
    if assignment is None:
        # Cache MISS path: no prior available in this scope (cache HIT
        # short-circuits above without touching detect_communities at
        # all). Pass `prior_mode="seeded"` even when `prior is None`.
        assignment = detect_communities(graph, prior=None, prior_mode="seeded")
        rich_club = rich_club_nodes(graph, percent=0.10)

    # Compute betweenness centrality once per build and attach to every
    # node as a graph attribute so the rank stage can read it O(1)
    # instead of calling graph.centrality() on every recall. Cache HIT
    # path already rehydrated centrality from node_payload into node
    # attrs above; we only (re)compute when the cache payload is absent
    # / stale or when node_payload centrality values are all-zero
    # placeholders.
    needs_centrality = True
    if use_cached_payload and cached_node_payload is not None:
        # If the cache carries real per-node centrality floats (possibly
        # non-zero) it is the current shape. If every value is exactly
        # 0.0 the cache predates the centrality field — recompute to
        # populate the live graph, then a subsequent save() below will
        # upgrade the cache.
        any_nonzero = any(
            float(p.get("centrality") or 0.0) != 0.0
            for p in cached_node_payload.values()
        )
        needs_centrality = not any_nonzero
    if needs_centrality:
        try:
            centrality_map = graph.centrality()
            for rid, cval in centrality_map.items():
                nid_str = str(rid)
                # Sidecar-keyed lookup: membership in graph._node_payload
                # implies the node exists in the topology (add_node populates
                # both atomically). Avoids reaching into private _nx state.
                if nid_str in graph._node_payload:
                    graph.set_node_centrality(rid, float(cval))
                    if (
                        node_payload_for_cache is not None
                        and nid_str in node_payload_for_cache
                    ):
                        node_payload_for_cache[nid_str]["centrality"] = (
                            float(cval)
                        )
        except (OSError, ValueError, RuntimeError) as exc:
            # Defensive: centrality is a ranking signal, not a
            # correctness invariant; fall back to zeros on failure.
            log.warning("centrality computation failed: %s", exc)
            for nid in graph.iter_nodes():
                # setdefault-equivalent on the sidecar: only seed 0.0 when
                # the centrality key is absent. Avoids stomping a real
                # value that an earlier path may have written.
                key = str(nid)
                if "centrality" not in graph._node_payload.get(key, {}):
                    graph.set_node_centrality(nid, 0.0)

    # Persist — fresh build, or the cache was a legacy node-payload shape.
    if cached_node_payload is None or needs_centrality:
        runtime_graph_cache.save(
            store, assignment, rich_club,
            node_payload=node_payload_for_cache,
            # max_degree travels with assignment + rich_club so warm-start
            # build_runtime_graph rehydrates without recompute.
            max_degree=int(getattr(graph, "_max_degree", 0) or 0),
        )

    # Register the graph-sync hook so future insert/update/delete calls
    # mutate the live graph instead of diverging. The store swallows
    # hook exceptions so a buggy hook never breaks a write.
    try:
        store.register_graph_sync_hook(_make_graph_sync_hook(graph))
    except (AttributeError, TypeError, RuntimeError) as exc:
        # Older store without register_graph_sync_hook — this is a
        # defensive upgrade path; the graph just won't stay live-sync'd.
        log.warning("graph_sync_hook registration failed: %s", exc)

    # Belt-and-braces: every code path above sets graph._max_degree,
    # but if some future refactor short-circuits before reaching the
    # live degree walk we still want the rank stage's
    # `getattr(graph, "_max_degree", 0)` to read a real int.
    if not hasattr(graph, "_max_degree"):
        graph._max_degree = 0

    return graph, assignment, rich_club
