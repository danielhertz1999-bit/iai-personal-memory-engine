"""Retrieval + reinforcement + contradiction paths.

- `recall`: baseline cosine top-k -- kept as a fallback for the
  empty-store case and for regression tests.
- `build_runtime_graph`: reconstruct a MemoryGraph + CommunityAssignment +
  rich-club from LanceDB state; consumed by core.py to drive `pipeline_recall`.
- `reinforce_edges`, `contradict`: unchanged from initial implementation.
- `link_temporal_next`: records a `record_inserted` event
  and creates a `temporal_next` edge from the previous same-session insertion
  to the new record if that event happened within the last 5 minutes.

Constitutional rules enforced here:
- every recall appends a provenance entry to every returned record.
- reinforce boosts pairwise Hebbian edges among co-retrieved ids.
- edge-based: contradict creates a linked record, preserves original.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from itertools import combinations
from uuid import UUID, uuid4

from iai_mcp.aaak import enforce_english_raw, generate_aaak_index
from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore
from iai_mcp.types import (
    EMBED_DIM,
    EdgeUpdate,
    MemoryHit,
    MemoryRecord,
    RecallResponse,
    ReconsolidationReceipt,
)


# / structured-log handle for the graph-build
# decrypt-failure path. Same one-liner the rest of the project uses
# (cf. capture.py:54, pipeline.py:33-imports). Used by the
# `graph_build_decrypt_failed` event when AES-GCM decrypt of a
# record's literal_surface raises during build_runtime_graph.
log = logging.getLogger(__name__)

# Per-process rate limit for graph_build_decrypt_failed (rid -> monotonic ts).
_GRAPH_DECRYPT_WARN_LAST: dict[str, float] = {}
_GRAPH_DECRYPT_WARN_INTERVAL_SEC = 300.0

# Downweight factor applied to MemoryHit.score when the derived valid_to < now
# (record was superseded by a newer contradicting record). 0.5 chosen as:
# aggressive enough that a fresh lower-cosine record can outrank a stale
# high-cosine hit on typical score distributions, while leaving the stale
# hit visible in the response for audit. Downweight, not hide, to preserve
# audit trail.
STALE_DOWNWEIGHT_FACTOR: float = 0.5

# Suffix appended to MemoryHit.reason when a record is downweighted as stale.
# Spaces around · are intentional — the reason field already carries
# "cosine X.XXX + …" segments separated by " + "; " · stale" stays readable.
_STALE_REASON_SUFFIX: str = " · stale"


# temporal_next window. Records inserted within this window
# in the same session are linked with a temporal_next edge.
TEMPORAL_NEXT_WINDOW = timedelta(minutes=5)


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
    """baseline retrieval.

    Fetches top (k_hits + k_anti) by cosine similarity; treats the top k_hits as
    excitatory hits and the bottom k_anti as a naive anti-hit stub. will
    replace anti-hits with real contradicts-edge + AAAK-opposition logic.

    Every returned hit gets a provenance entry appended.

    R7: `mode` kwarg defaults to 'verbatim'. The baseline
    is the conservative fallback path (used by core.dispatch when the runtime
    graph is unavailable / build fails / store is empty). Defaulting to
    verbatim protects the North-Star ≥99% essential variable on the degraded
    path — the user never silently lands on a schema-dominated surface even
    when the full pipeline is unreachable. Verbatim mode applies the same
    tier filter + schema exclusion as pipeline_recall verbatim mode so the
    contract on hits[] is identical regardless of which route core dispatched
    to. Concept mode preserves today's pure-cosine baseline (no filter).
    """
    raw = store.query_similar(cue_embedding, k=k_hits + k_anti)

    # R7: verbatim mode candidate filter on the baseline path.
    # tier='episodic' AND no pattern:* tag — same exclusion contract as
    # pipeline_recall verbatim mode (R5). Also excludes
    # tier='semantic_pruned' soft-deleted schemas naturally.
    if mode == "verbatim":
        raw = [
            (rec, score) for rec, score in raw
            if rec.tier == "episodic"
            and not any(t.startswith("pattern:") for t in (rec.tags or []))
        ]

    hits: list[MemoryHit] = []
    # (D5-01 effect c fix): collect provenance entries during the
    # hit-building loop, flush via ONE store.append_provenance_batch call
    # after the loop closes. Replaces the per-hit
    # `store.append_provenance(record.id, entry)` pattern that produced the
    # 64x wall-clock blow-up and rank perturbation under memory pressure
    # (pressplay 8 GB M1, 2026-04-19). Mirrors the L-02 fix already in
    # src/iai_mcp/pipeline.py::pipeline_recall (see SC-6).
    provenance_pending: list[tuple[UUID, dict]] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for record, score in raw[:k_hits]:
        hits.append(
            MemoryHit(
                record_id=record.id,
                score=float(score),
                reason=f"cosine {score:.3f}",
                literal_surface=record.literal_surface,
                adjacent_suggestions=[], # fills per AUTIST-07
            )
        )
        # every recall appends a provenance entry; write is batched
        # end-of-loop to preserve rank stability (effect c fix).
        provenance_pending.append((
            record.id,
            {
                "ts": now_iso,
                "cue": cue_text,
                "session_id": session_id,
            },
        ))

    # flush: single merge_insert transaction replaces N read-modify-writes.
    # Diagnostic-only: never block the user's recall on a provenance-write failure
    # (Rule 1 -- matches pipeline_recall's defensive contract).
    if provenance_pending:
        try:
            store.append_provenance_batch(provenance_pending)
        except Exception:
            pass

    anti_hits: list[MemoryHit] = []
    # Naive anti-hit stub: bottom-k of the same query. replaces with
    # real contradicts-edge + AAAK-opposition scoring.
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

    # Derive valid_from / valid_to from contradicts edges, then downweight
    # stale, then re-sort hits. anti_hits left in their semantic order:
    # they're an inhibitory tail, not a ranked list. Downweighting them
    # lowers their weight in any downstream consumer that uses anti-hit
    # score (rank stage, schema-induction reader) — without this, a user
    # who reverses an opinion twice would see ghosts of the first
    # reversal still actively inhibiting current recall.
    derive_temporal_validity(store, hits)
    derive_temporal_validity(store, anti_hits)
    apply_stale_downweight(hits)
    apply_stale_downweight(anti_hits)
    hits.sort(key=lambda h: h.score, reverse=True)

    # on-read S4 viability check on the baseline recall
    # path too, so behaviour is consistent regardless of which recall route
    # core.py dispatches to.
    try:
        from iai_mcp.s4 import on_read_check
        s4_hints = on_read_check(store, hits, session_id=session_id)
    except Exception:
        s4_hints = []

    response = RecallResponse(
        hits=hits,
        anti_hits=anti_hits,
        activation_trace=[h.record_id for h in hits],
        # ~4 chars per token heuristic; benchmark will use Anthropic count_tokens.
        budget_used=sum(len(h.literal_surface) for h in hits) // 4,
        hints=s4_hints,
        # surface mode on the baseline response too. The
        # baseline does not produce concept-mode patterns_observed (that's
        # the full pipeline's job — patterns_observed reflects displaced
        # candidates the rank stage would have surfaced; baseline has no
        # rank stage). Default [] is correct for both modes here.
        cue_mode=mode,
        patterns_observed=[],
    )

    # (M2 LIVE prerequisite): emit kind='retrieval_used' so M2
    # precision@5 can be computed live from production emits, not seeded
    # events. Diagnostic-only: never block the recall path on emit failure.
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
        )
    except Exception:
        pass

    return response


def reinforce_edges(
    store: MemoryStore, ids: list[UUID], delta: float = 0.1
) -> EdgeUpdate:
    """Hebbian boost on all pairwise edges among co-retrieved ids.

    Pairwise = C(n, 2) combinations. Delta 0.1 is the Phase-1 simple-increment
    default.
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
    """ edge-based reconsolidation.

    Creates a new record with `new_fact` and adds a `contradicts` edge from
    original -> new. Does NOT rewrite the original record -- full amend-in-place
    is deferred to a future version.
    """
    original = store.get(original_id)
    if original is None:
        raise ValueError(f"unknown record {original_id}")
    # validate against the store's actual embedding dim,
    # not the legacy hardcoded EMBED_DIM. Migrations and env overrides both
    # rely on store.embed_dim as source of truth.
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
        # propagate the original record's language tag to the contradiction.
        # A contradiction is a linguistic amendment; it lives in the same
        # conversational register as the source.
        language=getattr(original, "language", "en") or "en",
    )
    # H-02: constitutional guard must run on EVERY write path, not just the
    # L0 seed. A Cyrillic/CJK `new_fact` without an explicit `raw:<lang>` tag
    # would otherwise land in literal_surface unguarded. Callers who intentionally
    # store non-English raw capture pre-tag the record via the MCP surface.
    #
    # note: once Task 2 ships enforce_language_tagged, call sites in
    # core.py + retrieve should migrate. For Phase-1 back-compat we keep
    # enforce_english_raw here so the H-02 Cyrillic-rejection test keeps passing.
    enforce_english_raw(new_rec)
    new_rec.aaak_index = generate_aaak_index(new_rec)
    store.insert(new_rec)
    store.add_contradicts_edge(original_id, new_rec.id)

    # monotropic proactive check fires only in high-focus
    # domains. Hints aren't surfaced via contradict() (its signature is fixed
    # to ReconsolidationReceipt), but events land in the events table so the
    # user can inspect them via `iai-mcp contradictions` in .
    try:
        from iai_mcp.s4 import monotropic_proactive_check
        # Deliberately empty profile_state: callers of contradict() don't pass
        # one; core.py can inject a fuller state via its own wrapper once the
        # profile is wired to pipeline_recall.
        monotropic_proactive_check(store, new_rec, {}, session_id="-")
    except Exception:
        pass  # Rule 1: never block writes on S4 diagnostic path.

    return ReconsolidationReceipt(
        original_id=original_id,
        new_record_id=new_rec.id,
        edge_type="contradicts",
        ts=now,
    )


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
    instead of twice (p95 perf gate).

    Why hoisting matters: each store.db.open_table('records').to_pandas()
    call costs ~28ms at N=100 reference workload. Running it twice per
    recall (once for hits, once for anti_hits) doubled overhead to ~56ms
    and pushed p95 above the 100ms gate; one shared scan halves that.

    Perf-tight: the records table has wide rows (384d/1024d embedding +
    encrypted literal_surface + 1250-byte structure_hv). A bare
    `to_pandas()` materializes ALL columns. We only need (id, created_at)
    here — those two are the only fields consumed by derive_temporal_validity.
    The edges table similarly has weight/updated_at we don't need; we
    only consume (src, dst, edge_type). Column-subset scans (LanceDB
    `tbl.search().select([cols])`) skip the heavy payload columns and
    measurably reduce per-recall overhead vs the bare to_pandas() path.

    Known perf gap: at N=300 the column-subset scan still adds ~45 ms
    to recall_for_response, which exceeds the 75 ms / N=300 budget. The
    cheapest architectural fix — plumb `created_at` into the graph node
    payload so derive_temporal_validity reads from cache instead of
    scanning the records table — is deferred. See SUMMARY follow-up.
    """
    edges_tbl = store.db.open_table("edges")
    try:
        # Column-subset scan — skip weight + updated_at. limit() must be
        # big enough to capture every edge; LanceDB's default search
        # limit is 10, which would silently truncate large stores.
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
    except Exception:
        return None

    outgoing: dict[str, list[str]] = {}
    if edges_df is not None and not edges_df.empty:
        try:
            ctr = edges_df[edges_df["edge_type"] == "contradicts"]
        except Exception:
            return None
        if not ctr.empty:
            try:
                for src_s, dst_s in zip(
                    ctr["src"].tolist(), ctr["dst"].tolist(), strict=False
                ):
                    outgoing.setdefault(str(src_s), []).append(str(dst_s))
            except Exception:
                return None

    try:
        records_tbl = store.db.open_table("records")
        records_count = int(records_tbl.count_rows())
        if records_count > 0:
            # Column-subset scan — skip embedding (384/1024 floats),
            # literal_surface (encrypted AES-GCM blob), structure_hv
            # (1250 bytes), aaak_index, provenance, etc. id + created_at
            # is all derive_temporal_validity ever reads.
            records_df = (
                records_tbl.search()
                .select(["id", "created_at"])
                .limit(records_count)
                .to_pandas()
            )
            ts_by_id: dict[str, datetime] = dict(
                zip(
                    records_df["id"].tolist(),
                    records_df["created_at"].tolist(),
                    strict=False,
                )
            )
        else:
            ts_by_id = {}
    except Exception:
        return None
    return outgoing, ts_by_id


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
    a real `created_at` (graph-payload surface upgrade — currently
    SimpleRecordView.created_at is a wall-clock placeholder, see
    pipeline.SimpleRecordView). Today the helper bypasses the cache and
    reads (id, created_at) from the records table — see perf note in
    build_temporal_validity_maps above.

    PERF NOTE: per-hit `store.get(rid)` triggers the AES-GCM decrypt path
    on `literal_surface` (cost ~12ms per hit on M1). At N=100 reference
    workload, that broke the p95 perf gate (1296ms vs 100ms target).
    The to_pandas() scan returns RAW Lance rows (no decrypt until
    store._from_row), keeping the helper under the gate.

    Note: pipeline.py's stage 12 (_find_anti_hits) also reads the edges
    table; further hoisting to share with the post-rank pipeline is a
    follow-up optimization.
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
            # Strict ">": defensive against malformed older-pointing edges.
            # The dst must be NEWER than src to count as "contradicted by
            # a newer record".
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
) -> list[MemoryHit]:
    """Multiply MemoryHit.score by STALE_DOWNWEIGHT_FACTOR for hits whose
    derived valid_to < now. Append " · stale" to .reason for visibility.

    MUTATES hits in place. Returns the same list (NOT re-sorted — caller
    decides ranking semantics; anti_hits typically stay in their semantic
    order, ranked hits are re-sorted by the caller).

    Idempotent on both the reason-suffix append and the score multiplication:
    a second call on already-downweighted hits is a no-op. The score guard
    uses a private `_stale_downweighted` sentinel attribute that never
    crosses onto the JSON wire (core._hit_to_json emits only the public
    hit fields plus valid_from/valid_to).

    `now` is parameterizable for deterministic tests; defaults to
    datetime.now(timezone.utc).
    """
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
    """create temporal_next edge + record_inserted event.

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
        except Exception:
            # Diagnostic only; don't block the write path on edge failure.
            pass

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


def _make_graph_sync_hook(G):
    """factory for the store -> graph mutation callback.

    Returned callable dispatches on ``op`` (insert|update|delete) and
    mutates ``G`` (a NetworkX Graph) in-place. On unknown op or any
    payload shape error, the hook is a quiet no-op — the store's
    try/except surface turns exceptions into stderr events anyway, but
    we stay defensive here so hook-level bugs never reach the store.
    """
    def _hook(op: str, record) -> None:
        nid = str(record.id)
        if op == "insert":
            payload = {
                "embedding": list(record.embedding),
                "surface": record.literal_surface,
                "centrality": float(record.centrality),
                "tier": record.tier,
                "pinned": bool(record.pinned),
                "tags": list(getattr(record, "tags", []) or []),
                "language": str(getattr(record, "language", "en") or "en"),
            }
            G.add_node(nid, **payload)
        elif op == "update":
            payload = {
                "embedding": list(record.embedding),
                "surface": record.literal_surface,
                "centrality": float(record.centrality),
                "tier": record.tier,
                "pinned": bool(record.pinned),
                "tags": list(getattr(record, "tags", []) or []),
                "language": str(getattr(record, "language", "en") or "en"),
            }
            if nid in G.nodes:
                G.nodes[nid].update(payload)
            else:
                G.add_node(nid, **payload)
        elif op == "delete":
            if nid in G.nodes:
                G.remove_node(nid)
        # Unknown op: silently ignore. The store writes are authoritative;
        # unknown ops will be picked up on the next full rebuild.
    return _hook


def build_runtime_graph(store: MemoryStore):
    """Reconstruct MemoryGraph + CommunityAssignment + rich-club from LanceDB.

    Called by core.py's `memory_recall` dispatch when the store is non-empty.
    (P4.A): the expensive pieces -- Leiden community
    detection + rich-club selection -- are cached to disk in
    ``runtime_graph_cache.json`` keyed on the store's (records_count,
    edges_count, schema_version, embed_dim) tuple. Cache hit skips
    ~230 ms of Leiden + rich-club work. MemoryGraph itself is rebuilt
    on every call from the LanceDB rows because caching it would
    require a non-JSON format for the NetworkX object.

    (hot-path switch): every graph node carries the record's
    payload (embedding, surface, centrality, tier, pinned) as NetworkX
    node attributes. ``pipeline._read_record_payload`` reads from these
    attributes at seed + spread stages, eliminating the per-id
    ``store.get`` LanceDB round-trips that dominated at N=1k
    (737 ms -> target ~20-30 ms). A ``_graph_sync_hook`` is registered
    on the store so insert/update/delete mirror their mutations to the
    in-RAM graph; hook failures are logged, never raised (write-path
    authoritative). On cache HIT the node_payload blob rehydrates the
    NetworkX attributes directly; MISS rebuilds them from the fresh
    store.all_records() walk that was already happening for the graph.

    Returns (graph, assignment, rich_club).

    Local imports keep the heavy graph/community modules out of Plan-01's
    hot path (core.py module-load time stays small).
    """
    from iai_mcp.community import CommunityAssignment, detect_communities
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.richclub import rich_club_nodes
    from iai_mcp import runtime_graph_cache

    graph = MemoryGraph()

    # try the on-disk cache before running Leiden + rich-club.
    # Cache-first so we can consult the v2 node_payload blob for free.
    cached = runtime_graph_cache.try_load(store)
    assignment = None
    rich_club = None
    cached_node_payload: dict[str, dict] | None = None
    # R2: cached max_degree rehydrates without re-walking the
    # NetworkX graph. Used as a defensive fallback if the live degree
    # walk below fails for any reason.
    cached_max_degree: int = 0
    if cached is not None:
        assignment, rich_club, cached_node_payload, cached_max_degree = cached

    # Build nodes. If the cache gave us a node_payload blob AND the store
    # record count matches, reuse it — skips the encrypted LanceDB scan.
    # Otherwise fall through to the full row walk so node attrs stay
    # strictly derived from the authoritative store.
    records_tbl = store.db.open_table("records")
    records_count = int(records_tbl.count_rows())
    use_cached_payload = (
        cached_node_payload is not None
        and len(cached_node_payload) == records_count
    )

    if use_cached_payload:
        # Fast path: graph nodes + attributes come from the cache JSON.
        for nid, payload in cached_node_payload.items():
            # MemoryGraph.add_node has a fixed signature; use it for
            # topology, then pour the full payload into the NetworkX
            # node attribute dict.
            graph.add_node(
                UUID(nid),
                community_id=None,
                embedding=list(payload.get("embedding") or []),
            )
            graph._nx.nodes[nid].update({
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
            rid = UUID(row["id"])
            community_id = (
                UUID(row["community_id"])
                if row["community_id"]
                else None
            )
            embedding = (
                list(row["embedding"])
                if row["embedding"] is not None
                else [0.0] * EMBED_DIM
            )
            # literal_surface is AES-GCM encrypted at rest.
            # Decrypt here via the store's helper so the graph payload
            # carries plaintext the pipeline can use directly.
            literal_raw = row.get("literal_surface") or ""
            try:
                from iai_mcp.crypto import is_encrypted
                if is_encrypted(literal_raw):
                    literal_raw = store._decrypt_for_record(rid, literal_raw)
            except Exception:
                # / (V2-03 fix): a decrypt failure here
                # used to assign ``literal_raw = ""`` and then fall through
                # to update the live NetworkX node + persist to
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
            # tags travel on graph nodes so the rank stage's
            # SimpleRecordView carries tags for profile_modulation_for_record
            # without needing a store.get fallback in the hot path.
            tags_raw = row.get("tags_json") or "[]"
            try:
                import json as _json
                tags_list = _json.loads(tags_raw) if isinstance(tags_raw, str) else list(tags_raw)
                if not isinstance(tags_list, list):
                    tags_list = []
            except Exception:
                tags_list = []
            language = str(row.get("language") or "en")

            graph.add_node(
                rid,
                community_id=community_id,
                embedding=embedding,
            )
            # /05-13: attach record payload to the NetworkX node dict.
            graph._nx.nodes[str(rid)].update({
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

    # R2: cache the maximum graph degree so the rank stage
    # can normalise log(1+deg) into [0,1] (sample-rank-comparable to
    # cosine; W_DEGREE * deg_norm bounded by W_DEGREE itself instead of
    # by an unbounded log term that scales with hub connectivity).
    # Computed once per build; rehydrated from disk on warm starts via
    # the runtime_graph_cache.json payload. Defensive: fall back to the
    # cached value if the live degree() walk fails for any reason — and
    # never let a bare AttributeError reach the rank stage.
    try:
        deg_values = [d for _, d in graph._nx.degree()]
        max_degree = max(deg_values) if deg_values else 0
    except Exception:
        max_degree = cached_max_degree
    if max_degree == 0 and cached_max_degree > 0:
        # Live walk produced 0 (no edges yet) but the cache held a real
        # value — prefer the cached value. Triggers when an upstream
        # path stripped edges before the rebuild reached us.
        max_degree = cached_max_degree
    graph._max_degree = int(max_degree)

    # Run (or reuse cached) Leiden + rich-club.
    if assignment is None:
        assignment = detect_communities(graph, prior=None)
        rich_club = rich_club_nodes(graph, percent=0.10)

    # compute betweenness centrality ONCE per build
    # and attach to every node as a NetworkX attribute so the rank stage
    # can read it O(1) instead of calling graph.centrality() on every
    # recall (the pre-05-13 hot path). Cache HIT path already rehydrated
    # centrality from node_payload into node attrs above; we only
    # (re)compute when the cache payload is absent / stale or when
    # node_payload centrality values are all-zero placeholders.
    needs_centrality = True
    if use_cached_payload and cached_node_payload is not None:
        # If the cache was written AFTER 05-13 the per-node centrality
        # floats are real (possibly non-zero). If every value is exactly
        # 0.0 the cache was written pre-05-13 shape — recompute to
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
                if nid_str in graph._nx.nodes:
                    graph._nx.nodes[nid_str]["centrality"] = float(cval)
                    if (
                        node_payload_for_cache is not None
                        and nid_str in node_payload_for_cache
                    ):
                        node_payload_for_cache[nid_str]["centrality"] = (
                            float(cval)
                        )
        except Exception:
            # Defensive: centrality is a ranking signal, not a
            # correctness invariant; fall back to zeros on failure.
            for nid_str in graph._nx.nodes:
                graph._nx.nodes[nid_str].setdefault("centrality", 0.0)

    # Persist — fresh build, or cache was legacy 05-09 / 05-12 shape.
    if cached_node_payload is None or needs_centrality:
        runtime_graph_cache.save(
            store, assignment, rich_club,
            node_payload=node_payload_for_cache,
            # R2: max_degree travels with assignment + rich_club
            # so warm-start build_runtime_graph rehydrates without recompute.
            max_degree=int(getattr(graph, "_max_degree", 0) or 0),
        )

    # register the graph-sync hook so future insert/update/
    # delete calls mutate the live graph instead of diverging. The store
    # swallows hook exceptions so a buggy hook never breaks a write.
    try:
        store.register_graph_sync_hook(_make_graph_sync_hook(graph._nx))
    except Exception:
        # Older store without register_graph_sync_hook — this is a
        # defensive upgrade path; the graph just won't stay live-sync'd.
        pass

    # R2 belt-and-braces: every code path above sets
    # graph._max_degree, but if some future refactor short-circuits
    # before reaching the live degree walk we still want the rank
    # stage's `getattr(graph, "_max_degree", 0)` to read a real int.
    if not hasattr(graph, "_max_degree"):
        graph._max_degree = 0

    return graph, assignment, rich_club
