"""Retrieval + reinforcement + contradiction paths.

- `recall`: baseline cosine top-k -- kept as a fallback for the
  empty-store case and for regression tests.
- `build_runtime_graph`: reconstruct a MemoryGraph + CommunityAssignment +
  rich-club from LanceDB state; consumed by core.py to drive `pipeline_recall`.
- `reinforce_edges`, `contradict`: unchanged from Plan 01.
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


# Plan 07.11-02 / structured-log handle for the graph-build
# decrypt-failure path. Same one-liner the rest of the project uses
# (cf. capture.py:54, pipeline.py:33-imports). Used by the
# `graph_build_decrypt_failed` event when AES-GCM decrypt of a
# record's literal_surface raises during build_runtime_graph.
log = logging.getLogger(__name__)

# Per-process rate limit for graph_build_decrypt_failed (rid -> monotonic ts).
_GRAPH_DECRYPT_WARN_LAST: dict[str, float] = {}
_GRAPH_DECRYPT_WARN_INTERVAL_SEC = 300.0


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
    """Phase 1 baseline retrieval.

    Fetches top (k_hits + k_anti) by cosine similarity; treats the top k_hits as
    excitatory hits and the bottom k_anti as a naive anti-hit stub. Plan 02 will
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
    # pipeline_recall verbatim mode (R5). Also excludes D-09
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
    # src/iai_mcp/pipeline.py::pipeline_recall (see D-SPEED SC-6).
    provenance_pending: list[tuple[UUID, dict]] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for record, score in raw[:k_hits]:
        hits.append(
            MemoryHit(
                record_id=record.id,
                score=float(score),
                reason=f"cosine {score:.3f}",
                literal_surface=record.literal_surface,
                adjacent_suggestions=[],  # Plan 03 fills per AUTIST-07
            )
        )
        # every recall appends a provenance entry; write is batched
        # end-of-loop to preserve rank stability (Plan 05-02 effect c fix).
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
    # Naive anti-hit stub: bottom-k of the same query. Plan 02 replaces with
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
        # ~4 chars per token heuristic; Plan 03 benchmark will use Anthropic count_tokens.
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
    """MEM-05 edge-based reconsolidation.

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
    # user can inspect them via `iai-mcp contradictions` in Plan 02-04.
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
                # Plan 07.11-02 / (V2-03 fix): a decrypt failure here
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
            # Plan 05-12/05-13: attach record payload to the NetworkX node dict.
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
