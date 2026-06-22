from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from itertools import combinations
from uuid import UUID, uuid4

import pandas as pd

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


log = logging.getLogger(__name__)

_GRAPH_DECRYPT_WARN_LAST: dict[str, float] = {}
_GRAPH_DECRYPT_WARN_INTERVAL_SEC = 300.0


TEMPORAL_NEXT_WINDOW = timedelta(minutes=5)


STALE_DOWNWEIGHT_FACTOR: float = 0.5

_STALE_REASON_SUFFIX: str = " · stale"


# --- build_runtime_graph single-flight (WAKE CPU-storm fix) -----------------
# At daemon WAKE several background subsystems (boot preload, sigma identity
# audit, foraging weak-bridge detection, hippea cascade warming) each call
# build_runtime_graph concurrently. On a cache MISS each one independently runs
# the full O(n^2) community detection (mosaic), GIL-bound, in its own to_thread
# worker. Three+ of those at once contend for the GIL, starve the asyncio event
# loop, and the liveness watchdog's socket probe times out -> SIGKILL -> relaunch
# loop. This single-flight gate collapses the concurrent burst into ONE compute:
# the first caller (leader) computes and saves the on-disk cache; concurrent
# callers (followers) wait on its Event and then re-load the freshly-saved cache
# via the cheap path. No mutable graph object is shared between callers (each
# rebuilds its own MemoryGraph shell + sync hook), and recall is independent of
# the community assignment, so a slightly-stale shared result is harmless.
#
# Followers RE-CONTEND in a bounded loop rather than recomputing unconditionally:
# if the leader fails before saving (e.g. detect_communities raises), or the cache
# key shifts mid-burst, or the leader overruns the wait timeout, the woken
# followers loop back, and exactly ONE of them becomes the next leader while the
# rest wait again. That degrades those edge cases to *sequential* single-flight
# (one compute at a time) instead of an N-way concurrent re-storm.
_BRG_INFLIGHT_LOCK = threading.Lock()
_BRG_INFLIGHT: dict[str, threading.Event] = {}
_BRG_WAIT_TIMEOUT_SEC: float = 120.0
_BRG_MAX_ATTEMPTS: int = 4


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
    raw = store.query_similar(cue_embedding, k=k_hits + k_anti)

    if mode == "verbatim":
        raw = [
            (rec, score) for rec, score in raw
            if rec.tier == "episodic"
            and not any(t.startswith("pattern:") for t in (rec.tags or []))
        ]

    hits: list[MemoryHit] = []
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
                adjacent_suggestions=[],
                session_id=_prov.get("session_id"),
                captured_at=record.created_at.isoformat() if record.created_at else None,
            )
        )
        provenance_pending.append((
            record.id,
            {
                "ts": now_iso,
                "cue": cue_text,
                "session_id": session_id,
            },
        ))

    if provenance_pending:
        try:
            store.queue_provenance_batch(provenance_pending)
        except (OSError, ValueError, RuntimeError) as exc:
            log.warning("provenance_batch write failed: %s", exc)

    anti_hits: list[MemoryHit] = []
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

    derive_temporal_validity(store, hits)
    derive_temporal_validity(store, anti_hits)
    apply_stale_downweight(hits)
    apply_stale_downweight(anti_hits)
    # Rank on the internal unclamped key (falls back to score), so ordering
    # survives the display clamp applied at serialization.
    hits.sort(
        key=lambda h: (h.sort_score if h.sort_score is not None else h.score),
        reverse=True,
    )

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
        budget_used=sum(len(h.literal_surface) for h in hits) // 4,
        hints=s4_hints,
        cue_mode=mode,
        patterns_observed=[],
    )

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
    pairs: list[tuple[UUID, UUID]] = list(combinations(ids, 2))
    new_weights = store.boost_edges(pairs, delta=delta)
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
    flush_record_buffer(store)
    original = store.get(original_id)
    if original is None:
        raise ValueError(f"unknown record {original_id}")
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
        language=getattr(original, "language", "en") or "en",
    )
    enforce_english_raw(new_rec)
    new_rec.aaak_index = generate_aaak_index(new_rec)
    store.insert(new_rec)
    store.add_contradicts_edge(original_id, new_rec.id)
    invalidate_temporal_validity_cache(store)

    try:
        from iai_mcp.s4 import monotropic_proactive_check
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
    if not hits:
        return hits

    if outgoing is None or ts_by_id is None:
        if store is None:
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
    if cue_intent == "historical_verbatim":
        return hits
    now_value = now or datetime.now(timezone.utc)
    for hit in hits:
        if hit.valid_to is None or hit.valid_to >= now_value:
            continue
        if not getattr(hit, "_stale_downweighted", False):
            hit.score *= STALE_DOWNWEIGHT_FACTOR
            # Keep the internal ranking key in lock-step with the displayed
            # score so stale hits demote in the actual ordering too.
            if getattr(hit, "sort_score", None) is not None:
                hit.sort_score *= STALE_DOWNWEIGHT_FACTOR
            hit._stale_downweighted = True
        if not hit.reason.endswith(_STALE_REASON_SUFFIX):
            hit.reason = f"{hit.reason}{_STALE_REASON_SUFFIX}"
    return hits


def link_temporal_next(
    store: MemoryStore,
    new_record: MemoryRecord,
    session_id: str,
) -> UUID | None:
    now = datetime.now(timezone.utc)
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
        break

    if previous_id is not None:
        try:
            store.boost_edges(
                [(previous_id, new_record.id)],
                edge_type="temporal_next",
                delta=1.0,
            )
        except (OSError, ValueError, RuntimeError) as exc:
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
            try:
                from iai_mcp import runtime_graph_cache as _rgc
                _rgc.increment_dirty_counter()
            except Exception:  # noqa: BLE001 -- never break a record write
                pass
        elif op == "delete":
            graph.remove_node(nid)
            try:
                from iai_mcp import runtime_graph_cache as _rgc
                _rgc.increment_dirty_counter()
            except Exception:  # noqa: BLE001 -- never break a record delete
                pass
    return _hook


def build_runtime_graph(store: MemoryStore):
    """Single-flight wrapper around the real graph build.

    On a cache HIT this is cheap and runs directly. On a cache MISS it
    serialises concurrent callers so the expensive community detection runs
    exactly once per cache generation: the leader computes + saves the cache,
    followers wait and then reload it cheaply. See the _BRG_* notes above.
    """
    from iai_mcp import runtime_graph_cache as _rgc

    cached = None
    for _attempt in range(_BRG_MAX_ATTEMPTS):
        try:
            cached = _rgc.try_load(store)
        except Exception:  # noqa: BLE001 -- never let cache I/O break the build
            cached = None

        # Cache HIT: no contention risk, run directly (the impl reloads cheaply).
        if cached is not None and cached[0] is not None:
            return _build_runtime_graph_impl(store, cached)

        # Cache MISS: single-flight on the cache key so a WAKE burst of callers
        # does not all recompute the full mosaic concurrently.
        try:
            keystr = repr(_rgc._cache_key(store))
        except Exception:  # noqa: BLE001 -- if we can't key it, just compute
            return _build_runtime_graph_impl(store, cached)

        with _BRG_INFLIGHT_LOCK:
            event = _BRG_INFLIGHT.get(keystr)
            is_leader = event is None
            if is_leader:
                event = threading.Event()
                _BRG_INFLIGHT[keystr] = event

        if is_leader:
            # Leader: compute (the impl saves the cache), then release followers.
            try:
                return _build_runtime_graph_impl(store, cached)
            finally:
                with _BRG_INFLIGHT_LOCK:
                    # Only drop our own slot (a key shift could have replaced it).
                    if _BRG_INFLIGHT.get(keystr) is event:
                        _BRG_INFLIGHT.pop(keystr, None)
                event.set()

        # Follower: wait for the leader, then loop. Next iteration's try_load
        # HITS if the leader saved; if the leader failed / the key shifted /
        # the wait timed out, we re-contend and one follower becomes the next
        # leader (sequential single-flight — never an N-way concurrent re-storm).
        event.wait(timeout=_BRG_WAIT_TIMEOUT_SEC)

    # Attempts exhausted (e.g. the leader keeps failing): compute directly as a
    # last resort. Bounded, and still correct.
    return _build_runtime_graph_impl(store, cached)


def _build_runtime_graph_impl(store: MemoryStore, cached):
    from iai_mcp.community import detect_communities
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.richclub import rich_club_nodes
    from iai_mcp import runtime_graph_cache

    graph = MemoryGraph()

    assignment = None
    rich_club = None
    cached_node_payload: dict[str, dict] | None = None
    cached_max_degree: int = 0
    if cached is not None:
        assignment, rich_club, cached_node_payload, cached_max_degree = cached

    records_tbl = store.db.open_table("records")
    records_count = store.active_records_count()
    use_cached_payload = (
        cached_node_payload is not None
        and len(cached_node_payload) == records_count
    )

    if cached_node_payload is not None and len(cached_node_payload) > records_count:
        # The cache holds MORE nodes than are live: records were tombstoned
        # (dedup/erasure) since it was built, so the cached assignment/rich_club
        # were computed over now-dead nodes -- drop them and recompute on the
        # fresh live graph (rebuilt from the records table below, which already
        # excludes tombstoned rows). Pure GROWTH (cache has FEWER nodes than live)
        # must NOT drop them: the node set is still rebuilt fresh from the table
        # (so new records are present and drift/parity stay correct), but
        # detect_communities is not re-run, so a single insert is absorbed without
        # an O(n^2) recompute -- the staleness-window contract.
        assignment = None
        rich_club = None

    if use_cached_payload:
        for nid, payload in cached_node_payload.items():
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
        df = records_tbl.to_pandas()
        node_payload_for_cache = {}
        decrypt_fail_events = 0
        decrypt_fail_unique: set[str] = set()
        for _, row in df.iterrows():
            if int(row.get("embedding_pending") or 0) != 0:
                continue
            # Exclude tombstoned (soft-deleted / deduped / erased) records from the
            # runtime graph: they must not pollute communities, centrality, rich_club
            # or the sigma topology audit, and including them keeps the node count out
            # of sync with store.active_records_count() (the cache-validity anchor at
            # line ~552), permanently invalidating the cache -> a full rebuild every
            # wake. Matches active_records_count(): tombstoned_at IS NULL.
            #
            # NULL is the LIVE case and pandas materialises it differently per
            # dtype: None (object column), float('nan'), pd.NaT (a datetime64
            # column -- the shape the reembed Arrow schema produces), or pd.NA
            # (nullable extension dtypes). The old guard only caught float-NaN
            # (`isinstance(float) and v != v`); a NaT/NA live value slipped past
            # it, stringified to "NaT"/"<NA>", read truthy, and the LIVE record
            # was wrongly dropped -- collapsing the whole graph to empty on a
            # reembedded store. pd.isna() covers None/NaN/NaT/NA uniformly, so
            # only a real timestamp string survives to mark a tombstone.
            _tomb = row.get("tombstoned_at")
            if _tomb is not None and not pd.isna(_tomb) and str(_tomb).strip():
                continue
            rid = UUID(row["id"])
            _comm_raw = row["community_id"]
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
            literal_raw = row.get("literal_surface") or ""
            try:
                from iai_mcp.crypto import is_encrypted
                if is_encrypted(literal_raw):
                    literal_raw = store._decrypt_for_record(rid, literal_raw)
            except Exception:  # noqa: BLE001 -- InvalidTag / OSError / ValueError / RuntimeError
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
        # Skip edges whose endpoints are not live nodes: graph.add_edge() does
        # setdefault() on both endpoints, so an edge referencing a tombstoned record
        # would re-create it as a payload-less node and undo the tombstone filter
        # above (re-bloating the graph and the sigma audit).
        src_s, dst_s = row["src"], row["dst"]
        if not graph.has_node(src_s) or not graph.has_node(dst_s):
            continue
        graph.add_edge(
            UUID(src_s),
            UUID(dst_s),
            weight=float(row["weight"]),
            edge_type=row["edge_type"],
        )

    try:
        deg_values = [d for _, d in graph.degrees()]
        max_degree = max(deg_values) if deg_values else 0
    except (ValueError, RuntimeError, AttributeError):
        max_degree = cached_max_degree
    if max_degree == 0 and cached_max_degree > 0:
        max_degree = cached_max_degree
    graph._max_degree = int(max_degree)

    if assignment is None:
        assignment = detect_communities(graph, prior=None, prior_mode="seeded")
        rich_club = rich_club_nodes(graph, percent=0.10)

    needs_centrality = True
    if use_cached_payload and cached_node_payload is not None:
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
            log.warning("centrality computation failed: %s", exc)
            for nid in graph.iter_nodes():
                key = str(nid)
                if "centrality" not in graph._node_payload.get(key, {}):
                    graph.set_node_centrality(nid, 0.0)

    if cached_node_payload is None or needs_centrality:
        runtime_graph_cache.save(
            store, assignment, rich_club,
            node_payload=node_payload_for_cache,
            max_degree=int(getattr(graph, "_max_degree", 0) or 0),
        )

    try:
        store.register_graph_sync_hook(_make_graph_sync_hook(graph))
    except (AttributeError, TypeError, RuntimeError) as exc:
        log.warning("graph_sync_hook registration failed: %s", exc)

    if not hasattr(graph, "_max_degree"):
        graph._max_degree = 0

    return graph, assignment, rich_club
