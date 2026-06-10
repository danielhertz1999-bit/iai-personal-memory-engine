from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from uuid import UUID

from cachetools import TTLCache

logger = logging.getLogger(__name__)


_WARM_MAXSIZE = 200
_WARM_TTL_SECONDS = 1800


_warm_lru: TTLCache[UUID, Any] = TTLCache(maxsize=_WARM_MAXSIZE, ttl=_WARM_TTL_SECONDS)
_warm_lru_lock = asyncio.Lock()


def snapshot_warm_ids() -> list[UUID]:
    try:
        return list(_warm_lru.keys())
    except RuntimeError:
        return []


def get_warm_record(rid: UUID) -> Any | None:
    try:
        return _warm_lru.get(rid)
    except (KeyError, TypeError):
        return None


def fetch_warm_records(store: Any, ids: Iterable[UUID]) -> list:
    result = []
    for rid in ids:
        try:
            rec = store.get(rid)
            if rec is not None:
                result.append(rec)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            logger.debug("fetch_warm_failed", extra={"rid": str(rid), "err": str(exc)[:80]})
            continue
    return result


async def _install_warm(records: list) -> int:
    inserted = 0
    async with _warm_lru_lock:
        for rec in records:
            try:
                rid = getattr(rec, "id", None)
                if rid is not None:
                    _warm_lru[rid] = rec
                    inserted += 1
            except Exception as exc:  # noqa: BLE001 -- never let a bad record crash the warmer
                logger.debug("install_warm_failed", extra={"err": str(exc)[:80]})
                continue
    return inserted


async def warm_records(record_ids: Iterable[UUID], store: Any) -> int:
    return await _install_warm(fetch_warm_records(store, record_ids))


def compute_salient_communities(
    store: Any,
    assignment: Any,
    *,
    lookback_days: int = 7,
    top_k: int = 3,
) -> list[UUID]:
    from iai_mcp.events import query_events

    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    try:
        sessions = query_events(store, kind="session_started", since=since, limit=10000)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("hippea_session_query_failed", extra={"err": str(exc)[:80]})
        sessions = []

    if len(sessions) < 3:
        return list(getattr(assignment, "top_communities", []))[:top_k]

    try:
        retrievals = query_events(
            store, kind="retrieval_used", since=since, limit=50000,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("hippea_retrieval_query_failed", extra={"err": str(exc)[:80]})
        retrievals = []

    per_session_counter: dict[str, Counter] = defaultdict(Counter)
    for ev in retrievals:
        data = ev.get("data", {}) if isinstance(ev, dict) else {}
        sid = data.get("session_id") or ev.get("session_id", "")
        cid = data.get("community_id") or data.get("community", "")
        if sid and cid:
            per_session_counter[sid][str(cid)] += 1
    session_comm: dict[str, str] = {
        sid: ctr.most_common(1)[0][0]
        for sid, ctr in per_session_counter.items()
        if ctr
    }

    total_sessions = len(sessions)
    community_pool: list[UUID] = list(getattr(assignment, "top_communities", []) or [])
    seen: set[str] = set(session_comm.values())
    for cid in (str(c) for c in community_pool):
        seen.add(cid)
    if not seen:
        return []
    p = 1.0 / len(seen)

    freq: Counter = Counter(session_comm.values())

    day_buckets: dict[str, list[int]] = defaultdict(lambda: [0] * lookback_days)
    now = datetime.now(timezone.utc)
    for sev in sessions:
        ts = sev.get("ts") if isinstance(sev, dict) else None
        try:
            if isinstance(ts, str):
                t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif hasattr(ts, "to_pydatetime"):
                t = ts.to_pydatetime()
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
            elif hasattr(ts, "tzinfo") and ts is not None:
                t = ts
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
            else:
                t = now
            delta = (now - t).days
            day_idx = max(0, min(lookback_days - 1, delta))
        except (TypeError, ValueError, AttributeError):
            day_idx = 0
        data = sev.get("data", {}) if isinstance(sev, dict) else {}
        sid = data.get("session_id") or sev.get("session_id", "")
        c = session_comm.get(sid)
        if c:
            day_buckets[c][day_idx] += 1

    scores: dict[str, float] = {}
    for c in seen:
        f_c = freq.get(c, 0) / max(1, total_sessions)
        pe = abs(f_c - p)
        bucket = day_buckets.get(c, [0] * lookback_days)
        n = len(bucket) or 1
        mean = sum(bucket) / n
        variance = sum((x - mean) ** 2 for x in bucket) / n
        w = 1.0 / (variance + 0.01)
        scores[c] = w * pe

    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )
    top: list[UUID] = []
    for cid_str, _ in ranked:
        try:
            top.append(UUID(cid_str))
        except (TypeError, ValueError):
            continue
        if len(top) >= top_k:
            break
    return top


def _top_n_records_by_centrality(
    store: Any, assignment: Any, community_id: UUID, n: int,
) -> list[UUID]:
    mid_regions = getattr(assignment, "mid_regions", {}) or {}
    member_ids = list(mid_regions.get(community_id) or [])
    if not member_ids:
        return []
    centrality_map: dict[UUID, float] = store.centrality_for_ids(member_ids)
    scored: list[tuple[float, UUID]] = []
    for rid in member_ids:
        if rid not in centrality_map:
            continue
        scored.append((centrality_map[rid], rid))
    scored.sort(key=lambda kv: (-kv[0], str(kv[1])))
    return [rid for _c, rid in scored[:n]]


def compute_core_side_warm_snapshot(
    store: Any,
    assignment: Any,
    *,
    top_k: int = 3,
    per_community: int | None = None,
    max_records: int = 50,
) -> list[UUID]:
    top = compute_salient_communities(store, assignment, top_k=top_k)
    if not top:
        return []
    per_c = per_community or max(1, max_records // max(1, len(top)))
    out: list[UUID] = []
    for cid in top:
        try:
            out.extend(_top_n_records_by_centrality(store, assignment, cid, per_c))
        except (OSError, KeyError, ValueError, RuntimeError):
            continue
    return out[:max_records]


def compute_and_fetch_warm(
    store: Any,
    assignment: Any,
    *,
    top_k: int = 3,
    per_community: int | None = None,
) -> tuple:
    top = compute_salient_communities(store, assignment, top_k=top_k)
    if not top:
        return [], []

    per_c = per_community or max(1, _WARM_MAXSIZE // max(1, len(top)))
    to_warm: list[UUID] = []
    for cid in top:
        try:
            rec_ids = _top_n_records_by_centrality(store, assignment, cid, per_c)
            to_warm.extend(rec_ids)
        except (OSError, KeyError, ValueError, RuntimeError):
            continue
    records = fetch_warm_records(store, to_warm[:_WARM_MAXSIZE])
    return records, top


async def run_cascade(
    store: Any,
    assignment: Any,
    *,
    top_k: int = 3,
    per_community: int | None = None,
) -> dict:
    recs, top = compute_and_fetch_warm(
        store, assignment, top_k=top_k, per_community=per_community,
    )
    if not top:
        return {"communities_selected": 0, "records_warmed": 0, "top_communities": []}
    inserted = await _install_warm(recs)
    return {
        "communities_selected": len(top),
        "records_warmed": inserted,
        "top_communities": [str(c) for c in top],
    }
