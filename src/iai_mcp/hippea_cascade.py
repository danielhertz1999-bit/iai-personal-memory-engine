"""HIPPEA activation-cascade prefetch.

Daemon receives ``session_open`` over the unix socket and this module
computes precision-weighted salience over 7 days of ``session_started`` +
``retrieval_used`` events, selects top-K communities, and pre-warms their
top-N records into a process-local LRU cache (cachetools.TTLCache) guarded
by an asyncio.Lock.

Salience formula (Van de Cruys 2014 HIPPEA):
    f(c)   = count(session_gated_to_community=c, last_7_days) / total_sessions_7d
    p(c)   = 1 / |communities|
    PE(c)  = |f(c) - p(c)|
    sigma2 = Var[day_i_count(c) : i in 7 days]
    w(c)   = 1 / (sigma2(c) + 0.01)
    S(c)   = w(c) * PE(c)
    top_K  = argmax_K S(c)                                  # K=3 default
    warm   = union over c in top_K of top_N_by_centrality(records(c))

Cold-fallback (<3 sessions in 7-day window): return
assignment.top_communities[:top_k] without variance weighting.

Off-loop dispatch contract (event-loop safety):
  - ``fetch_warm_records`` is SYNC and LOCK-FREE: runs the store.get loop
    on the caller's thread (must be an executor thread, not the event loop).
  - ``_install_warm`` is ASYNC, on-loop: only touches the in-memory _warm_lru
    dict under _warm_lru_lock. No store.get, no SQLite, no decrypt.
  - ``compute_and_fetch_warm`` is SYNC, the off-loop callable the daemon
    submits to a dedicated bounded executor. Returns (records, top) so the
    caller can reconstruct the full stats dict without a second store pass.
  - ``warm_records`` and ``run_cascade`` are thin async wrappers preserved
    for direct (non-daemon) callers and existing tests.

Invariants (asserted by grep guards in the test suite):
- Human-first: cascade task yields on shutdown within 5s.
- Zero API cost: pure local -- no paid-API env var, no Anthropic SDK import.
- Read-only: no store.insert / store.append_provenance / store.update calls.
"""
from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from uuid import UUID

from cachetools import TTLCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------- process-local LRU

# Cache constants: maxsize=200, ttl=1800 (30 min).
# These keep the cache small enough to fit in MCP core RAM headroom.
_WARM_MAXSIZE = 200
_WARM_TTL_SECONDS = 1800


_warm_lru: TTLCache[UUID, Any] = TTLCache(maxsize=_WARM_MAXSIZE, ttl=_WARM_TTL_SECONDS)
_warm_lru_lock = asyncio.Lock()


def snapshot_warm_ids() -> list[UUID]:
    """Lock-free snapshot of warm record IDs.

    CPython GIL makes `list(dict.keys())` atomic for simple types. A concurrent
    mutator may race and invalidate the iterator -- we catch RuntimeError and
    return an empty list rather than propagating the rare race.
    """
    try:
        return list(_warm_lru.keys())
    except RuntimeError:
        return []


def get_warm_record(rid: UUID) -> Any | None:
    """Return the warmed record or None. Silent on miss / structural error."""
    try:
        return _warm_lru.get(rid)
    except (KeyError, TypeError):
        return None


def fetch_warm_records(store: Any, ids: Iterable[UUID]) -> list:
    """Fetch records from the store for the given ids. SYNC, LOCK-FREE.

    Intended to run on an off-loop executor thread. Performs the store.get
    loop with per-record exception swallowing. Does NOT touch _warm_lru and
    acquires no asyncio.Lock — safe to call from any thread.

    C6: READ-ONLY (store.get only, no mutations).
    """
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
    """Insert already-fetched records into the warm LRU. ASYNC, on-loop.

    Only touches the in-memory _warm_lru dict under _warm_lru_lock.
    No store.get, no SQLite, no decrypt. The records must already be
    resolved (i.e. returned by fetch_warm_records).
    """
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
    """Load records into the LRU. Returns count inserted.

    Thin wrapper over fetch_warm_records + _install_warm.
    Signature and return value are unchanged so direct callers and
    existing tests are unaffected.

    C6: READ-ONLY against the store -- only store.get is called.
    """
    return await _install_warm(fetch_warm_records(store, record_ids))


# ---------------------------------------------------------- salience formula


def compute_salient_communities(
    store: Any,
    assignment: Any,
    *,
    lookback_days: int = 7,
    top_k: int = 3,
) -> list[UUID]:
    """Return top-K community UUIDs by HIPPEA salience S(c) = w(c) * PE(c).

    Cold fallback (<3 sessions in window): return
    `assignment.top_communities[:top_k]` with no variance weighting.
    """
    # Lazy import to keep the module's surface clean of store-mutating paths.
    from iai_mcp.events import query_events

    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    try:
        sessions = query_events(store, kind="session_started", since=since, limit=10000)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("hippea_session_query_failed", extra={"err": str(exc)[:80]})
        sessions = []

    if len(sessions) < 3:
        # Cold fallback (<3 sessions): simplified formula drops the variance term.
        # Use the existing Leiden top-communities as a reasonable default.
        return list(getattr(assignment, "top_communities", []))[:top_k]

    try:
        retrievals = query_events(
            store, kind="retrieval_used", since=since, limit=50000,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("hippea_retrieval_query_failed", extra={"err": str(exc)[:80]})
        retrievals = []

    # session_id -> dominant community for that session (most retrieved).
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
    # Also admit any community seen in retrievals during the window even if it
    # isn't in top_communities -- the salience formula evaluates all observed
    # communities, not just the Leiden-top.
    seen: set[str] = set(session_comm.values())
    for cid in (str(c) for c in community_pool):
        seen.add(cid)
    if not seen:
        return []
    p = 1.0 / len(seen)

    # f(c) across the window.
    freq: Counter = Counter(session_comm.values())

    # Day-bucketed counts (0 = today, lookback_days-1 = oldest).
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

    # Compute S(c) per community.
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
        key=lambda kv: (-kv[1], kv[0]),  # deterministic tiebreak by cid str
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


# ---------------------------------------------------------- centrality helper


def _top_n_records_by_centrality(
    store: Any, assignment: Any, community_id: UUID, n: int,
) -> list[UUID]:
    """READ-ONLY: return top-N record ids for ``community_id`` by centrality.

    Uses ``assignment.mid_regions[community_id]`` to enumerate member records.
    Reads all members' centrality values in ONE bulk projection — no per-member
    full-record fetch and no AES-GCM decrypt.  The projection flows through the
    connection-lock-guarded ``to_batches`` path and scales as a single bounded
    scan over the table, not as member-count x per-record-decrypt.

    Sort order: descending centrality, then ascending ``str(id)`` as a
    deterministic tiebreak.  Members absent from the store are omitted.
    Missing or NULL centrality is treated as 0.0.
    """
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


# ---------------------------------------------------------- sync core-side helper


def compute_core_side_warm_snapshot(
    store: Any,
    assignment: Any,
    *,
    top_k: int = 3,
    per_community: int | None = None,
    max_records: int = 50,
) -> list[UUID]:
    """Synchronous counterpart to :func:`run_cascade`'s compute path.

    The MCP core runs in a different process from the sleep daemon, so
    the daemon's ``_warm_lru`` is invisible to core --
    ``snapshot_warm_ids()`` returns ``[]`` in the core on every fresh
    process boot. This helper lets the core compute its OWN cascade
    inline (no asyncio dependency) and write the warmed record ids into
    its own process-local LRU. Duplicates daemon work by design; that
    is the price of not having shared-memory IPC between the two
    processes.

    Reuses :func:`compute_salient_communities` (already sync) and
    :func:`_top_n_records_by_centrality` (sync) -- no new salience
    formula; only the orchestration that :func:`run_cascade` would do
    asynchronously.

    READ-ONLY against store; no async I/O; no paid-API import.
    """
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


# ---------------------------------------------------------- off-loop entrypoint


def compute_and_fetch_warm(
    store: Any,
    assignment: Any,
    *,
    top_k: int = 3,
    per_community: int | None = None,
) -> tuple:
    """Compute community salience, select top-K records, and fetch them. SYNC.

    Intended to run on a dedicated off-loop executor thread. Replicates
    run_cascade's exact cap math so the warm set is identical.

    Returns ``(records, top)`` where ``records`` is the list of fetched
    MemoryRecord objects and ``top`` is the list of selected community UUIDs.
    The caller (daemon or run_cascade wrapper) assembles the stats dict from
    these values and installs the records via ``_install_warm`` on the loop.

    Does NOT touch _warm_lru. No asyncio.Lock. Safe to call from any thread.
    C6: READ-ONLY (no store mutations).
    """
    top = compute_salient_communities(store, assignment, top_k=top_k)
    if not top:
        return [], []

    # Same cap math as run_cascade (NOT compute_core_side_warm_snapshot's
    # max_records=50 scheme, which uses a different default).
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


# ---------------------------------------------------------- public entrypoint


async def run_cascade(
    store: Any,
    assignment: Any,
    *,
    top_k: int = 3,
    per_community: int | None = None,
) -> dict:
    """Pre-warm records for top-K salient communities.

    Thin async wrapper over compute_and_fetch_warm + _install_warm.
    Signature and return value are unchanged so direct callers and
    existing tests are unaffected.

    Returns a stats dict: {
        "communities_selected": int,
        "records_warmed": int,
        "top_communities": list[str],
    }
    """
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
