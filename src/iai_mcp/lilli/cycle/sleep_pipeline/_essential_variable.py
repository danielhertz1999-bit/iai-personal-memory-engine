from __future__ import annotations

import json
import logging
import os

from iai_mcp.exceptions import StoreError
from iai_mcp.lifecycle_state import _utc_now_iso

logger = logging.getLogger(__name__)


def clear_crisis_mode_via_s2_or_fallback(self, *, reason: str) -> bool:
    s2 = getattr(self, "_s2_coordinator", None)
    loop = getattr(self, "_loop", None)
    if s2 is None:
        return False
    try:
        import asyncio
        coro = s2.set_crisis_mode(False, reason)
        if loop is not None and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            fut.result(timeout=5.0)
        else:
            asyncio.run(coro)
        return True
    except (OSError, RuntimeError, TimeoutError) as exc:
        logger.debug("S2 clear_crisis_mode failed, falling back: %s", exc)
        return False


def set_crisis_mode_via_s2_or_fallback(
    self, *, value: bool, reason: str,
) -> bool:
    s2 = getattr(self, "_s2_coordinator", None)
    loop = getattr(self, "_loop", None)
    if s2 is not None:
        try:
            import asyncio
            coro = s2.set_crisis_mode(value, reason)
            if loop is not None and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                fut.result(timeout=5.0)
            else:
                asyncio.run(coro)
            return True
        except (OSError, RuntimeError, TimeoutError) as exc:
            logger.debug("S2 set_crisis_mode failed, falling back: %s", exc)
    try:
        rec = self._load_state_record()
        rec["crisis_mode"] = bool(value)
        rec["crisis_mode_since_ts"] = _utc_now_iso() if bool(value) else None
        self._save_state_record(rec)
        return False
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("crisis_mode fallback save_state failed: %s", exc)
        return False


def run_essential_variable_tracker_hook(self) -> None:
    from iai_mcp.daemon_config import _load_sleep_overhaul_config
    from iai_mcp.ashby_step import (
        EssentialVariableTracker,
        TopologySnapshot,
    )
    from iai_mcp.events import write_event
    from iai_mcp.store import RECORDS_TABLE
    from iai_mcp.lilli.cycle.sleep_pipeline._live_graph import (
        _is_tombstoned,
        build_live_graph,
    )

    cfg = _load_sleep_overhaul_config()
    dry_run = cfg.dry_run

    try:
        recs = (
            self._store.db.open_table(RECORDS_TABLE)
            .search().to_pandas()
        )
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("essential_variable_tracker records query failed: %s", exc)
        return
    if recs.empty:
        return

    import uuid as _uuid
    # Build the graph on LIVE records + live-only edges only. Previously this
    # hook constructed the graph from ALL records (tombstoned included) and ALL
    # edges, so rich_club_coefficient was computed on a graph polluted by the
    # 3000+ deduped/tombstoned nodes (and phantom nodes re-created by add_edge).
    # That pushed rich_club below its floor and re-armed crisis on every sleep
    # cycle even on a healthy store. build_live_graph mirrors retrieve.py's fix
    # (53f04f9) so the crisis view matches recall's view of the graph.
    g = build_live_graph(self._store)
    community_ids: set = set()
    _community_embeddings: dict[str, list[list[float]]] = {}
    for _, row in recs.iterrows():
        if _is_tombstoned(row.get("tombstoned_at")):
            continue
        try:
            emb = row.get("embedding")
            emb_list = list(emb) if emb is not None else []
            cid_raw = row.get("community_id")
            if cid_raw is not None:
                try:
                    _cid_str = str(_uuid.UUID(str(cid_raw)))
                    community_ids.add(_cid_str)
                    if emb_list:
                        _community_embeddings.setdefault(
                            _cid_str, []
                        ).append(emb_list)
                except (ValueError, TypeError):
                    pass
        except (ValueError, TypeError, AttributeError):
            continue

    total_nodes = g.node_count()
    if total_nodes == 0:
        return

    try:
        rc_ratio = g.rich_club_coefficient()
    except (ValueError, RuntimeError, ZeroDivisionError) as exc:
        logger.debug("rich_club_coefficient failed: %s", exc)
        rc_ratio = 0.0
    nedges = sum(1 for _ in g.iter_edges_with_weight())
    edge_density = (
        (2.0 * nedges) / (total_nodes * (total_nodes - 1))
        if total_nodes >= 2 else 0.0
    )

    snapshot = TopologySnapshot(
        rich_club_ratio=float(rc_ratio),
        community_count=int(len(community_ids)),
        edge_density=float(edge_density),
        total_nodes=int(total_nodes),
    )
    tracker = EssentialVariableTracker(cfg)
    breaches = tracker.check(snapshot)

    # rich_club_ratio is kept as a DIAGNOSTIC only, not a crisis trigger. On the
    # real corpus the live (tombstone-filtered) rich_club sits ~0.019, just under
    # the 0.02 floor, so it false-positives on a demonstrably healthy graph and
    # is non-discriminant at this scale (a true collapse instead shows up as
    # edge_density/community_count). edge_density and community_count remain the
    # crisis triggers (both healthy with clear margin). The rich_club breach is
    # still recorded as an event for observability.
    _CRISIS_TRIGGER_VARS = {"edge_density", "community_count"}
    crisis_mode_already_set_this_cycle = False
    for var_name, breach in breaches.items():
        if breach is None:
            continue
        is_trigger = var_name in _CRISIS_TRIGGER_VARS
        crisis_mode_set = False
        if is_trigger and not dry_run and not crisis_mode_already_set_this_cycle:
            self._set_crisis_mode_via_s2_or_fallback(
                value=True,
                reason=f"essential_variable_breach:{var_name}",
            )
            crisis_mode_already_set_this_cycle = True
            crisis_mode_set = True
        elif is_trigger and not dry_run and crisis_mode_already_set_this_cycle:
            crisis_mode_set = True
        write_event(
            self._store,
            "essential_variable_breach",
            {
                "variable_name": str(var_name),
                "observed_value": float(breach.observed_value),
                "threshold": float(breach.threshold),
                "direction": str(breach.direction),
                "total_nodes": int(total_nodes),
                "crisis_mode_set": bool(crisis_mode_set),
                "is_crisis_trigger": bool(is_trigger),
                "dry_run_mode": bool(dry_run),
            },
            severity="warning",
        )

    if os.environ.get(
        "IAI_MCP_ORTHO_ENABLED", "",
    ).lower() in {"1", "true"}:
        try:
            from iai_mcp.pattern_separation import detect_hubness
            if _community_embeddings:
                _largest_cid = max(
                    _community_embeddings,
                    key=lambda k: len(_community_embeddings[k]),
                )
                _largest = _community_embeddings[_largest_cid][:100]
                if len(_largest) >= 2:
                    _hubness = detect_hubness(_largest, threshold=0.85)
                    write_event(
                        self._store,
                        "community_hubness_diagnostic",
                        {
                            "community_id": _largest_cid,
                            "mean_similarity": float(
                                _hubness.get("mean_similarity", 0.0)
                            ),
                            "max_similarity": float(
                                _hubness.get("max_similarity", 0.0)
                            ),
                            "is_hub": bool(_hubness.get("is_hub", False)),
                            "size": int(_hubness.get("size", 0)),
                        },
                        severity="info",
                    )
        except Exception as _hub_exc:  # noqa: BLE001 -- diagnostic MUST NOT crash sleep
            logger.debug(
                "detect_hubness diagnostic skipped: %s",
                str(_hub_exc)[:120],
            )
