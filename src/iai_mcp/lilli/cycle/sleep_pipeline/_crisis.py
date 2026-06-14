from __future__ import annotations

import json
import logging
from typing import Any, Callable

from iai_mcp.exceptions import StoreError
from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_crisis_recluster(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    if self._check_interrupt(
        SleepStep.CRISIS_RECLUSTER, 0, interrupt_check,
    ):
        return False, {}

    state_rec = self._load_state_record()
    if not state_rec.get("crisis_mode", False):
        return True, {"communities_dropped": 0, "crisis_mode": False}

    from iai_mcp.daemon_config import _load_sleep_overhaul_config
    cfg = _load_sleep_overhaul_config()
    drop_quartile = cfg.crisis_drop_quartile
    dry_run = cfg.dry_run

    from iai_mcp.events import write_event
    from iai_mcp.store import RECORDS_TABLE
    tbl = self._store.db.open_table(RECORDS_TABLE)

    try:
        df = tbl.search().to_pandas()
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("crisis_recluster records query failed: %s", exc)
        df = None

    communities_dropped = 0
    records_reassigned = 0
    new_community_count = 0
    modularity = 0.0
    backend = "flat"

    if df is not None and not df.empty and "community_id" in df.columns:
        non_null = df[df["community_id"].notna()]
        if not non_null.empty:
            sizes = (
                non_null.groupby("community_id").size().sort_values()
            )
            total_communities = len(sizes)
            n_to_drop = int(total_communities * drop_quartile)
            drop_ids = list(sizes.index[:n_to_drop])
            communities_dropped = n_to_drop

            if drop_ids and not dry_run:
                for cid in drop_ids:
                    try:
                        tbl.update(
                            where=f"community_id = '{str(cid)}'",
                            values={"community_id": None},
                        )
                    except (OSError, ValueError, RuntimeError, StoreError):
                        pass

            if not dry_run:
                tbl = self._store.db.open_table(RECORDS_TABLE)
                try:
                    df2 = tbl.search().to_pandas()
                except (OSError, ValueError, RuntimeError, StoreError):
                    df2 = df

                try:
                    from iai_mcp.community import detect_communities
                    from iai_mcp.graph import MemoryGraph
                    from iai_mcp.store import EDGES_TABLE
                    import uuid as _uuid

                    g = MemoryGraph()
                    for _, row in df2.iterrows():
                        try:
                            rid = _uuid.UUID(str(row["id"]))
                            emb = row.get("embedding")
                            emb_list = (
                                list(emb) if emb is not None else []
                            )
                            g.add_node(rid, None, emb_list)
                        except (ValueError, TypeError, AttributeError):
                            continue

                    try:
                        edges_df = (
                            self._store.db.open_table(EDGES_TABLE)
                            .search()
                            .to_pandas()
                        )
                        for _, e in edges_df.iterrows():
                            try:
                                src_u = _uuid.UUID(str(e["src"]))
                                dst_u = _uuid.UUID(str(e["dst"]))
                                g.add_edge(
                                    src_u, dst_u,
                                    weight=float(
                                        e.get("weight", 1.0) or 1.0
                                    ),
                                )
                            except (ValueError, TypeError, KeyError):
                                continue
                    except (OSError, ValueError, RuntimeError, StoreError) as exc:
                        logger.debug("crisis_recluster edges query failed: %s", exc)

                    _assignment = detect_communities(
                        g, prior=None, prior_mode="cold"
                    )
                    modularity = float(_assignment.modularity)
                    backend = _assignment.backend
                    _uuid_to_int: dict[_uuid.UUID, int] = {}
                    _next_int = 0
                    partition: dict[_uuid.UUID, int] = {}
                    for _node_uuid, _comm_uuid in _assignment.node_to_community.items():
                        if _comm_uuid not in _uuid_to_int:
                            _uuid_to_int[_comm_uuid] = _next_int
                            _next_int += 1
                        partition[_node_uuid] = _uuid_to_int[_comm_uuid]
                    new_uuids: dict[int, str] = {}
                    for node, lbl in partition.items():
                        if lbl not in new_uuids:
                            new_uuids[lbl] = str(_uuid.uuid4())
                        new_cid = new_uuids[lbl]
                        try:
                            tbl.update(
                                where=f"id = '{str(node)}'",
                                values={"community_id": new_cid},
                            )
                            records_reassigned += 1
                        except (OSError, ValueError, RuntimeError, StoreError):
                            continue
                    new_community_count = len(new_uuids)
                except Exception as exc:  # noqa: BLE001 -- Leiden/graph rebuild
                    logger.warning("crisis_recluster Leiden rebuild failed: %s", exc, exc_info=True)

    if not dry_run:
        cleared = self._clear_crisis_mode_via_s2_or_fallback(
            reason="crisis_recluster_complete",
        )
        if not cleared:
            try:
                rec = self._load_state_record()
                rec["crisis_mode"] = False
                self._save_state_record(rec)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("crisis_mode clear last-resort write failed: %s", exc)

    write_event(
        self._store,
        "crisis_recluster_pass",
        {
            "communities_dropped": int(communities_dropped),
            "records_reassigned": int(records_reassigned),
            "new_community_count": int(new_community_count),
            "modularity": float(modularity),
            "backend": str(backend),
            "dry_run_mode": bool(dry_run),
        },
        severity="warning" if communities_dropped > 0 else "info",
    )

    return True, {
        "communities_dropped": int(communities_dropped),
        "dry_run": bool(dry_run),
    }
