from __future__ import annotations

import logging
import uuid as _uuid

import pandas as pd

from iai_mcp.exceptions import StoreError
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import EDGES_TABLE, RECORDS_TABLE

logger = logging.getLogger(__name__)


def _is_tombstoned(value) -> bool:
    """True if a records.tombstoned_at value marks a (soft-)deleted record.

    NULL is the LIVE case and pandas materialises it differently per dtype:
    None (object column), float('nan'), pd.NaT (a datetime64 column -- the shape
    the reembed Arrow schema produces) or pd.NA (nullable extension dtypes).
    pd.isna() covers all of them, so only a real, non-empty timestamp string
    marks a tombstone. Mirrors retrieve.py::_build_runtime_graph_impl so the
    crisis hooks compute topology on exactly the same live node set as recall.
    """
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def build_live_graph(store) -> MemoryGraph:
    """Build a MemoryGraph of LIVE records + live-only edges.

    Excludes tombstoned and embedding-pending records (matching
    store.active_records_count()), and drops any edge whose endpoint is not a
    live node -- graph.add_edge() does setdefault() on both endpoints, so an edge
    to a tombstoned record would re-create it as a payload-less node and re-bloat
    the graph. That pollution is exactly what drove rich_club below its floor and
    re-armed crisis on every sleep cycle. This mirrors the fix in retrieve.py
    (53f04f9) so the crisis hooks no longer diverge from recall's view of the
    graph. On a store error it returns an empty graph (callers already guard on
    node_count == 0).
    """
    g = MemoryGraph()
    try:
        recs = store.db.open_table(RECORDS_TABLE).search().to_pandas()
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("build_live_graph records query failed: %s", exc)
        return g
    if recs.empty:
        return g

    for _, row in recs.iterrows():
        try:
            if int(row.get("embedding_pending") or 0) != 0:
                continue
            if _is_tombstoned(row.get("tombstoned_at")):
                continue
            rid = _uuid.UUID(str(row["id"]))
            cid_raw = row.get("community_id")
            cid_uuid = None
            if cid_raw is not None and not _is_tombstoned(cid_raw) and str(cid_raw).strip():
                try:
                    cid_uuid = _uuid.UUID(str(cid_raw))
                except (ValueError, TypeError):
                    cid_uuid = None
            emb = row.get("embedding")
            emb_list = list(emb) if emb is not None else []
            g.add_node(rid, cid_uuid, emb_list)
        except (ValueError, TypeError, AttributeError):
            continue

    try:
        edges_df = store.db.open_table(EDGES_TABLE).search().to_pandas()
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("build_live_graph edges query failed: %s", exc)
        return g
    for _, e in edges_df.iterrows():
        try:
            src_s, dst_s = e["src"], e["dst"]
            # Both endpoints must already be live nodes; add_edge() setdefault
            # would otherwise resurrect a tombstoned endpoint as a phantom node.
            if not g.has_node(src_s) or not g.has_node(dst_s):
                continue
            g.add_edge(
                _uuid.UUID(str(src_s)),
                _uuid.UUID(str(dst_s)),
                weight=float(e.get("weight", 1.0) or 1.0),
            )
        except (ValueError, TypeError, KeyError):
            continue
    return g
