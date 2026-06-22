"""Regression tests for excluding tombstoned records from the runtime graph.

build_runtime_graph used to add every record (and every edge) to the graph,
including soft-deleted / deduped / erased records (tombstoned_at IS NOT NULL).
That (a) polluted communities / centrality / rich_club / the sigma topology
audit with dead nodes, and (b) desynced the node count from
store.active_records_count() -- the cache-validity anchor -- so the payload
cache was permanently invalid and every wake did a full rebuild.

These tests pin: tombstoned records (and edges touching them) are excluded, the
live node count equals active_records_count(), and assignment/rich_club are
recomputed on the fresh live graph rather than reused from a stale-node cache.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

import iai_mcp.retrieve as retrieve
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _rec(seed: int):
    now = datetime.now(timezone.utc)
    rid = uuid4()
    return rid, MemoryRecord(
        id=rid, tier="episodic", literal_surface=f"rec-{seed}", aaak_index="",
        embedding=_vec(seed), community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=0.0, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False, provenance=[], created_at=now,
        updated_at=now, tags=[], language="en",
    )


def _make_store(tmp_path: Path, monkeypatch) -> MemoryStore:
    root = tmp_path / "store"
    monkeypatch.setenv("IAI_MCP_STORE", str(root))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    return MemoryStore(path=root)


def _tombstone(store: MemoryStore, rid) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with store.db._conn_lock:
        store.db._conn.execute(
            "UPDATE records SET tombstoned_at = ? WHERE id = ?",
            (now, str(rid)),
        )


def test_build_runtime_graph_excludes_tombstoned(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    live_ids = []
    for i in range(6):
        rid, rec = _rec(i)
        store.insert(rec)
        live_ids.append(rid)
    dead_ids = []
    for i in range(6, 10):
        rid, rec = _rec(i)
        store.insert(rec)
        _tombstone(store, rid)
        dead_ids.append(rid)

    assert store.active_records_count() == 6

    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    nodes = {str(n) for n in graph.nodes()}
    assert len(nodes) == 6, f"expected 6 live nodes, got {len(nodes)}"
    for rid in dead_ids:
        assert str(rid) not in nodes, f"tombstoned {rid} leaked into the graph"
    for rid in live_ids:
        assert str(rid) in nodes
    # rich_club is a fraction of the LIVE graph, never references dead nodes
    assert all(str(r) not in {str(d) for d in dead_ids} for r in (rich_club or []))


def test_node_count_matches_active_records_count(tmp_path, monkeypatch):
    """The cache-validity anchor: graph node count must equal active count, so the
    payload cache validates on the next build instead of rebuilding forever."""
    store = _make_store(tmp_path, monkeypatch)
    for i in range(8):
        _rid, rec = _rec(i)
        store.insert(rec)
    rid, rec = _rec(99)
    store.insert(rec)
    _tombstone(store, rid)

    graph, _assignment, _rc = retrieve.build_runtime_graph(store)
    assert len({str(n) for n in graph.nodes()}) == store.active_records_count() == 8
