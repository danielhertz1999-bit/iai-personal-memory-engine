"""Regression guards for the 2026-06-21 tombstone-handling remediation:

- the runtime-graph node filter runs on the cache-MISS pandas path (cache neutralised);
- a LIVE record on a datetime64/NaT tombstoned_at column is NOT dropped (pd.isna guard);
- an edge live->tombstoned does not resurrect the dead endpoint (has_node guard);
- build_live_graph (crisis hooks) excludes tombstoned, matching active_records_count().

Each test fails if its corresponding fix is reverted.
"""
from __future__ import annotations

import pandas as pd
import pytest

import iai_mcp.retrieve as retrieve
import iai_mcp.runtime_graph_cache as runtime_graph_cache
from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer

from tests.test_graph_excludes_tombstoned import _make_store, _rec, _tombstone


@pytest.fixture
def _no_graph_cache(monkeypatch):
    """Force build_runtime_graph to MISS the payload cache so the live pandas
    node/edge-skip loop is always exercised (never a cheap cache reload)."""
    monkeypatch.setattr(runtime_graph_cache, "try_load", lambda *_a, **_k: None)


def test_node_skip_runs_on_cache_miss(_no_graph_cache, tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    live = []
    for i in range(5):
        rid, rec = _rec(i)
        store.insert(rec)
        live.append(rid)
    dead_id, dead = _rec(77)
    store.insert(dead)
    _tombstone(store, dead_id)

    assert store.active_records_count() == 5
    graph, _a, _rc = retrieve.build_runtime_graph(store)
    nodes = {str(n) for n in graph.nodes()}
    assert nodes == {str(r) for r in live}
    assert str(dead_id) not in nodes


def test_live_record_survives_datetime64_nat_column(_no_graph_cache, tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    live = []
    for i in range(5):
        rid, rec = _rec(i)
        store.insert(rec)
        live.append(rid)
    flush_record_buffer(store)
    assert store.active_records_count() == 5

    records_tbl = store.db.open_table("records")
    table_cls = type(records_tbl)
    original_to_pandas = table_cls.to_pandas

    def _coerce(self, *a, **k):
        df = original_to_pandas(self, *a, **k)
        if "tombstoned_at" in df.columns:
            df = df.copy()
            df["tombstoned_at"] = pd.to_datetime(df["tombstoned_at"], utc=True)
        return df

    monkeypatch.setattr(table_cls, "to_pandas", _coerce)
    coerced = store.db.open_table("records").to_pandas()
    assert str(coerced["tombstoned_at"].dtype).startswith("datetime64")
    assert all(pd.isna(v) for v in coerced["tombstoned_at"])

    graph, _a, _rc = retrieve.build_runtime_graph(store)
    nodes = {str(n) for n in graph.nodes()}
    assert nodes == {str(r) for r in live}, (
        "live records dropped on a datetime64/NaT tombstoned_at column"
    )


def test_edge_to_tombstoned_dst_is_skipped(_no_graph_cache, tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    ids = []
    for i in range(5):
        rid, rec = _rec(i)
        store.insert(rec)
        ids.append(rid)
    flush_record_buffer(store)

    src, dst = ids[0], ids[4]
    store.boost_edges([(src, dst)], delta=1.0, edge_type="hebbian")
    flush_edge_buffer(store)
    _tombstone(store, dst)
    assert store.active_records_count() == 4

    graph, _a, _rc = retrieve.build_runtime_graph(store)
    nodes = {str(n) for n in graph.nodes()}
    assert str(dst) not in nodes, "tombstoned edge endpoint leaked back as a node"
    assert len(nodes) == 4


def test_build_live_graph_excludes_tombstoned(tmp_path, monkeypatch):
    from iai_mcp.lilli.cycle.sleep_pipeline._live_graph import build_live_graph

    store = _make_store(tmp_path, monkeypatch)
    live = []
    for i in range(5):
        rid, rec = _rec(i)
        store.insert(rec)
        live.append(rid)
    dead_id, dead = _rec(88)
    store.insert(dead)
    _tombstone(store, dead_id)
    flush_record_buffer(store)

    g = build_live_graph(store)
    nodes = {str(n) for n in g.nodes()}
    assert str(dead_id) not in nodes
    assert g.node_count() == store.active_records_count() == 5
