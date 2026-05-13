"""Task 2 Step 5: M2 precision@5 LIVE tests.

Reads ``kind='retrieval_used'`` events emitted by retrieve.py / pipeline.py.
"""
from __future__ import annotations

import pytest

from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore
from iai_mcp.trajectory import m2_precision_at_5_live


def test_m2_returns_zero_on_empty_store(tmp_path):
    store = MemoryStore(path=tmp_path)
    assert m2_precision_at_5_live(store) == 0.0


def test_m2_with_ground_truth_precision(tmp_path):
    """5 events, ground_truth coverage 4/5 in top-5 -> precision 0.8."""
    store = MemoryStore(path=tmp_path)
    for _ in range(5):
        write_event(
            store,
            kind="retrieval_used",
            data={
                "hit_ids": ["a", "b", "c", "d", "e"],
                "ground_truth": ["a", "b", "c", "d", "x"],
            },
            severity="info",
            session_id="s1",
        )
    val = m2_precision_at_5_live(store)
    assert val == pytest.approx(0.8, abs=1e-6)


def test_m2_fallback_hit_presence_at_5(tmp_path):
    """Without ground_truth, value falls back to (events with hits) / total."""
    store = MemoryStore(path=tmp_path)
    # 4 events with hits, 1 empty
    for _ in range(4):
        write_event(
            store, kind="retrieval_used",
            data={"hit_ids": ["x"]}, severity="info", session_id="s",
        )
    write_event(
        store, kind="retrieval_used",
        data={"hit_ids": []}, severity="info", session_id="s",
    )
    val = m2_precision_at_5_live(store)
    assert val == pytest.approx(0.8, abs=1e-6)
