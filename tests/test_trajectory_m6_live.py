from __future__ import annotations

import pytest

from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore
from iai_mcp.trajectory import m6_context_repeat_rate_live

def test_m6_zero_on_empty_store(tmp_path):
    store = MemoryStore(path=tmp_path)
    assert m6_context_repeat_rate_live(store) == 0.0

def test_m6_repeat_rate_three_repeats_in_ten(tmp_path):
    store = MemoryStore(path=tmp_path)
    distinct = [f"h{i}" for i in range(7)]
    for h in distinct:
        write_event(
            store, kind="session_started",
            data={"session_state_hash": h, "session_id": "s"},
            severity="info",
        )
    for _ in range(3):
        write_event(
            store, kind="session_started",
            data={"session_state_hash": "h0", "session_id": "s"},
            severity="info",
        )
    val = m6_context_repeat_rate_live(store)
    assert val == pytest.approx(0.3, abs=1e-6)

def test_m6_all_unique_returns_zero(tmp_path):
    store = MemoryStore(path=tmp_path)
    for i in range(5):
        write_event(
            store, kind="session_started",
            data={"session_state_hash": f"u{i}"},
            severity="info",
        )
    assert m6_context_repeat_rate_live(store) == 0.0

def test_m6_all_repeats_returns_high(tmp_path):
    store = MemoryStore(path=tmp_path)
    for _ in range(5):
        write_event(
            store, kind="session_started",
            data={"session_state_hash": "same"},
            severity="info",
        )
    val = m6_context_repeat_rate_live(store)
    assert val == pytest.approx(0.8, abs=1e-6)
