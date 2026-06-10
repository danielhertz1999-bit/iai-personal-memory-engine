from __future__ import annotations

from unittest.mock import patch

import pytest

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    import keyring as _keyring

    fake_store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake_store.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake_store.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake_store.pop((s, u), None))
    yield fake_store

def test_monitor_no_store_no_op():
    from iai_mcp.lilli.ops.hebbian import monitor_similarity_window

    result = monitor_similarity_window(None, [0.5] * 20)
    assert result is None

def test_monitor_short_window_no_op(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.events import query_events
    from iai_mcp.lilli.ops.hebbian import monitor_similarity_window
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    try:
        monitor_similarity_window(store, [0.5] * 5)
        events = query_events(store, kind="rank_deficiency_warning", limit=10)
        assert len(events) == 0, f"Expected no events but got {len(events)}"
    finally:
        store.close()

def test_monitor_healthy_window_no_emit(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.events import query_events
    from iai_mcp.lilli.ops.hebbian import monitor_similarity_window
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    try:
        healthy_window = [0.3, 0.7, 0.5, 0.4, 0.6, 0.3, 0.8, 0.5, 0.4, 0.7]
        monitor_similarity_window(store, healthy_window)
        events = query_events(store, kind="rank_deficiency_warning", limit=10)
        assert len(events) == 0, f"Expected no events for healthy window but got {len(events)}"
    finally:
        store.close()

def test_monitor_collapsed_window_emits_telemetry(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.events import query_events
    from iai_mcp.lilli.ops.hebbian import monitor_similarity_window
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    try:
        collapsed_window = [0.5, 0.501, 0.499, 0.5, 0.5, 0.501, 0.499, 0.5, 0.5, 0.501]
        monitor_similarity_window(store, collapsed_window)

        events = query_events(store, kind="rank_deficiency_warning", limit=10)
        assert len(events) >= 1, "Expected at least one rank_deficiency_warning event"

        evt = events[0]
        data = evt["data"]
        assert data["window_size"] == 10
        assert data["stddev"] < 0.05
        assert evt["domain"] == "lilli.ops.hebbian.monitor_similarity_window"
    finally:
        store.close()

def test_monitor_threshold_argument_respected(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.events import query_events
    from iai_mcp.lilli.ops.hebbian import monitor_similarity_window
    from iai_mcp.store import MemoryStore

    collapsed_window = [0.5, 0.501, 0.499, 0.5, 0.5, 0.501, 0.499, 0.5, 0.5, 0.501]

    store = MemoryStore()
    try:
        monitor_similarity_window(store, collapsed_window, threshold=0.0)
        events_no_emit = query_events(store, kind="rank_deficiency_warning", limit=10)
        assert len(events_no_emit) == 0, (
            f"Expected no event at threshold=0.0 but got {len(events_no_emit)}"
        )
    finally:
        store.close()

    store2 = MemoryStore()
    try:
        monitor_similarity_window(store2, collapsed_window, threshold=0.1)
        events_emit = query_events(store2, kind="rank_deficiency_warning", limit=10)
        assert len(events_emit) >= 1, (
            f"Expected at least one event at threshold=0.1 but got {len(events_emit)}"
        )
    finally:
        store2.close()

def test_monitor_does_not_raise_on_events_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.lilli.ops.hebbian import monitor_similarity_window
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    try:
        collapsed_window = [0.5, 0.501, 0.499, 0.5, 0.5, 0.501, 0.499, 0.5, 0.5, 0.501]

        import iai_mcp.events as events_module

        with patch.object(events_module, "write_event", side_effect=RuntimeError("simulated failure")):
            monitor_similarity_window(store, collapsed_window)
    finally:
        store.close()
