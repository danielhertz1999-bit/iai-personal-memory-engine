from __future__ import annotations

import platform
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="daemon module is POSIX-only on this project",
)


def _states():
    from iai_mcp.lifecycle_state import LifecycleState
    return LifecycleState


def test_drowsy_transition_triggers_drain_once():
    from iai_mcp.daemon import _should_drain_on_drowsy_edge
    L = _states()

    assert _should_drain_on_drowsy_edge(L.WAKE, L.DROWSY) is True
    assert _should_drain_on_drowsy_edge(L.DROWSY, L.DROWSY) is False


def test_subsequent_ticks_in_drowsy_do_not_redrain():
    from iai_mcp.daemon import _should_drain_on_drowsy_edge
    L = _states()

    prev = L.WAKE
    cur = L.DROWSY
    edges = []
    for _ in range(5):
        if _should_drain_on_drowsy_edge(prev, cur):
            edges.append(1)
        prev = cur
        cur = L.DROWSY

    assert sum(edges) == 1, edges


def test_wake_to_drowsy_to_wake_to_drowsy_drains_twice():
    from iai_mcp.daemon import _should_drain_on_drowsy_edge
    L = _states()

    trajectory = [L.WAKE, L.DROWSY, L.DROWSY, L.WAKE, L.DROWSY, L.DROWSY]
    triggers = 0
    prev = trajectory[0]
    for cur in trajectory[1:]:
        if _should_drain_on_drowsy_edge(prev, cur):
            triggers += 1
        prev = cur

    assert triggers == 2


def test_drain_failure_does_not_crash_helper():
    from iai_mcp.daemon import _run_drowsy_drain

    events: list[tuple] = []

    def write_event(store, kind, data, severity="info"):
        events.append((kind, data, severity))

    def failing_drain(store):
        raise RuntimeError("drain blew up")

    fake_store = SimpleNamespace()
    _run_drowsy_drain(fake_store, drain_fn=failing_drain, write_event_fn=write_event)

    kinds = [e[0] for e in events]
    assert "deferred_drain_failed" in kinds, events
    failed = next(e for e in events if e[0] == "deferred_drain_failed")
    assert failed[1].get("phase") == "drowsy", failed


def test_drain_success_writes_drowsy_event_when_files_processed():
    from iai_mcp.daemon import _run_drowsy_drain

    events: list[tuple] = []

    def write_event(store, kind, data, severity="info"):
        events.append((kind, data, severity))

    def good_drain(store):
        return {
            "files_drained": 2,
            "files_failed": 0,
            "events_inserted": 7,
            "events_reinforced": 0,
            "events_skipped_intentional": 0,
            "events_skipped_insert_failed": 0,
        }

    _run_drowsy_drain(SimpleNamespace(), drain_fn=good_drain, write_event_fn=write_event)

    kinds = [e[0] for e in events]
    assert "deferred_drain_drowsy" in kinds, events
    assert "deferred_drain_failed" not in kinds


def test_drain_zero_work_is_quiet():
    from iai_mcp.daemon import _run_drowsy_drain

    events: list[tuple] = []

    def write_event(store, kind, data, severity="info"):
        events.append((kind, data, severity))

    def empty_drain(store):
        return {
            "files_drained": 0,
            "files_failed": 0,
            "events_inserted": 0,
            "events_reinforced": 0,
            "events_skipped_intentional": 0,
            "events_skipped_insert_failed": 0,
        }

    _run_drowsy_drain(SimpleNamespace(), drain_fn=empty_drain, write_event_fn=write_event)

    assert events == []
