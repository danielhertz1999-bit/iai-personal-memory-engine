"""Tests for core._inject_overnight_digest -- (DAEMON-11).

Covers 5 behaviours:
1. First memory_recall of the day (>18h since last shown) gets overnight_digest.
2. Second recall within <18h does NOT include overnight_digest.
3. Empty state / no pending digest -> no overnight_digest key.
4. Digest is cleared from state after one delivery (D-24 once-per-window).
5. Exception in get_pending_digest does NOT break memory_recall (silent fail).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    from iai_mcp import daemon_state
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    return state_path


# digest shape -- every required field populated.
_FULL_DIGEST = {
    "rem_cycles_completed": 4,
    "episodes_processed": 10,
    "schemas_induced_tier0": 3,
    "claude_call_used": True,
    "quota_used_pct": 0.003,
    "main_insight_text": "today's unifying insight",
    "sigma_observed": 1.2,
    "s5_drift_alerts": [],
    "daemon_uptime_hours": 8,
    "timed_out_cycles": 0,
}


# ---------------------------------------------------------------------------
# Test 1: first recall of day gets digest
# ---------------------------------------------------------------------------


def test_first_recall_gets_digest(isolated_state):
    from iai_mcp.core import _inject_overnight_digest
    from iai_mcp.daemon_state import save_state

    # Seed state: pending digest + last shown 20h ago (past the 18h threshold).
    now = datetime.now(timezone.utc)
    save_state({
        "pending_digest": dict(_FULL_DIGEST),
        "last_digest_shown_at": (now - timedelta(hours=20)).isoformat(),
    })

    response: dict = {"hits": [], "anti_hits": [], "activation_trace": [], "budget_used": 0}
    _inject_overnight_digest(response)

    assert "overnight_digest" in response
    dig = response["overnight_digest"]
    # required fields surface.
    assert dig["rem_cycles_completed"] == 4
    assert dig["episodes_processed"] == 10
    assert dig["schemas_induced_tier0"] == 3
    assert dig["claude_call_used"] is True
    assert dig["quota_used_pct"] == 0.003
    assert dig["main_insight_text"] == "today's unifying insight"
    assert dig["sigma_observed"] == 1.2
    assert dig["s5_drift_alerts"] == []
    assert dig["daemon_uptime_hours"] == 8


# ---------------------------------------------------------------------------
# Test 2: second recall within 18h window does NOT include digest
# ---------------------------------------------------------------------------


def test_not_twice(isolated_state):
    """the same digest must not appear twice inside the 18h window."""
    from iai_mcp.core import _inject_overnight_digest
    from iai_mcp.daemon_state import save_state

    now = datetime.now(timezone.utc)
    # last_shown 4h ago -- inside the window.
    save_state({
        "pending_digest": dict(_FULL_DIGEST),
        "last_digest_shown_at": (now - timedelta(hours=4)).isoformat(),
    })

    response: dict = {"hits": []}
    _inject_overnight_digest(response)
    assert "overnight_digest" not in response


# ---------------------------------------------------------------------------
# Test 3: no pending digest -> no key added
# ---------------------------------------------------------------------------


def test_no_digest_when_none_pending(isolated_state):
    from iai_mcp.core import _inject_overnight_digest
    from iai_mcp.daemon_state import save_state

    save_state({})  # empty state
    response: dict = {"hits": []}
    _inject_overnight_digest(response)
    assert "overnight_digest" not in response


# ---------------------------------------------------------------------------
# Test 4: digest cleared from state after one delivery
# ---------------------------------------------------------------------------


def test_digest_cleared_after_delivery(isolated_state):
    """after surfacing the digest, state must no longer carry
    pending_digest so a subsequent recall (even after another 18h) does not
    re-show the stale digest."""
    from iai_mcp.core import _inject_overnight_digest
    from iai_mcp.daemon_state import load_state, save_state

    now = datetime.now(timezone.utc)
    save_state({
        "pending_digest": dict(_FULL_DIGEST),
        "last_digest_shown_at": (now - timedelta(hours=20)).isoformat(),
    })

    response: dict = {"hits": []}
    _inject_overnight_digest(response)
    assert "overnight_digest" in response

    # Persisted state: pending_digest consumed.
    on_disk = load_state()
    assert "pending_digest" not in on_disk
    # last_digest_shown_at advanced to roughly now.
    shown_at = datetime.fromisoformat(on_disk["last_digest_shown_at"])
    if shown_at.tzinfo is None:
        shown_at = shown_at.replace(tzinfo=timezone.utc)
    assert shown_at >= now - timedelta(seconds=5)


# ---------------------------------------------------------------------------
# Test 5: exception in get_pending_digest does NOT break memory_recall
# ---------------------------------------------------------------------------


def test_exception_is_silent(isolated_state, monkeypatch):
    """If get_pending_digest raises (corrupt state, unexpected schema), the
    response must still be returned without an overnight_digest key. The
    memory_recall hot path NEVER breaks on daemon-digest faults."""
    from iai_mcp import core

    def boom(*args, **kwargs):
        raise RuntimeError("simulated state corruption")

    monkeypatch.setattr("iai_mcp.core.get_pending_digest", boom)

    response: dict = {"hits": [], "existing": True}
    # Must not raise.
    core._inject_overnight_digest(response)
    assert response.get("existing") is True
    assert "overnight_digest" not in response
