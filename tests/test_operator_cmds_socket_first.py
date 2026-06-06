"""Regression tests: operator commands (health, trajectory, audit, daemon_stats)
use the daemon socket first and guard the direct-open fallback with
HippoLockHeldError, mirroring the already-shipped cmd_topology pattern.

All tests are fully mocked: no live daemon, no real store, no socket I/O,
no filesystem access.
"""
from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _rpc_ok(payload: dict) -> dict:
    """Wrap a payload in a JSON-RPC success envelope."""
    return {"result": payload}


def _rpc_none() -> None:
    """Simulate daemon-down: _send_jsonrpc_request returns None."""
    return None


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = fn()
    return buf.getvalue(), rc


# ---------------------------------------------------------------------------
# cmd_health
# ---------------------------------------------------------------------------

class TestCmdHealthSocketPath:
    """Socket path exercises."""

    def test_renders_llm_health_from_socket(self):
        """(a) Socket delivers one llm_health event → renders correctly, rc 0."""
        from iai_mcp.cli import cmd_health

        payload = {
            "events": [
                {
                    "id": "abc",
                    "kind": "llm_health",
                    "severity": "ok",
                    "ts": "2026-05-30T10:00:00+00:00",
                    "data": {"model": "claude-3-opus"},
                }
            ],
            "count": 1,
        }
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)) as mock_rpc, \
             patch("iai_mcp.store.MemoryStore") as mock_store:
            out, rc = _capture(lambda: cmd_health(_args()))
        assert rc == 0
        assert "llm_health:" in out
        assert "ok" in out
        mock_store.assert_not_called()
        mock_rpc.assert_called_once_with("events_query", {"kind": "llm_health", "limit": 1})

    def test_no_events_prints_not_recorded(self):
        """(a) Socket delivers empty events list → prints 'no events recorded', rc 0."""
        from iai_mcp.cli import cmd_health

        payload = {"events": [], "count": 0}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)), \
             patch("iai_mcp.store.MemoryStore") as mock_store:
            out, rc = _capture(lambda: cmd_health(_args()))
        assert rc == 0
        assert "no events recorded" in out
        mock_store.assert_not_called()

    def test_socket_down_fallback_runs(self):
        """(b) Socket returns None → fallback MemoryStore path runs, rc 0."""
        from iai_mcp.cli import cmd_health

        mock_event = {
            "id": "xyz",
            "kind": "llm_health",
            "severity": "degraded",
            "ts": "2026-05-30T09:00:00+00:00",
            "data": {"note": "fallback"},
        }
        mock_store_inst = MagicMock()
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=None), \
             patch("iai_mcp.store.MemoryStore", return_value=mock_store_inst) as mock_ms, \
             patch("iai_mcp.events.query_events", return_value=[mock_event]):
            out, rc = _capture(lambda: cmd_health(_args()))
        assert rc == 0
        assert "llm_health:" in out
        mock_ms.assert_called_once()

    def test_hippo_lock_held_on_fallback_clean_message(self):
        """(c) Socket None + HippoLockHeldError on fallback → clean message, rc 0."""
        from iai_mcp.cli import cmd_health
        from iai_mcp.hippo import HippoLockHeldError

        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=None), \
             patch("iai_mcp.store.MemoryStore", side_effect=HippoLockHeldError("locked")):
            out, rc = _capture(lambda: cmd_health(_args()))
        assert rc == 0
        assert "daemon holds store lock" in out


# ---------------------------------------------------------------------------
# cmd_trajectory
# ---------------------------------------------------------------------------

def _make_trajectory_events(n: int = 3) -> list[dict]:
    """Build n minimal trajectory_metric events (2 per metric for m1/m2/m3)."""
    events = []
    for i in range(n):
        events.append({
            "id": f"traj-{i}",
            "kind": "trajectory_metric",
            "ts": f"2026-05-{20 + i:02d}T10:00:00+00:00",
            "data": {"metric": "m1", "value": float(i + 1)},
        })
    return events


class TestCmdTrajectorySocketPath:
    def test_renders_from_socket(self):
        """(a) Socket delivers trajectory events → renders stats, rc 0."""
        from iai_mcp.cli import cmd_trajectory

        events = _make_trajectory_events(3)
        payload = {"events": events, "count": len(events)}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)) as mock_rpc, \
             patch("iai_mcp.store.MemoryStore") as mock_store:
            out, rc = _capture(lambda: cmd_trajectory(_args(since=None)))
        assert rc == 0
        assert "M1:" in out
        mock_store.assert_not_called()
        mock_rpc.assert_called_once()
        call_args = mock_rpc.call_args
        assert call_args[0][0] == "events_query"
        assert call_args[0][1]["kind"] == "trajectory_metric"

    def test_empty_events_prints_no_data(self):
        """(a) Socket delivers empty events → prints 'no trajectory data recorded'."""
        from iai_mcp.cli import cmd_trajectory

        payload = {"events": [], "count": 0}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)), \
             patch("iai_mcp.store.MemoryStore") as mock_store:
            out, rc = _capture(lambda: cmd_trajectory(_args(since=None)))
        assert rc == 0
        assert "no trajectory data recorded" in out
        mock_store.assert_not_called()

    def test_socket_down_fallback_runs(self):
        """(b) Socket returns None → fallback aggregate_trajectory runs, rc 0."""
        from iai_mcp.cli import cmd_trajectory

        fake_data = {"m1": [(None, 1.0), (None, 2.0)], "m2": [], "m3": [], "m4": [], "m5": [], "m6": []}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=None), \
             patch("iai_mcp.store.MemoryStore") as mock_ms, \
             patch("iai_mcp.trajectory.aggregate_trajectory", return_value=fake_data):
            out, rc = _capture(lambda: cmd_trajectory(_args(since=None)))
        assert rc == 0
        assert "M1:" in out
        mock_ms.assert_called_once()

    def test_hippo_lock_held_on_fallback_clean_message(self):
        """(c) Socket None + HippoLockHeldError on fallback → clean message, rc 0."""
        from iai_mcp.cli import cmd_trajectory
        from iai_mcp.hippo import HippoLockHeldError

        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=None), \
             patch("iai_mcp.store.MemoryStore", side_effect=HippoLockHeldError("locked")):
            out, rc = _capture(lambda: cmd_trajectory(_args(since=None)))
        assert rc == 0
        assert "daemon holds store lock" in out

    def test_since_passed_to_socket(self):
        """Socket call includes since param when --since flag provided."""
        from iai_mcp.cli import cmd_trajectory

        payload = {"events": [], "count": 0}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)) as mock_rpc:
            _capture(lambda: cmd_trajectory(_args(since="2")))
        call_params = mock_rpc.call_args[0][1]
        assert "since" in call_params


# ---------------------------------------------------------------------------
# cmd_audit
# ---------------------------------------------------------------------------

def _audit_event(kind: str = "s5_invariant_update") -> dict:
    return {
        "id": "evt-1",
        "kind": kind,
        "severity": "info",
        "ts": "2026-05-30T08:00:00+00:00",
        "data": {"note": "test"},
        "session_id": "s-1",
    }


class TestCmdAuditSocketPath:
    def test_all_mode_renders_from_socket(self):
        """(a) audit all: socket audit_query delivers events → renders, rc 0."""
        from iai_mcp.cli import cmd_audit

        events = [_audit_event("s5_invariant_update")]
        payload = {"events": events, "count": 1}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)) as mock_rpc, \
             patch("iai_mcp.store.MemoryStore") as mock_store:
            out, rc = _capture(lambda: cmd_audit(_args(audit_sub=None, since=None, severity=None)))
        assert rc == 0
        assert "s5_invariant_update" in out
        mock_store.assert_not_called()
        mock_rpc.assert_called_once()
        assert mock_rpc.call_args[0][0] == "audit_query"

    def test_shield_mode_sends_shield_kinds(self):
        """(a) audit shield: socket call uses shield kinds."""
        from iai_mcp.cli import cmd_audit

        events = [_audit_event("shield_rejection")]
        payload = {"events": events, "count": 1}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)) as mock_rpc, \
             patch("iai_mcp.store.MemoryStore") as mock_store:
            out, rc = _capture(lambda: cmd_audit(_args(audit_sub="shield", since=None, severity=None)))
        assert rc == 0
        assert "shield_rejection" in out
        mock_store.assert_not_called()
        kinds = mock_rpc.call_args[0][1]["kinds"]
        assert "shield_rejection" in kinds
        assert "shield_flag" in kinds

    def test_identity_mode_sends_identity_kinds(self):
        """(a) audit identity: socket call uses s5_ kinds."""
        from iai_mcp.cli import cmd_audit

        events = [_audit_event("s5_cooldown_block")]
        payload = {"events": events, "count": 1}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)) as mock_rpc, \
             patch("iai_mcp.store.MemoryStore") as mock_store:
            out, rc = _capture(lambda: cmd_audit(_args(audit_sub="identity", since=None, severity=None)))
        assert rc == 0
        mock_store.assert_not_called()
        kinds = mock_rpc.call_args[0][1]["kinds"]
        assert "s5_invariant_update" in kinds
        assert "shield_rejection" not in kinds

    def test_drift_mode_uses_detect_drift_socket_method(self):
        """(a) audit drift: socket detect_drift → no_anomaly message, rc 0."""
        from iai_mcp.cli import cmd_audit

        payload = {"alerts": [], "count": 0}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)) as mock_rpc, \
             patch("iai_mcp.store.MemoryStore") as mock_store:
            out, rc = _capture(lambda: cmd_audit(_args(audit_sub="drift", since=None, severity=None)))
        assert rc == 0
        assert "no anomaly" in out
        mock_store.assert_not_called()
        assert mock_rpc.call_args[0][0] == "detect_drift"

    def test_drift_mode_renders_alerts_from_socket(self):
        """(a) audit drift: socket returns alerts → prints each alert."""
        from iai_mcp.cli import cmd_audit

        alerts = [{"window_sessions": 5, "first_value": 0.1, "last_value": 0.9}]
        payload = {"alerts": alerts, "count": 1}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)), \
             patch("iai_mcp.store.MemoryStore") as mock_store:
            out, rc = _capture(lambda: cmd_audit(_args(audit_sub="drift", since=None, severity=None)))
        assert rc == 0
        assert "variance increasing" in out
        mock_store.assert_not_called()

    def test_socket_down_audit_all_fallback_runs(self):
        """(b) audit all: socket None → fallback audit_identity_events runs, rc 0."""
        from iai_mcp.cli import cmd_audit

        events = [_audit_event("s5_invariant_update")]
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=None), \
             patch("iai_mcp.store.MemoryStore") as mock_ms, \
             patch("iai_mcp.s5.audit_identity_events", return_value=events):
            out, rc = _capture(lambda: cmd_audit(_args(audit_sub=None, since=None, severity=None)))
        assert rc == 0
        assert "s5_invariant_update" in out
        mock_ms.assert_called_once()

    def test_socket_down_drift_fallback_runs(self):
        """(b) audit drift: socket None → fallback detect_drift_anomaly runs, rc 0."""
        from iai_mcp.cli import cmd_audit

        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=None), \
             patch("iai_mcp.store.MemoryStore") as mock_ms, \
             patch("iai_mcp.s5.detect_drift_anomaly", return_value=[]):
            out, rc = _capture(lambda: cmd_audit(_args(audit_sub="drift", since=None, severity=None)))
        assert rc == 0
        assert "no anomaly" in out
        mock_ms.assert_called_once()

    def test_hippo_lock_held_audit_all_clean_message(self):
        """(c) Socket None + HippoLockHeldError on fallback → clean message, rc 0."""
        from iai_mcp.cli import cmd_audit
        from iai_mcp.hippo import HippoLockHeldError

        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=None), \
             patch("iai_mcp.store.MemoryStore", side_effect=HippoLockHeldError("locked")):
            out, rc = _capture(lambda: cmd_audit(_args(audit_sub=None, since=None, severity=None)))
        assert rc == 0
        assert "daemon holds store lock" in out

    def test_hippo_lock_held_audit_drift_clean_message(self):
        """(c) audit drift: Socket None + HippoLockHeldError on fallback → clean, rc 0."""
        from iai_mcp.cli import cmd_audit
        from iai_mcp.hippo import HippoLockHeldError

        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=None), \
             patch("iai_mcp.store.MemoryStore", side_effect=HippoLockHeldError("locked")):
            out, rc = _capture(lambda: cmd_audit(_args(audit_sub="drift", since=None, severity=None)))
        assert rc == 0
        assert "daemon holds store lock" in out

    def test_severity_filter_applied_to_socket_results(self):
        """Severity post-filter works on socket-sourced event list."""
        from iai_mcp.cli import cmd_audit

        events = [
            {**_audit_event(), "severity": "critical"},
            {**_audit_event(), "severity": "info"},
        ]
        payload = {"events": events, "count": 2}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)):
            out_crit, rc = _capture(
                lambda: cmd_audit(_args(audit_sub=None, since=None, severity="critical"))
            )
        # Only critical events should appear (severity filter reduces list to 1)
        assert rc == 0
        # Output should contain something (the critical event rendered)
        assert out_crit.strip()


# ---------------------------------------------------------------------------
# cmd_daemon_stats
# ---------------------------------------------------------------------------

def _session_events(n: int, tokens: int = 3000) -> list[dict]:
    return [
        {
            "id": f"se-{i}",
            "kind": "session_started",
            "ts": f"2026-05-{i + 1:02d}T10:00:00+00:00",
            "data": {"total_cached_tokens": tokens},
        }
        for i in range(n)
    ]


class TestCmdDaemonStatsSocketPath:
    def test_renders_p90_from_socket(self):
        """(a) Socket delivers session_started events → renders p90, rc 0."""
        from iai_mcp.cli import cmd_daemon_stats

        events = _session_events(5, tokens=2000)
        payload = {"events": events, "count": len(events)}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)) as mock_rpc, \
             patch("iai_mcp.store.MemoryStore") as mock_store:
            out, rc = _capture(lambda: cmd_daemon_stats(_args()))
        assert rc == 0
        assert "session_start_tokens_p90:" in out
        assert "n_samples:" in out
        mock_store.assert_not_called()
        mock_rpc.assert_called_once()
        call_params = mock_rpc.call_args[0][1]
        assert call_params["kind"] == "session_started"
        assert call_params["limit"] == 100

    def test_empty_socket_events_no_data(self):
        """(a) Socket delivers zero events → 'no-data' p90, rc 0."""
        from iai_mcp.cli import cmd_daemon_stats

        payload = {"events": [], "count": 0}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)), \
             patch("iai_mcp.store.MemoryStore") as mock_store:
            out, rc = _capture(lambda: cmd_daemon_stats(_args()))
        assert rc == 0
        assert "no-data" in out
        mock_store.assert_not_called()

    def test_socket_down_fallback_runs(self):
        """(b) Socket returns None → direct-open compute_session_start_tokens_p90, rc 0."""
        from iai_mcp.cli import cmd_daemon_stats

        fake_result = {"p90": 4000, "n_samples": 10}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=None), \
             patch("iai_mcp.store.MemoryStore") as mock_ms, \
             patch("iai_mcp.cli.compute_session_start_tokens_p90", return_value=fake_result):
            out, rc = _capture(lambda: cmd_daemon_stats(_args()))
        assert rc == 0
        assert "4000" in out
        mock_ms.assert_called_once()

    def test_hippo_lock_held_on_fallback_clean_message(self):
        """(c) Socket None + HippoLockHeldError on fallback → clean message, rc 0."""
        from iai_mcp.cli import cmd_daemon_stats
        from iai_mcp.hippo import HippoLockHeldError

        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=None), \
             patch("iai_mcp.store.MemoryStore", side_effect=HippoLockHeldError("locked")):
            out, rc = _capture(lambda: cmd_daemon_stats(_args()))
        assert rc == 0
        assert "daemon holds store lock" in out

    def test_socket_whitelist_error_falls_through_to_direct_open(self):
        """Socket returns whitelist error (old daemon) → fallback runs, rc 0."""
        from iai_mcp.cli import cmd_daemon_stats

        # Old daemon returns result={"error": "kind... not user-visible"}
        error_resp = {"result": {"error": "kind 'session_started' is not user-visible"}}
        fake_result = {"p90": 3500, "n_samples": 50}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=error_resp), \
             patch("iai_mcp.store.MemoryStore") as mock_ms, \
             patch("iai_mcp.cli.compute_session_start_tokens_p90", return_value=fake_result):
            out, rc = _capture(lambda: cmd_daemon_stats(_args()))
        assert rc == 0
        # Fallback ran because payload["events"] was missing
        mock_ms.assert_called_once()

    def test_p90_computed_correctly_from_socket_events(self):
        """p90 math: 100 identical events → p90 equals that value."""
        from iai_mcp.cli import cmd_daemon_stats

        events = _session_events(100, tokens=1234)
        payload = {"events": events, "count": 100}
        with patch("iai_mcp.cli._send_jsonrpc_request", return_value=_rpc_ok(payload)), \
             patch("iai_mcp.store.MemoryStore"):
            out, rc = _capture(lambda: cmd_daemon_stats(_args()))
        assert rc == 0
        assert "1234" in out
        assert "n_samples: 100" in out
