"""REQ-7: `iai last` daemon-down live-only fallback.

All tests use a tmp HOME and a dead IAI_DAEMON_SOCKET_PATH so the
real _send_jsonrpc_request returns None via connect-failure — the same
code path that fires in production when the daemon socket is absent or
unreachable. No test contacts the live daemon.
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import sys
from pathlib import Path

import pytest


SID = "last-dd-test-abc"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_live_file(home: Path, session_id: str, events: list[dict]) -> Path:
    """Write a.live.jsonl file with the given event dicts under tmp HOME."""
    deferred = home / ".iai-mcp" / ".deferred-captures"
    deferred.mkdir(parents=True, exist_ok=True)
    ts_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    live = deferred / f"{session_id}.live.jsonl"
    with live.open("w") as fh:
        fh.write(json.dumps({"version": 1, "session_id": session_id}) + "\n")
        for ev in events:
            ev_full = {
                "ts": ts_now,
                "tier": "episodic",
                "cue": f"session {session_id} turn",
            }
            ev_full.update(ev)
            fh.write(json.dumps(ev_full) + "\n")
    return live


def _run_cmd_last(n: int, session: str | None, monkeypatch, tmp_path, capture_stdout=True):
    """Call cmd_last with an isolated tmp HOME and a dead socket path.

    Returns (rc, stdout_text).
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "nonexistent.sock"))
    monkeypatch.setenv("NO_COLOR", "1")  # disable ANSI codes for clean assertions

    args = argparse.Namespace(n=n, session=session)

    from iai_mcp.iai_cli import cmd_last

    if capture_stdout:
        buf = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        rc = cmd_last(args)
        return rc, buf.getvalue()
    else:
        rc = cmd_last(args)
        return rc, ""


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_cmd_last_daemon_down_returns_live_user_turn(tmp_path, monkeypatch):
    """Socket dead + a user turn in live.jsonl -> rc 0 and turn text in stdout."""
    _write_live_file(tmp_path, SID, [
        {"text": "hello live fallback world", "role": "user"},
    ])
    rc, out = _run_cmd_last(n=5, session=SID, monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert rc == 0
    assert "hello live fallback world" in out


def test_cmd_last_daemon_down_role_filter(tmp_path, monkeypatch):
    """Assistant events in the live file do NOT appear in stdout (role filter)."""
    _write_live_file(tmp_path, SID, [
        {"text": "user text here", "role": "user"},
        {"text": "assistant reply here", "role": "assistant"},
    ])
    rc, out = _run_cmd_last(n=10, session=SID, monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert rc == 0
    assert "user text here" in out
    assert "assistant reply here" not in out


def test_cmd_last_daemon_down_empty_live_shows_no_turns_found(tmp_path, monkeypatch):
    """No live file (or empty) -> stdout contains the expected empty message, rc 0."""
    # Do not write any live file — the deferred dir may not even exist.
    rc, out = _run_cmd_last(n=5, session=SID, monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert rc == 0
    assert "(no user turns found)" in out


def test_cmd_last_daemon_down_n_zero_shows_empty(tmp_path, monkeypatch):
    """n=0 does not crash and yields the empty message (NOT all events).

    events[:0] always returns empty regardless of how many events exist,
    matching the daemon path where n=0 returns an empty turns list.
    """
    _write_live_file(tmp_path, SID, [
        {"text": "some user turn", "role": "user"},
    ])
    rc, out = _run_cmd_last(n=0, session=SID, monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert rc == 0
    assert "(no user turns found)" in out
    assert "some user turn" not in out


def test_cmd_last_daemon_down_format_matches_daemon_path(tmp_path, monkeypatch):
    """Output format matches the daemon path: [{ts_iso[:19]}] {sid[:8]}: {text[:120]}."""
    ts_iso = "2026-05-31T12:34:56+00:00"
    _write_live_file(tmp_path, SID, [
        {"text": "format check text", "role": "user", "ts": ts_iso},
    ])
    rc, out = _run_cmd_last(n=5, session=SID, monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert rc == 0
    # The line should contain the expected timestamp prefix and sid prefix.
    expected_ts = ts_iso[:19]
    expected_sid = SID[:8]
    assert expected_ts in out
    assert expected_sid in out
    assert "format check text" in out
    # Verify the format token: "[<ts>] <sid>:"
    assert f"[{expected_ts}] {expected_sid}:" in out
