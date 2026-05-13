"""Tests for iai-mcp audit CLI .

`iai-mcp audit [--since WEEKS] [--severity SEV]` renders an identity-event
audit log, TZ-aware timestamps, and REDACTED shield match counts (D-30
threat T-02-05-02: leaking matched patterns in CLI output would hand the
attacker a dictionary of what the shield is watching for).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from iai_mcp.cli import main as cli_main
from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore


def test_cli_audit_empty(tmp_path, capsys, monkeypatch):
    """No identity events -> 'No identity events recorded' message, exit 0."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    code = cli_main(["audit"])
    assert code == 0
    out = capsys.readouterr().out
    assert (
        "no identity events" in out.lower()
        or "no events" in out.lower()
    )


def test_cli_audit_renders_events(tmp_path, capsys, monkeypatch):
    """Pre-populated events render with kind + ts (in user TZ) + severity."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="s5_invariant_update",
        data={"anchor_id": "abc", "new_record_id": "def"},
        severity="info", session_id="s1",
    )
    code = cli_main(["audit"])
    assert code == 0
    out = capsys.readouterr().out
    # Kind appears.
    assert "s5_invariant_update" in out
    # Severity visible.
    assert "info" in out


def test_cli_audit_since_weeks(tmp_path, capsys, monkeypatch):
    """`audit --since=2` filters to 2-week window without crashing."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="s5_invariant_update",
        data={"anchor_id": "abc"},
        severity="info", session_id="s1",
    )
    code = cli_main(["audit", "--since=2"])
    assert code == 0


def test_cli_audit_severity_filter_warning_only(tmp_path, capsys, monkeypatch):
    """`--severity=warning` filters out info-severity events."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="s5_invariant_update",
        data={"anchor_id": "abc"},
        severity="info", session_id="s1",
    )
    write_event(
        store, kind="s5_drift_alert",
        data={"first_value": 0.1, "last_value": 0.5},
        severity="warning", session_id="s2",
    )
    code = cli_main(["audit", "--severity=warning"])
    assert code == 0
    out = capsys.readouterr().out
    # Warning event mentioned; info event NOT.
    assert "s5_drift_alert" in out
    assert "s5_invariant_update" not in out


def test_cli_audit_shows_shield_rejections_redacted(tmp_path, capsys, monkeypatch):
    """shield_rejection events appear but matched patterns are redacted to
    count only (not the literal words)."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="shield_rejection",
        data={
            "tier": "hard_block",
            "matched": ["forget", "you are now", "override"],
            "record_id": "aabbcc",
            "action": "reject",
        },
        severity="critical", session_id="s1",
    )
    code = cli_main(["audit"])
    assert code == 0
    out = capsys.readouterr().out
    # kind visible.
    assert "shield_rejection" in out
    # matched COUNT visible (3 patterns).
    assert "3" in out or "matched_count=3" in out.replace(" ", "")
    # Literal signal words MUST NOT appear (redaction).
    assert "forget" not in out
    assert "you are now" not in out


# ---------------------------------------------------------------- subcommands


def test_cli_audit_shield_subcommand(tmp_path, capsys, monkeypatch):
    """`iai-mcp audit shield --since=7` returns shield events."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="shield_rejection",
        data={"tier": "hard_block", "matched": ["forget"], "action": "reject"},
        severity="critical", session_id="s1",
    )
    # Exercise the subcommand; no crash is the contract.
    code = cli_main(["audit", "shield", "--since=7"])
    assert code == 0


def test_cli_audit_drift_subcommand(tmp_path, capsys, monkeypatch):
    """`iai-mcp audit drift` runs detection + surfaces present alert."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    # Seed monotonically increasing M4 variance to trigger drift.
    for i, v in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
        write_event(
            store, kind="trajectory_metric",
            data={"metric": "m4", "value": v},
            severity="info", session_id=f"s{i}",
        )
    code = cli_main(["audit", "drift"])
    assert code == 0
    out = capsys.readouterr().out
    # Drift detected and surfaced.
    assert "drift" in out.lower()


def test_cli_audit_identity_subcommand(tmp_path, capsys, monkeypatch):
    """`iai-mcp audit identity` shows only s5_* events."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="s5_invariant_update",
        data={"anchor_id": "abc"},
        severity="info", session_id="s1",
    )
    write_event(
        store, kind="shield_rejection",
        data={"tier": "hard_block", "matched": ["forget"], "action": "reject"},
        severity="critical", session_id="s2",
    )
    code = cli_main(["audit", "identity"])
    assert code == 0
    out = capsys.readouterr().out
    # s5 event present; shield_rejection filtered out.
    assert "s5_invariant_update" in out
    assert "shield_rejection" not in out
