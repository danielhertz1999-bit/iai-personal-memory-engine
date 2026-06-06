"""Tests for iai-mcp trajectory CLI (Task 3).

The `trajectory` subcommand aggregates M1..M6 events via
trajectory.aggregate_trajectory and prints one summary line per metric.
Supports --since WEEKS to scope history.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from iai_mcp.cli import main as cli_main
from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore


def test_trajectory_empty_output(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    # No trajectory data recorded yet.
    code = cli_main(["trajectory"])
    assert code == 0
    out = capsys.readouterr().out
    assert "no trajectory data" in out.lower() or "no data" in out.lower()


def test_trajectory_renders_m1_to_m6(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    # Seed one event for each metric.
    for i, m in enumerate(["m1", "m2", "m3", "m4", "m5", "m6"]):
        write_event(
            store,
            kind="trajectory_metric",
            data={"metric": m, "value": float(i + 1)},
            severity="info",
            session_id="s1",
        )
    code = cli_main(["trajectory"])
    assert code == 0
    out = capsys.readouterr().out
    # Every metric mentioned (M1... M6 uppercase).
    for m in ("M1", "M2", "M3", "M4", "M5", "M6"):
        assert m in out


def test_trajectory_since_weeks_flag(tmp_path, capsys, monkeypatch):
    """--since=N accepts the flag without crashing. (Filter behaviour is
    tested at the trajectory.aggregate_trajectory level; the CLI contract
    here is: flag is recognised and 0 on success.)"""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="trajectory_metric",
        data={"metric": "m1", "value": 1.0},
        severity="info", session_id="s1",
    )
    code = cli_main(["trajectory", "--since=2"])
    assert code == 0


def test_trajectory_prints_aggregate_stats(tmp_path, capsys, monkeypatch):
    """Output for a populated M1 mentions min/max/mean."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    for v in (1.0, 2.0, 3.0):
        write_event(
            store, kind="trajectory_metric",
            data={"metric": "m1", "value": v},
            severity="info", session_id="s1",
        )
    code = cli_main(["trajectory"])
    assert code == 0
    out = capsys.readouterr().out
    # Some aggregate indicator visible.
    assert "mean" in out.lower() or "avg" in out.lower() or "=" in out
