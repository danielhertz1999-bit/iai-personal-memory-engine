from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytest


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def _settings_path(home: Path) -> Path:
    return home / ".claude" / "settings.json"


def test_install_adds_sessionstart_entry_idempotent(fake_home):
    from iai_mcp import cli as cli_mod

    rc1 = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc1 == 0
    rc2 = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc2 == 0

    data = json.loads(_settings_path(fake_home).read_text())
    ss_entries = data.get("hooks", {}).get("SessionStart", [])
    matching = [
        e for e in ss_entries
        if any("iai-mcp-session-recall.sh" in (h.get("command") or "")
               for h in (e.get("hooks") or []))
    ]
    assert len(matching) == 1, ss_entries
    entry = matching[0]
    assert entry.get("matcher") == "startup|resume|clear|compact", entry
    cmd = entry["hooks"][0]["command"]
    assert re.search(r"bash .*iai-mcp-session-recall\.sh", cmd), cmd

    stop_entries = data.get("hooks", {}).get("Stop", [])
    assert any(
        "iai-mcp-session-capture.sh" in (h.get("command") or "")
        for e in stop_entries for h in (e.get("hooks") or [])
    ), stop_entries

    assert (fake_home / ".claude" / "hooks" / "iai-mcp-session-recall.sh").exists()


def test_uninstall_removes_sessionstart_entry_and_script(fake_home):
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())

    recall_dst = fake_home / ".claude" / "hooks" / "iai-mcp-session-recall.sh"
    assert recall_dst.exists(), "install did not copy recall hook"
    data = json.loads(_settings_path(fake_home).read_text())
    ss = data.get("hooks", {}).get("SessionStart", [])
    assert any(
        "iai-mcp-session-recall.sh" in (h.get("command") or "")
        for e in ss for h in (e.get("hooks") or [])
    ), ss

    rc = cli_mod.cmd_capture_hooks_uninstall(argparse.Namespace())
    assert rc == 0

    settings = _settings_path(fake_home)
    if settings.exists():
        data = json.loads(settings.read_text())
        ss = data.get("hooks", {}).get("SessionStart", [])
        for e in ss:
            for h in (e.get("hooks") or []):
                assert "iai-mcp-session-recall.sh" not in (h.get("command") or "")

    assert not recall_dst.exists()


def test_status_reports_session_recall_alongside_stop(fake_home, capsys):
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    capsys.readouterr()
    cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "iai-mcp-session-recall.sh" in out, out
    assert "iai-mcp-session-capture.sh" in out, out
