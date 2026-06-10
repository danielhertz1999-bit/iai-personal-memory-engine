from __future__ import annotations

import argparse
import uuid


def test_daemon_unreachable_empty_stdout_exit_zero(tmp_path, monkeypatch, capsys):
    from iai_mcp import cli as cli_mod

    bad_sock = tmp_path / f"iai-mcp-does-not-exist-{uuid.uuid4().hex}.sock"
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(bad_sock))

    rc = cli_mod.cmd_session_start(argparse.Namespace(session_id="-"))

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""
