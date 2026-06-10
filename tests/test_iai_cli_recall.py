from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch, MagicMock

import pytest


def _make_args(cue: str = "test", limit: int = 5):
    import argparse
    ns = argparse.Namespace()
    ns.cue = cue
    ns.limit = limit
    return ns


def test_recall_daemon_hit_prints_hits(monkeypatch):
    from iai_mcp import iai_cli

    fake_resp = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "hits": [
                {"literal_surface": "first memory", "score": 0.95},
                {"literal_surface": "second memory", "score": 0.82},
            ]
        },
    }
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", lambda *a, **k: fake_resp)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_recall(_make_args(cue="test cue", limit=5))

    assert rc == 0
    out = buf.getvalue()
    assert "via daemon" in out
    assert "first memory" in out
    assert "second memory" in out
    assert "0.950" in out
    assert "0.820" in out


def test_recall_no_hits_returns_empty_marker(monkeypatch):
    from iai_mcp import iai_cli

    fake_resp = {"jsonrpc": "2.0", "id": 1, "result": {"hits": []}}
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", lambda *a, **k: fake_resp)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_recall(_make_args())

    assert rc == 0
    assert "no hits" in buf.getvalue().lower()


def test_recall_daemon_down_falls_back_to_bank(monkeypatch, tmp_path):
    from iai_mcp import iai_cli

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "nonexistent.sock"))

    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", lambda *a, **k: None)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        result = MagicMock()
        result.returncode = 0
        result.stdout = "fake bank hit 1\nfake bank hit 2\n"
        result.stderr = ""
        return result

    monkeypatch.setattr("subprocess.run", _fake_run)

    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = iai_cli.cmd_recall(_make_args(cue="when daemon dead", limit=3))

    assert rc == 0
    assert len(calls) == 1
    assert calls[0][0] == "iai-mcp"
    assert calls[0][1] == "bank-recall"
    assert "--query" in calls[0]
    assert "when daemon dead" in calls[0]
    assert "fake bank hit 1" in buf.getvalue()


def test_recall_bank_fallback_failure_returns_nonzero(monkeypatch, tmp_path):
    from iai_mcp import iai_cli

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "nonexistent.sock"))

    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", lambda *a, **k: None)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    def _fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 2
        result.stdout = ""
        result.stderr = "bank-recall: simulated failure\n"
        return result

    monkeypatch.setattr("subprocess.run", _fake_run)

    err = io.StringIO()
    with redirect_stderr(err):
        rc = iai_cli.cmd_recall(_make_args())

    assert rc == 1
    assert "simulated failure" in err.getvalue()


def test_recall_daemon_down_store_present_uses_direct_store(monkeypatch, tmp_path):
    from iai_mcp import iai_cli

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    store_dir = fake_home / ".iai-mcp" / "hippo"
    store_dir.mkdir(parents=True)
    (store_dir / "brain.sqlite3").write_bytes(b"")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "nonexistent.sock"))
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", lambda *a, **k: None)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    fake_hit = {"literal_surface": "store-backed memory hit", "score": 0.88, "_source": "direct-store"}
    monkeypatch.setattr("iai_mcp.semantic_recall.recall_semantic_warm", lambda *a, **kw: [fake_hit])

    subprocess_calls: list = []
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: subprocess_calls.append(a) or MagicMock(returncode=0, stdout="", stderr=""))

    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = iai_cli.cmd_recall(_make_args(cue="recall from store", limit=3))

    assert rc == 0
    assert len(subprocess_calls) == 0
    assert "store-backed memory hit" in buf.getvalue() or "store-backed memory hit" in err.getvalue() or "store recall" in err.getvalue()


def test_recall_respects_limit_arg(monkeypatch):
    from iai_mcp import iai_cli

    hits = [
        {"literal_surface": f"memory {i}", "score": 0.9 - i * 0.1}
        for i in range(10)
    ]
    fake_resp = {"jsonrpc": "2.0", "id": 1, "result": {"hits": hits}}
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", lambda *a, **k: fake_resp)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_recall(_make_args(limit=3))

    assert rc == 0
    out = buf.getvalue()
    assert "memory 0" in out
    assert "memory 1" in out
    assert "memory 2" in out
    assert "memory 3" not in out
