"""Tests for `iai status` -- short user-tier health summary."""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout


def _make_args():
    import argparse
    return argparse.Namespace()


def _stub_topology(monkeypatch, *, ok: bool, n: int = 100, regime: str = "healthy"):
    """Mock the daemon socket helper for topology probing."""
    if ok:
        fake_resp = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "N": n,
                "C": 0.1,
                "L": 5.0,
                "sigma": 2.0,
                "community_count": 1,
                "rich_club_ratio": 0.05,
                "regime": regime,
            },
        }
    else:
        fake_resp = None
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", lambda *a, **k: fake_resp)


def test_status_daemon_up_prints_summary(monkeypatch):
    """Daemon alive + valid subscription -> 5 lines, all populated."""
    from iai_mcp import iai_cli

    _stub_topology(monkeypatch, ok=True, n=6266, regime="healthy")
    monkeypatch.setattr(
        "iai_mcp.claude_cli.verify_credentials_subscription",
        lambda: {"ok": True, "subscription_type": "max"},
    )
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_status(_make_args())

    assert rc == 0
    out = buf.getvalue()
    assert "iai status" in out
    assert "UP" in out
    assert "6266" in out
    assert "healthy" in out
    assert "max" in out


def test_status_daemon_down_prints_down(monkeypatch):
    """Daemon dead -> DOWN, but the subscription row still renders."""
    from iai_mcp import iai_cli

    _stub_topology(monkeypatch, ok=False)
    monkeypatch.setattr(
        "iai_mcp.claude_cli.verify_credentials_subscription",
        lambda: {"ok": True, "subscription_type": "pro"},
    )
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_status(_make_args())

    assert rc == 0
    out = buf.getvalue()
    assert "DOWN" in out
    assert "pro" in out


def test_status_subscription_missing_shown(monkeypatch):
    """Subscription gate denies -> 'missing' label + reason."""
    from iai_mcp import iai_cli

    _stub_topology(monkeypatch, ok=True, n=10, regime="developmental")
    monkeypatch.setattr(
        "iai_mcp.claude_cli.verify_credentials_subscription",
        lambda: {"ok": False, "reason": "credentials_file_missing"},
    )
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_status(_make_args())

    assert rc == 0
    out = buf.getvalue()
    assert "missing" in out.lower()
    assert "credentials_file_missing" in out


def test_status_subcommand_registered():
    """`iai status` is a valid subcommand."""
    from iai_mcp.iai_cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["status"])
    assert args.cmd == "status"


def test_status_no_color_strips_ansi(monkeypatch):
    """NO_COLOR env strips ANSI from the status header."""
    from iai_mcp import iai_cli

    _stub_topology(monkeypatch, ok=True)
    monkeypatch.setattr(
        "iai_mcp.claude_cli.verify_credentials_subscription",
        lambda: {"ok": True, "subscription_type": "max"},
    )
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        iai_cli.cmd_status(_make_args())

    assert "\x1b[" not in buf.getvalue()
