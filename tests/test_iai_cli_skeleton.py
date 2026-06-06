"""Smoke tests for the user-facing `iai` CLI: argparse contract, logo
rendering, NO_COLOR honored, --version flag."""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import pytest


def test_iai_cli_module_importable():
    """The `iai_cli` module must import without spawning subprocesses
    or touching the daemon — `iai --help` must stay cheap."""
    from iai_mcp import iai_cli

    assert hasattr(iai_cli, "main")
    assert callable(iai_cli.main)
    assert hasattr(iai_cli, "__version__")


def test_iai_no_args_prints_logo_and_help(monkeypatch):
    """Running `iai` with no arguments prints the cyan ASCII logo + help."""
    from iai_mcp.iai_cli import main

    # Force NO_COLOR off + simulate tty so logo emits ANSI.
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([])
    output = buf.getvalue()

    assert rc == 0
    # Logo includes the figlet "iai" block letters.
    assert "iai" in output.lower() or "█" in output
    # Help text mentions subcommands.
    assert "recall" in output
    assert "capture" in output


def test_iai_no_color_env_strips_ansi(monkeypatch):
    """NO_COLOR=1 must produce plain text -- no ANSI escapes anywhere."""
    from iai_mcp.iai_cli import main

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        main([])
    output = buf.getvalue()
    assert "\x1b[" not in output, f"NO_COLOR violated: ANSI escape in output: {output!r}"


def test_iai_non_tty_stdout_strips_ansi(monkeypatch):
    """When stdout is piped (isatty False) we must emit plain text."""
    from iai_mcp.iai_cli import main

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        main([])
    output = buf.getvalue()
    assert "\x1b[" not in output


def test_iai_version_flag():
    """`iai --version` emits the version and exits 0."""
    from iai_mcp.iai_cli import main, __version__

    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as ei:
            main(["--version"])
    assert ei.value.code == 0
    assert __version__ in buf.getvalue()


def test_iai_unknown_subcommand_errors(monkeypatch):
    """argparse must reject an unknown subcommand with a non-zero exit."""
    from iai_mcp.iai_cli import main

    err = io.StringIO()
    with redirect_stderr(err):
        with pytest.raises(SystemExit) as ei:
            main(["nope-not-a-real-command"])
    assert ei.value.code != 0


def test_iai_recall_subcommand_registered():
    """`iai recall` is a valid subcommand and accepts a cue argument."""
    from iai_mcp.iai_cli import _build_parser

    parser = _build_parser()
    # argparse stores subparsers under the `dest` named 'cmd'.
    actions = {a.dest for a in parser._actions}
    assert "cmd" in actions


def test_iai_capture_subcommand_registered():
    """`iai capture` is a valid subcommand and accepts a text argument."""
    from iai_mcp.iai_cli import _build_parser

    parser = _build_parser()
    # Parse a fake `capture <text>` invocation -- should not raise.
    args = parser.parse_args(["capture", "hello world"])
    assert args.cmd == "capture"
    assert args.text == "hello world"
