"""Unit tests for the build-native CLI subcommand (cmd_build_native).

Covers two paths:
- success: cargo present + maturin exits 0 → handler returns 0, correct argv
- missing-cargo: shutil.which("cargo") returns None → non-zero, no subprocess,
  stderr contains rustup hint
"""
from __future__ import annotations

import argparse
from unittest.mock import Mock

import pytest

from iai_mcp.cli import cmd_build_native


def _make_args() -> argparse.Namespace:
    """Return a minimal Namespace — the handler reads no user-controlled attrs."""
    return argparse.Namespace()


def test_build_native_success(monkeypatch, capsys):
    """cargo present + maturin exits 0 → returns 0, correct argv sent to subprocess."""
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/cargo" if name == "cargo" else None)

    completed = Mock(returncode=0)
    run_mock = Mock(return_value=completed)
    monkeypatch.setattr("iai_mcp.cli.subprocess.run", run_mock)

    rc = cmd_build_native(_make_args())

    assert rc == 0
    run_mock.assert_called_once()
    call_args = run_mock.call_args
    argv = call_args[0][0]  # first positional arg is the cmd list

    assert "maturin" in argv
    assert "develop" in argv
    assert "--release" in argv

    # Locate --manifest-path value
    manifest_idx = argv.index("--manifest-path")
    manifest_val = argv[manifest_idx + 1]
    assert manifest_val.endswith("rust/iai_mcp_native/Cargo.toml")


def test_build_native_missing_cargo(monkeypatch, capsys):
    """cargo absent → non-zero exit, subprocess not called, rustup hint in stderr."""
    monkeypatch.setattr("shutil.which", lambda name: None)

    run_mock = Mock()
    monkeypatch.setattr("iai_mcp.cli.subprocess.run", run_mock)

    rc = cmd_build_native(_make_args())

    assert rc != 0
    run_mock.assert_not_called()

    captured = capsys.readouterr()
    stderr_text = captured.err
    assert "rustup" in stderr_text or "https://rustup.rs" in stderr_text
