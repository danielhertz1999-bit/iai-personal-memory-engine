"""Task 3 — mempalace / claude-mem subprocess adapters in
``bench/total_session_cost.py``.

These adapters let the reference column carry a live measurement
from the mempalace CLI when it is installed locally, falling back to
honest "adapter unavailable" disclosure when absent. They never block
the bench: subprocess timeouts and non-zero exits return None and emit
a ``bench_adapter_unavailable`` stderr event.

Covered contracts:

    Test 1 _run_mempalace_adapter signature exists and accepts the 10-turn script
    Test 2 mempalace CLI absent -> None + stderr event, no exception
    Test 3 mempalace CLI present -> sums per-turn token counts via the 3-tier counter
    Test 4 --measure-mempalace flag wires the live adapter into refs["mempalace_measured"]
    Test 5 _run_claude_mem_adapter mirrors mempalace shape for forward compat
    Test 6 manual --ref-mempalace alongside --measure-mempalace keeps both values,
            but LIVE measurement is the comparator for the `passed` flag
"""
from __future__ import annotations

import json
import subprocess
from unittest import mock

import pytest

from bench.total_session_cost import (
    _SCRIPT,
    _run_claude_mem_adapter,
    _run_mempalace_adapter,
    main,
    run_total_session_cost,
)


# --------------------------------------------------------------------------- helpers


def _fixed_counter(text: str) -> int:
    """Deterministic counter: 1 token per word. Keeps assertions stable
    across tiktoken / anthropic / char4 drift."""
    return max(1, len(text.split()))


# --------------------------------------------------------------------------- Test 1


def test_mempalace_adapter_signature():
    # Signature must accept the canonical 10-turn script and a counter.
    result = _run_mempalace_adapter(_SCRIPT, _fixed_counter)
    # Will be None on a machine without mempalace *responding cleanly*, but
    # the function must exist and not raise — callers depend on that contract.
    assert result is None or isinstance(result, int)


# --------------------------------------------------------------------------- Test 2


def test_mempalace_adapter_absent_cli_returns_none(capsys):
    with mock.patch("bench.total_session_cost.shutil.which", return_value=None):
        result = _run_mempalace_adapter(_SCRIPT, _fixed_counter)
    assert result is None
    err = capsys.readouterr().err
    assert "bench_adapter_unavailable" in err
    assert "mempalace" in err


# --------------------------------------------------------------------------- Test 3


def test_mempalace_adapter_live_run_sums_stdout_tokens():
    """With ``shutil.which`` finding the CLI and ``subprocess.run`` returning
    deterministic stdout, the adapter sums the token counts across all 10
    turns using the injected counter."""

    def fake_which(name):
        return "/fake/bin/mempalace" if name == "mempalace" else None

    def fake_run(*args, **kwargs):
        # stdout carries 3 words per turn -> 3 tokens per turn under _fixed_counter.
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=0,
            stdout="one two three",
            stderr="",
        )

    with mock.patch("bench.total_session_cost.shutil.which", side_effect=fake_which), \
         mock.patch("bench.total_session_cost.subprocess.run", side_effect=fake_run):
        result = _run_mempalace_adapter(_SCRIPT, _fixed_counter)
    assert result == 3 * len(_SCRIPT)


# --------------------------------------------------------------------------- Test 4


def test_measure_mempalace_flag_populates_refs(monkeypatch, capsys):
    """End-to-end: running `main` with --measure-mempalace populates
    refs["mempalace_measured"] when the adapter returns a number."""

    def fake_which(name):
        return "/fake/bin/mempalace" if name == "mempalace" else None

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=0,
            stdout="hello world",
            stderr="",
        )

    with mock.patch("bench.total_session_cost.shutil.which", side_effect=fake_which), \
         mock.patch("bench.total_session_cost.subprocess.run", side_effect=fake_run):
        rc = main(["--wake-depth", "minimal", "--measure-mempalace"])

    captured = capsys.readouterr()
    result = json.loads(captured.out.strip())
    assert "mempalace_measured" in result["refs"]
    assert isinstance(result["refs"]["mempalace_measured"], int)
    assert result["refs"]["mempalace_measured"] > 0


# --------------------------------------------------------------------------- Test 5


def test_claude_mem_adapter_mirrors_mempalace_shape(capsys):
    """The claude-mem adapter has the same signature and absent-CLI fallback
    as the mempalace adapter, even though claude-mem is not installed
    locally. This keeps the forward-compat path live."""
    with mock.patch("bench.total_session_cost.shutil.which", return_value=None):
        result = _run_claude_mem_adapter(_SCRIPT, _fixed_counter)
    assert result is None
    err = capsys.readouterr().err
    assert "bench_adapter_unavailable" in err
    assert "claude-mem" in err


# --------------------------------------------------------------------------- Test 6


def test_live_measurement_wins_over_manual_ref():
    """When both ``--measure-mempalace`` and ``--ref-mempalace <int>`` are
    supplied, the live measurement lands in ``refs["mempalace_measured"]``
    and is the comparator for ``passed``; the manual int is recorded in
    ``refs["mempalace_manual"]`` for audit trail."""

    with mock.patch("bench.total_session_cost.shutil.which",
                    side_effect=lambda n: "/fake/bin/mempalace" if n == "mempalace" else None), \
         mock.patch("bench.total_session_cost.subprocess.run",
                    return_value=subprocess.CompletedProcess(
                        args=[], returncode=0,
                        stdout="token " * 5000,  # 5000 tokens across 10 turns
                        stderr="",
                    )):
        result = run_total_session_cost(
            wake_depth="minimal",
            mempalace_ref=10,  # manual ref — deliberately tiny to force fail IF used
            measure_mempalace=True,
            count_tokens_fn=_fixed_counter,
        )
    assert "mempalace_measured" in result["refs"]
    assert "mempalace_manual" in result["refs"]
    assert result["refs"]["mempalace_manual"] == 10
    # LIVE measurement is the gate; with 50000+ tokens live, IAI total
    # (<~3000) is well below, so passed is True.
    assert result["passed"] is True
