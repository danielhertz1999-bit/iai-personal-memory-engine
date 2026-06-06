"""Regression gate for the memory_footprint RSS budget.

Runs ``bench/memory_footprint.py`` at N=1000 in a subprocess and asserts
that the JSON output reports ``passed=true`` and that
``rss_mb_peak <= threshold_mb`` (both read straight from the JSON so the
test stays in lock-step with whatever the bench module declares as the
current target).

Two assertions:

  * ``test_threshold_constant_is_documented`` — cheap unit check that the
    module-level constant is non-zero and properly imported.
  * ``test_rss_under_validated_target`` — end-to-end gate at N=1000.

The end-to-end gate is marked ``@pytest.mark.slow`` because the bench
takes ~3-4 minutes wall-time at N=1000 on the dev Mac. Run normally via
``pytest`` (slow markers are not skipped by default in this project).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_SCRIPT = REPO_ROOT / "bench" / "memory_footprint.py"


def test_threshold_constant_is_documented() -> None:
    """Cheap unit check: THRESHOLD_MB is set, positive, and importable.

    Acts as a smoke test for the bench module — a typo that nullifies the
    constant would otherwise only show up at end-to-end run time.
    """
    # Insert REPO_ROOT into sys.path so the bench package is importable
    # without leaning on the parent venv's editable install.
    src_str = str(REPO_ROOT)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    from bench import memory_footprint  # noqa: PLC0415 -- intentional late import

    assert hasattr(memory_footprint, "THRESHOLD_MB"), (
        "bench.memory_footprint must declare THRESHOLD_MB"
    )
    assert memory_footprint.THRESHOLD_MB > 0, (
        f"THRESHOLD_MB must be positive, got {memory_footprint.THRESHOLD_MB}"
    )


@pytest.mark.slow
def test_rss_under_validated_target(tmp_path: Path) -> None:
    """End-to-end gate: bench passes at N=1000 against the declared target.

    Spawns the bench in a subprocess from a clean ``tmp_path`` cwd so the
    test cannot accidentally pick up bench-local state. Asserts both the
    bench-side ``passed`` flag and the explicit
    ``rss_mb_peak <= threshold_mb`` invariant straight off the JSON.
    """
    env = os.environ.copy()
    # Bench requires a passphrase to derive its AES key (
    # crypto gate); use the same literal as other bench scripts.
    env.setdefault(
        "IAI_MCP_CRYPTO_PASSPHRASE",
        "iai-mcp-bench-falsifiability-deterministic-2026",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(BENCH_SCRIPT),
            "--n", "1000",
            "--skip-graph",
            "--seed", "42",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=900,  # 15 min ceiling; healthy runs complete in 3-4 min.
    )
    assert result.returncode == 0, (
        f"bench exit {result.returncode}\nstderr (tail):\n"
        f"{result.stderr[-2000:]}"
    )
    # Stdout is exactly one JSON line per the bench docstring; tolerate
    # extra log lines by picking the last line that looks like JSON.
    lines = [
        ln for ln in result.stdout.splitlines() if ln.strip().startswith("{")
    ]
    assert lines, (
        f"no JSON line in stdout:\nSTDOUT (tail):\n{result.stdout[-2000:]}"
    )
    data = json.loads(lines[-1])
    assert "rss_mb_peak" in data, f"JSON missing rss_mb_peak: {data}"
    assert "threshold_mb" in data, f"JSON missing threshold_mb: {data}"
    assert "passed" in data, f"JSON missing passed: {data}"
    assert data["passed"] is True, (
        f"bench failed: rss_mb_peak={data['rss_mb_peak']} > "
        f"threshold_mb={data['threshold_mb']}; full result={data}"
    )
    assert data["rss_mb_peak"] <= data["threshold_mb"], data
