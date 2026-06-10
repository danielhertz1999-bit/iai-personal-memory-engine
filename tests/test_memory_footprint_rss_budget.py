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
    src_str = str(REPO_ROOT)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    from bench import memory_footprint  # noqa: PLC0415  -- intentional late import

    assert hasattr(memory_footprint, "THRESHOLD_MB"), (
        "bench.memory_footprint must declare THRESHOLD_MB"
    )
    assert memory_footprint.THRESHOLD_MB > 0, (
        f"THRESHOLD_MB must be positive, got {memory_footprint.THRESHOLD_MB}"
    )


@pytest.mark.slow
def test_rss_under_validated_target(tmp_path: Path) -> None:
    env = os.environ.copy()
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
        timeout=900,
    )
    assert result.returncode == 0, (
        f"bench exit {result.returncode}\nstderr (tail):\n"
        f"{result.stderr[-2000:]}"
    )
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
