
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYZER_PATH = REPO_ROOT / "bench" / "analyze_efe_ab.py"


SAMPLE_HEADER = [
    "probe_id", "seed", "n_slice", "condition", "topic",
    "pipeline_rank", "cosine_rank",
    "pipeline_hit_at_k", "cosine_hit_at_k",
    "s4_contradiction_emitted", "anti_hits_count", "hint_kinds",
    "pipeline_top1_text",
    "route", "cue_hash",
]


def _write_csv(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    csv_path = results_dir / "contradiction_longitudinal_synth.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SAMPLE_HEADER)
        w.writeheader()
        for r in rows:
            full = {k: "" for k in SAMPLE_HEADER}
            full.update(r)
            w.writerow(full)
    return csv_path


def _make_row(seed: int, route: str, rank: int, **kw: Any) -> dict[str, Any]:
    return {
        "probe_id": kw.get("probe_id", f"p-{seed}-{route}-{rank}"),
        "seed": str(seed),
        "n_slice": "0",
        "condition": "post_flip",
        "topic": "launch_date",
        "pipeline_rank": str(rank),
        "pipeline_hit_at_k": "1" if 0 < rank <= 10 else "0",
        "route": route,
        "cue_hash": kw.get("cue_hash", "deadbeef"),
    }


def _run_analyzer(results_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ANALYZER_PATH), str(results_dir)],
        capture_output=True,
        text=True,
    )


def _rows_for_seed_arm(seed: int, route: str, hits: int, misses: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(hits):
        rows.append(_make_row(seed, route, rank=1 + (i % 10)))
    for i in range(misses):
        rows.append(_make_row(seed, route, rank=-1))
    return rows


def test_clear_pass_exits_zero(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for seed in (13, 42, 137):
        rows += _rows_for_seed_arm(seed, "efe_real", hits=25, misses=5)
        rows += _rows_for_seed_arm(seed, "efe_shadow", hits=10, misses=20)
    _write_csv(tmp_path, rows)

    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 0, (
        f"expected exit 0, got {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )

    summary_json = tmp_path / "results" / "EFE-AB-SUMMARY.json"
    summary_md = tmp_path / "results" / "EFE-AB-SUMMARY.md"
    assert summary_json.exists(), "EFE-AB-SUMMARY.json not written"
    assert summary_md.exists(), "EFE-AB-SUMMARY.md not written"

    summary = json.loads(summary_json.read_text())
    assert summary["ship_gate_hit"] is True
    assert summary["cross_seed_mean_delta"] > 0.10
    assert summary["threshold"] == pytest.approx(0.10)
    assert summary["n_seeds"] == 3


def test_clear_fail_exits_one(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for seed in (13, 42, 137):
        rows += _rows_for_seed_arm(seed, "efe_real", hits=12, misses=18)
        rows += _rows_for_seed_arm(seed, "efe_shadow", hits=10, misses=20)
    _write_csv(tmp_path, rows)

    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 1, (
        f"expected exit 1, got {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    summary = json.loads((tmp_path / "results" / "EFE-AB-SUMMARY.json").read_text())
    assert summary["ship_gate_hit"] is False
    assert summary["cross_seed_mean_delta"] < 0.10


def test_missing_efe_real_exits_two(tmp_path: Path) -> None:
    rows = _rows_for_seed_arm(13, "efe_shadow", hits=10, misses=20)
    _write_csv(tmp_path, rows)

    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 2, (
        f"expected exit 2 (setup error), got {proc.returncode}; "
        f"stderr={proc.stderr!r}"
    )
    assert "efe_real" in proc.stderr, (
        f"stderr should name the missing arm; got {proc.stderr!r}"
    )


def test_missing_efe_shadow_exits_two(tmp_path: Path) -> None:
    rows = _rows_for_seed_arm(13, "efe_real", hits=10, misses=20)
    _write_csv(tmp_path, rows)

    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 2
    assert "efe_shadow" in proc.stderr


def test_no_csv_in_dir_exits_two(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 2
    assert "contradiction_longitudinal" in proc.stderr.lower() or "no" in proc.stderr.lower()


def test_per_seed_aggregation() -> None:
    rows: list[dict[str, Any]] = []
    seed_to_counts = {
        13: ((4, 16), (3, 17)),
        42: ((10, 10), (7, 13)),
        137: ((12, 8), (8, 12)),
    }
    for seed, ((real_hit, real_miss), (shadow_hit, shadow_miss)) in seed_to_counts.items():
        rows += _rows_for_seed_arm(seed, "efe_real", hits=real_hit, misses=real_miss)
        rows += _rows_for_seed_arm(seed, "efe_shadow", hits=shadow_hit, misses=shadow_miss)

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from bench.analyze_efe_ab import (
            aggregate_across_seeds,
            compute_per_route_rescue_at_k,
        )
    finally:
        sys.path.pop(0)

    per_seed = compute_per_route_rescue_at_k(rows)
    agg = aggregate_across_seeds(per_seed)

    assert agg["per_seed"]["13"]["delta"] == pytest.approx(0.05, abs=1e-9)
    assert agg["per_seed"]["42"]["delta"] == pytest.approx(0.15, abs=1e-9)
    assert agg["per_seed"]["137"]["delta"] == pytest.approx(0.20, abs=1e-9)
    assert agg["cross_seed_mean_delta"] == pytest.approx(0.4 / 3.0, abs=1e-9)
    assert agg["ship_gate_hit"] is True


def test_per_seed_aggregation_exit_zero_on_threshold_boundary(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    seed_to_counts = {
        13: ((4, 16), (3, 17)),
        42: ((10, 10), (7, 13)),
        137: ((12, 8), (8, 12)),
    }
    for seed, ((rh, rm), (sh, sm)) in seed_to_counts.items():
        rows += _rows_for_seed_arm(seed, "efe_real", hits=rh, misses=rm)
        rows += _rows_for_seed_arm(seed, "efe_shadow", hits=sh, misses=sm)
    _write_csv(tmp_path, rows)
    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 0, (
        f"expected exit 0 (mean=0.133 > 0.10); got {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )


def test_efe_skip_rows_excluded(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    rows += _rows_for_seed_arm(13, "efe_real", hits=60, misses=40)
    rows += _rows_for_seed_arm(13, "efe_shadow", hits=20, misses=80)
    rows += _rows_for_seed_arm(13, "efe_skip", hits=50, misses=50)
    _write_csv(tmp_path, rows)

    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 0, (
        f"expected exit 0 with delta=0.40; got {proc.returncode}; "
        f"stderr={proc.stderr!r}"
    )
    summary = json.loads((tmp_path / "results" / "EFE-AB-SUMMARY.json").read_text())
    assert summary["n_rows"] == 200, (
        f"efe_skip rows should be excluded; got n_rows={summary['n_rows']}"
    )
    assert summary["per_seed"]["13"]["delta"] == pytest.approx(0.40, abs=1e-9)


def test_pipeline_rank_zero_or_negative_or_overK_is_miss(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from bench.analyze_efe_ab import _is_hit_at_k
    finally:
        sys.path.pop(0)

    assert _is_hit_at_k("1") is True
    assert _is_hit_at_k("5") is True
    assert _is_hit_at_k("10") is True
    assert _is_hit_at_k("0") is False
    assert _is_hit_at_k("-1") is False
    assert _is_hit_at_k("11") is False
    assert _is_hit_at_k("100") is False
    assert _is_hit_at_k("not-a-number") is False
    assert _is_hit_at_k("") is False


def test_empty_route_rows_dropped(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    rows += _rows_for_seed_arm(13, "efe_real", hits=10, misses=10)
    rows += _rows_for_seed_arm(13, "efe_shadow", hits=2, misses=18)
    for i in range(50):
        rows.append(_make_row(13, "", rank=1))
    _write_csv(tmp_path, rows)

    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 0
    summary = json.loads((tmp_path / "results" / "EFE-AB-SUMMARY.json").read_text())
    assert summary["n_rows"] == 40, (
        f"empty-route rows should be excluded; got n_rows={summary['n_rows']}"
    )


def test_missing_required_column_exits_two(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    csv_path = results_dir / "contradiction_longitudinal_legacy.csv"
    legacy_header = [c for c in SAMPLE_HEADER if c not in ("route", "cue_hash")]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=legacy_header)
        w.writeheader()
        w.writerow({k: "x" for k in legacy_header})

    proc = _run_analyzer(results_dir)
    assert proc.returncode == 2
    assert "route" in proc.stderr or "missing" in proc.stderr.lower()


def test_help_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, str(ANALYZER_PATH), "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower()
