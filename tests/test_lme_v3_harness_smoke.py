"""Task 2 — v3 harness smoke regression fence.

Two test classes covering 's backward-compat assertion + flag-coverage
smoke for the four (granularity × dataset) combinations.

## Class A: TestV2BaselineReproduction

Per: ``--granularity turn --dataset raw`` MUST reproduce v2's
per-row R@5 / R@10 measurements exactly on six pinned qids (one per
question_type, all R@5_X = R@5_Y = R@10_X = R@10_Y = 1.0 in
``bench/lme500/output/lme500-v2.json.jsonl``). If a future harness change
drifts from v2 baseline on these qids, this test fails loudly.

The test invokes the harness as a subprocess (using ``sys.executable`` so
the venv that has ``iai_mcp`` installed carries through) with
``--qid-include`` to scope to only the 6 pinned qids — avoiding the
multi-hour full-bench wall-clock.

Wall-clock budget: ~5-10 min on Apple M2 Max (real embedder, real
pipeline, real graph build per row). This
is host-portable: re-runnable on the bench host if construction-host
execution is preferred to be skipped. Set
``IAI_MCP_SKIP_LME_V3_SMOKE=1`` to skip on slow hosts; default is RUN.

## Class B: TestFourCombinationCoverage

Argparse-only coverage for the four (granularity × dataset) combinations.
No embedder load, no HF download — pure ``parser.parse_args(...)``
introspection. Verifies 's default switch (``session`` not ``turn``)
and 's default switch (``cleaned`` not ``raw``).

NO ``src/iai_mcp/`` modifications anywhere in this file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# Six v2-baseline pinned qids (one per question_type), all
# R@5_X = R@5_Y = R@10_X = R@10_Y = 1.0 in bench/lme500/output/lme500-v2.json.jsonl.
V2_BASELINE_QIDS: list[tuple[str, str]] = [
    ("e47becba",        "single-session-user"),
    ("0a995998",        "multi-session"),
    ("8a2466db",        "single-session-preference"),
    ("gpt4_59149c77",   "temporal-reasoning"),
    ("6a1eabeb",        "knowledge-update"),
    ("7161e7e2",        "single-session-assistant"),
]


REPO_ROOT = Path(__file__).resolve().parent.parent
V2_JSONL = REPO_ROOT / "bench" / "lme500" / "output" / "lme500-v2.json.jsonl"


@pytest.mark.skipif(
    os.environ.get("IAI_MCP_SKIP_LME_V3_SMOKE") == "1",
    reason="IAI_MCP_SKIP_LME_V3_SMOKE=1; smoke is host-portable to bench host",
)
class TestV2BaselineReproduction:
    """backward-compat fence: --granularity turn --dataset raw
    reproduces v2 per-qid measurements EXACTLY on the six pinned qids."""

    def _load_v2_truth(self) -> dict[str, dict]:
        """Return ``{qid: row}`` from lme500-v2.json.jsonl for the six pinned qids."""
        truth: dict[str, dict] = {}
        wanted = {qid for qid, _ in V2_BASELINE_QIDS}
        with V2_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = rec.get("question_id")
                if qid in wanted:
                    truth[qid] = rec
        missing = wanted - set(truth.keys())
        assert not missing, (
            f"v2 baseline JSONL missing pinned qids: {missing} — "
            f"the v2-baseline-reproduction fence is invalid"
        )
        return truth

    def test_v2_baseline_reproduction_six_qids(self, tmp_path: Path) -> None:
        """Run the harness on the six pinned qids in v1/v2 mode and assert
        R@5 / R@10 match v2 baseline per-row exactly (1.0 / 1.0 for all six)."""
        truth = self._load_v2_truth()
        qid_csv = ",".join(qid for qid, _ in V2_BASELINE_QIDS)

        out_path = tmp_path / "lme_v3_smoke_v2_repro.json"
        ckpt_path = tmp_path / "lme_v3_smoke_v2_repro.jsonl"

        cmd = [
            sys.executable,
            "-m",
            "bench.longmemeval_blind",
            "--split", "S",
            "--granularity", "turn",
            "--dataset", "raw",
            "--qid-include", qid_csv,
            "--out", str(out_path),
            "--checkpoint", str(ckpt_path),
        ]

        # Wall-clock budget: ~5-10 min on M2 Max for 6 full pipeline rows.
        # 30 minutes timeout for pessimistic CI/cold-cache hosts.
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30 * 60,
        )
        assert proc.returncode == 0, (
            f"harness subprocess failed:\n"
            f"stdout: {proc.stdout[-2000:]}\n"
            f"stderr: {proc.stderr[-2000:]}"
        )

        # Read per-row JSONL output (one record per qid).
        assert out_path.exists(), f"output JSON not written: {out_path}"
        per_row: dict[str, dict] = {}
        with ckpt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = rec.get("question_id")
                if qid:
                    per_row[qid] = rec

        # Per-qid pinning: every one of the six MUST match v2 baseline.
        # If even one differs, the test fails — no population averaging.
        mismatches: list[str] = []
        for qid, qtype in V2_BASELINE_QIDS:
            assert qid in per_row, (
                f"qid {qid} ({qtype}) missing from v3 smoke output"
            )
            v2 = truth[qid]
            v3 = per_row[qid]
            for metric in (
                "r_at_5_retrieve",
                "r_at_10_retrieve",
                "r_at_5_pipeline",
                "r_at_10_pipeline",
            ):
                v2_val = float(v2[metric])
                v3_val = float(v3.get(metric, -1.0))
                if v2_val != v3_val:
                    mismatches.append(
                        f"qid={qid} qtype={qtype} {metric}: "
                        f"v2={v2_val} v3={v3_val}"
                    )
        assert not mismatches, (
            "v2 baseline reproduction FAILED — harness has drifted on the "
            "six pinned qids:\n  " + "\n  ".join(mismatches)
        )


class TestFourCombinationCoverage:
    """argparse coverage: the four (granularity × dataset)
    combinations are all accepted; defaults match LOCKED decisions."""

    def _build_parser(self):
        """Construct the harness argparse parser without running main()."""
        # We exercise argparse directly via subprocess so we don't risk
        # importing iai_mcp / loading the embedder. The harness's --help
        # output already covers flag presence; here we test argparse
        # acceptance + defaults by running with --help / dry combinations.
        return None  # marker; tests below use subprocess --help

    def _normalize_help(self, text: str) -> str:
        """Collapse whitespace/newlines so argparse's help-line wrapping
        does not break substring assertions."""
        import re as _re
        return _re.sub(r"\s+", " ", text)

    def test_help_lists_granularity_flag_with_session_default(self) -> None:
        """--granularity choices = {session, turn} and default = session."""
        proc = subprocess.run(
            [sys.executable, "-m", "bench.longmemeval_blind", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, f"--help failed: {proc.stderr}"
        help_text = proc.stdout
        assert "--granularity" in help_text, "--granularity flag missing from --help"
        assert "{session,turn}" in help_text, (
            "--granularity choices != {session, turn}"
        )
        # Default disclosure on the help line.
        # Normalize whitespace because argparse wraps long help strings
        # onto multiple lines; the substring may straddle a newline.
        flat = self._normalize_help(help_text)
        assert "'session' (default)" in flat, (
            "--granularity default disclosure ('session' (default)) "
            "missing from --help text"
        )

    def test_help_lists_dataset_flag_with_cleaned_default(self) -> None:
        """--dataset choices = {cleaned, raw} and default = cleaned."""
        proc = subprocess.run(
            [sys.executable, "-m", "bench.longmemeval_blind", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, f"--help failed: {proc.stderr}"
        help_text = proc.stdout
        assert "--dataset" in help_text, "--dataset flag missing from --help"
        assert "{cleaned,raw}" in help_text, (
            "--dataset choices != {cleaned, raw}"
        )
        # Default disclosure on the help line.
        flat = self._normalize_help(help_text)
        assert "'cleaned' (default)" in flat, (
            "--dataset default disclosure ('cleaned' (default)) "
            "missing from --help text"
        )

    def test_help_lists_qid_include_flag(self) -> None:
        """--qid-include is wired."""
        proc = subprocess.run(
            [sys.executable, "-m", "bench.longmemeval_blind", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, f"--help failed: {proc.stderr}"
        help_text = proc.stdout
        assert "--qid-include" in help_text, (
            "--qid-include flag missing from --help"
        )

    def test_four_combinations_are_argparse_valid(self) -> None:
        """All four (granularity × dataset) combinations parse without
        raising. We verify by pre-flight: invoke the harness with each
        combination + --qid-include EMPTY (resolves to no rows after the
        filter, so the harness exits 0 without loading the embedder
        beyond import time).

        This isolates argparse acceptance from full-pipeline cost while
        still exercising the dataset adapter import path."""
        # We don't actually run any rows — passing an unmatched qid CSV
        # makes row_order empty after filtering, harness reaches the
        # output-write step with n_rows=0 and exits 0. This proves
        # argparse + adapter import both work for all 4 combos.
        combos = [
            ("turn", "raw"),
            ("session", "raw"),
            ("turn", "cleaned"),
            ("session", "cleaned"),
        ]
        with tempfile.TemporaryDirectory() as tdir:
            tdir_p = Path(tdir)
            for granularity, dataset in combos:
                out_path = tdir_p / f"smoke_{granularity}_{dataset}.json"
                ckpt_path = tdir_p / f"smoke_{granularity}_{dataset}.jsonl"
                cmd = [
                    sys.executable,
                    "-m", "bench.longmemeval_blind",
                    "--split", "S",
                    "--granularity", granularity,
                    "--dataset", dataset,
                    "--qid-include", "__nonexistent_qid_smoke_test__",
                    "--out", str(out_path),
                    "--checkpoint", str(ckpt_path),
                ]
                proc = subprocess.run(
                    cmd,
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=10 * 60,
                )
                assert proc.returncode == 0, (
                    f"combo granularity={granularity} dataset={dataset} "
                    f"failed:\n"
                    f"stdout: {proc.stdout[-1500:]}\n"
                    f"stderr: {proc.stderr[-1500:]}"
                )
                # Output JSON must record the chosen granularity + dataset
                # for v3 reproducibility.
                assert out_path.exists(), (
                    f"combo {granularity}/{dataset}: output JSON not written"
                )
                with out_path.open("r", encoding="utf-8") as f:
                    out = json.load(f)
                assert out["granularity"] == granularity, (
                    f"combo {granularity}/{dataset}: output JSON granularity "
                    f"= {out['granularity']!r}, expected {granularity!r}"
                )
                assert out["dataset_choice"] == dataset, (
                    f"combo {granularity}/{dataset}: output JSON "
                    f"dataset_choice = {out['dataset_choice']!r}, "
                    f"expected {dataset!r}"
                )
                assert out["n_rows"] == 0, (
                    f"combo {granularity}/{dataset}: expected n_rows=0 "
                    f"after no-match qid filter, got {out['n_rows']}"
                )
