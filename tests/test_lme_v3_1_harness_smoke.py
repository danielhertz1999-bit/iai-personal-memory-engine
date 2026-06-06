"""Harness smoke for the bench --embedder flag.

Asserts:
1. --embedder bge-small-en-v1.5 reproduces v3 output on 1 pinned qid
   (default-embedder baseline; ensures the flag does not break the v3 path).
2. Output JSON top-level metadata pins the embedder identity for reproducibility.

Each test runs bench/longmemeval_blind.py as a subprocess on n=1 question
(--qid-include e47becba -- first qid in v3 per_row, single-session-user
type, deterministic baseline R@5_X = R@5_Y = 1.0 with bge-small).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# First qid in v3 per_row (single-session-user; deterministic;
# both R@5_X = R@5_Y = 1.0 in v3 baseline). Verified via:
# python3 -c "import json; d=json.load(open('bench/lme500/output/lme500-v3.json'));
# print(d['per_row'][0]['question_id'])"
PINNED_QID_FOR_SMOKE = "e47becba"

FIXTURES = REPO / "tests" / "fixtures"


def _run_harness(embedder: str, qid_filter: str = PINNED_QID_FOR_SMOKE) -> dict:
    """Run bench/longmemeval_blind.py and return parsed JSON output.

    Note: harness CLI uses --limit (not --num-questions), --out (not --output),
    --checkpoint (not --jsonl-output). JSON output stores rows under 'per_row' key.
    """
    out_path = FIXTURES / f"smoke-v3.1-{embedder}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "bench.longmemeval_blind",
        "--split",
        "S",
        "--granularity",
        "session",
        "--dataset",
        "cleaned",
        "--embedder",
        embedder,
        "--qid-include",
        qid_filter,
        "--out",
        str(out_path),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"harness failed for --embedder {embedder}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return json.loads(out_path.read_text())


def test_bge_small_baseline_reproduces_v3() -> None:
    """Default embedder reproduces v3 output verbatim on the pinned qid."""
    out = _run_harness(embedder="bge-small-en-v1.5")
    per_row = out["per_row"]
    assert len(per_row) == 1, f"expected 1 row, got {len(per_row)}"
    row = per_row[0]
    assert row["question_id"] == PINNED_QID_FOR_SMOKE

    # Pull v3 baseline for this qid.
    v3 = json.loads((REPO / "bench" / "lme500" / "output" / "lme500-v3.json").read_text())
    v3_per_row = v3["per_row"]
    v3_row = next(r for r in v3_per_row if r["question_id"] == PINNED_QID_FOR_SMOKE)

    assert row["r_at_5_pipeline"] == v3_row["r_at_5_pipeline"], (
        f"R@5 pipeline drift: smoke={row['r_at_5_pipeline']} v3={v3_row['r_at_5_pipeline']}"
    )
    assert row["r_at_10_pipeline"] == v3_row["r_at_10_pipeline"]
    assert row["r_at_5_retrieve"] == v3_row["r_at_5_retrieve"]
    assert row["r_at_10_retrieve"] == v3_row["r_at_10_retrieve"]


def test_embedder_metadata_recorded_in_output() -> None:
    """Output JSON top-level metadata pins the English embedder identity for reproducibility."""
    # Re-load the fixture written by test_bge_small_baseline_reproduces_v3.
    bge_path = FIXTURES / "smoke-v3.1-bge-small-en-v1.5.json"
    if not bge_path.exists():
        # Run the harness if the fixture is not yet on disk.
        _run_harness(embedder="bge-small-en-v1.5")
    bge_out = json.loads(bge_path.read_text())
    assert bge_out.get("embedder_model_key") == "bge-small-en-v1.5"
    assert bge_out.get("embedder_hf_id") == "BAAI/bge-small-en-v1.5"
