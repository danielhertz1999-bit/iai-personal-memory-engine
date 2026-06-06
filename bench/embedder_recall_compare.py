"""Monotonic recall@k regression gate for the Rust forward-pass embedder.

Runs LongMemEval-S blind retrieval twice (PyTorch baseline + Rust path)
and asserts STRICT monotonic improvement:

    recall@5_rust >= recall@5_pytorch (per-question AND aggregate)
    recall@10_rust >= recall@10_pytorch (per-question AND aggregate)

NO tolerance band in the "drop allowed" direction. The user-locked
constitutional rule: results may only RISE, never DROP.

Per-question recall@k is BINARY (0 or 1 — gold hit in top-k or not),
so "rust >= pytorch per-question" reduces to: any question PyTorch got
right must NOT be missed by Rust. Rust may pick up additional questions
that PyTorch missed (monotonic improvement) — that is the only
acceptable delta direction.

## Backend isolation

Two clean subprocess runs, NOT in-process backend swap. Reason: the
production embedder loads the model at `Embedder.__init__` and caches
in module-level `_MODEL_CACHE`. Swapping the env var mid-process would
risk cross-contamination of the cached SentenceTransformer instance vs
the freshly-spawned `iai_mcp_embed.Embedder()`. Subprocess isolation is
the deterministic path.

## Retrieval prong selection

The underlying harness (`bench/longmemeval_blind.py`) emits both
`r_at_5_retrieve` (prong X = flat-cosine over L2-normalized 384-d
vectors) AND `r_at_5_pipeline` (prong Y = full graph-native architecture
with rich-club, community gates, mode-bias).

This bench consumes prong X ONLY. Prong Y introduces variance from
graph features that are NOT embedder-attributable; mixing it into a
monotonicity gate would conflate embedder regressions with graph
pipeline drift. Prong X is pure embedder -> cosine ranking, which is
exactly the scope this gate measures.

## CLI

    python bench/embedder_recall_compare.py
        [--n-questions 500]
        [--seed 42]
        [--skip-pytorch]
        [--out-dir bench/]

The defaults reproduce the SPEC.md R8 gate (N=500 full LongMemEval-S,
seed=42 audit metadata, both backends).

## Output files

  - bench/embedder_recall_compare.pytorch.json -- PyTorch baseline run
  - bench/embedder_recall_compare.rust.json -- Rust path run

Each file follows the shape:

    {
      "backend": "pytorch" | "rust",
      "model_revision_sha": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
      "dataset_id": "xiaowu0162/longmemeval-cleaned",
      "dataset_revision_sha": "98d7416c24...",
      "split": "S",
      "seed": int,
      "n_questions": int,
      "per_question": [
        {
          "qid": "...",
          "recall_at_5": 0 | 1,
          "recall_at_10": 0 | 1,
          "question_type": "..."
        },
        ...
      ],
      "aggregate": {
        "recall_at_5": float,
        "recall_at_10": float
      },
      "wall_seconds": float,
      "generated_at_utc": "..."
    }

## Exit codes

  - 0 monotonic verdict (Rust >= PyTorch on every metric, per-question
        and aggregate)
  - 1 monotonicity violation (one or more drops detected)
  - 2 subprocess crash / setup error (mostly user-actionable; e.g.
        wheel not installed)

The gate is the bench result, not a metric. If a single LongMemEval-S
question drops, this script exits 1 -- the phase does NOT close until
the violation is resolved (route back to the embedder forward pass).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BLIND_SCRIPT = REPO / "bench" / "longmemeval_blind.py"
DEFAULT_OUT_DIR = REPO / "bench"

# Pinned against bench/embedder_baseline/metadata.json.
# Recorded in every output JSON for audit / reproducibility.
PINNED_MODEL_REVISION_SHA = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
PINNED_DATASET_ID = "xiaowu0162/longmemeval-cleaned"
PINNED_DATASET_REVISION_SHA = "98d7416c24c778c2fee6e6f3006e7a073259d48f"


def _harness_output_path(backend: str, out_dir: Path) -> Path:
    """Intermediate harness output (long-form blind-run JSON)."""
    return out_dir / f"_embedder_recall_compare.{backend}.raw.json"


def _compare_output_path(backend: str, out_dir: Path) -> Path:
    """Compare-shape output, committed to git as audit trail."""
    return out_dir / f"embedder_recall_compare.{backend}.json"


def run_backend(
    backend: str,
    n_questions: int,
    seed: int,
    out_dir: Path,
) -> dict:
    """Spawn a subprocess that runs longmemeval_blind.py with the given
    backend, then transform its output into the compare-shape JSON.

    Returns the compare-shape dict (also written to disk).
    """
    raw_out = _harness_output_path(backend, out_dir)
    cmp_out = _compare_output_path(backend, out_dir)

    # Wipe any previous raw output so subprocess writes a fresh file
    # (defensive: avoids confusion between stale results across runs).
    if raw_out.exists():
        raw_out.unlink()

    # Wipe checkpoint too — clean run each time (deterministic gate).
    # The harness defaults to <out>.jsonl as checkpoint path.
    cp = Path(str(raw_out) + ".jsonl")
    if cp.exists():
        cp.unlink()

    env = os.environ.copy()
    env["IAI_MCP_EMBED_BACKEND"] = backend
    # Bench process owns its own embedder (no daemon involvement).
    env.setdefault("IAI_MCP_TEST_MODE", "1")
    # Quiet HF tokenizer fork warnings inside the subprocess.
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    cmd = [
        sys.executable,
        str(BLIND_SCRIPT),
        "--split", "S",
        "--limit", str(n_questions),
        "--granularity", "session",
        "--dataset", "cleaned",
        "--embedder", "bge-small-en-v1.5",
        "--out", str(raw_out),
        "--fresh",
    ]
    print(
        f"[recall_compare] backend={backend} cmd={' '.join(cmd)}",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.time()
    # Stream stderr live to our stderr so progress lines (per-row R@5
    # prints from the harness) are visible during the run.
    result = subprocess.run(
        cmd,
        env=env,
        check=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    wall = round(time.time() - t0, 2)

    if result.returncode != 0:
        raise SystemExit(
            f"backend={backend} run failed (exit {result.returncode}). "
            f"See stderr above for the harness's own diagnostic output."
        )
    if not raw_out.exists():
        raise SystemExit(
            f"backend={backend} run completed but no output file at {raw_out}; "
            f"the harness contract may have changed."
        )

    raw = json.loads(raw_out.read_text())
    compare_doc = _to_compare_shape(
        backend=backend,
        raw=raw,
        seed=seed,
        n_questions_requested=n_questions,
        wall_seconds=wall,
    )

    # Commit the compare-shape JSON (this is the audit file).
    cmp_out.parent.mkdir(parents=True, exist_ok=True)
    cmp_out.write_text(json.dumps(compare_doc, indent=2) + "\n")
    # Keep the raw harness file alongside for forensics on a failed run.

    return compare_doc


def _to_compare_shape(
    backend: str,
    raw: dict,
    seed: int,
    n_questions_requested: int,
    wall_seconds: float,
) -> dict:
    """Translate the harness's `r_at_5_retrieve / per_row` shape into the
    compare-shape contract this script publishes.

    The harness shape is rich (it ships both prong X and prong Y); for
    the monotonicity gate we project to prong X only (flat-cosine
    retrieval — the embedder-pure path). Per-question recall is read
    from `per_row[].r_at_5_retrieve` / `r_at_10_retrieve`.
    """
    per_row = raw.get("per_row", [])
    per_question = []
    for row in per_row:
        per_question.append(
            {
                "qid": row.get("question_id"),
                # recall@k is 0.0 or 1.0 in the harness; cast to int for
                # binary clarity in the JSON contract.
                "recall_at_5": int(float(row.get("r_at_5_retrieve", 0.0))),
                "recall_at_10": int(float(row.get("r_at_10_retrieve", 0.0))),
                "question_type": row.get("question_type", "unknown"),
            }
        )
    # The harness aggregates over the union of success + error rows; we
    # mirror that contract (silent zero on error counts as a miss, NOT a
    # softer "skip"). Use prong X (retrieve, flat-cosine).
    aggregate = {
        "recall_at_5": float(raw.get("r_at_5_retrieve", 0.0)),
        "recall_at_10": float(raw.get("r_at_10_retrieve", 0.0)),
    }
    return {
        "backend": backend,
        "model_revision_sha": PINNED_MODEL_REVISION_SHA,
        "dataset_id": raw.get("dataset_id", PINNED_DATASET_ID),
        "dataset_revision_sha": raw.get("revision", PINNED_DATASET_REVISION_SHA),
        "split": raw.get("split", "S"),
        "seed": seed,
        "n_questions_requested": n_questions_requested,
        "n_questions": int(raw.get("n_rows", len(per_question))),
        "granularity": raw.get("granularity", "session"),
        "embedder_model_key": raw.get("embedder_model_key", "bge-small-en-v1.5"),
        "embedder_hf_id": raw.get("embedder_hf_id"),
        "retrieval_prong": "X_retrieve_flat_cosine",
        "per_question": per_question,
        "aggregate": aggregate,
        "n_errors": int(raw.get("n_errors", 0)),
        "wall_seconds": wall_seconds,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def assert_monotonic(rust: dict, pytorch: dict) -> list[str]:
    """Detect monotonicity violations. Returns a list of human-readable
    violation lines (empty list = PASS).

    A violation is any case where Rust scored strictly worse than
    PyTorch on a per-question OR aggregate recall@k metric. Equal is OK
    (parity is the floor); strictly-better is the only acceptable
    delta direction.
    """
    violations: list[str] = []
    by_qid_rust = {q["qid"]: q for q in rust["per_question"]}
    by_qid_py = {q["qid"]: q for q in pytorch["per_question"]}

    rust_qids = set(by_qid_rust)
    py_qids = set(by_qid_py)
    if rust_qids != py_qids:
        only_rust = rust_qids - py_qids
        only_py = py_qids - rust_qids
        if only_rust:
            violations.append(
                f"QID-SET DRIFT: {len(only_rust)} qids in rust missing from pytorch "
                f"(sample: {sorted(only_rust)[:5]})"
            )
        if only_py:
            violations.append(
                f"QID-SET DRIFT: {len(only_py)} qids in pytorch missing from rust "
                f"(sample: {sorted(only_py)[:5]})"
            )
        # Even on drift we still compare the intersection, because the
        # gate is per-question monotonicity and dropping a question
        # entirely IS a drift signal the reviewer needs to see.

    intersection = sorted(rust_qids & py_qids)
    for qid in intersection:
        r = by_qid_rust[qid]
        p = by_qid_py[qid]
        if r["recall_at_5"] < p["recall_at_5"]:
            violations.append(
                f"REGRESSION qid={qid} R@5 rust={r['recall_at_5']} "
                f"< pytorch={p['recall_at_5']} (qtype={p.get('question_type', 'unknown')})"
            )
        if r["recall_at_10"] < p["recall_at_10"]:
            violations.append(
                f"REGRESSION qid={qid} R@10 rust={r['recall_at_10']} "
                f"< pytorch={p['recall_at_10']} (qtype={p.get('question_type', 'unknown')})"
            )

    # Aggregate check — even if every per-question check passed, a
    # silent aggregate drift would be a separate violation (it can't
    # actually happen if per-question all pass and qid sets match, but
    # the assertion is defensive belt-and-braces, and useful if qid
    # sets DO drift).
    if rust["aggregate"]["recall_at_5"] < pytorch["aggregate"]["recall_at_5"]:
        violations.append(
            f"REGRESSION AGGREGATE R@5: rust={rust['aggregate']['recall_at_5']:.4f} "
            f"< pytorch={pytorch['aggregate']['recall_at_5']:.4f}"
        )
    if rust["aggregate"]["recall_at_10"] < pytorch["aggregate"]["recall_at_10"]:
        violations.append(
            f"REGRESSION AGGREGATE R@10: rust={rust['aggregate']['recall_at_10']:.4f} "
            f"< pytorch={pytorch['aggregate']['recall_at_10']:.4f}"
        )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-questions",
        type=int,
        default=500,
        help=(
            "LongMemEval-S row cap forwarded as --limit to the blind "
            "harness. SPEC.md R8 target is 500 (full dataset). Smaller "
            "values are valid for smoke iteration but NOT for the final "
            "gate."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Audit metadata only -- the LongMemEval dataset order is "
            "deterministic from the adapter, and the embedders are "
            "deterministic per backend. The value is recorded in each "
            "output JSON for reproducibility."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where compare JSONs are written (default: bench/).",
    )
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help=(
            "Reuse a previously written pytorch.json instead of re-running "
            "the PyTorch backend. Debug only -- the final gate requires "
            "both backends run from clean."
        ),
    )
    parser.add_argument(
        "--skip-rust",
        action="store_true",
        help=(
            "Reuse a previously written rust.json instead of re-running "
            "the Rust backend. Debug only -- the final gate requires "
            "both backends run from clean."
        ),
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pytorch_compare_path = _compare_output_path("pytorch", out_dir)
    rust_compare_path = _compare_output_path("rust", out_dir)

    # PyTorch baseline.
    if args.skip_pytorch:
        if not pytorch_compare_path.exists():
            raise SystemExit(
                f"--skip-pytorch was passed but {pytorch_compare_path} does not exist."
            )
        print(
            f"[recall_compare] reusing cached PyTorch baseline at "
            f"{pytorch_compare_path}",
            file=sys.stderr,
        )
        pytorch_doc = json.loads(pytorch_compare_path.read_text())
    else:
        print(
            f"[recall_compare] running PyTorch backend (N={args.n_questions}) ...",
            file=sys.stderr,
            flush=True,
        )
        pytorch_doc = run_backend(
            backend="pytorch",
            n_questions=args.n_questions,
            seed=args.seed,
            out_dir=out_dir,
        )

    # Rust path.
    if args.skip_rust:
        if not rust_compare_path.exists():
            raise SystemExit(
                f"--skip-rust was passed but {rust_compare_path} does not exist."
            )
        print(
            f"[recall_compare] reusing cached Rust path at {rust_compare_path}",
            file=sys.stderr,
        )
        rust_doc = json.loads(rust_compare_path.read_text())
    else:
        print(
            f"[recall_compare] running Rust backend (N={args.n_questions}) ...",
            file=sys.stderr,
            flush=True,
        )
        rust_doc = run_backend(
            backend="rust",
            n_questions=args.n_questions,
            seed=args.seed,
            out_dir=out_dir,
        )

    # Verdict.
    n = rust_doc["n_questions"]
    py_agg = pytorch_doc["aggregate"]
    rs_agg = rust_doc["aggregate"]
    print(
        "\n=== Recall@k monotonicity gate "
        f"({n} LongMemEval-S questions, prong X = flat-cosine) ===",
        flush=True,
    )
    print(
        f"PyTorch  R@5={py_agg['recall_at_5']:.4f}  R@10={py_agg['recall_at_10']:.4f}  "
        f"errors={pytorch_doc.get('n_errors', 0)}"
    )
    print(
        f"Rust     R@5={rs_agg['recall_at_5']:.4f}  R@10={rs_agg['recall_at_10']:.4f}  "
        f"errors={rust_doc.get('n_errors', 0)}"
    )
    print(
        f"Delta    R@5={rs_agg['recall_at_5'] - py_agg['recall_at_5']:+.4f}  "
        f"R@10={rs_agg['recall_at_10'] - py_agg['recall_at_10']:+.4f}",
        flush=True,
    )

    violations = assert_monotonic(rust_doc, pytorch_doc)
    if violations:
        # Persist the violation report alongside the JSONs for audit.
        report_path = out_dir / "embedder_recall_compare.violations.txt"
        report_path.write_text("\n".join(violations) + "\n")
        print(
            f"\n*** {len(violations)} MONOTONICITY VIOLATION(S) "
            f"-- monotonicity gate FAILS ***",
            file=sys.stderr,
        )
        for v in violations[:20]:
            print(f"  {v}", file=sys.stderr)
        if len(violations) > 20:
            print(f"  ... ({len(violations) - 20} more)", file=sys.stderr)
        print(
            f"\n  Full violation report: {report_path}",
            file=sys.stderr,
        )
        print(
            "\n  Results may only RISE, never DROP. "
            "Investigate FP-determinism drift on the failing qids.",
            file=sys.stderr,
        )
        return 1

    print(
        "\n*** MONOTONIC: Rust >= PyTorch on every per-question + "
        "aggregate metric ***",
        flush=True,
    )
    print(
        f"  SPEC.md Acceptance Criteria 49a items 10 & 11: CLOSED",
        flush=True,
    )
    print(
        f"  Per-question check: {n} questions, 0 regressions",
        flush=True,
    )
    print(
        f"  Outputs: {pytorch_compare_path}",
        flush=True,
    )
    print(
        f"           {rust_compare_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
