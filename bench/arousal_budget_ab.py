#!/usr/bin/env python3
"""ship-gate harness — runs contradiction_longitudinal_claude.py
with arousal_route attribution captured, then invokes
bench/analyze_arousal_ab.py for the ship-gate verdict.

Mirrors EFE A/B harness pattern. Threshold +0.05 (vs EFE's
+0.10) reflects smaller-effect-size hypothesis for budget-tokens variation.

A/B route lives inside `_recall_core`, observable from both
`recall_for_response` (production) and `recall_for_benchmark` (bench). All
three RetrievalParams fields (max_hops, rank_threshold, mode) are plumbed;
under default ArousalState (level=0.5 -> balanced regime) only rank_threshold
has measurable effect on the bench (~0.45 cosine floor on cosine_top_indices).

Usage:
    python bench/arousal_budget_ab.py \\
        --output-dir bench/results//iteration-25-arousal-ab \\
        --seeds 13 42 137 \\
        --scale honest
"""
from __future__ import annotations

# No `iai_mcp.*` import in this harness (pure subprocess wrapper; mirrors
# bench/contradiction_longitudinal.py wrapper-only pattern).
# sys.path shim block intentionally OMITTED — see
# tests/test_bench_worktree_resolution.py:191 contract: BENCH_SCRIPTS_NO_SHIM
# scripts MUST NOT carry the shim. The harness only invokes contradiction_
# longitudinal_claude.py + analyze_arousal_ab.py via subprocess; both child
# scripts handle their own iai_mcp resolution (the child bench script has
# the shim per BENCH_SCRIPTS_NEEDING_SHIM contract).

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="arousal_budget A/B ship-gate harness",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 137])
    parser.add_argument(
        "--scale", default="honest",
        choices=["smoke", "mvp", "honest", "stress"],
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Ship-gate threshold (delta_rescue >= threshold -> keep)",
    )
    parser.add_argument(
        "--store-dir", default=None,
        help=(
            "Bench-isolated IAI_MCP_STORE path passed through to inner bench. "
            "Default lets the inner bench use its DEFAULT_STORE_DIR "
            "(/tmp/iai-mcp-bench-claude/store)."
        ),
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    worktree_root = Path(__file__).resolve().parent.parent

    # Single run produces CSV with arousal_route attribution
    # (env var unset = MD5-routed inside _recall_core).
    env = os.environ.copy()
    env.pop("IAI_MCP_AROUSAL_USE_SHADOW", None)
    cmd = [
        sys.executable,
        str(worktree_root / "bench" / "contradiction_longitudinal_claude.py"),
        "--output-dir", str(args.output_dir),
        "--scale", args.scale,
        "--seeds", *(str(s) for s in args.seeds),
    ]
    if args.store_dir:
        cmd += ["--store-dir", args.store_dir]
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        return rc

    # Analyze the produced CSV.
    return subprocess.run(
        [
            sys.executable,
            str(worktree_root / "bench" / "analyze_arousal_ab.py"),
            str(args.output_dir),
            "--threshold", str(args.threshold),
        ],
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
