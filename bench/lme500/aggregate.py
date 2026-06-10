"""Post-process LongMemEval-S blind-run output into report + summary JSON."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(input_path: Path) -> list[dict[str, Any]]:
    text = input_path.read_text(encoding="utf-8")
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "per_row" in data:
                return list(data["per_row"])
        except json.JSONDecodeError:
            pass
    if stripped.startswith("["):
        try:
            return list(json.loads(text))
        except json.JSONDecodeError:
            pass
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(
                f"[aggregate] WARN: skipping corrupt line {lineno}: {exc}",
                file=sys.stderr,
            )
    return rows


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = max(0, int(0.025 * n_resamples))
    hi_idx = min(n_resamples - 1, int(0.975 * n_resamples))
    return statistics.fmean(values), means[lo_idx], means[hi_idx]


def _get_prong_value(row: dict[str, Any], prong: str, k: int) -> float:
    if "error" in row and isinstance(row.get("error"), dict):
        return 0.0
    return float(row.get(f"r_at_{k}_{prong}", 0.0))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"overall": {"n": 0, "n_errors": 0}, "per_type": {}}

    by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"x5": [], "x10": [], "y5": [], "y10": []}
    )
    overall: dict[str, list[float]] = {"x5": [], "x10": [], "y5": [], "y10": []}
    n_errors = 0

    for row in rows:
        is_error = "error" in row and isinstance(row.get("error"), dict)
        if is_error:
            n_errors += 1
        qtype = str(row.get("question_type", "unknown"))
        x5 = _get_prong_value(row, "retrieve", 5)
        x10 = _get_prong_value(row, "retrieve", 10)
        y5 = _get_prong_value(row, "pipeline", 5)
        y10 = _get_prong_value(row, "pipeline", 10)
        overall["x5"].append(x5)
        overall["x10"].append(x10)
        overall["y5"].append(y5)
        overall["y10"].append(y10)
        by_type[qtype]["x5"].append(x5)
        by_type[qtype]["x10"].append(x10)
        by_type[qtype]["y5"].append(y5)
        by_type[qtype]["y10"].append(y10)

    def _prong_block(vals_5: list[float], vals_10: list[float]) -> dict:
        m5, lo5, hi5 = bootstrap_ci(vals_5)
        m10, lo10, hi10 = bootstrap_ci(vals_10)
        return {
            "r_at_5": {"mean": m5, "ci_lo": lo5, "ci_hi": hi5},
            "r_at_10": {"mean": m10, "ci_lo": lo10, "ci_hi": hi10},
        }

    overall_block = {
        "n": len(rows),
        "n_errors": n_errors,
        "X_retrieve": _prong_block(overall["x5"], overall["x10"]),
        "Y_pipeline": _prong_block(overall["y5"], overall["y10"]),
    }
    overall_block["lift_Y_minus_X"] = {
        "r_at_5": (
            overall_block["Y_pipeline"]["r_at_5"]["mean"]
            - overall_block["X_retrieve"]["r_at_5"]["mean"]
        ),
        "r_at_10": (
            overall_block["Y_pipeline"]["r_at_10"]["mean"]
            - overall_block["X_retrieve"]["r_at_10"]["mean"]
        ),
    }

    per_type_out: dict[str, dict[str, Any]] = {}
    for qt in sorted(by_type.keys()):
        data = by_type[qt]
        block = {
            "n": len(data["x5"]),
            "X_retrieve": _prong_block(data["x5"], data["x10"]),
            "Y_pipeline": _prong_block(data["y5"], data["y10"]),
        }
        block["lift_Y_minus_X"] = {
            "r_at_5": (
                block["Y_pipeline"]["r_at_5"]["mean"]
                - block["X_retrieve"]["r_at_5"]["mean"]
            ),
            "r_at_10": (
                block["Y_pipeline"]["r_at_10"]["mean"]
                - block["X_retrieve"]["r_at_10"]["mean"]
            ),
        }
        per_type_out[qt] = block

    return {"overall": overall_block, "per_type": per_type_out}


def format_markdown_report(agg: dict[str, Any], source_path: Path) -> str:
    overall = agg["overall"]
    lines: list[str] = []
    lines.append("# LongMemEval-S Aggregate Report")
    lines.append("")
    lines.append(f"- Source: `{source_path}`")
    lines.append(f"- n = {overall['n']}, errors = {overall['n_errors']}")
    lines.append(
        "- 95% CI via bootstrap percentile method (10000 resamples, seed=42)"
    )
    lines.append("")

    if overall["n"] == 0:
        lines.append("**No rows loaded.**")
        return "\n".join(lines) + "\n"

    lines.append("## Overall")
    lines.append("")
    lines.append("| Prong | R@5 | R@5 95% CI | R@10 | R@10 95% CI |")
    lines.append("|---|---|---|---|---|")
    x = overall["X_retrieve"]
    y = overall["Y_pipeline"]
    lift = overall["lift_Y_minus_X"]
    lines.append(
        f"| X (retrieve_recall — flat-cosine baseline) "
        f"| {x['r_at_5']['mean']:.3f} "
        f"| [{x['r_at_5']['ci_lo']:.3f}, {x['r_at_5']['ci_hi']:.3f}] "
        f"| {x['r_at_10']['mean']:.3f} "
        f"| [{x['r_at_10']['ci_lo']:.3f}, {x['r_at_10']['ci_hi']:.3f}] |"
    )
    lines.append(
        f"| Y (recall_for_benchmark — full graph-native pipeline) "
        f"| {y['r_at_5']['mean']:.3f} "
        f"| [{y['r_at_5']['ci_lo']:.3f}, {y['r_at_5']['ci_hi']:.3f}] "
        f"| {y['r_at_10']['mean']:.3f} "
        f"| [{y['r_at_10']['ci_lo']:.3f}, {y['r_at_10']['ci_hi']:.3f}] |"
    )
    lines.append(
        f"| **Architecture lift Y − X** "
        f"| **{lift['r_at_5']:+.3f}** "
        f"| — "
        f"| **{lift['r_at_10']:+.3f}** "
        f"| — |"
    )
    lines.append("")

    lines.append("## Per question type")
    lines.append("")
    lines.append(
        "| Type | n | X R@5 | Y R@5 | Lift R@5 "
        "| X R@10 | Y R@10 | Lift R@10 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for qt, block in agg["per_type"].items():
        n = block["n"]
        flag = " ⚠️" if n < 30 else ""
        x = block["X_retrieve"]
        y = block["Y_pipeline"]
        lift = block["lift_Y_minus_X"]
        lines.append(
            f"| `{qt}`{flag} | {n} "
            f"| {x['r_at_5']['mean']:.3f} | {y['r_at_5']['mean']:.3f} "
            f"| {lift['r_at_5']:+.3f} "
            f"| {x['r_at_10']['mean']:.3f} | {y['r_at_10']['mean']:.3f} "
            f"| {lift['r_at_10']:+.3f} |"
        )
    lines.append("")
    lines.append("⚠️ = n < 30, low statistical power for that bin.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Errors (graph-build failures, malformed rows, etc.) are counted "
        "as miss for **both** prongs (R@k = 0)."
    )
    lines.append(
        "- Mean is the unweighted row average; CI is bootstrap percentile."
    )
    lines.append(
        "- Architecture lift = mean(Y) − mean(X). The CI of the lift "
        "itself is not computed here (would require paired bootstrap on "
        "the (Y_i, X_i) tuples — TODO if needed)."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="input",
        required=True,
        help="Path to per-row JSON / JSONL file",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Output path for markdown report; default: <input>-report.md",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Output path for aggregated JSON; default: <input>-summary.json",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[aggregate] ERROR: {input_path} does not exist", file=sys.stderr)
        return 1
    rows = load_rows(input_path)
    if not rows:
        print(f"[aggregate] WARN: 0 rows loaded from {input_path}", file=sys.stderr)
        return 1

    agg = aggregate(rows)

    summary_path = (
        Path(args.summary)
        if args.summary
        else input_path.with_name(input_path.stem + "-summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    report_path = (
        Path(args.report)
        if args.report
        else input_path.with_name(input_path.stem + "-report.md")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_markdown_report(agg, input_path), encoding="utf-8")

    overall = agg["overall"]
    x = overall["X_retrieve"]
    y = overall["Y_pipeline"]
    lift = overall["lift_Y_minus_X"]
    print(
        f"[aggregate] n={overall['n']} errors={overall['n_errors']}",
        file=sys.stderr,
    )
    print(
        f"[aggregate] X (retrieve)  R@5={x['r_at_5']['mean']:.3f} "
        f"[{x['r_at_5']['ci_lo']:.3f},{x['r_at_5']['ci_hi']:.3f}]  "
        f"R@10={x['r_at_10']['mean']:.3f}",
        file=sys.stderr,
    )
    print(
        f"[aggregate] Y (pipeline)  R@5={y['r_at_5']['mean']:.3f} "
        f"[{y['r_at_5']['ci_lo']:.3f},{y['r_at_5']['ci_hi']:.3f}]  "
        f"R@10={y['r_at_10']['mean']:.3f}",
        file=sys.stderr,
    )
    print(
        f"[aggregate] Lift Y − X    R@5={lift['r_at_5']:+.3f}  "
        f"R@10={lift['r_at_10']:+.3f}",
        file=sys.stderr,
    )
    print(f"[aggregate] -> {summary_path}", file=sys.stderr)
    print(f"[aggregate] -> {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
