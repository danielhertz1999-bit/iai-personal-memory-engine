"""latency gate: Rust ≤ PyTorch / 2 on single-embed p50 AND p95.

Methodology:
  - Fixed 50 baseline texts (deterministic — texts.json prefix).
  - Cold-load timing captured separately (model construction + 1st encode).
    Cold is informational only — NOT a gate input.
  - Warm-up phase: discard the first ``n_warmup`` encodes (BLAS/cache settling).
  - Measurement: run ``n_rounds`` rounds of ``n_samples`` encodes each.
    Per-round percentiles are computed independently, then the cross-round
    MEDIAN is the reported number — robust to OS jitter / spurious spikes.
  - When ``n_rounds == 1``, the per-round percentile IS the reported value
    (no median over a single value needed; the script supports this so a
    spawn-context "200-sample sustained" run is equivalent to 4 × 50 rounds
    statistically while keeping the schema constant).

JSON contract (consumed by 49-VERIFICATION-49b.md + gate() in this script):
  - p50_ms, p95_ms, p99_ms # the gate inputs
  - samples_ms # flat list, all rounds, for audit
  - n_warmup, n_samples_per_round,
    n_rounds # methodology metadata
  - cold_ms # informational: 1st-encode incl. cache warm
  - platform, daemon_running_at_start,
    generated_at_utc # environment metadata

Subprocess isolation: outer process orchestrates; per-backend inner subprocess
loads ``iai_mcp.embed.Embedder`` exactly once with the backend env set. This
prevents ``_MODEL_CACHE`` cross-contamination.

Usage:
  python bench/embedder_latency.py # both, defaults
  python bench/embedder_latency.py --backend rust
  python bench/embedder_latency.py --backend pytorch
  python bench/embedder_latency.py --n-warmup 20 --n-samples 200 --n-rounds 1
"""

from __future__ import annotations

import argparse
import json
import os
import platform as platform_mod
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Worktree-resolution shim: lazy iai_mcp.embed import needs src/ on sys.path
# when the script is invoked directly from a fresh worktree.
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

REPO = Path(__file__).resolve().parent.parent
BASELINE_TEXTS = REPO / "bench" / "embedder_baseline" / "texts.json"

# Default output paths (overridable via --out-dir for ad-hoc runs).
PYTORCH_OUT = REPO / "bench" / "embedder_latency.pytorch.json"
RUST_OUT = REPO / "bench" / "embedder_latency.rust.json"

# Default methodology. CLI overrides supported.
DEFAULT_N_WARMUP = 5
DEFAULT_N_SAMPLES = 50
DEFAULT_N_ROUNDS = 3


def daemon_running() -> bool:
    """Cheap probe: socket file exists AND pid file claims a live process.

    Returns False on any uncertainty — the bench refuses to run when this
    returns True, so a False-negative is a safety-on default, not silent skip.
    """
    sock = Path.home() / ".iai-mcp" / ".daemon.sock"
    state = Path.home() / ".iai-mcp" / ".daemon-state.json"
    if not sock.exists() or not state.exists():
        return False
    try:
        doc = json.loads(state.read_text())
        pid = doc.get("daemon_pid")
        if pid is None:
            return False
        # crude pid liveness check (signal 0 = "does the process exist?")
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    except Exception:
        return False


def measure_in_subprocess(
    backend: str,
    n_warmup: int,
    n_samples: int,
    n_rounds: int,
) -> dict:
    """Run measurement loop in a fresh subprocess with the backend env set."""
    env = os.environ.copy()
    env["IAI_MCP_EMBED_BACKEND"] = backend
    # Inner-subprocess sentinel; same script, different entry point.
    env["IAI_MCP_LATENCY_INNER"] = "1"
    env["IAI_MCP_LATENCY_BACKEND"] = backend
    env["IAI_MCP_LATENCY_N_WARMUP"] = str(n_warmup)
    env["IAI_MCP_LATENCY_N_SAMPLES"] = str(n_samples)
    env["IAI_MCP_LATENCY_N_ROUNDS"] = str(n_rounds)
    cmd = [sys.executable, str(Path(__file__))]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(
            f"latency subprocess for backend={backend} exited {result.returncode}"
        )
    # The inner subprocess prints exactly one JSON blob as the LAST stdout line.
    # (sentence-transformers prints loader progress on stderr, not stdout,
    # but we still defensively pick `splitlines()[-1]` rather than parse all.)
    try:
        blob_line = result.stdout.strip().splitlines()[-1]
        return json.loads(blob_line)
    except Exception as exc:
        raise SystemExit(
            f"could not parse inner stdout JSON: {exc}\n"
            f"stdout tail: {result.stdout[-1500:]}\n"
            f"stderr tail: {result.stderr[-1500:]}"
        )


def _percentile(sorted_samples: list[float], p: float) -> float:
    """p in [0, 1]. Nearest-rank percentile on a sorted list."""
    if not sorted_samples:
        raise ValueError("empty samples")
    idx = max(0, min(len(sorted_samples) - 1, int(round(p * len(sorted_samples))) - 1))
    return sorted_samples[idx]


def measure_latency_inner(
    backend: str,
    n_warmup: int,
    n_samples: int,
    n_rounds: int,
) -> dict:
    """Inner-subprocess entrypoint — actually load the Embedder and time encodes.

    Returns the dict the outer process writes to its JSON output file. The last
    line of stdout is `json.dumps(...)` of this dict (IPC contract).
    """
    all_texts = json.loads(BASELINE_TEXTS.read_text())
    # Deterministic prefix per PLAN. If n_samples > len(all_texts), cycle the
    # prefix (still deterministic — first 100 baseline texts repeated). The
    # latency distribution shape is unchanged by repetition since each encode
    # is independent w.r.t. internal state (no KV-cache in BERT inference).
    if n_samples <= len(all_texts):
        texts = all_texts[:n_samples]
    else:
        texts = [all_texts[i % len(all_texts)] for i in range(n_samples)]

    # Cold-load timing: import + Embedder() construction. Informational only.
    t_cold0 = time.perf_counter_ns()
    from iai_mcp.embed import Embedder

    e = Embedder()
    # Fail-loud if the env routing didn't take.
    if e._backend != backend:
        raise SystemExit(
            f"backend env mismatch: IAI_MCP_EMBED_BACKEND={backend} but "
            f"Embedder._backend={e._backend} — check env propagation"
        )
    # First encode — captures first-call code-path overhead (BLAS init, allocator
    # warmup, model lazy materialization on some paths).
    _ = e.embed(texts[0])
    t_cold1 = time.perf_counter_ns()
    cold_ms = (t_cold1 - t_cold0) / 1_000_000.0

    # Warmup — discard.
    for t in texts[: max(0, min(n_warmup, len(texts)))]:
        e.embed(t)

    # Measurement.
    all_samples_ms: list[float] = []
    per_round_p50: list[float] = []
    per_round_p95: list[float] = []
    per_round_p99: list[float] = []
    for _round in range(n_rounds):
        round_samples: list[float] = []
        for t in texts:
            t0 = time.perf_counter_ns()
            _ = e.embed(t)
            t1 = time.perf_counter_ns()
            round_samples.append((t1 - t0) / 1_000_000.0)
        all_samples_ms.extend(round_samples)
        rs = sorted(round_samples)
        per_round_p50.append(_percentile(rs, 0.50))
        per_round_p95.append(_percentile(rs, 0.95))
        per_round_p99.append(_percentile(rs, 0.99))

    # Cross-round median = jitter-robust point estimate.
    p50 = statistics.median(per_round_p50)
    p95 = statistics.median(per_round_p95)
    p99 = statistics.median(per_round_p99)

    return {
        "backend": backend,
        "n_warmup": n_warmup,
        "n_samples_per_round": n_samples,
        "n_rounds": n_rounds,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "cold_ms": cold_ms,
        "per_round_p50_ms": per_round_p50,
        "per_round_p95_ms": per_round_p95,
        "per_round_p99_ms": per_round_p99,
        "samples_ms": all_samples_ms,
        "platform": platform_mod.platform(),
        "daemon_running_at_start": daemon_running(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def gate(rust: dict, pytorch: dict) -> tuple[bool, list[str], dict]:
    """Returns (passed, violations, ratios)."""
    violations: list[str] = []
    ratio_p50 = pytorch["p50_ms"] / rust["p50_ms"] if rust["p50_ms"] > 0 else float("inf")
    ratio_p95 = pytorch["p95_ms"] / rust["p95_ms"] if rust["p95_ms"] > 0 else float("inf")
    if not (rust["p50_ms"] <= pytorch["p50_ms"] / 2.0):
        violations.append(
            f"GATE FAIL p50: rust={rust['p50_ms']:.2f}ms > pytorch/2="
            f"{pytorch['p50_ms']/2.0:.2f}ms  (pytorch={pytorch['p50_ms']:.2f}ms, "
            f"ratio={ratio_p50:.2f}×)"
        )
    if not (rust["p95_ms"] <= pytorch["p95_ms"] / 2.0):
        violations.append(
            f"GATE FAIL p95: rust={rust['p95_ms']:.2f}ms > pytorch/2="
            f"{pytorch['p95_ms']/2.0:.2f}ms  (pytorch={pytorch['p95_ms']:.2f}ms, "
            f"ratio={ratio_p95:.2f}×)"
        )
    return (not violations), violations, {"p50": ratio_p50, "p95": ratio_p95}


def main_outer() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend",
        choices=["both", "rust", "pytorch"],
        default="both",
        help="Which backend(s) to measure (default: both).",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=DEFAULT_N_WARMUP,
        help=f"Warmup encodes (discarded) before measurement (default: {DEFAULT_N_WARMUP}).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Samples per measurement round (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=DEFAULT_N_ROUNDS,
        help=f"Number of measurement rounds (default: {DEFAULT_N_ROUNDS}; "
        "set to 1 for a single sustained run).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory (default: bench/).",
    )
    args = parser.parse_args()

    if daemon_running():
        sys.stderr.write(
            "ERROR: daemon is running. This gate requires daemon OFF for "
            "clean measurement (Hippo/lifecycle interference distorts latency).\n"
            "Run `iai-mcp daemon stop` and re-try.\n"
        )
        return 2

    out_dir = args.out_dir if args.out_dir is not None else (REPO / "bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    pytorch_out = (out_dir / PYTORCH_OUT.name) if args.out_dir else PYTORCH_OUT
    rust_out = (out_dir / RUST_OUT.name) if args.out_dir else RUST_OUT

    runs: dict[str, dict] = {}
    if args.backend in ("both", "pytorch"):
        print("[latency] measuring PyTorch path...", file=sys.stderr)
        wall0 = time.perf_counter()
        runs["pytorch"] = measure_in_subprocess(
            "pytorch", args.n_warmup, args.n_samples, args.n_rounds
        )
        runs["pytorch"]["wall_seconds"] = time.perf_counter() - wall0
        pytorch_out.write_text(json.dumps(runs["pytorch"], indent=2))
        print(
            f"[latency] PyTorch done in {runs['pytorch']['wall_seconds']:.1f}s "
            f"→ {pytorch_out}",
            file=sys.stderr,
        )
    if args.backend in ("both", "rust"):
        print("[latency] measuring Rust path...", file=sys.stderr)
        wall0 = time.perf_counter()
        runs["rust"] = measure_in_subprocess(
            "rust", args.n_warmup, args.n_samples, args.n_rounds
        )
        runs["rust"]["wall_seconds"] = time.perf_counter() - wall0
        rust_out.write_text(json.dumps(runs["rust"], indent=2))
        print(
            f"[latency] Rust done in {runs['rust']['wall_seconds']:.1f}s "
            f"→ {rust_out}",
            file=sys.stderr,
        )

    if args.backend == "both":
        r, p = runs["rust"], runs["pytorch"]
        print()
        print(
            f"PyTorch  p50={p['p50_ms']:.2f}ms  p95={p['p95_ms']:.2f}ms  "
            f"p99={p['p99_ms']:.2f}ms  cold={p['cold_ms']:.0f}ms"
        )
        print(
            f"Rust     p50={r['p50_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms  "
            f"p99={r['p99_ms']:.2f}ms  cold={r['cold_ms']:.0f}ms"
        )
        ok, violations, ratios = gate(r, p)
        print(
            f"Ratios   p50={ratios['p50']:.2f}×  p95={ratios['p95']:.2f}×  "
            f"(gate: ≥ 2.0× on both)"
        )
        if not ok:
            print()
            for v in violations:
                print(f"  {v}", file=sys.stderr)
            print("\n*** FAIL ***", file=sys.stderr)
            return 1
        print("\n*** PASS: Rust ≥ 2× PyTorch on both p50 and p95 ***")
    return 0


def main_inner() -> int:
    backend = os.environ["IAI_MCP_LATENCY_BACKEND"]
    n_warmup = int(os.environ.get("IAI_MCP_LATENCY_N_WARMUP", DEFAULT_N_WARMUP))
    n_samples = int(os.environ.get("IAI_MCP_LATENCY_N_SAMPLES", DEFAULT_N_SAMPLES))
    n_rounds = int(os.environ.get("IAI_MCP_LATENCY_N_ROUNDS", DEFAULT_N_ROUNDS))
    result = measure_latency_inner(backend, n_warmup, n_samples, n_rounds)
    # Last stdout line MUST be the JSON blob for IPC parse in main_outer.
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    if os.environ.get("IAI_MCP_LATENCY_INNER") == "1":
        raise SystemExit(main_inner())
    else:
        raise SystemExit(main_outer())
