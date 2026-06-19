"""Capture dedup-lock overhead: capture_turn() solo latency and concurrent throughput.

Quantifies the cost of _CAPTURE_DEDUP_LOCK (the fix for the daemon's
concurrent-thread capture_turn() dedup race -- see capture.py). All
captures used here have distinct content, so every call falls through to
an actual insert; the question this bench answers is how much serializing
the dedup-check-through-insert critical section costs, both with zero
contention (solo) and under real thread contention (concurrent).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import os
if not os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
    os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = (
        "iai-mcp-bench-falsifiability-deterministic-2026"
    )

from iai_mcp.capture import capture_turn
from iai_mcp.store import MemoryStore

SESSION_ID = "bench-capture-dedup-lock"
MIN_TEXT = (
    "with enough length to clear the MIN_CAPTURE_LEN floor for a realistic capture"
)


def _solo_latency(store: MemoryStore, n: int) -> dict:
    samples_ms: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        capture_turn(
            store,
            cue=f"bench solo turn {i}",
            text=f"Capture dedup-lock bench solo turn {i} {MIN_TEXT}.",
            tier="episodic",
            session_id=SESSION_ID,
            role="user",
            source_uuid=str(uuid4()),
        )
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    samples_ms.sort()
    return {
        "n": n,
        "p50_ms": statistics.median(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "p95_ms": samples_ms[max(0, int(round(0.95 * (n - 1))))],
    }


def _concurrent_throughput(store: MemoryStore, n_threads: int, per_thread: int) -> dict:
    def worker(thread_idx: int) -> None:
        for i in range(per_thread):
            capture_turn(
                store,
                cue=f"bench concurrent t{thread_idx} turn {i}",
                text=(
                    f"Capture dedup-lock bench concurrent thread {thread_idx} "
                    f"turn {i} {MIN_TEXT}."
                ),
                tier="episodic",
                session_id=SESSION_ID,
                role="user",
                source_uuid=str(uuid4()),
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    total = n_threads * per_thread
    return {
        "n_threads": n_threads,
        "per_thread": per_thread,
        "total_captures": total,
        "elapsed_s": elapsed,
        "throughput_per_s": (total / elapsed) if elapsed > 0 else float("inf"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--solo-n", type=int, default=60,
        help="Number of single-threaded distinct captures (default: 60).",
    )
    parser.add_argument(
        "--threads", type=int, default=8,
        help="Concurrent threads for the throughput phase (default: 8).",
    )
    parser.add_argument(
        "--per-thread", type=int, default=20,
        help="Distinct captures per thread (default: 20).",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(path=Path(td))

        solo = _solo_latency(store, args.solo_n)
        print(
            f"solo capture_turn latency (n={solo['n']}, distinct content, "
            f"zero lock contention): "
            f"p50={solo['p50_ms']:.2f}ms mean={solo['mean_ms']:.2f}ms "
            f"p95={solo['p95_ms']:.2f}ms"
        )

        conc = _concurrent_throughput(store, args.threads, args.per_thread)
        print(
            f"concurrent throughput ({conc['n_threads']} threads x "
            f"{conc['per_thread']} distinct captures each): "
            f"{conc['throughput_per_s']:.1f} captures/sec "
            f"({conc['total_captures']} total in {conc['elapsed_s']:.2f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
