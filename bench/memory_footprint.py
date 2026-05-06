"""M-03 RAM footprint bench. Reports RSS at store size N.

Target: RSS <= 300 MB warm at N=10k on a 16+ GB machine.

Pressplay 8 GB M1 hung mid-run on 2026-04-19 while trying to build the
runtime graph at N=10k (Pitfall 4 from 05-RESEARCH: bge-m3 ~2 GB +
NetworkX ~200 MB + LanceDB ~50 MB + Python overhead -> swap thrash).
Phase 5 measures on this 16 GB dev Mac; pressplay cross-validates at
N <= 2000 per D5-09.

JSON output (one line to stdout):

    {
      "n": int,
      "rss_mb_peak": float,           # platform-adjusted MB
      "threshold_mb": 300.0,
      "passed": bool,                 # True iff rss_mb_peak <= threshold_mb
      "platform": "darwin"|"linux"|"win32",
      "stage_ms": {"seed": float, "graph": float},
      "seed_n": int,                  # records that actually made it in
      "graph_built": bool,            # True iff build_runtime_graph finished
    }

Exit codes:
    0 if passed, 1 otherwise.

CLI:
    python -m bench.memory_footprint [--n 10000] [--dim 1024] [--seed 42]
                                     [--skip-graph]

--skip-graph keeps the RSS reading to the seeded-store baseline (no
NetworkX graph build); useful when the graph build is the timeout cause
and we want to isolate the store-only overhead.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

THRESHOLD_MB = 300.0


def _isolate_keyring_in_memory() -> None:
    """Install an in-memory keyring backend so MemoryStore's crypto layer
    never calls macOS Keychain (which hangs under SecItemCopyMatching when
    the bench is invoked from a non-interactive shell).

    Idempotent: if the current backend already has our sentinel attribute,
    it's a no-op. This is strictly bench-scope — production code paths do
    NOT touch this function.
    """
    import keyring
    from keyring.backend import KeyringBackend

    if getattr(keyring.get_keyring(), "_iai_bench_noop", False):
        return

    class _BenchNoOpKeyring(KeyringBackend):
        priority = 99
        _iai_bench_noop = True
        _kv: dict[tuple[str, str], str] = {}

        def get_password(self, service: str, username: str) -> str | None:
            return self._kv.get((service, username))

        def set_password(self, service: str, username: str, password: str) -> None:
            self._kv[(service, username)] = password

        def delete_password(self, service: str, username: str) -> None:
            self._kv.pop((service, username), None)

    keyring.set_keyring(_BenchNoOpKeyring())


def _rss_mb() -> float:
    """Peak RSS in MB, platform-adjusted.

    macOS returns ru_maxrss in BYTES.
    Linux returns ru_maxrss in KB.
    Windows via resource is not supported; the Windows branch falls back to
    a best-effort reading and the platform marker in the JSON output lets
    the report flag it.
    """
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(r) / 1024.0 / 1024.0
    # Linux reports kilobytes; everything else treated as KB for safety.
    return float(r) / 1024.0


def _make_noise_record(i: int, rng: np.random.Generator, dim: int) -> MemoryRecord:
    """Inline noise-record maker that does not pull in bench/verbatim.

    Keeps this bench self-contained so imports don't drag heavy deps.
    """
    now = datetime.now(timezone.utc)
    vec = rng.standard_normal(dim)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"bench noise record {i}",
        aaak_index="",
        embedding=vec.tolist(),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["bench", "ops-11"],
        language="en",
    )


def _seed_store(
    store: MemoryStore, n: int, dim: int, seed: int, *, concurrent: bool = False
) -> int:
    """Seed N synthetic records. Returns the count actually inserted.

    When ``concurrent`` is True, inserts are dispatched from a thread
    pool so the coalescing AsyncWriteQueue can actually batch records
    inside its 100 ms window. Sequential blocking inserts (the default
    sync path) see no coalesce benefit because each insert waits on its
    own batch flush before the next enqueue even happens.
    """
    rng = np.random.default_rng(seed)
    records = [_make_noise_record(i, rng, dim=dim) for i in range(n)]
    if not concurrent:
        for r in records:
            store.insert(r)
        return len(records)

    # Concurrent path: a thread pool fires enqueues from many threads so
    # the queue's coalesce window fills. Pool size ~256 is large enough
    # to always fill a max_batch=128 window on this hardware.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=256) as pool:
        list(pool.map(store.insert, records))
    return len(records)


def run_memory_footprint(
    n: int = 10_000,
    store_path: Path | str | None = None,
    dim: int = EMBED_DIM,
    seed: int = 42,
    *,
    skip_graph: bool = False,
    isolate_keyring: bool = True,
    async_writes: bool = False,
) -> dict:
    """Seed N records, optionally build the runtime graph, measure RSS.

    `isolate_keyring` (default True) installs an in-memory keyring backend
    so MemoryStore's crypto layer never hits macOS Keychain. Set False only
    when benching against an existing ~/.iai-mcp store whose real key lives
    in the user keyring.

    Returns a JSON-shaped dict with the keys described in the module docstring.
    """
    if isolate_keyring:
        _isolate_keyring_in_memory()

    cleanup: tempfile.TemporaryDirectory | None = None
    if store_path is None:
        cleanup = tempfile.TemporaryDirectory(prefix="iai-bench-ops11-")
        path = Path(cleanup.name)
    else:
        path = Path(store_path)
        path.mkdir(parents=True, exist_ok=True)

    # Honour the caller's --dim request by setting IAI_MCP_EMBED_DIM BEFORE
    # the MemoryStore is constructed. The store reads this env var via
    # store._resolve_embed_dim() on first table creation (see store.py:115).
    # Restore the prior value after the run so other benches/tests are not
    # contaminated.
    prev_embed_dim = os.environ.get("IAI_MCP_EMBED_DIM")
    if dim != EMBED_DIM:
        os.environ["IAI_MCP_EMBED_DIM"] = str(dim)

    try:
        store = MemoryStore(path=path)
        # Match the store's actual embed dim so inserts don't get silently
        # rejected when the env override was ignored (e.g. existing table
        # on disk pins a different dim).
        eff_dim = store.embed_dim

        # if --async-writes is set, enable the coalescing
        # write queue before the seed loop so every store.insert() below
        # routes through it. The queue is drained + torn down after the
        # seed completes, keeping the graph build / RSS reading on the
        # legacy sync path.
        if async_writes:
            import asyncio as _asyncio

            async def _enable():
                await store.enable_async_writes()

            _asyncio.run(_enable())

        t0 = time.perf_counter()
        seed_n = _seed_store(
            store, n, dim=eff_dim, seed=seed, concurrent=async_writes,
        )
        seed_ms = (time.perf_counter() - t0) * 1000.0

        if async_writes:
            import asyncio as _asyncio

            async def _disable():
                await store.disable_async_writes()

            _asyncio.run(_disable())

        graph_built = False
        graph_ms = 0.0
        if not skip_graph:
            # Lazy import so --skip-graph runs don't pay the NetworkX load.
            from iai_mcp import retrieve

            t1 = time.perf_counter()
            try:
                _graph, _assignment, _rc = retrieve.build_runtime_graph(store)
                graph_built = True
            except Exception:
                # Graph build can OOM on small hosts; surface that as the
                # diagnostic rather than crashing the bench. The RSS reading
                # still reflects peak consumed up to the failure.
                graph_built = False
            graph_ms = (time.perf_counter() - t1) * 1000.0

        gc.collect()
        rss_mb_peak = _rss_mb()

        return {
            "n": n,
            "rss_mb_peak": round(rss_mb_peak, 2),
            "threshold_mb": THRESHOLD_MB,
            "passed": rss_mb_peak <= THRESHOLD_MB,
            "platform": sys.platform,
            "stage_ms": {
                "seed": round(seed_ms, 2),
                "graph": round(graph_ms, 2),
            },
            "seed_n": seed_n,
            "graph_built": graph_built,
            "dim": eff_dim,
            "async_writes": bool(async_writes),
        }
    finally:
        # Restore IAI_MCP_EMBED_DIM so other benches / tests run with the
        # host default.
        if dim != EMBED_DIM:
            if prev_embed_dim is None:
                os.environ.pop("IAI_MCP_EMBED_DIM", None)
            else:
                os.environ["IAI_MCP_EMBED_DIM"] = prev_embed_dim
        if cleanup is not None:
            cleanup.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.memory_footprint",
        description=(
            "OPS-11 / RAM bench. Seeds N records, optionally builds "
            "the runtime graph, reports peak RSS. Target: <=300 MB at "
            "N=10k on a 16+ GB host."
        ),
    )
    parser.add_argument(
        "--n", "--n-records", dest="n", type=int, default=10_000,
        help="record count to seed (default 10000)",
    )
    parser.add_argument(
        "--dim", type=int, default=EMBED_DIM,
        help=f"embedding dimension (default {EMBED_DIM}; tests use 32/64 for speed)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed (default 42)",
    )
    parser.add_argument(
        "--skip-graph", action="store_true",
        help="Skip build_runtime_graph; isolate store-only RSS",
    )
    parser.add_argument(
        "--async-writes", action="store_true",
        help=(
            "enable MemoryStore.enable_async_writes() before the "
            "seed loop so inserts go through the coalescing AsyncWriteQueue. "
            "Target: amortise the ~0.3 MB/insert LanceDB buffer overhead by "
            "batching 128 inserts per flush."
        ),
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Write the JSON result to this file (in addition to stdout).",
    )
    args = parser.parse_args(argv)
    result = run_memory_footprint(
        n=args.n, dim=args.dim, seed=args.seed,
        skip_graph=args.skip_graph, async_writes=args.async_writes,
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh)
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
