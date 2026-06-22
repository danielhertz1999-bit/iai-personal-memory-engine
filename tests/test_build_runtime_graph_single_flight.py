"""Regression test for the build_runtime_graph WAKE CPU-storm fix.

At daemon WAKE several background subsystems (boot preload, sigma identity
audit, foraging, hippea cascade) call retrieve.build_runtime_graph concurrently.
Before the fix, each one independently ran the full (GIL-bound) community
detection on a cache miss, so 3+ concurrent runs contended for the GIL, starved
the asyncio event loop, and the liveness watchdog SIGKILLed the daemon.

The single-flight gate must collapse a concurrent burst into ONE compute (the
leader), with the others reusing the freshly-saved cache — while never sharing a
mutable MemoryGraph object between callers.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

import iai_mcp.community as community
import iai_mcp.retrieve as retrieve
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _rec(seed: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface="rec", aaak_index="",
        embedding=_vec(seed), community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=0.0, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False, provenance=[], created_at=now,
        updated_at=now, tags=[], language="en",
    )


def _make_store(tmp_path: Path, monkeypatch) -> MemoryStore:
    root = tmp_path / "store"
    monkeypatch.setenv("IAI_MCP_STORE", str(root))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    return MemoryStore(path=root)


def test_concurrent_build_runtime_graph_runs_detect_once(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    for i in range(8):
        store.insert(_rec(i))

    n_threads = 4
    calls: list[int] = []
    calls_lock = threading.Lock()
    start = threading.Barrier(n_threads)

    # The expensive community detection runs in a short-lived spawn-context child
    # (_detect_communities_isolated) to keep the JIT allocator arenas out of the
    # long-lived process. The single-flight gate the WAKE CPU-storm fix added
    # collapses a concurrent cache-miss burst to ONE detection regardless of
    # where it runs, so the single-flight intent is asserted at that one
    # isolated-detection unit (a module-attribute patch on the parent's
    # detect_communities would never see the child's fresh re-import).
    orig_detect = retrieve._detect_communities_isolated

    def slow_detect(*args, **kwargs):
        with calls_lock:
            calls.append(1)
        # Hold long enough that the other callers pile up behind the leader.
        time.sleep(0.6)
        return orig_detect(*args, **kwargs)

    monkeypatch.setattr(retrieve, "_detect_communities_isolated", slow_detect)

    results: list[tuple] = []
    results_lock = threading.Lock()

    def worker():
        start.wait()
        graph, assignment, _rc = retrieve.build_runtime_graph(store)
        with results_lock:
            # Hold the graph object itself, not id(graph): a graph that fell out
            # of scope could be GC'd and its address reused by the next caller's
            # graph, making id() collide between genuinely-distinct objects.
            results.append((assignment, graph))

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert len(results) == n_threads, "some build_runtime_graph callers hung"
    # Single-flight: the expensive community detection ran exactly once even
    # though all four callers raced into a cache miss simultaneously.
    assert len(calls) == 1, (
        f"expected community detection to run once (single-flight); "
        f"ran {len(calls)} times"
    )
    # Every caller got a valid assignment...
    assert all(a is not None for a, _ in results)
    # ...and its OWN MemoryGraph object — the mutable graph (which receives the
    # store sync hook) must never be shared between concurrent callers. Compare
    # object identity directly (the graphs are kept alive in `results`).
    graph_ids = {id(g) for _, g in results}
    assert len(graph_ids) == n_threads, (
        f"callers shared a MemoryGraph object ({len(graph_ids)} distinct of "
        f"{n_threads}); the single-flight must memoise only immutable products"
    )


def test_build_runtime_graph_hit_path_skips_detect(tmp_path, monkeypatch):
    """A warm cache must short-circuit without running community detection."""
    store = _make_store(tmp_path, monkeypatch)
    for i in range(8):
        store.insert(_rec(i))

    # Prime the cache.
    retrieve.build_runtime_graph(store)

    calls: list[int] = []
    orig_detect = community.detect_communities

    def counting_detect(*args, **kwargs):
        calls.append(1)
        return orig_detect(*args, **kwargs)

    monkeypatch.setattr(community, "detect_communities", counting_detect)

    # Second call on the same generation: cache HIT, no recompute.
    graph, assignment, _rc = retrieve.build_runtime_graph(store)
    assert assignment is not None
    assert calls == [], "warm-cache build_runtime_graph must not recompute communities"
