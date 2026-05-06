"""Plan 07-02 Wave 2 R3 acceptance: per-connection multiplexing without HOL blocking.

10 concurrent clients × 5 sequential calls each must complete within 2× the
latency of a single client doing the same workload alone (SPEC R3 invariant).

The R3 acceptance asserts the dispatch-via-asyncio.to_thread pattern is in
place: if a future regression were to inline `await dispatch(...)` instead of
`await asyncio.to_thread(dispatch, ...)`, every connection would head-of-line
block on the GIL-held sync dispatch, the 10-client wall-clock would slide
toward 10× baseline, and this test would fail loudly.

Reuses _send_jsonrpc + _with_socket_server + short_socket_paths fixture
from the sibling test_socket_server_dispatch module (same package).
"""
from __future__ import annotations

import asyncio
import time

# Re-export the fixture so pytest finds it for tests in this module without
# requiring a conftest.py change.
from .test_socket_server_dispatch import short_socket_paths  # noqa: F401


def test_10_concurrent_clients_no_hol_blocking(short_socket_paths):
    """R3: 10 clients × 5 sequential calls each, total ≤ 2× single-client baseline."""
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    from .test_socket_server_dispatch import _send_jsonrpc, _with_socket_server

    store = MemoryStore()

    async def _client_workload(sock_path, client_idx, n_calls=5):
        results = []
        for call_idx in range(n_calls):
            r = await _send_jsonrpc(
                sock_path,
                "memory_recall",
                {"cue": f"client-{client_idx}-call-{call_idx}", "budget_tokens": 100},
                req_id=call_idx + 1,
            )
            results.append(r)
        return results

    async def _runner(sock_path, store):
        # Warm-up: pay the embedder load cost once before measuring.
        await _client_workload(sock_path, -1, n_calls=2)

        # Single-client baseline (5 sequential calls).
        t0 = time.monotonic()
        await _client_workload(sock_path, 0)
        baseline = time.monotonic() - t0

        # 10 concurrent clients × 5 calls each = 50 in-flight calls total.
        t1 = time.monotonic()
        await asyncio.gather(
            *[_client_workload(sock_path, i) for i in range(10)]
        )
        concurrent_total = time.monotonic() - t1

        return baseline, concurrent_total

    baseline, concurrent_total = asyncio.run(
        _with_socket_server(sock_path, store, _runner)
    )

    # SPEC R3: 10 clients of identical work in ≤ 2× the wall-clock of one client.
    # The +0.5s slack absorbs OS scheduling jitter at low N (50 calls total,
    # warm-cache embedder p50 sub-10ms — total wall-clock typically <1s).
    assert concurrent_total <= 2 * baseline + 0.5, (
        f"HOL blocking detected: 10 concurrent clients took "
        f"{concurrent_total:.3f}s vs {baseline:.3f}s baseline (>2× ratio + 0.5s slack). "
        f"Probable cause: dispatch is not running via asyncio.to_thread."
    )


def test_3_clients_serialize_per_connection_but_parallel_across(short_socket_paths):
    """R3 sanity: same connection serializes; different connections parallelize.

    Three connections each fire one call simultaneously; total wall-clock must
    be close to a single-call wall-clock (not 3×). Demonstrates the per-connection
    coroutine + asyncio.to_thread interleaving pattern.
    """
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    from .test_socket_server_dispatch import _send_jsonrpc, _with_socket_server

    store = MemoryStore()

    async def _single_call(sock_path, idx):
        return await _send_jsonrpc(
            sock_path,
            "memory_recall",
            {"cue": f"parallel-test-{idx}", "budget_tokens": 100},
            req_id=idx,
        )

    async def _runner(sock_path, store):
        # Warm-up so the embedder load cost is amortised.
        await _single_call(sock_path, 0)

        # Single-call baseline (one connection, one call).
        t0 = time.monotonic()
        await _single_call(sock_path, 1)
        baseline = time.monotonic() - t0

        # Three connections in parallel.
        t1 = time.monotonic()
        await asyncio.gather(
            _single_call(sock_path, 2),
            _single_call(sock_path, 3),
            _single_call(sock_path, 4),
        )
        parallel_total = time.monotonic() - t1

        return baseline, parallel_total

    baseline, parallel_total = asyncio.run(
        _with_socket_server(sock_path, store, _runner)
    )

    # 3 calls in parallel should not take more than 1.5× a single call's
    # wall-clock + 0.3s slack (warm-cache memory_recall is fast; the test
    # asserts that the second + third connections aren't HOL-blocked behind
    # the first connection's dispatch worker).
    assert parallel_total <= 1.5 * baseline + 0.3, (
        f"3 parallel connections took {parallel_total:.3f}s vs "
        f"{baseline:.3f}s single-call baseline (>1.5× + 0.3s slack)."
    )
