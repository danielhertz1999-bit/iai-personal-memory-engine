from __future__ import annotations

import asyncio
import time

import pytest

from _perf_helpers import skip_if_loaded

from .test_socket_server_dispatch import short_socket_paths  # noqa: F401

@pytest.mark.perf
def test_10_concurrent_clients_no_hol_blocking(short_socket_paths):
    skip_if_loaded()
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
        await _client_workload(sock_path, -1, n_calls=2)

        t0 = time.monotonic()
        await _client_workload(sock_path, 0)
        baseline = time.monotonic() - t0

        t1 = time.monotonic()
        await asyncio.gather(
            *[_client_workload(sock_path, i) for i in range(10)]
        )
        concurrent_total = time.monotonic() - t1

        return baseline, concurrent_total

    baseline, concurrent_total = asyncio.run(
        _with_socket_server(sock_path, store, _runner)
    )

    assert concurrent_total <= 2 * baseline + 0.5, (
        f"HOL blocking detected: 10 concurrent clients took "
        f"{concurrent_total:.3f}s vs {baseline:.3f}s baseline (>2× ratio + 0.5s slack). "
        f"Probable cause: dispatch is not running via asyncio.to_thread."
    )

@pytest.mark.perf
def test_3_clients_serialize_per_connection_but_parallel_across(short_socket_paths):
    skip_if_loaded()
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
        await _single_call(sock_path, 0)

        t0 = time.monotonic()
        await _single_call(sock_path, 1)
        baseline = time.monotonic() - t0

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

    assert parallel_total <= 1.5 * baseline + 0.3, (
        f"3 parallel connections took {parallel_total:.3f}s vs "
        f"{baseline:.3f}s single-call baseline (>1.5× + 0.3s slack)."
    )
