from __future__ import annotations

import asyncio
import pytest

from iai_mcp.write_queue import AsyncWriteQueue

class MockAsyncTable:

    def __init__(self, *, raise_on_add: BaseException | None = None) -> None:
        self.calls: list[list] = []
        self.raise_on_add = raise_on_add
        self._delay_s: float = 0.0

    async def add(self, batch) -> None:
        self.calls.append(list(batch))
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if self.raise_on_add is not None:
            raise self.raise_on_add

def test_single_enqueue_flushes_within_coalesce_window():
    table = MockAsyncTable()

    async def run() -> None:
        q = AsyncWriteQueue(table, coalesce_ms=50, max_batch=128)
        await q.start()
        try:
            fut = await q.enqueue({"id": "r1"})
            await asyncio.wait_for(fut, timeout=0.5)
        finally:
            await q.stop()

    asyncio.run(run())
    assert len(table.calls) == 1
    assert len(table.calls[0]) == 1
    assert table.calls[0][0]["id"] == "r1"

def test_coalesce_window_batches_concurrent_enqueues():
    table = MockAsyncTable()

    async def run() -> None:
        q = AsyncWriteQueue(table, coalesce_ms=80, max_batch=128)
        await q.start()
        try:
            fut1 = await q.enqueue({"id": "r1"})
            fut2 = await q.enqueue({"id": "r2"})
            await asyncio.wait_for(asyncio.gather(fut1, fut2), timeout=0.5)
        finally:
            await q.stop()

    asyncio.run(run())
    assert len(table.calls) == 1, f"expected one batched add, got {len(table.calls)}"
    ids = [r["id"] for r in table.calls[0]]
    assert ids == ["r1", "r2"]

def test_max_batch_splits_into_two_flushes():
    table = MockAsyncTable()

    async def run() -> None:
        q = AsyncWriteQueue(table, coalesce_ms=50, max_batch=4)
        await q.start()
        try:
            futs = [await q.enqueue({"id": f"r{i}"}) for i in range(5)]
            await asyncio.wait_for(asyncio.gather(*futs), timeout=1.0)
        finally:
            await q.stop()

    asyncio.run(run())
    batch_sizes = [len(c) for c in table.calls]
    assert sum(batch_sizes) == 5
    assert len(table.calls) >= 2
    assert all(sz <= 4 for sz in batch_sizes)

def test_stop_drains_pending_records():
    table = MockAsyncTable()

    async def run() -> None:
        q = AsyncWriteQueue(table, coalesce_ms=30, max_batch=128)
        await q.start()
        fut = await q.enqueue({"id": "r1"})
        await q.stop()
        assert fut.done(), "stop() must drain pending futures"
        with pytest.raises(RuntimeError):
            await q.enqueue({"id": "r2"})

    asyncio.run(run())
    assert sum(len(c) for c in table.calls) == 1

def test_backpressure_awaits_when_queue_full():
    table = MockAsyncTable()
    table._delay_s = 0.05

    async def run() -> int:
        q = AsyncWriteQueue(
            table, coalesce_ms=30, max_batch=2, max_queue_size=2,
        )
        await q.start()
        try:
            f1 = await q.enqueue({"id": "r1"})
            f2 = await q.enqueue({"id": "r2"})
            t0 = asyncio.get_event_loop().time()
            f3 = await q.enqueue({"id": "r3"})
            waited = asyncio.get_event_loop().time() - t0
            await asyncio.gather(f1, f2, f3)
            return 1 if waited >= 0.01 else 0
        finally:
            await q.stop()

    waited_flag = asyncio.run(run())
    assert waited_flag == 1, "back-pressure enqueue must await at least one flush"

def test_on_flushed_fires_per_record_in_batch_order():
    table = MockAsyncTable()
    flushed: list[dict] = []

    def on_flushed(batch):
        flushed.extend(batch)

    async def run() -> None:
        q = AsyncWriteQueue(
            table, coalesce_ms=40, max_batch=128, on_flushed=on_flushed,
        )
        await q.start()
        try:
            futs = [await q.enqueue({"id": f"r{i}"}) for i in range(3)]
            await asyncio.gather(*futs)
        finally:
            await q.stop()

    asyncio.run(run())
    assert [r["id"] for r in flushed] == ["r0", "r1", "r2"]

def test_flush_exception_propagates_to_all_futures_in_batch():
    err = RuntimeError("lancedb boom")
    table = MockAsyncTable(raise_on_add=err)

    async def run() -> tuple[list, MockAsyncTable]:
        q = AsyncWriteQueue(table, coalesce_ms=30, max_batch=128)
        await q.start()
        try:
            f1 = await q.enqueue({"id": "r1"})
            f2 = await q.enqueue({"id": "r2"})
            results = []
            for f in (f1, f2):
                try:
                    await f
                    results.append(None)
                except RuntimeError as exc:
                    results.append(exc)

            table.raise_on_add = None
            f3 = await q.enqueue({"id": "r3"})
            await f3
            return results, table
        finally:
            await q.stop()

    results, _ = asyncio.run(run())
    assert len(results) == 2
    assert all(isinstance(r, RuntimeError) and str(r) == "lancedb boom" for r in results)
