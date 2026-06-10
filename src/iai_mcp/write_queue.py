from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

__all__ = ["AsyncWriteQueue"]


class AsyncWriteQueue:

    def __init__(
        self,
        table: Any,
        *,
        coalesce_ms: int = 100,
        max_batch: int = 128,
        max_queue_size: int = 4096,
        on_flushed: Optional[Callable[[list], None]] = None,
        pre_flush_gate: Optional[Callable[[list], list]] = None,
    ) -> None:
        self._table = table
        self._coalesce_s: float = max(coalesce_ms, 1) / 1000.0
        self._max_batch: int = max(max_batch, 1)
        self._max_queue_size: int = max(max_queue_size, 1)
        self._on_flushed = on_flushed
        self._pre_flush_gate = pre_flush_gate

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._flush_event: Optional[asyncio.Event] = None
        self._coalesce_task: Optional[asyncio.Task] = None
        self._stopping: bool = False
        self._stopped: bool = False


    async def start(self) -> None:
        if self._coalesce_task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._flush_event = asyncio.Event()
        self._stopping = False
        self._stopped = False
        self._coalesce_task = asyncio.create_task(
            self._coalesce_loop(), name="iai-mcp-write-coalesce"
        )

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopping = True
        assert self._queue is not None
        await self._queue.put(_SENTINEL)
        if self._coalesce_task is not None:
            await self._coalesce_task
            self._coalesce_task = None
        self._stopped = True


    async def enqueue(self, record: Any) -> asyncio.Future:
        if self._stopped or self._stopping:
            raise RuntimeError("AsyncWriteQueue is stopped; cannot enqueue")
        assert self._queue is not None and self._flush_event is not None

        while self._queue.qsize() >= self._max_queue_size:
            self._flush_event.clear()
            await self._flush_event.wait()

        fut: asyncio.Future = self._loop.create_future()  # type: ignore[union-attr]
        await self._queue.put(_Pending(record=record, future=fut))
        return fut


    async def _coalesce_loop(self) -> None:
        assert self._queue is not None and self._flush_event is not None
        while True:
            batch: list[_Pending] = []
            first = await self._queue.get()
            if first is _SENTINEL:
                while not self._queue.empty():
                    item = self._queue.get_nowait()
                    if item is _SENTINEL:
                        continue
                    batch.append(item)
                if batch:
                    await self._flush(batch)
                return
            batch.append(first)

            deadline = self._loop.time() + self._coalesce_s  # type: ignore[union-attr]
            while len(batch) < self._max_batch:
                remaining = deadline - self._loop.time()  # type: ignore[union-attr]
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    break
                if item is _SENTINEL:
                    await self._flush(batch)
                    await self._queue.put(_SENTINEL)
                    batch = []
                    break
                batch.append(item)

            if batch:
                await self._flush(batch)

    async def _flush(self, batch: list[_Pending]) -> None:
        records = [p.record for p in batch]

        insert_records: list = records
        insert_batch: list[_Pending] = batch
        skipped_pairs: list[tuple[_Pending, Any]] = []
        if self._pre_flush_gate is not None:
            try:
                decisions = self._pre_flush_gate(records)
            except BaseException as exc:  # noqa: BLE001
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(exc)
                self._notify_flushed()
                return
            insert_records = []
            insert_batch = []
            for p, dec in zip(batch, decisions):
                action = dec[0]
                payload = dec[1]
                if action == "skip":
                    skipped_pairs.append((p, payload))
                else:
                    insert_records.append(p.record)
                    insert_batch.append(p)

        if insert_records:
            try:
                await self._table.add(insert_records)
            except BaseException as exc:  # noqa: BLE001
                for p in insert_batch:
                    if not p.future.done():
                        p.future.set_exception(exc)
                for p, merged_uuid in skipped_pairs:
                    if not p.future.done():
                        p.future.set_result(merged_uuid)
                self._notify_flushed()
                return

        if self._on_flushed is not None and insert_records:
            try:
                self._on_flushed(insert_records)
            except Exception:  # noqa: BLE001 -- hook must never break write path
                pass
        for p in insert_batch:
            if not p.future.done():
                p.future.set_result(None)
        for p, merged_uuid in skipped_pairs:
            if not p.future.done():
                p.future.set_result(merged_uuid)
        self._notify_flushed()

    def _notify_flushed(self) -> None:
        if self._flush_event is not None and not self._flush_event.is_set():
            self._flush_event.set()


class _Pending:

    __slots__ = ("record", "future")

    def __init__(self, record: Any, future: asyncio.Future) -> None:
        self.record = record
        self.future = future


class _Sentinel:

    __repr__ = lambda self: "<AsyncWriteQueue.sentinel>"  # noqa: E731


_SENTINEL = _Sentinel()
