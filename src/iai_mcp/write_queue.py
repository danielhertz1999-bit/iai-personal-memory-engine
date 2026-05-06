"""Plan 05-10 — asyncio-backed coalescing write queue for LanceDB.

Motivation (from 05-08 diagnosis + 05-10 plan): each synchronous
``tbl.add([row])`` call against a LanceDB table allocates roughly
~0.3 MB of pyarrow working-set overhead that is sub-linear per call
but linear in call count. Seeding the store record-by-record (one
call per record) drives peak RSS to ~1.3 GB at N=5k. This module
coalesces inserts inside a 100 ms window (or ``max_batch`` records,
whichever fires first) and forwards them as a single ``await
tbl.add(batch)`` call. At N=10k with max_batch=128 the buffer
overhead drops from ~3 GB (10000 * 0.3 MB) to ~24 MB (79 * 0.3 MB).

Contract (see ``tests/test_write_queue.py`` for the machine-checked
version):

- ``enqueue(record)`` returns an ``asyncio.Future`` that resolves
  only after the record's batch has landed on disk. Callers that
  want sync-equivalent durability **must** await the future.
- A single ``tbl.add(batch)`` call carries all records coalesced
  inside one window, up to ``max_batch``.
- ``stop()`` drains pending records and flushes them synchronously
  before returning. Enqueues after ``stop()`` raise ``RuntimeError``.
- Back-pressure: when the buffer is already at ``max_queue_size``
  the next ``enqueue()`` awaits the next flush before accepting —
  never unbounded memory growth.
- Flush failures propagate: if ``tbl.add(batch)`` raises, every
  pending Future in that batch resolves with that exception. The
  queue itself stays running so subsequent enqueues still work.
- ``on_flushed(batch)`` (optional) fires once per successful flush,
  synchronously inside the loop, **before** futures are resolved.
  The callback receives the exact list of records in the order
  they were flushed — use this to mirror writes to a secondary
  index (Plan 05-12 runtime-graph hook).

Constitutional invariants:
- C3 (no paid-API): pure stdlib + a LanceDB async table handle.
- C6 (LanceDB authoritative): nothing in this module short-circuits
  the write; ``tbl.add(batch)`` is the only persistence path.
- (no drift): a resolved Future means the batch reached
  disk. An exception means no Future in that batch reached disk;
  the caller is expected to retry or surface the error.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

__all__ = ["AsyncWriteQueue"]


class AsyncWriteQueue:
    """Coalescing write queue on top of a LanceDB AsyncTable.

    The table object only needs to expose ``await add(batch)`` — the
    tests ship a minimal ``MockAsyncTable`` that satisfies this shape.

    Parameters
    ----------
    table
        LanceDB ``AsyncTable`` (or any object with ``async def
        add(self, batch: list[dict]) -> None``).
    coalesce_ms
        Flush window in milliseconds. On every iteration of the
        coalesce loop we wait at most this long for the next record
        before flushing whatever we have.
    max_batch
        Hard cap on records per ``tbl.add`` call. Reached before the
        timeout, triggers an immediate flush.
    max_queue_size
        Hard cap on buffered (queued + pending) records. The
        ``enqueue()`` call awaits the next flush once the cap is hit.
    on_flushed
        Optional callback ``callable(batch: list) -> None`` fired
        after each successful flush, inside the queue's event loop,
        before pending futures are resolved. Exceptions raised by the
        callback are swallowed (logged as a no-op) so a bad hook can
        never break the write path.
    """

    def __init__(
        self,
        table: Any,
        *,
        coalesce_ms: int = 100,
        max_batch: int = 128,
        max_queue_size: int = 4096,
        on_flushed: Optional[Callable[[list], None]] = None,
    ) -> None:
        self._table = table
        self._coalesce_s: float = max(coalesce_ms, 1) / 1000.0
        self._max_batch: int = max(max_batch, 1)
        self._max_queue_size: int = max(max_queue_size, 1)
        self._on_flushed = on_flushed

        # Runtime state (set in start()).
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        # Event set after every flush so back-pressured enqueues can wake.
        self._flush_event: Optional[asyncio.Event] = None
        self._coalesce_task: Optional[asyncio.Task] = None
        self._stopping: bool = False
        self._stopped: bool = False

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Attach to the current loop and spin up the coalesce task."""
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
        """Drain pending records, flush them, then shut the loop down.

        Idempotent: calling stop() on an already-stopped queue is a
        no-op.
        """
        if self._stopped:
            return
        self._stopping = True
        assert self._queue is not None
        # Sentinel wakes the coalesce loop out of its wait_for on an
        # otherwise-empty queue.
        await self._queue.put(_SENTINEL)
        if self._coalesce_task is not None:
            await self._coalesce_task
            self._coalesce_task = None
        self._stopped = True

    # ------------------------------------------------------------------ enqueue

    async def enqueue(self, record: Any) -> asyncio.Future:
        """Append ``record`` to the coalesce buffer.

        Returns a Future that resolves to ``None`` after the record's
        batch has been flushed (``tbl.add`` returned), or resolves
        with the exception raised by ``tbl.add`` for that batch.

        Blocks (awaits) when the queue is already at ``max_queue_size``
        until a flush frees a slot.
        """
        if self._stopped or self._stopping:
            raise RuntimeError("AsyncWriteQueue is stopped; cannot enqueue")
        assert self._queue is not None and self._flush_event is not None

        # Back-pressure: wait for a flush if we're already at the cap.
        # Use a loop because multiple concurrent enqueues may race on
        # the same wake-up.
        while self._queue.qsize() >= self._max_queue_size:
            self._flush_event.clear()
            await self._flush_event.wait()

        fut: asyncio.Future = self._loop.create_future()  # type: ignore[union-attr]
        await self._queue.put(_Pending(record=record, future=fut))
        return fut

    # ------------------------------------------------------------------ internals

    async def _coalesce_loop(self) -> None:
        """Main loop: collect up to ``max_batch`` records per window,
        then flush. Exits after the sentinel drain when ``stop()``
        is called.
        """
        assert self._queue is not None and self._flush_event is not None
        while True:
            batch: list[_Pending] = []
            # First item: block indefinitely until we get something or
            # the sentinel arrives.
            first = await self._queue.get()
            if first is _SENTINEL:
                # Drain any stragglers that snuck in before the sentinel.
                while not self._queue.empty():
                    item = self._queue.get_nowait()
                    if item is _SENTINEL:
                        continue
                    batch.append(item)
                if batch:
                    await self._flush(batch)
                return
            batch.append(first)

            # Fill the batch within the coalesce window.
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
                    # Flush what we have, then re-enter the outer loop
                    # to let the sentinel branch above handle shutdown.
                    await self._flush(batch)
                    # Re-queue the sentinel so the outer loop sees it.
                    await self._queue.put(_SENTINEL)
                    batch = []
                    break
                batch.append(item)

            if batch:
                await self._flush(batch)

    async def _flush(self, batch: list[_Pending]) -> None:
        """Push a batch through ``tbl.add`` and resolve each Future."""
        records = [p.record for p in batch]
        try:
            await self._table.add(records)
        except BaseException as exc:  # noqa: BLE001
            for p in batch:
                if not p.future.done():
                    p.future.set_exception(exc)
            self._notify_flushed()
            return

        # Hook first (synchronous, in-loop) — so graph-sync observes
        # the write before any caller that awaits the future can race
        # against the in-RAM graph.
        if self._on_flushed is not None:
            try:
                self._on_flushed(records)
            except Exception:
                # Invariant: a bad hook can never break the write
                # path. Swallow; structured logging lives in the
                # hook owner (store._fire_graph_sync_hook already
                # handles this for the graph-sync case).
                pass
        for p in batch:
            if not p.future.done():
                p.future.set_result(None)
        self._notify_flushed()

    def _notify_flushed(self) -> None:
        """Wake any enqueue() calls that are back-pressured."""
        if self._flush_event is not None and not self._flush_event.is_set():
            self._flush_event.set()


# ---------------------------------------------------------------------- internals


class _Pending:
    """Record + the Future its caller is awaiting. Tiny wrapper so we
    can drop it onto asyncio.Queue without worrying about dataclass
    equality semantics (Futures don't hash)."""

    __slots__ = ("record", "future")

    def __init__(self, record: Any, future: asyncio.Future) -> None:
        self.record = record
        self.future = future


class _Sentinel:
    """Marker object for graceful shutdown."""

    __repr__ = lambda self: "<AsyncWriteQueue.sentinel>"  # noqa: E731


_SENTINEL = _Sentinel()
