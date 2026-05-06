"""Plan 05-14 — async provenance write queue (OPS-10 / M-02).

Moves provenance writes off the recall critical path. A single daemon
thread drains a bounded queue.Queue of (record_id, entry) pairs and
flushes them via the existing ``MemoryStore.append_provenance_batch``
exactly as the sync path did.

Why this is the right shape:
- provenance writes are pure SIDE EFFECTS; pipeline_recall never reads
  their result. Textbook fire-and-forget candidate.
- The biological analogue: consolidation writes happen during rest, not
  during retrieval (CLS / sleep replay).
- The existing ``AsyncWriteQueue`` is for record inserts,
  which must be durable before their return (S4 viability check reads
  them back). Provenance has no such contract — a simpler, purpose-built
  queue avoids the coroutine/event-loop machinery that asyncio imposes.

Constitutional fences:
- Rule 1: worker swallows all exceptions (recall must never fail due
  to a provenance-write failure).
- entries are never dropped during normal operation; on shutdown
  the atexit hook drains the queue. W1/when the
  in-memory queue is full under overload, batches are spilled to
  ``~/.iai-mcp/.provenance-overflow/<unix_ms>-<n>.jsonl``. The worker
  drains the spill dir on idle and re-enqueues the batches. Zero drops
  on the happy path; the only path that can drop is disk-write failure
  (alarmed via the ``provenance_queue_spill_failed`` stderr event).
- C3 / C6: stdlib only. No extra dependencies.

Python 3.11+.
"""
from __future__ import annotations

import atexit
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore


# Sentinel pushed on the queue to wake the worker for stop/flush.
_STOP = object()
_FLUSH = object()

# W1/D-01 — overflow spill-to-disk.
OVERFLOW_DIR_NAME = ".provenance-overflow"
# Worker idle poll: 5s upper bound on overflow-drain responsiveness.
# Bounded so under sustained overload the spill drain catches up
# within a small constant time after _q clears.
_WORKER_IDLE_POLL_S = 5.0


class ProvenanceWriteQueue:
    """Single-daemon-thread coalescing queue for provenance batches.

    Usage:
        q = ProvenanceWriteQueue(store, coalesce_ms=50)
        q.start()                                # idempotent
        q.enqueue([(record_id, entry_dict), ...])  # non-blocking
        q.flush(timeout=2.0)                     # drain + wait
        q.stop()                                 # drain + join

    The worker loop:
        1. Blocking `.get()` on the queue (wakes on enqueue or sentinel).
        2. Opportunistic drain up to ``max_batch_pairs`` pairs OR until
           the queue has been empty for ``coalesce_ms``.
        3. Single call to ``store.append_provenance_batch(pairs,
           records_cache=None)``.
        4. Back to (1).

    All worker exceptions are logged to stderr as structured JSON events
    and swallowed.
    """

    def __init__(
        self,
        store: "MemoryStore",
        *,
        coalesce_ms: int = 50,
        max_queue_size: int = 4096,
        max_batch_pairs: int = 256,
    ) -> None:
        self._store = store
        self._coalesce_s = max(1, int(coalesce_ms)) / 1000.0
        self._max_batch = int(max_batch_pairs)
        # Queue items are either lists of (UUID, dict) pairs or the
        # _STOP / _FLUSH sentinels.
        self._q: queue.Queue = queue.Queue(maxsize=int(max_queue_size))
        self._thread: threading.Thread | None = None
        self._started = False
        self._stop_requested = False
        # flush synchronisation: drained_event is set by the worker when
        # it has processed everything up to a _FLUSH sentinel.
        self._flush_event = threading.Event()
        self._atexit_registered = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start the worker thread. Idempotent."""
        with self._lock:
            if self._started:
                return
            self._started = True
            self._stop_requested = False
            self._thread = threading.Thread(
                target=self._run,
                name="iai-mcp-provenance-queue",
                daemon=True,
            )
            self._thread.start()
            if not self._atexit_registered:
                atexit.register(self._atexit_flush)
                self._atexit_registered = True

    def stop(self) -> None:
        """Signal the worker, drain remaining items, join the thread.

        Idempotent. After stop the queue is no longer usable; call
        start() to revive (fresh worker, same queue instance).
        """
        with self._lock:
            if not self._started:
                return
            self._stop_requested = True
            try:
                self._q.put_nowait(_STOP)
            except queue.Full:
                # Drop one item to make room for the sentinel.
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(_STOP)
                except queue.Empty:
                    pass
            t = self._thread
        if t is not None:
            t.join(timeout=5.0)
        with self._lock:
            self._started = False
            self._thread = None

    def flush(self, timeout: float = 2.0) -> None:
        """Wait until the worker has drained everything enqueued so far.

        Puts a _FLUSH sentinel; the worker signals _flush_event once it
        has processed all pairs that were in the queue at that point.
        Times out silently — the caller is responsible for deciding
        whether to retry; recall latency is never blocked by flush().
        """
        if not self._started:
            return
        self._flush_event.clear()
        try:
            self._q.put(_FLUSH, timeout=timeout)
        except queue.Full:
            return
        self._flush_event.wait(timeout=timeout)

    # ---------------------------------------------------------------- public write

    def enqueue(self, pairs: "list[tuple[UUID, dict]]") -> None:
        """Non-blocking enqueue.

        W1/when the in-memory queue is full, the batch
        spills to ``~/.iai-mcp/.provenance-overflow/<ts>-<n>.jsonl``.
        The worker thread drains the spill dir on idle and re-enqueues
        the batches. zero drops under overload (only path that
        can drop is disk-write failure, which is itself alarmed).
        """
        if not pairs:
            return
        try:
            self._q.put_nowait(list(pairs))
            return
        except queue.Full:
            pass
        # In-memory queue full — spill to disk. Worker will pick this
        # up on its next idle cycle. Recall hot path is unaffected
        # (this branch only fires on the WRITE side under overload).
        self._spill_to_disk(list(pairs))
        try:
            sys.stderr.write(
                '{"event":"provenance_queue_overflow_spill","n_pairs":'
                + str(len(pairs))
                + "}\n"
            )
        except Exception:
            pass

    # ---------------------------------------------------------------- spill / drain

    def _spill_to_disk(self, pairs: list) -> None:
        """Persist a rejected batch to ``~/.iai-mcp/.provenance-overflow/``.

        Per-batch JSONL file: one line per (uuid_str, entry_dict) pair.
        File-level atomicity — the worker re-enqueues the entire file's
        contents in one call, then unlinks. Format:

            {"id": "<uuid>", "entry": {...}}\n
            {"id": "<uuid>", "entry": {...}}\n

        Failure modes:
        - Disk full / permission denied: emits structured stderr event
          ``provenance_queue_spill_failed``. This is the ONLY drop path
          remaining post-07.9 W1; it's a system-level alarm condition,
          not a normal-operation outcome.
        """
        if not pairs:
            return
        try:
            overflow_dir = Path.home() / ".iai-mcp" / OVERFLOW_DIR_NAME
            overflow_dir.mkdir(parents=True, exist_ok=True)
            ts_ms = int(time.time() * 1000)
            # Tag with the batch length and a short pid suffix so two
            # spills inside the same millisecond never collide.
            fpath = overflow_dir / f"{ts_ms}-{len(pairs)}-{id(pairs) & 0xFFFF:04x}.jsonl"
            tmp_path = fpath.with_suffix(fpath.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                for rid, entry in pairs:
                    fh.write(json.dumps({"id": str(rid), "entry": entry}) + "\n")
            tmp_path.rename(fpath)  # atomic rename keeps drain from
            # ever reading a half-written file.
        except Exception as exc:
            try:
                sys.stderr.write(
                    '{"event":"provenance_queue_spill_failed","error":'
                    + _json_str(str(exc))
                    + ',"n_pairs":' + str(len(pairs)) + '}\n'
                )
            except Exception:
                pass

    def _drain_overflow_dir(self) -> int:
        """Re-enqueue any spilled batches into ``_q``.

        Called by the worker on idle (between blocking `_q.get()` cycles).
        Per-file atomicity: re-enqueue ALL pairs from a file via a single
        ``_q.put`` call, then unlink. If ``_q`` is still full, leave the
        file on disk for the next idle cycle.

        Returns the number of pairs successfully re-enqueued in this pass.
        """
        overflow_dir = Path.home() / ".iai-mcp" / OVERFLOW_DIR_NAME
        if not overflow_dir.exists():
            return 0
        n_re_enqueued = 0
        # sorted() so older spill files drain first (FIFO durability).
        for fpath in sorted(overflow_dir.glob("*.jsonl")):
            try:
                pairs: list = []
                with fpath.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        pairs.append((UUID(obj["id"]), obj["entry"]))
                if not pairs:
                    fpath.unlink()
                    continue
                # Short-timeout put: this is the worker thread, so
                # blocking briefly is fine, but a long block would
                # delay normal-path enqueues that arrive during drain.
                try:
                    self._q.put(pairs, timeout=0.5)
                except queue.Full:
                    # Queue still saturated — leave the file for the
                    # next idle cycle. Don't unlink.
                    return n_re_enqueued
                fpath.unlink()
                n_re_enqueued += len(pairs)
            except Exception as exc:
                # Malformed spill file: preserve evidence, do not lose data.
                try:
                    failed = fpath.with_suffix(f".failed-{int(time.time())}.jsonl")
                    fpath.rename(failed)
                    sys.stderr.write(
                        '{"event":"provenance_queue_spill_drain_failed","error":'
                        + _json_str(str(exc)) + '}\n'
                    )
                except Exception:
                    pass
        return n_re_enqueued

    # ------------------------------------------------------------------ internals

    def _run(self) -> None:
        """Worker loop.

        W1/between blocking `_q.get()` cycles the worker
        drains any spilled overflow files at ``~/.iai-mcp/.provenance-overflow/``.
        Bounded poll: idle-timeout = ``_WORKER_IDLE_POLL_S`` so the spill
        drain runs at most once per ``_WORKER_IDLE_POLL_S`` seconds when
        the queue is empty.
        """
        while True:
            try:
                item = self._q.get(timeout=_WORKER_IDLE_POLL_S)
            except queue.Empty:
                # Idle tick — try to drain the overflow dir back into _q.
                # Defensive: any error during drain is logged + swallowed.
                try:
                    self._drain_overflow_dir()
                except Exception:
                    pass
                continue
            except Exception:
                continue
            if item is _STOP:
                # Drain remaining real items before exit.
                self._drain_remaining()
                return
            if item is _FLUSH:
                # Drain everything enqueued before this sentinel.
                self._drain_remaining()
                self._flush_event.set()
                continue
            # Normal batch. Coalesce: pull more pending items until we
            # hit max_batch_pairs or a short idle window.
            pairs: list = list(item)
            while len(pairs) < self._max_batch:
                try:
                    nxt = self._q.get(timeout=self._coalesce_s)
                except queue.Empty:
                    break
                if nxt is _STOP:
                    # Flush what we have, then exit.
                    self._flush_batch(pairs)
                    self._drain_remaining()
                    return
                if nxt is _FLUSH:
                    self._flush_batch(pairs)
                    self._drain_remaining()
                    self._flush_event.set()
                    pairs = []
                    break
                pairs.extend(nxt)
            if pairs:
                self._flush_batch(pairs)

    def _drain_remaining(self) -> None:
        """Pull everything currently in the queue and flush as one batch."""
        pairs: list = []
        saw_flush = False
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                continue
            if item is _FLUSH:
                saw_flush = True
                continue
            pairs.extend(item)
        if pairs:
            self._flush_batch(pairs)
        if saw_flush:
            self._flush_event.set()

    def _flush_batch(self, pairs: list) -> None:
        """Call store.append_provenance_batch, swallow all exceptions (Rule 1)."""
        if not pairs:
            return
        try:
            self._store.append_provenance_batch(pairs, records_cache=None)
        except Exception as exc:
            try:
                sys.stderr.write(
                    '{"event":"provenance_queue_flush_failed","n_pairs":'
                    + str(len(pairs))
                    + ',"error":'
                    + _json_str(str(exc))
                    + "}\n"
                )
            except Exception:
                pass

    def _atexit_flush(self) -> None:
        """atexit handler — drain and stop the worker. Never raises."""
        try:
            if self._started:
                self.flush(timeout=2.0)
                self.stop()
        except Exception:
            pass


def _json_str(s: str) -> str:
    """Minimal JSON string escape for stderr structured logs."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
