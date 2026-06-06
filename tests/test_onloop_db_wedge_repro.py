"""Mechanism lock: a synchronous store read on the event-loop thread, blocked on
a contended shared connection lock, wedges the loop-served liveness probe — and
the SAME read dispatched off-loop keeps the probe served.

This is a self-contained reproduction of the confirmed daemon-wedge mechanism. It
does not depend on any production-code change: both arms pass under the current
code.

  ARM-1 (WEDGE): a worker thread holds the real ``MemoryStore.db._conn_lock`` for
    a controlled window. A coroutine awaited ON the event loop calls
    ``store.get(rid)`` inline; ``store.get`` -> ``HippoQuery.to_pandas`` acquires
    the SAME ``_conn_lock`` and blocks on the Python re-entrant lock for the FULL
    hold window (it never reaches SQLite, so the SQLite busy_timeout is bypassed
    and there is no ~2s cap). Because that block runs on the loop thread, the
    loop cannot serve the cheap pure-dict-read ``status`` probe, so the round-trip
    probe (sampled from its own thread + loop, exactly like the liveness watchdog)
    exceeds its timeout -> WEDGE.

  ARM-2 (SERVED, the discriminator): the SAME ``store.get(rid)`` dispatched via
    ``await asyncio.to_thread(store.get, rid)`` while the worker still holds
    ``_conn_lock``. The worker-side read still stalls for ~the hold window (it
    contends on ``_conn_lock`` off the loop), but the loop stays free, so the
    probe round-trip stays SERVED (sub-second). This arm is served NOW: the read
    is moved off the loop inside the test itself, so it needs no production-code
    change.

Why deterministic (no race, hence no xfail): the worker sets its ``started``
event ONLY after it has acquired ``_conn_lock``, the arm waits on that event
before issuing the get, and the hold is a fixed ``time.sleep``. The contention is
guaranteed, the timing is fixed.

Hermeticity: the conftest autouse fixtures redirect HOME / socket / store defaults
under a per-test tmp dir and supply a file-passphrase crypto path; the store is
opened with an explicit ``path=`` under ``tmp_path``. An in-test assertion fails
if the resolved store root is ever under the operator's real ``~/.iai-mcp``. The
real daemon is never addressed; only a private socket + a hermetic store are
opened. Probe threads, the throwaway loops, the socket server, and the store are
all torn down in ``finally``.
"""
from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from iai_mcp.community import CommunityAssignment
from iai_mcp.daemon import WATCHDOG_PROBE_TIMEOUT_SEC, _probe_status_roundtrip
from iai_mcp.socket_server import SocketServer
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


# Mechanism, not magnitude: a small representative store is enough — the
# diagnosis measured the magnitude separately. Each record carries a populated
# literal + a few provenance entries + a profile-modulation gain so a real
# ``store.get`` performs the same three AES-GCM field decrypts as production.
_N_SEED = 80
# A shorter probe read-timeout shrinks the controlled hold window while keeping
# the WEDGE / SERVED discrimination strict (block must exceed it; served must be
# well under it). The probe also keeps a fixed internal connect timeout.
_PROBE_READ_TIMEOUT = 1.0
_HOLD_SEC = 3.0  # > _PROBE_READ_TIMEOUT, so a held block reliably wedges the probe
_SERVED_RTT_CEIL = 1.0  # an unblocked loop serves the dict-read probe well under this


def _make_representative_record(vec, community_id, centrality: float) -> MemoryRecord:
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    literal = ("verbatim recall content " * 40)[:960]
    provenance = [
        {"ts": now.isoformat(), "cue": "recall cue text", "session_id": f"sess-{i}"}
        for i in range(3)
    ]
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=literal,
        aaak_index="",
        embedding=vec.tolist(),
        community_id=community_id,
        centrality=centrality,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=provenance,
        created_at=now,
        updated_at=now,
        tags=["topic:alpha", "topic:beta", "kind:note"],
        language="en",
        profile_modulation_gain={"empathy_gain": 0.5, "detail_gain": 0.7},
    )


def _seed_store(store: MemoryStore, n: int):
    """Insert n representative records; return (live_ids, assignment)."""
    dim = store._embed_dim
    cid = uuid4()
    ids = []
    rng = np.random.default_rng(7)
    for i in range(n):
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        rec = _make_representative_record(vec, cid, float(i % 31))
        store.insert(rec)
        ids.append(rec.id)
    assignment = CommunityAssignment(
        top_communities=[cid],
        community_centroids={cid: [0.0] * dim},
        mid_regions={cid: ids},
    )
    return ids, assignment


class _ThreadedProbe:
    """Sample the REAL probe coroutine from its own thread + throwaway loop.

    Faithful to the liveness watchdog: it never touches the daemon's loop. RTT is
    recorded as +inf on any failure or timeout (the wedge signal).
    """

    def __init__(self, sock_path: str, read_timeout: float):
        self._sock_path = sock_path
        self._read_timeout = read_timeout
        self._stop = threading.Event()
        self._worst = 0.0
        self._samples: list[float] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                ok = asyncio.run(
                    _probe_status_roundtrip(self._sock_path, self._read_timeout)
                )
            except Exception:  # noqa: BLE001 -- probe failure == unservable
                ok = False
            rtt = (time.monotonic() - t0) if ok else float("inf")
            self._samples.append(rtt)
            if rtt == float("inf"):
                self._worst = float("inf")
            elif self._worst != float("inf"):
                self._worst = max(self._worst, rtt)
            time.sleep(0.02)

    def start(self) -> None:
        self._thread.start()

    def reset(self) -> None:
        self._worst = 0.0
        self._samples = []

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=8.0)

    def report(self):
        return self._worst, list(self._samples)


def _hold_conn_lock(store: MemoryStore, hold_sec: float,
                    started: threading.Event, done: threading.Event) -> None:
    """Hold the real db._conn_lock for exactly hold_sec.

    ``started`` is set AFTER the lock is acquired, so any reader that waits on it
    is guaranteed to contend on the lock. A store read blocks on the Python
    re-entrant lock acquire before it ever reaches SQLite, so the hold body is
    irrelevant — time.sleep isolates the lock coupling and bypasses busy_timeout.
    """
    with store.db._conn_lock:
        started.set()
        time.sleep(hold_sec)
    done.set()


def _measured_get(store: MemoryStore, rid) -> float:
    t0 = time.monotonic()
    try:
        store.get(rid)
    except Exception:  # noqa: BLE001 -- timing the block, not asserting the value
        pass
    return time.monotonic() - t0


def _short_socket_path() -> Path:
    """A short unix-socket path under the system temp dir.

    The store stays hermetic under ``tmp_path``, but pytest's ``tmp_path`` can be
    long enough that ``<tmp_path>/.iai-mcp/.daemon.sock`` exceeds the platform
    ``sun_path`` limit (~104 bytes on macOS), which makes the probe's connect
    fail silently. The socket is a transient endpoint, not stored data, so a
    short unique path is correct and stays off the operator's real paths.
    """
    d = Path(tempfile.mkdtemp(prefix="iai-wedge-"))
    return d / "d.sock"


def _build_state() -> dict:
    return {
        "fsm_state": "WAKE",
        "daemon_started_at": None,
        "last_tick_at": None,
        "quiet_window": None,
        "pending_digest": None,
        "scheduler_paused": False,
    }


def _assert_hermetic(store: MemoryStore, tmp_path: Path) -> None:
    """Fail loudly if the store ever resolves under the operator's real home."""
    root = Path(store.root).resolve()
    assert str(root).startswith(str(tmp_path.resolve())), (
        f"store root {root} escaped tmp_path {tmp_path}"
    )
    real_home_store = (Path.home() / ".iai-mcp").resolve()
    # Path.home() is redirected to tmp by the conftest autouse fixture, so the
    # real operator store is never resolvable here; assert it regardless.
    assert real_home_store not in root.parents and root != real_home_store, (
        f"store root {root} resolved under the real ~/.iai-mcp"
    )


async def _serve(store: MemoryStore, sock_path: Path):
    server = SocketServer(store, state=_build_state())
    serve_task = asyncio.create_task(server.serve(socket_path=sock_path))
    for _ in range(100):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)
    return server, serve_task


async def _teardown_server(server: SocketServer, serve_task) -> None:
    server.shutdown_event.set()
    try:
        await asyncio.wait_for(serve_task, timeout=5.0)
    except Exception:  # noqa: BLE001
        serve_task.cancel()
        try:
            await serve_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


def test_fixture_smoke(tmp_path):
    """Baseline: an uncontended store.get is sub-10ms; the probe is served fast
    when nothing blocks the loop."""
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)
    ids, _assignment = _seed_store(store, _N_SEED)
    rid = ids[len(ids) // 2]

    sock_path = _short_socket_path()

    async def _body():
        server, serve_task = await _serve(store, sock_path)
        probe = _ThreadedProbe(str(sock_path), _PROBE_READ_TIMEOUT)
        try:
            probe.start()
            await asyncio.sleep(0.4)

            base_get = _measured_get(store, rid)
            assert base_get < 0.1, f"uncontended get too slow: {base_get:.4f}s"

            worst, samples = probe.report()
            assert samples, "probe produced no samples"
            assert worst < _SERVED_RTT_CEIL, (
                f"idle-loop probe should be served fast, worst={worst}"
            )
        finally:
            probe.stop()
            await _teardown_server(server, serve_task)

    try:
        asyncio.run(_body())
    finally:
        store.close()


def _served_fraction(samples: list[float], ceil: float) -> float:
    if not samples:
        return 0.0
    served = sum(1 for s in samples if s != float("inf") and s <= ceil)
    return served / len(samples)


def test_on_loop_store_read_under_held_lock_wedges_probe(tmp_path):
    """ARM-1: an inline (on-loop) store.get under a held _conn_lock wedges the
    REAL probe (RTT exceeds the watchdog timeout while the loop is blocked)."""
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)
    ids, _assignment = _seed_store(store, _N_SEED)
    rid = ids[len(ids) // 2]

    sock_path = _short_socket_path()

    async def _body():
        server, serve_task = await _serve(store, sock_path)
        probe = _ThreadedProbe(str(sock_path), _PROBE_READ_TIMEOUT)
        try:
            probe.start()
            await asyncio.sleep(0.4)

            probe.reset()
            started = threading.Event()
            done = threading.Event()
            worker = threading.Thread(
                target=_hold_conn_lock,
                args=(store, _HOLD_SEC, started, done),
                daemon=True,
            )
            worker.start()
            # Wait for the lock-acquired signal OFF the loop so the loop is free
            # right up until the inline (on-loop) get is what blocks it.
            ok = await asyncio.to_thread(started.wait, 10.0)
            assert ok, "worker never acquired _conn_lock"
            # INLINE on the event loop: blocks the loop for the full hold window.
            block_sec = _measured_get(store, rid)
            await asyncio.to_thread(done.wait, _HOLD_SEC + 10.0)
            await asyncio.to_thread(worker.join, 5.0)
            await asyncio.sleep(0.2)

            arm1_worst, arm1_samples = probe.report()
            assert block_sec >= _PROBE_READ_TIMEOUT, (
                "on-loop get did not block long enough to exercise the wedge "
                f"(block={block_sec:.3f}s, hold={_HOLD_SEC}s)"
            )
            assert arm1_samples, "ARM-1 probe produced no samples"
            assert arm1_worst == float("inf") or arm1_worst > WATCHDOG_PROBE_TIMEOUT_SEC, (
                "ARM-1 probe should have wedged (RTT > timeout) while the loop "
                f"was blocked on the on-loop get; worst={arm1_worst}"
            )
        finally:
            probe.stop()
            await _teardown_server(server, serve_task)

    try:
        asyncio.run(_body())
    finally:
        store.close()


def test_off_loop_store_read_under_held_lock_keeps_probe_served(tmp_path):
    """ARM-2 (the discriminator, served NOW): the SAME store.get dispatched via
    asyncio.to_thread keeps the loop-served probe served even while a worker
    holds _conn_lock for the full window. The worker-side read still stalls
    behind the lock off-loop (proving it exercises the same primitive)."""
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)
    ids, _assignment = _seed_store(store, _N_SEED)
    rid = ids[len(ids) // 2]

    sock_path = _short_socket_path()

    async def _body():
        server, serve_task = await _serve(store, sock_path)
        probe = _ThreadedProbe(str(sock_path), _PROBE_READ_TIMEOUT)
        # Loop-liveness ticker: advances only when the loop gets turns.
        ticks = {"n": 0}
        stop_ticker = asyncio.Event()

        async def _ticker():
            while not stop_ticker.is_set():
                await asyncio.sleep(0.05)
                ticks["n"] += 1

        ticker_task = asyncio.create_task(_ticker())
        try:
            probe.start()
            await asyncio.sleep(0.4)

            probe.reset()
            started = threading.Event()
            done = threading.Event()
            worker = threading.Thread(
                target=_hold_conn_lock,
                args=(store, _HOLD_SEC, started, done),
                daemon=True,
            )
            worker.start()
            ok = await asyncio.to_thread(started.wait, 10.0)
            assert ok, "worker never acquired _conn_lock"
            ticks_before = ticks["n"]
            # OFF the event loop: the read contends on _conn_lock in a pool
            # thread; the loop stays free to serve the probe.
            worker_get_sec = await asyncio.to_thread(_measured_get, store, rid)
            ticks_after = ticks["n"]
            await asyncio.to_thread(done.wait, _HOLD_SEC + 10.0)
            await asyncio.to_thread(worker.join, 5.0)
            await asyncio.sleep(0.2)

            arm2_worst, arm2_samples = probe.report()
            # The off-loop read really contended on the lock.
            assert worker_get_sec >= _PROBE_READ_TIMEOUT, (
                "to_thread get should have stalled behind the held _conn_lock "
                f"(get={worker_get_sec:.3f}s, hold={_HOLD_SEC}s)"
            )
            # The loop kept ticking through the hold window -> it was never blocked.
            assert ticks_after - ticks_before >= 5, (
                "the event loop stalled during the off-loop read "
                f"(ticks advanced {ticks_after - ticks_before} over ~{_HOLD_SEC}s)"
            )
            assert arm2_samples, "ARM-2 probe produced no samples"
            # The probe stayed served for the overwhelming majority of samples;
            # a stray boundary outlier (a sample straddling teardown) must not
            # flip the verdict, so assert the served fraction, not the max.
            served = _served_fraction(arm2_samples, _SERVED_RTT_CEIL)
            assert served >= 0.8, (
                "ARM-2 probe should have stayed served (loop free) even while a "
                f"worker held _conn_lock; served_fraction={served:.2f} "
                f"samples={arm2_samples}"
            )
        finally:
            stop_ticker.set()
            ticker_task.cancel()
            try:
                await ticker_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            probe.stop()
            await _teardown_server(server, serve_task)

    try:
        asyncio.run(_body())
    finally:
        store.close()
