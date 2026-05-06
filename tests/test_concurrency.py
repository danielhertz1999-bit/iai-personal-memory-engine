"""Tests for iai_mcp.concurrency -- Task 1.

Covers 10 behaviours from the plan:
1. acquire_shared + try_acquire_exclusive blocking semantics.
2. Exclusive-then-exclusive: second blocks.
3. flock fd-close safety (Pitfall 2): closing /etc/passwd doesn't release lock.
4. Multi-MCP: 2 and 3 shared holders keep daemon blocked.
5. SIGKILL releases lock automatically (kernel).
6. Unix socket NDJSON status round-trip.
7. Unix socket dispatcher receives exact dict for pause/force_rem/tail_logs.
8. Stale socket cleanup (Pitfall 10) lets server bind without EADDRINUSE.
9. Lock file + socket file mode 0o600.
10. holds_exclusive_nb -- cooperative-yield probe; returns False when
    contended and never propagates BlockingIOError / EWOULDBLOCK.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import multiprocessing
import os
import signal
import time
from pathlib import Path

import pytest


# Use spawn so fork+LanceDB+multithread hazards (Pitfall 6) never apply.
_SPAWN = multiprocessing.get_context("spawn")


# ---------------------------------------------------------------------------
# helpers that run inside spawn children
# ---------------------------------------------------------------------------

def _child_hold_shared(lock_path_str: str, acquired_flag: str, release_flag: str) -> int:
    """Open the lock file, take LOCK_SH, touch acquired_flag, wait for release_flag, exit."""
    fd = os.open(lock_path_str, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        Path(acquired_flag).write_text("ok")
        # Wait for parent to signal release.
        release = Path(release_flag)
        for _ in range(300):  # up to 30s
            if release.exists():
                break
            time.sleep(0.1)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)
    return 0


def _child_hold_shared_sigkillable(lock_path_str: str, acquired_flag: str) -> int:
    """Take LOCK_SH, touch flag, sleep forever (until SIGKILL from parent)."""
    fd = os.open(lock_path_str, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_SH)
    Path(acquired_flag).write_text("ok")
    while True:
        time.sleep(1)


# ---------------------------------------------------------------------------
# fixture: isolate LOCK_PATH / SOCKET_PATH into tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture
def lock_and_socket_paths(tmp_path, monkeypatch):
    """Redirect module-level LOCK_PATH + SOCKET_PATH to tmp_path.

    AF_UNIX on macOS caps the path at 104 chars; pytest's tmp_path is often
    too long. We place the lock in tmp_path and the socket under a short
    /tmp/iai-<pid>-<n>/ directory so `bind()` succeeds.
    """
    from iai_mcp import concurrency
    lock_path = tmp_path / ".lock"
    # Short socket dir to stay inside the AF_UNIX 104-byte limit on macOS.
    sock_dir = Path(f"/tmp/iai-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    monkeypatch.setattr(concurrency, "LOCK_PATH", lock_path)
    monkeypatch.setattr(concurrency, "SOCKET_PATH", sock_path)
    try:
        yield lock_path, sock_path
    finally:
        # Best-effort cleanup so /tmp doesn't accumulate.
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Test 1: shared vs exclusive
# ---------------------------------------------------------------------------

def test_shared_blocks_exclusive(tmp_path, lock_and_socket_paths):
    """ProcessLock.acquire_shared() holder blocks try_acquire_exclusive()."""
    from iai_mcp.concurrency import ProcessLock

    lock_path, _ = lock_and_socket_paths
    reader = ProcessLock(lock_path)
    reader.acquire_shared()
    try:
        writer = ProcessLock(lock_path)
        try:
            # Separate fd on same file: exclusive must NOT be acquirable.
            assert writer.try_acquire_exclusive() is False
        finally:
            writer.close()
    finally:
        reader.release()
        reader.close()


# ---------------------------------------------------------------------------
# Test 2: exclusive-then-exclusive
# ---------------------------------------------------------------------------

def test_exclusive_then_exclusive_nonblocking(tmp_path, lock_and_socket_paths):
    """First exclusive holder succeeds; second gets False (non-blocking)."""
    from iai_mcp.concurrency import ProcessLock

    lock_path, _ = lock_and_socket_paths
    first = ProcessLock(lock_path)
    try:
        assert first.try_acquire_exclusive() is True
        second = ProcessLock(lock_path)
        try:
            assert second.try_acquire_exclusive() is False
        finally:
            second.close()
    finally:
        first.release()
        first.close()


# ---------------------------------------------------------------------------
# Test 3: flock fd-close safety (Pitfall 2 guard)
# ---------------------------------------------------------------------------

def test_flock_fd_close_safe(tmp_path, lock_and_socket_paths):
    """Closing an unrelated fd must NOT release our flock lock.

    flock is owned by process + open-file-description; closing /etc/passwd's fd
    doesn't touch our lock. This is the reason we use flock not lockf (Pitfall 2).
    """
    from iai_mcp.concurrency import ProcessLock

    lock_path, _ = lock_and_socket_paths
    holder = ProcessLock(lock_path)
    try:
        assert holder.try_acquire_exclusive() is True

        # Open + close an unrelated file to provoke the lockf close-fd trap.
        unrelated = os.open("/etc/passwd", os.O_RDONLY)
        os.close(unrelated)

        # Confirm another process cannot grab exclusive -- our lock still held.
        other = ProcessLock(lock_path)
        try:
            assert other.try_acquire_exclusive() is False
        finally:
            other.close()
    finally:
        holder.release()
        holder.close()


# ---------------------------------------------------------------------------
# Test 4: multi-MCP shared holders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_holders", [2, 3])
def test_multi_mcp(tmp_path, lock_and_socket_paths, n_holders):
    """N parallel shared holders block exclusive until ALL release."""
    from iai_mcp.concurrency import ProcessLock

    lock_path, _ = lock_and_socket_paths
    lock_path_str = str(lock_path)

    # Spawn N children, each holding LOCK_SH.
    acquired_flags = [tmp_path / f".acquired_{i}" for i in range(n_holders)]
    release_flag = tmp_path / ".release"

    procs = []
    for i in range(n_holders):
        p = _SPAWN.Process(
            target=_child_hold_shared,
            args=(lock_path_str, str(acquired_flags[i]), str(release_flag)),
        )
        p.start()
        procs.append(p)

    try:
        # Wait for all children to acquire shared.
        deadline = time.time() + 15
        while time.time() < deadline:
            if all(f.exists() for f in acquired_flags):
                break
            time.sleep(0.05)
        assert all(f.exists() for f in acquired_flags), "children failed to take LOCK_SH"

        # Daemon cannot take exclusive.
        daemon = ProcessLock(lock_path)
        try:
            assert daemon.try_acquire_exclusive() is False
        finally:
            daemon.close()

        # Release ALL children, then daemon can acquire.
        release_flag.write_text("go")
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

    # After all children exit, exclusive must succeed.
    daemon2 = ProcessLock(lock_path)
    try:
        assert daemon2.try_acquire_exclusive() is True
    finally:
        daemon2.release()
        daemon2.close()


# ---------------------------------------------------------------------------
# Test 5: SIGKILL releases lock (kernel-enforced)
# ---------------------------------------------------------------------------

def test_sigkill_releases_lock(tmp_path, lock_and_socket_paths):
    """Kernel auto-releases flock on process death (threat model: user kill -9)."""
    from iai_mcp.concurrency import ProcessLock

    lock_path, _ = lock_and_socket_paths
    lock_path_str = str(lock_path)

    acquired_flag = tmp_path / ".acquired_sigkill"
    child = _SPAWN.Process(
        target=_child_hold_shared_sigkillable,
        args=(lock_path_str, str(acquired_flag)),
    )
    child.start()
    try:
        deadline = time.time() + 15
        while time.time() < deadline and not acquired_flag.exists():
            time.sleep(0.05)
        assert acquired_flag.exists(), "child didn't acquire shared"

        # Parent observes shared holder -> cannot take exclusive.
        attempt = ProcessLock(lock_path)
        try:
            assert attempt.try_acquire_exclusive() is False
        finally:
            attempt.close()

        # Kill child -9.
        os.kill(child.pid, signal.SIGKILL)
        child.join(timeout=10)
        assert not child.is_alive()
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=2)

    # Kernel released child's lock -> exclusive now succeeds.
    daemon = ProcessLock(lock_path)
    try:
        # Give the kernel a brief moment to propagate the release.
        deadline = time.time() + 3
        acquired = False
        while time.time() < deadline:
            if daemon.try_acquire_exclusive():
                acquired = True
                break
            time.sleep(0.05)
        assert acquired, "exclusive still blocked after SIGKILL"
    finally:
        daemon.release()
        daemon.close()


# ---------------------------------------------------------------------------
# Test 6: socket NDJSON status round-trip
# ---------------------------------------------------------------------------

def test_socket_status_round_trip(tmp_path, lock_and_socket_paths):
    """serve_control_socket answers status with ok=true + state + uptime_sec."""
    from iai_mcp.concurrency import ProcessLock, serve_control_socket

    _, sock_path = lock_and_socket_paths
    lock = ProcessLock(lock_and_socket_paths[0])
    state = {"fsm_state": "WAKE", "daemon_started_at": "2026-04-18T00:00:00+00:00"}

    async def runner():
        shutdown = asyncio.Event()
        server_task = asyncio.create_task(
            serve_control_socket(store=None, lock=lock, state=state, shutdown=shutdown,
                                 socket_path=sock_path)
        )
        # Wait for socket to appear.
        for _ in range(100):
            if sock_path.exists():
                break
            await asyncio.sleep(0.02)
        assert sock_path.exists(), "socket never bound"

        reader, writer = await asyncio.open_unix_connection(path=str(sock_path))
        writer.write(b'{"type":"status"}\n')
        await writer.drain()
        line = await reader.readline()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        return json.loads(line)

    try:
        resp = asyncio.run(runner())
    finally:
        lock.close()

    assert resp["ok"] is True
    assert resp["state"] == "WAKE"
    # uptime_sec is a non-negative number.
    assert isinstance(resp["uptime_sec"], (int, float))


# ---------------------------------------------------------------------------
# Test 7: injected dispatcher receives request dicts unchanged
# ---------------------------------------------------------------------------

def test_socket_injected_dispatcher(tmp_path, lock_and_socket_paths):
    """pause/force_rem/tail_logs routed through injected dispatcher unchanged."""
    from iai_mcp.concurrency import ProcessLock, serve_control_socket

    _, sock_path = lock_and_socket_paths
    lock = ProcessLock(lock_and_socket_paths[0])

    received: list[dict] = []

    async def custom_dispatcher(req: dict) -> dict:
        received.append(req)
        return {"ok": True, "seen": req.get("type")}

    requests = [
        {"type": "pause", "seconds": 60},
        {"type": "force_rem"},
        {"type": "tail_logs", "n": 10},
    ]

    async def runner():
        shutdown = asyncio.Event()
        server_task = asyncio.create_task(
            serve_control_socket(
                store=None, lock=lock, state={}, shutdown=shutdown,
                dispatcher=custom_dispatcher, socket_path=sock_path,
            )
        )
        for _ in range(100):
            if sock_path.exists():
                break
            await asyncio.sleep(0.02)
        assert sock_path.exists()

        responses = []
        for req in requests:
            r, w = await asyncio.open_unix_connection(path=str(sock_path))
            w.write((json.dumps(req) + "\n").encode())
            await w.drain()
            line = await r.readline()
            responses.append(json.loads(line))
            w.close()
            try:
                await w.wait_closed()
            except Exception:
                pass

        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        return responses

    try:
        responses = asyncio.run(runner())
    finally:
        lock.close()

    assert received == requests, f"dispatcher saw {received!r}"
    for resp, req in zip(responses, requests):
        assert resp == {"ok": True, "seen": req["type"]}


# ---------------------------------------------------------------------------
# Test 8: stale socket cleanup (Pitfall 10)
# ---------------------------------------------------------------------------

def test_stale_socket_cleanup(tmp_path, lock_and_socket_paths):
    """Pre-existing socket file (SIGKILL-orphaned) is cleaned so bind succeeds."""
    from iai_mcp.concurrency import ProcessLock, serve_control_socket

    _, sock_path = lock_and_socket_paths
    # Simulate orphaned socket file.
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    sock_path.write_text("stale")
    assert sock_path.exists()

    lock = ProcessLock(lock_and_socket_paths[0])

    async def runner():
        shutdown = asyncio.Event()
        server_task = asyncio.create_task(
            serve_control_socket(store=None, lock=lock, state={}, shutdown=shutdown,
                                 socket_path=sock_path)
        )
        for _ in range(100):
            if sock_path.exists() and sock_path.stat().st_size == 0:
                # Socket replaces stale file; content is empty binary.
                break
            await asyncio.sleep(0.02)
        # Quick status round-trip to confirm server is live.
        r, w = await asyncio.open_unix_connection(path=str(sock_path))
        w.write(b'{"type":"status"}\n')
        await w.drain()
        line = await r.readline()
        w.close()
        try:
            await w.wait_closed()
        except Exception:
            pass
        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        return json.loads(line)

    try:
        resp = asyncio.run(runner())
    finally:
        lock.close()

    assert resp.get("ok") is True


# ---------------------------------------------------------------------------
# Test 9: 0o600 permissions on lock file + socket
# ---------------------------------------------------------------------------

def test_file_permissions_user_only(tmp_path, lock_and_socket_paths):
    """Lock + socket files must be 0o600 (user-only rw)."""
    from iai_mcp.concurrency import ProcessLock, serve_control_socket

    lock_path, sock_path = lock_and_socket_paths

    lock = ProcessLock(lock_path)
    # Lock file exists and has 0o600 mode.
    assert lock_path.exists()
    mode = lock_path.stat().st_mode & 0o777
    assert mode == 0o600, f"lock mode is {oct(mode)}, expected 0o600"

    async def runner():
        shutdown = asyncio.Event()
        server_task = asyncio.create_task(
            serve_control_socket(store=None, lock=lock, state={}, shutdown=shutdown,
                                 socket_path=sock_path)
        )
        for _ in range(100):
            if sock_path.exists():
                break
            await asyncio.sleep(0.02)
        # Check socket file mode.
        sock_mode = sock_path.stat().st_mode & 0o777
        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        return sock_mode

    try:
        sock_mode = asyncio.run(runner())
    finally:
        lock.close()
    assert sock_mode == 0o600, f"socket mode is {oct(sock_mode)}, expected 0o600"


# ---------------------------------------------------------------------------
# Test 10: holds_exclusive_nb cooperative-yield probe
# ---------------------------------------------------------------------------

def test_holds_exclusive_nb(tmp_path, lock_and_socket_paths):
    """holds_exclusive_nb returns True when we hold EX; False when contended.

    The probe MUST catch BlockingIOError/EWOULDBLOCK internally and never
    propagate the exception.
    """
    from iai_mcp.concurrency import ProcessLock

    lock_path, _ = lock_and_socket_paths
    daemon = ProcessLock(lock_path)
    try:
        # 1. Held exclusive -> probe returns True (no-op re-acquire).
        assert daemon.try_acquire_exclusive() is True
        assert daemon.holds_exclusive_nb() is True

        # 2. Release and let a child grab shared; probe now returns False.
        daemon.release()

        lock_path_str = str(lock_path)
        acquired_flag = tmp_path / ".shared_holder_acquired"
        release_flag = tmp_path / ".shared_holder_release"
        child = _SPAWN.Process(
            target=_child_hold_shared,
            args=(lock_path_str, str(acquired_flag), str(release_flag)),
        )
        child.start()
        try:
            deadline = time.time() + 15
            while time.time() < deadline and not acquired_flag.exists():
                time.sleep(0.05)
            assert acquired_flag.exists()

            # Daemon no longer holds EX, and child holds SH.
            # holds_exclusive_nb should return False without raising.
            assert daemon.holds_exclusive_nb() is False
        finally:
            release_flag.write_text("go")
            child.join(timeout=10)
            if child.is_alive():
                child.terminate()
                child.join(timeout=2)
    finally:
        daemon.close()
