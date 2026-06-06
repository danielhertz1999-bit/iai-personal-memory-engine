"""doctor.py multi-binder detection + repair.

Test matrix (8 tests):
  A. _extract_binder_pids parses lsof -F pn output → set[int]
  B. _extract_binder_pids skips PIDs bound to UNRELATED sockets
  C. _extract_binder_pids handles empty input → empty set
  D. check_g_no_dup_binders skips when socket file absent (PASS-with-skip)
  E. check_g_no_dup_binders PASSes with single binder (multiprocessing worker)
  F. check_g_no_dup_binders FAILs with two binders (regression-trap centerpiece)
  G. _kill_dup_binders keeps oldest, kills the rest (real subprocess daemons)
  H. iai-mcp doctor --apply --yes recovers from dup-binder scenario (e2e)

A-D: pure unit tests, no daemon, fast (<1s combined).
E-F: in-process multiprocessing workers — distinct PIDs, lsof-visible.
G-H: real iai_mcp.daemon subprocesses — required because _kill_dup_binders
     filters by 'iai_mcp.daemon' substring in psutil cmdline (wrong-PID-kill
     mitigation). Isolated by HIGH-4 LOCK env propagation pattern from
     test_doctor_apply_recovery.py:isolated_daemon_paths.

Skip on non-POSIX (AF_UNIX requirement).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX AF_UNIX required (lsof -U + multiprocessing socket binders)",
)


# ---------------------------------------------------------------------------
# Section 1 — pure unit tests for _extract_binder_pids (A, B, C)
# ---------------------------------------------------------------------------


def test_extract_binder_pids_parses_lsof_output():
    """A: hand-crafted lsof -F pn output → expected PID set.

    lsof -F pn format alternates lines `p<pid>` and `n<filename>`. Each
    PID is followed by 0+ name entries until the next `p<pid>`.
    """
    from iai_mcp.doctor import _extract_binder_pids

    target = Path("/tmp/iai-test/d.sock")
    lsof_output = "\n".join([
        "p12345",
        f"n{target}",
        "p67890",
        f"n{target}",
        "p99999",
        "n/tmp/other-app/socket",
    ])

    pids = _extract_binder_pids(lsof_output, target)

    assert pids == {12345, 67890}, f"expected {{12345, 67890}}, got {pids}"


def test_extract_binder_pids_skips_unrelated_sockets():
    """B: lsof output with multiple sockets; only PIDs holding OUR path are returned."""
    from iai_mcp.doctor import _extract_binder_pids

    target = Path("/tmp/iai-test/d.sock")
    lsof_output = "\n".join([
        "p1001",
        "n/var/run/some-other-daemon.sock",
        "p2002",
        f"n{target}",
        "p3003",
        "n/tmp/X11-unix/X0",
        "p4004",
        f"n{target}",
        "n/some/extra/name/for/p4004",  # PID 4004 holds multiple fds
    ])

    pids = _extract_binder_pids(lsof_output, target)

    assert pids == {2002, 4004}, f"expected {{2002, 4004}}, got {pids}"


def test_extract_binder_pids_handles_empty_output():
    """C: empty input → empty set (defensive corner case)."""
    from iai_mcp.doctor import _extract_binder_pids

    target = Path("/tmp/anywhere.sock")
    assert _extract_binder_pids("", target) == set()
    assert _extract_binder_pids("\n\n\n", target) == set()
    # Malformed: PID line without name line; name line without preceding PID.
    assert _extract_binder_pids("p123\nXgarbage\np\n", target) == set()


# ---------------------------------------------------------------------------
# Section 2 — check_g_no_dup_binders (D, E, F) using monkeypatched socket path
# ---------------------------------------------------------------------------


@pytest.fixture
def short_socket_path(tmp_path, monkeypatch):
    """Yield a short socket path under /tmp (AF_UNIX 104-byte cap on macOS).

    Honors the IAI_DAEMON_SOCKET_PATH env override that doctor._resolve_socket_path
    consults. Cleans up the socket file on teardown.
    """
    sock_dir = Path(f"/tmp/iai-mb-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))
    try:
        yield sock_path
    finally:
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass


def test_check_g_no_socket_skips(short_socket_path, monkeypatch):
    """D: socket file absent → PASS-with-skip detail "no socket file (skip)".

    Mirrors check_d_no_orphan_core's skip pattern when the resource isn't
    present (no false-positive on a clean machine).
    """
    from iai_mcp.doctor import check_g_no_dup_binders

    # Fixture set the env var; ensure no file exists.
    assert not short_socket_path.exists()

    result = check_g_no_dup_binders()

    assert result.passed is True
    assert "no socket file" in result.detail


# --- Multiprocessing worker for Tests E and F (distinct PIDs) ---------------


def _bind_socket_worker(sock_path_str: str, ready_event: mp.Event, exit_event: mp.Event) -> None:
    """Subprocess worker: bind an AF_UNIX socket to sock_path, signal ready,
    block until exit_event is set.

    Each multiprocessing.Process child has a distinct PID and lsof reports
    its socket fd. Used by Tests E (1 binder) and F (2 binders) to construct
    deterministic dup-binder scenarios without a real iai_mcp.daemon (whose
    boot cost is ~3-10s).
    """
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        # Each worker handles its own bind; for the 2-binder scenario, the
        # parent unlinks the path between worker spawns so each worker
        # successfully bind()s a fresh inode at the same name.
        s.bind(sock_path_str)
        s.listen(5)
        ready_event.set()
        # Block until parent signals shutdown.
        exit_event.wait(timeout=30)
    finally:
        try:
            s.close()
        except OSError:
            pass


def test_check_g_single_binder_passes(short_socket_path):
    """E: ONE binder bound to the socket → check_g returns PASS with "1 binder(s)".

    Uses a multiprocessing.Process worker (distinct PID from the pytest
    process) so lsof has something to enumerate.
    """
    from iai_mcp.doctor import check_g_no_dup_binders

    # NOTE: use 'spawn' (not 'fork') even on Darwin — lancedb is not fork-safe
    # (UserWarning surfaces with fork on macOS). Workers don't touch lancedb,
    # but the parent test process has it imported transitively; spawn isolates.
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    exit_signal = ctx.Event()
    worker = ctx.Process(
        target=_bind_socket_worker,
        args=(str(short_socket_path), ready, exit_signal),
    )
    worker.start()
    try:
        assert ready.wait(timeout=10), "binder worker never signaled ready"
        # Tiny settle so lsof's cache reflects the bind.
        time.sleep(0.2)

        result = check_g_no_dup_binders()

        assert result.passed is True, (
            f"single-binder scenario should PASS; got detail={result.detail!r}"
        )
        assert "1 binder" in result.detail, f"unexpected detail: {result.detail!r}"
    finally:
        exit_signal.set()
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2)


def test_check_g_two_binders_fails(short_socket_path):
    """F: TWO binders bound to the same socket path → check_g returns FAIL.

    REGRESSION-TRAP CENTERPIECE. Spawns 2 multiprocessing workers, each
    binding to the same socket path with an unlink between them so both
    bind() calls succeed at the OS level. lsof reports both PIDs as
    holding the path; check_g detects the singleton-invariant violation.

    This is exactly the failure mode 's launchd architecture
    structurally prevents in production — the test bypasses launchd by
    hand-binding sockets in worker processes. On post- production,
    this scenario can only occur if a user manually bypasses launchd.
    """
    from iai_mcp.doctor import _extract_binder_pids, check_g_no_dup_binders

    # NOTE: use 'spawn' (not 'fork') even on Darwin — lancedb is not fork-safe
    # (UserWarning surfaces with fork on macOS). Workers don't touch lancedb,
    # but the parent test process has it imported transitively; spawn isolates.
    ctx = mp.get_context("spawn")

    # Worker 1
    ready1 = ctx.Event()
    exit1 = ctx.Event()
    w1 = ctx.Process(
        target=_bind_socket_worker,
        args=(str(short_socket_path), ready1, exit1),
    )
    w1.start()

    # Worker 2 — race-window simulation: unlink the path so worker 2's bind()
    # creates a fresh inode at the same name. Worker 1's fd still holds the
    # ORIGINAL inode (unlinked but kept alive by the open fd); worker 2 holds
    # the NEW inode at the same path. lsof reports both PIDs.
    ready2 = ctx.Event()
    exit2 = ctx.Event()
    w2 = None
    try:
        assert ready1.wait(timeout=10), "worker 1 never signaled ready"
        # Unlink so the second bind doesn't EADDRINUSE.
        try:
            short_socket_path.unlink()
        except OSError:
            pass
        w2 = ctx.Process(
            target=_bind_socket_worker,
            args=(str(short_socket_path), ready2, exit2),
        )
        w2.start()
        assert ready2.wait(timeout=10), "worker 2 never signaled ready"
        time.sleep(0.3)  # let lsof catch up

        # Belt-and-suspenders: confirm via the parser directly that lsof sees both.
        lsof_out = subprocess.run(
            ["lsof", "-U", "-F", "pn"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
        binder_pids = _extract_binder_pids(lsof_out, short_socket_path)
        assert {w1.pid, w2.pid}.issubset(binder_pids), (
            f"lsof should report both worker PIDs as binders; got {binder_pids} "
            f"(workers: {w1.pid}, {w2.pid})"
        )

        # Centerpiece assertion: check_g detects the dup-binder scenario.
        result = check_g_no_dup_binders()

        assert result.passed is False, (
            f"two-binder scenario should FAIL; got detail={result.detail!r}"
        )
        # Detail mentions both PIDs.
        assert str(w1.pid) in result.detail, f"detail missing PID {w1.pid}: {result.detail!r}"
        assert str(w2.pid) in result.detail, f"detail missing PID {w2.pid}: {result.detail!r}"
    finally:
        exit1.set()
        if w2 is not None:
            exit2.set()
        for proc in (w1, w2):
            if proc is None:
                continue
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)


# ---------------------------------------------------------------------------
# Section 3 — _kill_dup_binders + e2e doctor --apply (G, H)
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_daemon_paths(tmp_path, monkeypatch):
    """HOME + socket + store + crypto env propagation for real-daemon tests.

    Mirrors test_doctor_apply_recovery.py:isolated_daemon_paths verbatim
    (HIGH-4 LOCK precedent). Required because _kill_dup_binders
    filters by 'iai_mcp.daemon' substring in psutil cmdline — only real
    iai_mcp.daemon subprocesses are killable, so multiprocessing workers
    cannot serve Tests G/H.
    """
    iai_dir = tmp_path / ".iai-mcp"
    iai_dir.mkdir(parents=True, exist_ok=True)

    state_path = iai_dir / ".daemon-state.json"
    lock_path = iai_dir / ".lock"
    store_dir = iai_dir / "store"
    store_dir.mkdir(parents=True, exist_ok=True)

    sock_dir = Path(f"/tmp/iai-mb2-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"

    real_hf_home = Path.home() / ".cache" / "huggingface"

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(real_hf_home))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(store_dir))
    monkeypatch.setenv("IAI_DAEMON_IDLE_SHUTDOWN_SECS", "99999")
    monkeypatch.setenv(
        "PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring"
    )
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-mb-passphrase")
    import keyring.core

    keyring.core._keyring_backend = None

    from iai_mcp import cli, daemon_state

    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    monkeypatch.setattr(cli, "LOCK_PATH", lock_path)
    monkeypatch.setattr(cli, "SOCKET_PATH", sock_path)

    try:
        yield sock_path, state_path, store_dir, lock_path
    finally:
        _kill_test_daemons(sock_path)
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass
        keyring.core._keyring_backend = None


def _spawn_daemon(sock_path: Path, store_dir: Path, home: Path) -> subprocess.Popen:
    """Spawn `python -m iai_mcp.daemon` with the test's env propagated."""
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_MCP_STORE"] = str(store_dir)
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = "99999"
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = "test-mb-passphrase"
    return subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_socket(sock_path: Path, timeout_sec: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if sock_path.exists():
            return True
        time.sleep(0.1)
    return False


def _kill_test_daemons(sock_path: Path) -> None:
    """Match-by-env cleanup: SIGTERM iai_mcp.daemon subprocesses whose
    psutil environ has our IAI_DAEMON_SOCKET_PATH value. Avoids touching
    the user's real production daemon.
    """
    target = str(sock_path)
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            if "iai_mcp.daemon" not in cl:
                continue
            try:
                env = p.environ()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            if env.get("IAI_DAEMON_SOCKET_PATH") == target:
                try:
                    p.send_signal(signal.SIGTERM)
                    p.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    try:
                        p.send_signal(signal.SIGKILL)
                    except psutil.NoSuchProcess:
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def _spawn_dup_daemons(
    sock_path: Path, store_dir: Path, home: Path
) -> tuple[subprocess.Popen, subprocess.Popen]:
    """Spawn 2 real iai_mcp.daemon subprocesses both bound to sock_path.

    Race-window simulation: spawn daemon #1, wait for
    socket, unlink (so daemon #2 can bind a fresh inode at the same path),
    spawn daemon #2, wait for socket. Daemon #1's listening fd still holds
    the original (now unlinked) inode; daemon #2 holds the new inode. lsof
    reports both PIDs as binders of the same path.
    """
    p1 = _spawn_daemon(sock_path, store_dir, home)
    if not _wait_for_socket(sock_path, timeout_sec=30):
        try:
            p1.kill()
        except ProcessLookupError:
            pass
        raise AssertionError("daemon #1 never bound socket within 30s")

    # Race-window: unlink so daemon #2's bind() succeeds without EADDRINUSE.
    try:
        sock_path.unlink()
    except OSError:
        pass

    p2 = _spawn_daemon(sock_path, store_dir, home)
    if not _wait_for_socket(sock_path, timeout_sec=30):
        try:
            p2.kill()
        except ProcessLookupError:
            pass
        try:
            p1.kill()
        except ProcessLookupError:
            pass
        raise AssertionError("daemon #2 never bound socket within 30s")

    # Settle so lsof reflects both binders.
    time.sleep(0.5)
    return p1, p2


@pytest.mark.skip(
    reason=(
        "Single-machine LifecycleLock prevents two daemons from both "
        "binding the same IAI_MCP_STORE. Daemon #2 raises "
        "LifecycleLockConflict and exits 1 before bind. The dup-binder "
        "integration scenario is now impossible by design. The unit tests "
        "in this file (test_extract_binder_pids_*, test_check_g_*) still "
        "cover check_g's detection logic without spawning two real daemons."
    )
)
def test_kill_dup_binders_keeps_oldest(isolated_daemon_paths):
    """G: 2 real daemons → _kill_dup_binders kills younger, keeps oldest.

    Re-running check_g afterward returns PASS (1 binder remaining).
    """
    from iai_mcp.doctor import (
        _extract_binder_pids,
        _kill_dup_binders,
        check_g_no_dup_binders,
    )

    sock_path, _, store_dir, _ = isolated_daemon_paths
    home = Path(os.environ["HOME"])

    p1, p2 = _spawn_dup_daemons(sock_path, store_dir, home)
    try:
        # Pre-condition: both daemons must show up as binders for our socket.
        lsof_out = subprocess.run(
            ["lsof", "-U", "-F", "pn"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
        binders = _extract_binder_pids(lsof_out, sock_path)
        assert {p1.pid, p2.pid}.issubset(binders), (
            f"expected both daemon PIDs in binders; got {binders} "
            f"(daemons: {p1.pid}, {p2.pid})"
        )
        pre_check = check_g_no_dup_binders()
        assert pre_check.passed is False, (
            f"pre-condition: dup-binder scenario should FAIL check_g; "
            f"got {pre_check.detail!r}"
        )

        # Kill the younger daemon. p1 was spawned first → has greater etime →
        # is the keep_pid; p2 should be killed.
        ok, msg, ms = _kill_dup_binders()

        assert ok is True, f"_kill_dup_binders returned ok=False: {msg}"
        assert "kept PID" in msg, f"msg missing 'kept PID': {msg!r}"
        assert "killed" in msg, f"msg missing 'killed': {msg!r}"
        assert ms < 10_000, f"_kill_dup_binders took {ms}ms (>10s); too slow"

        # After kill, a follow-up check_g should report 1 (or 0 — race) binder.
        post_check = check_g_no_dup_binders()
        assert post_check.passed is True, (
            f"post-kill check_g should PASS; got {post_check.detail!r}"
        )

        # The kept daemon (p1) should still be alive; the other should be dead
        # within a generous timeout (kill is SIGKILL, instant on macOS).
        assert p1.poll() is None, "expected oldest daemon (p1) to survive"
        # Allow up to 2s for SIGKILL signal delivery + reap.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and p2.poll() is None:
            time.sleep(0.1)
        assert p2.poll() is not None, "expected younger daemon (p2) to be dead"
    finally:
        for proc in (p1, p2):
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGKILL)
                    proc.wait(timeout=3)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    pass


@pytest.mark.skip(
    reason=(
        "Single-machine LifecycleLock prevents two daemons from both "
        "binding the same IAI_MCP_STORE. Daemon #2 raises "
        "LifecycleLockConflict and exits 1 before bind. End-to-end "
        "recovery from dup-binders cannot run because the dup-binders "
        "state is now impossible to construct."
    )
)
def test_doctor_apply_yes_recovers_from_dup_binders(isolated_daemon_paths):
    """H: end-to-end. 2 dup-binder daemons → cmd_doctor(apply=True, yes=True)
    drives the kill_dup_binders repair → re-check returns 0 OR exit 2 only
    if a non-related check (e.g., (a) state desync) FAILs.

    NB: spawning two real daemons against the same socket inevitably leaves
    daemon-state.json pointing at one of the two PIDs (whichever wrote last).
    After kill_dup_binders, if the survivor is the one daemon-state recorded,
    check_a passes; if the survivor is the OTHER daemon, check_a FAILs and the
    respawn action triggers, which (because the surviving daemon already binds
    the socket) yields a launchd-react-noop OR a benign respawn-timeout. The
    relevant assertion for THIS test is the dup-binder repair specifically:
    after recovery, lsof reports exactly 1 binder for our socket path. The
    overall rc and check_a status are looser assertions because they depend
    on the state-file-vs-survivor coincidence.
    """
    from iai_mcp.doctor import (
        _extract_binder_pids,
        check_g_no_dup_binders,
        cmd_doctor,
    )

    sock_path, _, store_dir, _ = isolated_daemon_paths
    home = Path(os.environ["HOME"])

    p1, p2 = _spawn_dup_daemons(sock_path, store_dir, home)
    try:
        # Sanity: dup-binder is detectable.
        pre = check_g_no_dup_binders()
        assert pre.passed is False, f"pre: dup-binder should FAIL; got {pre.detail!r}"

        args = argparse.Namespace(apply=True, yes=True)
        rc = cmd_doctor(args)

        # The critical observable: dup-binders cleared.
        post_check = check_g_no_dup_binders()
        assert post_check.passed is True, (
            f"post-recovery: check_g should PASS; got {post_check.detail!r}"
        )
        # rc may be 0 (everything green) or 2 (only check_a survived as FAIL
        # because state-file PID points at the killed survivor); both prove
        # the dup-binder repair mechanism worked. rc=1 would mean --apply
        # never ran the repair (regression).
        assert rc in (0, 2), (
            f"cmd_doctor rc={rc} unexpected; allowed 0 (full recovery) or 2 "
            f"(dup-binders fixed but state-file desync persists)."
        )

        # Belt-and-suspenders: lsof confirms exactly 1 binder remains.
        lsof_out = subprocess.run(
            ["lsof", "-U", "-F", "pn"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
        binders = _extract_binder_pids(lsof_out, sock_path)
        assert len(binders) <= 1, (
            f"after recovery, expected ≤1 binder for {sock_path}; got {binders}"
        )
    finally:
        for proc in (p1, p2):
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGKILL)
                    proc.wait(timeout=3)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    pass
