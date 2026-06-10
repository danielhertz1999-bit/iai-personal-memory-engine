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


def test_extract_binder_pids_parses_lsof_output():
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
        "n/some/extra/name/for/p4004",
    ])

    pids = _extract_binder_pids(lsof_output, target)

    assert pids == {2002, 4004}, f"expected {{2002, 4004}}, got {pids}"


def test_extract_binder_pids_handles_empty_output():
    from iai_mcp.doctor import _extract_binder_pids

    target = Path("/tmp/anywhere.sock")
    assert _extract_binder_pids("", target) == set()
    assert _extract_binder_pids("\n\n\n", target) == set()
    assert _extract_binder_pids("p123\nXgarbage\np\n", target) == set()


@pytest.fixture
def short_socket_path(tmp_path, monkeypatch):
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
    from iai_mcp.doctor import check_g_no_dup_binders

    assert not short_socket_path.exists()

    result = check_g_no_dup_binders()

    assert result.passed is True
    assert "no socket file" in result.detail


def _bind_socket_worker(sock_path_str: str, ready_event: mp.Event, exit_event: mp.Event) -> None:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.bind(sock_path_str)
        s.listen(5)
        ready_event.set()
        exit_event.wait(timeout=30)
    finally:
        try:
            s.close()
        except OSError:
            pass


def test_check_g_single_binder_passes(short_socket_path):
    from iai_mcp.doctor import check_g_no_dup_binders

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
    from iai_mcp.doctor import _extract_binder_pids, check_g_no_dup_binders

    ctx = mp.get_context("spawn")

    ready1 = ctx.Event()
    exit1 = ctx.Event()
    w1 = ctx.Process(
        target=_bind_socket_worker,
        args=(str(short_socket_path), ready1, exit1),
    )
    w1.start()

    ready2 = ctx.Event()
    exit2 = ctx.Event()
    w2 = None
    try:
        assert ready1.wait(timeout=10), "worker 1 never signaled ready"
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
        time.sleep(0.3)

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

        result = check_g_no_dup_binders()

        assert result.passed is False, (
            f"two-binder scenario should FAIL; got detail={result.detail!r}"
        )
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


@pytest.fixture
def isolated_daemon_paths(tmp_path, monkeypatch):
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
    p1 = _spawn_daemon(sock_path, store_dir, home)
    if not _wait_for_socket(sock_path, timeout_sec=30):
        try:
            p1.kill()
        except ProcessLookupError:
            pass
        raise AssertionError("daemon #1 never bound socket within 30s")

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
    from iai_mcp.doctor import (
        _extract_binder_pids,
        _kill_dup_binders,
        check_g_no_dup_binders,
    )

    sock_path, _, store_dir, _ = isolated_daemon_paths
    home = Path(os.environ["HOME"])

    p1, p2 = _spawn_dup_daemons(sock_path, store_dir, home)
    try:
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

        ok, msg, ms = _kill_dup_binders()

        assert ok is True, f"_kill_dup_binders returned ok=False: {msg}"
        assert "kept PID" in msg, f"msg missing 'kept PID': {msg!r}"
        assert "killed" in msg, f"msg missing 'killed': {msg!r}"
        assert ms < 10_000, f"_kill_dup_binders took {ms}ms (>10s); too slow"

        post_check = check_g_no_dup_binders()
        assert post_check.passed is True, (
            f"post-kill check_g should PASS; got {post_check.detail!r}"
        )

        assert p1.poll() is None, "expected oldest daemon (p1) to survive"
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
    from iai_mcp.doctor import (
        _extract_binder_pids,
        check_g_no_dup_binders,
        cmd_doctor,
    )

    sock_path, _, store_dir, _ = isolated_daemon_paths
    home = Path(os.environ["HOME"])

    p1, p2 = _spawn_dup_daemons(sock_path, store_dir, home)
    try:
        pre = check_g_no_dup_binders()
        assert pre.passed is False, f"pre: dup-binder should FAIL; got {pre.detail!r}"

        args = argparse.Namespace(apply=True, yes=True)
        rc = cmd_doctor(args)

        post_check = check_g_no_dup_binders()
        assert post_check.passed is True, (
            f"post-recovery: check_g should PASS; got {post_check.detail!r}"
        )
        assert rc in (0, 2), (
            f"cmd_doctor rc={rc} unexpected; allowed 0 (full recovery) or 2 "
            f"(dup-binders fixed but state-file desync persists)."
        )

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
