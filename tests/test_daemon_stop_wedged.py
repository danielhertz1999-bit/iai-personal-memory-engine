"""Hermetic proof that the cross-process SIGTERM->bounded-wait->SIGKILL
escalation (the mechanism `iai-mcp daemon stop` now uses by default)
terminates a WEDGED asyncio event loop within a bound.

This mirrors the shipped stop guarantee without touching the real daemon:

  - We spawn a tiny CHILD process whose asyncio loop installs an in-loop
    SIGTERM handler (exactly like the daemon: `loop.add_signal_handler`),
    signals readiness, then WEDGES the loop with a long synchronous sleep
    so the loop can never service the handler callback.
  - We then drive the SAME escalation the production stop uses: a direct
    `os.kill(pid, SIGTERM)` followed, on a bounded `_is_pid_alive` poll
    timeout, by `os.kill(pid, SIGKILL)`.
  - Assert the SIGTERM phase ALONE does NOT terminate the wedged child
    within the bound (the gap is real -- the wedged loop absorbs it), and
    the SIGKILL escalation DOES terminate it within the bound.

Hermeticity: a private socket path + a TEST process label argv token --
NEVER the real `com.iai-mcp.daemon`, NEVER a real PID, no launchctl. The
child is reaped in a `finally` block.

The child argv carries an inert `iai_mcp.daemon` token ONLY so the
PID-recycle-safe `_is_pid_alive` gate (which cross-checks the cmdline for
that substring) recognises the hermetic child. The child does NOT import
`iai_mcp` -- it is a self-contained stdlib-only process.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from iai_mcp.lifecycle_lock import _is_pid_alive


# Child program: install an in-loop SIGTERM handler, touch the ready file,
# then wedge the loop with a blocking sleep so the handler never runs.
# argv: [python, "-c", <prog>, <ready_path>, <test_socket_path>, "iai_mcp.daemon"]
_WEDGED_CHILD = r"""
import asyncio, os, signal, sys, time

ready_path = sys.argv[1]
sock_path = sys.argv[2]  # test-only socket path; never the real daemon socket

async def _main():
    loop = asyncio.get_running_loop()
    # Faithful to the daemon: register a loop-routed SIGTERM handler that a
    # wedged loop can never service.
    serviced = {"sigterm": False}
    loop.add_signal_handler(signal.SIGTERM, lambda: serviced.__setitem__("sigterm", True))
    # Signal readiness AFTER the handler is installed.
    with open(ready_path, "w") as fh:
        fh.write(str(os.getpid()))
        fh.flush()
        os.fsync(fh.fileno())
    # Wedge the loop: a long synchronous sleep holds the loop thread so the
    # SIGTERM callback can never fire. (A busy `while True: pass` would model
    # it too; a blocking sleep wedges just as effectively without pegging a
    # core for the duration of the bounded test.)
    time.sleep(3600)

asyncio.run(_main())
"""


def _spawn_wedged_child(tmp_path):
    ready_path = tmp_path / "child_ready"
    sock_path = tmp_path / "test_daemon.sock"  # never the real socket
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _WEDGED_CHILD,
            str(ready_path),
            str(sock_path),
            # Inert recycle-gate token; the child does NOT import iai_mcp.
            "iai_mcp.daemon",
        ],
    )
    # Wait for the child to install the handler + signal ready.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if ready_path.exists():
            return proc
        if proc.poll() is not None:
            raise AssertionError(
                f"wedged child exited early rc={proc.returncode}"
            )
        time.sleep(0.02)
    raise AssertionError("wedged child did not signal readiness within 5s")


def test_wedged_loop_survives_sigterm_then_killed_by_escalation(tmp_path):
    proc = _spawn_wedged_child(tmp_path)
    pid = proc.pid
    try:
        # Sanity: the recycle-safe gate recognises our hermetic child.
        assert _is_pid_alive(pid), "child not recognised as alive by the gate"

        # --- Step 1: SIGTERM ALONE must NOT terminate the wedged child ---
        os.kill(pid, signal.SIGTERM)
        sigterm_bound = time.monotonic() + 1.5
        while time.monotonic() < sigterm_bound:
            if not _is_pid_alive(pid):
                break
            time.sleep(0.05)
        assert _is_pid_alive(pid), (
            "wedged child died from SIGTERM alone -- the in-loop handler gap "
            "is not being reproduced (the loop must absorb the SIGTERM)"
        )

        # --- Step 2: escalate to SIGKILL after the bounded wait ---
        # (This is exactly the production stop's path-B escalation.)
        os.kill(pid, signal.SIGKILL)
        kill_bound = time.monotonic() + 5.0
        dead = False
        while time.monotonic() < kill_bound:
            if not _is_pid_alive(pid):
                dead = True
                break
            time.sleep(0.02)
        # Reap so the gate sees a fully-gone process (not a zombie).
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass
        assert dead or proc.poll() is not None, (
            "SIGKILL escalation did not terminate the wedged child within "
            "the bound"
        )
        assert proc.poll() is not None, "child still running after SIGKILL"
        # Returncode reflects the uncatchable signal kill.
        assert proc.returncode in (-signal.SIGKILL, signal.SIGKILL, 137), (
            f"unexpected child returncode {proc.returncode}"
        )
    finally:
        # Defensive reap: never leave the hermetic child behind.
        if proc.poll() is None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass


def test_escalation_only_targets_the_spawned_child_pid(tmp_path):
    """Guard: the escalation we drive signals only our own child PID -- the
    test never references the real daemon label/socket/PID."""
    proc = _spawn_wedged_child(tmp_path)
    pid = proc.pid
    try:
        assert pid != os.getpid()
        assert _is_pid_alive(pid)
        os.kill(pid, signal.SIGKILL)
        proc.wait(timeout=5.0)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            try:
                os.kill(pid, signal.SIGKILL)
                proc.wait(timeout=5.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
