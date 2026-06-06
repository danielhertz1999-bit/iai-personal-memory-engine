"""Doctor 6-row PASS/FAIL checklist.

Each individual failure scenario produces a FAIL on the matching check
and the doctor exits with the documented code (0=all pass,
1=any FAIL no --apply, 2=--apply but FAIL persists).

Checks (in order):
  (a) daemon process alive       — daemon_pid in .daemon-state.json
  (b) socket file fresh          — kernel-level connect() succeeds <1s
  (c) lock file healthy          — fcntl probe doesn't error
  (d) no orphan iai_mcp.core procs — psutil scan returns 0
  (e) daemon state file valid    — fsm_state ∈ {WAKE, SLEEPING, DREAMING}
  (f) store readable             — MemoryStore() opens without error

Tests use monkeypatching to construct each failure scenario in isolation
without booting a real daemon (test_doctor_apply_recovery.py covers the
end-to-end recovery scenario with a real subprocess daemon).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures: tmp socket + state + lock + store paths
# ---------------------------------------------------------------------------


@pytest.fixture
def short_socket_paths(tmp_path, monkeypatch):
    """Yield (lock_path, sock_path, state_path) under tmp dirs.

    AF_UNIX on macOS caps socket paths at ~104 bytes; pytest's tmp_path can
    be too long under xdist. Use a short /tmp/iai-doc-<pid>-<n>/ fallback
    for the socket.

    Monkeypatches:
      - IAI_DAEMON_SOCKET_PATH env (read by doctor._resolve_socket_path)
      - iai_mcp.daemon_state.STATE_PATH (read by check (a)/(e) load_state)
      - iai_mcp.cli.LOCK_PATH (read by check (c) ProcessLock)
      - IAI_MCP_STORE env (read by check (f) MemoryStore)
    """
    lock_path = tmp_path / ".lock"
    sock_dir = Path(f"/tmp/iai-doc-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    state_path = tmp_path / ".daemon-state.json"
    store_dir = tmp_path / "store"
    store_dir.mkdir(parents=True, exist_ok=True)

    from iai_mcp import cli, daemon_state

    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(store_dir))
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    monkeypatch.setattr(cli, "LOCK_PATH", lock_path)
    # Also patch cli.SOCKET_PATH as a defensive fallback — doctor's
    # _resolve_socket_path prefers the env var, but if env propagation is
    # ever removed this guarantees test isolation.
    monkeypatch.setattr(cli, "SOCKET_PATH", sock_path)

    try:
        yield lock_path, sock_path, state_path
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clean_environment_yields_check_a_fail_exit_1(short_socket_paths, capsys):
    """Clean tmp env (no daemon, no state file) → cmd_doctor returns 1.

    Check (a) reports ABSENT (no daemon_pid). Check (e) PASSES (no state file
    is acceptable — daemon never booted). Other FAILs depend on host process
    table for (d), but exit code is 1 either way (any FAIL → 1 without --apply).
    """
    from iai_mcp.doctor import cmd_doctor

    args = argparse.Namespace(apply=False, yes=False)
    rc = cmd_doctor(args)
    captured = capsys.readouterr()

    assert rc == 1, f"expected 1 (FAIL no --apply), got {rc}"
    assert "iai doctor" in captured.out
    assert "(a) daemon process alive" in captured.out
    assert "ABSENT" in captured.out, "check (a) should say ABSENT when no daemon_pid"


@pytest.mark.parametrize(
    "scenario,expected_fail_check",
    [
        ("no_daemon_pid", "(a) daemon process alive"),
        ("dead_pid_in_state", "(a) daemon process alive"),
        ("stale_socket_unconnectable", "(b) socket file fresh"),
        ("orphan_core_procs", "(d) no orphan iai_mcp.core procs"),
        ("corrupt_state_fsm", "(e) daemon state file valid"),
    ],
)
def test_individual_failure_modes(
    scenario, expected_fail_check, short_socket_paths, monkeypatch
):
    """Each failure scenario produces a FAIL on the matching check.

    Cascading FAILs are allowed (e.g. dead daemon → check_a + check_b both
    fail) but the named expected_fail_check MUST appear in the FAIL list.
    """
    _, sock_path, state_path = short_socket_paths

    if scenario == "no_daemon_pid":
        # State file absent → check (a) FAIL with ABSENT.
        # Default fixture state — nothing more to do.
        pass

    elif scenario == "dead_pid_in_state":
        # Stamp a high PID that almost certainly doesn't exist on a fresh
        # macOS / Linux box. Stay well under INT_MAX (2^31-1) so os.kill
        # doesn't raise OverflowError before the ProcessLookupError path.
        # PID_MAX defaults: macOS 99_999, Linux 4_194_304 — value 2_000_000
        # is above both default ranges (effectively guaranteed unallocated).
        state_path.write_text(json.dumps({"daemon_pid": 2_000_000, "fsm_state": "WAKE"}))

    elif scenario == "stale_socket_unconnectable":
        # Create the socket file as a regular file (not a real socket) → connect
        # raises ConnectionRefusedError or OSError. check (b) FAIL.
        sock_path.write_text("")

    elif scenario == "orphan_core_procs":
        # Monkeypatch psutil.process_iter to return a synthetic orphan hit.
        # Avoids actually spawning python -m iai_mcp.core (which would launch
        # a real Python core and pollute the process table for sibling tests).
        import psutil

        class _FakeProc:
            def __init__(self, pid: int, cmdline: list[str]):
                self.info = {"pid": pid, "cmdline": cmdline}

        fake = _FakeProc(99_999, ["python", "-m", "iai_mcp.core"])
        monkeypatch.setattr(
            psutil, "process_iter", lambda *a, **kw: [fake]
        )

    elif scenario == "corrupt_state_fsm":
        # Write an invalid fsm_state value → check (e) FAIL.
        state_path.write_text(json.dumps({"fsm_state": "INVALID_STATE_VALUE"}))

    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    fail_names = [r.name for r in results if not r.passed]
    assert expected_fail_check in fail_names, (
        f"Expected FAIL on '{expected_fail_check}' for scenario '{scenario}'; "
        f"got fails: {fail_names}"
    )


def test_print_checklist_format_six_rows(short_socket_paths, monkeypatch, capsys):
    """print_checklist always emits 6 PASS/FAIL rows with consistent header.

    Forces all 6 checks to PASS via monkeypatching to verify the formatter
    handles a fully-green checklist (default scenario in the other tests
    only verifies the FAIL path).
    """
    from iai_mcp import doctor

    forced_results = [
        doctor.CheckResult("(a) daemon process alive", True, "PID 99999 (iai_mcp.daemon)"),
        doctor.CheckResult("(b) socket file fresh", True, "connected in 5 ms"),
        doctor.CheckResult("(c) lock file healthy", True, "acquirable"),
        doctor.CheckResult("(d) no orphan iai_mcp.core procs", True, "0 found"),
        doctor.CheckResult("(e) daemon state file valid", True, "fsm_state=WAKE"),
        doctor.CheckResult("(f) hippo storage readable", True, "Hippo storage opens without error"),
    ]
    doctor.print_checklist(forced_results)
    out = capsys.readouterr().out

    assert "iai doctor" in out
    assert out.count("[PASS]") == 6
    assert out.count("[FAIL]") == 0


def test_all_pass_returns_exit_0(short_socket_paths, monkeypatch, capsys):
    """Exit 0: when run_diagnosis returns all PASS, cmd_doctor returns 0.

    Monkeypatches run_diagnosis itself rather than constructing a passing
    world — the latter requires a real daemon subprocess (covered by
    test_doctor_apply_recovery.py).
    """
    from iai_mcp import doctor

    forced_pass = [
        doctor.CheckResult(name, True, "synthetic pass") for name in (
            "(a) daemon process alive",
            "(b) socket file fresh",
            "(c) lock file healthy",
            "(d) no orphan iai_mcp.core procs",
            "(e) daemon state file valid",
            "(f) hippo storage readable",
        )
    ]
    monkeypatch.setattr(doctor, "run_diagnosis", lambda: forced_pass)

    args = argparse.Namespace(apply=False, yes=False)
    rc = doctor.cmd_doctor(args)
    out = capsys.readouterr().out

    assert rc == 0
    assert "All checks passed" in out


def test_apply_without_yes_warns_when_yes_alone(short_socket_paths, monkeypatch, capsys):
    """UX: --yes without --apply prints a warning to stderr but still
    runs diagnosis (does not block the user).
    """
    from iai_mcp import doctor

    args = argparse.Namespace(apply=False, yes=True)
    rc = doctor.cmd_doctor(args)
    captured = capsys.readouterr()

    # The warning goes to stderr.
    assert "--yes without --apply is meaningless" in captured.err
    # Diagnosis still runs — exit code mirrors check outcome (likely 1
    # because no daemon is running in the tmp env).
    assert rc in (0, 1)


def test_exit_code_2_when_apply_cannot_fix(short_socket_paths, monkeypatch, capsys):
    """--apply runs all repair actions but final re-check still has
    FAIL → exit 2.

    Construct a scenario where the FAIL is unfixable: corrupt fsm_state in
    the state file. _plan_repair_actions has no action mapped to check (e),
    so the FAIL persists through the re-check and cmd_doctor returns 2.
    """
    _, _, state_path = short_socket_paths
    # Write an invalid fsm_state so check (e) always FAILs.
    state_path.write_text(json.dumps({"fsm_state": "TOTALLY_BOGUS"}))

    # Also force every other check to PASS via run_diagnosis monkeypatch
    # so we isolate check (e) as the persistent FAIL. The first call returns
    # the bogus-state results; the second (after --apply) returns the same.
    from iai_mcp import doctor

    def _forced_fail_e_only():
        return [
            doctor.CheckResult("(a) daemon process alive", True, "synthetic"),
            doctor.CheckResult("(b) socket file fresh", True, "synthetic"),
            doctor.CheckResult("(c) lock file healthy", True, "synthetic"),
            doctor.CheckResult("(d) no orphan iai_mcp.core procs", True, "synthetic"),
            doctor.CheckResult(
                "(e) daemon state file valid",
                False,
                "fsm_state='TOTALLY_BOGUS' not in [...]",
            ),
            doctor.CheckResult("(f) hippo storage readable", True, "synthetic"),
        ]

    monkeypatch.setattr(doctor, "run_diagnosis", _forced_fail_e_only)

    args = argparse.Namespace(apply=True, yes=True)
    rc = doctor.cmd_doctor(args)
    out = capsys.readouterr().out

    assert rc == 2, f"expected 2 (--apply tried but FAIL persists), got {rc}"
    assert "STILL BROKEN" in out
    assert "(e) daemon state file valid" in out


def test_check_b_returns_fail_when_socket_missing(short_socket_paths):
    """Check (b) returns FAIL with explicit "does not exist" diagnosis when
    the socket file is missing entirely (not just unreachable).
    """
    _, sock_path, _ = short_socket_paths
    # Defensive: ensure socket truly absent.
    if sock_path.exists():
        sock_path.unlink()

    from iai_mcp.doctor import check_b_socket_fresh

    result = check_b_socket_fresh()
    assert result.passed is False
    assert "does not exist" in result.detail


def test_check_e_passes_when_state_file_absent(short_socket_paths):
    """Check (e) PASSES when state file is absent (daemon never booted is
    not a bug at this layer — check (a) catches it as ABSENT).
    """
    _, _, state_path = short_socket_paths
    if state_path.exists():
        state_path.unlink()

    from iai_mcp.doctor import check_e_state_file_valid

    result = check_e_state_file_valid()
    assert result.passed is True
    assert "no state file" in result.detail


def test_check_b_passes_against_silent_listening_socket(short_socket_paths):
    """Check (b) PASSes when the socket accepts connections but never replies.

    Regression for a past false-positive: the previous implementation
    issued a {type: status} round-trip with a 250 ms wall, which false-FAILed
    on a healthy daemon whose status reply path took 1-8 s. The fix is
    connect-only: a successful kernel-level connect() means the socket is
    fresh; daemon-responsiveness belongs to a separate diagnostic.

    Reproduce by standing up a unix socket listener that accepts but never
    sends a reply. The probe must PASS (connect succeeded), not FAIL on
    a missing reply.
    """
    import socket as _socket
    import threading
    import time as _time

    _, sock_path, _ = short_socket_paths
    if sock_path.exists():
        sock_path.unlink()

    server = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(8)
    # Accept-and-hold loop in a background thread: pulls one connection,
    # then sits silent. The probe only needs the kernel accept() to succeed.
    stop = threading.Event()
    accepted: list = []

    def _accept_loop():
        while not stop.is_set():
            try:
                server.settimeout(0.05)
                try:
                    conn, _ = server.accept()
                except (_socket.timeout, OSError):
                    continue
                accepted.append(conn)  # hold ref so it doesn't garbage-collect
            except Exception:
                break

    th = threading.Thread(target=_accept_loop, daemon=True)
    th.start()

    try:
        # Give the listen() backlog a moment to be ready.
        _time.sleep(0.05)
        from iai_mcp.doctor import check_b_socket_fresh

        result = check_b_socket_fresh()
        assert result.passed is True, (
            f"check_b should PASS against a silent listening socket; "
            f"got: {result.detail}"
        )
        assert "connected" in result.detail
    finally:
        stop.set()
        try:
            for c in accepted:
                try:
                    c.close()
                except OSError:
                    pass
        finally:
            try:
                server.close()
            except OSError:
                pass
        th.join(timeout=1.0)


def test_check_b_fails_when_socket_is_regular_file(short_socket_paths):
    """Check (b) FAILs when the socket path is a regular file (not a socket).

    Mirrors the existing `stale_socket_unconnectable` parametrized scenario
    but asserts the post-fix error string still surfaces "unreachable" so
    `_plan_repair_actions` can map this FAIL to `unlink_stale_socket`.
    """
    _, sock_path, _ = short_socket_paths
    if sock_path.exists():
        sock_path.unlink()
    sock_path.write_text("")  # regular file, not a socket

    from iai_mcp.doctor import check_b_socket_fresh

    result = check_b_socket_fresh()
    assert result.passed is False
    # Either ConnectionRefused or OSError errno=38 (ENOTSOCK) — both
    # are "unreachable" per the post-fix error message contract.
    assert "unreachable" in result.detail
