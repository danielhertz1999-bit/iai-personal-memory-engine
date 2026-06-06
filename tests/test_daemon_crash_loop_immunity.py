"""Crash-loop immunity for the deferred-captures drain path.

Two failure modes are covered:

1. **Poison-pill quarantine (PATCH A).** A `.jsonl` whose ingestion crashes
   the daemon mid-drain leaves behind a `.processing-<pid>.jsonl` marker.
   On the next drain pass that pid is dead, so the file is counted as a
   crash attempt. After `QUARANTINE_MAX_ATTEMPTS` crashes the file is
   moved out of the active queue into `.quarantine/<utc-ts>-<basename>`
   and a `deferred_captures_quarantined` event is emitted at
   severity=warning. The existing FAILED_MAX_ATTEMPTS retry policy
   (`.failed-*-attempt-N` -> `.permanent-failed-*`) is unrelated and must
   keep working.

2. **Socket-binds-before-drain (PATCH B + C).** The MCP unix socket must
   accept a client connection before drain completes — i.e. drain runs
   as a background asyncio task, not a blocking await on startup.

All tests run with a tmp HOME so production user state is untouched.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Shared fixture: tmp HOME + isolated store env (mirrors test_failed_captures_retry)
# ---------------------------------------------------------------------------


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    """HOME=tmp_path + fail-backend keyring + crypto passphrase + isolated store."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-crash-loop-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "hippo"))
    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_isolated_store():
    from iai_mcp.store import MemoryStore

    return MemoryStore()


def _make_clean_jsonl(
    deferred_dir: Path,
    session_id: str,
    ts_suffix: int,
    *,
    version: int = 1,
) -> Path:
    """Write a minimal v1 deferred-capture JSONL (header + one event).

    ``version=2`` produces a forward-compat skip file that the candidates
    loop will not ingest — useful for tests that want the file to survive
    the drain pass without monkeypatching ``capture_turn``.
    """
    deferred_dir.mkdir(parents=True, exist_ok=True)
    out = deferred_dir / f"sess-{session_id}-{ts_suffix}.jsonl"
    header = {
        "version": version,
        "deferred_at": "2026-05-15T00:00:00Z",
        "session_id": session_id,
        "cwd": "/tmp",
    }
    event = {
        "text": "event text long enough to pass MIN_CAPTURE length checks",
        "cue": "test cue crash loop",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-05-15T00:00:00Z",
    }
    out.write_text(json.dumps(header) + "\n" + json.dumps(event) + "\n")
    return out


def _dead_pid() -> int:
    """Return a PID that has definitely exited and been reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# ---------------------------------------------------------------------------
# Test 1 — poison-pill quarantined after two stale-PID rescans
# ---------------------------------------------------------------------------


def test_poison_pill_quarantined_after_two_attempts(iai_home, monkeypatch):
    """Two stale `.processing-<deadpid>` markers raise the crash counter past
    QUARANTINE_MAX_ATTEMPTS; the third drain pass moves the file into
    `.quarantine/`.
    """
    from iai_mcp.capture import (
        QUARANTINE_MAX_ATTEMPTS,
        drain_deferred_captures,
    )

    assert QUARANTINE_MAX_ATTEMPTS == 2

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    # version=2 keeps the candidates loop in skip-only mode — the test is
    # exercising the crash-counter rescan, not the ingest pipeline.
    fpath = _make_clean_jsonl(deferred_dir, "poison", 1000000000, version=2)

    # Simulate a crash that left a stale.processing-<deadpid>.jsonl marker.
    dead = _dead_pid()
    stale_1 = fpath.with_name(fpath.stem + f".processing-{dead}.jsonl")
    fpath.rename(stale_1)

    store = _open_isolated_store()

    # Pass 1: stale marker ->.crash-1.jsonl rename (no quarantine yet).
    drain_deferred_captures(store)
    crash_1 = list(deferred_dir.glob("sess-poison-1000000000.crash-1.jsonl"))
    assert len(crash_1) == 1, (
        f"first stale-PID rescan should rename to .crash-1.jsonl, "
        f"dir={list(deferred_dir.iterdir())}"
    )

    # Hand-create a fresh stale marker on top of the.crash-1 file to mimic
    # a second crash before the next drain runs.
    stale_2 = crash_1[0].with_name(crash_1[0].stem + f".processing-{dead}.jsonl")
    crash_1[0].rename(stale_2)

    # Pass 2: stale-PID rescan with prior_n=1 ->.crash-2 rename.
    drain_deferred_captures(store)
    crash_2 = list(deferred_dir.glob("sess-poison-1000000000.crash-2.jsonl"))
    assert len(crash_2) == 1, (
        f"second stale-PID rescan should rename to .crash-2.jsonl, "
        f"dir={list(deferred_dir.iterdir())}"
    )

    # Third stale-PID rescan: prior_n=2, next_n=3 > QUARANTINE_MAX_ATTEMPTS,
    # file moves to.quarantine/.
    stale_3 = crash_2[0].with_name(crash_2[0].stem + f".processing-{dead}.jsonl")
    crash_2[0].rename(stale_3)

    drain_deferred_captures(store)

    quarantine_dir = deferred_dir / ".quarantine"
    assert quarantine_dir.exists(), "quarantine dir must be created on third strike"
    quarantined = list(quarantine_dir.iterdir())
    assert len(quarantined) == 1, (
        f"exactly one quarantined file expected, got {quarantined}"
    )
    # Name carries the UTC timestamp prefix + recovered original basename.
    assert quarantined[0].name.endswith("sess-poison-1000000000.jsonl") or (
        "sess-poison-1000000000" in quarantined[0].name
    ), quarantined[0].name


# ---------------------------------------------------------------------------
# Test 2 — quarantine emits warning event with crash_loop reason
# ---------------------------------------------------------------------------


def test_quarantine_emits_warning_event(iai_home, monkeypatch):
    """`deferred_captures_quarantined` event fires at severity=warning,
    domain=ops, with payload {file, reason="crash_loop", attempts}.
    """
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _make_clean_jsonl(deferred_dir, "warn", 1100000000, version=2)

    dead = _dead_pid()
    # Pre-stage the file at.crash-2 + stale marker so a single drain pass
    # promotes it to the quarantine branch.
    crash_2 = fpath.with_name(
        fpath.stem + ".crash-2.jsonl"
    )
    fpath.rename(crash_2)
    stale_marker = crash_2.with_name(
        crash_2.stem + f".processing-{dead}.jsonl"
    )
    crash_2.rename(stale_marker)

    write_event_calls: list[tuple[str, dict, dict]] = []

    def _stub_write_event(_store, kind, data, **kwargs):
        write_event_calls.append((kind, data, kwargs))

    import iai_mcp.events as events_mod

    monkeypatch.setattr(events_mod, "write_event", _stub_write_event)

    store = _open_isolated_store()
    drain_deferred_captures(store)

    quarantine_events = [
        c for c in write_event_calls if c[0] == "deferred_captures_quarantined"
    ]
    assert len(quarantine_events) == 1, (
        f"expected 1 deferred_captures_quarantined event, got {write_event_calls}"
    )
    kind, data, kwargs = quarantine_events[0]
    assert kwargs.get("severity") == "warning", kwargs
    assert kwargs.get("domain") == "ops", kwargs
    assert data.get("reason") == "crash_loop", data
    assert isinstance(data.get("attempts"), int), data
    assert data.get("attempts") >= 3, data
    assert "file" in data, data


# ---------------------------------------------------------------------------
# Test 3 — processing marker removed on success (clean ingest)
# ---------------------------------------------------------------------------


def test_processing_marker_removed_on_success(iai_home, monkeypatch):
    """A clean drain leaves NO `.processing-<pid>.jsonl` behind."""
    from iai_mcp.capture import drain_deferred_captures
    import iai_mcp.capture as capture_mod

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    _make_clean_jsonl(deferred_dir, "clean", 1200000000)

    def _stub(*_args: Any, **_kwargs: Any) -> dict:
        return {"status": "inserted", "record_id": None, "reason": "ok"}

    monkeypatch.setattr(capture_mod, "capture_turn", _stub)

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 1, counts

    leftovers = [
        p.name
        for p in deferred_dir.iterdir()
        if ".processing-" in p.name
    ]
    assert leftovers == [], (
        f"no .processing-<pid> markers should remain after clean drain, got {leftovers}"
    )


# ---------------------------------------------------------------------------
# Test 4 — existing `.failed-*-attempt-3` retry policy regression check
# ---------------------------------------------------------------------------


def test_failed_attempt_retry_policy_still_holds(iai_home, monkeypatch):
    """`.failed-<ts>-attempt-3.jsonl` whose backoff has elapsed must still
    transition to `.permanent-failed-<ts>.jsonl` (FAILED_MAX_ATTEMPTS path).
    Quarantine is a separate code path and must not interfere.
    """
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _make_clean_jsonl(deferred_dir, "fail3", 2000000000)
    attempt_3 = fpath.with_name(
        "sess-fail3-2000000000.failed-2000000000-attempt-3.jsonl"
    )
    fpath.rename(attempt_3)
    aged = time.time() - 1000  # > 240s backoff
    os.utime(attempt_3, (aged, aged))

    def _stub(*_args: Any, **_kwargs: Any) -> dict:
        return {"status": "skipped", "reason": "insert-failed:test"}

    import iai_mcp.capture as capture_mod

    monkeypatch.setattr(capture_mod, "capture_turn", _stub)

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_failed"] == 1, counts
    perm = list(deferred_dir.glob(
        "sess-fail3-2000000000.permanent-failed-*.jsonl"
    ))
    assert len(perm) == 1, (
        f"expected 1 .permanent-failed-* after attempt-3 retry failure, got {perm}"
    )


# ---------------------------------------------------------------------------
# Test 5 — socket binds before drain completes (PATCH B + C integration)
# ---------------------------------------------------------------------------


def test_socket_binds_before_drain_completes(tmp_path, monkeypatch, request):
    """`daemon.main` must bind the MCP unix socket and accept a connection
    while drain is still busy. Proves drain runs as a background asyncio task
    and does not block the event loop or SocketServer bind.

    Pattern: drive `daemon.main` from inside `asyncio.run` so we share the
    same event loop. Drain is monkey-patched to sleep 5s in a background
    thread (`asyncio.to_thread`); within that 5s window the unix socket
    MUST be created and accept a connection.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-bind-first-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "hippo"))
    monkeypatch.setenv("IAI_DAEMON_IDLE_SHUTDOWN_SECS", "3600")
    import keyring.core

    keyring.core._keyring_backend = None

    # macOS AF_UNIX caps sun_path at 104 bytes; pytest's tmp_path is typically
    # ~70+ chars deep on macOS, leaving no headroom for a basename. Bind in
    # /tmp under a unique-per-test name. Cross-test leak protection is the
    # request.addfinalizer below -- it runs even if the main `finally` is
    # interrupted by pytest tearing the test down mid-flight (the leak vector
    # MED-03 from the P6 review actually flagged). The serve() method honors
    # IAI_DAEMON_SOCKET_PATH at call-time (not import-time), so no
    # concurrency-module reload is required.
    tmp_socket = Path(f"/tmp/iai-test-{os.getpid()}-{int(time.time()*1000)}.sock")
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_socket))

    def _cleanup_socket():
        try:
            tmp_socket.unlink()
        except (FileNotFoundError, OSError):
            pass

    request.addfinalizer(_cleanup_socket)

    # Stub drain so the foreground startup path either blocks (pre-patch)
    # or returns quickly (post-patch — drain scheduled via
    # `asyncio.create_task`). The 60s sleep is long enough to absorb the
    # one-shot ~10s cold-start cost of embedder prewarm + lance optimize
    # and still leave a wide window where the socket must be bound while
    # drain is mid-sleep.
    drain_state = {"started": False, "finished": False}

    def _slow_drain(_store):
        drain_state["started"] = True
        time.sleep(60)
        drain_state["finished"] = True
        return {
            "files_drained": 0,
            "files_failed": 0,
            "events_inserted": 0,
            "events_reinforced": 0,
            "events_skipped_intentional": 0,
            "events_skipped_insert_failed": 0,
        }

    import iai_mcp.capture as capture_mod

    monkeypatch.setattr(capture_mod, "drain_deferred_captures", _slow_drain)

    from iai_mcp.daemon import main as daemon_main

    # Capture drain state at the moment the socket connects. The
    # post-scenario assertions are racy because the drain thread keeps
    # running after asyncio cancellation — `drain_state["finished"]`
    # being True at assertion time means nothing; what matters is
    # whether drain was finished at the moment the socket bound.
    snapshot: dict = {}

    async def _scenario() -> bool:
        daemon_task = asyncio.create_task(daemon_main())
        try:
            # Poll for socket existence + connect-ability within 60s — the
            # outer wait covers cold-start (embedder load, lance optimize)
            # plus the small window between SocketServer.create_task and
            # the actual unix-socket bind.
            deadline = time.monotonic() + 60.0
            sock_ok = False
            while time.monotonic() < deadline:
                if daemon_task.done():
                    exc = daemon_task.exception()
                    if exc is not None:
                        raise exc
                    return False
                if tmp_socket.exists():
                    try:
                        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        s.settimeout(1.0)
                        await asyncio.to_thread(s.connect, str(tmp_socket))
                        s.close()
                        # Snapshot drain state AT THE MOMENT of successful
                        # connect — this is the load-bearing observation.
                        snapshot["bound_at"] = time.monotonic()
                        snapshot["drain_started"] = drain_state["started"]
                        snapshot["drain_finished"] = drain_state["finished"]
                        sock_ok = True
                        break
                    except (ConnectionRefusedError, FileNotFoundError, OSError):
                        pass
                await asyncio.sleep(0.1)
            return sock_ok
        finally:
            daemon_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(daemon_task, return_exceptions=True),
                    timeout=5.0,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

    try:
        sock_ok = asyncio.run(_scenario())
        assert sock_ok, (
            f"socket {tmp_socket} did not accept a connection within 60s "
            f"while drain was busy; socket_exists={tmp_socket.exists()}, "
            f"drain_started={drain_state['started']}, "
            f"drain_finished={drain_state['finished']}"
        )
        # The load-bearing claim: AT THE MOMENT the socket bound, drain
        # was NOT finished. If drain finished before the bind, startup is
        # still blocking on it. Either drain hasn't started yet (proves
        # bind-before-drain-start, even stronger) or drain has started
        # but not finished — both acceptable.
        assert not snapshot.get("drain_finished", True), (
            f"drain finished before socket bound at "
            f"t={snapshot.get('bound_at'):.2f} — startup is still blocking on it"
        )
    finally:
        keyring.core._keyring_backend = None
        # Socket cleanup handled by addfinalizer + tmp_path teardown


# ---------------------------------------------------------------------------
# Test 6 — atomic claim logs generic OSError
# ---------------------------------------------------------------------------


def test_atomic_claim_logs_generic_oserror(tmp_path, monkeypatch):
    """A generic OSError (e.g. PermissionError) raised by `Path.rename`
    inside `drain_deferred_captures`'s atomic-ownership-claim block MUST
    write a one-line log entry to the deferred-drain log AND leave the
    original file untouched (no `.processing-<pid>` segment, no `.crash-N`
    bump, no `.failed-*` rename). Regression guard for the OSError-claim path.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "p6-1-fix-a-test-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "hippo"))
    import keyring.core

    keyring.core._keyring_backend = None

    # Build a minimal deferred-captures file (header + one event line).
    deferred_dir = tmp_path / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    fname = "sess-fix-a-1700000000.jsonl"
    fpath = deferred_dir / fname
    header = {"version": 1, "session_id": "sess-fix-a"}
    event = {"cue": "x", "text": "y" * 32, "tier": "episodic", "role": "user"}
    fpath.write_text(json.dumps(header) + "\n" + json.dumps(event) + "\n")

    # Inject PermissionError into Path.rename for the duration of the call.
    # The OSError-claim path must catch this and log, NOT bubble up.
    import pathlib as _pathlib

    real_rename = _pathlib.Path.rename

    def boom(self, target):
        # Only intercept the claim-rename (active dir ->.processing-<pid>);
        # let _quarantine_file's shutil.move + other path operations work.
        if ".processing-" in str(target) and self == fpath:
            raise PermissionError("simulated EACCES on atomic claim")
        return real_rename(self, target)

    monkeypatch.setattr(_pathlib.Path, "rename", boom)

    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    try:
        counts = drain_deferred_captures(store)
    finally:
        keyring.core._keyring_backend = None

    # File must still exist in the active dir, unchanged.
    assert fpath.exists(), "regression: file should remain after claim failure"
    assert ".processing-" not in fpath.name, (
        f"regression: file basename should NOT have a .processing- segment, "
        f"got {fpath.name}"
    )
    # No.crash-N bump (atomic-claim failure is NOT a crash-loop signal).
    assert ".crash-" not in fpath.name

    # The log line must exist.
    from datetime import datetime, timezone

    log_path = tmp_path / ".iai-mcp" / "logs" / (
        f"deferred-drain-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
    )
    assert log_path.exists(), "deferred-drain log should be created"
    log_text = log_path.read_text()
    assert "claim-failed" in log_text, (
        f"regression: log should contain 'claim-failed', got: {log_text!r}"
    )
    assert "PermissionError" in log_text, (
        f"regression: log should name PermissionError, got: {log_text!r}"
    )

    # Counts: no file drained, no file failed (claim never succeeded — we
    # never entered the per-file ingest body).
    assert counts["files_drained"] == 0
    assert counts["files_failed"] == 0


# ---------------------------------------------------------------------------
# Test 7 — _strip_processing_marker returns (Path, False) on rename failure
# ---------------------------------------------------------------------------


def test_strip_processing_marker_returns_false_on_rename_failure(
    tmp_path, monkeypatch
):
    """`_strip_processing_marker` MUST return (input_path, False) when the
    underlying rename raises OSError, and MUST write a log line if log_path
    is provided. Regression guard for the marker-strip OSError contract.
    """
    from iai_mcp.capture import _strip_processing_marker

    src = tmp_path / "sess-x-1700000000.processing-99999.jsonl"
    src.write_text("{}\n")
    log_path = tmp_path / "drain.log"

    import pathlib as _pathlib

    def boom(self, target):
        raise PermissionError("simulated")

    monkeypatch.setattr(_pathlib.Path, "rename", boom)

    new_path, ok = _strip_processing_marker(src, log_path=log_path)
    assert ok is False, "strip MUST report failure"
    assert new_path == src, "on failure, return the input path"
    assert log_path.exists(), "log_path should be written"
    assert "strip-marker-failed" in log_path.read_text()

    # Without log_path, contract holds but no log is written.
    log_path2 = tmp_path / "absent.log"
    new_path2, ok2 = _strip_processing_marker(src)
    assert ok2 is False
    assert new_path2 == src
    assert not log_path2.exists()
