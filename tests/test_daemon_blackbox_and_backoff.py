"""Hermetic tests for two reliability features in the daemon:

1. Pre-kill forensic black box: on a failed probe tick (below the kill threshold)
   the watchdog captures a lock-free forensic dump exactly ONCE per failure
   episode. It never calls _self_kill, never changes the tick decision, and never
   touches the store connection lock.

2. Bounded restart lock-backoff at boot: when the EXCLUSIVE store open raises
   HippoLockHeldError the boot helper retries with bounded backoff rather than
   failing immediately. If the lock frees within the bound, the open succeeds.
   If the lock never frees, the helper re-raises and does NOT hang indefinitely.

Tests are hermetic: tmp paths, injected fixtures, mocked os.kill (the test
process is never killed). No real ~/.iai-mcp/ path is touched. No live daemon
is started or stopped.
"""
from __future__ import annotations

import asyncio
import os
import signal
import time

import pytest

from iai_mcp import daemon
from iai_mcp.hippo import HippoLockHeldError


# ---------------------------------------------------------------------------
# Shared constants (mirrored from test_daemon_watchdog.py for consistency)
# ---------------------------------------------------------------------------
DEBOUNCE_N = 3
GRACE = 600.0
NORMAL = 1
WARN = 2
RSS_LOW = 300 * 1024 * 1024  # 300 MB — never a kill trigger


# ---------------------------------------------------------------------------
# Shared harness: hermetic watchdog tick environment
# ---------------------------------------------------------------------------

@pytest.fixture
def watchdog_env(tmp_path, monkeypatch):
    """Hermetic watchdog harness (breadcrumb fd + no-kill os.kill mock).

    Returns a namespace with:
      - log_path:    tmp breadcrumb log (opened as _WATCHDOG_LOG_FD)
      - sock_path:   tmp socket path (probe is always injected — no real bind)
      - kill_calls:  list accumulating (pid, sig) from the mocked os.kill
      - bb_log_path: expected black-box dump log path
    Sets the daemon past cold-start grace by default.
    """
    log_path = tmp_path / ".daemon-watchdog.log"
    bb_log_path = tmp_path / ".daemon-blackbox.log"
    sock_path = str(tmp_path / ".daemon.sock")

    # Open the breadcrumb fd exactly as _liveness_watchdog does.
    fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", fd)

    # Reset the episode-tracking flag if present so tests don't bleed.
    if hasattr(daemon, "_WATCHDOG_BLACKBOX_EPISODE_FIRED"):
        monkeypatch.setattr(daemon, "_WATCHDOG_BLACKBOX_EPISODE_FIRED", False)

    # Past cold-start grace so memory/leak triggers are live.
    monkeypatch.setattr(
        daemon, "_daemon_started_monotonic", time.monotonic() - (GRACE + 60.0)
    )

    kill_calls: list[tuple[int, int]] = []

    def _fake_kill(pid, sig):
        kill_calls.append((pid, sig))
        # Do NOT actually signal — the test process must survive.

    monkeypatch.setattr(daemon.os, "kill", _fake_kill)

    class _Ns:
        pass

    ns = _Ns()
    ns.log_path = log_path
    ns.bb_log_path = bb_log_path
    ns.sock_path = sock_path
    ns.kill_calls = kill_calls
    ns.fd = fd
    yield ns
    try:
        os.close(fd)
    except OSError:
        pass


def _probe(result: bool):
    """Build an async probe fn that always returns the fixed result."""
    async def _p(_sock, _timeout):
        return result
    return _p


def _tick(
    env,
    consec: int,
    probe_result: bool,
    *,
    blackbox_fn=None,
) -> tuple[float, int]:
    """Run one _watchdog_tick through the hermetic harness."""
    return daemon._watchdog_tick(
        object(),       # store — never touched on the non-kill path
        env.sock_path,
        env.log_path,
        consec,
        probe_fn=_probe(probe_result),
        pressure_fn=lambda: NORMAL,
        rss_fn=lambda: RSS_LOW,
        blackbox_fn=blackbox_fn,
    )


# ===========================================================================
# Pre-kill forensic black box
# ===========================================================================


class TestBlackboxOnFailedProbe:
    """The black box fires exactly once per failure episode on a non-kill
    failing tick, never on clean ticks, and never when the kill fires."""

    def test_blackbox_called_once_on_first_failing_tick(self, watchdog_env):
        """On the very first failing tick (consec==0 -> consec==1 after tick),
        the blackbox_fn is called exactly once."""
        calls: list[int] = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append(consec)

        _interval, consec = _tick(watchdog_env, 0, False, blackbox_fn=_bb)
        assert consec == 1  # debounce counter advanced
        assert len(calls) == 1, "black box must fire on the first failing tick"

    def test_blackbox_not_called_on_subsequent_failing_ticks_same_episode(
        self, watchdog_env
    ):
        """Once per episode: subsequent failing ticks of the same episode do
        NOT re-fire the black box.  Episode ends only when a clean tick resets
        the counter."""
        calls: list[int] = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append(consec)

        consec = 0
        # Drive DEBOUNCE_N - 1 failing ticks (stays below kill threshold).
        for _ in range(DEBOUNCE_N - 1):
            _interval, consec = _tick(watchdog_env, consec, False, blackbox_fn=_bb)

        # The black box should have fired exactly once.
        assert len(calls) == 1, (
            "black box must fire exactly once per failure episode, "
            f"but was called {len(calls)} times"
        )
        assert watchdog_env.kill_calls == [], "no kill below debounce threshold"

    def test_blackbox_not_called_on_clean_tick(self, watchdog_env):
        """A clean tick (probe_ok) NEVER invokes the black box."""
        calls: list[int] = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append(consec)

        _interval, consec = _tick(watchdog_env, 0, True, blackbox_fn=_bb)
        assert consec == 0  # clean tick resets counter
        assert calls == [], "black box must NOT fire on a clean probe tick"

    def test_episode_flag_resets_after_clean_tick(self, watchdog_env):
        """After a clean tick the episode resets, so the next failure episode
        fires the black box again."""
        calls: list[int] = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append(consec)

        # Episode 1: one failing tick.
        _interval, consec = _tick(watchdog_env, 0, False, blackbox_fn=_bb)
        assert len(calls) == 1

        # Clean tick — episode resets.
        _interval, consec = _tick(watchdog_env, consec, True, blackbox_fn=_bb)
        assert consec == 0
        assert len(calls) == 1, "no extra call on clean tick"

        # Episode 2: another failing tick — black box should fire again.
        _interval, consec = _tick(watchdog_env, consec, False, blackbox_fn=_bb)
        assert len(calls) == 2, "black box must re-arm after a clean tick"

    def test_blackbox_never_called_on_kill_tick(self, watchdog_env):
        """At DEBOUNCE_N failures (the kill tick), the black box must NOT be
        called — it must only fire below the kill threshold."""
        calls: list[int] = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append(consec)

        consec = 0
        # Drive to DEBOUNCE_N failures total.
        for _ in range(DEBOUNCE_N):
            _interval, consec = _tick(watchdog_env, consec, False, blackbox_fn=_bb)

        # Kill should have fired.
        assert watchdog_env.kill_calls, "kill must fire at debounce_n failures"
        # The black box fired exactly once (on the first failing tick, not on
        # the kill tick).
        assert len(calls) == 1, (
            "black box must fire exactly once (not on the kill tick)"
        )

    def test_self_kill_not_called_on_non_kill_failing_tick(self, watchdog_env):
        """Below the kill threshold, os.kill is NEVER invoked — the black box
        is logs-only and does not alter the kill decision."""
        consec = 0
        for _ in range(DEBOUNCE_N - 1):
            _interval, consec = _tick(watchdog_env, consec, False, blackbox_fn=None)

        assert watchdog_env.kill_calls == [], "_self_kill must not fire below debounce"

    def test_returned_tuple_unchanged_with_blackbox(self, watchdog_env):
        """The (next_interval, consecutive_failures) returned by _watchdog_tick
        must be identical whether blackbox_fn is provided or not."""
        # Without black box.
        _iv1, _c1 = _tick(watchdog_env, 0, False, blackbox_fn=None)

        # Rearm the episode flag so a second call also fires once.
        if hasattr(daemon, "_WATCHDOG_BLACKBOX_EPISODE_FIRED"):
            import importlib
            # Reset by patching directly — we just need the episode flag clear.
            daemon._WATCHDOG_BLACKBOX_EPISODE_FIRED = False

        calls: list = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append(True)

        _iv2, _c2 = _tick(watchdog_env, 0, False, blackbox_fn=_bb)

        assert _iv1 == _iv2, "interval must not change due to black box"
        assert _c1 == _c2, "consecutive_failures must not change due to black box"

    def test_blackbox_fn_receives_correct_args(self, watchdog_env):
        """The blackbox_fn receives the expected arguments (log_fd, probe_ok,
        consec_at_call_time, debounce_n)."""
        received: list[tuple] = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            received.append((log_fd, probe_ok, consec, debounce_n))

        _interval, _consec = _tick(watchdog_env, 0, False, blackbox_fn=_bb)

        assert len(received) == 1
        _fd, _probe_ok, _consec_val, _dbn = received[0]
        assert _probe_ok is False
        assert _dbn == daemon.WATCHDOG_FAILURE_DEBOUNCE_N
        # fd: could be the real blackbox fd (int) or None if not yet opened
        # (when the function is injected, the fd arg is passed as configured).

    def test_no_store_lock_taken_in_blackbox(self, watchdog_env):
        """The black box must be entirely lock-free. We verify this by injecting
        a blackbox_fn and confirming it fires without the test needing any store
        lock — no MemoryStore opened, no hippo._conn_lock taken."""
        # If blackbox_fn is injected, it replaces any real dump, so no actual
        # faulthandler / os.write occurs. The test just confirms the wiring
        # accepts a fully injected fn and fires it without any store dep.
        calls: list = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append("fired")

        _interval, consec = _tick(watchdog_env, 0, False, blackbox_fn=_bb)
        assert calls == ["fired"], "injected blackbox_fn must be called"


def test_capture_blackbox_writes_real_traceback(tmp_path):
    """Smoke-test the real _capture_blackbox path: a pre-opened int fd must
    receive a structured header, a real faulthandler all-thread traceback,
    and the trailing separator. This confirms faulthandler.dump_traceback
    accepts an integer fd and that the load-bearing forensic payload lands."""
    log_path = tmp_path / "bb.log"
    fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        daemon._capture_blackbox(fd, False, 1, daemon.WATCHDOG_FAILURE_DEBOUNCE_N)
    finally:
        os.close(fd)

    content = log_path.read_text(encoding="utf-8", errors="replace")
    assert "pre_kill_forensic_dump" in content, "header must be present"
    # faulthandler all-threads dump: on CPython the output contains "Thread"
    # or "File" (from the stack frame repr). Accept either.
    assert ('File "' in content) or ("Thread" in content) or ("Stack" in content), (
        "faulthandler all-thread traceback must be present in the dump"
    )
    assert "--- end dump ---" in content, "trailing separator must be present"
    # task_names from the watchdog thread: asyncio.all_tasks() raises RuntimeError
    # in a non-loop thread, so the field is reliably empty (best-effort only).
    assert "tasks=" in content, "tasks field must appear in the header"


# ===========================================================================
# Bounded restart lock-backoff on HippoLockHeldError
# ===========================================================================


class TestBootLockBackoff:
    """The boot helper retries the EXCLUSIVE store open on HippoLockHeldError
    (a transient race with a dying predecessor). It succeeds when the lock frees
    within the bound and re-raises loud when it does not."""

    def test_backoff_retries_and_succeeds_when_lock_frees(
        self, tmp_path
    ):
        """Constructor raises HippoLockHeldError for the first K calls, then
        succeeds. The boot helper must retry and return the store object."""
        sentinel = object()  # stand-in for a real MemoryStore
        calls: list[int] = []

        def _factory_first_k_fail(k: int):
            def _f():
                calls.append(len(calls) + 1)
                if len(calls) <= k:
                    raise HippoLockHeldError(
                        tmp_path / ".lock", f"attempt-{len(calls)}"
                    )
                return sentinel
            return _f

        # Fail first 2 calls, succeed on attempt 3.
        result = asyncio.run(daemon._open_exclusive_store_with_backoff(
            _factory_first_k_fail(2),
            max_attempts=5,
            backoff_sec=0.01,  # tiny for test speed
        ))
        assert result is sentinel, "helper must return the store on eventual success"
        assert len(calls) == 3, f"expected 3 attempts, got {len(calls)}"

    def test_backoff_exhausts_and_reraises_when_lock_never_frees(
        self, tmp_path
    ):
        """Constructor always raises HippoLockHeldError. After exhausting the
        bound the helper must re-raise — it must NOT hang indefinitely and must
        NOT swallow the error."""
        calls: list[int] = []
        lock_path = tmp_path / ".lock"

        def _always_fail():
            calls.append(len(calls) + 1)
            raise HippoLockHeldError(lock_path, "always-held")

        with pytest.raises(HippoLockHeldError):
            asyncio.run(daemon._open_exclusive_store_with_backoff(
                _always_fail,
                max_attempts=3,
                backoff_sec=0.01,
            ))
        assert len(calls) == 3, (
            f"helper must attempt exactly max_attempts times, got {len(calls)}"
        )

    def test_backoff_succeeds_immediately_no_lock_error(self, tmp_path):
        """When the constructor succeeds on the first call (no HippoLockHeldError),
        the helper returns immediately without any retry delay."""
        sentinel = object()
        calls: list[int] = []

        def _succeed():
            calls.append(1)
            return sentinel

        result = asyncio.run(daemon._open_exclusive_store_with_backoff(
            _succeed,
            max_attempts=5,
            backoff_sec=0.01,
        ))
        assert result is sentinel
        assert len(calls) == 1, "no retry when first attempt succeeds"

    def test_backoff_non_lock_error_propagates_immediately(self, tmp_path):
        """Non-HippoLockHeldError exceptions from the constructor must propagate
        immediately (no retry), preserving the original exception type."""
        class _UnrelatedError(RuntimeError):
            pass

        calls: list[int] = []

        def _wrong_error():
            calls.append(1)
            raise _UnrelatedError("some other boot failure")

        with pytest.raises(_UnrelatedError):
            asyncio.run(daemon._open_exclusive_store_with_backoff(
                _wrong_error,
                max_attempts=5,
                backoff_sec=0.01,
            ))
        assert len(calls) == 1, "must NOT retry on non-HippoLockHeldError"

    def test_backoff_single_owner_guarantee(self, tmp_path):
        """After the bound is exhausted the re-raise preserves the original
        HippoLockHeldError (not a swallowed None), confirming the single-owner
        guarantee: if the predecessor never releases, boot still surfaces loud."""
        lock_path = tmp_path / ".lock"

        def _always_fail():
            raise HippoLockHeldError(lock_path, "unknown")

        exc: HippoLockHeldError | None = None
        try:
            asyncio.run(daemon._open_exclusive_store_with_backoff(
                _always_fail,
                max_attempts=2,
                backoff_sec=0.01,
            ))
        except HippoLockHeldError as e:
            exc = e

        assert exc is not None, "HippoLockHeldError must surface on exhaustion"
        # The exception must carry the lock path context (not be a generic error).
        assert "lock" in str(exc).lower() or str(lock_path) in str(exc)
