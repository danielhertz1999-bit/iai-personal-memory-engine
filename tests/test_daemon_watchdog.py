"""Hermetic decision-table + thread-level tests for the daemon self-watchdog.

The watchdog detects a WEDGED daemon (active status round-trip fails N
consecutive times) AND approaching-jetsam memory pressure, then performs a
controlled self-recovery: a LOCK-FREE breadcrumb followed by an UNCONDITIONAL
self-SIGKILL -> launchd respawn. CPU is never a kill signal.

Two layers:
  - Pure-function tests on ``_evaluate_watchdog`` + ``_next_poll_interval`` — no
    threads, no sleeps, no real kills.
  - Synchronous "thread-level" tests that drive ``_watchdog_tick`` directly (no
    real thread spun, no real process killed: ``os.kill`` is monkeypatched), plus
    a real connect-no-reply probe test and a real lock-independence test where the
    breadcrumb fd is invalid (EBADF) yet the SIGKILL still fires for BOTH paths.

Every test uses tmp paths; the live daemon and ``~/.iai-mcp`` are never touched.
Generic ``alice``/tmp data only.
"""
from __future__ import annotations

import asyncio
import os
import signal
import socket
import threading
import time

import pytest

from iai_mcp import daemon

# Threshold constants the watchdog consumes (measured-peak-derived).
HARD_CAP = 2_684_354_560  # 2.5 GiB
FLOOR = 1_610_612_736  # 1.5 GiB
DEBOUNCE_N = 3
GRACE = 600.0
MAX_RECOVERIES = 3
WINDOW = 600.0

# RSS sentinels well clear of the bounds.
RSS_LOW = 300 * 1024 * 1024  # 300 MB — steady-state, never a contributor
RSS_BIG = 2 * 1024 * 1024 * 1024  # 2.0 GiB — > floor, < cap (a real contributor)
RSS_LEAK = 3 * 1024 * 1024 * 1024  # 3.0 GiB — > cap (a leak)

# Pressure levels: 1=NORMAL, 2=WARN, 4=CRITICAL.
NORMAL = 1
WARN = 2
CRITICAL = 4


def _evaluate(
    probe_ok=True,
    rss=RSS_LOW,
    pressure=NORMAL,
    uptime=GRACE + 1.0,  # default past cold-start grace
    consecutive=0,
    recoveries=None,
    now_wall=1_000_000.0,
):
    """Thin wrapper so each test only sets the dimensions it cares about."""
    return daemon._evaluate_watchdog(
        probe_ok,
        rss,
        pressure,
        uptime,
        consecutive,
        list(recoveries or []),
        now_wall,
        hard_cap=HARD_CAP,
        contributor_floor=FLOOR,
        debounce_n=DEBOUNCE_N,
        cold_start_grace_sec=GRACE,
        max_recoveries=MAX_RECOVERIES,
        recovery_window_sec=WINDOW,
    )


# ---------------------------------------------------------------------------
# Pure decision table: _evaluate_watchdog
# ---------------------------------------------------------------------------
def test_healthy_idle_no_kill():
    """probe_ok, low RSS, NORMAL pressure -> none."""
    assert _evaluate() == ("none", "healthy")


def test_busy_healthy_no_kill_cpu_never_a_signal():
    """A busy/high-CPU but responsive daemon is never killed: the function has no
    CPU input at all, and a fast status reply (probe_ok) + safe memory -> none."""
    assert _evaluate(probe_ok=True, rss=RSS_LOW, pressure=NORMAL) == (
        "none",
        "healthy",
    )


def test_single_wedge_blip_does_not_kill_debounce():
    """probe_ok=False for 1 tick (< N) -> none (a single transient miss)."""
    assert _evaluate(probe_ok=False, consecutive=1) == ("none", "debounce")
    assert _evaluate(probe_ok=False, consecutive=DEBOUNCE_N - 1) == (
        "none",
        "debounce",
    )


def test_wedge_after_n_consecutive_failures_kills():
    """probe_ok=False sustained for N ticks -> kill, reason=wedge."""
    assert _evaluate(probe_ok=False, consecutive=DEBOUNCE_N) == ("kill", "wedge")


def test_wedge_not_grace_covered():
    """A wedge kills even within cold-start grace — a daemon that cannot serve a
    status round-trip N times is wedged regardless of age."""
    assert _evaluate(probe_ok=False, consecutive=DEBOUNCE_N, uptime=1.0) == (
        "kill",
        "wedge",
    )


def test_leak_kills_even_at_normal_pressure():
    """rss > HARD_CAP is always a leak -> kill, even at NORMAL pressure (after
    debounce)."""
    assert _evaluate(rss=RSS_LEAK, pressure=NORMAL, consecutive=DEBOUNCE_N) == (
        "kill",
        "leak",
    )


def test_leak_single_tick_does_not_kill():
    """A leak still respects the debounce — one tick over the cap is not a kill."""
    assert _evaluate(rss=RSS_LEAK, pressure=NORMAL, consecutive=1) == (
        "none",
        "debounce",
    )


def test_warn_plus_big_kills_memory():
    """sustained WARN AND rss > CONTRIBUTOR_FLOOR -> kill, reason=memory."""
    assert _evaluate(rss=RSS_BIG, pressure=WARN, consecutive=DEBOUNCE_N) == (
        "kill",
        "memory",
    )


def test_critical_plus_big_also_kills_memory():
    """CRITICAL (>= WARN) AND big -> kill, reason=memory."""
    assert _evaluate(rss=RSS_BIG, pressure=CRITICAL, consecutive=DEBOUNCE_N) == (
        "kill",
        "memory",
    )


def test_warn_but_another_process_owns_ram_no_kill():
    """WARN pressure but rss < CONTRIBUTOR_FLOOR (another process owns the RAM)
    -> none. Killing a small daemon frees nothing and re-WARNs -> kill loop."""
    assert _evaluate(rss=RSS_LOW, pressure=WARN, consecutive=DEBOUNCE_N) == (
        "none",
        "healthy",
    )


def test_unreadable_pressure_does_not_kill():
    """pressure_level None (unreadable) AND rss < HARD_CAP -> none (fail-open:
    never kill on an unreadable signal). The RSS-leak backstop still guards."""
    assert _evaluate(rss=RSS_BIG, pressure=None, consecutive=DEBOUNCE_N) == (
        "none",
        "healthy",
    )


def test_unreadable_pressure_leak_backstop_still_fires():
    """Even with an unreadable pressure signal, rss > HARD_CAP is a leak -> kill."""
    assert _evaluate(rss=RSS_LEAK, pressure=None, consecutive=DEBOUNCE_N) == (
        "kill",
        "leak",
    )


def test_cold_start_grace_suppresses_memory_trigger():
    """Within cold-start grace (uptime < 600s), a WARN+big condition is
    suppressed (the boot RSS ramp is legitimate)."""
    assert _evaluate(
        rss=RSS_BIG, pressure=WARN, uptime=10.0, consecutive=DEBOUNCE_N
    ) == ("none", "healthy")


def test_cold_start_grace_suppresses_leak_trigger():
    """The leak backstop is also suppressed during cold-start grace."""
    assert _evaluate(
        rss=RSS_LEAK, pressure=NORMAL, uptime=10.0, consecutive=DEBOUNCE_N
    ) == ("none", "healthy")


def test_circuit_breaker_trips_to_needs_operator_not_kill():
    """>= K recoveries inside the window -> needs_operator (STOP killing)."""
    now = 1_000_000.0
    recent = [now - 10, now - 20, now - 30]  # 3 == MAX_RECOVERIES, all in-window
    assert _evaluate(
        probe_ok=False, consecutive=DEBOUNCE_N, recoveries=recent, now_wall=now
    ) == ("needs_operator", "circuit_breaker")


def test_circuit_breaker_ignores_out_of_window_recoveries():
    """Recoveries older than the window do not count toward the breaker — a
    daemon that wedged once last hour and again now is still allowed to recover."""
    now = 1_000_000.0
    old = [now - WINDOW - 100, now - WINDOW - 200, now - WINDOW - 300]
    assert _evaluate(
        probe_ok=False, consecutive=DEBOUNCE_N, recoveries=old, now_wall=now
    ) == ("kill", "wedge")


def test_circuit_breaker_below_threshold_still_kills():
    """K-1 in-window recoveries: still allowed to kill (breaker not yet tripped)."""
    now = 1_000_000.0
    recent = [now - 10, now - 20]  # 2 < MAX_RECOVERIES
    assert _evaluate(
        probe_ok=False, consecutive=DEBOUNCE_N, recoveries=recent, now_wall=now
    ) == ("kill", "wedge")


# ---------------------------------------------------------------------------
# Pure adaptive cadence: _next_poll_interval
# ---------------------------------------------------------------------------
def test_poll_interval_normal_is_steady():
    assert daemon._next_poll_interval(NORMAL) == daemon.WATCHDOG_LIVENESS_POLL_SEC


def test_poll_interval_none_is_steady():
    assert daemon._next_poll_interval(None) == daemon.WATCHDOG_LIVENESS_POLL_SEC


def test_poll_interval_warn_is_tightened():
    assert daemon._next_poll_interval(WARN) == daemon.WATCHDOG_WARN_POLL_SEC


def test_poll_interval_critical_is_tightened():
    assert daemon._next_poll_interval(CRITICAL) == daemon.WATCHDOG_WARN_POLL_SEC


def test_warn_poll_is_strictly_tighter_than_steady():
    """The tightened cadence must actually be faster, or the debounce window
    is not shortened under pressure."""
    assert daemon.WATCHDOG_WARN_POLL_SEC < daemon.WATCHDOG_LIVENESS_POLL_SEC


# ---------------------------------------------------------------------------
# Thread-level (synchronous via _watchdog_tick — os.kill mocked, real process
# NEVER killed; tmp socket + tmp breadcrumb log; live daemon never touched).
# ---------------------------------------------------------------------------
@pytest.fixture
def watchdog_env(tmp_path, monkeypatch):
    """Hermetic watchdog harness.

    Returns a namespace with:
      - log_path: a tmp breadcrumb log (also opened as _WATCHDOG_LOG_FD)
      - sock_path: a tmp socket path (probe is injected, so no real bind needed)
      - kill_calls: list capturing (pid, sig) from the mocked os.kill
    Past cold-start grace by default. The real os.kill is replaced so the test
    process is NEVER killed.
    """
    log_path = tmp_path / ".daemon-watchdog.log"
    sock_path = str(tmp_path / ".daemon.sock")

    # Open the breadcrumb fd exactly as the thread does.
    fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", fd)

    # Past cold-start grace so memory/leak triggers are live by default.
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
    ns.sock_path = sock_path
    ns.kill_calls = kill_calls
    ns.fd = fd
    yield ns
    try:
        os.close(fd)
    except OSError:
        pass


def _probe(result: bool):
    """Build an async probe fn that ignores args and returns a fixed result."""

    async def _p(_sock, _timeout):
        return result

    return _p


def _read_breadcrumb(log_path):
    return log_path.read_text(encoding="utf-8")


# --- the wedge path -------------------------------------------------------
def test_thread_wedge_after_n_consecutive_kills(watchdog_env):
    """A probe that fails N consecutive ticks -> self-kill via the lock-free
    helper: a wedge breadcrumb is written AND os.kill(getpid, SIGKILL) fires."""
    store = object()  # never touched on the kill path
    consec = 0
    # First N-1 ticks: debounce, no kill.
    for _ in range(DEBOUNCE_N - 1):
        _interval, consec = daemon._watchdog_tick(
            store,
            watchdog_env.sock_path,
            watchdog_env.log_path,
            consec,
            probe_fn=_probe(False),
            pressure_fn=lambda: NORMAL,
            rss_fn=lambda: RSS_LOW,
        )
    assert watchdog_env.kill_calls == []
    # Nth tick: kill.
    _interval, consec = daemon._watchdog_tick(
        store,
        watchdog_env.sock_path,
        watchdog_env.log_path,
        consec,
        probe_fn=_probe(False),
        pressure_fn=lambda: NORMAL,
        rss_fn=lambda: RSS_LOW,
    )
    assert watchdog_env.kill_calls == [(os.getpid(), signal.SIGKILL)]
    crumb = _read_breadcrumb(watchdog_env.log_path)
    assert daemon.DAEMON_WEDGE_KILL in crumb
    assert "reason=wedge" in crumb


def test_thread_healthy_busy_not_killed(watchdog_env):
    """A responsive (probe_ok) daemon with safe memory is never killed, even
    across many ticks — CPU is never a kill signal (no CPU input at all)."""
    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N + 5):
        _interval, consec = daemon._watchdog_tick(
            store,
            watchdog_env.sock_path,
            watchdog_env.log_path,
            consec,
            probe_fn=_probe(True),
            pressure_fn=lambda: NORMAL,
            rss_fn=lambda: RSS_LOW,
        )
    assert watchdog_env.kill_calls == []
    assert consec == 0  # clean ticks reset the debounce counter


# --- the memory path ------------------------------------------------------
def test_thread_warn_plus_big_memory_kill(watchdog_env):
    """Sustained WARN + big RSS -> memory-kill via the SAME lock-free helper."""
    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N):
        _interval, consec = daemon._watchdog_tick(
            store,
            watchdog_env.sock_path,
            watchdog_env.log_path,
            consec,
            probe_fn=_probe(True),  # responsive — only memory triggers
            pressure_fn=lambda: WARN,
            rss_fn=lambda: RSS_BIG,
        )
    assert watchdog_env.kill_calls == [(os.getpid(), signal.SIGKILL)]
    crumb = _read_breadcrumb(watchdog_env.log_path)
    assert daemon.DAEMON_MEMORY_PRESSURE_KILL in crumb
    assert "reason=memory" in crumb


def test_thread_warn_another_process_owns_ram_no_kill(watchdog_env):
    """WARN pressure but LOW daemon RSS (another process owns the RAM) -> no
    kill (the contributor-floor gate), across many ticks."""
    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N + 5):
        _interval, consec = daemon._watchdog_tick(
            store,
            watchdog_env.sock_path,
            watchdog_env.log_path,
            consec,
            probe_fn=_probe(True),
            pressure_fn=lambda: WARN,
            rss_fn=lambda: RSS_LOW,
        )
    assert watchdog_env.kill_calls == []


def test_thread_adaptive_cadence_tightens_under_warn(watchdog_env):
    """Under sustained WARN the chosen next-sleep is the tightened poll, not the
    steady poll — so a fast RSS balloon keeps its debounce window."""
    store = object()
    interval_normal, _ = daemon._watchdog_tick(
        store,
        watchdog_env.sock_path,
        watchdog_env.log_path,
        0,
        probe_fn=_probe(True),
        pressure_fn=lambda: NORMAL,
        rss_fn=lambda: RSS_LOW,
    )
    interval_warn, _ = daemon._watchdog_tick(
        store,
        watchdog_env.sock_path,
        watchdog_env.log_path,
        0,
        probe_fn=_probe(True),
        pressure_fn=lambda: WARN,
        rss_fn=lambda: RSS_LOW,  # low RSS -> WARN does not kill, only tightens
    )
    assert interval_normal == daemon.WATCHDOG_LIVENESS_POLL_SEC
    assert interval_warn == daemon.WATCHDOG_WARN_POLL_SEC


# --- the circuit-breaker (cross-process, reconstructed from disk) ---------
def test_thread_circuit_breaker_emits_needs_operator_not_kill(
    watchdog_env, monkeypatch
):
    """K prior kill breadcrumbs in the window (as if SIGKILL->respawn happened K
    times) -> the watchdog STOPS killing and emits a loud needs-operator event.
    The recovery state is reconstructed from the on-disk breadcrumb (in-memory
    state is wiped on each SIGKILL), proving the breaker is cross-process."""
    # Pre-seed K recent kill lines on disk (as prior respawns would have left).
    now = time.time()
    lines = [
        f"{daemon.datetime.fromtimestamp(now - 10 * (i + 1), daemon.timezone.utc).isoformat()} "
        f"{daemon.DAEMON_WEDGE_KILL} reason=wedge pid=999\n"
        for i in range(MAX_RECOVERIES)
    ]
    watchdog_env.log_path.write_text("".join(lines), encoding="utf-8")
    # Re-open the fd (the seed truncated/rewrote the file).
    os.close(watchdog_env.fd)
    fd = os.open(
        str(watchdog_env.log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600
    )
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", fd)

    emitted: list[tuple] = []

    def _fake_write_event(store, kind, data, **kw):
        emitted.append((kind, data, kw))
        return "evt-id"

    monkeypatch.setattr(daemon, "write_event", _fake_write_event)

    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N):
        _interval, consec = daemon._watchdog_tick(
            store,
            watchdog_env.sock_path,
            watchdog_env.log_path,
            consec,
            probe_fn=_probe(False),  # wedge — but the breaker has tripped
            pressure_fn=lambda: NORMAL,
            rss_fn=lambda: RSS_LOW,
        )
    # NO kill (breaker tripped); a loud needs-operator event WAS emitted.
    assert watchdog_env.kill_calls == []
    assert any(k == daemon.DAEMON_WATCHDOG_NEEDS_OPERATOR for k, _d, _kw in emitted)
    try:
        os.close(fd)
    except OSError:
        pass


# --- THE LOCK-INDEPENDENCE TEST (the safety core) -------------------------
def test_self_kill_is_unconditional_when_breadcrumb_fails_wedge(
    tmp_path, monkeypatch
):
    """SAFETY CORE: even when the breadcrumb sink RAISES (an invalid fd ->
    EBADF, modelling the os.write blocking/failing under a held store lock),
    the SIGKILL STILL fires for the WEDGE path. The kill is never gated on the
    emit and never traverses a lock-taking path."""
    monkeypatch.setattr(
        daemon, "_daemon_started_monotonic", time.monotonic() - (GRACE + 60.0)
    )
    # An invalid fd: real os.write(fd, ...) raises EBADF inside _self_kill's
    # try/except (the TRUE lock-free path), without globally patching os.write.
    bad_fd = os.open(str(tmp_path / "tmp"), os.O_WRONLY | os.O_CREAT, 0o600)
    os.close(bad_fd)  # now closed -> writing raises EBADF
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", bad_fd)

    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        daemon.os, "kill", lambda pid, sig: kill_calls.append((pid, sig))
    )

    log_path = tmp_path / ".daemon-watchdog.log"  # empty -> no recoveries
    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N):
        _interval, consec = daemon._watchdog_tick(
            store,
            str(tmp_path / ".daemon.sock"),
            log_path,
            consec,
            probe_fn=_probe(False),
            pressure_fn=lambda: NORMAL,
            rss_fn=lambda: RSS_LOW,
        )
    # The breadcrumb write raised, yet the kill fired — and targeted self-PID.
    assert kill_calls == [(os.getpid(), signal.SIGKILL)]


def test_self_kill_is_unconditional_when_breadcrumb_fails_memory(
    tmp_path, monkeypatch
):
    """SAFETY CORE: the SAME unconditional behavior for the MEMORY path — the
    one shared _self_kill helper guarantees both paths are lock-independent and
    cannot diverge."""
    monkeypatch.setattr(
        daemon, "_daemon_started_monotonic", time.monotonic() - (GRACE + 60.0)
    )
    bad_fd = os.open(str(tmp_path / "tmp2"), os.O_WRONLY | os.O_CREAT, 0o600)
    os.close(bad_fd)
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", bad_fd)

    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        daemon.os, "kill", lambda pid, sig: kill_calls.append((pid, sig))
    )

    log_path = tmp_path / ".daemon-watchdog.log"
    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N):
        _interval, consec = daemon._watchdog_tick(
            store,
            str(tmp_path / ".daemon.sock"),
            log_path,
            consec,
            probe_fn=_probe(True),  # responsive -> only memory triggers
            pressure_fn=lambda: WARN,
            rss_fn=lambda: RSS_BIG,
        )
    assert kill_calls == [(os.getpid(), signal.SIGKILL)]


def test_self_kill_direct_breadcrumb_failure_still_kills(tmp_path, monkeypatch):
    """Directly exercise _self_kill with a forced breadcrumb failure (the unit
    proof, independent of the tick): _write_breadcrumb raises, os.kill fires."""

    def _raise(_line):
        raise OSError("simulated blocked/failed breadcrumb sink")

    monkeypatch.setattr(daemon, "_write_breadcrumb", _raise)
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        daemon.os, "kill", lambda pid, sig: kill_calls.append((pid, sig))
    )
    daemon._self_kill("wedge", daemon.DAEMON_WEDGE_KILL)
    assert kill_calls == [(os.getpid(), signal.SIGKILL)]


# --- the active round-trip probe is REAL (connect-only is the wrong primitive)
def test_probe_returns_false_when_no_socket(tmp_path):
    """A missing socket -> probe False (connect fails)."""
    sock_path = str(tmp_path / "absent.sock")
    assert asyncio.run(daemon._probe_status_roundtrip(sock_path, 0.2)) is False


def test_probe_returns_false_on_connect_but_no_reply(tmp_path):
    """The discriminating proof: a server that ACCEPTS the connection but never
    writes a reply -> probe False within read_timeout. This is exactly the
    wedged-loop case connect-only would WRONGLY pass."""
    sock_path = str(tmp_path / "silent.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    accepted: list = []

    def _accept_and_hang():
        try:
            conn, _ = srv.accept()
            accepted.append(conn)  # hold it open, never reply
        except OSError:
            pass

    t = threading.Thread(target=_accept_and_hang, daemon=True)
    t.start()
    try:
        # read_timeout small so the test is fast; the read MUST time out.
        result = asyncio.run(daemon._probe_status_roundtrip(sock_path, 0.3))
        assert result is False
    finally:
        for c in accepted:
            try:
                c.close()
            except OSError:
                pass
        srv.close()


def test_probe_returns_true_on_full_roundtrip(tmp_path):
    """A server that replies with a line -> probe True (a healthy daemon)."""
    sock_path = str(tmp_path / "healthy.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    held: list = []

    def _accept_and_reply():
        try:
            conn, _ = srv.accept()
            held.append(conn)
            conn.recv(4096)  # read the {"type":"status"} request
            conn.sendall(b'{"ok": true}\n')
        except OSError:
            pass

    t = threading.Thread(target=_accept_and_reply, daemon=True)
    t.start()
    try:
        result = asyncio.run(daemon._probe_status_roundtrip(sock_path, 1.0))
        assert result is True
    finally:
        for c in held:
            try:
                c.close()
            except OSError:
                pass
        srv.close()


# --- cross-process recovery-timestamp read-back (the breaker's disk side) --
def test_load_recovery_timestamps_reads_back_kill_lines(tmp_path):
    """The on-disk breadcrumb read-back: only *_kill lines are counted, with
    their wall-clock epochs; a needs-operator line and garbage are ignored."""
    log_path = tmp_path / ".daemon-watchdog.log"
    now = time.time()
    iso = lambda ts: daemon.datetime.fromtimestamp(ts, daemon.timezone.utc).isoformat()
    log_path.write_text(
        f"{iso(now - 5)} {daemon.DAEMON_WEDGE_KILL} reason=wedge pid=1\n"
        f"{iso(now - 6)} {daemon.DAEMON_MEMORY_PRESSURE_KILL} reason=memory pid=2\n"
        f"{iso(now - 7)} {daemon.DAEMON_WATCHDOG_NEEDS_OPERATOR} reason=x pid=3\n"
        "garbage line that should be skipped\n",
        encoding="utf-8",
    )
    ts = daemon._load_recovery_timestamps(
        log_path, (daemon.DAEMON_WEDGE_KILL, daemon.DAEMON_MEMORY_PRESSURE_KILL)
    )
    assert len(ts) == 2  # the 2 kill lines only — not needs-operator, not garbage
    # And they round-trip near the seeded epochs.
    assert abs(ts[0] - (now - 5)) < 1.0
    assert abs(ts[1] - (now - 6)) < 1.0


def test_load_recovery_timestamps_missing_file_is_empty(tmp_path):
    assert daemon._load_recovery_timestamps(
        tmp_path / "nope.log", (daemon.DAEMON_WEDGE_KILL,)
    ) == []
