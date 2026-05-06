"""Tests for core.py additions -- DAEMON-06 / DAEMON-09.

Covers 8 behaviours:
1. consent=False short-circuits: socket is NEVER opened (C2 guard)
2. consent=True opens socket, sends NDJSON, returns daemon response
3. Missing / wrong-typed consent raises ValueError (ASVS V5 schema)
4. force_wake opens socket, sends NDJSON with 900s timeout
5. force_wake handles daemon-unreachable gracefully
6. memory_recall dispatch injects sleep_suggestion when dual-gate passes
7. memory_recall dispatch does NOT include sleep_suggestion key when gate fails
8. memory_recall does NOT break if detect_wind_down raises (silent fail)
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from iai_mcp import core


# ----------------------------------------------------------- threaded helper


class _ThreadedFakeDaemon:
    """Fake daemon that survives across multiple asyncio.run() calls.

    `core.dispatch` uses its own asyncio.run per JSON-RPC method, which tears
    down the event loop each call. A server started via asyncio.run() inside
    the test body dies when that call returns, so the next asyncio.run can
    connect to the socket file but no task is accepting -> timeout. Running
    the server on a private background loop in a daemon thread keeps the
    accept loop alive for the full test lifetime.
    """

    def __init__(self, path: Path, captured: list, reply: dict) -> None:
        self.path = path
        self.captured = captured
        self.reply = reply
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.AbstractServer | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                try:
                    line = await reader.readline()
                    if line:
                        self.captured.append(json.loads(line.decode("utf-8")))
                    writer.write((json.dumps(self.reply) + "\n").encode("utf-8"))
                    await writer.drain()
                finally:
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass

            async def _serve() -> None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._server = await asyncio.start_unix_server(_handle, path=str(self.path))
                self._ready.set()
                async with self._server:
                    await self._server.serve_forever()

            try:
                self._loop.run_until_complete(_serve())
            except asyncio.CancelledError:
                pass
            finally:
                self._loop.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        assert self._ready.wait(timeout=5.0), "fake daemon failed to start within 5s"

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return

        async def _shutdown() -> None:
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()

        fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
        try:
            fut.result(timeout=5.0)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def tmp_socket(tmp_path: Path) -> Path:
    """Provide a short unique unix-socket path.

    Unix domain sockets have a ~104-byte path limit on macOS; tmp_path can be
    too long when driven by `pytest-xdist` worker names. Fall back to /tmp
    when tmp_path would overflow.
    """
    candidate = tmp_path / "d.sock"
    if len(str(candidate)) > 100:
        candidate = Path(tempfile.mkdtemp(prefix="iai-sock-")) / "d.sock"
    return candidate


async def _run_fake_server(
    sock: Path,
    captured: list,
    reply: dict,
    *,
    delay_before_reply: float = 0.0,
) -> asyncio.AbstractServer:
    """Spin up a single-shot fake daemon over unix socket.

    Reads one NDJSON line, records it in `captured`, sleeps `delay_before_reply`
    seconds, writes `reply` as NDJSON back, closes. Returns the server object
    so the caller can close it afterwards.
    """

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if line:
                captured.append(json.loads(line.decode("utf-8")))
            if delay_before_reply > 0:
                await asyncio.sleep(delay_before_reply)
            writer.write((json.dumps(reply) + "\n").encode("utf-8"))
            await writer.drain()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    sock.parent.mkdir(parents=True, exist_ok=True)
    return await asyncio.start_unix_server(_handle, path=str(sock))


# ---------------------------------------------------------------- consent gate


def test_consent_false_short_circuits_no_socket_touch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2 invariant: consent=False must NEVER open the daemon socket."""

    async def _explode(*args, **kwargs):
        raise AssertionError(
            "C2 violation: asyncio.open_unix_connection reached with consent=False"
        )

    monkeypatch.setattr(asyncio, "open_unix_connection", _explode)

    result = asyncio.run(
        core.handle_initiate_sleep_mode({"consent": False, "reason": "not ready"})
    )
    assert result == {"ok": False, "reason": "consent_declined"}


def test_consent_missing_raises_value_error() -> None:
    with pytest.raises(ValueError, match="consent"):
        asyncio.run(core.handle_initiate_sleep_mode({"reason": "missing"}))


def test_consent_wrong_type_raises_value_error() -> None:
    # Strings / ints / None must all be rejected; only literal bool passes.
    for bad in ["true", 1, 0, None, [True]]:
        with pytest.raises(ValueError):
            asyncio.run(
                core.handle_initiate_sleep_mode({"consent": bad, "reason": "x"})
            )


def test_reason_missing_raises_value_error() -> None:
    with pytest.raises(ValueError, match="reason"):
        asyncio.run(core.handle_initiate_sleep_mode({"consent": True}))


def test_reason_wrong_type_raises_value_error() -> None:
    with pytest.raises(ValueError, match="reason"):
        asyncio.run(
            core.handle_initiate_sleep_mode({"consent": True, "reason": 42})
        )


def test_consent_true_opens_socket_and_returns_reply(
    tmp_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """consent=True path: real socket round-trip against a fake daemon."""
    captured: list[dict] = []

    async def _runner() -> dict:
        server = await _run_fake_server(
            tmp_socket, captured, {"ok": True, "state": "TRANSITIONING"},
        )
        try:
            async with server:
                # Monkeypatch core's SOCKET_PATH so _send_to_daemon uses ours.
                monkeypatch.setattr(core, "SOCKET_PATH", tmp_socket)
                return await core.handle_initiate_sleep_mode(
                    {"consent": True, "reason": "good night"},
                )
        finally:
            server.close()
            await server.wait_closed()

    result = asyncio.run(_runner())
    assert result == {"ok": True, "state": "TRANSITIONING"}
    assert len(captured) == 1
    sent = captured[0]
    assert sent["type"] == "user_initiated_sleep"
    assert sent["reason"] == "good night"
    assert "ts" in sent  # ISO timestamp attached


def test_consent_true_daemon_unreachable_returns_graceful_error(
    tmp_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Daemon down (socket file absent) must return daemon_not_running."""
    # Do NOT start a server.
    assert not tmp_socket.exists()
    monkeypatch.setattr(core, "SOCKET_PATH", tmp_socket)
    result = asyncio.run(
        core.handle_initiate_sleep_mode(
            {"consent": True, "reason": "night"},
        )
    )
    assert result["ok"] is False
    assert result["reason"] == "daemon_not_running"


# ---------------------------------------------------------------- force_wake


def test_force_wake_sends_correct_message(
    tmp_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    async def _runner() -> dict:
        server = await _run_fake_server(
            tmp_socket, captured, {"ok": True, "state": "WAKE"},
        )
        try:
            async with server:
                monkeypatch.setattr(core, "SOCKET_PATH", tmp_socket)
                return await core.handle_force_wake({})
        finally:
            server.close()
            await server.wait_closed()

    result = asyncio.run(_runner())
    assert result == {"ok": True, "state": "WAKE"}
    assert len(captured) == 1
    assert captured[0]["type"] == "force_wake"
    assert "ts" in captured[0]


def test_force_wake_daemon_unreachable_graceful(
    tmp_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not tmp_socket.exists()
    monkeypatch.setattr(core, "SOCKET_PATH", tmp_socket)
    result = asyncio.run(core.handle_force_wake({}))
    assert result["ok"] is False
    assert result["reason"] == "daemon_not_running"


def test_force_wake_timeout_is_fifteen_minutes() -> None:
    """cooperative cap is 15 minutes = 900 seconds."""
    assert core.FORCE_WAKE_TIMEOUT_SEC == 900


# ---------------------------------------------------------------- inject helper


def _window_covering_now() -> tuple[int, int]:
    """Return a quiet_window (start_bucket, duration) that contains `now`.

    Uses the current local time so the dual-gate is satisfied deterministically
    regardless of the test-host clock.
    """
    from iai_mcp.tz import load_user_tz

    tz = load_user_tz()
    now_local = datetime.now(timezone.utc).astimezone(tz)
    cur_bucket = (now_local.hour * 60 + now_local.minute) // 30
    # Make the window start 2 buckets (1h) before now and last 4h (8 buckets).
    start = (cur_bucket - 2) % 48
    return (start, 8)


def test_inject_sleep_suggestion_dual_gate_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When phrase + window both pass, response gains sleep_suggestion."""
    fake_state = {"quiet_window": _window_covering_now()}

    def _load() -> dict:
        return dict(fake_state)

    monkeypatch.setattr("iai_mcp.daemon_state.load_state", _load)

    response: dict = {"hits": [], "anti_hits": []}
    core._inject_sleep_suggestion(response, cue="good night", language="en")
    assert "sleep_suggestion" in response, (
        f"expected injection on dual-gate pass, got {response!r}"
    )
    assert response["sleep_suggestion"]["message_hint"] == "user_wind_down_detected"


def test_inject_sleep_suggestion_no_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No phrase match -> response has no sleep_suggestion key."""
    fake_state = {"quiet_window": _window_covering_now()}
    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state",
        lambda: dict(fake_state),
    )

    response: dict = {"hits": [], "anti_hits": []}
    core._inject_sleep_suggestion(
        response, cue="how do I configure pytest", language="en",
    )
    assert "sleep_suggestion" not in response


def test_inject_sleep_suggestion_no_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phrase match but no quiet_window -> response has no sleep_suggestion."""
    monkeypatch.setattr("iai_mcp.daemon_state.load_state", lambda: {})

    response: dict = {"hits": [], "anti_hits": []}
    core._inject_sleep_suggestion(response, cue="good night", language="en")
    assert "sleep_suggestion" not in response


def test_inject_sleep_suggestion_detector_raises_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If detect_wind_down raises, response goes out untouched."""
    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic bedtime failure")

    monkeypatch.setattr("iai_mcp.bedtime.detect_wind_down", _boom)

    response: dict = {"hits": [], "anti_hits": [], "budget_used": 0}
    # Must not propagate the RuntimeError.
    core._inject_sleep_suggestion(response, cue="good night", language="en")
    assert "sleep_suggestion" not in response
    # Pre-existing keys untouched.
    assert response == {"hits": [], "anti_hits": [], "budget_used": 0}


# ---------------------------------------------------------------- dispatch wiring


def test_dispatch_routes_initiate_sleep_mode(
    tmp_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synchronous `core.dispatch` entrypoint must route the new
    methods through asyncio.run -- verified by having a fake daemon
    respond to a real socket round-trip.

    The fake daemon runs in a background thread/loop so it survives
    dispatch()'s own asyncio.run (which tears down the calling loop).
    """
    captured: list[dict] = []
    daemon = _ThreadedFakeDaemon(tmp_socket, captured, {"ok": True})
    daemon.start()
    try:
        monkeypatch.setattr(core, "SOCKET_PATH", tmp_socket)
        # store arg is unused by our handlers -- pass None sentinel.
        result = core.dispatch(
            None,
            "initiate_sleep_mode",
            {"consent": True, "reason": "test"},
        )
        assert result == {"ok": True}
        assert captured[0]["type"] == "user_initiated_sleep"
    finally:
        daemon.stop()


def test_dispatch_routes_force_wake(
    tmp_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []
    daemon = _ThreadedFakeDaemon(tmp_socket, captured, {"ok": True, "state": "WAKE"})
    daemon.start()
    try:
        monkeypatch.setattr(core, "SOCKET_PATH", tmp_socket)
        result = core.dispatch(None, "force_wake", {})
        assert result == {"ok": True, "state": "WAKE"}
        assert captured[0]["type"] == "force_wake"
    finally:
        daemon.stop()
