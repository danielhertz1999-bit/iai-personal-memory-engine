"""Tests for — the 7th unix-socket message type `session_open`.

Covers:
- Valid session_open message is accepted; reply = {"ok": True, "reason": "session_open_queued"}.
- Missing session_id is tolerated (optional field per spec).
- Wrong-typed session_id is rejected at validation.
- After a valid session_open, state contains:
    * first_turn_pending[session_id] = True
    * hippea_cascade_request with pending=True
- The 6 prior message types still work (no regression).

Uses a real `serve_control_socket(store, lock, state, shutdown)` behind a
threaded background event-loop so asyncio.run() calls in the test body don't
tear the server down between requests.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from iai_mcp import concurrency, daemon_state
from iai_mcp.concurrency import (
    ProcessLock,
    _dispatch_socket_request,
    _validate_socket_message,
    serve_control_socket,
)


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def tmp_socket(tmp_path: Path) -> Path:
    """Short unique unix-socket path (macOS ~104-byte limit)."""
    candidate = tmp_path / "d.sock"
    if len(str(candidate)) > 100:
        candidate = Path(tempfile.mkdtemp(prefix="iai-sock-")) / "d.sock"
    return candidate


@pytest.fixture
def tmp_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect daemon_state.STATE_PATH to a hermetic tmp file."""
    p = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", p)
    return p


# ---------------------------------------------------------------- unit tests


def test_validate_session_open_accepts_valid_message() -> None:
    ok, err = _validate_socket_message(
        {"type": "session_open", "session_id": "s1", "ts": "2026-04-19T00:00:00Z"}
    )
    assert ok is True
    assert err is None


def test_validate_session_open_accepts_missing_session_id() -> None:
    """session_id is optional per spec; absence is tolerated."""
    ok, err = _validate_socket_message({"type": "session_open"})
    assert ok is True
    assert err is None


def test_validate_session_open_rejects_non_string_session_id() -> None:
    ok, err = _validate_socket_message(
        {"type": "session_open", "session_id": 123, "ts": "x"}
    )
    assert ok is False
    assert err is not None
    assert "session_id" in err


def test_validate_session_open_rejects_non_string_ts() -> None:
    ok, err = _validate_socket_message(
        {"type": "session_open", "session_id": "s1", "ts": 42}
    )
    assert ok is False
    assert err is not None
    assert "ts" in err


# ---------------------------------------------------------------- dispatcher unit


def _make_fake_store() -> Any:
    return MagicMock()


def _make_fake_lock() -> Any:
    return MagicMock(spec=ProcessLock)


# We call asyncio.run() directly in tests below; no asyncio marker needed.


def test_dispatch_session_open_queues_first_turn_and_cascade(
    tmp_state: Path,
) -> None:
    """session_open handler: sets first_turn_pending[session_id]=True AND
    hippea_cascade_request with pending=True; persists via save_state."""
    state: dict = {"fsm_state": "WAKE"}
    req = {
        "type": "session_open",
        "session_id": "sess-abc",
        "ts": "2026-04-19T12:00:00Z",
    }
    resp = asyncio.run(
        _dispatch_socket_request(req, _make_fake_store(), _make_fake_lock(), state)
    )
    assert resp == {"ok": True, "reason": "session_open_queued"}
    # Flag set for first-turn hook.
    pending = state.get("first_turn_pending")
    assert isinstance(pending, dict)
    stamp = pending.get("sess-abc")
    assert isinstance(stamp, str) and stamp  # ISO-8601 timestamp, post-fix
    # Flag set for cascade task.
    cascade = state.get("hippea_cascade_request")
    assert isinstance(cascade, dict)
    assert cascade.get("pending") is True
    assert cascade.get("session_id") == "sess-abc"
    # Echo for introspection.
    last = state.get("last_session_open")
    assert isinstance(last, dict)
    assert last.get("session_id") == "sess-abc"
    # Persisted to disk.
    assert tmp_state.exists()
    on_disk = json.loads(tmp_state.read_text())
    assert on_disk.get("hippea_cascade_request", {}).get("pending") is True


def test_dispatch_session_open_missing_session_id_ok(tmp_state: Path) -> None:
    """No session_id -> defaults to empty string; still queues cascade."""
    state: dict = {"fsm_state": "WAKE"}
    req = {"type": "session_open", "ts": "2026-04-19T12:00:00Z"}
    resp = asyncio.run(
        _dispatch_socket_request(req, _make_fake_store(), _make_fake_lock(), state)
    )
    assert resp.get("ok") is True
    assert resp.get("reason") == "session_open_queued"


def test_dispatch_session_open_clips_long_session_id(tmp_state: Path) -> None:
    """session_id is clipped to 128 chars (ASVS V5 output hardening)."""
    state: dict = {"fsm_state": "WAKE"}
    long_id = "a" * 1000
    req = {"type": "session_open", "session_id": long_id, "ts": "x"}
    resp = asyncio.run(
        _dispatch_socket_request(req, _make_fake_store(), _make_fake_lock(), state)
    )
    assert resp["ok"] is True
    last = state.get("last_session_open") or {}
    assert len(last.get("session_id", "")) <= 128


# ---------------------------------------------------------------- no-regression


def test_dispatch_force_wake_still_works(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE"}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "force_wake", "ts": "x"},
            _make_fake_store(),
            _make_fake_lock(),
            state,
        )
    )
    assert resp == {"ok": True, "reason": "wake_queued"}


def test_dispatch_force_rem_still_works(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE"}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "force_rem", "ts": "x"},
            _make_fake_store(),
            _make_fake_lock(),
            state,
        )
    )
    assert resp == {"ok": True, "reason": "rem_queued"}


def test_dispatch_pause_still_works(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE"}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "pause"},
            _make_fake_store(),
            _make_fake_lock(),
            state,
        )
    )
    assert resp == {"ok": True, "paused": True}
    assert state["scheduler_paused"] is True


def test_dispatch_resume_still_works(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE", "scheduler_paused": True}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "resume"},
            _make_fake_store(),
            _make_fake_lock(),
            state,
        )
    )
    assert resp == {"ok": True, "paused": False}
    assert state["scheduler_paused"] is False


def test_dispatch_user_initiated_sleep_still_works(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE"}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "user_initiated_sleep", "reason": "night", "ts": "x"},
            _make_fake_store(),
            _make_fake_lock(),
            state,
        )
    )
    assert resp.get("ok") is True
    assert resp.get("state") == "TRANSITIONING"


def test_dispatch_status_still_works(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE"}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "status"},
            _make_fake_store(),
            _make_fake_lock(),
            state,
        )
    )
    assert resp.get("ok") is True
    assert resp.get("state") == "WAKE"
    # Version echoed in session-open response.
    assert "version" in resp


# ---------------------------------------------------------------- round-trip


class _ThreadedDaemon:
    """Real serve_control_socket on background thread + event loop.

    Reuses the pattern from tests/test_core_bedtime_inject.py but drives the
    production _dispatch_socket_request so we exercise the real 7th-message
    path end-to-end.
    """

    def __init__(self, path: Path, state: dict) -> None:
        self.path = path
        self.state = state
        self.lock = MagicMock(spec=ProcessLock)
        self.store = MagicMock()
        self.shutdown = None  # populated on the loop thread
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self.shutdown = asyncio.Event()

            async def _serve() -> None:
                # Hand the real dispatcher the state we own.
                async def _dispatcher(req: dict) -> dict:
                    return await _dispatch_socket_request(
                        req, self.store, self.lock, self.state
                    )

                task = asyncio.create_task(
                    serve_control_socket(
                        self.store,
                        self.lock,
                        self.state,
                        self.shutdown,  # type: ignore[arg-type]
                        dispatcher=_dispatcher,
                        socket_path=self.path,
                    )
                )
                # Give the server a moment to bind before signalling ready.
                await asyncio.sleep(0.1)
                self._ready.set()
                await task

            try:
                self._loop.run_until_complete(_serve())
            except Exception:
                pass
            finally:
                try:
                    self._loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        assert self._ready.wait(timeout=5.0), "threaded daemon failed to start"

    def stop(self) -> None:
        if self._loop is None:
            return
        if self.shutdown is not None:
            self._loop.call_soon_threadsafe(self.shutdown.set)
        self._thread and self._thread.join(timeout=5.0)


async def _send(path: Path, msg: dict, *, timeout: float = 5.0) -> dict:
    reader, writer = await asyncio.open_unix_connection(str(path))
    try:
        writer.write((json.dumps(msg) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        return json.loads(line)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def test_session_open_end_to_end_round_trip(
    tmp_socket: Path, tmp_state: Path,
) -> None:
    """Real NDJSON round-trip over a unix socket — the 7th message type."""
    state: dict = {"fsm_state": "WAKE"}
    daemon = _ThreadedDaemon(tmp_socket, state)
    daemon.start()
    try:
        resp = asyncio.run(
            _send(
                tmp_socket,
                {
                    "type": "session_open",
                    "session_id": "e2e-sess-1",
                    "ts": "2026-04-19T12:00:00Z",
                },
            )
        )
        assert resp == {"ok": True, "reason": "session_open_queued"}
        # State mutations visible to the test after the reply.
        pending = state.get("first_turn_pending")
        assert isinstance(pending, dict)
        stamp = pending.get("e2e-sess-1")
        assert isinstance(stamp, str) and stamp  # ISO-8601 timestamp, post-fix
        cascade = state.get("hippea_cascade_request")
        assert isinstance(cascade, dict)
        assert cascade.get("pending") is True
    finally:
        daemon.stop()


def test_session_open_does_not_regress_other_6_types(
    tmp_socket: Path, tmp_state: Path,
) -> None:
    """Force_wake / force_rem / pause / resume / status / user_initiated_sleep
    all still succeed end-to-end."""
    state: dict = {"fsm_state": "WAKE"}
    daemon = _ThreadedDaemon(tmp_socket, state)
    daemon.start()
    try:
        # force_wake
        r = asyncio.run(_send(tmp_socket, {"type": "force_wake", "ts": "x"}))
        assert r == {"ok": True, "reason": "wake_queued"}
        # force_rem
        r = asyncio.run(_send(tmp_socket, {"type": "force_rem", "ts": "x"}))
        assert r == {"ok": True, "reason": "rem_queued"}
        # pause
        r = asyncio.run(_send(tmp_socket, {"type": "pause"}))
        assert r.get("ok") is True
        # resume
        r = asyncio.run(_send(tmp_socket, {"type": "resume"}))
        assert r.get("ok") is True
        # status
        r = asyncio.run(_send(tmp_socket, {"type": "status"}))
        assert r.get("ok") is True
        # user_initiated_sleep (state is WAKE so this transitions)
        r = asyncio.run(
            _send(
                tmp_socket,
                {"type": "user_initiated_sleep", "reason": "night", "ts": "x"},
            )
        )
        assert r.get("ok") is True
    finally:
        daemon.stop()
