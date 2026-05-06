"""Phase 4 daemon concurrency primitives (DAEMON-04, DAEMON-05).

Persistent-fd flock wrapper. Hold one instance for process lifetime.
fcntl.flock (NOT lockf) -- fd-close does not release (see apenwarr 2010, Pitfall 2).

Constitutional guard:
- C1 HUMAN-FIRST: ProcessLock.try_acquire_exclusive is non-blocking; daemon
  yields immediately when any shared lockholder exists.
- C-USER-CONSENT (formerly C2 per D7-16): the user_initiated_sleep
  branch of _dispatch_socket_request only sets pending flags after receiving
  an explicit consent payload from the wrapper; the FSM transition itself is
  performed by _tick_body, never by the dispatcher (C-DISPATCHER-FSM-ISOLATION).
- C-DISPATCHER-FSM-ISOLATION (Phase 7 structural; supersedes the bare `C2`
  inline-comment shorthand previously used at the FSM-yield call sites): the
  socket dispatcher MUST NOT transition the FSM directly; it only sets pending
  flags consumed by _tick_body under the FSM lock. New socket_server
  inherits this invariant.
- T-04-06 mitigation: flock is bound to process + open-file-description,
  so closing an unrelated fd (e.g. /etc/passwd) does NOT release our lock.
- T-04-02 mitigation: cleanup_stale_socket + asyncio cleanup_socket kwarg
  survive SIGKILL-orphaned sockets.
- T-04-07 mitigation: lock + socket created with mode 0o600 so cross-user
  access requires OS privilege escalation (out of scope).

This module has NO LLM code and NO paid-API env var references.
"""
from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

LOCK_PATH: Path = Path.home() / ".iai-mcp" / ".lock"
SOCKET_PATH: Path = Path.home() / ".iai-mcp" / ".daemon.sock"


class ProcessLock:
    """Persistent-fd flock wrapper.

    Hold one instance per process for the entire process lifetime.
    fcntl.flock (BSD) NOT lockf (POSIX) -- closing an unrelated fd does NOT
    release our lock (see apenwarr 2010, Pitfall 2).

    Semantics:
    - acquire_shared():           blocking LOCK_SH (MCP pattern)
    - try_acquire_exclusive():    LOCK_EX | LOCK_NB (daemon heavy-op pattern)
    - holds_exclusive_nb():       cooperative-yield probe
    - release():                  LOCK_UN (release without closing fd)
    - close():                    os.close() the fd (shutdown only)
    """

    def __init__(self, path: Path = LOCK_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # O_CREAT so lock file is created if missing; mode 0o600 keeps it user-only.
        self._fd: int | None = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        # Ensure mode is actually 0o600 even if umask altered it on create.
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        self._path = path

    def acquire_shared(self) -> None:
        """Blocking LOCK_SH. MCP sessions call this at session start."""
        if self._fd is None:
            raise RuntimeError("ProcessLock closed; cannot acquire")
        fcntl.flock(self._fd, fcntl.LOCK_SH)

    def try_acquire_exclusive(self) -> bool:
        """Non-blocking LOCK_EX | LOCK_NB.

        Returns True if acquired, False if any shared holder blocks us.
        Daemon calls this before heavy ops; False -> yield to MCP.
        """
        if self._fd is None:
            return False
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            raise

    def holds_exclusive_nb(self) -> bool:
        """D-06 cooperative-yield probe.

        Non-blocking check: do we still hold the exclusive lock?

        Returns True if our fd has the exclusive lock. Returns False if
        another process (e.g., MCP) acquired a shared lock while we were
        working between REM cycles.

        Implementation: fcntl.flock with LOCK_EX | LOCK_NB on our existing fd.
        On Linux/macOS, re-acquiring an already-held lock is a no-op success.
        On contention (shared lock held by another process), raises BlockingIOError
        which we catch and translate to False. EWOULDBLOCK/EAGAIN may surface as
        OSError on some platforms -- caught the same way.
        """
        if self._fd is None:
            return False
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            raise

    def release(self) -> None:
        """LOCK_UN: release lock but keep fd open for later reacquisition."""
        if self._fd is None:
            return
        fcntl.flock(self._fd, fcntl.LOCK_UN)

    def close(self) -> None:
        """Close fd. Only call at process shutdown -- closing releases the lock."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None


def cleanup_stale_socket(path: Path = SOCKET_PATH) -> None:
    """Remove a stale socket file left over from SIGKILL-orphaned daemon.

    Pitfall 10 mitigation: the in-process case is handled either by the
    3.13+ kwarg (see serve_control_socket) or by the 3.12 finally-block
    emulation, but a prior daemon killed with SIGKILL never got to run its
    cleanup. Call this BEFORE the server binds.
    """
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # Path may be a non-socket file -- still try to unlink. If even that
        # fails (e.g. permission), let asyncio surface the EADDRINUSE.
        try:
            path.unlink()
        except OSError:
            pass


def _validate_socket_message(req: dict) -> tuple[bool, str | None]:
    """Per-type schema validation (ASVS V5).

    Returns (ok, error_message). `req` must already be known to be a dict.
    """
    req_type = req.get("type")
    if not isinstance(req_type, str):
        return False, "type must be a string"

    if req_type == "status":
        # No required fields.
        return True, None

    if req_type == "user_initiated_sleep":
        reason = req.get("reason")
        ts = req.get("ts")
        if not isinstance(reason, str):
            return False, "reason must be a string"
        if not isinstance(ts, str):
            return False, "ts must be a string"
        return True, None

    if req_type in ("force_wake", "force_rem"):
        ts = req.get("ts")
        if not isinstance(ts, str):
            return False, "ts must be a string"
        return True, None

    if req_type in ("pause", "resume"):
        # pause may optionally carry `seconds`; we don't persist it as a timer
        # (the flag is binary) but we DO validate the type if supplied.
        if "seconds" in req:
            seconds = req.get("seconds")
            if not isinstance(seconds, int) or isinstance(seconds, bool):
                return False, "seconds must be an int"
        return True, None

    # TOK-14 / D5-05: 7th message type `session_open`.
    # Both session_id and ts are OPTIONAL; when supplied, they must be strings.
    # Absence is tolerated so the TS wrapper can emit a bare ping on MCP boot
    # without stalling on id/ts bookkeeping.
    if req_type == "session_open":
        if "session_id" in req and not isinstance(req["session_id"], str):
            return False, "session_id must be a string"
        if "ts" in req and not isinstance(req["ts"], str):
            return False, "ts must be a string"
        return True, None

    # Unknown types are not rejected at validation time; the dispatcher
    # returns a structured unknown_message_type response so the caller sees
    # a different reason code from "invalid_message".
    return True, None


async def _dispatch_socket_request(
    req: dict,
    store: Any,
    lock: ProcessLock,
    state: dict,
) -> dict:
    """Default dispatcher for NDJSON socket requests.

    Handles seven message types; mutates `state` in-place and persists via
    `save_state` when the message changes scheduler control flags. The
    dispatcher thread NEVER transitions the FSM directly
    (C-DISPATCHER-FSM-ISOLATION; renamed from bare `C2` per D7-16) --
    it only sets pending flags that `_tick_body` reads under the FSM lock.

    Handled types:
    - status                  -> state snapshot including version
    - user_initiated_sleep    -> set user_sleep_request pending flag
    - force_wake              -> set force_wake_request pending flag
    - force_rem               -> set force_rem_request pending flag
    - pause                   -> scheduler_paused=True
    - resume                  -> scheduler_paused=False
    - session_open            -> set first_turn_pending + hippea_cascade_request
                                 (Plan 05-04 TOK-14 / D5-05)
    - any other               -> {"ok": False, "reason": "unknown_message_type"}
    """
    # Reject non-dict requests (defence-in-depth; caller already json.loaded).
    if not isinstance(req, dict):
        return {
            "ok": False,
            "reason": "invalid_message",
            "error": "request must be a JSON object",
        }

    # Per-type schema validation (ASVS V5).
    ok, err = _validate_socket_message(req)
    if not ok:
        return {
            "ok": False,
            "reason": "invalid_message",
            "error": err or "schema_validation_failed",
        }

    req_type = req.get("type")

    # Lazy imports so test monkeypatches of STATE_PATH (via daemon_state) and
    # __version__ (via iai_mcp) always resolve to the current module state.
    from datetime import datetime, timezone

    from iai_mcp import __version__ as pkg_version
    from iai_mcp.daemon_state import save_state

    # -------------------------------------------------------- status snapshot
    if req_type == "status":
        fsm_state = state.get("fsm_state", "WAKE")
        started_at = state.get("daemon_started_at")
        uptime_sec: float | None = None
        if started_at:
            try:
                start_dt = datetime.fromisoformat(started_at)
                uptime_sec = (datetime.now(timezone.utc) - start_dt).total_seconds()
            except (TypeError, ValueError):
                uptime_sec = None

        # Truncate pending_digest to the top-level counters for socket
        # transport; the full digest can be multi-KB once insights are baked.
        pending_digest = state.get("pending_digest")
        if isinstance(pending_digest, dict):
            truncated_digest = {
                "rem_cycles_completed": pending_digest.get("rem_cycles_completed", 0),
                "episodes_processed": pending_digest.get("episodes_processed", 0),
                "schemas_induced_tier0": pending_digest.get(
                    "schemas_induced_tier0", 0,
                ),
                "claude_call_used": pending_digest.get("claude_call_used", False),
            }
        else:
            truncated_digest = None

        return {
            "ok": True,
            # Backwards-compat key used by tests/test_concurrency.py Test 6.
            "state": fsm_state,
            "uptime_sec": uptime_sec,
            # Plan 04-gap-1 additions:
            "version": pkg_version,
            "fsm_state": fsm_state,
            "last_tick_at": state.get("last_tick_at"),
            "quiet_window": state.get("quiet_window"),
            "pending_digest": truncated_digest,
            "daemon_started_at": started_at,
            "scheduler_paused": bool(state.get("scheduler_paused", False)),
        }

    # -------------------------------------------------- user_initiated_sleep
    if req_type == "user_initiated_sleep":
        current_fsm = state.get("fsm_state", "WAKE")
        if current_fsm in ("SLEEP", "DREAMING", "TRANSITIONING"):
            return {"ok": False, "reason": "already_sleeping"}

        # Clip reason to 500 chars (ASVS V5 output hardening mirror).
        reason = str(req.get("reason", ""))[:500]
        ts = str(req.get("ts", ""))
        state["user_sleep_request"] = {
            "reason": reason,
            "ts": ts,
            "pending": True,
        }
        try:
            save_state(state)
        except Exception as exc:  # noqa: BLE001 -- socket must never crash daemon
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        # Tell the caller we queued the transition; the scheduler owns the FSM
        # and will move WAKE->TRANSITIONING->SLEEP on the next tick
        # (C-DISPATCHER-FSM-ISOLATION; renamed from bare `C2` per D7-16).
        return {"ok": True, "state": "TRANSITIONING"}

    # ---------------------------------------------------------- force_wake
    if req_type == "force_wake":
        ts = str(req.get("ts", ""))
        state["force_wake_request"] = {"ts": ts, "pending": True}
        try:
            save_state(state)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "reason": "wake_queued"}

    # ----------------------------------------------------------- force_rem
    if req_type == "force_rem":
        ts = str(req.get("ts", ""))
        state["force_rem_request"] = {"ts": ts, "pending": True}
        try:
            save_state(state)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "reason": "rem_queued"}

    # --------------------------------------------------------- pause/resume
    if req_type == "pause":
        state["scheduler_paused"] = True
        try:
            save_state(state)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "paused": True}

    if req_type == "resume":
        state["scheduler_paused"] = False
        try:
            save_state(state)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "paused": False}

    # ---------------------------------------------------------- session_open
    # TOK-14 / D5-05: 7th message type. Sets two flags:
    #   - first_turn_pending[session_id] = True  -> consumed by core's
    #     _first_turn_recall_hook exactly once per session.
    #   - hippea_cascade_request {pending=True, session_id, ts} -> polled by
    #     daemon._hippea_cascade_loop which pre-warms the LRU with records
    #     from the top-K salient communities (Van de Cruys HIPPEA operational
    #     form).
    # Both flags are idempotent under a re-emit: set_overwrite is intentional
    # so a client that retries session_open gets a fresh cascade.
    if req_type == "session_open":
        # Clip session_id to 128 chars (ASVS V5 output hardening — matches
        # user_initiated_sleep.reason clip at 500).
        session_id = str(req.get("session_id", ""))[:128]
        ts = str(req.get("ts", ""))
        state["last_session_open"] = {"session_id": session_id, "ts": ts}
        # first-turn hook flag. Co-exists with existing dict form
        # written by daemon_state.mark_session_opened.
        first_turn = state.setdefault("first_turn_pending", {})
        now_iso = datetime.now(timezone.utc).isoformat()
        if isinstance(first_turn, dict):
            first_turn[session_id] = now_iso
        else:
            # Legacy scalar-bool state -> upgrade in place to the dict form.
            state["first_turn_pending"] = {session_id: now_iso}
        # cascade flag.
        state["hippea_cascade_request"] = {
            "session_id": session_id,
            "ts": ts,
            "pending": True,
        }
        try:
            save_state(state)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "reason": "session_open_queued"}

    # ------------------------------------------------------------ unknown
    return {
        "ok": False,
        "reason": "unknown_message_type",
        "type": req_type,
    }


async def serve_control_socket(
    store: Any,
    lock: ProcessLock,
    state: dict,
    shutdown: asyncio.Event,
    *,
    dispatcher: Callable[[dict], Awaitable[dict]] | None = None,
    socket_path: Path = SOCKET_PATH,
) -> None:
    """Unix socket NDJSON server at ~/.iai-mcp/.daemon.sock.

    Protocol: each line from client is a JSON request; each response is one
    JSON line back. The cleanup_socket kwarg (Python 3.13+) auto-removes the
    socket file on server shutdown; on 3.12 we emulate in the finally-block.
    Stale-socket pre-cleanup protects against SIGKILL-orphaned files.

    Permissions: chmod 0o600 immediately after bind so cross-user access
    requires privilege escalation (T-04-04 accepted risk).

    When dispatcher is provided it receives only the parsed request dict and
    must return a dict. When None, the default _dispatch_socket_request is used.
    """
    cleanup_stale_socket(socket_path)
    # Ensure parent dir exists (Path.home() / .iai-mcp could be first-run).
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    # Python 3.13 added a `cleanup_socket` kwarg to the event-loop unix server
    # that auto-removes the socket file on shutdown. On 3.12 we emulate the
    # same behaviour by unlinking in the finally-block below. See:
    # https://docs.python.org/3.13/library/asyncio-stream.html
    _supports_cleanup_socket = False
    try:
        import inspect as _inspect
        import asyncio as _asyncio_mod
        _loop_sig = _inspect.signature(
            _asyncio_mod.get_event_loop_policy().new_event_loop().create_unix_server
        )
        _supports_cleanup_socket = "cleanup_socket" in _loop_sig.parameters
    except Exception:
        _supports_cleanup_socket = False

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            try:
                req = json.loads(line)
            except (TypeError, ValueError) as exc:
                writer.write((json.dumps({"error": f"invalid_json: {exc}"}) + "\n").encode("utf-8"))
                await writer.drain()
                return
            try:
                if dispatcher is not None:
                    resp = await dispatcher(req)
                else:
                    resp = await _dispatch_socket_request(req, store, lock, state)
            except Exception as exc:  # noqa: BLE001 -- socket must never crash daemon
                resp = {"error": str(exc)}
            writer.write((json.dumps(resp) + "\n").encode("utf-8"))
            await writer.drain()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # Build server kwargs. The native 3.13+ behaviour is opted in via
    # `cleanup_socket=True`; on 3.12 the finally-block emulates the same unlink
    # so a subsequent daemon boot cannot hit EADDRINUSE.
    _server_kwargs = {"cleanup_socket": True} if _supports_cleanup_socket else {}
    server = await asyncio.start_unix_server(
        handle, path=str(socket_path), **_server_kwargs,
    )
    # chmod 0o600 immediately after bind (T-04-07 mitigation).
    try:
        os.chmod(str(socket_path), 0o600)
    except OSError:
        pass

    try:
        async with server:
            await shutdown.wait()
    finally:
        # Python 3.12 cleanup-socket emulation: remove the socket file on
        # shutdown so the next daemon boot doesn't hit EADDRINUSE. 3.13+ does
        # this natively inside the server.__aexit__.
        if not _supports_cleanup_socket:
            try:
                socket_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
