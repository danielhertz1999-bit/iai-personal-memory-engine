"""Daemon socket-server.

NDJSON JSON-RPC 2.0 server over the daemon control socket. Reuses
core.dispatch() with stdio (both transports share one function).

Guards:
- FSM isolation: the socket dispatcher MUST NOT transition the
  daemon FSM directly; it calls core.dispatch which returns a dict. FSM
  transitions remain owned by the daemon FSM tick.
- Human-first: in-process cooperative yield via last_activity_ts and
  active_connections probes; the daemon REM scheduler reads these between cycles.
- Zero API cost: imports stdlib + core.dispatch only; no SDK references.
- Literal preservation: zero record mutation paths; transport-only adapter.
- Fail-loud surface: daemon-side raises become JSON-RPC error code -32001;
  wrapper-side socket-death surfaces as -32002.
- Backward-compat: imports core.dispatch; no transport branching.

Single-socket dispatcher fork: each accepted NDJSON line is parsed once, then
routed by shape:
  - jsonrpc=='2.0'  -> core.dispatch (MCP methods)
  - 'type' in CONTROL_MSG_TYPES (control plane) -> forward verbatim to
    concurrency._dispatch_socket_request (state must be wired by daemon.main()
    via SocketServer(store, state=...); standalone tests do not exercise this
    branch -- the forks are independent).
  - else -> JSON-RPC ERR_INVALID_REQUEST.

launchd socket activation: serve() forks on the LISTEN_FDS env var. When
launchd-managed (LISTEN_FDS=1, LISTEN_PID==os.getpid()), inherit the pre-bound
fd 3 via the systemd-compatible inherited-fd protocol; SKIP cleanup_stale_socket,
mkdir, chmod, and post-serve unlink (launchd owns the socket file). Otherwise
binds the path manually (development, tests, non-Darwin). See
_inherit_launchd_socket.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from iai_mcp.concurrency import SOCKET_PATH, cleanup_stale_socket
from iai_mcp.core import UnknownMethodError

# JSON-RPC 2.0 server-error codes (jsonrpc.org/specification reserves
# -32099..-32000 for "implementation-defined server-errors").
ERR_DAEMON_INTERNAL = -32001    # internal dispatch failure
ERR_INVALID_REQUEST = -32600    # malformed JSON-RPC envelope
ERR_METHOD_NOT_FOUND = -32601   # core.dispatch raised UnknownMethodError
ERR_INVALID_PARAMS = -32602     # core.dispatch raised TypeError or KeyError on params
ERR_PARSE_ERROR = -32700        # json.loads failed

# `IDLE_SECS_DEFAULT` and `idle_secs` are kept on the SocketServer
# constructor for backward compat with existing tests, but no
# in-process loop consumes them anymore. The lifecycle state machine
# (heartbeat scanner + idle detector + sleep_pipeline + Hibernation
# transition) owns idle-shutdown responsibility.
IDLE_SECS_DEFAULT = 1800        # 30 minutes (kept for compat)


def _inherit_launchd_socket() -> socket.socket | None:
    """Return inherited unix socket from launchd, or None for manual run.

    Implements the systemd-style inherited-fd protocol (also honored by
    macOS launchd):
      - LISTEN_FDS env var = number of inherited fds (must be >= 1).
      - LISTEN_PID env var = pid of process meant to inherit (must == os.getpid()).
      - First inherited fd is 3 (SD_LISTEN_FDS_START).

    Returns None on ANY mismatch / parse-failure / env-absent so caller can
    fall back to the manual bind path. Defensive against:
      - env vars absent (manual `python -m iai_mcp.daemon` from terminal)
      - LISTEN_PID inherited from a parent but not meant for us
      - LISTEN_FDS=0 (launchd would never set this, but be safe)
      - non-integer values (raise-free; return None)
    """
    listen_fds = os.environ.get("LISTEN_FDS")
    listen_pid = os.environ.get("LISTEN_PID")
    if listen_fds is None or listen_pid is None:
        return None
    try:
        if int(listen_pid) != os.getpid():
            return None
        if int(listen_fds) < 1:
            return None
    except ValueError:
        return None
    inherited_fd = 3  # SD_LISTEN_FDS_START
    sock = socket.socket(fileno=inherited_fd)
    sock.setblocking(False)
    return sock


def _validate_jsonrpc_envelope(req: Any) -> tuple[bool, str | None]:
    """Envelope schema check: jsonrpc=='2.0', id present and non-null, method is string."""
    if not isinstance(req, dict):
        return False, "request must be a JSON object"
    if req.get("jsonrpc") != "2.0":
        return False, "jsonrpc must be '2.0'"
    if "id" not in req or req["id"] is None:
        return False, "id required and non-null"
    if not isinstance(req.get("method"), str):
        return False, "method must be a string"
    if "params" in req and not isinstance(req["params"], (dict, list)):
        return False, "params must be object or array"
    return True, None


class SocketServer:
    """Per-connection multiplexed JSON-RPC 2.0 server over unix socket.

    Single-socket dispatcher: the same accept loop handles both daemon control
    messages (forwarded to concurrency._dispatch_socket_request when state is
    wired) and JSON-RPC MCP envelopes (routed via core.dispatch on a worker
    thread).

    Constructor args:
      store: shared MemoryStore (singleton in daemon.main(); fresh in tests).
      idle_secs: idle-shutdown threshold; falls back to env override then
                 IDLE_SECS_DEFAULT when None.
      state: shared state dict for the control-plane fork (daemon.main() wires
             it; a standalone path leaves None and the control branch returns a
             structured "control_plane_unwired" error if exercised).
    """

    # Daemon control-message types -- used by the dispatcher fork.
    # Source of truth: concurrency.py:_dispatch_socket_request branches.
    # embed_cue: lightweight warm-embedder RPC (awake-accelerator role, no
    # store open — CLIENT-facing, not a daemon self-call).
    CONTROL_MSG_TYPES = frozenset({
        "status", "user_initiated_sleep", "force_wake", "force_rem",
        "pause", "resume", "session_open", "embed_cue",
    })

    def __init__(
        self,
        store: Any,
        idle_secs: int | None = None,
        *,
        state: dict | None = None,
    ) -> None:
        self.store = store
        # The env override `IAI_DAEMON_IDLE_SHUTDOWN_SECS` was removed; the
        # constructor default falls through to IDLE_SECS_DEFAULT (1800). The
        # attribute is kept for back-compat with telemetry / tests but no
        # in-process loop reads it anymore.
        if idle_secs is None:
            idle_secs = IDLE_SECS_DEFAULT
        self.idle_secs = idle_secs
        self.last_activity_ts: float = time.monotonic()
        self.active_connections: int = 0
        # asyncio.Event lazy-binds to the running loop on first wait/set, so it
        # is safe to construct here even before the loop starts (Python 3.10+).
        self.shutdown_event: asyncio.Event = asyncio.Event()
        # Control-plane fork wiring (daemon.main() supplies this).
        self._state = state

    async def handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """One coroutine per accepted connection. Reads NDJSON lines, dispatches each.

        Dispatch fork on each line:
          - jsonrpc=='2.0'  -> core.dispatch (MCP JSON-RPC 2.0)
          - 'type' in CONTROL_MSG_TYPES and no jsonrpc -> daemon control plane
          - else -> JSON-RPC ERR_INVALID_REQUEST.
        """
        self.active_connections += 1
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                self.last_activity_ts = time.monotonic()
                req_id: Any = None
                try:
                    req = json.loads(line)
                except json.JSONDecodeError as e:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": ERR_PARSE_ERROR, "message": str(e)},
                    }
                    writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                    await writer.drain()
                    continue

                # Dispatcher fork branch 1: control message (no jsonrpc field).
                if (
                    isinstance(req, dict)
                    and req.get("type") in self.CONTROL_MSG_TYPES
                    and "jsonrpc" not in req
                ):
                    if self._state is None:
                        # Standalone path: the control plane needs daemon
                        # context (daemon.main() wires it).
                        result = {
                            "ok": False,
                            "reason": "control_plane_unwired",
                            "error": (
                                "SocketServer constructed without state; "
                                "control-plane fork unavailable in this context"
                            ),
                        }
                    else:
                        try:
                            # Lazy local import; signature: (req, store, state).
                            from iai_mcp.concurrency import _dispatch_socket_request
                            result = await _dispatch_socket_request(
                                req, self.store, self._state,
                            )
                        except Exception as e:  # noqa: BLE001
                            # Control-plane errors must not crash the daemon.
                            # Return a structured error.
                            result = {"ok": False, "reason": "control_plane_error",
                                      "error": str(e)[:200]}
                    if result is not None:
                        writer.write((json.dumps(result) + "\n").encode("utf-8"))
                        await writer.drain()
                    continue

                # JSON-RPC 2.0 envelope branch.
                ok, err = _validate_jsonrpc_envelope(req)
                req_id = req.get("id") if isinstance(req, dict) else None
                if not ok:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": ERR_INVALID_REQUEST, "message": err},
                    }
                    writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                    await writer.drain()
                    continue
                method = req["method"]
                params = req.get("params") or {}
                try:
                    # Lazy local import keeps daemon startup snappy and dodges
                    # circular-import edge cases during async test fixture setup.
                    from iai_mcp.core import dispatch
                    # dispatch is sync + can take 50-500 ms. asyncio.to_thread
                    # prevents head-of-line blocking across connections.
                    # The threading.RLock in core.py keeps profile mutations
                    # safe under concurrent worker-thread access.
                    result = await asyncio.to_thread(
                        dispatch, self.store, method, params,
                    )
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
                except UnknownMethodError as e:
                    # V3-03 fix: unknown method now raises (was: in-band {error:...} dict).
                    # e.args[0] is the unknown method name (per core.UnknownMethodError contract).
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": ERR_METHOD_NOT_FOUND,
                            "message": f"unknown method '{e.args[0]}'",
                        },
                    }
                except KeyError as e:
                    # V3-04 fix: KeyError from missing required params (e.g. params["cue"]).
                    # Was incorrectly mapped to -32601; correct code is -32602 INVALID_PARAMS.
                    # e.args[0] is the missing key name.
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": ERR_INVALID_PARAMS,
                            "message": f"missing required param: {e.args[0]!r}",
                        },
                    }
                except TypeError as e:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": ERR_INVALID_PARAMS, "message": str(e)},
                    }
                except Exception as e:  # noqa: BLE001 -- socket must never crash daemon
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": ERR_DAEMON_INTERNAL, "message": str(e)},
                    }
                writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            # Client closed the socket mid-write (common when the MCP wrapper
            # in Claude Code/Cursor exits or the host kills its pipe). Expected
            # behavior — not a daemon fault. Falls through to finally cleanup
            # without the asyncio "Unhandled exception in client_connected_cb"
            # noise that previously flooded launchd-stderr.log.
            pass
        finally:
            self.active_connections -= 1
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, ConnectionError):  # noqa: BLE001 -- cleanup is best-effort
                pass

    # `idle_watcher` removed. The lifecycle state machine +
    # heartbeat scanner + idle detector supersede this in-process timer.
    # `last_activity_ts` / `active_connections` accounting on this object
    # is preserved (used by tests + future observability) but no internal
    # loop consumes them.

    async def serve(self, socket_path: Path | None = None) -> None:
        """Bind socket, run server until shutdown_event set, drain in-flight, unlink socket.

        When launchd has pre-bound the listener (LISTEN_FDS env set and
        LISTEN_PID==os.getpid()), inherit fd 3 and call
        asyncio.start_unix_server with sock=. SKIP cleanup_stale_socket,
        mkdir, chmod, post-serve unlink, and the cleanup_socket=True kwarg
        -- launchd owns the socket file's lifecycle (SockPathMode=384
        already applied at bind time). Otherwise (development, tests,
        non-Darwin) use the manual-bind path: cleanup_stale -> mkdir ->
        bind -> chmod, with post-serve unlink on Python < 3.13.
        """
        if socket_path is None:
            # Honor IAI_DAEMON_SOCKET_PATH env override for test isolation.
            env_path = os.environ.get("IAI_DAEMON_SOCKET_PATH")
            socket_path = Path(env_path) if env_path else SOCKET_PATH

        # Detect Python 3.13+ cleanup_socket kwarg (mirror the same probe used
        # in concurrency.py to keep behavior identical between the two servers).
        sig = inspect.signature(asyncio.start_unix_server)
        supports_cleanup_socket = "cleanup_socket" in sig.parameters

        inherited = _inherit_launchd_socket()
        if inherited is not None:
            # launchd socket activation. launchd owns the socket file:
            # do NOT cleanup_stale_socket (would unlink launchd's listener and
            # brick subsequent activations), do NOT mkdir (path already exists
            # since launchd bound it), do NOT chmod (SockPathMode=384 applied
            # at bind), do NOT pass cleanup_socket=True (asyncio would unlink
            # on close), do NOT post-serve unlink. launchd manages the file.
            server = await asyncio.start_unix_server(
                self.handle,
                sock=inherited,
            )
        else:
            # Manual-run fallback (development, tests, non-Darwin).
            cleanup_stale_socket(socket_path)
            socket_path.parent.mkdir(parents=True, exist_ok=True)
            server_kwargs: dict[str, Any] = (
                {"cleanup_socket": True} if supports_cleanup_socket else {}
            )
            server = await asyncio.start_unix_server(
                self.handle,
                path=str(socket_path),
                **server_kwargs,
            )
            # chmod 0o600 immediately after bind to restrict socket access.
            try:
                os.chmod(str(socket_path), 0o600)
            except OSError:
                pass

        # idle_task removed; lifecycle state machine drives shutdown via
        # Hibernation transitions.
        try:
            async with server:
                await self.shutdown_event.wait()
                # Graceful shutdown: stop accepting new connections, drain in-flight.
                server.close()
                await server.wait_closed()
        finally:
            # Manual unlink fallback ONLY for the manual-bind branch on
            # Python <3.13. Under launchd, NEVER unlink -- launchd owns the file.
            if inherited is None and not supports_cleanup_socket:
                try:
                    socket_path.unlink()
                except (FileNotFoundError, OSError):
                    pass
