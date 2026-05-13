"""V3-05 regression test: bridge reconnect race + socket-death window.

-01 / . Reproduces the race in `mcp-wrapper/src/bridge.ts`
where a `bridge.call()` arriving in the gap between socket close and
reconnect-completion would reject with `daemon_unreachable` even though
the daemon is healthy. Pre-fix: the EventEmitter "close" handler fires
fire-and-forget against an async `handleSocketDeath`; Node does not
await the returned Promise, so a concurrent call sees `this.sock === null`
and short-circuits to rejection. Post-fix: `handleSocketDeath` writes
its async work to a `reconnectPromise: Promise<void> | null` field and
`call()` awaits it before checking socket state.

Pattern: per PATTERNS.md B-01, this test lives Python-side
(not in `mcp-wrapper/tests/integration/`) because `mcp-wrapper/` has no
TS test runner configured. The wrapper-spawn helpers mirror
`tests/test_mcp_tools.py:139-181` (`_spawn_wrapper`, `_initialize`,
`_mcp_call`).

The harness uses a minimal Python unix-socket listener (the "fake
daemon") rather than the real `iai_mcp.daemon` because the real
daemon's cold start (~7-8s for bge-small embedder load + LanceDB open)
exceeds the wrapper's `SOCKET_CONNECT_TIMEOUT_MS = 5000` reconnect
budget — a realistic kill-and-respawn scenario can't reliably win the
5s reconnect race even with warm caches. The fake daemon binds within
milliseconds and stays bound throughout the test; only the wrapper's
*accepted* connection is forcibly closed via a stdin DROP command. This
isolates exactly the V3-05 race: socket-close event, in-flight
reconnect, racing call, reconnect succeeds.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"


def _wrapper_ready() -> bool:
    return (WRAPPER / "dist" / "index.js").exists()


@pytest.fixture(scope="module")
def built_wrapper() -> Path:
    if not _wrapper_ready():
        if not (WRAPPER / "node_modules").exists():
            subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
        subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = WRAPPER / "dist" / "index.js"
    if not dist.exists():
        pytest.skip(f"mcp-wrapper not built; missing {dist}")
    return dist


# ---------------------------------------------------------------------------
# Fake daemon: minimal JSON-RPC NDJSON listener.
#
# Real daemon cold-start (~7-8s for bge-small embedder load + LanceDB open)
# exceeds the wrapper's 5s reconnect timeout (SOCKET_CONNECT_TIMEOUT_MS in
# mcp-wrapper/src/bridge.ts:18). To exercise the V3-05 race fix we need a
# substitute listener that BINDS within milliseconds of being asked, so
# the wrapper's at-most-one reconnect actually succeeds. The fake daemon
# answers every JSON-RPC request with a valid `{"result": {...}}` payload
# — sufficient to confirm `bridge.call()` did NOT short-circuit to
# `daemon_unreachable`.
# ---------------------------------------------------------------------------


_FAKE_DAEMON_SCRIPT = r"""
# Minimal stand-in for the real iai-mcp daemon's socket_server. Binds the
# unix socket the wrapper is configured to dial; answers every JSON-RPC
# request with a synthetic result. A DROP command on stdin closes the
# wrapper's currently-accepted connection WITHOUT touching the listening
# socket — so the wrapper sees "close", fires its EE handler, and the
# next reconnect attempt immediately re-accepts.
import json, os, socket, sys, threading

sock_path = sys.argv[1]
try:
    os.unlink(sock_path)
except FileNotFoundError:
    pass

srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(sock_path)
srv.listen(8)

state_lock = threading.Lock()
live_conns = []  # type: list[socket.socket]

sys.stdout.write("BOUND\n")
sys.stdout.flush()


def serve(conn):
    buf = b""
    try:
        while True:
            data = conn.recv(65536)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                rid = req.get("id")
                method = req.get("method", "")
                resp = {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "ok": True,
                        "method": method,
                        "fake_daemon": True,
                    },
                }
                try:
                    conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                except Exception:
                    return
    except Exception:
        pass
    finally:
        with state_lock:
            try:
                live_conns.remove(conn)
            except ValueError:
                pass
        try:
            conn.close()
        except Exception:
            pass


def stdin_reader():
    for raw in sys.stdin:
        cmd = raw.strip()
        if cmd == "DROP":
            # Close every live wrapper-accepted connection. The wrapper's
            # EE "close" handler fires; the listening socket stays bound
            # so the wrapper's reconnect immediately re-accepts.
            with state_lock:
                victims = list(live_conns)
                live_conns.clear()
            for c in victims:
                try:
                    c.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    c.close()
                except Exception:
                    pass
            sys.stdout.write("DROPPED\n")
            sys.stdout.flush()
        elif cmd == "QUIT":
            break


threading.Thread(target=stdin_reader, daemon=True).start()


while True:
    try:
        conn, _ = srv.accept()
    except Exception:
        break
    with state_lock:
        live_conns.append(conn)
    threading.Thread(target=serve, args=(conn,), daemon=True).start()
"""


def _spawn_fake_daemon(sock_path: Path) -> subprocess.Popen:
    """Spawn the minimal fake daemon. Binds within milliseconds.

    Returns a Popen with stdin/stdout pipes:
    - Write `b"DROP\n"` to stdin to close every live wrapper connection
      while keeping the listening socket bound (forces the wrapper to
      observe socket_close and trigger handleSocketDeath).
    - Read `b"DROPPED\n"` from stdout to confirm the drop was processed.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", _FAKE_DAEMON_SCRIPT, str(sock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for the BOUND signal so the caller is sure the socket is live.
    deadline = time.monotonic() + 10.0
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line.strip() == b"BOUND":
            return proc
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr is not None else b""
            raise RuntimeError(
                f"fake daemon exited before binding: {err.decode(errors='replace')}"
            )
    proc.kill()
    raise RuntimeError("fake daemon did not bind within 10s")


def _drop_fake_daemon_conn(proc: subprocess.Popen) -> None:
    """Tell the fake daemon to close every live accepted connection."""
    assert proc.stdin is not None
    proc.stdin.write(b"DROP\n")
    proc.stdin.flush()
    # Wait for the DROPPED ack so we know the close has been issued.
    assert proc.stdout is not None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line.strip() == b"DROPPED":
            return
    raise RuntimeError("fake daemon did not ack DROP within 5s")


@pytest.fixture
def fake_daemon():
    """Function-scoped fake-daemon harness. Returns dict with:

    - `path`: the unix socket path the listener is bound to.
    - `proc`: the underlying Popen handle.
    - `drop_connections()`: tell the listener to close every currently
      accepted wrapper connection without touching the listening socket;
      forces the wrapper to observe socket_close and fire its
      handleSocketDeath path.

    Why a fake daemon and not the real one: the real daemon's cold start
    (bge-small embedder load + LanceDB open) is ~7-8s on macOS, which
    exceeds the wrapper's `SOCKET_CONNECT_TIMEOUT_MS = 5000` reconnect
    budget. To exercise the V3-05 fix in isolation we need a listener
    that is **always bound** so the wrapper's at-most-one reconnect
    attempt actually succeeds. The fake daemon answers every JSON-RPC
    request with a synthetic `{"result": {...}}` payload — sufficient
    to confirm `bridge.call()` did NOT short-circuit to
    `daemon_unreachable`. The wrapper's bridge code path (the unit
    under test) is exercised end-to-end; the daemon-side dispatch is
    not.
    """
    sock_dir = Path(f"/tmp/iai-mcp-disconnect-{os.getpid()}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"

    proc = _spawn_fake_daemon(sock_path)

    def drop_connections() -> None:
        _drop_fake_daemon_conn(proc)

    yield {"path": sock_path, "proc": proc, "drop_connections": drop_connections}

    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        sock_path.unlink()
    except OSError:
        pass
    try:
        shutil.rmtree(sock_dir, ignore_errors=True)
    except OSError:
        pass


def _spawn_wrapper(
    built_wrapper: Path,
    daemon_sock: Path,
    reconnect_delay_ms: int = 1000,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["IAI_MCP_PYTHON"] = sys.executable
    tmpdir = tempfile.mkdtemp(prefix="iai-mcp-disconnect-test-")
    env["IAI_MCP_STORE"] = tmpdir
    env["IAI_DAEMON_SOCKET_PATH"] = str(daemon_sock)
    # Widen the V3-05 race window deterministically so the racing call()
    # below can land BEFORE the wrapper's reconnectPromise resolves.
    # Production keeps this unset → 0 ms → no-op. See bridge.ts
    # handleSocketDeath IIFE for the production-safe gate.
    env["IAI_MCP_RECONNECT_TEST_DELAY_MS"] = str(reconnect_delay_ms)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        ["node", str(built_wrapper)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _mcp_call(
    proc: subprocess.Popen,
    method: str,
    params: dict,
    rpc_id: int = 99,
    timeout_s: float = 10.0,
) -> dict:
    req = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(req) + "\n").encode())
    proc.stdin.flush()
    assert proc.stdout is not None
    # Naive readline; the wrapper writes one JSON line per response.
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("wrapper closed stdout before replying")
        try:
            return json.loads(line.decode())
        except json.JSONDecodeError:
            # Skip non-JSON noise lines.
            continue
    raise RuntimeError(f"timeout waiting for {method} response")


def _initialize(proc: subprocess.Popen, rpc_id: int = 1) -> None:
    resp = _mcp_call(
        proc,
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "iai-mcp-disconnect-test", "version": "0.1.0"},
        },
        rpc_id,
    )
    assert "result" in resp, f"initialize failed: {resp}"
    assert proc.stdin is not None
    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    proc.stdin.write((json.dumps(note) + "\n").encode())
    proc.stdin.flush()


def test_call_during_socket_death_resolves_after_reconnect(
    built_wrapper: Path,
    fake_daemon: dict,
) -> None:
    """V3-05 regression: tools/call issued in the socket-death window must
    not reject with daemon_unreachable when the daemon is still
    reachable.

    Pre-fix (bridge.ts un-modified): the EventEmitter "close" handler
    fires fire-and-forget against an async handleSocketDeath; Node does
    NOT await the returned Promise. A racing tools/call arrives, sees
    this.sock === null, rejects daemon_unreachable BEFORE the reconnect
    attempt commits the new socket back to this.sock.

    Post-fix: handleSocketDeath assigns its async reconnect work to
    this.reconnectPromise; bridge.call() awaits that promise BEFORE
    checking !this.sock, so the racing call serializes onto the
    reconnect outcome. With the listening socket continuously bound,
    the wrapper's at-most-one reconnect succeeds against the SAME
    listener that just dropped its connection, and the racing call
    resolves cleanly.

    Test harness uses a minimal Python unix-socket listener (not the
    real daemon) because the real daemon's cold start (~7-8s for
    bge-small embedder load + LanceDB open) exceeds the wrapper's
    `SOCKET_CONNECT_TIMEOUT_MS = 5000` reconnect budget. The fake
    daemon's listening socket is always bound; only the wrapper's
    accepted connection is forcibly closed via a stdin DROP command.

    The test sets `IAI_MCP_RECONNECT_TEST_DELAY_MS=1000` in the wrapper
    process env so the wrapper's reconnect IIFE sleeps 1s before
    re-connecting. Production runs leave the env var unset → 0 ms →
    no-op. Without this widener the race window between socket close
    and reconnect-completion is sub-millisecond on a unix-socket loopback,
    so the test cannot deterministically discriminate pre-fix from
    post-fix behavior. With the widener, the racing tools/call lands at
    t≈50ms while the reconnect IIFE is still sleeping; pre-fix that
    triggers daemon_unreachable, post-fix it awaits reconnectPromise.
    """
    sock_path = fake_daemon["path"]
    wrapper = _spawn_wrapper(built_wrapper, sock_path)
    try:
        _initialize(wrapper)

        # Sanity: first tools/call round-trips through the fake daemon.
        # The fake daemon answers every method with a synthetic result;
        # the wrapper does NOT short-circuit to daemon_unreachable here.
        r1 = _mcp_call(
            wrapper,
            "tools/call",
            {"name": "topology", "arguments": {}},
            rpc_id=2,
        )
        err_str_1 = json.dumps(r1)
        assert "daemon_unreachable" not in err_str_1, (
            f"baseline call already broken: {r1}"
        )

        # Race step: instruct the fake daemon to drop the wrapper's
        # accepted connection. The listening socket stays bound so
        # the wrapper's at-most-one reconnect immediately re-accepts.
        # The wrapper's EE "close" handler fires; handleSocketDeath
        # starts its reconnectPromise IIFE.
        fake_daemon["drop_connections"]()

        # Brief grace so the close event surfaces in the wrapper's
        # EventEmitter loop and the reconnectPromise field is populated
        # before our racing tools/call arrives. Without this nudge the
        # racing call could land BEFORE the close event has been observed
        # at all, in which case `this.sock` is still the (now-dead) live
        # socket and `bridge.write` succeeds but never gets a reply.
        time.sleep(0.05)

        # Issue the racing tools/call.
        # Pre-fix: bridge.call() is sync; it sees this.sock === null
        # (handleSocketDeath nulled it) and short-circuits to
        # daemon_unreachable, NOT awaiting the in-flight reconnect.
        # Post-fix: bridge.call() is async and awaits
        # this.reconnectPromise; reconnect succeeds against the
        # always-bound listening socket; call proceeds and gets a real
        # JSON-RPC response. The assertion below only forbids the
        # daemon_unreachable string.
        r2 = _mcp_call(
            wrapper,
            "tools/call",
            {"name": "topology", "arguments": {}},
            rpc_id=3,
            timeout_s=20.0,
        )
        err_str_2 = json.dumps(r2)
        assert "daemon_unreachable" not in err_str_2, (
            f"V3-05 race not closed: {r2}"
        )
    finally:
        try:
            wrapper.terminate()
            wrapper.wait(timeout=5)
        except subprocess.TimeoutExpired:
            wrapper.kill()
