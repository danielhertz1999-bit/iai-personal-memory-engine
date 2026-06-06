"""Acceptance: concurrent wrapper cold-start regression trap.

The regression-trap test for N parallel wrapper cold-starts when no daemon
exists.

Contract:
    - PASSES with the launchd-managed listener:
      bridge.ts is a pure connector -> all 5 wrappers connect
      to the SAME launchd-pre-bound socket -> launchd spawns the daemon
      ONCE in response to the first connection -> all 5 wrappers share it.
    - FAILS deterministically on the spawn-fallback baseline:
      bridge.ts spawn-fallback wins the TOCTOU race for multiple wrappers,
      2-5 daemons end up bound, the singleton assertion fires.

This test provides runtime invariant coverage, not just architectural code
coverage.

Test isolation: a per-test LaunchAgent with a unique Label
``com.iai-mcp.daemon.test-<pid>-<tmp_id>`` is rendered into ``tmp_path/
Library/LaunchAgents/`` (NOT the user's real ``~/Library/LaunchAgents/``,
to avoid pollution if teardown is interrupted) and loaded via
``launchctl load -w``. The test socket lives under
``/tmp/iai-cspawn-<pid>-<tmp_id>/d.sock`` (within macOS's 104-byte
AF_UNIX path cap). Teardown unloads the agent, removes the plist, kills
any spawned test daemon (env-filtered to never touch the user's real
production daemon), and removes the socket.

Total runtime: ~25-30s (5 staggered cold-starts + 15s settle + readline
poll). Override with ``IAI_MCP_SKIP_LAUNCHCTL_TESTS=1`` to skip.

This module is macOS-only (LaunchAgent + launchctl). Skipped on Linux/Windows.
"""
from __future__ import annotations

import json
import os
import platform
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="LaunchAgent + launchctl is macOS-only",
)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_wrapper() -> Path:
    """Build the TS wrapper once per test module; reuse across tests.

    Same pattern as ``tests/test_socket_subagent_reuse.py:built_wrapper``.
    """
    if not (WRAPPER / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
    subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = WRAPPER / "dist" / "index.js"
    assert dist.exists(), "npm run build should have produced dist/index.js"
    return dist


@pytest.fixture
def test_launchagent(tmp_path):
    """Render + load a tmp LaunchAgent against an isolated test socket path.

    The plist is written into ``tmp_path/Library/LaunchAgents/`` (NOT the
    user's real ``~/Library/LaunchAgents/``) so any teardown failure leaves
    no pollution under the user's home directory. ``launchctl load -w``
    accepts any absolute plist path; the loaded agent is identified
    internally by its ``Label`` value, which is unique per-test
    (PID + ``tmp_path`` id).

    Injects ``IAI_DAEMON_SOCKET_PATH`` into the plist's EnvironmentVariables
    so the daemon carries it.  The launchd socket-activation path ignores
    this env value (uses fd 3 for binding), but the daemon itself reads it
    and it keeps the wrapper env consistent with the plist.  Teardown uses
    lsof on the specific tmp socket path to find and kill the test daemon —
    setproctitle-proof, never touches the production daemon (which binds
    ``~/.iai-mcp/.daemon.sock``, a different path).

    Yields: ``(sock_path, plist_path, label, env, store_dir)`` -- env is
    suitable for spawning wrappers via subprocess.Popen; store_dir is the
    isolated Hippo store root used to identify the test daemon.
    """
    if os.environ.get("IAI_MCP_SKIP_LAUNCHCTL_TESTS") == "1":
        pytest.skip("IAI_MCP_SKIP_LAUNCHCTL_TESTS=1")

    # Use /tmp/ for the socket directory (macOS AF_UNIX 104-byte path cap;
    # tmp_path under /private/var/folders/... is too long for some labels).
    sock_dir = Path(f"/tmp/iai-cspawn-{os.getpid()}-{id(tmp_path) & 0xFFFFFF:x}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    if sock_path.exists():
        sock_path.unlink()

    # Isolated store root for the test daemon. Hippo acquires an exclusive
    # fcntl lock on <store>/hippo/.lock; without this redirect the test
    # daemon would try to open ~/.iai-mcp and fail with HippoLockHeldError
    # whenever the production daemon is running. store_dir lives under
    # tmp_path (pytest auto-cleans it; no path-length constraint here).
    store_dir = tmp_path / "store"
    store_dir.mkdir(parents=True, exist_ok=True)

    label = f"com.iai-mcp.daemon.test-{os.getpid()}-{id(tmp_path) & 0xFFFFFF:x}"

    # Render plist under tmp_path/Library/LaunchAgents/ (NOT the user's
    # real ~/Library/LaunchAgents/ -- avoids pollution if teardown is
    # interrupted on a dev box where the production daemon is OFF).
    plist_dir = tmp_path / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / f"{label}.plist"

    # Read template and substitute placeholders. Then:
    #   1. Replace the production label string ONLY at the
    #      <key>Label</key> binding site (anchor on the surrounding
    #      <string>...</string> so we don't accidentally rewrite the
    #      docstring comment block at the top, which mentions the
    #      production label by name).
    #   2. Replace the production socket path with the test socket path.
    #   3. Inject IAI_DAEMON_SOCKET_PATH and PYTHONPATH into
    #      EnvironmentVariables. IAI_DAEMON_SOCKET_PATH is harmless to the
    #      launchd path (which uses fd 3 for socket activation, ignoring the
    #      env value for binding). Previously this tag was needed by the
    #      psutil-environ teardown filter; teardown now uses lsof on the
    #      socket path instead (setproctitle-proof), but the injection is
    #      kept for the daemon's own use and for symmetry with the wrapper
    #      env below.
    #   4. Inject IAI_MCP_STORE so the daemon opens an isolated Hippo
    #      store under tmp_path. Without this, the test daemon races the
    #      production daemon for the exclusive fcntl lock on
    #      ~/.iai-mcp/hippo/.lock and crashes with HippoLockHeldError
    #      whenever production is running.
    template = (REPO / "scripts" / "com.iai-mcp.daemon.plist.template").read_text()
    label_old_xml = "<string>com.iai-mcp.daemon</string>"
    label_new_xml = f"<string>{label}</string>"
    if template.count(label_old_xml) != 1:
        pytest.fail(
            f"plist template invariant broken: expected exactly one "
            f"<string>com.iai-mcp.daemon</string> occurrence (the "
            f"<key>Label</key> binding); found "
            f"{template.count(label_old_xml)}",
        )
    rendered = (
        template
        .replace("{PYTHON_PATH}", sys.executable)
        .replace("{HOME}", str(Path.home()))
        .replace(label_old_xml, label_new_xml)
        .replace(
            f"{Path.home()}/.iai-mcp/.daemon.sock",
            str(sock_path),
        )
        .replace(
            "<key>IAI_MCP_LAUNCHD_MANAGED</key>\n    <string>1</string>",
            "<key>IAI_MCP_LAUNCHD_MANAGED</key>\n    <string>1</string>\n"
            f"    <key>IAI_DAEMON_SOCKET_PATH</key>\n    <string>{sock_path}</string>\n"
            f"    <key>PYTHONPATH</key>\n    <string>{REPO / 'src'}</string>\n"
            f"    <key>IAI_MCP_STORE</key>\n    <string>{store_dir}</string>\n"
            "    <key>IAI_MCP_CRYPTO_PASSPHRASE</key>\n    <string>test-cspawn-key</string>",
        )
    )
    plist_path.write_text(rendered)

    # Pre-clean (idempotent). Ignore any "not loaded" errors.
    subprocess.run(
        ["launchctl", "unload", "-w", str(plist_path)],
        capture_output=True, check=False,
    )

    # Load the test LaunchAgent.
    res = subprocess.run(
        ["launchctl", "load", "-w", str(plist_path)],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        # Common causes: TCC denial on macOS Sequoia/Sonoma, missing
        # /Library/LaunchAgents permission, plist syntax error.
        pytest.skip(f"launchctl load failed (rc={res.returncode}): {res.stderr.strip()}")

    # Verify registration. If load returned 0 but the label is missing,
    # something is off -- fail rather than silently skip.
    list_res = subprocess.run(
        ["launchctl", "list"], capture_output=True, text=True, check=False,
    )
    if label not in list_res.stdout:
        subprocess.run(
            ["launchctl", "unload", "-w", str(plist_path)],
            capture_output=True, check=False,
        )
        pytest.fail(
            f"LaunchAgent {label!r} not present in `launchctl list` after load",
        )

    env = {
        **os.environ,
        "IAI_MCP_PYTHON": sys.executable,
        "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "IAI_DAEMON_SOCKET_PATH": str(sock_path),
        "IAI_MCP_STORE": str(store_dir),
        "IAI_MCP_CRYPTO_PASSPHRASE": "test-cspawn-key",
    }

    try:
        yield sock_path, plist_path, label, env, store_dir
    finally:
        # Teardown: unload, kill any spawned test daemon, remove socket file.
        # The plist itself lives under tmp_path which pytest cleans up.
        subprocess.run(
            ["launchctl", "unload", "-w", str(plist_path)],
            capture_output=True, check=False,
        )
        # Socket-holder kill — setproctitle-proof and path-exact.
        #
        # setproctitle() clobbers the external psutil.Process.environ() view
        # on macOS, making IAI_DAEMON_SOCKET_PATH invisible to env-based
        # filters after the title is set.  lsof on the specific tmp socket
        # path is immune to setproctitle and scopes precisely to this test's
        # daemon — NEVER the production daemon, which binds a different path
        # (~/.iai-mcp/.daemon.sock).  We capture PIDs from the SIGTERM pass
        # and reuse them in the SIGKILL pass so the kill targets the right
        # processes even after the daemon releases the socket fd on SIGTERM.
        target = str(sock_path)
        term_pids: set[int] = set()
        lsof_res = subprocess.run(
            ["lsof", "-U", "-F", "pn"],
            capture_output=True, text=True, check=False,
        )
        current: int | None = None
        for line in lsof_res.stdout.splitlines():
            if line.startswith("p"):
                try:
                    current = int(line[1:])
                except ValueError:
                    current = None
            elif line.startswith("n") and current is not None and line[1:] == target:
                term_pids.add(current)
        # Filter to iai_mcp.daemon cmdline (excludes launchd, which may also
        # hold a pre-bound listener fd for socket-activation).
        for pid in list(term_pids):
            try:
                cl = " ".join(psutil.Process(pid).cmdline())
                if "iai_mcp.daemon" not in cl:
                    term_pids.discard(pid)
                    continue
                psutil.Process(pid).send_signal(signal.SIGTERM)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                term_pids.discard(pid)
        # Brief settle, then SIGKILL any stragglers by PID (captured above).
        # We do NOT re-run lsof: SIGTERM'd daemon releases the socket fd
        # before fully exiting, so a second lsof would miss it.
        time.sleep(0.5)
        for pid in term_pids:
            try:
                if psutil.pid_exists(pid):
                    cl = " ".join(psutil.Process(pid).cmdline())
                    if "iai_mcp.daemon" in cl:
                        psutil.Process(pid).kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        try:
            sock_path.unlink()
        except (FileNotFoundError, OSError):
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _spawn_wrapper_send_initialize(
    built_wrapper: Path, env: dict,
) -> subprocess.Popen:
    """Spawn one wrapper subprocess; send MCP initialize on stdin.

    Returns the Popen handle. Caller polls stdout (with select+timeout) to
    read the initialize response after the daemon settle window expires.
    """
    proc = subprocess.Popen(
        ["node", str(built_wrapper)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "concurrent-spawn-test", "version": "0.0"},
        },
    }
    try:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(init_req) + "\n").encode("utf-8"))
        proc.stdin.flush()
    except BrokenPipeError:
        # Wrapper crashed before reading stdin; readline below will see
        # empty bytes and the test will report 0/5 successes.
        pass
    return proc


def _read_initialize_response(
    proc: subprocess.Popen, timeout_sec: float = 2.0,
) -> dict | None:
    """Poll wrapper stdout for one JSON-RPC line (the initialize response)."""
    if proc.stdout is None:
        return None
    try:
        ready, _, _ = select.select([proc.stdout], [], [], timeout_sec)
        if not ready:
            return None
        line = proc.stdout.readline()
        if not line:
            return None
        return json.loads(line.decode("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _count_daemons_for_socket(sock_path: Path, store_dir: Path) -> int:
    """Count iai_mcp.daemon processes associated with this test's store_dir.

    Two signals that are both immune to setproctitle() on macOS:

    1. setproctitle() clobbers the external psutil.Process.environ() view —
       the daemon's own os.environ is intact but external readers (like
       psutil) lose injected vars after the title is set.  We cannot use
       psutil.environ() for scoping.

    2. For launchd socket-activated daemons, the socket fd is inherited from
       launchd (passed as fd 3 via LISTEN_FDS).  lsof attributes the socket
       path to launchd, not the daemon — so lsof on sock_path yields 0 for
       the daemon process in this path.

    The reliable, activation-path-agnostic signal: the daemon opens its Hippo
    store files directly (hippo/.lock, brain.sqlite3, etc.) regardless of
    whether it was manually spawned or launchd-activated.  psutil.open_files()
    sees these and returns them with the correct on-disk path (resolving /tmp
    → /private/tmp automatically).  We intersect "iai_mcp.daemon" cmdline +
    open-files-under-store_dir so we never count the production daemon (which
    opens a different store under ~/.iai-mcp).
    """
    store_str = str(store_dir.resolve())
    count = 0
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cl = " ".join(proc.info.get("cmdline") or [])
            if "iai_mcp.daemon" not in cl:
                continue
            # Check if any open file lives inside our isolated store_dir.
            for f in proc.open_files():
                if str(Path(f.path).resolve()).startswith(store_str):
                    count += 1
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return count


def _count_binders(sock_path: Path) -> int:
    """Count distinct PIDs that hold sock_path open (lsof -U)."""
    res = subprocess.run(
        ["lsof", "-U", "-F", "pn"],
        capture_output=True, text=True, check=False,
    )
    pids: set[int] = set()
    current: int | None = None
    target = str(sock_path)
    for line in res.stdout.splitlines():
        if line.startswith("p"):
            try:
                current = int(line[1:])
            except ValueError:
                current = None
        elif line.startswith("n") and current is not None and line[1:] == target:
            pids.add(current)
    return len(pids)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_5_concurrent_wrapper_cold_starts_yield_singleton(
    built_wrapper, test_launchagent,
):
    """5 staggered cold-starts -> exactly 1 daemon after settle.

    Setup (via test_launchagent fixture):
        - Tmp LaunchAgent loaded against an isolated test socket path.
        - Plist has RunAtLoad=false. Empirically (macOS Sequoia 15.x),
          launchctl load -w for a Sockets-activated agent may spawn the
          daemon eagerly anyway -- the test tolerates this via the
          relaxed pre-condition (<= 1) and asserts the singleton
          invariant on the post-condition (== 1).

    Body:
        - Spawn 5 wrapper subprocesses with staggered start times
          (~0/50/100/150/200 ms apart). Each sends MCP initialize.
        - Wait 15s for the daemon to settle (cold-start ~8s embedder
          load + Hippo open + buffer).
        - Read each wrapper's initialize response (with 2s readline
          timeout per wrapper -- they should all be ready by t+15s).
        - Terminate wrappers (releases their connect-side fds before the
          binder count assertion).

    Assertions:
        (a) ``_count_daemons_for_socket(sock_path, store_dir) == 1`` -- exactly
            one iai_mcp.daemon process has the test store open. The
            singleton invariant (immune to setproctitle and launchd
            socket-activation path).
        (b) ``_count_binders(sock_path) <= 1`` -- lsof reports at most
            one process holding the socket file. Wrappers are clients
            of the abstract socket connection, not file-holders -- after
            their fds close they don't show up here. The launchd
            pre-bound listener is owned by launchd itself, which may
            or may not appear in lsof depending on the version.
        (c) all 5 wrapper subprocesses received a successful MCP
            initialize JSON-RPC response.

    With bridge.ts as a pure connector (no spawn-fallback): all 5 wrappers
    connect to the SAME launchd-pre-bound socket, launchd's spawn-once
    contract gives them the SAME daemon, all 3 assertions hold. THIS is what
    the test proves.

    Regression-trap caveat: the "FAILS deterministically on the spawn-fallback
    baseline" framing is platform-conditional. On macOS Sequoia 15.x,
    ``launchctl load -w`` eagerly spawns the daemon when the plist has Sockets
    defined (despite RunAtLoad=false). With the launchd-pre-bound socket
    already up and a daemon already bound, a spawn-fallback bridge.ts would
    also succeed -- its fallback would never fire because the initial connect
    succeeds. This test therefore PROVES the connector invariant cleanly (its
    primary job) but is NOT a deterministic regression trap on macOS Sequoia.
    On older macOS versions where launchctl-load defers spawn until first
    connection, the regression-trap behavior would hold.
    """
    sock_path, plist_path, label, env, store_dir = test_launchagent

    # Pre-condition: at most 1 daemon bound to this socket. RunAtLoad=false
    # in the plist is documented as "spawn lazily on first connection",
    # but on macOS Sequoia (15.x) `launchctl load -w` for a Sockets-
    # activated agent eagerly spawns the daemon despite RunAtLoad=false.
    # Empirically verified: the daemon may be PID-listed immediately
    # after `launchctl load -w` returns. This does NOT defeat the
    # singleton invariant -- it just shifts the spawn moment. The
    # critical assertion is the post-condition (`== 1` after 5 wrappers),
    # not whether the daemon was 0 or 1 before.
    initial_daemon_count = _count_daemons_for_socket(sock_path, store_dir)
    assert initial_daemon_count <= 1, (
        f"expected <= 1 daemon before test, found {initial_daemon_count} "
        f"(stale daemons from earlier test? cleanup leak?)"
    )

    # Spawn 5 wrappers staggered by ~50 ms each. Total stagger window
    # ~200 ms -- well within the launchd socket-activation race window
    # this test exercises.
    procs: list[subprocess.Popen] = []
    stagger_intervals = [0.0, 0.05, 0.05, 0.05, 0.05]
    for delay in stagger_intervals:
        if delay > 0:
            time.sleep(delay)
        procs.append(_spawn_wrapper_send_initialize(built_wrapper, env))

    # Wait 15s for the daemon to settle. Cold start = 8s embedder load
    # + Hippo open + buffer. Per advisor: do NOT shorten this -- the
    # 8s embedder cold-start is the empirical reality.
    time.sleep(15)

    # Read each wrapper's initialize response.
    init_responses: list[dict | None] = [
        _read_initialize_response(p, timeout_sec=2.0) for p in procs
    ]

    # Snapshot the singleton + binder counts BEFORE terminating wrappers.
    # Terminating may take 2s+ per wrapper; we want the assertion to fire
    # against the steady state we just observed.
    daemon_count = _count_daemons_for_socket(sock_path, store_dir)
    binder_count = _count_binders(sock_path)

    # Cleanup wrappers (release their connect-side fds; daemon still up
    # for the fixture teardown to handle).
    for proc in procs:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Assertion (a) -- THE singleton invariant.
    assert daemon_count == 1, (
        f"singleton invariant violated: {daemon_count} daemons bound to "
        f"{sock_path} after 5 concurrent wrapper cold-starts. Contract: "
        f"launchd handles the spawn-once; all wrappers join the same daemon. "
        f"The spawn-fallback baseline reproduces 2-5 daemons via a TOCTOU "
        f"race in bridge.ts."
    )
    # Assertion (b) -- file-holder confirmation. Either 0 (the socket
    # file is owned by launchd's pre-bind, not a daemon process fd entry)
    # or 1 (the spawned daemon also shows in lsof). In either case the
    # COUNT must be <= 1: 2+ would mean dueling binders.
    assert binder_count <= 1, (
        f"lsof reports {binder_count} binders for {sock_path}; "
        f"expected <= 1 (singleton)"
    )
    # Assertion (c) -- all 5 wrappers handshook successfully. A wrapper
    # that received an initialize result proves it connected to a real
    # daemon and got a real response (not just a launchd-side accept).
    success_count = sum(
        1 for r in init_responses if r is not None and "result" in r
    )
    assert success_count == 5, (
        f"only {success_count}/5 wrappers received successful initialize "
        f"response. Responses: {init_responses}"
    )


@pytest.mark.skip(
    reason="manual baseline regression check; run only against the "
    "spawn-fallback baseline to demonstrate the regression-trap behavior",
)
def test_pre_phase_7_1_baseline_fails():
    """Documentation marker: how to run against the spawn-fallback baseline.

    Manual procedure to demonstrate the regression-trap behavior:

        1. Check out a revision with the spawn-fallback bridge.ts.
        2. ``cd mcp-wrapper && npm run build``  (rebuild bridge.ts with
           the spawn-fallback restored)
        3. ``pytest tests/test_concurrent_wrapper_spawn.py::\\
           test_5_concurrent_wrapper_cold_starts_yield_singleton -v``
        4. Expected: assertion (a) FAILS with daemon_count >= 2 (the
           TOCTOU race produces multiple daemons that all bind in
           parallel before any of them notice the others).
        5. Restore the pure-connector bridge.ts.
        6. Rebuild + rerun: assertion passes.

    A maintainer who wants to re-prove the regression-trap behavior follows
    the procedure above.
    """
    pass
