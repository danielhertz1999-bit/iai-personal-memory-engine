"""RED-by-design + companion hermetic test for the asleep-detection skip in cmd_recall.

TWO CASES:

SLEEP-skip case (RED): When lifecycle_state.json records SLEEP with a
``since_ts`` well in the past (confidently asleep, past any freshness margin),
cmd_recall should detect asleep and skip the ~2s RPC, going straight to the
in-process construct.  TODAY it always pays the RPC first → wall-time ~2s →
the <1.5s assertion FAILS (correct RED).  Goes GREEN once the asleep-detection
skip ships.

Fail-fast preservation companion (GREEN): When lifecycle_state.json shows WAKE
(or is absent), cmd_recall must STILL pay the ~2s fail-fast probe against the
stalling socket.  This is the regression guard: the SLEEP-skip must fire ONLY
on confident SLEEP, never on WAKE.  This case passes today and must keep passing
after the fix ships.

Asleep-detection contract (for the fix to honor):
- Read the lifecycle file from the resolved store root:
  ``{IAI_MCP_STORE or ~/.iai-mcp}/lifecycle_state.json``
- Only skip when ``current_state in {SLEEP, HIBERNATION}`` AND
  ``since_ts`` is older than ``_ASLEEP_MARGIN_SEC`` (a small freshness window).
  The test writes ``since_ts`` 60 s in the past, implying
  ``_ASLEEP_MARGIN_SEC < 60``.
- Any read failure (absent / corrupt / parse error) → fall through to RPC
  (status quo, never hard-fail).
- IAI_RECALL_READ_TIMEOUT env var is honored (the test uses a short timeout
  so the stall case degrades quickly, consistent with the test ceiling).

Socket routing:
  IAI_DAEMON_SOCKET_PATH must be set for cli.py to reach the fake stalling
  socket (without it, cli.py short-circuits on custom IAI_MCP_STORE →
  returns None immediately, bypassing the read_timeout entirely).

Hermetic: tmp HOME + IAI_MCP_STORE + short system-temp socket path, generic
data, no live-daemon touch.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make


# ---------------------------------------------------------------------------
# Wall-clock ceilings
# ---------------------------------------------------------------------------

# The SLEEP-skip case must return well UNDER the ~2s stall timeout.
SLEEP_SKIP_CEILING_S = 1.5

# The fail-fast preservation companion must still stall ~2s.
# Lower bound: must actually wait (> 1.8s so we're past the read_timeout).
# Upper bound: stall + degrade path overhead (generous ceiling).
WAKE_FAILFAST_LOWER_S = 1.8
WAKE_FAILFAST_CEILING_S = 3.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FILLER = 5


def _make_hermetic_store(tmp_path: Path) -> Path:
    """Build a small MemoryStore with a brain.sqlite3 so recall_semantic_warm
    has something to open.  Closes explicitly to release EXCLUSIVE lock before
    the test's degraded-recall path (which opens SHARED with a short timeout).
    """
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM
    import numpy as np

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))
    rng = np.random.default_rng(42)
    for i in range(N_FILLER):
        v = rng.random(EMBED_DIM).astype(np.float32)
        store.insert(_make(text=f"User record {i}", vec=v.tolist()))
    flush_record_buffer(store)
    try:
        store.close()
    except Exception:
        pass
    return store_root


def _short_sock_path() -> str:
    """Return a short AF_UNIX path in system temp (avoids macOS 104-char sun_path limit).

    A file in tempfile.gettempdir() is always short enough for AF_UNIX bind.
    """
    fd, path = tempfile.mkstemp(prefix="iai_asleep_", suffix=".sock", dir=tempfile.gettempdir())
    os.close(fd)
    os.unlink(path)  # mkstemp creates the file; remove so bind() can create a socket
    return path


def _start_stall_server(sock_path: str, stall_seconds: float = 60.0) -> threading.Event:
    """Start a fake AF_UNIX server that accepts then stalls (never replies).

    Returns a threading.Event set once the server is bound and ready.
    The client connects successfully but readline() blocks until the timeout fires.
    """
    ready = threading.Event()

    def _server():
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(sock_path)
        srv.listen(5)
        ready.set()
        srv.settimeout(120.0)
        try:
            conn, _ = srv.accept()
            try:
                conn.recv(4096)
            except OSError:
                pass
            time.sleep(stall_seconds)
            conn.close()
        except OSError:
            pass
        finally:
            srv.close()

    t = threading.Thread(target=_server, daemon=True)
    t.start()
    ready.wait(timeout=2.0)
    return ready


def _write_lifecycle_state(store_root: Path, state: str, seconds_ago: float) -> None:
    """Write lifecycle_state.json under store_root with a past since_ts.

    Args:
        store_root: the IAI_MCP_STORE path (lifecycle file lives here directly).
        state: e.g. "SLEEP", "WAKE", "DROWSY", "HIBERNATION".
        seconds_ago: how many seconds in the past to set since_ts.
    """
    from iai_mcp.lifecycle_state import save_state, default_state

    lc_path = store_root / "lifecycle_state.json"
    record = dict(default_state())
    record["current_state"] = state
    since = (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()
    record["since_ts"] = since
    record["last_activity_ts"] = since
    save_state(record, path=lc_path)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hermetic env + embedder stub fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path: Path):
    """Redirect HOME + IAI_MCP_STORE to tmp; remove live socket env."""
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
    yield


@pytest.fixture(autouse=True)
def _stub_embedder(monkeypatch):
    """Stub the embedder funnel to raise so the recall path degrades instantly.

    Under tmp HOME the real HF model is not reachable.  Stubbing to raise
    routes cmd_recall through the instant recency floor (daemon-down-degrade,
    non-empty) without a real construct.  Timing is the load-bearing RED
    assertion; this stub keeps tests fast.
    """
    import iai_mcp.embed as _embed_mod
    import iai_mcp.semantic_recall as _sr

    def _raising_funnel(_store):
        raise RuntimeError("hermetic: no real embedder in asleep-skip tests")

    _sr._WARM_LOCAL_STORE = None
    monkeypatch.setattr(_embed_mod, "embedder_for_store", _raising_funnel)
    yield
    _sr._WARM_LOCAL_STORE = None


# ---------------------------------------------------------------------------
# SLEEP-skip case: confidently-SLEEP state → must skip the ~2s RPC (RED today)
# ---------------------------------------------------------------------------

def test_sleep_skip_avoids_2s_rpc(monkeypatch, tmp_path):
    """Confidently-SLEEP lifecycle_state causes cmd_recall to skip the RPC.

    Today cmd_recall always issues the ~2s read-timeout RPC regardless of
    lifecycle state → wall-time ~2s → FAILS the <1.5s assertion.
    Goes GREEN once the asleep-detection short-circuit ships.
    """
    store_root = _make_hermetic_store(tmp_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    # Write a SLEEP lifecycle state with since_ts well in the past (60s ago),
    # so it exceeds any reasonable freshness margin.
    # (implies _ASLEEP_MARGIN_SEC < 60 in the production fix)
    _write_lifecycle_state(store_root, "SLEEP", seconds_ago=60.0)

    # Start the stall server at a SHORT system-temp path (macOS 104-char limit).
    sock_path = _short_sock_path()
    ready = _start_stall_server(sock_path, stall_seconds=60.0)
    assert ready.is_set(), "Stall server failed to bind"

    # IAI_DAEMON_SOCKET_PATH must be set so cli.py does NOT short-circuit.
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", sock_path)

    # Short read-timeout bounds the stall case wall time.
    # The fix should skip the RPC entirely → wall < SLEEP_SKIP_CEILING_S.
    monkeypatch.setenv("IAI_RECALL_READ_TIMEOUT", "1.5")

    import iai_mcp.iai_cli as _iai_cli

    args = argparse.Namespace(cue="test recall cue", limit=5, json=True)

    t0 = time.perf_counter()
    _iai_cli.cmd_recall(args)
    elapsed = time.perf_counter() - t0

    # RED assertion: today this fails because the RPC is issued (wall ≈ 1.5s+).
    # Once the asleep-detection skip ships, wall < SLEEP_SKIP_CEILING_S.
    assert elapsed < SLEEP_SKIP_CEILING_S, (
        f"iai recall against a confidently-SLEEP daemon took {elapsed:.2f}s "
        f"(expected < {SLEEP_SKIP_CEILING_S}s after asleep-skip). "
        "Today this fails because the ~2s RPC is still issued. "
        "Goes GREEN once the lifecycle-state-based skip ships."
    )


# ---------------------------------------------------------------------------
# Fail-fast preservation companion: WAKE state → still pays ~2s probe (GREEN)
# ---------------------------------------------------------------------------

def test_wake_state_still_pays_failfast_rpc(monkeypatch, tmp_path):
    """WAKE lifecycle_state preserves the fail-fast ~2s probe (not skipped).

    Regression guard: the SLEEP-skip must fire ONLY when the daemon is
    confidently SLEEP.  A WAKE state must still issue the RPC and degrade
    via the ~2s read-timeout path.

    This case PASSES today and must KEEP passing after the fix ships.
    """
    store_root = _make_hermetic_store(tmp_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    # Write a WAKE lifecycle state — the skip must NOT fire.
    _write_lifecycle_state(store_root, "WAKE", seconds_ago=10.0)

    sock_path = _short_sock_path()
    ready = _start_stall_server(sock_path, stall_seconds=60.0)
    assert ready.is_set(), "Stall server failed to bind"

    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", sock_path)
    monkeypatch.setenv("IAI_RECALL_READ_TIMEOUT", "2.0")

    import iai_mcp.iai_cli as _iai_cli

    args = argparse.Namespace(cue="wake regression check", limit=5, json=True)

    t0 = time.perf_counter()
    _iai_cli.cmd_recall(args)
    elapsed = time.perf_counter() - t0

    # Must have paid the RPC (stalled for the read_timeout).
    assert elapsed >= WAKE_FAILFAST_LOWER_S, (
        f"WAKE state recall returned in {elapsed:.2f}s — too fast; "
        "the ~2s RPC should still be issued for a WAKE daemon. "
        f"Expected >= {WAKE_FAILFAST_LOWER_S}s (read_timeout = 2.0s)."
    )
    assert elapsed < WAKE_FAILFAST_CEILING_S, (
        f"WAKE state recall took {elapsed:.2f}s — too slow; expected < {WAKE_FAILFAST_CEILING_S}s."
    )
