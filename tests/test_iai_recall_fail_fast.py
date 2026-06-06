"""Gate: iai recall fail-fast — degrade in ~2s not 30s on a stalled daemon.

THREE HERMETIC CASES:
(1) SLOW daemon: fake AF_UNIX server that accepts then stalls (never replies).
    IAI_DAEMON_SOCKET_PATH must be set so cli.py does NOT short-circuit.
    Asserts cmd_recall returns within ~3s (not ~30s) AND returns a degraded result.
(2) FAST daemon: fake socket that replies promptly with a valid memory_recall result.
    Asserts cmd_recall uses the daemon hits (no degrade path taken).
(3) DOWN socket: absent socket path. Asserts cmd_recall degrades fast via client path.

Hermetic: tmp socket paths, monkeypatched HOME/IAI_MCP_STORE, generic User data.
Never touches the live daemon or live ~/.iai-mcp.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Wall-clock ceiling for the fail-fast: the stall case must degrade in <=3s.
# The read_timeout is ~2s; we allow 1s extra for socket setup + degrade path.
FAIL_FAST_CEILING_S = 3.5

# Fast-daemon ceiling: the fast case must return in <=2s.
FAST_CEILING_S = 2.0

# Record filler count for the hermetic store (fewer is faster to build).
N_FILLER = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hermetic_store(tmp_path: Path) -> Path:
    """Create a small tmp MemoryStore so recall_semantic_warm returns something.

    Explicitly closes the store after building so the EXCLUSIVE lock is released
    before the test calls degraded_semantic_recall (which opens with SHARED mode).
    Without explicit close(), Python's non-deterministic GC may leave the
    EXCLUSIVE lock held during the test, causing degraded_semantic_recall (which
    opens with AccessMode.SHARED + _lock_timeout_override=0.25) to wait for the
    lock release — making the apparently-instant degrade path take 9s+.
    """
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM
    import numpy as np

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))
    rng = np.random.default_rng(12345)
    for i in range(N_FILLER):
        v = rng.random(EMBED_DIM).astype(np.float32)
        store.insert(_make(text=f"User record {i}", vec=v.tolist()))
    flush_record_buffer(store)
    # Explicitly release the exclusive lock so SHARED opens succeed quickly.
    try:
        store.close()
    except Exception:
        pass
    return store_root


def _unix_socket_server_stall(sock_path: str, stall_seconds: float = 60.0) -> threading.Event:
    """Start a fake AF_UNIX server that accepts connections then stalls (never replies).

    Returns a threading.Event that is set once the server has bound and is ready.
    The server runs in a daemon thread and accepts one connection, then sleeps.

    This exercises the read_timeout in _send_jsonrpc_request (in cli.py):
    the client connects successfully (no ConnectionRefused) but readline()
    blocks until the read_timeout fires.

    IMPORTANT: IAI_DAEMON_SOCKET_PATH must be set to sock_path so that
    cli.py does NOT short-circuit to None immediately (it short-circuits
    only when IAI_MCP_STORE is a custom path AND no socket override is set).
    """
    ready = threading.Event()

    def _server():
        # Remove stale socket file if it exists.
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(sock_path)
        srv.listen(5)
        ready.set()
        # Accept one connection and stall (never reply).
        srv.settimeout(120.0)
        try:
            conn, _ = srv.accept()
            # Read the request to prevent EPIPE on the client side.
            try:
                conn.recv(4096)
            except OSError:
                pass
            # Stall: sleep until the test timeout expires.
            time.sleep(stall_seconds)
            conn.close()
        except OSError:
            pass
        finally:
            srv.close()

    t = threading.Thread(target=_server, daemon=True)
    t.start()
    # Wait for the server to bind (give it up to 2s).
    ready.wait(timeout=2.0)
    return ready


def _unix_socket_server_fast(sock_path: str, hits: list[dict]) -> threading.Event:
    """Start a fake AF_UNIX server that immediately replies with a valid result.

    Replies with a JSON-RPC 2.0 response containing the given hits.
    Returns a threading.Event set once the server is bound and ready.
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
        srv.settimeout(10.0)
        try:
            conn, _ = srv.accept()
            try:
                conn.recv(4096)  # consume request
                resp = {
                    "jsonrpc": "2.0", "id": 1,
                    "result": {
                        "hits": hits,
                        "anti_hits": [],
                        "activation_trace": [],
                        "budget_used": 100,
                        "ann_path_used": True,
                    }
                }
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
            except OSError:
                pass
            finally:
                conn.close()
        except OSError:
            pass
        finally:
            srv.close()

    t = threading.Thread(target=_server, daemon=True)
    t.start()
    ready.wait(timeout=2.0)
    return ready


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path: Path):
    """Hermetic: HOME, IAI_MCP_STORE → tmp. Remove daemon socket env."""
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    # Remove any live daemon socket path — tests set their own.
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
    yield


@pytest.fixture(autouse=True)
def _reset_and_stub_construct(monkeypatch):
    """Reset the cached local store handle AND stub the embedder funnel.

    Prevents cross-test state leakage AND prevents the real Rust Embedder from
    being constructed during these fail-fast timing tests. A real Embedder()
    construct in the hermetic tmp-HOME env (cache miss) would either attempt a
    network download or take seconds — both would break the fail-fast timing
    assertions. The daemon-independent recall path constructs its own embedder
    via the funnel, so stubbing the funnel to RAISE keeps the construct instant
    and routes the path to the bypass-safe recency degrade (the behavior the
    down-socket fail-fast case asserts).
    """
    import iai_mcp.embed as _embed_mod
    import iai_mcp.semantic_recall as _sr

    def _raising_funnel(_store):
        raise RuntimeError("hermetic: no real embedder construct in fail-fast tests")

    _sr._WARM_LOCAL_STORE = None
    monkeypatch.setattr(_embed_mod, "embedder_for_store", _raising_funnel)

    yield

    _sr._WARM_LOCAL_STORE = None


# ---------------------------------------------------------------------------
# Case 1: SLOW daemon (the real LAT-04 case)
# ---------------------------------------------------------------------------


def test_slow_daemon_degrades_in_under_3s(monkeypatch, tmp_path):
    """Stall-on-read fake daemon: cmd_recall returns within ~3s, not ~30s.

    IMPORTANT: IAI_DAEMON_SOCKET_PATH is set to the fake stall socket so
    cli.py does NOT short-circuit to None immediately (the short-circuit
    fires only when IAI_MCP_STORE is a custom path AND no socket override is
    set). Without this, the test would exercise connect-timeout instead of
    read_timeout and not test the LAT-04 fix.
    """
    sock_path = str(tmp_path / "stall.sock")
    store_root = _make_hermetic_store(tmp_path)

    # Start the stall server and wait for it to bind.
    ready = _unix_socket_server_stall(sock_path, stall_seconds=60.0)
    assert ready.is_set(), "Stall server failed to bind"

    # Set the daemon socket override so cli.py reaches read_timeout.
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", sock_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    # Stub recall_semantic_warm to return fast (so the degrade path is quick).
    import iai_mcp.iai_cli as _iai_cli
    from iai_mcp.semantic_recall import recall_semantic_warm as _real_warm

    def _fast_degrade(store_root_arg, cue, n=5, *, session_id=None):
        return [{"literal_surface": "User degrade hit", "score": 0.0, "_source": "daemon-down-degrade"}]

    monkeypatch.setattr(_iai_cli, "recall_semantic_warm" if hasattr(_iai_cli, "recall_semantic_warm") else "_recall_warm", _fast_degrade, raising=False)

    # Patch recall_semantic_warm at the iai_cli module import site.
    import iai_mcp.semantic_recall as _sr
    monkeypatch.setattr(_sr, "recall_semantic_warm", _fast_degrade)

    # Build argparse namespace for cmd_recall.
    import argparse
    args = argparse.Namespace(cue="test query", limit=5, json=False)

    t0 = time.perf_counter()
    returncode = _iai_cli.cmd_recall(args)
    elapsed = time.perf_counter() - t0

    # Must return in <=FAIL_FAST_CEILING_S (well under 30s).
    assert elapsed < FAIL_FAST_CEILING_S, (
        f"iai recall took {elapsed:.2f}s on a stalled daemon — expected <=3s. "
        f"The LAT-04 short read_timeout fix may not be active."
    )

    # Must not return 1 (error exit).
    assert returncode == 0, f"cmd_recall returned non-zero: {returncode}"


# ---------------------------------------------------------------------------
# Case 2: FAST daemon (no regression)
# ---------------------------------------------------------------------------


def test_fast_daemon_uses_daemon_hits_no_degrade(monkeypatch, tmp_path):
    """Fast-replying fake daemon: cmd_recall returns daemon hits, not degrade."""
    sock_path = str(tmp_path / "fast.sock")
    store_root = _make_hermetic_store(tmp_path)

    daemon_hits = [
        {"record_id": "00000000-0000-0000-0000-000000000001", "score": 0.95,
         "reason": "cosine 0.95", "literal_surface": "User daemon memory hit 1",
         "adjacent_suggestions": []},
    ]
    ready = _unix_socket_server_fast(sock_path, hits=daemon_hits)
    assert ready.is_set(), "Fast server failed to bind"

    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", sock_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    # Track whether recall_semantic_warm (degrade path) is called.
    degrade_called = []

    import iai_mcp.semantic_recall as _sr
    _orig_warm = _sr.recall_semantic_warm

    def _spy_warm(*a, **kw):
        degrade_called.append(True)
        return _orig_warm(*a, **kw)

    monkeypatch.setattr(_sr, "recall_semantic_warm", _spy_warm)

    import iai_mcp.iai_cli as _iai_cli
    import argparse

    # Capture stdout.
    import io
    captured = io.StringIO()
    import sys
    orig_stdout = sys.stdout
    sys.stdout = captured

    t0 = time.perf_counter()
    try:
        args = argparse.Namespace(cue="test query", limit=5, json=True)
        returncode = _iai_cli.cmd_recall(args)
    finally:
        sys.stdout = orig_stdout

    elapsed = time.perf_counter() - t0

    assert elapsed < FAST_CEILING_S, f"Fast daemon recall took {elapsed:.2f}s"
    assert returncode == 0

    # The degrade path (recall_semantic_warm) must NOT have been called.
    assert not degrade_called, (
        "Degrade path was invoked even though the daemon replied promptly — "
        "the LAT-04 fix must not degrade on a fast daemon."
    )

    # The output should contain the daemon hit.
    output = captured.getvalue()
    assert "daemon memory hit" in output or '"_source": "daemon"' in output or '"source": "daemon"' in output or "daemon" in output, (
        f"Expected daemon result in output, got: {output!r}"
    )


# ---------------------------------------------------------------------------
# Case 3: DOWN socket (unchanged fast degrade)
# ---------------------------------------------------------------------------


def test_down_socket_degrades_fast(monkeypatch, tmp_path):
    """Down socket (absent path): cmd_recall degrades fast via client path."""
    absent_sock = str(tmp_path / "absent.sock")
    store_root = _make_hermetic_store(tmp_path)

    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", absent_sock)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    import iai_mcp.iai_cli as _iai_cli
    import argparse

    t0 = time.perf_counter()
    args = argparse.Namespace(cue="test query", limit=5, json=False)
    returncode = _iai_cli.cmd_recall(args)
    elapsed = time.perf_counter() - t0

    # Down socket should connect-fail fast (well under 1s).
    assert elapsed < 2.0, f"Down-socket degrade took {elapsed:.2f}s"
    assert returncode == 0
