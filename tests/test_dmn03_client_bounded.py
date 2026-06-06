"""A client read never blocks indefinitely on a store-owner-held lock.

The store owner (a long-lived process) holds the cross-process EXCLUSIVE
lock on ``hippo/.lock`` while it works (e.g. during consolidation). A
separate client process that wants to read MUST NOT block forever waiting
for that lock — it gets a bounded SHARED wait (the absolute guard is
``_SHARED_LOCK_TIMEOUT_S`` = 1.45 s, well under the 1.5 s SLO) and then
either acquires or raises a recoverable error. The client then degrades
(direct-read override / bank fallback) rather than hanging.

flock is per-process: a same-process EXCLUSIVE holder would make a
same-process SHARED open raise immediately (``same-process-holds-EXCLUSIVE``),
which would NOT exercise the cross-process bounded-wait path. So the
EXCLUSIVE holder here is a SEPARATE process that acknowledges (via a sentinel)
that it holds the lock before the client attempts its SHARED open. The
assertion is wall-clock: the client open returns OR raises within a generous
finite bound, and crucially never hangs past it.

Hermetic: tmp store, a spawned holder PID with explicit env, no real store
or daemon touched.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from iai_mcp.hippo import (
    AccessMode,
    ConsolidationPendingError,
    HippoDB,
    HippoLockHeldError,
    _SHARED_LOCK_TIMEOUT_S,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


_TEST_PASSPHRASE = "iai-mcp-test-passphrase-2026-04-30-phase-07.10"
_HOLDER_BARRIER_TIMEOUT_S = 60.0


# Holder program: opens the store EXCLUSIVE (acquiring the cross-process
# LOCK_EX), signals that it holds the lock via a sentinel, then sleeps so the
# lock stays held while the parent's client attempts its bounded SHARED open.
_HOLDER_PROGRAM = r"""
import os, sys, time

src = os.environ["IAI_MCP_TEST_SRC"]
if src not in sys.path:
    sys.path.insert(0, src)

from iai_mcp.store import MemoryStore

store_root = os.environ["IAI_MCP_STORE"]
sentinel = os.environ["IAI_MCP_TEST_SENTINEL"]

# EXCLUSIVE open == holds the cross-process LOCK_EX on hippo/.lock.
store = MemoryStore(path=store_root)

with open(sentinel, "w") as fh:
    fh.write("held")
    fh.flush()
    os.fsync(fh.fileno())

# Hold the lock well past the client's bounded wait; parent SIGKILLs us.
time.sleep(120)
"""


def _seed_one_record(store_root: Path) -> None:
    store = MemoryStore(path=store_root)
    try:
        now = datetime.now(timezone.utc)
        store.insert(
            MemoryRecord(
                id=uuid4(),
                tier="episodic",
                literal_surface="alice baseline record for the bounded-client check",
                aaak_index="",
                embedding=[0.1] * EMBED_DIM,
                community_id=None,
                centrality=0.0,
                detail_level=2,
                pinned=False,
                stability=0.0,
                difficulty=0.0,
                last_reviewed=None,
                never_decay=False,
                never_merge=False,
                provenance=[],
                created_at=now,
                updated_at=now,
                tags=["baseline"],
                language="en",
            )
        )
    finally:
        store.close()


def _spawn_exclusive_holder(store_root: Path, sentinel: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_MCP_TEST_SENTINEL"] = str(sentinel)
    env["IAI_MCP_TEST_SRC"] = str(Path(__file__).resolve().parent.parent / "src")
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = _TEST_PASSPHRASE
    return subprocess.Popen(
        [sys.executable, "-c", _HOLDER_PROGRAM],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_client_shared_read_is_bounded_under_held_lock():
    """A SHARED client open returns or raises within the bound — never hangs.

    A separate process holds the EXCLUSIVE lock; the client's SHARED open must
    complete (acquire OR raise a recoverable lock error) inside a finite
    wall-clock bound comfortably above the 1.45 s SHARED guard — not block
    indefinitely. A hang would fail this test by exceeding the bound.
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="iai-bounded-client-"))
    sentinel = tmp_root / ".holder-held"
    holder: subprocess.Popen | None = None
    try:
        # Seed + initialise the store, then release so the holder can take EX.
        _seed_one_record(tmp_root)

        holder = _spawn_exclusive_holder(tmp_root, sentinel)
        deadline = time.monotonic() + _HOLDER_BARRIER_TIMEOUT_S
        while not sentinel.exists():
            if holder.poll() is not None:
                _out, err = holder.communicate()
                raise AssertionError(
                    "exclusive holder exited before signalling "
                    f"(rc={holder.returncode}); stderr=\n{err.decode(errors='replace')}"
                )
            if time.monotonic() >= deadline:
                raise AssertionError("exclusive holder never signalled lock-held")
            time.sleep(0.01)

        # The holder now owns the cross-process LOCK_EX. A SHARED client open
        # must NOT hang: it returns (acquired) OR raises a recoverable lock
        # error within the bound. Generous ceiling: the 1.45 s SHARED guard
        # plus slack — comfortably under ~3 s.
        bound_s = max(3.0, _SHARED_LOCK_TIMEOUT_S + 1.5)
        t0 = time.monotonic()
        client: HippoDB | None = None
        try:
            client = HippoDB(
                tmp_root,
                access_mode=AccessMode.SHARED,
                read_only=True,
            )
        except (ConsolidationPendingError, HippoLockHeldError):
            # Bounded-and-degraded path: the client backed off cleanly instead
            # of hanging. This is the expected DoS-avoidance behaviour.
            pass
        finally:
            if client is not None:
                client.close()
        elapsed = time.monotonic() - t0

        assert elapsed < bound_s, (
            f"client SHARED open took {elapsed:.3f}s (>= bound {bound_s:.3f}s) — "
            "the client blocked instead of degrading within the bound"
        )
    finally:
        if holder is not None:
            try:
                os.kill(holder.pid, signal.SIGKILL)
                holder.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)


def test_shared_lock_timeout_constant_is_bounded():
    """The SHARED-wait guard is a finite sub-1.5 s bound (the SLO)."""
    assert 0 < _SHARED_LOCK_TIMEOUT_S < 1.5
