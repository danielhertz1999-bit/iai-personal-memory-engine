"""RED scaffolds for REQ-1: direct no-flock recency read, daemon-free, ≤1.5 s.

Validation rows: F6 (latency SLO), partial F2 (no daemon gating), REQ-1.

Both tests are xfail(strict=True) because the direct-read primary path does not
yet exist. The production code currently routes every recency read through the
daemon socket; the daemon-down fallback reads only the live deferred-captures
layer, not the Hippo store. These tests will flip from xfail to pass when the
direct LOCK_SH recency read path is wired.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user_turn(text: str = "generic user turn"):
    """Return a minimal episodic role:user MemoryRecord with a zero-vector embedding.

    The zero vector is valid here because REQ-1 recency reads are embedding-
    independent — recency never calls into hnswlib.
    """
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.0] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": "test-session", "role": "user"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["role:user"],
        language="en",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recency_read_daemon_up_steady(hermetic_store: Path) -> None:
    """F6 / REQ-1: a stored turn is returned by the direct recency path in ≤1.5 s.

    Constructs a HippoDB on the hermetic store, inserts a role:user turn,
    then asserts the *direct* primary recency path (not the daemon-socket
    path) returns that turn within the 1.5 s SLO.

    RED: this test imports the not-yet-existing direct_recency_read helper
    so it fails with ImportError (collection stays green; body xfails).
    """
    # Import the future direct recency helper inside the body so a collection-
    # time ImportError does not prevent other tests from being collected.
    from iai_mcp.store import MemoryStore, flush_record_buffer

    # Import the not-yet-wired direct primary read path. This import will
    # raise ImportError until the direct path is implemented, which is the
    # correct RED failure mode.
    from iai_mcp.direct_recency import read_recent_user_turns_direct  # type: ignore[import]

    store = MemoryStore(hermetic_store)
    try:
        rec = _make_user_turn("direct recency probe text")
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    t0 = time.monotonic()
    turns = read_recent_user_turns_direct(hermetic_store, n=5)
    elapsed = time.monotonic() - t0

    assert elapsed <= 1.5, f"direct recency read took {elapsed:.3f} s (SLO ≤1.5 s)"
    surfaces = [t.literal_surface for t in turns]
    assert any("direct recency probe text" in s for s in surfaces), (
        "stored turn not found in direct recency results"
    )


def test_recency_read_daemon_down_sigkill(hermetic_store: Path, tmp_path: Path) -> None:
    """F6 / REQ-1: direct recency read survives a missing -shm file (SIGKILL residue).

    A non-clean daemon exit leaves brain.sqlite3-shm absent. Opening with
    mode=ro raises an error (SQLite READONLY_CANTINIT); the direct path must
    open with mode=memory (WAL + read-write) so it can create a fresh shm.
    This test asserts the turn is still returned in ≤1.5 s.

    RED: imports the not-yet-existing direct_recency helper.
    """
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.direct_recency import read_recent_user_turns_direct  # type: ignore[import]

    # Seed the store.
    store = MemoryStore(hermetic_store)
    try:
        rec = _make_user_turn("sigkill survival probe text")
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    # Simulate a SIGKILL residue: remove the WAL shm file if it exists.
    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    shm_path = Path(str(db_path) + "-shm")
    if shm_path.exists():
        shm_path.unlink()

    t0 = time.monotonic()
    turns = read_recent_user_turns_direct(hermetic_store, n=5)
    elapsed = time.monotonic() - t0

    assert elapsed <= 1.5, f"daemon-down recency read took {elapsed:.3f} s (SLO ≤1.5 s)"
    surfaces = [t.literal_surface for t in turns]
    assert any("sigkill survival probe text" in s for s in surfaces), (
        "stored turn not found after simulated SIGKILL residue"
    )
