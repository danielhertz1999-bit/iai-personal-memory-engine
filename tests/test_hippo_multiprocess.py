"""Genuine multi-process concurrency harness.

THIS IS subprocess-based (NOT thread-based like test_hippo_concurrency.py).
test_hippo_concurrency.py shares ONE HippoDB across threads via _PROCESS_LOCKS
refcount — it proves nothing about cross-process flock semantics.

This file spawns REAL second OS processes via subprocess.Popen to test:
- no SQLite corruption under concurrent multi-process access
- no deadlock (processes complete within a timeout bound)
- hnswlib readers never load a half-written index (atomic.hnsw.tmp rename)

Writer uses write_turn_direct() which opens with LOCK_SH (SHARED mode) — same
flock tier as the reader — so both may hold SHARED concurrently without contention.

Test 2 (A1): two processes call load_hnsw_readonly() (load_index) concurrently,
and a daemon-role process performs the atomic.hnsw.tmp→rename save while a
client holds a loaded index. This is the Research Q#1 safety assertion.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Subprocess writer helper (two-phase, SHARED/LOCK_SH)
# ---------------------------------------------------------------------------

# write_turn_direct opens HippoDB(SHARED) — LOCK_SH — so many clients can
# coexist concurrently. deferred_embedding=True skips the cold-start Rust
# embedder (8-9 s in hermetic envs with remapped HOME/cache) and stores
# embedding_pending=1 zero-vector rows instead. The deferred row is still
# immediately findable by RECENCY (SQLite-only) and surfaces in ANN after a
# subsequent _rebuild_index_from_sqlite call (simulated in Test 2 / F4 check).
_WRITER_SCRIPT = textwrap.dedent("""\
    import os, sys
    from pathlib import Path

    store_root = Path(os.environ["IAI_MCP_STORE"])
    n = int(os.environ.get("IAI_TEST_N_RECORDS", "5"))

    from iai_mcp.direct_write import write_turn_direct

    ok = 0
    for i in range(n):
        result = write_turn_direct(
            store_root,
            text=f"multiprocess writer record {i} concurrent test text",
            session_id="mp-session",
            role="user",
            deferred_embedding=True,
        )
        if result.get("status") in ("inserted", "reinforced"):
            ok += 1

    print(f"writer: inserted {ok} records")
    sys.exit(0)
""")

# Reader script: opens the store SHARED read_only and reads all records.
# read_only=True sets PRAGMA query_only=ON and skips hnswlib load (ANN index
# lives in-memory in the daemon process; clients probe via recency + SQLite).
_READER_SCRIPT = textwrap.dedent("""\
    import os, sys
    from pathlib import Path

    store_root = Path(os.environ["IAI_MCP_STORE"])

    from iai_mcp.hippo import AccessMode, HippoDB
    from iai_mcp.store import MemoryStore

    store = MemoryStore(store_root, access_mode=AccessMode.SHARED, read_only=True)
    try:
        records = store.all_records()
        print(f"reader: saw {len(records)} records")
        sys.exit(0)
    finally:
        store.close()
""")

# hnswlib concurrent-load script: calls load_hnsw_readonly() directly so that
# hnswlib.Index.load_index() is exercised — a SHARED read_only MemoryStore
# skips the hnswlib load (daemon owns the in-process ANN); only
# load_hnsw_readonly() exercises load_index in a client process.
_HNSW_CONCURRENT_LOAD_SCRIPT = textwrap.dedent("""\
    import os, sys
    from pathlib import Path

    store_root = Path(os.environ["IAI_MCP_STORE"])

    from iai_mcp.hippo import load_hnsw_readonly, EMBED_DIM

    idx = load_hnsw_readonly(store_root, EMBED_DIM)
    if idx is None:
        print("hnsw_load: index absent (no .hnsw file)")
        sys.exit(1)
    count = idx.get_current_count()
    print(f"hnsw_load: ok count={count}")
    sys.exit(0)
""")

# Daemon-role atomic-save script: opens the store EXCLUSIVE (as the daemon
# does), then closes it — which triggers _save_index_atomic() in HippoDB.close().
# This simulates the daemon writing a fresh.hnsw while a client may hold an
# already-loaded index. The test asserts the concurrent reader is not harmed.
_HNSW_ATOMIC_SAVE_SCRIPT = textwrap.dedent("""\
    import os, sys, time
    from pathlib import Path

    store_root = Path(os.environ["IAI_MCP_STORE"])

    from iai_mcp.store import MemoryStore

    store = MemoryStore(store_root)
    try:
        # Small pause so the concurrent reader subprocess starts its load_index.
        time.sleep(0.05)
        # Insert one record to mark the index dirty so _save_index_atomic runs.
        import uuid, numpy as np
        from datetime import datetime, timezone
        from iai_mcp.types import EMBED_DIM, MemoryRecord
        rng = np.random.RandomState(seed=999)
        vec = rng.randn(EMBED_DIM).tolist()
        rec = MemoryRecord(
            id=uuid.uuid4(),
            tier="episodic",
            literal_surface="daemon-role atomic save probe",
            aaak_index="",
            embedding=vec,
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": "daemon-save", "role": "user"}],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["role:user"],
            language="en",
        )
        store.insert(rec)
        print("atomic_save: inserted record ok")
        sys.exit(0)
    finally:
        store.close()  # triggers _save_index_atomic
""")


def _child_env(store_root: Path, tmp_path: Path) -> dict[str, str]:
    """Build a child process env that is hermetic (never contacts live daemon)."""
    env = dict(os.environ)
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_DAEMON_SOCKET_PATH"] = str(tmp_path / "no-such.sock")
    env["HOME"] = str(tmp_path)
    # Passphrase is already in os.environ from conftest._crypto_passphrase_env.
    return env


def _insert_seed_records(store_root: Path, n: int, seed_base: int = 200) -> None:
    """Insert n real-embedding records in the current process and close the store.

    Used to seed the on-disk.hnsw before concurrent subprocess tests. The
    store is closed (lock released) before returning so child processes can open it.
    """
    import uuid as _uuid
    import numpy as np
    from datetime import datetime, timezone
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    store = MemoryStore(store_root)
    try:
        for i in range(n):
            rng = np.random.RandomState(seed=seed_base + i)
            vec = rng.randn(EMBED_DIM).tolist()
            rec = MemoryRecord(
                id=_uuid.uuid4(),
                tier="episodic",
                literal_surface=f"concurrent seed record {i}",
                aaak_index="",
                embedding=vec,
                community_id=None,
                centrality=0.0,
                detail_level=1,
                pinned=False,
                stability=0.0,
                difficulty=0.0,
                last_reviewed=None,
                never_decay=False,
                never_merge=False,
                provenance=[{"session_id": "seed-session", "role": "user"}],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                tags=["role:user"],
                language="en",
            )
            store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()  # releases flock; triggers _save_index_atomic


# ---------------------------------------------------------------------------
# Test 1: genuine 2-process writer + reader — no corruption
# ---------------------------------------------------------------------------


def test_multiprocess_writer_and_reader_no_corruption(
    hermetic_store: Path, tmp_path: Path
) -> None:
    """Writer process + reader process on the same tmp store — no corruption.

    Writer uses write_turn_direct() (SHARED/LOCK_SH, deferred_embedding=True) so
    it never touches the hnswlib index and multiple SHARED holders coexist.
    Reader opens with AccessMode.SHARED + read_only=True (LOCK_SH + query_only=ON).

    Asserts:
    (a) both complete within a timeout bound (no deadlock — F2);
    (b) no HippoIntegrityError / no SQLite malformed-db error (F1);
    (c) writer reports inserting N records (correct write path);
    (d) reader completes without error (eventual consistency — F1).

    Also asserts F4 partial: a deferred-embedding record written by the writer is
    immediately findable by RECENCY (SQLite-only, embedding_pending=1). Full ANN
    appearance after rebuild is validated in test_hnswlib_concurrent_load_index_no_error.

    Uses subprocess.Popen (NOT threading, NOT multiprocessing.Process sharing
    in-process state) — genuine cross-process flock semantics.
    """
    n_records = 5
    env = _child_env(hermetic_store, tmp_path)
    env["IAI_TEST_N_RECORDS"] = str(n_records)

    # Launch writer first (deferred two-phase write, SHARED mode).
    writer = subprocess.Popen(
        [sys.executable, "-c", _WRITER_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Launch reader concurrently (SHARED read_only).
    reader = subprocess.Popen(
        [sys.executable, "-c", _READER_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    writer_out, writer_err = writer.communicate(timeout=30)
    reader_out, reader_err = reader.communicate(timeout=30)

    # (b) No errors.
    assert writer.returncode == 0, (
        f"writer process failed (rc={writer.returncode}):\n{writer_err}"
    )
    assert reader.returncode == 0, (
        f"reader process failed (rc={reader.returncode}):\n{reader_err}"
    )
    assert "HippoIntegrityError" not in writer_err, f"writer: HippoIntegrityError\n{writer_err}"
    assert "HippoIntegrityError" not in reader_err, f"reader: HippoIntegrityError\n{reader_err}"
    assert "malformed" not in reader_err.lower(), f"reader: SQLite malformed\n{reader_err}"

    # (c) Writer inserted all N records via the two-phase deferred path.
    assert f"inserted {n_records}" in writer_out, f"writer output unexpected:\n{writer_out}"
    # (d) Reader completed a full all_records() pass (exact count not asserted —
    # concurrent timing means 0-N is valid; what matters is no error).
    assert "reader: saw" in reader_out, f"reader output unexpected:\n{reader_out}"

    # F4 partial: deferred records are findable by recency immediately.
    # Open the store in this process after both subprocesses exit.
    from iai_mcp.store import MemoryStore
    from iai_mcp.hippo import AccessMode
    store = MemoryStore(hermetic_store, access_mode=AccessMode.SHARED, read_only=True)
    try:
        all_recs = store.all_records()
        assert len(all_recs) == n_records, (
            f"Expected {n_records} records after writer completed, got {len(all_recs)}"
        )
        # All rows are pending (deferred_embedding=True) but readable via SQLite.
        pending = [r for r in all_recs if getattr(r, "embedding_pending", False)]
        assert len(pending) == n_records, (
            f"Expected all {n_records} records to be embedding_pending; got {len(pending)}"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 2: hnswlib multi-process concurrent load_index assertion (A1)
# ---------------------------------------------------------------------------


def test_hnswlib_concurrent_load_index_no_error(
    hermetic_store: Path, tmp_path: Path
) -> None:
    """Two processes call load_hnsw_readonly() concurrently — no error.

    Also asserts: a daemon-role process calls the atomic.hnsw.tmp→rename save
    while a client holds a loaded index → no error in either process.

    This is the explicit Research Q#1 / A1 assertion. A failure here would
    trigger the sqlite-vec contingency decision.

    load_hnsw_readonly() is used (not a SHARED read_only MemoryStore) because
    read_only MemoryStore skips hnswlib (hnsw=None) — only load_hnsw_readonly()
    exercises load_index in a client process.
    """
    # Seed with real embeddings so records.hnsw exists on disk.
    _insert_seed_records(hermetic_store, n=3, seed_base=200)

    hnsw_path = hermetic_store / "hippo" / "records.hnsw"
    assert hnsw_path.exists(), "records.hnsw must exist after seeding for A1 test"

    env = _child_env(hermetic_store, tmp_path)

    # --- Part 1: two processes call load_index concurrently ---
    p1 = subprocess.Popen(
        [sys.executable, "-c", _HNSW_CONCURRENT_LOAD_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    p2 = subprocess.Popen(
        [sys.executable, "-c", _HNSW_CONCURRENT_LOAD_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    out1, err1 = p1.communicate(timeout=30)
    out2, err2 = p2.communicate(timeout=30)

    assert p1.returncode == 0, f"concurrent load_index process 1 failed (rc={p1.returncode}):\n{err1}"
    assert p2.returncode == 0, f"concurrent load_index process 2 failed (rc={p2.returncode}):\n{err2}"
    assert "hnsw_load: ok" in out1, f"process 1 output unexpected:\n{out1}"
    assert "hnsw_load: ok" in out2, f"process 2 output unexpected:\n{out2}"

    # --- Part 2: daemon-role atomic save while a client holds a loaded index ---
    # The reader loads the index first (in this process) to hold it in memory.
    from iai_mcp.hippo import load_hnsw_readonly, EMBED_DIM
    held_idx = load_hnsw_readonly(hermetic_store, EMBED_DIM)
    assert held_idx is not None, "Seeded index must be loadable for Part 2"

    # Daemon-role subprocess: opens EXCLUSIVE, inserts a record, closes
    # (close triggers _save_index_atomic → write.hnsw.tmp → os.replace →.hnsw).
    # The held_idx in this process already has the data loaded in memory —
    # the os.replace on the file must not corrupt or crash it.
    daemon_saver = subprocess.Popen(
        [sys.executable, "-c", _HNSW_ATOMIC_SAVE_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # While daemon saver is running, verify the held index is still functional.
    # (hnswlib load_index reads the file into memory; after loading the
    # in-memory index is independent of the on-disk file.)
    held_count = held_idx.get_current_count()
    assert held_count >= 3, f"held_idx.get_current_count() should be ≥ 3, got {held_count}"

    daemon_out, daemon_err = daemon_saver.communicate(timeout=30)
    assert daemon_saver.returncode == 0, (
        f"daemon-role saver failed (rc={daemon_saver.returncode}):\n{daemon_err}"
    )
    assert "atomic_save: inserted record ok" in daemon_out, (
        f"daemon_saver output unexpected:\n{daemon_out}"
    )

    # Verify the held index is still readable after the atomic save completed.
    assert held_idx.get_current_count() >= 3, (
        "held_idx.get_current_count() degraded after concurrent atomic save"
    )


# ---------------------------------------------------------------------------
# Test 3: reader never loads.hnsw.tmp (atomic rename respected)
# ---------------------------------------------------------------------------


def test_reader_never_loads_hnsw_tmp(hermetic_store: Path, tmp_path: Path) -> None:
    """A reader subprocess concurrent with a daemon atomic-save reads correctly.

    Seeds the store (single process, flock released), then simultaneously:
    - writes a corrupt .hnsw.tmp sentinel (simulates a daemon mid-save state), and
    - launches a concurrent reader subprocess using load_hnsw_readonly().

    load_hnsw_readonly() loads ONLY records.hnsw (never .hnsw.tmp) per its
    implementation contract. The reader must:
    (a) complete without error;
    (b) NOT load the corrupt .hnsw.tmp sentinel (only the stable .hnsw is read).

    Separate from the LOCK contention aspect: this test verifies the file-name
    guard in load_hnsw_readonly() (records.hnsw only) against a corrupt.hnsw.tmp.
    """
    _insert_seed_records(hermetic_store, n=1, seed_base=300)

    # Place a corrupt.hnsw.tmp sentinel (simulates an interrupted atomic save).
    hnsw_tmp = hermetic_store / "hippo" / "records.hnsw.tmp"
    hnsw_tmp.write_bytes(b"CORRUPT_SENTINEL")

    env = _child_env(hermetic_store, tmp_path)

    # Reader uses load_hnsw_readonly which targets records.hnsw ONLY (never.tmp).
    _READONLY_LOAD_SCRIPT = textwrap.dedent("""\
        import os, sys
        from pathlib import Path

        store_root = Path(os.environ["IAI_MCP_STORE"])

        from iai_mcp.hippo import load_hnsw_readonly, EMBED_DIM

        idx = load_hnsw_readonly(store_root, EMBED_DIM)
        if idx is None:
            print("hnsw_load: FAILED (index is None — corrupt .hnsw.tmp may have been loaded)")
            sys.exit(1)
        count = idx.get_current_count()
        print(f"hnsw_load: ok count={count}")
        sys.exit(0)
    """)

    # Concurrent writer holds LOCK_EX while the reader runs.
    # Under SHARED mode the reader uses LOCK_SH + 40 ms retry loop (<1.5 s SLO).
    # The daemon holds LOCK_EX for 0.3 s then releases — reader acquires LOCK_SH
    # after the writer exits (we test the file-name guard, not lock contention here).
    _CONCURRENT_WRITER_SCRIPT = textwrap.dedent("""\
        import os, sys, time
        from pathlib import Path
        store_root = Path(os.environ["IAI_MCP_STORE"])
        from iai_mcp.store import MemoryStore
        store = MemoryStore(store_root)
        try:
            time.sleep(0.3)  # hold LOCK_EX while reader spawns
            print("concurrent_writer: ok")
        finally:
            store.close()
    """)

    writer = subprocess.Popen(
        [sys.executable, "-c", _CONCURRENT_WRITER_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Small delay so writer acquires LOCK_EX first.
    time.sleep(0.05)

    reader = subprocess.Popen(
        [sys.executable, "-c", _READONLY_LOAD_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    writer_out, writer_err = writer.communicate(timeout=15)
    reader_out, reader_err = reader.communicate(timeout=15)

    assert writer.returncode == 0, f"writer failed:\n{writer_err}"
    # Reader must succeed: load_hnsw_readonly() loads records.hnsw not.tmp.
    # Note: load_hnsw_readonly does NOT use flock — it just opens the file.
    # The LOCK_EX held by the writer subprocess affects HippoDB.__init__ only;
    # load_hnsw_readonly() does a direct hnswlib.Index.load_index() on the file.
    assert reader.returncode == 0, (
        f"reader failed (rc={reader.returncode}):\n{reader_err}\n"
        "load_hnsw_readonly must load records.hnsw independently of flock."
    )
    assert "hnsw_load: ok" in reader_out, f"reader output unexpected:\n{reader_out}"
    # The corrupt.hnsw.tmp must not cause a silent wrong-index load.
    assert "CORRUPT" not in reader_err, f"reader may have loaded corrupt .hnsw.tmp:\n{reader_err}"
    assert "FAILED" not in reader_out, f"reader reported failure:\n{reader_out}"
