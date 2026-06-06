"""TDD regression tests for count_rows thread-safety and None-fetchone resilience.

Scenarios:
  (a) Baseline: count_rows returns correct int on a fresh store (single-thread).
  (b) Concurrent: count_rows called from worker threads while direct SQL writes
      run concurrently on the same connection — must return int, NEVER None.
  (c) Defensive raise: when fetchone returns None (simulated via mock), count_rows
      must raise HippoIntegrityError, not TypeError.
  (d) asyncio.to_thread simulate: count_rows via to_thread concurrent with other
      to_thread write operations — matches the exact daemon crash pattern.

Pre-fix (current source): scenario (b) and (d) fail — fetchone returns None
under concurrent access, producing TypeError 'NoneType' is not subscriptable.
Post-fix: all four scenarios pass.

Root cause: HippoDB shares a single sqlite3.Connection across all threads with
check_same_thread=False but no threading lock protecting execute()+fetchone()
pairs. When a concurrent thread calls conn.execute("BEGIN") between a reader's
execute(SELECT) and fetchone() calls, CPython sqlite3 resets the cursor state,
causing fetchone() to return None instead of the result row.
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from iai_mcp.hippo import HippoDB


# ---------------------------------------------------------------------------
# Direct-SQL helpers (avoid MemoryRecord complexity; test HippoTable directly)
# ---------------------------------------------------------------------------

def _insert_events(db: HippoDB, n: int = 5) -> None:
    """Insert n rows into the events table via direct BEGIN/INSERT/COMMIT
    to simulate flush_event_buffer's write pattern on the shared connection.
    """
    import json as _json
    for i in range(n):
        event_id = str(uuid.uuid4())
        ts = "2026-01-01T00:00:00+00:00"
        db._conn.execute("BEGIN")
        db._conn.execute(
            "INSERT INTO events (id, kind, severity, domain, ts, data_json, session_id, source_ids_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (event_id, "test_event", "info", "test", ts, _json.dumps({"i": i}), "sess", None),
        )
        db._conn.execute("COMMIT")


def _seed_records_direct(db: HippoDB, n: int = 10) -> None:
    """Insert n minimal records via direct SQL using only the NOT NULL columns:
    id, tier, embedding, created_at, hv_tier, structure_hv_payload.
    """
    embed_bytes = np.zeros(db._embed_dim, dtype=np.float32).tobytes()
    for i in range(n):
        rid = str(uuid.uuid4())
        ts = "2026-01-01T00:00:00+00:00"
        db._conn.execute(
            "INSERT INTO records (id, tier, embedding, created_at, hv_tier, structure_hv_payload)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (rid, "episodic", embed_bytes, ts, "bsc", b""),
        )


# ---------------------------------------------------------------------------
# Scenario (a): baseline single-thread count_rows
# ---------------------------------------------------------------------------

class TestCountRowsBaseline:
    def test_count_rows_returns_zero_on_fresh_table(self, tmp_path: Path) -> None:
        """count_rows on a fresh records table returns 0."""
        db = HippoDB(tmp_path)
        tbl = db.open_table("records")
        result = tbl.count_rows()
        assert result == 0, f"Expected 0, got {result!r}"
        db.close()

    def test_count_rows_returns_correct_int_after_inserts(self, tmp_path: Path) -> None:
        """count_rows returns the seeded row count."""
        db = HippoDB(tmp_path)
        tbl = db.open_table("records")
        _seed_records_direct(db, 7)
        result = tbl.count_rows()
        assert result == 7, f"Expected 7, got {result!r}"
        db.close()

    def test_count_rows_never_returns_none_single_thread(self, tmp_path: Path) -> None:
        """count_rows must never return None in a single-thread context."""
        db = HippoDB(tmp_path)
        tbl = db.open_table("records")
        result = tbl.count_rows()
        assert result is not None, "count_rows returned None — TypeError would follow at int(row[0])"
        db.close()


# ---------------------------------------------------------------------------
# Scenario (b): concurrent access — the direct reproducer
# ---------------------------------------------------------------------------

class TestCountRowsConcurrentAccess:
    """Regression for the daemon crash:
      TypeError: 'NoneType' object is not subscriptable
      at hippo.py count_rows: return int(row[0])

    Under concurrent writes to the same connection, fetchone() returns None.
    Pre-fix: ~28% failure rate. Post-fix: 0% failure rate.
    """

    def test_count_rows_never_none_under_concurrent_writes(self, tmp_path: Path) -> None:
        """count_rows must return int even when concurrent threads write to same conn.

        Simulates:
          Thread A: _hippea_cascade_loop -> asyncio.to_thread(build_runtime_graph)
                    -> records_tbl.count_rows()
          Thread B: _tick_body -> asyncio.to_thread(flush_event_buffer)
                    -> db._conn.execute("BEGIN") / INSERT / COMMIT
        """
        db = HippoDB(tmp_path)
        records_tbl = db.open_table("records")
        _seed_records_direct(db, 10)

        none_counts: list[str] = []
        type_errors: list[str] = []
        stop = threading.Event()

        def reader_thread() -> None:
            for _ in range(500):
                if stop.is_set():
                    break
                try:
                    result = records_tbl.count_rows()
                    if result is None:
                        none_counts.append("None from count_rows")
                except TypeError as exc:
                    type_errors.append(str(exc))

        def writer_thread() -> None:
            while not stop.is_set():
                try:
                    _insert_events(db, 1)
                except Exception:
                    pass

        readers = [threading.Thread(target=reader_thread) for _ in range(3)]
        writers = [threading.Thread(target=writer_thread) for _ in range(3)]
        for t in readers + writers:
            t.start()

        for t in readers:
            t.join(timeout=15)
        stop.set()
        for t in writers:
            t.join(timeout=5)

        db.close()

        assert not type_errors, (
            f"count_rows raised TypeError {len(type_errors)} time(s) under concurrent writes: "
            f"{type_errors[:3]}"
        )
        assert not none_counts, (
            f"count_rows returned None {len(none_counts)} time(s) under concurrent writes — "
            "pre-fix regression: execute()+fetchone() not protected by a threading lock. "
            "Fix: add threading.Lock to HippoDB and acquire it in count_rows."
        )


# ---------------------------------------------------------------------------
# Scenario (d): asyncio.to_thread — exact daemon crash pattern
# ---------------------------------------------------------------------------

class TestCountRowsAsyncioToThread:
    def test_count_rows_stable_under_asyncio_to_thread_concurrency(
        self, tmp_path: Path
    ) -> None:
        """Simulates the exact daemon asyncio.to_thread fan-out that caused the crash.

        _hippea_cascade_loop dispatches build_runtime_graph (which calls count_rows)
        via asyncio.to_thread(). While that worker thread is running, the event loop
        dispatches flush_event_buffer via another asyncio.to_thread() — both end up
        touching db._conn simultaneously.

        Pre-fix: fetchone returns None on ~28% of count_rows calls.
        Post-fix: always returns int.
        """
        db = HippoDB(tmp_path)
        records_tbl = db.open_table("records")
        _seed_records_direct(db, 15)

        type_errors: list[str] = []
        none_results: list[str] = []

        def flush_events_sync() -> None:
            try:
                _insert_events(db, 5)
            except Exception:
                pass

        def build_graph_sim() -> int | None:
            try:
                return records_tbl.count_rows()
            except TypeError as exc:
                type_errors.append(str(exc))
                return None

        async def run_concurrent() -> None:
            for _ in range(30):
                results = await asyncio.gather(
                    asyncio.to_thread(build_graph_sim),
                    asyncio.to_thread(flush_events_sync),
                )
                count_result = results[0]
                if count_result is None and not type_errors:
                    none_results.append("None returned from count_rows (no TypeError raised)")

        asyncio.run(run_concurrent())
        db.close()

        assert not type_errors, (
            f"count_rows raised TypeError under asyncio.to_thread: {type_errors[:3]}"
        )
        assert not none_results, (
            f"count_rows returned None under asyncio.to_thread {len(none_results)} time(s) — "
            "pre-fix regression: daemon crash pattern reproduced"
        )


# ---------------------------------------------------------------------------
# Scenario (c): defensive HippoIntegrityError when fetchone returns None
# ---------------------------------------------------------------------------

class TestCountRowsDefensiveRaise:
    """Regression: count_rows must raise HippoIntegrityError (not TypeError)
    when fetchone returns None. The original TypeError 'NoneType' object is not
    subscriptable is cryptic and hides the root cause at the source.

    sqlite3.Connection.execute is a C-level read-only attribute and cannot be
    patched via unittest.mock.patch. Instead we replace tbl._conn temporarily
    with a MagicMock that returns a cursor whose fetchone() is None.
    """

    def _make_tbl_with_none_fetchone(self, tmp_path: Path):
        """Return (db, tbl) where tbl._conn.execute(...).fetchone() returns None."""
        from iai_mcp.hippo import HippoIntegrityError
        db = HippoDB(tmp_path)
        tbl = db.open_table("records")

        # Replace tbl._conn with a mock whose execute().fetchone() returns None.
        # tbl._conn is a plain Python attribute (set in HippoTable.__init__),
        # so it CAN be replaced — unlike the C-level sqlite3.Connection.execute.
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.execute.return_value = mock_cursor
        mock_conn.in_transaction = False  # for the error message

        tbl._conn = mock_conn
        return db, tbl

    def test_raises_hippo_integrity_error_not_type_error(self, tmp_path: Path) -> None:
        """When fetchone() returns None, count_rows must raise HippoIntegrityError."""
        from iai_mcp.hippo import HippoIntegrityError

        db, tbl = self._make_tbl_with_none_fetchone(tmp_path)

        with pytest.raises(HippoIntegrityError) as exc_info:
            tbl.count_rows()

        err_msg = str(exc_info.value)
        assert any(kw in err_msg for kw in ("records", "COUNT", "None", "fetchone")), (
            f"HippoIntegrityError message lacks diagnostic context: {err_msg!r}"
        )
        db.close()

    def test_type_error_no_longer_escapes(self, tmp_path: Path) -> None:
        """TypeError must not escape count_rows — pre-fix behavior that caused daemon crash."""
        from iai_mcp.hippo import HippoIntegrityError

        db, tbl = self._make_tbl_with_none_fetchone(tmp_path)

        raised_type_error = False
        raised_integrity_error = False

        try:
            tbl.count_rows()
        except TypeError:
            raised_type_error = True
        except HippoIntegrityError:
            raised_integrity_error = True

        assert not raised_type_error, (
            "count_rows raised TypeError — pre-fix behavior. "
            "Must raise HippoIntegrityError instead."
        )
        assert raised_integrity_error, (
            "count_rows must raise HippoIntegrityError when fetchone returns None."
        )
        db.close()
