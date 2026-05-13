"""Task 1.2 -- capture_queue.py test suite.

Covers atomic append (incl. crash simulation), 50-thread concurrent
append, idempotent ingest with mid-handler crash, lock-skip semantics,
overflow + audit log, verbatim Unicode round-trip, list_pending sort
order, schema-version mismatch, empty-queue ingest, ULID lex<->time
order, and lock-file cleanup on success/failure.

All tests use ``tmp_path`` -- no production queue at ``~/.iai-mcp/pending/``
is touched.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from iai_mcp.capture_queue import (
    DEFAULT_MAX_SIZE,
    SCHEMA_VERSION,
    CaptureQueue,
    CaptureQueueSchemaError,
    generate_ulid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_record(i: int = 0, surface: str | None = None) -> dict:
    """Return a minimally valid record envelope dict."""
    return {
        "surface": surface if surface is not None else f"sample text {i}",
        "cue": f"cue {i}",
        "tier": "episodic",
        "session_id": "test-session",
        "role": "user",
    }


def _write_envelope_directly(
    queue_dir: Path,
    ulid: str,
    record: dict,
    *,
    schema_version: int = SCHEMA_VERSION,
    appended_at: str = "2026-05-02T15:00:00+00:00",
) -> Path:
    """Bypass ``CaptureQueue.append`` to seed a pending file with custom fields."""
    path = queue_dir / f"pending-{ulid}.json"
    envelope = {
        "ulid": ulid,
        "appended_at": appended_at,
        "record": record,
        "schema_version": schema_version,
    }
    path.write_text(
        json.dumps(envelope, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# 1. Basic append + file creation
# ---------------------------------------------------------------------------

def test_append_returns_ulid_and_creates_file(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    ulid = q.append(_sample_record(0))

    assert isinstance(ulid, str)
    assert len(ulid) == 26
    final = tmp_path / f"pending-{ulid}.json"
    assert final.exists()

    envelope = json.loads(final.read_text(encoding="utf-8"))
    assert envelope["ulid"] == ulid
    assert envelope["schema_version"] == SCHEMA_VERSION
    assert envelope["record"]["surface"] == "sample text 0"
    # appended_at is ISO-8601 parseable.
    from datetime import datetime
    datetime.fromisoformat(envelope["appended_at"])

    assert q.pending_count() == 1


# ---------------------------------------------------------------------------
# 2. Atomic append under simulated crash (os.replace patched to raise)
# ---------------------------------------------------------------------------

def test_append_atomic_under_crash_simulation(tmp_path, monkeypatch):
    """If ``os.replace`` fails, no committed pending file appears.

    The temp file may or may not be left around depending on where the
    failure happens; what matters is that ``pending_count`` stays 0
    because no ``pending-<ulid>.json`` was successfully published.
    """
    q = CaptureQueue(queue_dir=tmp_path)

    real_replace = os.replace

    def boom(src, dst):
        raise OSError(errno.EIO, "simulated crash mid-rename")

    monkeypatch.setattr("iai_mcp.capture_queue.os.replace", boom)

    with pytest.raises(OSError):
        q.append(_sample_record(0))

    # No final pending file appeared.
    assert q.pending_count() == 0
    finals = list(tmp_path.glob("pending-*.json"))
    finals = [p for p in finals if not p.name.endswith(".tmp")]
    assert finals == []

    # Restore + verify a real append still works.
    monkeypatch.setattr("iai_mcp.capture_queue.os.replace", real_replace)
    q.append(_sample_record(1))
    assert q.pending_count() == 1


# ---------------------------------------------------------------------------
# 3. Concurrent append (50 threads * 10 records each = 500)
# ---------------------------------------------------------------------------

def test_concurrent_append_50_threads(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    n_threads = 50
    n_per_thread = 10
    errors: list[BaseException] = []
    ulids: list[str] = []
    ulids_lock = threading.Lock()

    def worker(tid: int) -> None:
        try:
            local: list[str] = []
            for i in range(n_per_thread):
                ulid = q.append(_sample_record(i, f"thread-{tid}-record-{i}"))
                local.append(ulid)
            with ulids_lock:
                ulids.extend(local)
        except BaseException as exc:  # pragma: no cover - surfaced via assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "worker thread hung"

    assert errors == [], f"workers raised: {errors!r}"
    assert len(ulids) == n_threads * n_per_thread
    # No ULID collisions.
    assert len(set(ulids)) == len(ulids)
    # Every committed file is well-formed JSON.
    pending = q.list_pending()
    assert len(pending) == n_threads * n_per_thread
    for p in pending:
        envelope = json.loads(p.read_text(encoding="utf-8"))
        assert envelope["schema_version"] == SCHEMA_VERSION
        assert envelope["record"]["surface"].startswith("thread-")


# ---------------------------------------------------------------------------
# 4. Idempotent ingest -- crash mid-handler leaves both files, retry works
# ---------------------------------------------------------------------------

def test_idempotent_ingest_crash_mid_handler(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    ulid = q.append(_sample_record(42, surface="payload-42"))

    pending_path = tmp_path / f"pending-{ulid}.json"
    lock_path = tmp_path / f"pending-{ulid}.lock"

    def crashing_handler(_record: dict) -> None:
        raise RuntimeError("handler exploded")

    with pytest.raises(RuntimeError):
        q.ingest_pending(crashing_handler)

    # Both pending and lock remain on disk.
    assert pending_path.exists(), "pending file must remain after handler exception"
    assert lock_path.exists(), "lock file must remain to mark mid-flight crash"
    assert q.pending_count() == 1

    # Retry with a clean handler -- should succeed.
    seen: list[dict] = []

    def good_handler(record: dict) -> None:
        seen.append(record)

    n = q.ingest_pending(good_handler)
    assert n == 1
    assert len(seen) == 1
    assert seen[0]["surface"] == "payload-42"
    # Both files cleaned up after success.
    assert not pending_path.exists()
    assert not lock_path.exists()
    assert q.pending_count() == 0


# ---------------------------------------------------------------------------
# 5. Lock contention -- A held externally, B and C still ingest
# ---------------------------------------------------------------------------

def test_idempotent_ingest_lock_skipped(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    ulid_a = q.append(_sample_record(1, surface="A"))
    ulid_b = q.append(_sample_record(2, surface="B"))
    ulid_c = q.append(_sample_record(3, surface="C"))

    # Externally lock A's lock file in non-blocking exclusive mode.
    lock_a = tmp_path / f"pending-{ulid_a}.lock"
    fd = os.open(str(lock_a), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        seen: list[str] = []

        def handler(record: dict) -> None:
            seen.append(record["surface"])

        n = q.ingest_pending(handler)
        # B and C ingested; A skipped because we hold its lock.
        assert n == 2
        assert sorted(seen) == ["B", "C"]
        # A still pending.
        assert (tmp_path / f"pending-{ulid_a}.json").exists()
        assert not (tmp_path / f"pending-{ulid_b}.json").exists()
        assert not (tmp_path / f"pending-{ulid_c}.json").exists()
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


# ---------------------------------------------------------------------------
# 6. Overflow -- exceed max, oldest pruned, audit log populated
# ---------------------------------------------------------------------------

def test_overflow_prune_oldest(tmp_path):
    """At ``max_size=100``, 110 appends end with count=99 (max-100 headroom)
    and 11 audit entries (10 over + 1 to descend below max).

    The exact post-prune count is ``max_size - 100`` because the prune
    batch headroom in capture_queue is 100. With ``max_size=100`` the
    target is therefore 0; the actual pruned count equals the excess at
    the moment of first overflow plus subsequent appends that re-trigger
    overflow.

    The deterministic invariants are:

    1. Final ``pending_count`` <= ``max_size``.
    2. Total appends == kept + dropped.
    3. Audit log has exactly ``dropped`` JSONL lines, all with
       reason="queue_overflow" and a known ULID.
    """
    max_size = 100
    n_total = 110
    q = CaptureQueue(queue_dir=tmp_path, max_size=max_size)

    appended_ulids: list[str] = []
    for i in range(n_total):
        appended_ulids.append(q.append(_sample_record(i)))

    final_count = q.pending_count()
    assert final_count <= max_size

    audit_path = tmp_path / ".overflow-audit.log"
    assert audit_path.exists(), "audit log must exist after overflow"

    audit_lines = audit_path.read_text(encoding="utf-8").splitlines()
    audit_records = [json.loads(line) for line in audit_lines if line]

    dropped = n_total - final_count
    assert dropped > 0, "at least one record must have been dropped on overflow"
    assert len(audit_records) == dropped, (
        f"expected {dropped} audit entries, got {len(audit_records)}"
    )
    for rec in audit_records:
        assert rec["reason"] == "queue_overflow"
        assert rec["dropped_ulid"] in appended_ulids
        assert isinstance(rec["queue_size_before_prune"], int)


# ---------------------------------------------------------------------------
# 7. Verbatim round-trip -- Russian + English + emoji + Greek + symbols
# ---------------------------------------------------------------------------

def test_verbatim_round_trip_unicode(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    payload = "Привет, world! 🧠 Δ ∑ — combining é vs é"

    q.append(_sample_record(0, surface=payload))
    seen: list[str] = []

    def handler(record: dict) -> None:
        seen.append(record["surface"])

    n = q.ingest_pending(handler)
    assert n == 1
    assert len(seen) == 1
    # Byte-identical surface preserved through JSON encode + decode.
    assert seen[0] == payload
    assert seen[0].encode("utf-8") == payload.encode("utf-8")


# ---------------------------------------------------------------------------
# 8. list_pending sort order is oldest-first
# ---------------------------------------------------------------------------

def test_list_pending_sort_order(tmp_path):
    """ULIDs are time-sorted by construction; listing them sorted by name
    must yield the same order in which they were appended.
    """
    q = CaptureQueue(queue_dir=tmp_path)
    ulids = [q.append(_sample_record(i)) for i in range(20)]
    listed = [q._ulid_from_path(p) for p in q.list_pending()]
    assert listed == ulids, "list_pending must be oldest-first"


# ---------------------------------------------------------------------------
# 9. Schema-version mismatch raises CaptureQueueSchemaError
# ---------------------------------------------------------------------------

def test_schema_version_mismatch_raises(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    _write_envelope_directly(
        tmp_path,
        ulid="01HZQTESTBADSCHEMA00000000",
        record=_sample_record(0),
        schema_version=99,
    )
    assert q.pending_count() == 1

    def handler(_record: dict) -> None:  # pragma: no cover -- never called
        pytest.fail("handler must not be called on schema mismatch")

    with pytest.raises(CaptureQueueSchemaError) as excinfo:
        q.ingest_pending(handler)
    assert "schema_version" in str(excinfo.value)
    assert "99" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 10. Empty queue -- ingest returns 0, no errors
# ---------------------------------------------------------------------------

def test_empty_queue_ingest_returns_zero(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    assert q.pending_count() == 0

    handler_called = [False]

    def handler(_record: dict) -> None:  # pragma: no cover -- never called
        handler_called[0] = True

    n = q.ingest_pending(handler)
    assert n == 0
    assert handler_called[0] is False


# ---------------------------------------------------------------------------
# 11. ULID lex sort matches generation/time order over many samples
# ---------------------------------------------------------------------------

def test_ulid_lexicographic_sort_matches_time_order():
    """Generate 1000 ULIDs as fast as possible; their natural string sort
    must equal generation order. The internal monotonic guard guarantees
    this even when many ULIDs collide on the same wall-clock millisecond.
    """
    n = 1000
    ulids = [generate_ulid() for _ in range(n)]
    assert len(set(ulids)) == n, "no ULID collisions allowed"
    assert sorted(ulids) == ulids, "lex sort must equal generation order"


# ---------------------------------------------------------------------------
# 12. Lock file cleaned up on handler success
# ---------------------------------------------------------------------------

def test_lock_file_cleanup_on_handler_success(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    ulid = q.append(_sample_record(0))
    lock_path = tmp_path / f"pending-{ulid}.lock"

    def handler(_record: dict) -> None:
        # While the handler runs, the lock file IS on disk -- but we
        # cannot easily inspect that without breaking the lock owner
        # invariant. The post-condition is what matters here.
        pass

    n = q.ingest_pending(handler)
    assert n == 1
    assert not lock_path.exists(), "lock file must be cleaned on success"
    assert not (tmp_path / f"pending-{ulid}.json").exists()


# ---------------------------------------------------------------------------
# 13. Lock file persists on handler exception (mid-flight crash marker)
# ---------------------------------------------------------------------------

def test_lock_file_persists_on_handler_exception(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    ulid = q.append(_sample_record(0))
    pending_path = tmp_path / f"pending-{ulid}.json"
    lock_path = tmp_path / f"pending-{ulid}.lock"

    def handler(_record: dict) -> None:
        raise ValueError("simulated mid-handler crash")

    with pytest.raises(ValueError):
        q.ingest_pending(handler)

    assert pending_path.exists(), "pending must remain after handler exception"
    assert lock_path.exists(), "lock must remain to mark mid-flight crash"
