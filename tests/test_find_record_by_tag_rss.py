"""Regression gate: find_record_by_tag must not materialize full Arrow batches
on every call.

ROOT CAUSE confirmed here:
  drain_deferred_captures → capture_turn → find_record_by_tag
  The old implementation called iter_record_columns(["id","tags_json"]) which
  invokes HippoQuery.to_batches() — a FULL materialization of the entire
  records table wrapped in a PyArrow RecordBatch — on every call.
  With a large live store (175 MB in production) and a 31-file deferred
  backlog, the cumulative un-freed Arrow buffers pushed RSS past the 2.5 GiB
  watchdog cap, triggering a crash-loop.

Three assertions:
  RED (path): the old full-scan implementation calls to_batches() on every
              find_record_by_tag call — confirmed by spy call count.
  GREEN (path): the new SQL implementation does NOT call to_batches() —
                confirmed by spy call count being zero.
  CORRECTNESS: find_record_by_tag correctly returns the target UUID for a
               tagged record and None for absent tags.

The repro is HERMETIC:
  - tmp HOME + IAI_MCP_STORE + IAI_DAEMON_SOCKET_PATH (no real ~/.iai-mcp)
  - keyring fail-backend + passphrase env var (no macOS Keychain prompt)
  - embedder monkeypatched to a fast stub (test focuses on the scan path, not
    embed throughput)
  - generic 'User'/tmp_path only — no PII
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock
from uuid import UUID

import psutil
import pytest


# POSIX assumption for HOME isolation.
pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX paths; hermetic fixture uses HOME monkeypatching",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBED_DIM = 384
_SEED_RECORDS = 300    # enough to make a full scan non-trivial
_PROBE_TURNS = 30      # number of find_record_by_tag calls per workload run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_embedding() -> list[float]:
    return [0.0] * _EMBED_DIM


def _make_record(text: str, tags: list[str] | None = None):
    """Minimal episodic MemoryRecord using a zero-embedding stub."""
    from iai_mcp.types import MemoryRecord

    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=_zero_embedding(),
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
        tags=tags or ["role:user"],
        language="en",
    )


def _old_find_record_by_tag(self, tag: str) -> UUID | None:
    """The original full-scan implementation — used for RED path evidence."""
    tag_json_literal = json.dumps(tag)
    for row in self.iter_record_columns(["id", "tags_json"]):
        tags_raw = row.get("tags_json") or "[]"
        if tag_json_literal not in tags_raw:
            continue
        try:
            tags = json.loads(tags_raw)
        except (ValueError, TypeError):
            continue
        if tag in tags:
            raw_id = row.get("id")
            if raw_id is None:
                continue
            try:
                return UUID(str(raw_id))
            except (ValueError, AttributeError):
                continue
    return None


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def rss_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Hermetic HOME + store + keyring isolation.

    Matches the pattern in test_drain_deferred_captures.py: HOME=tmp,
    keyring fail-backend, passphrase env var, IAI_MCP_STORE under tmp.
    Returns tmp_path.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-rss-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "lancedb"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))

    import keyring.core
    keyring.core._keyring_backend = None

    # Stub the Rust embedder so seeding and drain turns don't load bge-small.
    from iai_mcp import embed as embed_mod

    class _FakeEmbedder:
        DIM = _EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return _zero_embedding()

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [_zero_embedding() for _ in texts]

    monkeypatch.setattr(embed_mod, "embedder_for_store", lambda store: _FakeEmbedder())

    yield tmp_path

    keyring.core._keyring_backend = None


def _open_store(tmp_path: Path):
    """Open a fresh MemoryStore under tmp_path."""
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _seed_store(store, n: int) -> None:
    """Insert n records with distinct text into the store."""
    for i in range(n):
        r = _make_record(
            text=f"seeded record {i:05d} with enough text to represent a real turn",
            tags=["role:user", f"seed:{i}"],
        )
        store.insert(r)


# ---------------------------------------------------------------------------
# Test: RED path — old full-scan calls to_batches on every probe
# ---------------------------------------------------------------------------


def test_old_full_scan_calls_to_batches_each_probe(
    rss_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED path: old iter_record_columns impl invokes to_batches on every call.

    Monkeypatches find_record_by_tag back to the original full-scan
    implementation. Spies on HippoQuery.to_batches to count invocations.
    With _PROBE_TURNS calls, to_batches must be called at least _PROBE_TURNS
    times — confirming the O(N_calls * store_size) materialization.

    This is the root cause of the drain crash-loop: each drain turn
    materializes the full store into Arrow buffers via to_batches.
    """
    from iai_mcp.store import MemoryStore
    from iai_mcp import hippo as hippo_mod

    monkeypatch.setattr(MemoryStore, "find_record_by_tag", _old_find_record_by_tag)

    to_batches_call_count: list[int] = [0]
    original_to_batches = hippo_mod.HippoQuery.to_batches

    def counting_to_batches(self, batch_size=1024):
        to_batches_call_count[0] += 1
        return original_to_batches(self, batch_size)

    monkeypatch.setattr(hippo_mod.HippoQuery, "to_batches", counting_to_batches)

    store = _open_store(rss_env)
    try:
        _seed_store(store, _SEED_RECORDS)

        # Reset counter AFTER seeding (seeding may also call iter_record_columns
        # in internal paths — we want to count only the probe calls).
        to_batches_call_count[0] = 0

        for i in range(_PROBE_TURNS):
            store.find_record_by_tag(f"nonexistent-idem-tag-{i}")
    finally:
        store.close()

    call_count = to_batches_call_count[0]
    print(f"\n[RED] Old full-scan: to_batches called {call_count} times for {_PROBE_TURNS} probes")

    assert call_count >= _PROBE_TURNS, (
        f"Expected old full-scan to call to_batches at least {_PROBE_TURNS} times"
        f" ({_PROBE_TURNS} probes); got {call_count}."
        " The old implementation must trigger a full materialization per call."
    )


# ---------------------------------------------------------------------------
# Test: GREEN gate — new SQL implementation does NOT call to_batches
# ---------------------------------------------------------------------------


def test_new_sql_does_not_call_to_batches(rss_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GREEN gate: the new targeted SQL find_record_by_tag bypasses to_batches entirely.

    Same workload as the RED test but with the fixed implementation. Spies on
    HippoQuery.to_batches — it must NOT be called at all during
    find_record_by_tag (the SQL path goes directly to db._conn.execute).
    """
    from iai_mcp import hippo as hippo_mod

    to_batches_call_count: list[int] = [0]
    original_to_batches = hippo_mod.HippoQuery.to_batches

    def counting_to_batches(self, batch_size=1024):
        to_batches_call_count[0] += 1
        return original_to_batches(self, batch_size)

    monkeypatch.setattr(hippo_mod.HippoQuery, "to_batches", counting_to_batches)

    store = _open_store(rss_env)
    try:
        _seed_store(store, _SEED_RECORDS)
        to_batches_call_count[0] = 0  # reset after seeding

        for i in range(_PROBE_TURNS):
            store.find_record_by_tag(f"nonexistent-idem-tag-{i}")
    finally:
        store.close()

    call_count = to_batches_call_count[0]
    print(f"\n[GREEN] New SQL: to_batches called {call_count} times for {_PROBE_TURNS} probes")

    assert call_count == 0, (
        f"New SQL find_record_by_tag must NOT call to_batches; got {call_count} calls."
        " The fix is not effective — the Arrow materialization path is still active."
    )


# ---------------------------------------------------------------------------
# Test: correctness — tag lookup round-trip
# ---------------------------------------------------------------------------


def test_find_record_by_tag_correctness(rss_env: Path) -> None:
    """find_record_by_tag returns the correct UUID and None for absent tags.

    Exercises three cases:
    - A record with a matching tag is found and returns its UUID.
    - A record without the tag is NOT returned.
    - A tag that doesn't exist anywhere returns None.
    Also verifies idempotency-key semantics: the idem tag format used by
    capture_turn round-trips correctly through find_record_by_tag.
    """
    from iai_mcp.capture import _idem_tag

    store = _open_store(rss_env)
    try:
        target_tag = "role:user"
        absent_tag = "definitely-not-present-zyxwv"
        idem = _idem_tag("test-sess", "user", "2026-06-05T12:00:00+00:00", "hello world")

        # Insert record with target_tag.
        r1 = _make_record("record with target tag", tags=[target_tag, "extra:tag"])
        store.insert(r1)

        # Insert record with idem tag.
        r2 = _make_record("record with idem tag", tags=[idem, "role:user"])
        store.insert(r2)

        # Insert record with neither tag.
        r3 = _make_record("record with different tags only", tags=["role:assistant"])
        store.insert(r3)

        # Should find r1 by target_tag (first inserted record with that tag).
        found = store.find_record_by_tag(target_tag)
        assert found == r1.id, (
            f"Expected r1.id={r1.id}, got {found}"
        )

        # Should find r2 by idem tag.
        found_idem = store.find_record_by_tag(idem)
        assert found_idem == r2.id, (
            f"Expected r2.id={r2.id}, got {found_idem}"
        )

        # Should return None for absent tag.
        not_found = store.find_record_by_tag(absent_tag)
        assert not_found is None, (
            f"Expected None for absent tag, got {not_found}"
        )

    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test: correctness — None return when store is empty
# ---------------------------------------------------------------------------


def test_find_record_by_tag_empty_store(rss_env: Path) -> None:
    """find_record_by_tag returns None gracefully on an empty store."""
    store = _open_store(rss_env)
    try:
        result = store.find_record_by_tag("idem:anything")
        assert result is None, f"Expected None on empty store, got {result}"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test: correctness — behavior parity (no tombstone filter, matching old impl)
# ---------------------------------------------------------------------------


def test_find_record_by_tag_no_tombstone_filter(rss_env: Path) -> None:
    """Behavior parity: old impl had no tombstone filter; new SQL matches.

    The new SQL query also has no WHERE tombstoned_at IS NULL filter, which
    preserves the old behavior: a tombstoned record's idem tag still blocks
    re-drain if present. This test confirms the row is found after insert
    (tombstone operation is a separate path not exercised here).
    """
    store = _open_store(rss_env)
    try:
        tag = "idem:tombstone-parity-test"
        r = _make_record("tombstone parity record", tags=[tag])
        store.insert(r)

        found = store.find_record_by_tag(tag)
        assert found == r.id, f"Expected {r.id}, got {found}"

        # Stable across repeated calls — no side effects.
        found_again = store.find_record_by_tag(tag)
        assert found_again == r.id, "tag lookup must be stable across multiple calls"

    finally:
        store.close()
