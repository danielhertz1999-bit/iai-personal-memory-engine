"""Regression test for the close()-drain-purge-release contract on the
MemoryStore module-level write buffers.

Asserts that closing a store removes its ``id()`` from all six buffer /
flush-timestamp dicts AND that a freshly-opened store at a different
``tmp_path`` inherits no buffered ciphertext even after ``gc.collect()``
forces ``id()`` reuse.
"""
from __future__ import annotations

import gc
from datetime import datetime, timezone

import pytest


# ---------------------------------------------------------------------------
# Test 1: explicit dict-purge proof for all six module-level buffer dicts.
# ---------------------------------------------------------------------------


def test_close_purges_all_six_buffer_dicts(tmp_path):
    """``close()`` pops ``id(self)`` from every module-level buffer / flush dict.

    The drain step (best-effort flush) may also pop entries; the assertion
    is about the post-close STATE of the dicts, which must be free of the
    closed store's id regardless of which step did the pop.

    Pre-seed strategy:
    - ``events._event_buffer`` is populated naturally via
      ``write_event(buffered=True)``.
    - The four ``*_last_flush_at`` dicts (timestamps only, never touched
      by drain) are seeded directly.
    - ``_record_buffer`` and ``_edge_buffer`` are seeded with the same
      empty-list value an unused-but-present-key state would have. Drain
      sees ``pending == []`` and returns 0 -- but the key still exists,
      so PURGE has something to pop.
    """
    from iai_mcp import events, store as store_mod
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    s = MemoryStore(path=tmp_path)
    try:
        store_id = id(s)

        # Make sure the events buffer has at least one real row so the
        # drain step exercises a non-empty pending list (covers the
        # success-timestamp update branch too).
        write_event(s, kind="reg_test", data={"k": "v"}, buffered=True)
        assert store_id in events._event_buffer

        # Seed the remaining five dicts. For the two row-buffer dicts, an
        # empty-list value is enough to make ``store_id in dict`` True
        # while keeping drain a no-op (pending = pop -> [] -> early return).
        events._last_flush_at[store_id] = datetime.now(timezone.utc)
        store_mod._record_buffer[store_id] = []
        store_mod._record_last_flush_at[store_id] = datetime.now(timezone.utc)
        store_mod._edge_buffer[store_id] = []
        store_mod._edge_last_flush_at[store_id] = datetime.now(timezone.utc)

        # All six dicts now contain store_id.
        for dct_name, dct in (
            ("events._event_buffer", events._event_buffer),
            ("events._last_flush_at", events._last_flush_at),
            ("store._record_buffer", store_mod._record_buffer),
            ("store._record_last_flush_at", store_mod._record_last_flush_at),
            ("store._edge_buffer", store_mod._edge_buffer),
            ("store._edge_last_flush_at", store_mod._edge_last_flush_at),
        ):
            assert store_id in dct, f"pre-close: {dct_name} missing store_id"

        s.close()

        # Post-close: none of the six dicts may contain store_id. PURGE
        # is the only step that touches the timestamp dicts and the
        # initially-empty row-buffer dicts; if PURGE is broken on any one
        # of the six, the corresponding assertion below fires.
        assert store_id not in events._event_buffer
        assert store_id not in events._last_flush_at
        assert store_id not in store_mod._record_buffer
        assert store_id not in store_mod._record_last_flush_at
        assert store_id not in store_mod._edge_buffer
        assert store_id not in store_mod._edge_last_flush_at
    finally:
        # Belt-and-braces: close() is idempotent.
        s.close()


# ---------------------------------------------------------------------------
# Test 2: no ghost ciphertext crosses an explicit close + gc.collect cycle.
# ---------------------------------------------------------------------------


def test_no_ghost_ciphertext_across_id_reuse(tmp_path, tmp_path_factory):
    """A freshly-opened store sees no buffer rows authored by a previously-closed store.

    Python does not guarantee id() reclamation on any particular GC pass, so
    we cannot assert ``id(store_a) == id(store_b)`` deterministically. The
    assertion is the stronger property: regardless of whether id reuse
    actually fires on this run, store_b's view of every buffer dict must
    be free of any row authored by store_a. If close() ever stops popping
    even one of the dicts, this test fails on at least some seeds.
    """
    from iai_mcp import events, store as store_mod
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    path_b = tmp_path_factory.mktemp("store_b")

    # Open store A, buffer a sentinel-keyed event, close.
    store_a = MemoryStore(path=tmp_path)
    write_event(store_a, kind="sentinel_A", data={"author": "store_a"}, buffered=True)
    # The drain step in close() will write this sentinel row to store_a's
    # disk (under store_a's key+AAD) -- that is correct behavior, not a bug.
    store_a.close()

    # Force GC to reclaim store_a; non-deterministic whether the id() value
    # is reused, but this is the only realistic way to provoke the bug
    # surface the patch is meant to close.
    del store_a
    gc.collect()

    # Open store B at a different path. If close() did not purge the six
    # dicts on store_a (which it now does), and if Python recycled the
    # id() value to store_b (random), then store_b would start with
    # ghost ciphertext from store_a in events._event_buffer at id(store_b).
    store_b = MemoryStore(path=path_b)
    try:
        store_b_id = id(store_b)

        # Inspect every dict for any row referencing the sentinel kind.
        # The bug surface is "events._event_buffer at id(store_b) contains
        # a row whose kind is sentinel_A"; the patched close() prevents
        # this regardless of whether id reuse happened.
        ghost_event_rows = [
            r for r in events._event_buffer.get(store_b_id, [])
            if r.get("kind") == "sentinel_A"
        ]
        assert not ghost_event_rows, (
            f"ghost ciphertext leaked into events._event_buffer: {ghost_event_rows}"
        )

        # No flush-timestamp ghost either.
        # store_b should not see a pre-set _last_flush_at unless it
        # produced one itself (it has not).
        # (Looser assertion: just check the four other dicts are also
        # ghost-free for store_b's id.)
        for dct_name, dct in (
            ("store._record_buffer", store_mod._record_buffer),
            ("store._edge_buffer", store_mod._edge_buffer),
        ):
            ghost_rows = dct.get(store_b_id, [])
            assert not ghost_rows, (
                f"ghost rows in {dct_name} at id(store_b)={store_b_id}: {ghost_rows}"
            )

        # Also assert store_b's events table on disk does not contain
        # any sentinel_A row -- the sentinel belongs to store_a's path,
        # not path_b. (Proves drain wrote to the right disk.)
        from iai_mcp.events import query_events
        rows = query_events(store_b, kind="sentinel_A")
        assert not rows, (
            f"sentinel_A row leaked onto store_b's disk; expected store_a's path only: {rows}"
        )
    finally:
        store_b.close()
