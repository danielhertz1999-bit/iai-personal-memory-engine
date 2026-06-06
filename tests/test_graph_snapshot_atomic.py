"""Atomic snapshot for ``runtime_graph_cache.json``.

The on-disk file (constant ``CACHE_FILENAME`` in
``src/iai_mcp/runtime_graph_cache.py``) is also referred to as
``graph.json``. The save path in ``runtime_graph_cache.py``
already uses a ``.tmp`` + ``os.replace`` rename. This file adds the
missing ``f.flush()`` + ``os.fsync(f.fileno())`` step BEFORE close /
replace so a mid-write crash on macOS APFS or Linux ext4 can never
leave a partially-written final file.

Three RED-witness contracts:

1. ``test_crash_after_tmp_write_leaves_previous_snapshot_intact`` — when
   ``os.replace`` raises mid-save (kernel-panic-between-write-and-rename
   simulation), the reader still sees the previous complete snapshot
   and the ``.tmp`` sidecar is cleaned up. This is a regression test
   that documents the OSError-handler-already-cleans-up contract; the
   real RED witnesses for the new fsync line are tests 2 + 3.

2. ``test_save_calls_fsync_before_replace`` — ``os.fsync`` is invoked
   on the open ``.tmp`` file descriptor BEFORE ``os.replace`` runs.
   Asserted via ``parent.mock_calls`` ordering across spies installed
   on both ``os.fsync`` and ``os.replace``. On the un-modified source
   the assertion fails because zero ``fsync`` calls land.

3. ``test_save_handles_fsync_failure_gracefully`` — when ``os.fsync``
   raises ``OSError("disk full")`` the existing ``except OSError``
   branch in ``runtime_graph_cache.py`` swallows it, returns
   ``False``, cleans up the ``.tmp`` file, and leaves any pre-existing
   cache intact. On the un-modified source ``os.fsync`` is never
   called so this contract is meaningless — the test fails RED by
   virtue of the assertion that the failure path actually fires.

Patterns reused from ``tests/test_runtime_graph_cache.py`` verbatim:
the ``_isolated_keyring`` autouse fixture, the ``store`` fixture and
the ``_make_assignment`` helper. Per-file factories are project canon.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pytest

from iai_mcp import runtime_graph_cache
from iai_mcp.community import CommunityAssignment
from iai_mcp.store import MemoryStore


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Project-canon dict-backed keyring fake. ``save()`` resolves the
    cache encryption key through ``_cache_encryption_key`` which reaches
    the keyring on the construction host — substitute a deterministic
    in-memory dict so the test never blocks on a real keyring prompt.
    Copied verbatim from ``tests/test_runtime_graph_cache.py``."""
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    """Fresh MemoryStore in ``tmp_path/hippo`` with the cache file path
    pinned to ``tmp_path/runtime_graph_cache.json`` — same shape as the
    sibling test file. Copied verbatim from
    ``tests/test_runtime_graph_cache.py``."""
    s = MemoryStore(path=tmp_path / "hippo")
    s.root = tmp_path
    return s


def _make_assignment(n_communities: int = 2) -> CommunityAssignment:
    """Per-file factory for a minimally-populated assignment. Mirrors
    ``tests/test_runtime_graph_cache.py``."""
    comms = [uuid4() for _ in range(n_communities)]
    nodes = [uuid4() for _ in range(n_communities * 3)]
    return CommunityAssignment(
        node_to_community={
            nodes[i]: comms[i // 3] for i in range(len(nodes))
        },
        community_centroids={c: [0.1, 0.2, 0.3] for c in comms},
        modularity=0.42,
        backend="leiden-networkx",
        top_communities=comms,
        mid_regions={c: nodes[i * 3:(i + 1) * 3] for i, c in enumerate(comms)},
    )


class _CrashAfterTmpWrite(OSError):
    """Sentinel exception class: a monkeypatched ``os.replace`` raises
    this to simulate a kernel panic between the ``.tmp`` write and the
    final rename. Production code catches ``OSError``, so the subclass
    flows through the existing ``except OSError`` branch in
    ``runtime_graph_cache.py``."""


# --------------------------------------------------------------------------- Test 1


def test_crash_after_tmp_write_leaves_previous_snapshot_intact(store, monkeypatch):
    """Reader sees the v1 snapshot after a v2 save crashes mid-rename.

    Sequence:
      1. Save a complete v1 snapshot — succeeds.
      2. Snapshot the on-disk bytes + ``try_load`` payload for v1.
      3. Monkeypatch ``os.replace`` so the next call raises
         ``_CrashAfterTmpWrite``.
      4. Save a v2 snapshot — returns ``False`` (the ``except OSError``
         branch handles the simulated crash).
      5. Verify: (a) the cache file still holds the v1 ciphertext
         byte-for-byte. (b) ``try_load`` still returns v1, not v2.
         (c) The ``.tmp`` sidecar has been cleaned up by the existing
         OSError-branch unlink at lines 633-637.
    """
    cache_path = runtime_graph_cache._cache_path(store)
    tmp_sidecar = cache_path.with_suffix(cache_path.suffix + ".tmp")

    # 1. v1 save.
    assignment_v1 = _make_assignment(n_communities=2)
    rich_club_v1 = [uuid4() for _ in range(3)]
    assert runtime_graph_cache.save(store, assignment_v1, rich_club_v1) is True

    # 2. Snapshot v1 state.
    v1_bytes = cache_path.read_bytes()
    v1_loaded = runtime_graph_cache.try_load(store)
    assert v1_loaded is not None
    v1_assignment, v1_rich, _v1_payload, _v1_maxdeg = v1_loaded
    assert v1_assignment.modularity == pytest.approx(0.42)
    assert set(v1_rich) == set(rich_club_v1)

    # 3. Crash-inject os.replace — but ONLY for the cache file. Patching
    #    ``runtime_graph_cache.os.replace`` swaps the global ``os`` singleton,
    #    so a lazy ``hnswlib`` ``.hnsw`` index flush (or any background atomic
    #    write) landing in this window would otherwise be crashed too. Scope
    #    the side-effect to ``dst == cache_path`` and pass everything else
    #    through to the real syscall so the test is order-independent.
    real_replace = os.replace

    def _crash_only_cache(src, dst, *a, **k):
        if str(dst) == str(cache_path):
            raise _CrashAfterTmpWrite("simulated crash")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(
        "iai_mcp.runtime_graph_cache.os.replace", _crash_only_cache
    )

    # 4. v2 save attempt — returns False on the simulated crash.
    assignment_v2 = _make_assignment(n_communities=5)
    rich_club_v2 = [uuid4() for _ in range(7)]
    assert runtime_graph_cache.save(store, assignment_v2, rich_club_v2) is False

    # 5a. The on-disk cache file is still the v1 ciphertext byte-for-byte.
    assert cache_path.read_bytes() == v1_bytes, (
        "v1 cache must remain intact after a v2 save crashes mid-rename"
    )
    # 5b. try_load still returns v1, not v2.
    after_crash = runtime_graph_cache.try_load(store)
    assert after_crash is not None
    after_assignment, after_rich, _ap, _am = after_crash
    assert after_assignment.modularity == pytest.approx(0.42)
    assert set(after_rich) == set(rich_club_v1)
    # 5c. The .tmp sidecar was cleaned up by the existing OSError branch.
    assert not tmp_sidecar.exists(), (
        ".tmp sidecar must be unlinked by the OSError-handler cleanup path"
    )


# --------------------------------------------------------------------------- Test 2


def test_save_calls_fsync_before_replace(store, monkeypatch):
    """``os.fsync`` is invoked BEFORE ``os.replace`` during ``save()``.

    Strategy: ``iai_mcp.runtime_graph_cache`` does ``import os``, so patching
    ``runtime_graph_cache.os.fsync`` / ``.os.replace`` patches the *global*
    ``os`` singleton — every ``os.replace`` / ``os.fsync`` anywhere in the
    process (including a lazy ``hnswlib`` index ``.hnsw`` flush left pending
    by a prior test's ``MemoryStore``, or any background writer) lands on the
    spy. A naive ``replace.call_count == 1`` / ``mock_calls.index("replace")``
    therefore flakes under random test order when an unrelated atomic write
    fires inside this window.

    To make the assertion contamination-proof we SCOPE both spies to the
    cache snapshot only:
      - the replace spy counts a call iff its destination is ``cache_path``;
      - the fsync spy counts a call iff the file descriptor's inode matches
        the ``.tmp`` source inode (the cache tmp file still exists at fsync
        time, pre-replace). Inode identity is portable across macOS APFS and
        Linux ext4.
    Unrelated ``os.replace`` (the ``.hnsw`` index) / ``os.fsync`` (any other
    fd) flow straight through to the real syscall and are NOT counted.

    On the un-modified source this stays the RED witness: zero cache-scoped
    ``fsync`` calls land because no ``os.fsync`` exists in the write block, so
    ``cache_fsync_count >= 1`` fails.
    """
    cache_path = runtime_graph_cache._cache_path(store)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")

    real_fsync = os.fsync
    real_replace = os.replace

    # Chronological order of cache-scoped events only.
    events: list[str] = []
    cache_fsync_count = 0
    cache_replace_count = 0

    def _scoped_fsync(fd: int) -> None:
        nonlocal cache_fsync_count
        try:
            same = os.fstat(fd).st_ino == os.stat(tmp_path).st_ino
        except OSError:
            same = False
        if same:
            cache_fsync_count += 1
            events.append("fsync")
        return real_fsync(fd)

    def _scoped_replace(src, dst, *a, **k):
        nonlocal cache_replace_count
        if str(dst) == str(cache_path):
            cache_replace_count += 1
            events.append("replace")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(
        "iai_mcp.runtime_graph_cache.os.fsync", _scoped_fsync
    )
    monkeypatch.setattr(
        "iai_mcp.runtime_graph_cache.os.replace", _scoped_replace
    )

    assignment = _make_assignment(n_communities=3)
    rich_club = [uuid4() for _ in range(4)]
    assert runtime_graph_cache.save(store, assignment, rich_club) is True

    # The cache fsync ran at least once; the cache replace ran exactly once.
    assert cache_fsync_count >= 1, (
        "os.fsync must be invoked on the cache tmp file before os.replace"
    )
    assert cache_replace_count == 1, (
        f"save() must os.replace the cache file exactly once; "
        f"got {cache_replace_count}"
    )

    # And the cache ``fsync`` precedes the cache ``replace`` chronologically.
    assert events.index("fsync") < events.index("replace"), (
        f"fsync must precede replace for the cache file; order was {events!r}"
    )

    # And the file is genuinely durable on disk after the round-trip.
    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None
    loaded_assignment, loaded_rich, _np, _md = loaded
    assert loaded_assignment.modularity == pytest.approx(0.42)
    assert set(loaded_rich) == set(rich_club)


# --------------------------------------------------------------------------- Test 3


def test_save_handles_fsync_failure_gracefully(store, monkeypatch):
    """``os.fsync`` raising ``OSError`` is swallowed and ``save`` returns
    ``False`` — no partial state lands on disk.

    Sequence:
      1. Save a v1 snapshot — succeeds. Remember its on-disk bytes.
      2. Replace ``os.fsync`` with a function that raises
         ``OSError("disk full")``.
      3. Call ``save()`` for v2 — must return ``False`` (the existing
         ``except OSError`` branch catches the fsync failure).
      4. Verify: (a) the cache file still has the v1 bytes (the failed
         save left the prior snapshot intact). (b) the ``.tmp`` sidecar
         was cleaned up.

    On the un-modified source ``os.fsync`` is never called inside
    ``save``, so the patched ``fsync`` never fires and the v2 save
    succeeds — the assertion ``save(...) is False`` is the RED witness.
    """
    cache_path = runtime_graph_cache._cache_path(store)
    tmp_sidecar = cache_path.with_suffix(cache_path.suffix + ".tmp")

    # 1. v1 baseline.
    assert runtime_graph_cache.save(
        store, _make_assignment(n_communities=2), [uuid4()]
    ) is True
    v1_bytes = cache_path.read_bytes()

    # 2. Disk-full simulation on fsync — scoped to the cache tmp file only.
    #    ``runtime_graph_cache.os.fsync`` is the global ``os`` singleton, so an
    #    unrelated fsync (e.g. a lazy ``.hnsw`` index flush from a prior test's
    #    store) firing in this window must NOT be crashed. Match the cache tmp
    #    file by inode (it exists at fsync time, pre-replace) and pass any
    #    other fd through to the real syscall.
    real_fsync = os.fsync

    def _raise_disk_full(fd: int) -> None:
        try:
            same = os.fstat(fd).st_ino == os.stat(tmp_sidecar).st_ino
        except OSError:
            same = False
        if same:
            raise OSError("disk full")
        return real_fsync(fd)

    monkeypatch.setattr(
        "iai_mcp.runtime_graph_cache.os.fsync", _raise_disk_full
    )

    # 3. v2 save attempt with fsync broken — save must return False.
    assert runtime_graph_cache.save(
        store, _make_assignment(n_communities=5), [uuid4() for _ in range(3)]
    ) is False

    # 4a. v1 ciphertext bytes still on disk.
    assert cache_path.read_bytes() == v1_bytes, (
        "v1 cache must remain intact when fsync fails on the v2 save"
    )
    # 4b. The .tmp sidecar was cleaned up.
    assert not tmp_sidecar.exists(), (
        ".tmp sidecar must be unlinked when fsync raises mid-save"
    )
