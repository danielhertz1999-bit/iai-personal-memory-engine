"""Event-buffer SIGKILL loss documentation test.

_event_buffer (events.py) is an in-process dict. It is NOT persisted on SIGKILL or OOM kill.
This is EXPECTED, DOCUMENTED behavior. This test asserts the loss and verifies post-restart
state is self-consistent. Do NOT add persistence — the in-process buffer with documented
hard-kill loss is the design contract.
"""
from __future__ import annotations

import pytest


# ----------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Standard project test isolation — verbatim from
    tests/test_pipeline_anti_hits_malformed.py. Without this fixture
    the test will fail on the construction host because the OS keyring is
    unavailable."""
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


# ----------------------------------------------------------- helpers


def _clear_buffer(store) -> None:
    """Pop any leftover buffer state for this store id."""
    from iai_mcp import events

    events._event_buffer.pop(id(store), None)
    events._last_flush_at.pop(id(store), None)


# ----------------------------------------------------------- Test 1


def test_event_buffer_sigkill_loss(tmp_path):
    """Buffered events are lost on SIGKILL (in-process dict, no persistence).

    Simulates SIGKILL by:
    1. Writing N events with buffered=True (events land in _event_buffer, NOT the store).
    2. Clearing _event_buffer without calling flush_event_buffer (models hard-kill loss).
    3. Re-opening MemoryStore at the same path (models post-restart process).

    DOCUMENTED LOSS: the N buffered events are not in the post-restart EVENTS table.
    This is EXPECTED, DOCUMENTED behavior — _event_buffer has no persistence mechanism.
    Documents this loss; no fix is planned (no-persistence-buffer is by design).

    Self-consistency assertion: the EVENTS table must be readable and schema-intact
    after restart (no corrupt rows, no schema damage).
    """
    from iai_mcp import events
    from iai_mcp.events import write_event
    from iai_mcp.store import EVENTS_TABLE, MemoryStore

    store = MemoryStore(path=tmp_path)
    _clear_buffer(store)
    N = 10

    # Write N buffered events (never flushed — simulates SIGKILL).
    # write_event signature: write_event(store, kind, data, *, buffered=False)
    # severity kwarg is optional keyword-only; omit here.
    for i in range(N):
        write_event(store, kind="test_sigkill_event", data={"seq": i}, buffered=True)

    # Assert buffer populated (events in-process only, not in the store).
    assert len(events._event_buffer.get(id(store), [])) == N, (
        f"expected {N} buffered events; got {len(events._event_buffer.get(id(store), []))}"
    )

    # Assert store table UNCHANGED (no rows added).
    events_tbl = store.db.open_table(EVENTS_TABLE)
    pre_kill_count = len(events_tbl.to_pandas())
    # Buffer is non-empty but table row count has not changed.
    # (There may be pre-existing rows from MemoryStore init — capture baseline, not assume 0.)

    # Simulate SIGKILL: abandon store without flush.
    # In a real SIGKILL the process is terminated; here we model it by
    # dereferencing the store. The _event_buffer entry is keyed by id(store);
    # after del store, the old id(store) key remains until GC or explicit clear.
    # For deterministic behavior, also clear the buffer to simulate memory loss.
    store_id_before = id(store)
    del store
    events._event_buffer.pop(store_id_before, None)

    # Simulate restart: new process opens same data directory.
    store2 = MemoryStore(path=tmp_path)
    post_restart_count = len(store2.db.open_table(EVENTS_TABLE).to_pandas())

    # DOCUMENTED LOSS: buffered events did not persist across restart.
    assert post_restart_count == pre_kill_count, (
        f"SIGKILL data loss confirmed: {N} buffered events lost on hard kill. "
        f"pre={pre_kill_count}, post={post_restart_count}. "
        "This is EXPECTED behavior — _event_buffer has no persistence mechanism. "
        "Documents this loss; no fix is planned (no-persistence-buffer is by design)."
    )

    # Self-consistency check: post-restart store is not corrupted.
    # Table must be accessible and schema intact.
    tbl_after = store2.db.open_table(EVENTS_TABLE)
    schema_names = tbl_after.schema.names
    assert "id" in schema_names, "EVENTS table schema corrupted after simulated SIGKILL"
    assert "kind" in schema_names, "EVENTS table schema corrupted after simulated SIGKILL"
