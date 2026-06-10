from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
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


def _clear_buffer(store) -> None:
    from iai_mcp import events

    events._event_buffer.pop(id(store), None)
    events._last_flush_at.pop(id(store), None)


def test_event_buffer_sigkill_loss(tmp_path):
    from iai_mcp import events
    from iai_mcp.events import write_event
    from iai_mcp.store import EVENTS_TABLE, MemoryStore

    store = MemoryStore(path=tmp_path)
    _clear_buffer(store)
    N = 10

    for i in range(N):
        write_event(store, kind="test_sigkill_event", data={"seq": i}, buffered=True)

    assert len(events._event_buffer.get(id(store), [])) == N, (
        f"expected {N} buffered events; got {len(events._event_buffer.get(id(store), []))}"
    )

    events_tbl = store.db.open_table(EVENTS_TABLE)
    pre_kill_count = len(events_tbl.to_pandas())

    store_id_before = id(store)
    del store
    events._event_buffer.pop(store_id_before, None)

    store2 = MemoryStore(path=tmp_path)
    post_restart_count = len(store2.db.open_table(EVENTS_TABLE).to_pandas())

    assert post_restart_count == pre_kill_count, (
        f"SIGKILL data loss confirmed: {N} buffered events lost on hard kill. "
        f"pre={pre_kill_count}, post={post_restart_count}. "
        "This is EXPECTED behavior — _event_buffer has no persistence mechanism. "
        "Documents this loss; no fix is planned (no-persistence-buffer is by design)."
    )

    tbl_after = store2.db.open_table(EVENTS_TABLE)
    schema_names = tbl_after.schema.names
    assert "id" in schema_names, "EVENTS table schema corrupted after simulated SIGKILL"
    assert "kind" in schema_names, "EVENTS table schema corrupted after simulated SIGKILL"
