from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

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

def _make(store: MemoryStore, text: str = "hello") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * store.embed_dim,
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
        tags=[],
        language="en",
    )

def test_async_insert_persists_record(tmp_path: Path):
    store = MemoryStore(path=tmp_path)

    async def drive() -> None:
        await store.enable_async_writes(coalesce_ms=50, max_batch=128)
        try:
            r = _make(store, "async-insert-1")
            store.insert(r)
            got = store.get(r.id)
            assert got is not None
            assert got.literal_surface == "async-insert-1"
        finally:
            await store.disable_async_writes()

    asyncio.run(drive())

def test_sync_insert_unchanged_when_async_never_enabled(tmp_path: Path):
    store = MemoryStore(path=tmp_path)
    r = _make(store, "sync-only")
    store.insert(r)
    got = store.get(r.id)
    assert got is not None
    assert got.literal_surface == "sync-only"

def test_disable_async_writes_falls_back_to_sync(tmp_path: Path):
    store = MemoryStore(path=tmp_path)

    async def drive() -> None:
        await store.enable_async_writes(coalesce_ms=50, max_batch=128)
        r1 = _make(store, "async-phase")
        store.insert(r1)
        await store.disable_async_writes()
        r2 = _make(store, "sync-phase")
        store.insert(r2)
        assert store.get(r1.id) is not None
        assert store.get(r2.id) is not None

    asyncio.run(drive())

def test_graph_sync_hook_fires_per_record_under_async_writes(tmp_path: Path):
    store = MemoryStore(path=tmp_path)
    seen: list[tuple[str, str]] = []

    def hook(op: str, record: MemoryRecord) -> None:
        seen.append((op, str(record.id)))

    store.register_graph_sync_hook(hook)

    async def drive() -> list[str]:
        await store.enable_async_writes(coalesce_ms=80, max_batch=128)
        try:
            records = [_make(store, f"r{i}") for i in range(3)]
            await asyncio.gather(
                *(asyncio.to_thread(store.insert, r) for r in records)
            )
            return [str(r.id) for r in records]
        finally:
            await store.disable_async_writes()

    ids = asyncio.run(drive())
    hook_ids = [rid for (op, rid) in seen if op == "insert"]
    assert sorted(hook_ids) == sorted(ids)
    assert len(hook_ids) == 3
