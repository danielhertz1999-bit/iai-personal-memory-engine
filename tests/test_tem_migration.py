from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    import keyring as _keyring

    fake_store: dict[tuple[str, str], str] = {}

    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake_store.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake_store.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake_store.pop((s, u), None))
    yield fake_store

def _make_record(**overrides):
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    base = dict(
        id=uuid4(),
        tier="episodic",
        literal_surface="hello world",
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )
    base.update(overrides)
    return MemoryRecord(**base)

def test_bind_structure_returns_correct_byte_length(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.tem import bind_structure
    from iai_mcp.types import STRUCTURE_HV_BYTES

    rec = _make_record()
    hv = bind_structure(rec)
    assert isinstance(hv, bytes)
    assert len(hv) == STRUCTURE_HV_BYTES

def test_insert_fills_empty_structure_hv_via_bind_structure(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import STRUCTURE_HV_BYTES

    store = MemoryStore()
    rec = _make_record()
    assert rec.structure_hv == b""

    store.insert(rec)
    fetched = store.get(rec.id)
    assert fetched is not None
    assert fetched.structure_hv != b""
    assert len(fetched.structure_hv) == STRUCTURE_HV_BYTES

def test_insert_preserves_explicit_structure_hv(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import STRUCTURE_HV_BYTES

    store = MemoryStore()
    explicit = bytes([0xAB] * STRUCTURE_HV_BYTES)
    rec = _make_record(structure_hv=explicit)
    store.insert(rec)
    fetched = store.get(rec.id)
    assert fetched is not None
    assert fetched.structure_hv == explicit

def test_round_trip_structure_hv_through_lancedb(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import STRUCTURE_HV_BYTES

    store = MemoryStore()
    rec = _make_record(literal_surface="round-trip test")
    store.insert(rec)
    fetched = store.get(rec.id)
    assert fetched is not None
    assert isinstance(fetched.structure_hv, bytes)
    assert len(fetched.structure_hv) == STRUCTURE_HV_BYTES
    assert fetched.literal_surface == "round-trip test"
