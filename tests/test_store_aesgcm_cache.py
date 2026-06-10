from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _RealAESGCM

from iai_mcp.crypto import CIPHERTEXT_PREFIX
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


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


def _make(
    text: str = "hello world",
    tier: str = "episodic",
    tags: list[str] | None = None,
    detail: int = 2,
    language: str = "en",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=detail,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=(detail >= 3),
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=tags if tags is not None else [],
        language=language,
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "hippo")


def test_decrypt_for_record_uses_cached_aesgcm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aesgcm_mock = MagicMock(wraps=_RealAESGCM)
    monkeypatch.setattr("iai_mcp.store.AESGCM", aesgcm_mock)

    store_local = MemoryStore(path=tmp_path / "hippo")
    for i in range(5):
        store_local.insert(_make(text=f"record-{i}"))

    aesgcm_mock.reset_mock()

    records = store_local.all_records()
    assert len(records) == 5

    assert aesgcm_mock.call_count <= 1, (
        f"expected cached AESGCM (≤1 construction across N decrypts); "
        f"got {aesgcm_mock.call_count} constructions"
    )


def test_decrypt_for_record_output_byte_identical_to_uncached_path(
    store: MemoryStore,
) -> None:
    verbatim = "ハロー世界 — ground truth literal"
    rec = _make(text=verbatim)
    store.insert(rec)

    via_all = store.all_records()
    assert len(via_all) == 1
    assert via_all[0].literal_surface == verbatim

    via_get = store.get(rec.id)
    assert via_get is not None
    assert via_get.literal_surface == verbatim


def test_cached_aesgcm_is_actually_cached(store: MemoryStore) -> None:
    first = store._cached_aesgcm
    second = store._cached_aesgcm
    assert first is second, (
        "expected cached_property to return the same AESGCM object on repeated access"
    )


def test_invalidate_aesgcm_cache_clears(store: MemoryStore) -> None:
    _ = store._cached_aesgcm
    assert "_cached_aesgcm" in store.__dict__

    store._invalidate_aesgcm_cache()
    assert "_cached_aesgcm" not in store.__dict__

    rec = _make(text="post-invalidation roundtrip")
    store.insert(rec)
    got = store.get(rec.id)
    assert got is not None
    assert got.literal_surface == "post-invalidation roundtrip"


def test_aesgcm_cache_handles_unique_per_record_nonce(store: MemoryStore) -> None:
    duplicate = "duplicate"
    recs = [_make(text=duplicate) for _ in range(3)]
    for r in recs:
        store.insert(r)

    for original in recs:
        got = store.get(original.id)
        assert got is not None
        assert got.literal_surface == duplicate


def test_decrypt_for_record_skips_cache_for_plaintext_passthrough(
    store: MemoryStore,
) -> None:
    rid = uuid4()
    plaintext = "plaintext that has no iai:enc:v1: prefix"

    assert "_cached_aesgcm" not in store.__dict__

    out = store._decrypt_for_record(rid, plaintext)
    assert out == plaintext

    assert "_cached_aesgcm" not in store.__dict__, (
        "is_encrypted() short-circuit must fire BEFORE _cached_aesgcm "
        "materialises; plaintext passthrough should not pay AESGCM cost"
    )
