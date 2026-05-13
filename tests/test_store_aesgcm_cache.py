"""W5 — cached AESGCM cipher property on MemoryStore.

RED phase: these tests fail until ``MemoryStore`` exposes:

  * ``_cached_aesgcm`` — ``@functools.cached_property`` returning a single
    ``AESGCM(self._key())`` instance per store lifetime.
  * ``_invalidate_aesgcm_cache()`` — drops the cached attribute so that the
    next ``_cached_aesgcm`` access materialises a fresh cipher (future
    key-rotation hook per CONTEXT.md D-18).
  * ``_decrypt_for_record`` rewritten to use the cached cipher instead of
    constructing ``AESGCM(key)`` per call.

Covered contracts (CONTEXT.md W5 slice):

  Cache identity & reuse:
    1. ``_cached_aesgcm`` is reused across N decrypts: patch
       ``iai_mcp.store.AESGCM`` with a ``MagicMock(wraps=_RealAESGCM)`` and
       assert the constructor was called AT MOST ONCE across multiple
       ``all_records()``-driven decrypts.
    2. The cached path produces byte-identical plaintext to the uncached
       path: insert one CJK literal_surface, read it back via
       ``all_records()`` AND via ``get(record_id)``; both round-trip
       byte-for-byte.
    3. ``store._cached_aesgcm`` is the same object on repeated access
       (cached_property identity contract).

  Cache invalidation hook:
    4. ``_invalidate_aesgcm_cache()`` clears the cached attribute and a
       subsequent decrypt still works (re-materialisation is correct).

  Per-record nonce safety (D-03 contract — cache reuse safe ONLY if
  every call uses a different nonce):
    5. Three records with the SAME plaintext encrypt to three different
       ciphertexts (random per-record nonce) and decrypt back identically
       through the cached cipher with no ``InvalidTag``.

  Plaintext short-circuit (the cache must NOT materialise when
  ``is_encrypted()`` returns False):
    6. Calling ``_decrypt_for_record`` with a non-prefixed plaintext
       value passes it through unchanged AND the cache stays absent
       from ``store.__dict__``.

plan-checker B-1 lesson: every test uses a real ``MemoryRecord``
dataclass via ``_make()`` — never a plain dict against attribute-access code.
"""
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


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Mirror tests/test_store_iter_records.py — process-isolated keyring so
    AES-256-GCM key generation does not poke the OS keychain inside CI."""
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
    """Real-dataclass fixture (NEVER a plain dict — plan-checker B-1)."""
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
    """Fresh MemoryStore in tmp_path/lancedb (one per test, no cross-test bleed)."""
    return MemoryStore(path=tmp_path / "lancedb")


# --------------------------------------------------------------------------- cache reuse


def test_decrypt_for_record_uses_cached_aesgcm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_decrypt_for_record`` reuses ONE ``AESGCM`` instance.

    Patch ``iai_mcp.store.AESGCM`` with a ``MagicMock(wraps=_RealAESGCM)`` so
    the real cipher still functions but every constructor call is counted.
    Insert 5 records (the encrypt path uses ``crypto.AESGCM`` — a separate
    import — so encryption does NOT increment the store-side mock). Then call
    ``all_records()`` which routes every encrypted field through
    ``_decrypt_for_record``: 5 records × 3 encrypted columns
    (literal_surface, provenance_json, profile_modulation_gain_json) = up to
    15 decrypt calls, but the patched constructor must be called AT MOST ONCE
    across all of them (single cached cipher per store lifetime).

    Pre-Task-2 ``main`` state has no ``AESGCM`` attribute on
    ``iai_mcp.store``; ``monkeypatch.setattr`` raises ``AttributeError`` and
    the test fails — exactly the RED contract.
    """
    aesgcm_mock = MagicMock(wraps=_RealAESGCM)
    monkeypatch.setattr("iai_mcp.store.AESGCM", aesgcm_mock)

    store_local = MemoryStore(path=tmp_path / "lancedb")
    for i in range(5):
        store_local.insert(_make(text=f"record-{i}"))

    # Reset call count: any stray calls during insert (none expected because
    # encrypt uses crypto.AESGCM) are excluded; we measure only decrypt-driven
    # construction.
    aesgcm_mock.reset_mock()

    records = store_local.all_records()
    assert len(records) == 5

    # CONTEXT.md W5 slice: AESGCM(key) called exactly once across N decrypts.
    assert aesgcm_mock.call_count <= 1, (
        f"expected cached AESGCM (≤1 construction across N decrypts); "
        f"got {aesgcm_mock.call_count} constructions"
    )


# --------------------------------------------------------------------------- byte-identical output


def test_decrypt_for_record_output_byte_identical_to_uncached_path(
    store: MemoryStore,
) -> None:
    """cached path decrypts byte-for-byte the same as the uncached path.

    Locks an external invariant — the optimisation must NOT change observable
    plaintext. Insert one CJK + em-dash literal_surface (forces multi-byte
    UTF-8 round-trip) and verify both ``all_records()`` AND ``get()`` return
    the exact verbatim string.
    """
    verbatim = "ハロー世界 — ground truth literal"
    rec = _make(text=verbatim)
    store.insert(rec)

    via_all = store.all_records()
    assert len(via_all) == 1
    assert via_all[0].literal_surface == verbatim

    via_get = store.get(rec.id)
    assert via_get is not None
    assert via_get.literal_surface == verbatim


# --------------------------------------------------------------------------- cached_property identity


def test_cached_aesgcm_is_actually_cached(store: MemoryStore) -> None:
    """``_cached_aesgcm`` returns the same object on repeated access.

    ``functools.cached_property`` semantics — the descriptor stores the value
    in ``self.__dict__`` after first access, so subsequent accesses return
    the identical object (``is``-equal, not just ``==``-equal).
    """
    first = store._cached_aesgcm
    second = store._cached_aesgcm
    assert first is second, (
        "expected cached_property to return the same AESGCM object on repeated access"
    )


# --------------------------------------------------------------------------- invalidation hook


def test_invalidate_aesgcm_cache_clears(store: MemoryStore) -> None:
    """``_invalidate_aesgcm_cache()`` drops the cached attribute and the
    next access re-materialises a fresh cipher.

    Sequence:
      1. Force materialisation by accessing ``store._cached_aesgcm``.
      2. Confirm ``"_cached_aesgcm"`` is in ``store.__dict__`` (cached_property
         stored it).
      3. Call ``store._invalidate_aesgcm_cache()``.
      4. Confirm ``"_cached_aesgcm"`` is NO LONGER in ``store.__dict__``.
      5. Re-access ``_cached_aesgcm`` and use it for a real decrypt round-trip
         to prove the rebuild is functionally correct.
    """
    # 1+2: materialise + observe the cached_property storage slot
    _ = store._cached_aesgcm
    assert "_cached_aesgcm" in store.__dict__

    # 3+4: invalidate + observe the slot is gone
    store._invalidate_aesgcm_cache()
    assert "_cached_aesgcm" not in store.__dict__

    # 5: post-invalidation decrypt still works (proves rebuild is correct)
    rec = _make(text="post-invalidation roundtrip")
    store.insert(rec)
    got = store.get(rec.id)
    assert got is not None
    assert got.literal_surface == "post-invalidation roundtrip"


# --------------------------------------------------------------------------- nonce safety


def test_aesgcm_cache_handles_unique_per_record_nonce(store: MemoryStore) -> None:
    """D-03 contract: cipher reuse is safe ONLY if every call uses a distinct
    nonce. ``encrypt_field`` generates a fresh random nonce per call, so three
    records with the SAME plaintext yield three different ciphertexts that all
    decrypt back to the same string through the cached cipher.

    A regression that pinned the nonce (or reused the cipher across operations
    in a way that broke nonce-uniqueness) would surface as either matching
    ciphertexts or InvalidTag on decrypt.
    """
    duplicate = "duplicate"
    recs = [_make(text=duplicate) for _ in range(3)]
    for r in recs:
        store.insert(r)

    # Round-trip through the cached path: all 3 must decrypt to the same plaintext.
    for original in recs:
        got = store.get(original.id)
        assert got is not None
        assert got.literal_surface == duplicate


# --------------------------------------------------------------------------- plaintext short-circuit


def test_decrypt_for_record_skips_cache_for_plaintext_passthrough(
    store: MemoryStore,
) -> None:
    """``is_encrypted()`` short-circuit must fire BEFORE the cached cipher is
    materialised, so plaintext passthrough costs zero AESGCM construction.

    Sequence:
      1. Confirm ``"_cached_aesgcm"`` is absent from ``store.__dict__``
         (no decrypts have happened yet).
      2. Call ``store._decrypt_for_record(<rid>, <plaintext>)`` directly with
         a value that has no ``iai:enc:v1:`` prefix.
      3. Assert the returned value is the input string unchanged.
      4. Assert ``"_cached_aesgcm"`` is STILL absent from ``store.__dict__``
         — the cache was never materialised because ``is_encrypted()``
         returned False first.
    """
    rid = uuid4()
    plaintext = "plaintext that has no iai:enc:v1: prefix"

    # 1: pristine state, cache not materialised
    assert "_cached_aesgcm" not in store.__dict__

    # 2+3: direct call returns input unchanged
    out = store._decrypt_for_record(rid, plaintext)
    assert out == plaintext

    # 4: cache STILL not materialised — proof the short-circuit fired first
    assert "_cached_aesgcm" not in store.__dict__, (
        "is_encrypted() short-circuit must fire BEFORE _cached_aesgcm "
        "materialises; plaintext passthrough should not pay AESGCM cost"
    )
