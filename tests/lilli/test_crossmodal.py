"""Crossmodal bridge: embedding <-> hypervector tests.

Tests:
 1. test_from_embedding_returns_1250_bytes
 2. test_from_embedding_deterministic
 3. test_from_embedding_rejects_wrong_shape
 4. test_from_embedding_rejects_nonfinite
 5. test_from_embedding_values_are_0_or_1_bits
 6. test_from_embedding_zero_emb_all_bits_one
 7. test_from_embedding_batch_store_none_no_telemetry
 8. test_from_embedding_batch_below_min_size_no_telemetry
 9. test_from_embedding_batch_clustered_emits_rank_deficiency
10. test_from_embedding_batch_healthy_below_threshold_no_emit
11. test_from_embedding_batch_explicit_low_threshold_trips
12. test_from_embedding_batch_returns_correct_count
13. test_to_embedding_neighbors_empty_store_returns_empty
14. test_to_embedding_neighbors_populated_store_returns_k
15. test_to_embedding_neighbors_wrong_hv_length_returns_empty
16. test_rank_deficiency_min_batch_size_constant
17. test_hv_dim_is_10000_bits
18. test_from_embedding_batch_single_call_subset_matches_individual
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iai_mcp.lilli.crossmodal.embed_to_hv import (
    RANK_DEFICIENCY_DEFAULT_THRESHOLD,
    RANK_DEFICIENCY_MIN_BATCH_SIZE,
    _TELEMETRY_RANK_DEFICIENCY_KIND,
    from_embedding,
    from_embedding_batch,
    to_embedding_neighbors,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMBED_DIM = 384
_HV_BYTES = 1250  # 10000 bits / 8


def _rand_emb(seed: int = 0) -> list[float]:
    """Return a deterministic, normalised 384-dim float embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _open_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Open an isolated MemoryStore with keyring bypass.

    Env vars are set via monkeypatch so they are automatically reverted after
    each test, preventing module-level env leaks into the rest of the suite.
    """
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_KEYRING_BYPASS", "true")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-crossmodal-pp")
    return MemoryStore(path=tmp_path / "store")


# ---------------------------------------------------------------------------
# 1. from_embedding: output shape
# ---------------------------------------------------------------------------


def test_from_embedding_returns_1250_bytes() -> None:
    """from_embedding must return exactly 1250 bytes (10000-bit HV)."""
    hv = from_embedding(_rand_emb(seed=1))
    assert isinstance(hv, bytes)
    assert len(hv) == _HV_BYTES


# ---------------------------------------------------------------------------
# 2. from_embedding: determinism
# ---------------------------------------------------------------------------


def test_from_embedding_deterministic() -> None:
    """Same embedding input must produce identical bytes across two calls."""
    emb = _rand_emb(seed=42)
    assert from_embedding(emb) == from_embedding(emb)


# ---------------------------------------------------------------------------
# 3. from_embedding: wrong shape
# ---------------------------------------------------------------------------


def test_from_embedding_rejects_wrong_shape() -> None:
    """from_embedding must raise ValueError for non-384 input."""
    with pytest.raises(ValueError, match="384"):
        from_embedding([0.1] * 100)
    with pytest.raises(ValueError, match="384"):
        from_embedding([])
    with pytest.raises(ValueError, match="384"):
        from_embedding([0.0] * 385)


# ---------------------------------------------------------------------------
# 4. from_embedding: non-finite values
# ---------------------------------------------------------------------------


def test_from_embedding_rejects_nonfinite() -> None:
    """from_embedding must raise ValueError when embedding contains NaN or Inf."""
    emb = _rand_emb(seed=7)
    bad = list(emb)
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        from_embedding(bad)
    bad2 = list(emb)
    bad2[100] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        from_embedding(bad2)


# ---------------------------------------------------------------------------
# 5. from_embedding: packed bits are 0 or 1
# ---------------------------------------------------------------------------


def test_from_embedding_values_are_0_or_1_bits() -> None:
    """Unpacked bits from from_embedding must contain only 0s and 1s."""
    hv = from_embedding(_rand_emb(seed=3))
    packed = np.frombuffer(hv, dtype=np.uint8)
    bits = np.unpackbits(packed)
    unique = set(bits.tolist())
    assert unique <= {0, 1}
    assert len(bits) == 10000


# ---------------------------------------------------------------------------
# 6. from_embedding: zero embedding → all bits 1 (non-negative ties go to 1)
# ---------------------------------------------------------------------------


def test_from_embedding_zero_emb_all_bits_one() -> None:
    """A zero embedding maps to all non-negative projections → all bits 1."""
    emb = [0.0] * _EMBED_DIM
    # With emb=0, emb@P == 0 for every dimension — all should be >= 0 (tie → 1).
    hv = from_embedding(emb)
    packed = np.frombuffer(hv, dtype=np.uint8)
    bits = np.unpackbits(packed)
    # All ties (0.0 projections) resolve to bit=1
    assert np.all(bits == 1), "Zero embedding must produce all-ones HV"


# ---------------------------------------------------------------------------
# 7. from_embedding_batch with store=None: no telemetry, no crash
# ---------------------------------------------------------------------------


def test_from_embedding_batch_store_none_no_telemetry() -> None:
    """Batch with store=None must succeed and return correct list without any I/O."""
    embs = [_rand_emb(seed=i) for i in range(RANK_DEFICIENCY_MIN_BATCH_SIZE + 5)]
    result = from_embedding_batch(embs, store=None)
    assert len(result) == len(embs)
    assert all(len(hv) == _HV_BYTES for hv in result)
    # Cross-check: each hv matches individual from_embedding call
    for emb, hv in zip(embs, result):
        assert from_embedding(emb) == hv


# ---------------------------------------------------------------------------
# 8. Batch below RANK_DEFICIENCY_MIN_BATCH_SIZE: no telemetry even with store
# ---------------------------------------------------------------------------


def test_from_embedding_batch_below_min_size_no_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch smaller than RANK_DEFICIENCY_MIN_BATCH_SIZE (8) must not emit."""
    from iai_mcp.events import query_events

    store = _open_store(tmp_path, monkeypatch)
    try:
        embs = [_rand_emb(seed=i) for i in range(RANK_DEFICIENCY_MIN_BATCH_SIZE - 1)]
        assert len(embs) == RANK_DEFICIENCY_MIN_BATCH_SIZE - 1
        result = from_embedding_batch(embs, store=store)
        assert len(result) == len(embs)
        events_found = query_events(store, kind=_TELEMETRY_RANK_DEFICIENCY_KIND)
        assert len(events_found) == 0, (
            f"Expected no rank_deficiency events for batch size {len(embs)}, "
            f"got {len(events_found)}"
        )
    finally:
        try:
            store.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 9. Clustered batch (near-identical) emits rank_deficiency_warning
# ---------------------------------------------------------------------------


def test_from_embedding_batch_clustered_emits_rank_deficiency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch of near-identical embeddings must trip the deviation threshold."""
    from iai_mcp.events import query_events

    store = _open_store(tmp_path, monkeypatch)
    try:
        base_emb = _rand_emb(seed=99)
        # Use identical embeddings — deviation from 0.5 will be large (~0.5).
        embs = [base_emb] * (RANK_DEFICIENCY_MIN_BATCH_SIZE + 2)
        result = from_embedding_batch(embs, store=store)
        assert len(result) == len(embs)
        events_found = query_events(store, kind=_TELEMETRY_RANK_DEFICIENCY_KIND)
        assert len(events_found) >= 1, (
            f"Expected >= 1 rank_deficiency event for clustered batch, "
            f"got {len(events_found)}"
        )
        payload = events_found[0]["data"]
        assert payload["batch_size"] == len(embs)
        assert payload["deviation"] > RANK_DEFICIENCY_DEFAULT_THRESHOLD
        assert payload["hv_dim"] == 10000
    finally:
        try:
            store.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 10. Healthy diverse embeddings below default threshold: no emit
# ---------------------------------------------------------------------------


def test_from_embedding_batch_healthy_below_threshold_no_emit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Diverse random embeddings must NOT trip the default threshold of 0.2."""
    from iai_mcp.events import query_events

    store = _open_store(tmp_path, monkeypatch)
    try:
        # 16 diverse random embeddings; their HV bits should be close to 50/50.
        embs = [_rand_emb(seed=100 + i) for i in range(16)]
        result = from_embedding_batch(embs, store=store)
        assert len(result) == 16
        events_found = query_events(store, kind=_TELEMETRY_RANK_DEFICIENCY_KIND)
        assert len(events_found) == 0, (
            f"Expected no rank_deficiency events for diverse batch, "
            f"got {len(events_found)}"
        )
    finally:
        try:
            store.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 11. Low explicit threshold trips on healthy batch (Gate 15(b) contract)
# ---------------------------------------------------------------------------


def test_from_embedding_batch_explicit_low_threshold_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """deviation_threshold=0.01 must trip even on a healthy diverse batch."""
    from iai_mcp.events import query_events

    store = _open_store(tmp_path, monkeypatch)
    try:
        # Random diverse batch; deviation should be around 0.13 (well above 0.01).
        embs = [_rand_emb(seed=200 + i) for i in range(RANK_DEFICIENCY_MIN_BATCH_SIZE + 4)]
        from_embedding_batch(embs, store=store, deviation_threshold=0.01)
        events_found = query_events(store, kind=_TELEMETRY_RANK_DEFICIENCY_KIND)
        assert len(events_found) >= 1, (
            f"Expected >= 1 rank_deficiency event with threshold=0.01, "
            f"got {len(events_found)}"
        )
        payload = events_found[0]["data"]
        assert payload["threshold"] == pytest.approx(0.01)
    finally:
        try:
            store.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 12. Batch returns correct count
# ---------------------------------------------------------------------------


def test_from_embedding_batch_returns_correct_count() -> None:
    """from_embedding_batch must return exactly len(embs) hypervectors."""
    for n in [0, 1, 5, 8, 20]:
        embs = [_rand_emb(seed=n + i) for i in range(n)]
        result = from_embedding_batch(embs)
        assert len(result) == n, f"Expected {n} results, got {len(result)}"


# ---------------------------------------------------------------------------
# 13. to_embedding_neighbors: empty store returns empty list
# ---------------------------------------------------------------------------


def test_to_embedding_neighbors_empty_store_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """to_embedding_neighbors on an empty store must return []."""
    store = _open_store(tmp_path, monkeypatch)
    try:
        hv = from_embedding(_rand_emb(seed=5))
        result = to_embedding_neighbors(hv, store, k=5)
        assert isinstance(result, list)
        assert len(result) == 0
    finally:
        try:
            store.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 14. to_embedding_neighbors: populated store returns up to k neighbors
# ---------------------------------------------------------------------------


def _make_record(emb: list[float], label: str):
    """Create a minimal valid MemoryRecord for testing."""
    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp.types import MemoryRecord

    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=label,
        aaak_index="",
        embedding=emb,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        language="en",
    )


def test_to_embedding_neighbors_populated_store_returns_k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With N records inserted, to_embedding_neighbors returns min(k, N) results."""
    from iai_mcp.types import MemoryRecord

    store = _open_store(tmp_path, monkeypatch)
    try:
        # Insert 5 records with random embeddings.
        n_records = 5
        for i in range(n_records):
            emb = _rand_emb(seed=300 + i)
            store.insert(_make_record(emb, f"record {i}"))

        hv = from_embedding(_rand_emb(seed=1))
        results = to_embedding_neighbors(hv, store, k=3)
        assert isinstance(results, list)
        assert 1 <= len(results) <= 3, (
            f"Expected 1..3 neighbors, got {len(results)}"
        )
        # Each result should be a (MemoryRecord, float) pair.
        for rec, score in results:
            assert isinstance(rec, MemoryRecord)
            assert isinstance(score, float)
    finally:
        try:
            store.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 15. to_embedding_neighbors: wrong HV length returns empty list
# ---------------------------------------------------------------------------


def test_to_embedding_neighbors_wrong_hv_length_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrong-length HV must return [] (graceful degradation, not exception)."""
    store = _open_store(tmp_path, monkeypatch)
    try:
        result = to_embedding_neighbors(b"\x00" * 512, store, k=5)
        assert result == []
        result2 = to_embedding_neighbors(b"", store, k=5)
        assert result2 == []
    finally:
        try:
            store.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 16. RANK_DEFICIENCY_MIN_BATCH_SIZE constant
# ---------------------------------------------------------------------------


def test_rank_deficiency_min_batch_size_constant() -> None:
    """RANK_DEFICIENCY_MIN_BATCH_SIZE must be 8."""
    assert RANK_DEFICIENCY_MIN_BATCH_SIZE == 8


# ---------------------------------------------------------------------------
# 17. HV dimension is 10000 bits
# ---------------------------------------------------------------------------


def test_hv_dim_is_10000_bits() -> None:
    """from_embedding must return 1250 bytes regardless of embedding content."""
    for seed in [0, 1, 42, 999]:
        hv = from_embedding(_rand_emb(seed=seed))
        assert len(hv) == 1250
        bits = np.unpackbits(np.frombuffer(hv, dtype=np.uint8))
        assert len(bits) == 10000


# ---------------------------------------------------------------------------
# 18. Batch subset consistency with individual calls
# ---------------------------------------------------------------------------


def test_from_embedding_batch_single_call_subset_matches_individual() -> None:
    """Each element in from_embedding_batch must match from_embedding for the same input."""
    embs = [_rand_emb(seed=400 + i) for i in range(12)]
    batch_hvs = from_embedding_batch(embs)
    for i, (emb, batch_hv) in enumerate(zip(embs, batch_hvs)):
        individual_hv = from_embedding(emb)
        assert batch_hv == individual_hv, (
            f"Batch result at index {i} does not match individual from_embedding call"
        )
