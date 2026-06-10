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

_EMBED_DIM = 384
_HV_BYTES = 1250

def _rand_emb(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()

def _open_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_KEYRING_BYPASS", "true")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-crossmodal-pp")
    return MemoryStore(path=tmp_path / "store")

def test_from_embedding_returns_1250_bytes() -> None:
    hv = from_embedding(_rand_emb(seed=1))
    assert isinstance(hv, bytes)
    assert len(hv) == _HV_BYTES

def test_from_embedding_deterministic() -> None:
    emb = _rand_emb(seed=42)
    assert from_embedding(emb) == from_embedding(emb)

def test_from_embedding_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="384"):
        from_embedding([0.1] * 100)
    with pytest.raises(ValueError, match="384"):
        from_embedding([])
    with pytest.raises(ValueError, match="384"):
        from_embedding([0.0] * 385)

def test_from_embedding_rejects_nonfinite() -> None:
    emb = _rand_emb(seed=7)
    bad = list(emb)
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        from_embedding(bad)
    bad2 = list(emb)
    bad2[100] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        from_embedding(bad2)

def test_from_embedding_values_are_0_or_1_bits() -> None:
    hv = from_embedding(_rand_emb(seed=3))
    packed = np.frombuffer(hv, dtype=np.uint8)
    bits = np.unpackbits(packed)
    unique = set(bits.tolist())
    assert unique <= {0, 1}
    assert len(bits) == 10000

def test_from_embedding_zero_emb_all_bits_one() -> None:
    emb = [0.0] * _EMBED_DIM
    hv = from_embedding(emb)
    packed = np.frombuffer(hv, dtype=np.uint8)
    bits = np.unpackbits(packed)
    assert np.all(bits == 1), "Zero embedding must produce all-ones HV"

def test_from_embedding_batch_store_none_no_telemetry() -> None:
    embs = [_rand_emb(seed=i) for i in range(RANK_DEFICIENCY_MIN_BATCH_SIZE + 5)]
    result = from_embedding_batch(embs, store=None)
    assert len(result) == len(embs)
    assert all(len(hv) == _HV_BYTES for hv in result)
    for emb, hv in zip(embs, result):
        assert from_embedding(emb) == hv

def test_from_embedding_batch_below_min_size_no_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

def test_from_embedding_batch_clustered_emits_rank_deficiency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iai_mcp.events import query_events

    store = _open_store(tmp_path, monkeypatch)
    try:
        base_emb = _rand_emb(seed=99)
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

def test_from_embedding_batch_healthy_below_threshold_no_emit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iai_mcp.events import query_events

    store = _open_store(tmp_path, monkeypatch)
    try:
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

def test_from_embedding_batch_explicit_low_threshold_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iai_mcp.events import query_events

    store = _open_store(tmp_path, monkeypatch)
    try:
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

def test_from_embedding_batch_returns_correct_count() -> None:
    for n in [0, 1, 5, 8, 20]:
        embs = [_rand_emb(seed=n + i) for i in range(n)]
        result = from_embedding_batch(embs)
        assert len(result) == n, f"Expected {n} results, got {len(result)}"

def test_to_embedding_neighbors_empty_store_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

def _make_record(emb: list[float], label: str):
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
    from iai_mcp.types import MemoryRecord

    store = _open_store(tmp_path, monkeypatch)
    try:
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
        for rec, score in results:
            assert isinstance(rec, MemoryRecord)
            assert isinstance(score, float)
    finally:
        try:
            store.close()
        except Exception:
            pass

def test_to_embedding_neighbors_wrong_hv_length_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

def test_rank_deficiency_min_batch_size_constant() -> None:
    assert RANK_DEFICIENCY_MIN_BATCH_SIZE == 8

def test_hv_dim_is_10000_bits() -> None:
    for seed in [0, 1, 42, 999]:
        hv = from_embedding(_rand_emb(seed=seed))
        assert len(hv) == 1250
        bits = np.unpackbits(np.frombuffer(hv, dtype=np.uint8))
        assert len(bits) == 10000

def test_from_embedding_batch_single_call_subset_matches_individual() -> None:
    embs = [_rand_emb(seed=400 + i) for i in range(12)]
    batch_hvs = from_embedding_batch(embs)
    for i, (emb, batch_hv) in enumerate(zip(embs, batch_hvs)):
        individual_hv = from_embedding(emb)
        assert batch_hv == individual_hv, (
            f"Batch result at index {i} does not match individual from_embedding call"
        )
