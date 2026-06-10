from __future__ import annotations

import sys
import time
from pathlib import Path
from uuid import UUID

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _recall_helpers import (  # noqa: E402
    EMBED_DIM,
    UUID_TWO_HOP_SURFACE,
    _deterministic_vec,
    _populate_store,
    _prime_structural_cache,
)

@pytest.fixture(autouse=True)
def _monkeypatch_env(monkeypatch, tmp_path: Path):
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_MCP_AROUSAL_USE_SHADOW", "1")
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
    yield

_FAKE_CUE = "User test cue for daemon-independent recall"

def _build_gold_store(tmp_path: Path) -> tuple:
    from iai_mcp.store import MemoryStore

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))
    cue_vec = _deterministic_vec(seed=99)
    _populate_store(store, cue_vec=cue_vec, n_filler=700)
    _prime_structural_cache(store)
    return store_root, store, cue_vec

class _FakeEmbedder:

    DIM = EMBED_DIM

    def __init__(self, vec: list[float]):
        self._vec = vec

    def embed(self, text: str) -> list[float]:
        return list(self._vec)

def _install_funnel(monkeypatch, embedder) -> None:
    import iai_mcp.embed as _embed_mod
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: embedder)

def test_daemon_down_first_call_returns_full_structural_gold(monkeypatch, tmp_path):
    from iai_mcp.pipeline import K_CANDIDATES
    import iai_mcp.semantic_recall as _sr
    from iai_mcp import core as _core_mod

    store_root, store, cue_vec = _build_gold_store(tmp_path)

    _install_funnel(monkeypatch, _FakeEmbedder(cue_vec))

    monkeypatch.setitem(_core_mod._profile_state, "literal_preservation", "medium")

    ann_top_k = {r.id for r, _ in store.query_similar(cue_vec, k=K_CANDIDATES)}
    assert UUID(int=5) not in ann_top_k, (
        f"PRECONDITION FAILED: UUID(5) is a DIRECT ANN top-{K_CANDIDATES} hit — "
        f"the 2-hop spread is not load-bearing; the gate would be hollow. "
        f"store size={store.active_records_count()}."
    )

    construct_calls: list[bool] = []
    _orig_construct = _sr._construct_with_budget

    def _spy_construct(root):
        construct_calls.append(True)
        return _orig_construct(root)

    monkeypatch.setattr(_sr, "_construct_with_budget", _spy_construct)

    _sr._WARM_LOCAL_STORE = None
    result = _sr.recall_semantic_warm(store_root, _FAKE_CUE, n=50)

    assert construct_calls, (
        "the construct path (_construct_with_budget) did NOT engage on the "
        "daemon-independent recall path — the daemon-down-full label would be "
        "force-stamped without the construct actually running"
    )

    assert result, "warm-construct daemon-independent recall returned empty hits"

    sources = {h.get("_source") for h in result}
    assert sources == {"daemon-down-full"}, (
        f"warm-construct path must tag EVERY hit _source='daemon-down-full' "
        f"exactly, got sources={sources!r}"
    )

    surfaces_with = {h.get("literal_surface", "") for h in result}
    assert UUID_TWO_HOP_SURFACE in surfaces_with, (
        f"STRUCTURAL PARITY GATE FAILED: UUID(5) hub-sensitive gold missing from "
        f"the warm-construct daemon-independent recall.\n"
        f"Got surfaces: {sorted(surfaces_with)}\n"
        f"2-hop spread (UUID(3)->UUID(4)->UUID(5)) must surface UUID(5); "
        f"UUID(5) cosine=0.02 is outside ANN top-{K_CANDIDATES} (precondition verified)."
    )

    control_hits = _sr._ann_only_daemon_down(store_root, cue_vec, 50, _FAKE_CUE, None)
    control_surfaces = {h.get("literal_surface", "") for h in control_hits}
    assert UUID_TWO_HOP_SURFACE not in control_surfaces, (
        "CONTROL FAILED: the ANN-only last-resort path (no 2-hop spread, no "
        "structural loader) surfaced UUID(5). The 2-hop / rich-club structural "
        "spread would not be the reason UUID(5) appears in the positive assertion."
    )

def test_daemon_down_construct_raises_degrades(monkeypatch, tmp_path):
    from iai_mcp.store import MemoryStore
    import iai_mcp.embed as _embed_mod
    import iai_mcp.semantic_recall as _sr
    import numpy as np
    from test_store import _make

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))
    rng = np.random.default_rng(7)
    for i in range(5):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.insert(_make(text=f"User record {i}", vec=v.tolist()))

    def _raising_funnel(_store):
        raise RuntimeError("simulated construct failure (model/weights missing)")

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _raising_funnel)

    _sr._WARM_LOCAL_STORE = None
    result = _sr.recall_semantic_warm(store_root, _FAKE_CUE, n=5)

    assert result, "construct-raise must still return a non-empty recency degrade"
    for row in result:
        assert row.get("_source") == "daemon-down-degrade", (
            f"construct-raise must degrade to daemon-down-degrade, got {row.get('_source')!r}"
        )

def test_daemon_down_smoke_encode_raises_degrades(monkeypatch, tmp_path):
    from iai_mcp.store import MemoryStore
    import iai_mcp.embed as _embed_mod
    import iai_mcp.semantic_recall as _sr
    import numpy as np
    from test_store import _make

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))
    rng = np.random.default_rng(8)
    for i in range(5):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.insert(_make(text=f"User record {i}", vec=v.tolist()))

    class _EncodeRaisesEmbedder:
        DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            raise RuntimeError("simulated encode failure")

    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: _EncodeRaisesEmbedder())

    _sr._WARM_LOCAL_STORE = None
    result = _sr.recall_semantic_warm(store_root, _FAKE_CUE, n=5)

    assert result, "smoke-encode-raise must still return a non-empty recency degrade"
    for row in result:
        assert row.get("_source") == "daemon-down-degrade", (
            f"smoke-encode-raise must degrade, got {row.get('_source')!r}"
        )

def test_daemon_down_construct_over_budget_degrades_promptly(monkeypatch, tmp_path):
    from iai_mcp.store import MemoryStore
    import iai_mcp.embed as _embed_mod
    import iai_mcp.semantic_recall as _sr
    import numpy as np
    from test_store import _make

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))
    rng = np.random.default_rng(9)
    for i in range(5):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.insert(_make(text=f"User record {i}", vec=v.tolist()))

    monkeypatch.setenv("IAI_MCP_EMBED_CONSTRUCT_BUDGET_MS", "100")
    SLOW_CONSTRUCT_S = 2.0

    class _SlowEmbedder:
        DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [0.0] * EMBED_DIM

    def _slow_funnel(_store):
        time.sleep(SLOW_CONSTRUCT_S)
        return _SlowEmbedder()

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _slow_funnel)

    _sr._WARM_LOCAL_STORE = None
    t0 = time.perf_counter()
    result = _sr.recall_semantic_warm(store_root, _FAKE_CUE, n=5)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, (
        f"over-budget construct must degrade PROMPTLY (join-with-timeout), "
        f"took {elapsed:.3f}s — must be << {SLOW_CONSTRUCT_S}s slow construct"
    )
    assert result, "over-budget construct must still return a non-empty recency degrade"
    for row in result:
        assert row.get("_source") == "daemon-down-degrade", (
            f"over-budget construct must degrade, got {row.get('_source')!r}"
        )

def test_daemon_down_smoke_encode_over_budget_degrades_promptly(monkeypatch, tmp_path):
    from iai_mcp.store import MemoryStore
    import iai_mcp.embed as _embed_mod
    import iai_mcp.semantic_recall as _sr
    import numpy as np
    from test_store import _make

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))
    rng = np.random.default_rng(10)
    for i in range(5):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.insert(_make(text=f"User record {i}", vec=v.tolist()))

    monkeypatch.setenv("IAI_MCP_EMBED_CONSTRUCT_BUDGET_MS", "100")
    SLOW_ENCODE_S = 2.0

    class _SlowEncodeEmbedder:
        DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            time.sleep(SLOW_ENCODE_S)
            return [0.0] * EMBED_DIM

    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: _SlowEncodeEmbedder())

    _sr._WARM_LOCAL_STORE = None
    t0 = time.perf_counter()
    result = _sr.recall_semantic_warm(store_root, _FAKE_CUE, n=5)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, (
        f"over-budget ENCODE must degrade PROMPTLY (budget covers construct+encode), "
        f"took {elapsed:.3f}s — must be << {SLOW_ENCODE_S}s slow encode"
    )
    assert result, "over-budget encode must still return a non-empty recency degrade"
    for row in result:
        assert row.get("_source") == "daemon-down-degrade", (
            f"over-budget encode must degrade, got {row.get('_source')!r}"
        )

def test_daemon_down_path_issues_no_embed_cue_rpc(monkeypatch, tmp_path):
    from iai_mcp.store import MemoryStore
    import iai_mcp.embed as _embed_mod
    import iai_mcp.semantic_recall as _sr
    import numpy as np
    from test_store import _make

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))
    rng = np.random.default_rng(11)
    for i in range(5):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.insert(_make(text=f"User record {i}", vec=v.tolist()))

    rpc_calls: list[bool] = []

    def _spy_rpc(cue, timeout_ms):
        rpc_calls.append(True)
        return None

    monkeypatch.setattr(_sr, "_send_embed_cue_rpc", _spy_rpc)

    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: _FakeEmbedder([0.0] * EMBED_DIM))

    _sr._WARM_LOCAL_STORE = None
    _sr.recall_semantic_warm(store_root, _FAKE_CUE, n=5)

    assert not rpc_calls, (
        "_send_embed_cue_rpc was called on the daemon-independent recall path — "
        "the redundant second RPC must be dropped from this path"
    )
