"""Gate: daemon-independent semantic recall — in-process construct under a budget.

The hippocampus is the always-available awake memory: semantic recall works
without the consolidation process. When the recall RPC has already failed, the
recall path CONSTRUCTS its own embedder synchronously under a join-with-timeout
budget guard (covering the construct AND one smoke-encode together) and feeds it
into the full structural-parity path -> _source="daemon-down-full". On a true
cold-disk construct (over budget) OR a construct/encode failure, it returns a
STORE-backed recency degrade NOW — the bypass-safe floor (never empty, never a
hard-fail, never blocking the full construct).

FOUR integrity proofs (causal — never a force-stamped label):
1. A STRUCTURAL-ONLY 2-hop gold UUID(5) is PRESENT in the warm result.
2. A hard PRECONDITION that UUID(5) is NOT in the ANN top-K (so the 2-hop spread
   is genuinely load-bearing — the gate has teeth).
3. An ANN-only CONTROL (_ann_only_daemon_down) that MISSES UUID(5) (proving the
   construct path fed the FULL structural pipeline, not ANN-only).
4. A construct-engaged SPY on _construct_with_budget (telemetry-independent) —
   the _source stamp alone can lie, so we prove the new construct path engaged.

Hermetic:
  - monkeypatched HOME / IAI_MCP_STORE -> tmp.
  - IAI_DAEMON_SOCKET_PATH absent (so the recall RPC fails fast and the
    daemon-independent branch is reached).
  - IAI_MCP_AROUSAL_USE_SHADOW=1 disables the rank_threshold filter so the
    structural spread is not gated by an arousal cosine threshold.
  - The embedder funnel (embedder_for_store) is monkeypatched in every test, so
    no real model load / network ever happens in the tmp-HOME env.
  - Generic User/tmp_path data; NEVER touching the live ~/.iai-mcp or daemon.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from uuid import UUID

import pytest

# Shared seeding + structural-cache helpers (also re-exported via conftest); the
# 63-04 real-subprocess gate reuses the SAME definitions to build the on-disk
# structural cache.
sys.path.insert(0, str(Path(__file__).parent))
from _recall_helpers import (  # noqa: E402
    EMBED_DIM,
    UUID_TWO_HOP_SURFACE,
    _deterministic_vec,
    _populate_store,
    _prime_structural_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _monkeypatch_env(monkeypatch, tmp_path: Path):
    """Redirect HOME / IAI_MCP_STORE and remove the daemon socket.

    Also sets IAI_MCP_AROUSAL_USE_SHADOW=1 so pipeline._recall_core uses the
    arousal_shadow route (rank_threshold=0.0). Without this, cues with
    hash[0]&1==1 trigger arousal_real with rank_threshold>=0.3 which filters
    UUID(5) (cosine=0.02) from the spread candidates — the hub-sensitive gold
    assertion requires that structural spread is not gated by an arousal cosine
    threshold. arousal_shadow is the documented test isolation for these gates.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_MCP_AROUSAL_USE_SHADOW", "1")
    # No IAI_DAEMON_SOCKET_PATH -> the recall RPC fails fast and the
    # daemon-independent construct branch is reached.
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_CUE = "User test cue for daemon-independent recall"


def _build_gold_store(tmp_path: Path) -> tuple:
    """Build a tmp store with hub-sensitive gold, prime the structural cache, and
    return (store_root, store, cue_vec) ready for the structural-parity gate.

    Built OFF-PATH (untimed). The cue_vec is deterministic so the fake embedder
    always returns the same vector -> the ANN lookup hits UUID(1)/UUID(3)
    near-neighbours and the 2-hop spread reaches UUID(5).

    n_filler=700: with standard_normal fillers and a mean-zero cue, the k=200 ANN
    cutoff is ~0.030, placing UUID(5) at cosine=0.02 OUTSIDE ANN top-200 so the
    2-hop spread UUID(3)->UUID(4)->UUID(5) is the ONLY way UUID(5) enters the pool.
    UUID(5)'s degree boost (50 real hebbian edges) gives it score=0.12 which ranks
    above standard_normal fillers at cosine~0.030.
    """
    from iai_mcp.store import MemoryStore

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))
    cue_vec = _deterministic_vec(seed=99)
    _populate_store(store, cue_vec=cue_vec, n_filler=700)
    _prime_structural_cache(store)
    return store_root, store, cue_vec


class _FakeEmbedder:
    """Returns a fixed cue vector. Its hits clear rank_threshold under shadow."""

    DIM = EMBED_DIM

    def __init__(self, vec: list[float]):
        self._vec = vec

    def embed(self, text: str) -> list[float]:
        return list(self._vec)


def _install_funnel(monkeypatch, embedder) -> None:
    """Monkeypatch the single embedder funnel to return `embedder` quickly.

    This replaces the real (cold, ~seconds) model construct with an instant stub,
    keeping every test hermetic in the tmp-HOME env (no model load, no network).
    """
    import iai_mcp.embed as _embed_mod
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: embedder)


# ---------------------------------------------------------------------------
# WARM CONSTRUCT -> daemon-down-full + the FOUR integrity proofs
# ---------------------------------------------------------------------------


def test_daemon_down_first_call_returns_full_structural_gold(monkeypatch, tmp_path):
    """Warm construct -> EXACT daemon-down-full + STRUCTURAL-ONLY UUID(5) present.

    FOUR integrity proofs in one test:
    (1) the STRUCTURAL-ONLY 2-hop gold UUID(5) is PRESENT;
    (2) PRECONDITION: UUID(5) is NOT in the ANN top-K (2-hop is load-bearing);
    (3) ANN-only CONTROL misses UUID(5);
    (4) a construct-engaged SPY on _construct_with_budget fired.

    Do NOT weaken, exclude UUID(5), or demote any proof to a warning.
    """
    from iai_mcp.pipeline import K_CANDIDATES
    import iai_mcp.semantic_recall as _sr
    from iai_mcp import core as _core_mod

    store_root, store, cue_vec = _build_gold_store(tmp_path)

    # Funnel returns the fake embedder fast -> construct + smoke-encode in budget.
    _install_funnel(monkeypatch, _FakeEmbedder(cue_vec))

    # literal_preservation="medium" — same as the bounded-assembler parity fixture.
    monkeypatch.setitem(_core_mod._profile_state, "literal_preservation", "medium")

    # --- PROOF 2 (PRECONDITION): UUID(5) is NOT a direct ANN hit ---
    ann_top_k = {r.id for r, _ in store.query_similar(cue_vec, k=K_CANDIDATES)}
    assert UUID(int=5) not in ann_top_k, (
        f"PRECONDITION FAILED: UUID(5) is a DIRECT ANN top-{K_CANDIDATES} hit — "
        f"the 2-hop spread is not load-bearing; the gate would be hollow. "
        f"store size={store.active_records_count()}."
    )

    # --- PROOF 4 (CONSTRUCT-ENGAGED SPY): wrap _construct_with_budget ---
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

    # --- EXACT source label: daemon-down-full (no loose negation) ---
    sources = {h.get("_source") for h in result}
    assert sources == {"daemon-down-full"}, (
        f"warm-construct path must tag EVERY hit _source='daemon-down-full' "
        f"exactly, got sources={sources!r}"
    )

    # --- PROOF 1 (STRUCTURAL-ONLY GOLD PRESENT): UUID(5) surface in result ---
    surfaces_with = {h.get("literal_surface", "") for h in result}
    assert UUID_TWO_HOP_SURFACE in surfaces_with, (
        f"STRUCTURAL PARITY GATE FAILED: UUID(5) hub-sensitive gold missing from "
        f"the warm-construct daemon-independent recall.\n"
        f"Got surfaces: {sorted(surfaces_with)}\n"
        f"2-hop spread (UUID(3)->UUID(4)->UUID(5)) must surface UUID(5); "
        f"UUID(5) cosine=0.02 is outside ANN top-{K_CANDIDATES} (precondition verified)."
    )

    # --- PROOF 3 (ANN-ONLY CONTROL): the ANN-only last-resort misses UUID(5) ---
    control_hits = _sr._ann_only_daemon_down(store_root, cue_vec, 50, _FAKE_CUE, None)
    control_surfaces = {h.get("literal_surface", "") for h in control_hits}
    assert UUID_TWO_HOP_SURFACE not in control_surfaces, (
        "CONTROL FAILED: the ANN-only last-resort path (no 2-hop spread, no "
        "structural loader) surfaced UUID(5). The 2-hop / rich-club structural "
        "spread would not be the reason UUID(5) appears in the positive assertion."
    )


# ---------------------------------------------------------------------------
# BYPASS-SAFE FLOOR: construct / encode raise -> recency degrade
# ---------------------------------------------------------------------------


def test_daemon_down_construct_raises_degrades(monkeypatch, tmp_path):
    """The construct (funnel) RAISES -> recency degrade, non-empty, never raises out."""
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
    """Construct succeeds but.embed() RAISES -> recency degrade.

    Proves the smoke-encode is INSIDE the budget guard: a half-usable embedder
    (constructs but cannot encode) is never handed to the structural path.
    """
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


# ---------------------------------------------------------------------------
# BYPASS-SAFE FLOOR: construct / encode over budget -> degrade PROMPTLY
# ---------------------------------------------------------------------------


def test_daemon_down_construct_over_budget_degrades_promptly(monkeypatch, tmp_path):
    """Construct (funnel) sleeps past the budget -> degrade returned PROMPTLY.

    Proves the join-with-timeout (NOT a post-hoc elapsed-check): the call must
    return well under the slow-construct sleep time.
    """
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

    # Low budget; a slow funnel that sleeps WELL past it (time.sleep releases GIL).
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

    # Must degrade in well under the slow construct time (budget 0.1 s + slack).
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
    """Construct returns fast but.embed() sleeps past the budget -> degrade PROMPTLY.

    Proves the budget covers construct + encode TOGETHER (HIGH-3 locked floor):
    an unbounded encode after a fast construct must NOT escape the guard.
    """
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
            time.sleep(SLOW_ENCODE_S)  # encode over budget (sleep releases GIL)
            return [0.0] * EMBED_DIM

    # Construct returns fast; the encode is what blows the budget.
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


# ---------------------------------------------------------------------------
# MEDIUM: no embed_cue RPC is issued on the daemon-independent path
# ---------------------------------------------------------------------------


def test_daemon_down_path_issues_no_embed_cue_rpc(monkeypatch, tmp_path):
    """The daemon-independent recall path issues NO _send_embed_cue_rpc call.

    The recall RPC has already failed before recall_semantic_warm is reached, so
    a second blocked RPC to the same socket is pure latency. Assert it is gone.
    """
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

    # A fast fake embedder so the path completes (either full or degrade).
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: _FakeEmbedder([0.0] * EMBED_DIM))

    _sr._WARM_LOCAL_STORE = None
    _sr.recall_semantic_warm(store_root, _FAKE_CUE, n=5)

    assert not rpc_calls, (
        "_send_embed_cue_rpc was called on the daemon-independent recall path — "
        "the redundant second RPC must be dropped from this path"
    )
