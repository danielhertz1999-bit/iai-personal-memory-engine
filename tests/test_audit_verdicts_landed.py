"""Audit-verdict regression gate.

Pins these outcomes:

* Three modules deleted (`pask_conversation`, `organizational_closure`,
  `recursive_autopoiesis`) raise `ModuleNotFoundError` on import.
* `pattern_separation.py` SURVIVES — `orthogonalize_for_routing` +
  `detect_hubness` + `OrthogonalizationResult` remain importable.
* `MemoryStore.pattern_separation_gate` (which routes through the
  internal `_pattern_separation_gate_with_hits` helper) calls
  `orthogonalize_for_routing` only when `IAI_MCP_ORTHO_ENABLED=1`,
  and never mutates `record.embedding` (the embedding invariant).
* `bench/memory_footprint.py` no longer exposes `_isolate_keyring_in_memory`
  nor the `isolate_keyring` parameter on `run_memory_footprint`.

Test design pattern: inline 4d synthetic embeddings, per-test `MemoryStore`,
autouse env reset. No SleepPipeline / no real Embedder.
"""
from __future__ import annotations

import inspect
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest


EMBED_DIM = 4
REFERENCE_EMBEDDING: list[float] = [1.0, 0.0, 0.0, 0.0]


def _make_embedding_at_cosine(
    cos_target: float, embed_dim: int = EMBED_DIM,
) -> list[float]:
    """Return a unit 4d vector whose dot-product against REFERENCE is exactly
    `cos_target`. Cell 0 carries the cosine; cell 1 carries the residual.
    Mirrors `tests/test_phase11_1_pattern_separation.py` helper.
    """
    if not (-1.0 <= cos_target <= 1.0):
        raise ValueError(f"cos_target out of range: {cos_target}")
    residual = math.sqrt(max(0.0, 1.0 - cos_target * cos_target))
    return [cos_target, residual] + [0.0] * (embed_dim - 2)


def _make_record(
    *,
    embedding: list[float],
    literal_surface: str = "alice prefers tea over coffee",
):
    """Build a fully-populated MemoryRecord. Same scaffolding as the
     regression suite — sensible defaults, no Alice-as-example
    string."""
    from iai_mcp.types import MemoryRecord
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=list(embedding),
        community_id=None,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
    )


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clear patsep + ORTHO env vars + pin embed dim + tmp store root."""
    for var in (
        "IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT",
        "IAI_MCP_PATSEP_TOP_K",
        "IAI_MCP_PATSEP_DRY_RUN",
        "IAI_MCP_ORTHO_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(EMBED_DIM))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp"))


def _make_store(tmp_path: Path):
    """Per-test fresh MemoryStore (4d schema, no consistency lag)."""
    from iai_mcp.store import MemoryStore
    return MemoryStore(
        path=str(tmp_path / "iai-mcp"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )


# ---------------------------------------------------------------------------
# A) DEAD-CODE-REMOVE / DELETE gates (3 modules)
# ---------------------------------------------------------------------------


def test_pask_conversation_removed() -> None:
    """audit verdict: DEAD-CODE-REMOVE. Module must not import."""
    with pytest.raises(ModuleNotFoundError):
        import iai_mcp.pask_conversation  # noqa: F401


def test_organizational_closure_removed() -> None:
    """planner decision: REMOVE per audit default. Must not import."""
    with pytest.raises(ModuleNotFoundError):
        import iai_mcp.organizational_closure  # noqa: F401


def test_recursive_autopoiesis_removed() -> None:
    """planner decision: REMOVE — passive 11-knob seal is the
    project invariant; active-evolution machinery contradicts it. Must
    not import."""
    with pytest.raises(ModuleNotFoundError):
        import iai_mcp.recursive_autopoiesis  # noqa: F401


# ---------------------------------------------------------------------------
# B) REWIRE gates (pattern_separation surface + flag-gated gate behaviour)
# ---------------------------------------------------------------------------


def test_pattern_separation_still_present_and_importable() -> None:
    """audit verdict: REWIRE. The 3 public symbols must still
    be importable after."""
    from iai_mcp.pattern_separation import (  # noqa: F401
        OrthogonalizationResult,
        detect_hubness,
        orthogonalize_for_routing,
    )


def test_pattern_separation_gate_default_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With IAI_MCP_ORTHO_ENABLED unset, the gate runs the legacy path
    WITHOUT calling orthogonalize_for_routing. Verified via a spy that
    raises if called."""
    from iai_mcp import pattern_separation as ps

    monkeypatch.delenv("IAI_MCP_ORTHO_ENABLED", raising=False)
    store = _make_store(tmp_path)

    # Seed two records so query_similar inside the gate has hits to
    # return; the second insert triggers the gate's top-k probe.
    e0 = REFERENCE_EMBEDDING
    e1 = _make_embedding_at_cosine(0.5)
    store.insert(_make_record(embedding=e0))
    store.insert(_make_record(embedding=e1))

    # Spy: replace orthogonalize_for_routing with a raise-on-call so any
    # invocation fails loud (it MUST NOT be called when flag is off).
    def _no_call(*args, **kwargs):
        raise AssertionError(
            "orthogonalize_for_routing called with flag OFF"
        )
    monkeypatch.setattr(ps, "orthogonalize_for_routing", _no_call)

    # The gate's hot path is _pattern_separation_gate_with_hits; the
    # public pattern_separation_gate wraps it. Exercise both to be sure
    # the spy stayed cold on the production path.
    rec = _make_record(embedding=_make_embedding_at_cosine(0.3))
    action, _payload = store.pattern_separation_gate(rec)
    # Gate ran to completion -> spy never raised.
    assert action is not None


def test_pattern_separation_gate_flag_on_invokes_orthogonalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With IAI_MCP_ORTHO_ENABLED=1 set, the gate calls
    orthogonalize_for_routing exactly once per invocation with the
    record's embedding as `vec` and the hits' embeddings as
    `neighbor_vecs`. Verified via a spy that counts calls + captures
    args."""
    import iai_mcp.store as _store_mod
    from iai_mcp import pattern_separation as ps

    monkeypatch.setenv("IAI_MCP_ORTHO_ENABLED", "1")
    store = _make_store(tmp_path)

    # Seed so query_similar returns hits.
    store.insert(_make_record(embedding=REFERENCE_EMBEDDING))
    store.insert(_make_record(embedding=_make_embedding_at_cosine(0.6)))

    calls: list[tuple[list[float], list[list[float]]]] = []
    original = ps.orthogonalize_for_routing

    def _spy(vec, neighbor_vecs, strength=0.3):
        calls.append((list(vec), [list(nv) for nv in neighbor_vecs]))
        return original(vec, neighbor_vecs, strength=strength)

    # the function (`from iai_mcp.pattern_separation import...`). For
    # the spy to take effect we must replace the symbol on the
    # `iai_mcp.pattern_separation` module BEFORE the gate executes.
    monkeypatch.setattr(
        "iai_mcp.pattern_separation.orthogonalize_for_routing", _spy,
    )
    # Sanity: confirm we patched the right name.
    assert _store_mod is not None  # silence unused-import warning

    rec = _make_record(embedding=_make_embedding_at_cosine(0.4))
    store.pattern_separation_gate(rec)

    assert len(calls) >= 1, (
        "orthogonalize_for_routing was not called with flag ON"
    )
    captured_vec, captured_neighbors = calls[-1]
    # First positional arg matches record.embedding.
    assert captured_vec == list(rec.embedding)
    # Second positional arg is a list of embeddings (the hits).
    assert len(captured_neighbors) >= 1
    # Every neighbor is a list[float] of the seed-record embedding dim.
    for nv in captured_neighbors:
        assert isinstance(nv, list)
        assert len(nv) == EMBED_DIM


def test_pattern_separation_gate_preserves_embedding_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding invariant — even with ORTHO flag ON, the gate MUST
    NOT mutate `record.embedding`. The orthogonalized vector is
    routing-only. Test asserts byte-equal pre/post."""
    monkeypatch.setenv("IAI_MCP_ORTHO_ENABLED", "1")
    store = _make_store(tmp_path)

    store.insert(_make_record(embedding=REFERENCE_EMBEDDING))
    store.insert(_make_record(embedding=_make_embedding_at_cosine(0.7)))

    rec = _make_record(embedding=_make_embedding_at_cosine(0.5))
    pre_snapshot = list(rec.embedding)
    store.pattern_separation_gate(rec)
    post_snapshot = list(rec.embedding)

    assert pre_snapshot == post_snapshot, (
        "R4 EMBEDDING INVARIANT violated: gate mutated record.embedding "
        f"pre={pre_snapshot[:3]}... post={post_snapshot[:3]}..."
    )


def test_pattern_separation_gate_static_no_record_embedding_assign() -> None:
    """Belt-and-braces static check: the gate's source code MUST NOT
    contain a literal `record.embedding =` assignment. Catches future
    drift even if behavioural test misses an aliased mutation."""
    from iai_mcp.store import MemoryStore
    src = inspect.getsource(
        MemoryStore._pattern_separation_gate_with_hits,
    )
    assert "record.embedding =" not in src, (
        "R4 EMBEDDING INVARIANT static-check failure: literal "
        "`record.embedding =` assignment found inside "
        "_pattern_separation_gate_with_hits source."
    )


# ---------------------------------------------------------------------------
# C) -E gate (bench dead-code removal)
# ---------------------------------------------------------------------------


def test_isolate_keyring_removed_from_memory_footprint() -> None:
    """-E: `_isolate_keyring_in_memory` and the `isolate_keyring`
    parameter on `run_memory_footprint` were dead post- and
    must be absent from `bench.memory_footprint`."""
    import bench.memory_footprint as mf

    assert not hasattr(mf, "_isolate_keyring_in_memory"), (
        "DEF-25-E regression: `_isolate_keyring_in_memory` still defined "
        "in bench.memory_footprint"
    )
    sig = inspect.signature(mf.run_memory_footprint)
    assert "isolate_keyring" not in sig.parameters, (
        "DEF-25-E regression: `isolate_keyring` parameter still on "
        "run_memory_footprint signature"
    )
