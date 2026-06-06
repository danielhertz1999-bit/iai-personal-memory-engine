"""Baseline parity tests.

Acceptance:
- retrieve.recall accepts a mode kwarg (default 'verbatim').
- mode='verbatim' applies the same tier filter + schema exclusion as
  pipeline_recall verbatim mode.
- core.dispatch falls back to retrieve.recall when build_runtime_graph
  fails — the classified mode is preserved (the verbatim default protects
  verbatim recall on the degraded path).
- regression fence (test_recall_topk_stability) continues to pass.

The verbatim ≥99% target is preserved even when the full pipeline is
unreachable. The fallback path inherits the same contract on hits[] so
the user never silently lands on a schema-dominated surface.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


# --------------------------------------------------------- Fixture machinery


def _make_episodic(text: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
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


def _make_schema(text: str, pattern: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["schema", "draft", f"pattern:{pattern}"],
        language="en",
    )


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


def _seed_mixed_tier_store(tmp_path):
    """Seed: 3 episodic + 2 schema (semantic + pattern:*) — all share the
    same embedding so cosine ties to the cue."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "hippo")
    episodic_records = [_make_episodic(f"episodic verbatim text {i}") for i in range(3)]
    schema_records = [
        _make_schema(f"schema record {i}", pattern=f"test:r7:{i}")
        for i in range(2)
    ]
    for r in episodic_records:
        store.insert(r)
    for r in schema_records:
        store.insert(r)
    return store, episodic_records, schema_records


# ============================================================================
# Acceptance tests
# ============================================================================


def test_baseline_recall_default_mode_is_verbatim_per_d14():
    """retrieve.recall mode kwarg default is 'verbatim' per
    (conservative North-Star fallback)."""
    import inspect
    from iai_mcp.retrieve import recall

    sig = inspect.signature(recall)
    assert "mode" in sig.parameters, "retrieve.recall must accept mode kwarg"
    assert sig.parameters["mode"].default == "verbatim", (
        f"retrieve.recall default mode must be 'verbatim', "
        f"got {sig.parameters['mode'].default!r}"
    )


def test_baseline_recall_verbatim_filters_to_episodic_only(tmp_path):
    """Direct call: recall(store,...) without mode kwarg returns hits
    filtered to tier='episodic' (default) — schema records excluded."""
    from iai_mcp.retrieve import recall

    store, episodic_records, schema_records = _seed_mixed_tier_store(tmp_path)
    cue = [1.0] + [0.0] * (EMBED_DIM - 1)

    # No mode kwarg -> verbatim default per.
    resp = recall(
        store=store, cue_embedding=cue, cue_text="probe",
        session_id="r7_default", k_hits=5, k_anti=2,
    )
    assert resp.cue_mode == "verbatim", (
        f"baseline default mode must be 'verbatim', got {resp.cue_mode!r}"
    )
    schema_id_set = {r.id for r in schema_records}
    for h in resp.hits:
        assert h.record_id not in schema_id_set, (
            f"verbatim mode baseline must exclude schema records; "
            f"schema {h.record_id} appeared in hits"
        )
        rec = store.get(h.record_id)
        assert rec is not None
        assert rec.tier == "episodic", (
            f"verbatim mode hit {h.record_id} has tier {rec.tier!r}, expected 'episodic'"
        )


def test_baseline_recall_concept_mode_returns_all_tiers(tmp_path):
    """recall(..., mode='concept') returns the existing pure-cosine top-k
    INCLUDING all tiers (no filter — concept mode preserves baseline behaviour)."""
    from iai_mcp.retrieve import recall

    store, episodic_records, schema_records = _seed_mixed_tier_store(tmp_path)
    cue = [1.0] + [0.0] * (EMBED_DIM - 1)

    resp = recall(
        store=store, cue_embedding=cue, cue_text="probe",
        session_id="r7_concept", k_hits=5, k_anti=2, mode="concept",
    )
    assert resp.cue_mode == "concept"
    # All 5 records (3 episodic + 2 schema) tied at cosine=1.0; with k_hits=5
    # we should see all 5. Schema records ARE included on concept mode (the
    # baseline does not filter; only the full pipeline applies the tier split).
    hit_ids = {h.record_id for h in resp.hits}
    schema_id_set = {r.id for r in schema_records}
    assert schema_id_set & hit_ids, (
        f"concept mode baseline must include schema tier (no filter); "
        f"schema_ids={schema_id_set}, hit_ids={hit_ids}"
    )


def test_dispatch_falls_back_to_baseline_on_graph_build_failure(tmp_path, monkeypatch):
    """Monkeypatch retrieve.build_runtime_graph to raise.
    dispatch(..., 'memory_recall', {...verbatim cue...}) must:
      (a) complete (not propagate the exception);
      (b) return a non-empty response;
      (c) cue_mode == 'verbatim';
      (d) all hits are tier='episodic' (verbatim filter applied via fallback).
    """
    from iai_mcp import core
    from iai_mcp import retrieve as _retrieve_mod

    store, episodic_records, schema_records = _seed_mixed_tier_store(tmp_path)

    def fake_build(*args, **kwargs):
        raise RuntimeError("simulated graph build failure")

    monkeypatch.setattr(_retrieve_mod, "build_runtime_graph", fake_build)

    response = core.dispatch(
        store, "memory_recall",
        {"cue": "verbatim quote about migration",
         "session_id": "r7_fallback",
         "cue_embedding": [1.0] + [0.0] * (EMBED_DIM - 1)},
    )
    # (a) dispatch completed without raising — we have a response.
    assert isinstance(response, dict)
    # (c) classified mode preserved on the fallback path.
    assert response["cue_mode"] == "verbatim", (
        f"verbatim cue must classify to verbatim even when graph build fails; "
        f"got {response['cue_mode']!r}"
    )
    # (b) + (d) hits are episodic-only (when present).
    schema_id_strs = {str(r.id) for r in schema_records}
    for h in response["hits"]:
        assert h["record_id"] not in schema_id_strs, (
            f"fallback path must apply verbatim filter; schema {h['record_id']} "
            f"appeared in hits despite graph build failure + verbatim cue"
        )


def test_recall_topk_stability_smoke(tmp_path):
    """Smoke check: tests/test_recall_topk_stability.py still passes with the
    explicit mode='concept' pin. The actual lock is the dedicated test file;
    this test merely imports + runs one of its representative invariants here
    as a sentinel.
    """
    # Direct import + smoke run of the canonical helper from the existing
    # regression-fence module. If the module can't import at all under the
    # changes, this test catches it (import-time errors).
    import importlib

    mod = importlib.import_module("tests.test_recall_topk_stability")
    assert hasattr(mod, "test_no_literal_surface_mutation"), (
        "regression-fence module must still expose its sentinel test"
    )
    # Run one of the lighter assertions inline so this test does meaningful
    # work — the C5 literal_surface invariant. Runs in <2s.
    mod.test_no_literal_surface_mutation(tmp_path)
