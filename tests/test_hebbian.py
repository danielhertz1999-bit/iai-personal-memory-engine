"""Tests for Hebbian reinforcement, L0 seed, profile knobs, consolidate stub."""
from __future__ import annotations

from uuid import UUID

from iai_mcp.core import DEFERRED_KNOBS, L0_ID, LIVE_KNOBS, _seed_l0_identity, dispatch
from iai_mcp.store import MemoryStore
from tests.test_store import _make


def test_reinforce_creates_pairwise_edges(tmp_path):
    """C(3,2) = 3 pairwise edges on three-way co-retrieval."""
    store = MemoryStore(path=tmp_path)
    recs = [_make() for _ in range(3)]
    for r in recs:
        store.insert(r)
    ids = [str(r.id) for r in recs]
    result = dispatch(store, "memory_reinforce", {"ids": ids})
    assert result["edges_boosted"] == 3


def test_reinforce_twice_doubles_weight(tmp_path):
    """calling reinforce twice on same ids stacks the delta (0.1 + 0.1 = 0.2)."""
    store = MemoryStore(path=tmp_path)
    recs = [_make() for _ in range(2)]
    for r in recs:
        store.insert(r)
    ids = [str(r.id) for r in recs]
    dispatch(store, "memory_reinforce", {"ids": ids})
    r2 = dispatch(store, "memory_reinforce", {"ids": ids})
    assert len(r2["new_weights"]) == 1
    key = next(iter(r2["new_weights"]))
    assert abs(r2["new_weights"][key] - 0.2) < 1e-5


def test_l0_identity_seeded(tmp_path):
    """D-14 + pinned L0 record exists with immutability flags."""
    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    l0 = store.get(L0_ID)
    assert l0 is not None
    assert l0.pinned is True
    assert l0.never_decay is True
    assert l0.never_merge is True
    assert l0.detail_level == 5
    assert l0.tier == "semantic"
    assert "IAI-MCP" in l0.literal_surface


def test_l0_seed_is_idempotent(tmp_path):
    """Multiple boots of the core must not duplicate the L0 record."""
    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    _seed_l0_identity(store)
    _seed_l0_identity(store)
    all_records = store.all_records()
    l0_count = sum(1 for r in all_records if r.id == L0_ID)
    assert l0_count == 1


def test_profile_get_returns_live_knobs(tmp_path):
    """15 live (14 autistic-kernel + wake_depth MCP-12) + 0 deferred."""
    store = MemoryStore(path=tmp_path)
    result = dispatch(store, "profile_get", {})
    assert result["live"]["literal_preservation"] == "strong"        # AUTIST-04
    assert result["live"]["masking_off"] is True                     # AUTIST-06
    assert result["live"]["task_support"] == "cued_recognition"      # AUTIST-07
    assert result["live"]["scene_construction_scaffold"] is True     # AUTIST-14
    assert result["live"]["wake_depth"] == "minimal"                 # MCP-12
    # Plan 07.12-02: 10 autistic-kernel + wake_depth = 11 live (AUTIST-02/08/11/12 removed).
    assert len(result["live"]) == 11
    assert len(result["deferred"]) == 0


def test_profile_get_specific_live_knob(tmp_path):
    store = MemoryStore(path=tmp_path)
    result = dispatch(store, "profile_get", {"knob": "literal_preservation"})
    assert result["knob"] == "literal_preservation"
    assert result["value"] == "strong"


def test_profile_get_camouflaging_now_live_after_autist13_flip(tmp_path):
    """AUTIST-13 camouflaging_relaxation is live; profile_get returns value."""
    # Reset per-process state in case earlier tests (e.g. relax_register) moved the knob.
    import iai_mcp.core as core
    core._profile_state["camouflaging_relaxation"] = 0.0

    store = MemoryStore(path=tmp_path)
    result = dispatch(store, "profile_get", {"knob": "camouflaging_relaxation"})
    assert result["knob"] == "camouflaging_relaxation"
    assert result["value"] == 0.0  # D-AUTIST13 default


def test_profile_set_camouflaging_relaxation_now_succeeds(tmp_path):
    """camouflaging_relaxation is live; profile_set accepts in-range float."""
    store = MemoryStore(path=tmp_path)
    result = dispatch(store, "profile_set", {"knob": "camouflaging_relaxation", "value": 0.3})
    assert result["status"] == "ok"
    # Reset for other tests
    dispatch(store, "profile_set", {"knob": "camouflaging_relaxation", "value": 0.0})


def test_profile_set_live_knob_succeeds(tmp_path):
    """live knob accepts valid enum values ("loose" is in the schema)."""
    store = MemoryStore(path=tmp_path)
    # Reset default before test to avoid test ordering issues
    LIVE_KNOBS["literal_preservation"] = "strong"
    # Plan 03 introduced schema validation (enum:strong|medium|loose).
    # Plan 01 accepted any value; now we use a valid enum entry.
    result = dispatch(store, "profile_set", {"knob": "literal_preservation", "value": "loose"})
    assert result["status"] == "ok"
    assert LIVE_KNOBS["literal_preservation"] == "loose"
    # Restore so other tests aren't affected
    LIVE_KNOBS["literal_preservation"] = "strong"


def test_memory_consolidate_real(tmp_path):
    """Plan 02-02 memory_consolidate now runs real heavy consolidation.

    The stub returned {"status": "queued", "phase": "placeholder"};
    replaces that with actual sleep-cycle output:
    {"mode": "heavy", "tier": "tier0"|"tier1", "summaries_created": int,
     "decay_result": {...}, "schema_candidates": [...]}.
    """
    store = MemoryStore(path=tmp_path)
    result = dispatch(store, "memory_consolidate", {})
    assert result["mode"] == "heavy"
    assert result["tier"] in ("tier0", "tier1")
    assert "summaries_created" in result
    assert "decay_result" in result
    assert "schema_candidates" in result
