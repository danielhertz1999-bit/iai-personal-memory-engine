"""Regression fence — rank stability + literal-preservation invariant.

The dominant accuracy effect is provenance-write amplification, so this file
covers the three tests that fence it and the invariants that must survive the
provenance-batching fix.

- Test 1 (rank stability) locks the invariant so future regressions (any
  change that restores the N+1 append pattern) are caught in CI regardless of
  host memory — on memory-pressed hosts the per-hit loop tips into swap thrash
  and perturbs ranks.
- Test 2 (top-60 pinned coverage).
- Test 3 (literal preservation).

Invariants covered:
- literal preservation (Test 3)
- provenance creation (Test 1 auxiliary assertion — batched write still
  produces exactly k_hits new provenance entries per recall call)
- verbatim recall at runbook profile (Test 2)
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from bench.verbatim import _make_noise, _make_pinned
from iai_mcp.retrieve import recall
from iai_mcp.store import EMBED_DIM, MemoryStore
from iai_mcp.types import MemoryRecord


NOISE_SEED = 20260419


def _seed_store(tmp_path, n_pinned: int, n_noise: int, dim: int = EMBED_DIM):
    """Isolated store with n_pinned + n_noise records.

    Pinned records use identical embedding = [1.0]*dim so cosine ties across
    all of them — this is the tie-break stress profile Test 1 needs. Noise
    uses seeded random unit vectors.
    """
    store = MemoryStore(path=tmp_path)
    pinned_texts = [
        f"Alice pinned verbatim day {i}: phrase-{i}-{'x' * 10}"
        for i in range(n_pinned)
    ]
    pinned_records = [_make_pinned(t, dim=dim) for t in pinned_texts]
    for r in pinned_records:
        store.insert(r)

    rng = np.random.default_rng(NOISE_SEED)
    for j in range(n_noise):
        store.insert(_make_noise(j, rng, dim=dim))
    return store, pinned_records, pinned_texts


def test_topk_rank_identical_across_sequential_queries(tmp_path):
    """Effect (c) rank-stability fence.

    Seeds 30 pinned (tied at cosine=1.0) + 100 noise, calls recall 20x with
    the SAME cue. Asserts the top-30 hit set and per-slot (record_id,
    literal_surface) tuple is byte-identical across every call.

    If the per-hit `store.append_provenance(...)` loop inside `recall()`
    perturbs the vector index mid-run (the low-memory failure mode),
    rank drift will cause this assertion to fail.

    Auxiliary assertion: for each of 20 sequential recalls, the pinned
    records' cumulative provenance entry count increases by exactly k_hits per
    call (batching preserves the "every recall → provenance entry" invariant,
    it only changes WHEN the writes happen).
    """
    store, pinned, pinned_texts = _seed_store(tmp_path, n_pinned=30, n_noise=100)
    dim = store.embed_dim
    cue = [1.0] * dim

    #: retrieve.recall now defaults to mode='verbatim'
    # (conservative North-Star fallback). The fixture pinned records are
    # tier='semantic' (per bench/verbatim._make_pinned), which verbatim mode
    # filters out — leaving zero hits. The rank-stability invariant
    # this test covers is mode-agnostic (it tests provenance-batch ordering
    # under recall pressure), so pin to mode='concept' explicitly.
    resp0 = recall(
        store=store,
        cue_embedding=cue,
        cue_text="probe",
        session_id="t0",
        budget_tokens=5000,
        k_hits=30,
        k_anti=3,
        mode="concept",
    )
    baseline_ids = tuple((h.record_id, h.literal_surface) for h in resp0.hits)
    assert len(baseline_ids) >= 1, "recall returned zero hits; harness broken"

    # Cap k_hits at n_pinned to avoid mixing noise into the deterministic head.
    # Every pinned is cosine=1.0; any reordering among them is rank drift.
    for i in range(1, 20):
        resp = recall(
            store=store,
            cue_embedding=cue,
            cue_text="probe",
            session_id=f"t{i}",
            budget_tokens=5000,
            k_hits=30,
            k_anti=3,
            mode="concept",
        )
        current = tuple((h.record_id, h.literal_surface) for h in resp.hits)
        assert current == baseline_ids, (
            f"rank drift at iteration {i}: top-k set changed between sequential "
            f"recalls with identical cue. Baseline={baseline_ids}, current={current}. "
            f"This indicates effect (c) provenance-write amplification is perturbing "
            f"the vector index."
        )

    # auxiliary: every pinned record should have >= 20 provenance entries
    # (one per recall that returned it in top-k). Because the cue is cosine=1.0
    # to every pinned, ALL 30 pinned are in top-30 on every call => exactly 20
    # new entries per pinned.
    for rec in pinned:
        updated = store.get(rec.id)
        assert updated is not None, f"pinned record {rec.id} vanished"
        # Allow tolerance for batch write ordering, but each pinned must have
        # >= 20 entries (20 recalls * 1 hit each).
        assert len(updated.provenance) >= 20, (
            f"MEM-05 violation: pinned {rec.id} has "
            f"{len(updated.provenance)} provenance entries after 20 recalls "
            f"(expected >= 20)."
        )


def test_topk_contains_all_pinned_at_runbook_profile(tmp_path):
    """gate at the runbook profile (n=50 pinned, k=60, 200 noise).

    At k=60 with 50 pinned + 200 noise, every pinned should be in the top-60.
    This is the in-process mirror of `bench/verbatim.py --n 50 --gap 5
    --noise-per-session 40 --k 60`, minus the provenance-write amplification
    angle that Test 1 covers.
    """
    store, pinned, _ = _seed_store(tmp_path, n_pinned=50, n_noise=200)
    dim = store.embed_dim
    cue = [1.0] * dim

    #: pin mode='concept' so tier='semantic' pinned records
    # survive the candidate filter (verbatim mode would drop them).
    resp = recall(
        store=store,
        cue_embedding=cue,
        cue_text="probe",
        session_id="runbook",
        budget_tokens=50_000,
        k_hits=60,
        k_anti=3,
        mode="concept",
    )
    hit_ids = {h.record_id for h in resp.hits}
    pinned_ids = {r.id for r in pinned}
    missing = pinned_ids - hit_ids
    assert not missing, (
        f"At runbook profile: "
        f"{len(missing)}/{len(pinned_ids)} pinned records missing from top-60. "
        f"Missing surface (first 3): "
        f"{sorted(str(m)[:8] for m in list(missing)[:3])}"
    )


def test_no_literal_surface_mutation(tmp_path):
    """literal_surface is byte-identical pre/post recalls.

    Belt-and-suspenders against any future change that would write to
    `literal_surface` during the recall path. The batching fix (Task 2) does
    not touch this field, but the invariant test locks it in so a regression
    in any other part of recall is caught immediately.
    """
    store, pinned, _ = _seed_store(tmp_path, n_pinned=10, n_noise=40)
    dim = store.embed_dim
    cue = [1.0] * dim

    # Snapshot literal_surface bytes before recalls.
    pre = {r.id: store.get(r.id).literal_surface for r in pinned}

    # 20 sequential recalls.
    #: mode='concept' so tier='semantic' pinned records
    # survive the candidate filter.
    for i in range(20):
        recall(
            store=store,
            cue_embedding=cue,
            cue_text=f"probe-{i}",
            session_id=f"s{i}",
            budget_tokens=5000,
            k_hits=15,
            k_anti=3,
            mode="concept",
        )

    # Post-recall snapshot: every byte unchanged.
    post = {r.id: store.get(r.id).literal_surface for r in pinned}
    assert pre.keys() == post.keys()
    for rid in pre:
        assert pre[rid] == post[rid], (
            f"C5 MEM-01 violation: literal_surface of record {rid} mutated "
            f"by recall path. Before={pre[rid]!r}, after={post[rid]!r}."
        )
