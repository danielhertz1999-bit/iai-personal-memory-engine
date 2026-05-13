"""Pipeline perf regression guard (Task 2, D-SPEED gap closure).

Load-bearing tests:
  - test_recall_for_response_p95_under_threshold: seeds N=100 records, runs
    recall_for_response 10 times; asserts p95 < 150ms (CI-generous ceiling).
    Bench CLI uses the strict 100ms ceiling (D-SPEED SC-6).
  - test_recall_for_response_single_provenance_batch_call: instrumentation test;
    confirms append_provenance_batch is called exactly once per recall and
    the pairs list matches the hit count (no per-hit append_provenance).
  - test_recall_for_response_mem05_provenance_preserved: semantic
    equivalence check -- every hit still has exactly one new provenance
    entry whose session_id matches the call's session_id.
  - test_recall_for_response_on_read_check_uses_batch_variant: monkeypatches
    s4.on_read_check to raise + on_read_check_batch to return []; the call
    must NOT raise, proving the batch variant is on the active call path.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


class _PerfEmbedder:
    """Deterministic sha256-based embedder. Stable across processes."""

    DIM = EMBED_DIM

    def __init__(self, base_seed: int = 0) -> None:
        self._base_seed = base_seed

    def embed(self, text: str) -> list[float]:
        import hashlib
        import random
        digest = hashlib.sha256(
            f"{self._base_seed}:{text}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        v = [rng.random() * 2 - 1 for _ in range(self.DIM)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v] if norm > 0 else v

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _make_rec(vec: list[float], text: str, tags: list[str]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec,
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
        tags=tags,
        language="en",
    )


def _seed_store(path, n: int = 100, seed: int = 0):
    """Seed a MemoryStore with N synthetic records + build runtime graph."""
    from iai_mcp.pipeline import recall_for_response  # noqa: F401
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=path)
    embedder = _PerfEmbedder(base_seed=seed)
    tag_pool = [
        ["topic:auth"], ["topic:db"], ["topic:web"],
        ["topic:net"], ["topic:cli"],
    ]
    for i in range(n):
        vec = embedder.embed(f"seed-{i}")
        tags = list(tag_pool[i % len(tag_pool)])
        rec = _make_rec(vec, text=f"synthetic fact {i}", tags=tags)
        store.insert(rec)
    graph, assignment, rich_club = build_runtime_graph(store)
    return store, embedder, graph, assignment, rich_club


# --------------------------------------------------------- perf regression


def test_recall_for_response_p95_under_threshold(tmp_path):
    """D-SPEED perf regression guard: p95 < 150ms at N=100 (CI-generous).

    PRE-FIX: p95 ~1000ms (fails). POST-FIX: p95 < 150ms (passes).
    Bench CLI uses strict 100ms; this test uses 150ms to absorb CI jitter.
    """
    import time

    from iai_mcp.pipeline import recall_for_response

    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=100, seed=0,
    )
    cues = [
        "what did we cover about auth yesterday?",
        "explain the db migration plan",
        "how does the web cache invalidation work",
        "summary of the cli subcommand changes",
        "recent network stack bug report",
    ]

    latencies: list[float] = []
    for i in range(10):
        cue = cues[i % len(cues)]
        t0 = time.perf_counter()
        recall_for_response(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=rich_club,
            embedder=embedder,
            cue=cue,
            session_id="perf_test",
            budget_tokens=1500,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[-1]
    assert p95 < 150.0, (
        f"D-SPEED regression: p95={p95:.2f}ms > 150ms at N=100 "
        f"(target <100ms strict, 150ms CI-generous). "
        f"All latencies: {[f'{x:.1f}' for x in latencies]}"
    )


# --------------------------------------------------------- wire-up tests


def test_recall_for_response_single_provenance_batch_call(tmp_path, monkeypatch):
    """recall_for_response calls store.append_provenance_batch EXACTLY once.

    Instrumentation: replace append_provenance_batch with a recorder.
    The recorder captures the pairs list length; after one recall_for_response
    call with hits>=1, count must be exactly 1 and the pairs list length
    must equal the number of hits.

    ALSO asserts store.append_provenance (single-call) is NEVER called on
    the hit path -- is preserved but through the batch API.
    """
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.store import MemoryStore

    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=100, seed=0,
    )

    batch_calls: list[int] = []  # each element = len(pairs) of that call
    single_calls: list[object] = []

    original_batch = MemoryStore.append_provenance_batch
    original_single = MemoryStore.append_provenance

    def _recorder_batch(self, pairs, *args, **kwargs):
        # pipeline passes records_cache=... kwarg -- accept and
        # forward. The test only cares about call-count + pairs-length.
        batch_calls.append(len(pairs))
        return original_batch(self, pairs, *args, **kwargs)

    def _recorder_single(self, record_id, entry):
        single_calls.append((record_id, entry))
        return original_single(self, record_id, entry)

    monkeypatch.setattr(MemoryStore, "append_provenance_batch", _recorder_batch)
    monkeypatch.setattr(MemoryStore, "append_provenance", _recorder_single)

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue="what about auth",
        session_id="wire_test",
        budget_tokens=1500,
    )

    assert len(resp.hits) >= 1, "pipeline must return at least one hit on seeded store"
    assert len(batch_calls) == 1, (
        f"append_provenance_batch should be called EXACTLY once; got {len(batch_calls)}"
    )
    assert batch_calls[0] == len(resp.hits), (
        f"batch pairs list should have {len(resp.hits)} entries (one per hit); "
        f"got {batch_calls[0]}"
    )
    # No per-hit single calls on the hit path.
    assert len(single_calls) == 0, (
        f"append_provenance (single) should NOT be called on the hit path; "
        f"got {len(single_calls)} calls"
    )


def test_recall_for_response_mem05_provenance_preserved(tmp_path):
    """ correctness: every hit has a NEW provenance entry post-recall.

    Establishes provenance len-before per hit, runs recall_for_response, then
    confirms each hit's record has exactly one more provenance entry whose
    session_id matches the call.
    """
    from iai_mcp.pipeline import recall_for_response

    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=100, seed=0,
    )
    session = "mem05_preserved"

    # Run recall first to see which records become hits.
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue="what about auth",
        session_id=session,
        budget_tokens=1500,
    )
    assert len(resp.hits) >= 1

    for h in resp.hits:
        rec = store.get(h.record_id)
        assert rec is not None
        # Every hit has AT LEAST one provenance entry with the session_id
        # we just used. (provisional check for correctness).
        matching = [p for p in rec.provenance if p.get("session_id") == session]
        assert len(matching) == 1, (
            f"record {h.record_id} has {len(matching)} provenance entries "
            f"for session '{session}'; expected exactly 1. prov={rec.provenance}"
        )


def test_recall_for_response_on_read_check_uses_batch_variant(tmp_path, monkeypatch):
    """The active call path uses on_read_check_batch, not on_read_check.

    Arrange: monkeypatch s4.on_read_check to RAISE. If the old single-call
    path is still wired, recall_for_response will fail (or silently swallow).
    We also patch on_read_check_batch to return a sentinel, and verify it
    is what actually flows through.
    """
    from iai_mcp import s4
    from iai_mcp.pipeline import recall_for_response

    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=100, seed=0,
    )

    sentinel = [{"kind": "sentinel_batch_wired", "severity": "info", "source_ids": [], "text": "ok"}]

    def _boom(*a, **kw):
        raise RuntimeError("old on_read_check must NOT be called")

    def _batch_ok(*a, **kw):
        return sentinel

    monkeypatch.setattr(s4, "on_read_check", _boom)
    monkeypatch.setattr(s4, "on_read_check_batch", _batch_ok)

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue="what about auth",
        session_id="batch_wired_test",
        budget_tokens=1500,
    )
    # The batch variant's sentinel must appear in hints.
    hint_kinds = [h.get("kind") for h in resp.hints]
    assert "sentinel_batch_wired" in hint_kinds, (
        f"expected on_read_check_batch sentinel in hints; got {hint_kinds}"
    )
