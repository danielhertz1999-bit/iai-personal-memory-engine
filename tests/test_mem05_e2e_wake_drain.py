"""E2E WAKE-drain test for deferred-provenance buffer.

Verifies that the deferred-provenance buffer at.deferred-provenance.jsonl
is drained on daemon WAKE transitions, satisfying the invariant
("every recall appends provenance") in production.

Test cases:
  - test_buffer_drains_on_wake: hot-path defer → explicit flush → buffer
    empty + provenance entries land on records (simulates daemon WAKE drain).
  - test_flush_idempotent_on_empty: flush on empty/missing buffer returns 0,
    no exception, no store write.
  - test_grep_production_callsite_exists: regression guard — daemon.py
    source MUST literally contain "flush_deferred_provenance" so we never
    silently de-wire the production call site again.

Background: `flush_deferred_provenance` once had ZERO production call sites;
a stale comment claimed it was "flushed during SLEEP" but no SLEEP step
called it, so the buffer file grew unboundedly. The flush is now wired into
the daemon WAKE handler, homologous to drain_deferred_captures.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.provenance_buffer import (
    _BUFFER_FILENAME,
    flush_deferred_provenance,
)
from iai_mcp.types import EMBED_DIM, MemoryRecord


class _PerfEmbedder:
    """Deterministic sha256-based embedder, copy of test_pipeline_perf helper."""

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


def _seed_small_store(path, n: int = 20, seed: int = 0):
    """Seed a small MemoryStore + runtime graph for fast E2E tests."""
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


def test_buffer_drains_on_wake(tmp_path, monkeypatch):
    """End-to-end: recall defers → flush drains → buffer empty + provenance lands.

    Simulates the daemon WAKE-handler drain by calling
    `flush_deferred_provenance(store)` directly (unit-scope; no FSM).
    Verifies the contract: every returned hit has a provenance
    entry tagged with the session_id used in the recall.

    Opts out of the conftest's `defer_provenance` auto-flush fixture so the
    buffer file is observable mid-test (the whole point of this test is to
    verify the deferred-then-drained lifecycle of that file). Seed runs
    BEFORE the env var is set so the records/edges auto-flush stays active
    during inserts; only the post-seed recall path observes the deferred
    JSONL buffer.
    """
    from iai_mcp.pipeline import recall_for_response

    store, embedder, graph, assignment, rich_club = _seed_small_store(
        tmp_path, n=20, seed=0,
    )

    # Opt out of conftest auto-flushes AFTER seeding so the recall below
    # leaves the deferred buffer un-drained (which is what this test asserts).
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    session = "wake-drain-test"

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
    assert len(resp.hits) >= 1, "recall must produce at least one hit"

    buffer_path = Path(store.root) / _BUFFER_FILENAME
    assert buffer_path.exists(), "defer_provenance must have created the buffer"
    pre_lines = buffer_path.read_text().strip().splitlines()
    assert len(pre_lines) > 0, (
        f"buffer must contain entries from the recall defer; got {len(pre_lines)}"
    )

    drained = flush_deferred_provenance(store)
    assert drained == len(pre_lines), (
        f"flush must drain all pending entries; expected {len(pre_lines)}, got {drained}"
    )

    post_text = buffer_path.read_text().strip()
    assert post_text == "", (
        f"buffer must be truncated after flush; got {len(post_text)} bytes"
    )

    for h in resp.hits:
        rec = store.get(h.record_id)
        assert rec is not None
        matching = [p for p in rec.provenance if p.get("session_id") == session]
        assert len(matching) == 1, (
            f"record {h.record_id} has {len(matching)} provenance entries "
            f"for session '{session}'; expected exactly 1. prov={rec.provenance}"
        )


def test_flush_idempotent_on_empty(tmp_path):
    """flush_deferred_provenance returns 0 on missing/empty buffer, no raise."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    buffer_path = Path(store.root) / _BUFFER_FILENAME

    assert not buffer_path.exists(), "fresh store must not have a buffer file"
    count_missing = flush_deferred_provenance(store)
    assert count_missing == 0

    buffer_path.write_text("")
    count_empty = flush_deferred_provenance(store)
    assert count_empty == 0


def test_grep_production_callsite_exists():
    """Regression guard: daemon.py source MUST reference flush_deferred_provenance.

    Catches future de-wiring of the production call site. Skipped if running
    from a sdist install where src/ is not on disk relative to repo root.
    """
    repo_root = Path(__file__).resolve().parent.parent
    daemon_src = repo_root / "src" / "iai_mcp" / "daemon.py"
    if not daemon_src.exists():
        pytest.skip("daemon.py source not on disk (sdist/wheel install)")
    text = daemon_src.read_text()
    assert "flush_deferred_provenance" in text, (
        "daemon.py MUST reference flush_deferred_provenance; otherwise the "
        "MEM-05 deferred-provenance buffer is never drained in production "
        "(regression)."
    )
