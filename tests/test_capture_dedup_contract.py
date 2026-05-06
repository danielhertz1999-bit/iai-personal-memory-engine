"""Phase 07.11 Plan 01 / — `memory_capture` dedup contract.

These four regression tests are the executable specification for D-01:

* `test_query_similar_accepts_tier_kwarg` — `query_similar` must accept a
  `tier` kwarg, must filter at the LanceDB where-layer when it is given, and
  must `ValueError` BEFORE any I/O on bad tier values.
* `test_capture_turn_dedups_on_high_cos_match` — capturing the same cue twice
  yields one inserted + one reinforced; the dedup branch is reachable.
* `test_capture_turn_inserts_on_low_cos` — distinct cues both insert; no
  false dedup.
* `test_reinforce_record_increments_edge_weight` — the new
  `store.reinforce_record` typed wrapper is a thin `boost_edges` delegate
  whose self-loop weight increases monotonically across calls.

Honesty constraint: every test below MUST fail on `git stash` of the
plan's source diffs and pass on `git stash pop`. RED-witness ran 2026-04-30
on un-fixed source: tier-kwarg + reinforce_record cases TypeError before the
fix; dedup cases fail because the dedup branch is unreachable dead code.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# --------------------------------------------------------------------------- fixtures
# Pattern copied verbatim from tests/test_pipeline_anti_hits_malformed.py:33-50
# (`_isolated_keyring` autouse fixture is the project canon for tests touching
# encrypted records on the construction host where the real keyring is absent
# or hangs).


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


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")


def _make_record(
    rid: UUID,
    surface: str = "topic",
    *,
    tier: str = "episodic",
    embedding: list[float] | None = None,
) -> MemoryRecord:
    """Minimal-record helper. Mirrors the shape used in the sibling test file
    `test_pipeline_anti_hits_malformed.py:_make_record` so existing fixture
    expectations transfer exactly. Defaults to a deterministic seed embedding
    (`[0.1] * EMBED_DIM`) so multiple records made with this helper share a
    high-cosine neighbourhood (the dedup tests need that)."""
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier=tier,
        literal_surface=surface,
        aaak_index="",
        embedding=list(embedding) if embedding is not None else [0.1] * EMBED_DIM,
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


# --------------------------------------------------------------------------- tests


def test_query_similar_accepts_tier_kwarg(store):
    """D-01 step 1: tier kwarg filters at the LanceDB where-layer.

    Pre-fix: TypeError("got an unexpected keyword argument 'tier'").
    Post-fix: returns only episodic rows; bad tier values raise ValueError
    BEFORE any I/O.
    """
    rid_e = uuid4()
    rid_s = uuid4()
    store.insert(_make_record(rid_e, "episodic-cue", tier="episodic"))
    store.insert(_make_record(rid_s, "semantic-cue", tier="semantic"))

    embedding = [0.1] * EMBED_DIM
    out = store.query_similar(embedding, k=10, tier="episodic")
    ids = {r.id for r, _ in out}
    assert rid_e in ids, "episodic record should be returned by tier='episodic'"
    assert rid_s not in ids, "semantic record must be filtered out by tier='episodic'"

    # Bad tier -> ValueError before any I/O.
    with pytest.raises(ValueError):
        store.query_similar(embedding, k=10, tier="bogus")

    # Backwards-compat: tier=None preserves the legacy behaviour (both rows
    # are returned by the cosine query, no where-clause applied).
    out_none = store.query_similar(embedding, k=10, tier=None)
    ids_none = {r.id for r, _ in out_none}
    assert rid_e in ids_none and rid_s in ids_none


def test_capture_turn_dedups_on_high_cos_match(store):
    """D-01 step 3: second capture of identical cue -> reinforced, not inserted.

    Pre-fix: dedup branch unreachable. Bug A (TypeError on tier kwarg) is
    swallowed by `except Exception`; `neighbours = []` so the loop never
    executes. Even if Bug A were fixed, Bug B (`getattr(n, "score", None)`
    on a tuple) returns None so the `if score is not None` guard never
    fires. Even if both A+B were fixed, Bug C (single-UUID list to
    boost_edges which expects pairs) crashes. Result: every capture inserts.

    Post-fix: dedup branch is reachable; second call returns
    `status="reinforced"` and the episodic-record count stays at 1.
    """
    text = "the user prefers Russian on the surface; English in storage"
    cue = "lang preference"

    r1 = capture_turn(
        store=store, text=text, cue=cue, tier="episodic",
        session_id="s1", role="user",
    )
    assert r1["status"] == "inserted", f"first capture should insert, got {r1}"

    r2 = capture_turn(
        store=store, text=text, cue=cue, tier="episodic",
        session_id="s1", role="user",
    )
    assert r2["status"] == "reinforced", f"second capture should reinforce, got {r2}"
    assert "cos=" in r2["reason"], f"reason should record cosine score, got {r2}"

    # Record count remains 1 -- no duplicate inserted.
    rows = list(store.iter_records())
    assert len([r for r in rows if r.tier == "episodic"]) == 1


def test_capture_turn_inserts_on_low_cos(store):
    """distinct cues -> two inserts, no false dedup.

    Asymmetric guard against an over-eager fix: if the dedup branch fires
    on EVERY capture (e.g. cos threshold misread), this test catches it.
    """
    r1 = capture_turn(
        store=store, text="apples are red", cue="apple",
        tier="episodic", session_id="s1", role="user",
    )
    r2 = capture_turn(
        store=store,
        text="quantum chromodynamics describes the strong force",
        cue="qcd", tier="episodic", session_id="s1", role="user",
    )
    assert r1["status"] == "inserted", f"first insert expected, got {r1}"
    assert r2["status"] == "inserted", f"second insert expected, got {r2}"

    rows = list(store.iter_records())
    assert len([r for r in rows if r.tier == "episodic"]) == 2


def test_reinforce_record_increments_edge_weight(store):
    """D-01 step 2: reinforce_record self-loop weight increases monotonically.

    Pre-fix: AttributeError -- `reinforce_record` does not exist on store.
    Post-fix: the typed wrapper builds `[(rid, rid)]` and delegates to
    `boost_edges`; the canonical-pair coalescer at boost_edges:1244-1247
    produces the canonical `(str(rid), str(rid))` self-loop key, and the
    weight strictly increases on each successive call.
    """
    rid = uuid4()
    store.insert(_make_record(rid, "anchor-record"))

    w1 = store.reinforce_record(rid)
    w2 = store.reinforce_record(rid)

    # Both calls return dict[(str, str), float] keyed by the canonical
    # sorted-self-loop pair.
    key = (str(rid), str(rid))
    assert key in w1, f"self-loop key missing from first call: {w1}"
    assert key in w2, f"self-loop key missing from second call: {w2}"
    assert w2[key] > w1[key], (
        f"weight must strictly increase across calls: w1={w1[key]} w2={w2[key]}"
    )
