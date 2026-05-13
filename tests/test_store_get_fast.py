"""— store.get filter-pushdown fast-path ( / M-02).

TDD RED scaffold for exit gate.

Goal: MemoryStore.get(record_id) must use a LanceDB filter-pushdown
point read instead of tbl.to_pandas() full-table-scan. At N=1k the old
path materialised every row + column into a pandas DataFrame and then
filtered in-process; on the prod schema (embedding 384d + encrypted
text + many columns) this ate ~34 ms per call -> ~340 ms per recall
iteration (L0 fast-path + anti-hit lookup = 10 calls/iter).

Invariants preserved:
  - unknown id -> None
  - known id   -> MemoryRecord via _from_row (AES-GCM decrypt fidelity)
  - semantics identical to the full-scan path (byte-identical fields)
"""
from __future__ import annotations

import random
import time
from uuid import UUID, uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord
from tests.test_store import _make


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

def _seed(
    store: MemoryStore, n: int, *, seed: int = 0, compact: bool = False
) -> list[UUID]:
    """Seed `n` records with deterministic embeddings; return ids in order.

    When ``compact=True``, run ``tbl.optimize()`` after the inserts so the
    table is in a single-fragment steady state -- this mirrors what the
    AsyncWriteQueue produces in production and what the
    bench actually measures after warm-up. Without compaction the
    per-insert fragments force every scan (filter-pushdown or not) to
    touch N fragments and perf-fence numbers are dominated by fragment
    open cost rather than the get-path cost we actually want to measure.
    """
    from iai_mcp.store import RECORDS_TABLE

    rnd = random.Random(seed)
    ids: list[UUID] = []
    for i in range(n):
        vec = [rnd.random() for _ in range(EMBED_DIM)]
        r = _make(text=f"fact {i} :: verbatim payload {rnd.random():.6f}", vec=vec)
        store.insert(r)
        ids.append(r.id)
    if compact:
        try:
            tbl = store.db.open_table(RECORDS_TABLE)
            tbl.optimize()
        except Exception:
            # optimize() requires pylance on some platforms; skipping is
            # non-fatal -- the test will just see the pre-compaction
            # numbers, which still exercise the filter-pushdown code path.
            pass
    return ids


# --------------------------------------------------------------------------- #
# G1: unknown id -> None                                                      #
# --------------------------------------------------------------------------- #

def test_get_unknown_id_returns_none(tmp_path):
    """G1: unknown uuid returns None (unchanged semantics)."""
    store = MemoryStore(path=tmp_path)
    _seed(store, n=5)
    phantom = uuid4()
    assert store.get(phantom) is None


# --------------------------------------------------------------------------- #
# G2: known id round-trips + literal_surface decrypts                         #
# --------------------------------------------------------------------------- #

def test_get_known_id_roundtrip_with_decrypt(tmp_path):
    """G2: known id -> MemoryRecord; encrypted literal_surface decrypts."""
    store = MemoryStore(path=tmp_path)
    verbatim = "пусть каждое слово сохранится точно — G2 fidelity"
    r = _make(text=verbatim)
    store.insert(r)
    got = store.get(r.id)
    assert got is not None
    assert got.id == r.id
    assert got.literal_surface == verbatim


# --------------------------------------------------------------------------- #
# G3: no unfiltered to_pandas() on MemoryStore.get                            #
# --------------------------------------------------------------------------- #

def test_get_does_not_call_unfiltered_to_pandas(tmp_path, monkeypatch):
    """G3: store.get must NOT call tbl.to_pandas() without a filter.

    Accept either:
      - tbl.search(...).where(...).to_pandas()
      - tbl.to_lance().to_table(filter=...).to_pandas()
    Reject: bare tbl.to_pandas() with no filter kwarg.
    """
    store = MemoryStore(path=tmp_path)
    _seed(store, n=20)
    target = _seed(store, n=1)[0]

    import lancedb.table as _lt

    # LanceTable is the concrete subclass of Table that open_table returns
    # in lancedb 0.30.x; it overrides to_pandas, so we must patch the
    # concrete class, not the ABC.
    target_cls = _lt.LanceTable
    base_to_pandas = target_cls.to_pandas
    unfiltered_calls: list[dict] = []

    def traced(self, *args, **kwargs):
        # If called on the Table directly (NOT on a search/query builder)
        # and no filter kwarg is passed, record it — that is the old
        # full-scan path.
        if "filter" not in kwargs:
            unfiltered_calls.append({"args": args, "kwargs": dict(kwargs)})
        return base_to_pandas(self, *args, **kwargs)

    monkeypatch.setattr(target_cls, "to_pandas", traced)

    got = store.get(target)
    assert got is not None
    assert got.id == target
    assert not unfiltered_calls, (
        "store.get called Table.to_pandas() without a filter — "
        "full-scan path still in use. Expected filter-pushdown via "
        "tbl.search(...).where(...) or tbl.to_lance().to_table(filter=...)."
    )


# --------------------------------------------------------------------------- #
# G4: perf fence — 100 sequential store.get at N=1k <= 500 ms total           #
# --------------------------------------------------------------------------- #

def test_get_perf_fence_n1k(tmp_path):
    """G4: 100 sequential store.get at N=1k <= 500 ms total (mean <=5 ms, p95 <=10 ms).

    Uses ``compact=True`` in the fixture so the table is a single-fragment
    steady state -- this is what the production AsyncWriteQueue
    produces and what the bench measures after warm-up. Without
    compaction, per-insert fragments dominate every scan and the numbers
    measure fragment open cost rather than the get-path cost the plan
    actually wants to fence.
    """
    store = MemoryStore(path=tmp_path)
    ids = _seed(store, n=1000, compact=True)
    rnd = random.Random(42)
    picks = [rnd.choice(ids) for _ in range(100)]

    # Warmup — pay the first-call LanceDB table-open / index compile once.
    store.get(picks[0])

    samples_ms: list[float] = []
    for rid in picks:
        t0 = time.perf_counter()
        rec = store.get(rid)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
        assert rec is not None and rec.id == rid

    total = sum(samples_ms)
    mean = total / len(samples_ms)
    samples_ms.sort()
    p95 = samples_ms[int(0.95 * len(samples_ms)) - 1]

    # Perf fence — generous margins so CI noise does not flake.
    assert total <= 500.0, f"N=1k 100x store.get total {total:.1f} ms > 500 ms budget"
    assert mean <= 5.0, f"N=1k store.get mean {mean:.2f} ms > 5 ms/call"
    assert p95 <= 10.0, f"N=1k store.get p95 {p95:.2f} ms > 10 ms/call"


# --------------------------------------------------------------------------- #
# G5: correctness fence vs full-scan baseline                                 #
# --------------------------------------------------------------------------- #

def test_get_matches_full_scan_baseline(tmp_path):
    """G5: for 50 random ids at N=1k, store.get output equals _from_row applied
    to the full-scan row — byte-identical on id, literal_surface, embedding,
    tags, provenance, language, community_id, centrality, stability,
    difficulty, last_reviewed, updated_at.
    """
    store = MemoryStore(path=tmp_path)
    ids = _seed(store, n=1000)
    rnd = random.Random(7)
    picks = [rnd.choice(ids) for _ in range(50)]

    # Build the baseline via the legacy full-scan reconstruction.
    tbl = store.db.open_table("records")
    df = tbl.to_pandas()

    for rid in picks:
        got = store.get(rid)
        assert got is not None
        baseline_row = df[df["id"] == str(rid)].iloc[0].to_dict()
        baseline = store._from_row(baseline_row)

        assert got.id == baseline.id
        assert got.literal_surface == baseline.literal_surface
        assert list(got.embedding) == list(baseline.embedding)
        assert got.tags == baseline.tags
        assert got.provenance == baseline.provenance
        assert got.language == baseline.language
        assert got.community_id == baseline.community_id
        assert got.centrality == baseline.centrality
        assert got.stability == baseline.stability
        assert got.difficulty == baseline.difficulty
        assert got.last_reviewed == baseline.last_reviewed
        assert got.updated_at == baseline.updated_at
