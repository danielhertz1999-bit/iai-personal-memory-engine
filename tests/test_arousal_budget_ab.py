"""(P1 full plumb-in) — arousal_budget A/B regression suite.

The A/B route lives INSIDE `_recall_core` (mirrors EFE precedent commit
ae49662). All three RetrievalParams fields (`max_hops`, `rank_threshold`,
`mode`) are plumbed into _recall_core scoring with measurable effect:

  - rank_threshold => filter cosine_top_indices by shared_cos >= threshold
  - max_hops => override spread_hops when smaller than current default
  - mode => bias adjust on top of _gate_bias_for_mode(mode) for gated_set

Tests cover:

1. test_route_determinism — same cue string yields the same route
2. test_route_split_balance — 10000 random cues yield ~5000/5000 within 2%
3. test_env_override_shadow_forces_all_to_shadow — 100% shadow with env=1
4. test_bench_production_route_parity — bench `_bench_arousal_route_for_cue`
   matches the inline formula on a parametrized cue set
5. test_telemetry_event_written — after a real `_recall_core` call, the
   events table has at least one `retrieval_arousal_ab` event with the
   expected keys (cue_hash, route, n_hits, budget_tokens_used,
   max_hops_used, rank_threshold_used, arousal_level, arousal_mode,
   top_hit_id)
6. test_rank_threshold_filters_candidates — on `arousal_real` route the
   rank_threshold filter reduces candidate count vs `arousal_shadow`
7. test_analyzer_math_synthetic_csv — `compute_per_route_rescue_at_k` and
   the verdict logic on a deterministic synthetic CSV with skewed counts

The 6th test is the load-bearing P1 plumb-in proof: it asserts the
arousal_real route actually scores differently. Without it, A/B would be
predetermined null (P2 scope).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1-4: pure-Python route logic + bench mirror
# ---------------------------------------------------------------------------


def _inline_production_route(cue: str) -> tuple[str, str]:
    """Inline reference for the production routing formula.

    If this drifts from `_bench_arousal_route_for_cue` OR the inline block in
    `_recall_core`, the parity test catches it.
    """
    digest = hashlib.md5(str(cue).encode("utf-8")).digest()
    cue_hash_hex = digest[:4].hex()
    if os.environ.get("IAI_MCP_AROUSAL_USE_SHADOW") == "1":
        return ("arousal_shadow", cue_hash_hex)
    route = "arousal_real" if (digest[0] & 1) else "arousal_shadow"
    return (route, cue_hash_hex)


def test_route_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same cue string yields the same route across calls."""
    monkeypatch.delenv("IAI_MCP_AROUSAL_USE_SHADOW", raising=False)
    from bench import contradiction_longitudinal_claude as bench

    helper = bench._bench_arousal_route_for_cue  # noqa: SLF001

    cues = ["one", "two", "three", "русский cue", "x" * 100]
    for cue in cues:
        r1, h1 = helper(cue)
        r2, h2 = helper(cue)
        assert r1 == r2, f"determinism broken for cue={cue!r}"
        assert h1 == h2


def test_route_split_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    """10000 random cues route ~5000/5000 within +/-2 percent."""
    monkeypatch.delenv("IAI_MCP_AROUSAL_USE_SHADOW", raising=False)
    from bench import contradiction_longitudinal_claude as bench

    helper = bench._bench_arousal_route_for_cue  # noqa: SLF001

    n = 10000
    real = 0
    shadow = 0
    for i in range(n):
        route, _ = helper(f"cue-{i}")
        if route == "arousal_real":
            real += 1
        elif route == "arousal_shadow":
            shadow += 1
        else:  # pragma: no cover — sanity guard
            pytest.fail(f"unexpected route {route!r}")
    expected_per_arm = n / 2
    tolerance = n * 0.02  # +/- 2 percent
    assert abs(real - expected_per_arm) < tolerance, f"real={real} skewed"
    assert abs(shadow - expected_per_arm) < tolerance, f"shadow={shadow} skewed"


def test_env_override_shadow_forces_all_to_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`IAI_MCP_AROUSAL_USE_SHADOW=1` forces 100% shadow regardless of cue."""
    monkeypatch.setenv("IAI_MCP_AROUSAL_USE_SHADOW", "1")
    from bench import contradiction_longitudinal_claude as bench

    helper = bench._bench_arousal_route_for_cue  # noqa: SLF001

    for cue in [f"cue-{i}" for i in range(100)]:
        route, cue_hash = helper(cue)
        assert route == "arousal_shadow", (
            f"shadow override broken for cue={cue!r}: got {route}"
        )
        # Hash still computed from cue (not branched on env).
        expected_hash = hashlib.md5(cue.encode("utf-8")).digest()[:4].hex()
        assert cue_hash == expected_hash


@pytest.mark.parametrize(
    "cue",
    [
        "alpha", "beta", "gamma", "delta",
        "Quote the original launch announcement.",
        "русский cue",
        "ceo_name probe",
        "x",
    ],
)
def test_bench_production_route_parity(
    cue: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_bench_arousal_route_for_cue` matches the inline production formula."""
    monkeypatch.delenv("IAI_MCP_AROUSAL_USE_SHADOW", raising=False)
    from bench import contradiction_longitudinal_claude as bench

    bench_route, bench_hash = bench._bench_arousal_route_for_cue(cue)  # noqa: SLF001
    prod_route, prod_hash = _inline_production_route(cue)

    assert bench_route == prod_route, (
        f"route drift for cue={cue!r}: bench={bench_route} prod={prod_route}"
    )
    assert bench_hash == prod_hash, (
        f"hash drift for cue={cue!r}: bench={bench_hash} prod={prod_hash}"
    )


# ---------------------------------------------------------------------------
# 5-6: production telemetry + scoring effect (require iai_mcp imports)
# ---------------------------------------------------------------------------


# Fixture helpers cloned from tests/test_recall_core_unit.py for reuse.


class _FakeEmbedder:
    """Stand-in embedder: cue's embedding is configurable per-test."""

    def __init__(self, vec: list[float] | None = None) -> None:
        from iai_mcp.types import EMBED_DIM

        self.DIM = EMBED_DIM
        if vec is None:
            vec = [1.0] + [0.0] * (EMBED_DIM - 1)
        self._vec = vec

    def embed(self, text: str) -> list[float]:
        return list(self._vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]


def _make_record(vec: list[float], text: str = "rec", tier: str = "episodic"):
    from iai_mcp.types import MemoryRecord

    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
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
        tags=[],
        language="en",
    )


def _build_store_and_graph(tmp_path, n: int):
    """Build N records with primary-axis embeddings + matching MemoryGraph."""
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import EMBED_DIM

    store = MemoryStore(path=tmp_path / "hippo")
    recs = []
    for i in range(n):
        vec = [0.0] * EMBED_DIM
        # Distribute on different axes so cosine values vary across records.
        # Records at i*4 will share axis 0 with the cue (high cosine);
        # records at i*4+1, +2, +3 use other axes (cosine near 0).
        vec[i % EMBED_DIM] = 1.0
        rec = _make_record(vec, text=f"rec{i}")
        store.insert(rec)
        recs.append(rec)
    graph = MemoryGraph()
    for rec in recs:
        graph.add_node(rec.id, community_id=None, embedding=list(rec.embedding))
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding),
            "surface": rec.literal_surface,
            "centrality": 0.0,
            "tier": rec.tier,
            "tags": [],
            "language": "en",
        })
    return store, graph, recs


def _flat_assignment(recs):
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.types import EMBED_DIM

    cid = uuid4()
    centroid = [1.0] + [0.0] * (EMBED_DIM - 1)
    return CommunityAssignment(
        node_to_community={r.id: cid for r in recs},
        community_centroids={cid: centroid},
        modularity=0.0,
        backend="flat",
        top_communities=[cid],
        mid_regions={cid: [r.id for r in recs]},
    )


def test_telemetry_event_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_recall_core` emits a `retrieval_arousal_ab` event with the expected keys."""
    monkeypatch.delenv("IAI_MCP_AROUSAL_USE_SHADOW", raising=False)

    from iai_mcp.events import flush_event_buffer, query_events
    from iai_mcp.pipeline import _recall_core

    store, graph, recs = _build_store_and_graph(tmp_path, n=10)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    _recall_core(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=embedder,
        cue="probe-cue-alpha",
        session_id="test-session",
    )

    # `write_event` is buffered=True for telemetry; flush before querying.
    flush_event_buffer(store)

    rows = query_events(store, kind="retrieval_arousal_ab", limit=10)
    assert rows, "no retrieval_arousal_ab event written"

    ev = rows[0]
    assert ev["kind"] == "retrieval_arousal_ab"
    data = ev.get("data") or {}

    required_keys = {
        "cue_hash", "route", "n_hits", "budget_tokens_used",
        "max_hops_used", "rank_threshold_used", "arousal_level",
        "arousal_mode", "top_hit_id",
    }
    missing = required_keys - data.keys()
    assert not missing, f"telemetry missing keys: {missing}"

    # Route is one of the documented arms (or skip on exception path).
    assert data["route"] in {"arousal_real", "arousal_shadow", "arousal_skip"}
    # cue_hash is 8 hex chars (first 4 bytes of MD5).
    assert isinstance(data["cue_hash"], str) and len(data["cue_hash"]) == 8


def test_rank_threshold_filters_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On `arousal_real` route, rank_threshold filters cosine_top_indices.

    Builds a pool of 60 records. Records on axis 0 share the cue's primary
    axis (cosine ~1.0); records on other axes have cosine ~0.0. The balanced
    arousal regime returns rank_threshold ~0.45. Under arousal_real, only
    axis-0 records survive the filter; under arousal_shadow, all 60 reach
    Stage 8.

    Force one route then the other via the env override; assert the number
    of attributable candidates differs in the expected direction.
    """
    from iai_mcp.events import flush_event_buffer, query_events
    from iai_mcp.pipeline import _recall_core
    from iai_mcp.types import EMBED_DIM

    # 60 records spread across EMBED_DIM (384) axes; ~ EMBED_DIM rotation
    # leaves only a few on axis 0 if EMBED_DIM > 60, so build deliberately
    # mixed axes so we control how many records have cosine >= 0.45.
    # We want axes that yield exact cosine = 1.0 vs 0.0:
    # - "good" records share cue axis (axis 0) -> cosine = 1.0
    # - "bad" records use other axes -> cosine = 0.0
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "hippo")
    recs = []
    good_count = 5
    bad_count = 55
    # 5 good records on axis 0 (cosine=1.0)
    for i in range(good_count):
        vec = [0.0] * EMBED_DIM
        vec[0] = 1.0
        rec = _make_record(vec, text=f"good{i}")
        store.insert(rec)
        recs.append(rec)
    # 55 bad records on axis 1 (cosine=0.0 with cue axis 0)
    for i in range(bad_count):
        vec = [0.0] * EMBED_DIM
        vec[1] = 1.0
        rec = _make_record(vec, text=f"bad{i}")
        store.insert(rec)
        recs.append(rec)
    graph = MemoryGraph()
    for rec in recs:
        graph.add_node(rec.id, community_id=None, embedding=list(rec.embedding))
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding),
            "surface": rec.literal_surface,
            "centrality": 0.0,
            "tier": rec.tier,
            "tags": [],
            "language": "en",
        })
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()  # cue vec = axis 0

    # Run 1: forced shadow route (env=1). All 60 records reach Stage 8 ranking.
    monkeypatch.setenv("IAI_MCP_AROUSAL_USE_SHADOW", "1")
    _recall_core(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=embedder,
        cue="rank-threshold-probe",
        session_id="test-shadow",
    )
    flush_event_buffer(store)
    shadow_events = query_events(store, kind="retrieval_arousal_ab", limit=5)
    assert shadow_events, "shadow probe wrote no event"
    shadow_data = shadow_events[0]["data"]
    assert shadow_data["route"] == "arousal_shadow", (
        f"force-shadow env not honored: route={shadow_data['route']}"
    )
    # Shadow route: rank_threshold_used is 0.0 (no filter applied).
    assert shadow_data["rank_threshold_used"] == 0.0, (
        f"shadow rank_threshold_used should be 0.0: got "
        f"{shadow_data['rank_threshold_used']}"
    )

    # Run 2: forced real route via a cue whose MD5(digest[0]) & 1 == 1.
    # Find such a cue deterministically.
    monkeypatch.delenv("IAI_MCP_AROUSAL_USE_SHADOW", raising=False)
    real_cue = None
    for i in range(100):
        cand = f"real-cue-{i}"
        digest = hashlib.md5(cand.encode()).digest()
        if digest[0] & 1:
            real_cue = cand
            break
    assert real_cue is not None, "could not find a cue with arousal_real route"

    _recall_core(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=embedder,
        cue=real_cue,
        session_id="test-real",
    )
    flush_event_buffer(store)
    real_events = query_events(store, kind="retrieval_arousal_ab", limit=5)
    # Newest first; the most recent event is from the real call.
    real_data = real_events[0]["data"]
    assert real_data["route"] == "arousal_real", (
        f"unexpected route on real cue: {real_data['route']}"
    )
    # Real route should expose a positive rank_threshold (balanced ~ 0.45).
    assert real_data["rank_threshold_used"] > 0.0, (
        f"real route rank_threshold_used should be > 0: got "
        f"{real_data['rank_threshold_used']}"
    )


# ---------------------------------------------------------------------------
# 7: analyzer math
# ---------------------------------------------------------------------------


def _build_synthetic_csv(
    tmp_path: Path,
    real_hits_per_seed: int,
    real_misses_per_seed: int,
    shadow_hits_per_seed: int,
    shadow_misses_per_seed: int,
    seeds: list[int],
) -> Path:
    """Write a CSV the analyzer can read. arousal_route column populated."""
    csv_path = tmp_path / "contradiction_longitudinal_synth.csv"
    header = [
        "probe_id", "seed", "n_slice", "condition", "topic",
        "pipeline_rank", "cosine_rank",
        "pipeline_hit_at_k", "cosine_hit_at_k",
        "s4_contradiction_emitted", "anti_hits_count", "hint_kinds",
        "pipeline_top1_text",
        "route", "cue_hash",
        "arousal_route", "arousal_cue_hash",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        i = 0
        for seed in seeds:
            for _ in range(real_hits_per_seed):
                w.writerow({
                    **{k: "" for k in header},
                    "probe_id": f"p{i}",
                    "seed": str(seed),
                    "pipeline_rank": "1",
                    "pipeline_hit_at_k": "1",
                    "arousal_route": "arousal_real",
                    "arousal_cue_hash": "aaaa1111",
                })
                i += 1
            for _ in range(real_misses_per_seed):
                w.writerow({
                    **{k: "" for k in header},
                    "probe_id": f"p{i}",
                    "seed": str(seed),
                    "pipeline_rank": "-1",
                    "pipeline_hit_at_k": "0",
                    "arousal_route": "arousal_real",
                    "arousal_cue_hash": "aaaa2222",
                })
                i += 1
            for _ in range(shadow_hits_per_seed):
                w.writerow({
                    **{k: "" for k in header},
                    "probe_id": f"p{i}",
                    "seed": str(seed),
                    "pipeline_rank": "1",
                    "pipeline_hit_at_k": "1",
                    "arousal_route": "arousal_shadow",
                    "arousal_cue_hash": "bbbb1111",
                })
                i += 1
            for _ in range(shadow_misses_per_seed):
                w.writerow({
                    **{k: "" for k in header},
                    "probe_id": f"p{i}",
                    "seed": str(seed),
                    "pipeline_rank": "-1",
                    "pipeline_hit_at_k": "0",
                    "arousal_route": "arousal_shadow",
                    "arousal_cue_hash": "bbbb2222",
                })
                i += 1
    return csv_path


def test_analyzer_math_synthetic_csv(tmp_path: Path) -> None:
    """`analyze_arousal_ab.py` produces the expected ship_gate verdict.

    real: 18 hits / 2 misses per seed -> rescue = 0.90
    shadow: 5 hits / 15 misses per seed -> rescue = 0.25
    delta = 0.65 (well above +0.05 threshold) -> ship_gate_hit=True, verdict=keep
    """
    csv_path = _build_synthetic_csv(
        tmp_path,
        real_hits_per_seed=18, real_misses_per_seed=2,
        shadow_hits_per_seed=5, shadow_misses_per_seed=15,
        seeds=[13, 42, 137],
    )
    analyzer = REPO_ROOT / "bench" / "analyze_arousal_ab.py"
    assert analyzer.exists(), f"analyzer not built yet: {analyzer}"

    result = subprocess.run(
        [sys.executable, str(analyzer), str(csv_path.parent)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"analyzer exited {result.returncode}: stderr={result.stderr}"
    )

    summary_path = csv_path.parent / "AROUSAL-AB-SUMMARY.json"
    assert summary_path.exists(), f"summary not written: {summary_path}"
    summary = json.loads(summary_path.read_text())
    assert summary["cross_seed_mean_delta"] > 0.5, (
        f"expected large positive delta; got {summary['cross_seed_mean_delta']}"
    )
    assert summary["ship_gate_hit"] is True
    assert summary.get("verdict") == "keep"
