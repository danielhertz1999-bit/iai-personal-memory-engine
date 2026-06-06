"""Stage 8 historical-verbatim anchor tests.

Bench scenario reconstruction:
- Gold record: "The fix ships in week 14."           (original / superseded)
- Wrong record: "Fix ETA revised: week 18."          (correction / current truth)
- Bench seeds the contradicts edge via retrieve.contradict(store, orig_id,
  corr_text, cue_emb) — which writes src=GOLD, dst=WRONG in the edges
  table. So `outgoing[GOLD] = [WRONG]` (gold has the outgoing contradicts
  edge); WRONG is the dst.

Current-fact primacy contract (softened anchor):
- On a historical_verbatim cue the corrector (WRONG / current truth) keeps
  its natural high-cosine rank — it ranks FIRST (or at least above the
  superseded original).
- The superseded original (GOLD) is anchored to just BELOW the corrector's
  natural score via the anchor pass, so it surfaces in the result set
  (top-10) even when its raw cosine rank is buried.
- historical_verbatim@10 = 1.000 is preserved (original still in top-10).
- The corrector is NOT downweighted; current-fact primacy is maintained.

Test contract:
1. Bench-mirroring scenario: corrector ranks above original; original in top hits.
2. Neutral cue (no historical marker): anchor does NOT fire.
3. Russian historical cue: corrector ranks above original; original surfaces.
4. Record with no contradicts edge: untouched; no spurious score change.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


class _BenchEmbedder:
    """Deterministic sha256-based embedder mirroring _PerfEmbedder shape.

    The Stage 8 test seeds records with hand-crafted cosines so the
    downweight effect is observable without depending on bge model load.
    """

    DIM = EMBED_DIM

    def __init__(self, base_seed: int = 0) -> None:
        self._base_seed = base_seed
        self._fixed: dict[str, list[float]] = {}

    def set_fixed(self, text: str, vec: list[float]) -> None:
        self._fixed[text] = list(vec)

    def embed(self, text: str) -> list[float]:
        if text in self._fixed:
            return list(self._fixed[text])
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


def _high_cos_variant(base_vec: list[float], noise_seed: int, noise_scale: float = 0.20) -> list[float]:
    """Build a vector that cosine-matches `base_vec` highly (>0.9) but is
    not identical to it. Used to give GOLD and WRONG embeddings that both
    score high vs the cue but are distinguishable by pattern_separation
    (cosine < 0.92 dedup threshold).
    """
    import hashlib
    import random
    digest = hashlib.sha256(f"{noise_seed}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    noise = [rng.random() * 2 - 1 for _ in range(len(base_vec))]
    # Normalize noise then mix with base.
    n_norm = sum(x * x for x in noise) ** 0.5
    if n_norm > 0:
        noise = [x / n_norm for x in noise]
    mixed = [
        (1.0 - noise_scale) * b + noise_scale * n
        for b, n in zip(base_vec, noise, strict=False)
    ]
    m_norm = sum(x * x for x in mixed) ** 0.5
    if m_norm > 0:
        mixed = [x / m_norm for x in mixed]
    return mixed


def _seed_bench_scenario(tmp_path, n_filler: int = 8):
    """Seed a store mirroring bench/contradiction_longitudinal_claude.py.

    Returns (store, embedder, graph, assignment, rich_club, gold_id, wrong_id,
    no_edge_id, cue_emb). Both GOLD and WRONG embeddings cosine-match the
    cue at >0.9 but are mutually distinguishable (cosine ~0.85, below the
    0.92 pattern_separation near_dup threshold) so contradict() creates a
    distinct WRONG record rather than deduping into GOLD.
    """
    from iai_mcp.retrieve import build_runtime_graph, contradict
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "hippo")
    embedder = _BenchEmbedder(base_seed=24)

    # Cue embedding — anchors the cosine geometry.
    cue_text = "Quote the original ETA wording."
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    # GOLD vector — high cosine vs cue but DELIBERATELY LOWER than WRONG.
    # This reproduces the bench failure mode: WRONG outscores GOLD on the
    # base cosine signal (wrong's score gap is ~0.18 from
    # cosine + degree). The Stage 8 historical_verbatim downweight is what
    # must flip this gap on the historical cue.
    gold_vec = _high_cos_variant(cue_vec, noise_seed=1001, noise_scale=0.30)
    gold_text = "The fix ships in week 14."
    embedder.set_fixed(gold_text, gold_vec)
    gold = _make_rec(gold_vec, gold_text, tags=["topic:bug_fix_eta"])
    store.insert(gold)
    gold_id_post_insert = gold.id  # captures dedup-merge id if any

    # WRONG vector — HIGHER cosine vs cue than GOLD. Different enough that
    # pattern_separation doesn't dedup it (cosine gold/wrong < 0.92).
    wrong_vec = _high_cos_variant(cue_vec, noise_seed=2002, noise_scale=0.15)
    corr_text = "Fix ETA revised: week 18."
    embedder.set_fixed(corr_text, wrong_vec)
    receipt = contradict(store, gold_id_post_insert, corr_text, list(wrong_vec))
    wrong_id = receipt.new_record_id
    assert wrong_id != gold_id_post_insert, (
        f"test fixture invariant broken: contradict() deduped WRONG into "
        f"GOLD (id={wrong_id}). Increase noise_scale or change seeds."
    )

    # NO_EDGE = a third record with no contradicts edges; mid cos vs cue
    # but no contradicts-edge participation.
    no_edge_vec = _high_cos_variant(cue_vec, noise_seed=3003, noise_scale=0.35)
    no_edge_text = "Unrelated auth tokens are rotated weekly."
    embedder.set_fixed(no_edge_text, no_edge_vec)
    no_edge = _make_rec(no_edge_vec, no_edge_text, tags=["topic:auth"])
    store.insert(no_edge)
    no_edge_id_post_insert = no_edge.id

    # Filler records (cosine-noise so the pool is non-trivial).
    tag_pool = [
        ["topic:auth"], ["topic:db"], ["topic:web"],
        ["topic:net"], ["topic:cli"],
    ]
    for i in range(n_filler):
        vec = embedder.embed(f"filler-{i}")
        tags = list(tag_pool[i % len(tag_pool)])
        rec = _make_rec(vec, text=f"unrelated fact {i}", tags=tags)
        store.insert(rec)

    graph, assignment, rich_club = build_runtime_graph(store)
    return (
        store, embedder, graph, assignment, rich_club,
        gold_id_post_insert, wrong_id, no_edge_id_post_insert, cue_vec,
    )


# ============================================================================
# Anchor tests — corrector-primacy on historical cue, no-anchor on neutral
# ============================================================================


def test_bench_failing_cue_corrector_ranks_above_original(tmp_path):
    """Bench-mirroring scenario: cue 'Quote the original ETA wording.' →
    the corrector 'Fix ETA revised: week 18.' (current truth) ranks above
    the superseded original 'The fix ships in week 14.' (GOLD). Both must
    appear in the result set; GOLD must be in top-10 (historical_verbatim@10
    = 1.000), but the corrector ranks strictly above it (current-fact
    primacy).

    Pre-softening: GOLD was anchored above the corrector (wrong contract).
    Post-softening: corrector keeps natural rank; GOLD anchors just below it.
    """
    from iai_mcp.pipeline import recall_for_benchmark

    (store, embedder, graph, assignment, rich_club,
     gold_id, wrong_id, _no_edge_id, _cue_vec) = _seed_bench_scenario(tmp_path)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue="Quote the original ETA wording.",
        session_id="bench-probe",
        k_hits=10, mode="concept",
    )
    assert len(resp.hits) >= 2, (
        f"need at least 2 hits to compare ranks; got {len(resp.hits)}"
    )

    hit_ids = [h.record_id for h in resp.hits]
    assert gold_id in hit_ids, (
        f"GOLD (superseded original) must be in top-10 hits "
        f"(historical_verbatim@10=1.000); not found. "
        f"Hits: {[(str(h.record_id)[:8], h.literal_surface) for h in resp.hits]}"
    )
    assert wrong_id in hit_ids, (
        f"Corrector (current truth) must also be present in hits. "
        f"Hits: {[(str(h.record_id)[:8], h.literal_surface) for h in resp.hits]}"
    )
    gold_rank = hit_ids.index(gold_id)
    wrong_rank = hit_ids.index(wrong_id)
    assert wrong_rank < gold_rank, (
        f"Corrector (current truth) must rank above the superseded original "
        f"(current-fact primacy). corrector_rank={wrong_rank} gold_rank={gold_rank}. "
        f"Hits: {[(str(h.record_id)[:8], h.literal_surface, round(h.score, 3)) for h in resp.hits[:5]]}"
    )


def test_superseded_original_in_top10_below_corrector_buried_cosine(tmp_path):
    """Regression guard for the post-migration embedder geometry: the
    superseded original (contradicts-src) must be in top-10 AND rank below
    its corrector on a historical-verbatim cue EVEN WHEN its raw cosine rank
    is buried far below the corrector by many higher-cosine distractors.

    The anchor pass lifts the buried original to just BELOW the corrector's
    natural score (corrector - epsilon), so the corrector (current truth)
    ranks first and the original surfaces in second position. Both are in
    top-10; historical_verbatim@10 = 1.000 is maintained.

    This is the exact geometry that dropped historical_verbatim 0.900 -> 0.713
    when the embedder swap buried keyword-less originals at cosine rank 53-152.
    """
    from iai_mcp.retrieve import build_runtime_graph, contradict
    from iai_mcp.store import MemoryStore
    from iai_mcp.pipeline import recall_for_benchmark

    store = MemoryStore(path=tmp_path / "hippo")
    embedder = _BenchEmbedder(base_seed=77)

    cue_text = "Quote the original ETA wording."
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    # GOLD (original): only WEAKLY similar to the cue (keyword-less original),
    # so distractors will out-cosine it — reproducing the buried-rank failure.
    gold_vec = _high_cos_variant(cue_vec, noise_seed=5101, noise_scale=0.62)
    gold_text = "The fix ships in week 14."
    embedder.set_fixed(gold_text, gold_vec)
    gold = _make_rec(gold_vec, gold_text, tags=["topic:bug_fix_eta"])
    store.insert(gold)
    gold_id = gold.id

    # WRONG (corrector): HIGH cosine to the cue — stays near the top.
    wrong_vec = _high_cos_variant(cue_vec, noise_seed=6202, noise_scale=0.12)
    corr_text = "Fix ETA revised: week 18."
    embedder.set_fixed(corr_text, wrong_vec)
    receipt = contradict(store, gold_id, corr_text, list(wrong_vec))
    wrong_id = receipt.new_record_id
    assert wrong_id != gold_id

    # Distractors that out-cosine GOLD but are NOT contradicts participants:
    # these are the "moderately-related records" that bury the keyword-less
    # original between the corrector and itself.
    for i in range(20):
        dvec = _high_cos_variant(cue_vec, noise_seed=7000 + i, noise_scale=0.40)
        embedder.set_fixed(f"distractor-{i}", dvec)
        store.insert(_make_rec(dvec, text=f"distractor fact {i}", tags=["topic:misc"]))

    graph, assignment, rich_club = build_runtime_graph(store)

    # Confirm the fixture actually buries GOLD on raw cosine (else the test
    # would pass trivially without exercising the anchor).
    import numpy as np
    cv = np.asarray(cue_vec, dtype=np.float32); cv /= np.linalg.norm(cv) + 1e-12
    cos_scored = []
    for r in store.all_records():
        v = np.asarray(r.embedding, dtype=np.float32); v /= np.linalg.norm(v) + 1e-12
        cos_scored.append((float(v @ cv), r.id))
    cos_scored.sort(key=lambda t: -t[0])
    gold_cos_rank = next(i for i, (_, rid) in enumerate(cos_scored, 1) if rid == gold_id)
    assert gold_cos_rank > 5, (
        f"fixture invariant broken: GOLD must be buried (cos_rank>5) for this "
        f"test to exercise the anchor; got cos_rank={gold_cos_rank}"
    )

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue=cue_text, session_id="bench-probe",
        k_hits=10, mode="concept",
    )
    hit_ids = [h.record_id for h in resp.hits]

    # Original must be in top-10 (historical_verbatim@10 = 1.000).
    assert gold_id in hit_ids, (
        f"Anchor must surface the buried superseded original (cos_rank="
        f"{gold_cos_rank}) into top-10 on the historical cue; not found. "
        f"Hits: {[(str(h.record_id)[:8], h.literal_surface, round(h.score,3)) for h in resp.hits]}"
    )
    # Corrector must also be present and rank above the original (current-fact primacy).
    assert wrong_id in hit_ids, (
        f"Corrector (current truth) must be in results on historical cue."
    )
    gold_rank = hit_ids.index(gold_id)
    wrong_rank = hit_ids.index(wrong_id)
    assert wrong_rank < gold_rank, (
        f"Corrector must rank above the superseded original (current-fact primacy). "
        f"corrector_rank={wrong_rank} gold_rank={gold_rank}. "
        f"Hits: {[(str(h.record_id)[:8], h.literal_surface, round(h.score,3)) for h in resp.hits[:5]]}"
    )


def test_neutral_cue_does_not_apply_downweight(tmp_path):
    """Negative case: on a NEUTRAL cue (no historical marker), the
    historical-verbatim anchor must NOT fire. WRONG (or GOLD — either
    can win normally) is allowed; the contract is that this cue does NOT
    invoke the anchor pass that places the original near the corrector.

    We verify this by asserting both records appear AND neither's score
    has been displaced by a large artificial offset. On a neutral cue
    only incidental signals (degree, age) can create a gap — the anchor
    is never applied.
    """
    from iai_mcp.pipeline import recall_for_benchmark

    (store, embedder, graph, assignment, rich_club,
     gold_id, wrong_id, _no_edge_id, _cue_vec) = _seed_bench_scenario(tmp_path)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue="What's the ETA?",  # neutral — no historical marker
        session_id="bench-probe",
        k_hits=5, mode="concept",
    )

    # Find both records in the result set.
    gold_hit = next((h for h in resp.hits if h.record_id == gold_id), None)
    wrong_hit = next((h for h in resp.hits if h.record_id == wrong_id), None)
    assert gold_hit is not None, "GOLD must appear in neutral cue results"
    # WRONG may or may not appear (it could be anti-hit-filtered) — only
    # check the downweight invariant when present.
    if wrong_hit is not None:
        # Both records share identical cue embedding, so their cos*W_COSINE
        # contribution is identical. Any score gap is from degree/age/
        # secondary signals — but neither should have lost 0.25 from the
        # historical_verbatim downweight.
        gap = abs(gold_hit.score - wrong_hit.score)
        from iai_mcp.pipeline import HISTORICAL_VERBATIM_DOWNWEIGHT
        # Sanity: a 0.25 downweight would make the gap ≥ 0.20. On the
        # neutral cue it should be much smaller (< 0.20) because only
        # incidental signals (degree, age) diverge.
        assert gap < HISTORICAL_VERBATIM_DOWNWEIGHT - 0.02, (
            f"On NEUTRAL cue the score gap should NOT reflect a "
            f"~{HISTORICAL_VERBATIM_DOWNWEIGHT} downweight; got gap={gap:.4f}. "
            f"GOLD={gold_hit.score:.4f} WRONG={wrong_hit.score:.4f}"
        )


def test_russian_historical_cue_corrector_ranks_above_original(tmp_path):
    """RU historical cue 'приведи оригинальную формулировку': corrector
    (current truth) ranks above the superseded original; original is present
    in top-10 (historical_verbatim@10 = 1.000).

    Same scenario, Russian phrasing. EN records (the bench gold/wrong are
    English text) ranked by the RU cue would have very low cosine, but
    the cue embedding here is shared with both records (test rigging) so
    the intent → anchor chain is the only discriminator.
    """
    from iai_mcp.pipeline import recall_for_benchmark

    (store, embedder, graph, assignment, rich_club,
     gold_id, wrong_id, _no_edge_id, cue_vec) = _seed_bench_scenario(tmp_path)
    # Pin the Russian cue embedding to the same vector so cosine ties out.
    ru_cue = "приведи оригинальную формулировку"
    embedder.set_fixed(ru_cue, cue_vec)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue=ru_cue,
        session_id="bench-probe",
        k_hits=10, mode="concept",
    )
    hit_ids = [h.record_id for h in resp.hits]
    assert gold_id in hit_ids, (
        f"RU historical cue: GOLD (superseded original) must be in top-10; not found. "
        f"Hits: {[(str(h.record_id)[:8], h.literal_surface) for h in resp.hits]}"
    )
    assert wrong_id in hit_ids, (
        f"RU historical cue: corrector must also be in top-10."
    )
    gold_rank = hit_ids.index(gold_id)
    wrong_rank = hit_ids.index(wrong_id)
    assert wrong_rank < gold_rank, (
        f"RU historical cue: corrector must rank above the superseded original "
        f"(current-fact primacy). corrector_rank={wrong_rank} gold_rank={gold_rank}."
    )


def test_record_without_contradicts_edge_unaffected_by_downweight(tmp_path):
    """A record with NO contradicts edges must be untouched by the
    historical-verbatim machinery — the superseded-original anchor must not
    move its score. The anchor is gated by participation in a contradicts
    edge (NO_EDGE participates in none), so its score on a historical cue
    must equal its score on a neutral cue.

    k_hits is set above the total record count (11 records in the scenario)
    to ensure NO_EDGE appears in the result even though the softened anchor
    places the corrector at rank 1 and original at rank 2 — the corrector no
    longer gets a downweight that might have pushed it below NO_EDGE.

    Note on why this is NOT a "gap vs GOLD" check: the historical path
    ANCHORS the superseded original (GOLD, the contradicts-src) just below
    its corrector so the original surfaces by association. GOLD legitimately
    scores near the corrector — that gap reflects GOLD being anchored, not
    NO_EDGE being penalized. The correct invariant for NO_EDGE is
    self-comparison across cue intents.
    """
    from iai_mcp.pipeline import recall_for_benchmark

    (store, embedder, graph, assignment, rich_club,
     _gold_id, _wrong_id, no_edge_id, cue_vec) = _seed_bench_scenario(tmp_path)

    # Pin a neutral cue to the SAME embedding as the historical cue so the
    # ONLY difference between the two recalls is the classified intent — not
    # the cosine geometry. (NO_EDGE's embedding is a high-cos variant of
    # cue_vec, so distinct cue text would change its cosine term and mask the
    # invariant we want to test.)
    neutral_cue = "current eta status"  # no historical/verbatim trigger
    embedder.set_fixed(neutral_cue, cue_vec)
    from iai_mcp.cue_router import _classify_cue
    assert _classify_cue(neutral_cue)[1] != "historical_verbatim", (
        "test fixture invariant: neutral cue must not classify as historical"
    )

    # k_hits=20 exceeds the scenario's 11 total records, ensuring NO_EDGE
    # is included regardless of its rank (the corrector is no longer depressed).
    hist = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue="Quote the original ETA wording.",  # historical_verbatim intent
        session_id="bench-probe",
        k_hits=20, mode="concept",
    )
    neutral = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue=neutral_cue,  # same embedding, neutral intent
        session_id="bench-probe",
        k_hits=20, mode="concept",
    )
    no_edge_hist = next((h for h in hist.hits if h.record_id == no_edge_id), None)
    no_edge_neutral = next(
        (h for h in neutral.hits if h.record_id == no_edge_id), None,
    )
    assert no_edge_hist is not None, (
        "NO_EDGE record should still appear in the ranked list (k_hits=20 "
        "covers all 11 records) — the historical machinery is targeted, not blanket"
    )
    assert no_edge_neutral is not None, "NO_EDGE must appear on the neutral cue"
    # Cue embedding is identical across both recalls; NO_EDGE's cosine term is
    # therefore identical. Its final score must not change between historical
    # and neutral intents, because it participates in no contradicts edge.
    # A leak in the superseded-original anchor would shift it.
    assert abs(no_edge_hist.score - no_edge_neutral.score) < 1e-6, (
        f"NO_EDGE score changed between historical and neutral cues "
        f"(hist={no_edge_hist.score:.6f} neutral={no_edge_neutral.score:.6f}); "
        f"it has no contradicts edge so the superseded-original anchor "
        f"must not touch it."
    )


def test_historical_verbatim_downweight_constant_is_module_level(tmp_path):
    """HISTORICAL_VERBATIM_DOWNWEIGHT must be a module-level constant on
    iai_mcp.pipeline and overridable via env IAI_MCP_HISTORICAL_VERBATIM_DOWNWEIGHT.
    """
    from iai_mcp import pipeline as _pipeline_mod

    assert hasattr(_pipeline_mod, "HISTORICAL_VERBATIM_DOWNWEIGHT"), (
        "pipeline.py must export HISTORICAL_VERBATIM_DOWNWEIGHT module constant"
    )
    assert isinstance(_pipeline_mod.HISTORICAL_VERBATIM_DOWNWEIGHT, float)
    # Default from RESEARCH.md A1 math: 0.25
    assert 0.0 < _pipeline_mod.HISTORICAL_VERBATIM_DOWNWEIGHT < 1.0, (
        f"HISTORICAL_VERBATIM_DOWNWEIGHT must be in (0, 1) for stability, "
        f"got {_pipeline_mod.HISTORICAL_VERBATIM_DOWNWEIGHT}"
    )


def test_bench_harness_calls_classify_cue_for_intent(tmp_path):
    """Bench harness wiring contract: bench/contradiction_longitudinal_claude.py
    must call _classify_cue before recall_for_benchmark so the intent flows
    through (or — equivalently — recall_for_benchmark calls _classify_cue
    internally so the bench gets intent routing without bench changes).

    We assert the END-TO-END behavior: when the bench passes the failing
    cue through recall_for_benchmark, the response reflects historical
    intent routing — corrector ranks first (current-fact primacy), original
    in top-10 (historical_verbatim@10 = 1.000).
    """
    from iai_mcp.pipeline import recall_for_benchmark

    (store, embedder, graph, assignment, rich_club,
     gold_id, wrong_id, _no_edge_id, _cue_vec) = _seed_bench_scenario(tmp_path)

    # This call mirrors bench/contradiction_longitudinal_claude.py
    # exactly: mode="concept", k_hits=200, cue passed verbatim.
    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue="Quote the original ETA wording.",
        session_id="bench-probe",
        k_hits=200,
        mode="concept",
    )
    assert len(resp.hits) >= 2
    hit_ids = [h.record_id for h in resp.hits]
    # Original must surface in top-10 (historical_verbatim@10 = 1.000).
    assert gold_id in hit_ids[:10], (
        "bench-shaped call (mode='concept', k_hits=200) must route "
        "historical_verbatim intent and include the original in top-10"
    )
    # Corrector must rank above the original (current-fact primacy).
    assert wrong_id in hit_ids, "corrector must be present in results"
    gold_rank = hit_ids.index(gold_id)
    wrong_rank = hit_ids.index(wrong_id)
    assert wrong_rank < gold_rank, (
        f"bench-shaped call: corrector must rank above original (current-fact primacy). "
        f"corrector_rank={wrong_rank} gold_rank={gold_rank}."
    )
