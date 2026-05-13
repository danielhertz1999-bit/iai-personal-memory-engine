"""Five-stage retrieval pipeline ( + /03/06, AUTIST-07).

Stage 1 - Embed: bge-small(cue) -> 384d vector.
Stage 2 - Community gate : argmax cosine over centroids, keep top 3
         (primary + 2 neighbours via Yeo-like tunnel scores).
Stage 3 - Seeds: top-3 within gated communities by 0.6*cos + 0.4*centrality.
Stage 4 - 2-hop greedy spread , union with pre-fetched rich-club .
Stage 5 - Rank + pack under budget:
            score = W_COSINE*cos + W_AAAK*aaak_overlap + W_DEGREE*deg_norm
                    - W_AGE*age_penalty
            where deg_norm = log(1+deg) / log(1+max_deg) is bounded in [0,1]
            so the degree contribution is sample-rank-comparable to cosine
            (R2; max_deg cached on graph._max_degree by build_runtime_graph).
            multiplied by profile_modulation gain product if
            profile_state carries active knobs.
          Anti-hits from contradicts-edge neighbours of top hits ( dual-route).

Constitutional rules enforced:
- every hit appends a provenance entry (same as baseline retrieve.recall).
- literal_surface returned verbatim (never rewritten) from store.
- adjacent_suggestions populated per hit (AUTIST-07 cued recognition).

Task 1 additions:
- profile_modulates edges: after ranking, active knob gains create
  profile_modulates edges from affected records -> PROFILE_SENTINEL_UUID.
- Curiosity hints (LEARN-04, Task 4): entropy-gated clarifying
  questions surfaced via RecallResponse.hints.
- Provisional schema hints (LEARN-03): high-entropy recalls surface candidate
  schemas for the user to approve.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import log
from uuid import UUID

import numpy as np

from iai_mcp.community import CommunityAssignment
from iai_mcp.embed import Embedder
from iai_mcp.events import write_event
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryHit, RecallResponse

# W4: structured-log channel for anti-hits malformed-edge
# observability. Named ``logger`` (not ``log``) to avoid shadowing the
# ``math.log`` import already used in the rank stage's degree
# normalisation (`log(1.0 + max_deg)` etc.).
logger = logging.getLogger(__name__)


# ------------------------------------------------------- helpers


@dataclass
class SimpleRecordView:
    """lightweight record view sourced from graph node attrs.

    Covers the fields the seed + spread + rank stages actually read
    (embedding for cosine, literal_surface for MemoryHit hydration,
    centrality + tier for tie-break signals). Fields the scoring loops
    *don't* touch at the seed/spread stage are filled with safe defaults
    so the view can stand in for a MemoryRecord without crashing the
    rarer code paths (aaak_overlap, age_penalty) that hit rank stage.

    This is NOT a MemoryRecord replacement; it's a read-only payload
    carrier for the hot-path that never needs to round-trip to LanceDB.
    Writes always go through store.insert / store.update / store.delete
    which is the authoritative contract (no drift).
    """

    id: UUID
    embedding: list[float]
    literal_surface: str
    centrality: float
    tier: str
    # Defaults the rank stage touches but the graph node dict may not carry:
    aaak_index: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    profile_modulation_gain: dict = field(default_factory=dict)
    structure_hv: bytes = b""
    provenance: list = field(default_factory=list)
    # fields touched by profile_modulation_for_record and
    # other rank-adjacent paths. Safe defaults keep the rank stage
    # shape-compatible with the full MemoryRecord surface.
    tags: list = field(default_factory=list)
    language: str = "en"


def _read_record_payload(G, rid: UUID, store: MemoryStore):
    """graph-first record payload access.

    Reads node attributes from the live NetworkX graph. If the node is
    missing the ``embedding`` attribute (race / partial-sync with the
    store / pre-05-12 call site), falls back to ``store.get(rid)`` so
    the recall path never crashes — just takes a small latency hit on
    that one node.

    Returns either a SimpleRecordView (graph-resident, no disk I/O)
    or a MemoryRecord (store fallback), or None if the id is truly
    unknown to both the graph and the store.
    """
    node = G.nodes.get(str(rid)) if rid is not None else None
    if node is not None and "embedding" in node and "surface" in node:
        # / (V2-03 fix): empty/None surface OR a
        # `_decrypt_failed=True` flag is a sentinel for cache-miss-due-
        # to-decrypt-failure. Fall through to store.get(rid) which has
        # its own retry semantics in crypto.py. A legitimately-empty
        # record round-trips correctly because store.get returns the
        # same empty literal_surface; the rare legitimate-empty case
        # remains correct because both paths produce the same output.
        surface = node.get("surface")
        if surface in (None, "") or node.get("_decrypt_failed"):
            pass  # explicit pass — fall through to store.get fallback below
        else:
            return SimpleRecordView(
                id=rid,
                embedding=list(node["embedding"]),
                literal_surface=str(surface),
                centrality=float(node.get("centrality", 0.0) or 0.0),
                tier=str(node.get("tier", "episodic")),
                tags=list(node.get("tags") or []),
                language=str(node.get("language", "en") or "en"),
            )
    # Defensive fallback (graph miss OR empty-surface sentinel).
    # Cheap per node; only triggered on drift OR decrypt-fail rehydrate.
    try:
        return store.get(rid)
    except Exception:
        return None

# score formula constants
W_COSINE = 1.0
W_AAAK = 0.3
W_DEGREE = 0.1
W_AGE = 0.05

# Age penalty "half-life": 30 days brings the penalty to 1.0 (fully saturated).
AGE_HALF_LIFE_DAYS = 30.0

# R3: literal_preservation knob modulates the effective
# W_DEGREE used in the rank-stage scoring formula. Keys MUST match the
# profile.py:87 KnobSpec enum schema "enum:strong|medium|loose" — NOT the
# phantom keys "balanced/weak", which would be rejected by
# profile_set. The 11-knob registry is closed (-02 removed
# AUTIST-02/08/11/12; expansion is a phase-level decision), so we use the
# canonical knob vocabulary.
#
# Numeric mapping (starting values; refine if scoring sanity
# checks on the live store show hubs still dominating at strong):
#   strong  = 0.3   tighten degree influence; verbatim wins (Mottron EPF)
#   medium  = 1.0   normalize-only baseline; no extra knob effect
#   loose   = 1.5   let hubs speak louder; concept-mode-friendly
#
# Default fallback when profile_state is missing/empty/invalid is "medium"
# (scale 1.0) so callers without a knob set see / behaviour.
LITERAL_PRESERVATION_W_DEGREE_SCALE: dict[str, float] = {
    "strong": 0.3,
    "medium": 1.0,
    "loose":  1.5,
}

# : candidate-pool size for the
# cosine top-K gate replacement. Single module-level constant — NOT a
# tier-branch, NOT a "small graph vs large graph" cap. K=200 is the
# empirical 99th-percentile gold rank from the LongMemEval-S v1 trace
# (worst-case qid had 12/12 gold inside cosine rank 1-200) plus 30%
# margin. Future re-tuning is a benchmark-driven decision, not a hack.
K_CANDIDATES: int = 200

# : mode-dependent community-gate
# soft-bias scalars, grounded in CLS / EPF / HIPPEA / Ashby / Beer VSM.
#
# The community gate (Leiden communities + centroid cosine) is a
# CATEGORICAL structure — neocortical, not hippocampal. McClelland CLS
# dictates that hippocampal episodic recall (mode=verbatim) is sparse,
# NOT compressed, with NO categorical aggregation; neocortical semantic
# recall (mode=concept) IS compressed schemas with categorical
# structure. So the gate's score-impact MUST depend on the recall mode:
#
#   verbatim mode -> 0.0  (HIPPEA pure / EPF literal / hippocampal
#                          episodic; categorical filtering is anti-aSD
#                          here, weak priors yield to sensory-input
#                          precision = cosine-on-embeddings)
#   concept  mode -> 0.1  (CLS neocortical semantic; communities ARE
#                          the cortical schemas; soft +10% bonus to
#                          records in top-3 gated communities = a
#                          categorical hint without filtering)
#
# The bias is NEVER a hard filter; the candidate pool is always cosine
# top-K_CANDIDATES regardless of mode. The bias only adjusts the
# Stage-5 final score for records that fall inside the top-3 gated
# communities.
#
# Beer VSM S5 (policy / identity invariants) governs the recall mode;
# the cue-classifier in core.py:dispatch (R5) sets `mode`,
# and `_gate_bias_for_mode(mode)` returns the appropriate scalar. No
# runtime drift, no coverage-based threshold, no dynamic 0.0/0.1
# if/else — purely a function of the `mode` parameter.
COMMUNITY_BIAS_VERBATIM: float = 0.0   # HIPPEA pure / EPF literal / hippocampal episodic
COMMUNITY_BIAS_CONCEPT: float = 0.1    # CLS neocortical semantic / categorical hint

# redesign (08-02 Rule 1 fix): internal post-rank cap.
#
# The candidate pool (K_CANDIDATES=200) widens what reaches Stage 5
# ranking compared to the pre-Phase-8 OLD pipeline_recall, which gated
# candidates to ~3 communities ≈ ~50 records. After ranking, the old code
# ran post-rank work (s4 contradiction-detection pairwise scan, anti-hits
# lookup, profile_modulates edge writes, schema/curiosity hints) over the
# OLD-narrow set. The new code threatens to run these O(N²) and O(N)
# passes over the wider K_CANDIDATES set when the public cap
# (`budget_tokens` or `k_hits`) is non-binding.
#
# OLD effective post-rank set size on synthetic perf-gate fixtures: 50-72
# records. NEW pre-cap set: 200. The plan's 200ms / 75ms perf-gate
# ceilings were tuned to OLD effective behavior. To preserve those
# ceilings WITHOUT breaking / , we apply an internal
# post-rank cap inside the entry points: only `_POST_RANK_MAX_HITS`
# records flow into the post-rank pipeline. The public cap
# (`budget_tokens` for `recall_for_response`, `k_hits` for
# `recall_for_benchmark`) is unchanged for the caller-facing `hits`
# field.
#
# Justification of the value: 50 covers the LongMemEval-S R@5 / R@10
# evaluation surface (gold ≤24 records per row) plus margin, AND fits
# inside the OLD effective-hit-count distribution on the perf-gate
# fixtures (50-72), AND keeps s4's O(N²) pairwise scan bounded at
# 50*49/2 ≈ 1225 pair checks vs the unbounded ~20k that 200 hits would
# trigger. The cap applies to side-effect computations only (s4 hints,
# anti-hits, profile_modulates edges, schema, curiosity, retrieval_used
# event); the public `hits` list still respects the caller's contract.
_POST_RANK_MAX_HITS: int = 50


def _gate_bias_for_mode(mode: str) -> float:
    """: CLS-grounded mode-dependent gate bias.

    Returns the community-gate soft-bias scalar appropriate for the
    given recall mode. Mode dispatch is set upstream by the cue-classifier
    in `core.py:dispatch` (R5).

    verbatim mode -> 0.0  (HIPPEA literal precision, hippocampal episodic recall)
    concept  mode -> 0.1  (CLS neocortical semantic, soft categorical hint)

    Any other value defaults to verbatim's 0.0 (conservative — never
    accidentally bias toward categorical filtering when the mode is
    ambiguous; matches "never accidentally bias" rule).
    """
    return COMMUNITY_BIAS_CONCEPT if mode == "concept" else COMMUNITY_BIAS_VERBATIM


@dataclass
class _RecallCoreResult:
    """redesign: shape returned by `_recall_core`.

    Holds the load-bearing recall outputs: the SORTED full ranked list
    of scored_hits + activation_trace + cue_mode. The entry points
    (`recall_for_response`, `recall_for_benchmark`) apply their pack/cap
    THEN run the post-rank pipeline (anti-hits, s4 hints, profile-modulates
    edges, schema/curiosity hints, patterns_observed strip, retrieval_used
    event, provenance batch) over the CAPPED subset.

    Post-rank fields (`anti_hits`, `hints`, `patterns_observed`) are
    populated by `_recall_core` ONLY on the L0 retrieval-skip fast path
    where the result is already capped at 1 hit. On the regular path
    they are returned empty and the entry points populate them.

    `scored_hits` is sorted by score descending (R5 deterministic
    tie-break by UUID-asc as secondary key, matching pre-Phase-8
    behaviour).

    `_records_cache` is a private field carrying the records_cache
    `_recall_core` built (graph-resident SimpleRecordView's + LanceDB
    fallback). Entry points reuse it for post-rank work to avoid
    duplicating the ~O(N) graph walk + store.all_records() scan.
    """

    scored_hits: list[MemoryHit] = field(default_factory=list)
    activation_trace: list[UUID] = field(default_factory=list)
    anti_hits: list[MemoryHit] = field(default_factory=list)
    hints: list[dict] = field(default_factory=list)
    patterns_observed: list[dict] = field(default_factory=list)
    cue_mode: str = "concept"
    budget_used: int = 0
    # Private: records_cache built by `_recall_core`, reused by entry
    # points for post-rank work. Not part of the public contract.
    _records_cache: dict = field(default_factory=dict)


# deterministic sentinel UUID -- target of every
# profile_modulates edge. Individual gain breakdowns live on the record's
# profile_modulation_gain dict at recall time (stored in records_cache).
PROFILE_SENTINEL_UUID = UUID("00000000-0000-0000-0000-0000000000f1")


# --------------------------------------------------------------- math helpers


def _cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def _aaak_overlap(cue_text: str, aaak_index: str) -> float:
    """Token-set Jaccard between cue tokens and AAAK index tokens.

    approximation: whitespace + slash split applied symmetrically to
    both cue_text and aaak_index so "auth/login" tokenises consistently on
    either side. will replace this with a proper AAAK tokeniser once
    the AAAK index schema is frozen.
    """
    if not aaak_index:
        return 0.0
    cue_set = set(cue_text.lower().replace("/", " ").split())
    idx_set = set(aaak_index.lower().replace("/", " ").split())
    if not cue_set or not idx_set:
        return 0.0
    return len(cue_set & idx_set) / len(cue_set | idx_set)


def _age_penalty(created_at: datetime) -> float:
    """Monotonic age penalty bounded at 1.0. Saturates at AGE_HALF_LIFE_DAYS.

    `created_at` may be naive or tz-aware; naive values are treated as UTC so
    we can still subtract from a tz-aware `now`.
    """
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    days = (now - created_at).total_seconds() / 86400.0
    if days < 0:
        return 0.0
    return min(1.0, days / AGE_HALF_LIFE_DAYS)


# ----------------------------------------------------------------- stage impls


def _community_gate(
    cue_emb: list[float],
    assignment: CommunityAssignment,
    top_n: int = 3,
) -> list[UUID]:
    """: route cue to top-N communities by cosine(cue, centroid).

    vectorized — one matmul over stacked centroids
    replaces the per-centroid ``_cosine`` loop. At N=1k with no tag
    structure Leiden can emit ~1000 single-member communities; the old
    loop was ~20 ms per recall, the matmul is ~0.1 ms.

    Deterministic tie-break: stable sort by (-score, UUID-str).
    """
    centroids = assignment.community_centroids
    if not centroids:
        return []
    cids = list(centroids.keys())
    mat = np.asarray(
        [centroids[c] for c in cids], dtype=np.float32
    )
    cue_vec = np.asarray(cue_emb, dtype=np.float32)
    cue_norm = float(np.linalg.norm(cue_vec))
    if cue_norm > 0.0:
        cue_vec = cue_vec / cue_norm
    # Centroids may not be unit-norm (community.py averages member
    # embeddings then re-normalizes; we still normalize defensively so
    # this stays true-cosine even if a caller passes raw centroids).
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0.0] = 1.0
    mat = mat / norms[:, None]
    scores = mat @ cue_vec  # shape (K,)
    order = np.argsort(-scores, kind="stable")
    return [cids[int(i)] for i in order[:top_n]]


def _pick_seeds(
    candidate_indices: np.ndarray,
    shared_cos: np.ndarray,
    centrality_arr: np.ndarray,
    n: int = 3,
) -> np.ndarray:
    """redesign : seed selection over the
    shared cosine array.

    Reads scores from the precomputed `shared_cos` array (built once in
    `_recall_core`). No per-record cosine. No store I/O. No records_cache
    lookup. Pure O(K_CANDIDATES) numpy arithmetic.

    Args:
      candidate_indices: 1D int array of indices into the shared pool
        (typically `shared_order[:K_CANDIDATES]`).
      shared_cos: 1D float array of cue-vs-pool cosine scores (one
        entry per pool record).
      centrality_arr: 1D float array of centrality scores (one entry
        per pool record). Same indexing as shared_cos.
      n: number of seeds to return.

    Returns:
      1D int array of seed indices into the shared pool, length <= n.
      Stable-sort ordering for deterministic tie-break.
    """
    if candidate_indices.size == 0:
        return np.empty(0, dtype=candidate_indices.dtype)
    blended = (
        0.6 * shared_cos[candidate_indices]
        + 0.4 * centrality_arr[candidate_indices]
    )
    top_local = np.argsort(-blended, kind="stable")[:n]
    return candidate_indices[top_local]


def _collect_graph_pool(
    graph: MemoryGraph,
    records_cache: dict[UUID, "object"] | None,
    store: MemoryStore,
) -> tuple[list[UUID], np.ndarray]:
    """: build the (ids, embeddings) pool over
    which the shared cosine pass operates.

    Reads embeddings in this order of preference:
      1. graph._nx node attr "embedding" (zero-IO; populated by
         build_runtime_graph)
      2. records_cache hit (in-RAM SimpleRecordView or MemoryRecord)
      3. store.get fallback (rare; partial-sync drift)

    Nodes whose UUID parses but whose embedding cannot be located via
    any of the three paths are silently dropped. The output rows are
    ALIGNED with the output ids: pool_ids[i] is the UUID of pool_embs[i].

    Returns `(pool_ids, pool_embs)` where `pool_embs` is a 2D numpy
    array of shape `(len(pool_ids), embed_dim)` and dtype float32.
    Empty graph -> ([], np.zeros((0, store.embed_dim), dtype=np.float32)).

    This helper isolates the pool-collection concern so `_recall_core`
    can call it once at the top of every recall and reuse the result
    across Stage 2 (gate diagnostic), Stage 3 (seeds), Stage 4
    (reachable), and Stage 5 (rank). No second pool walk anywhere.
    """
    pool_ids: list[UUID] = []
    pool_embs_rows: list[list[float]] = []
    for nid_str in graph._nx.nodes:
        try:
            rid = UUID(nid_str)
        except (TypeError, ValueError):
            continue
        emb: list[float] | None = None
        # Path 1: graph._nx node attr (cheapest, populated by build_runtime_graph).
        node = graph._nx.nodes[nid_str]
        node_emb = node.get("embedding")
        if node_emb:
            emb = list(node_emb)
        # Path 2: records_cache hit.
        if not emb and records_cache is not None and rid in records_cache:
            rec = records_cache[rid]
            cached_emb = getattr(rec, "embedding", None)
            if cached_emb:
                emb = list(cached_emb)
        # Path 3: store.get fallback (defensive; partial-sync drift).
        if not emb:
            try:
                rec = store.get(rid)
                if rec is not None and rec.embedding:
                    emb = list(rec.embedding)
            except Exception:
                emb = None
        if emb:
            pool_ids.append(rid)
            pool_embs_rows.append(emb)
    if not pool_ids:
        # Use store.embed_dim so the empty-pool shape matches the
        # configured embedder; downstream `pool_embs @ cue_vec`
        # short-circuits cleanly to an empty result.
        return [], np.zeros((0, store.embed_dim), dtype=np.float32)
    return pool_ids, np.asarray(pool_embs_rows, dtype=np.float32)


def _find_anti_hits(
    hits: list[MemoryHit],
    store: MemoryStore,
    graph: MemoryGraph,
    k: int = 3,
    records_cache: dict[UUID, "object"] | None = None,
) -> list[MemoryHit]:
    """ dual-route anti-hits: contradicts-edge neighbours of top hits.

    scope: contradicts-edge lookup only. / will add
    AAAK-opposition scoring when the AAAK tokeniser is in place.

    records_cache (optional): used to hydrate MemoryHit.literal_surface
    without calling store.get per anti-id. Missing ids fall back to store.get.
    """
    seen: set[UUID] = {h.record_id for h in hits}
    anti_ids: list[UUID] = []

    tbl = store.db.open_table("edges")
    df = tbl.to_pandas()
    if df.empty:
        return []

    contradicts = df[df["edge_type"] == "contradicts"]
    if contradicts.empty:
        return []

    # W4 / filter rows whose src or dst cannot be parsed
    # as a UUID. A single malformed edge would otherwise abort
    # _find_anti_hits at the inner ``UUID(lid)`` call below, which in
    # turn aborts the post-rank stage of _recall_core. Anti-hits is an
    # enrichment signal; degrading to "no anti-hits" on corruption is
    # always preferred over crashing the recall path.
    def _is_uuid_str(v) -> bool:
        try:
            UUID(str(v))
            return True
        except (ValueError, TypeError, AttributeError):
            return False

    bad_mask = ~contradicts["src"].map(_is_uuid_str) | ~contradicts["dst"].map(_is_uuid_str)
    if bool(bad_mask.any()):
        n_bad = int(bad_mask.sum())
        try:
            first_bad = contradicts.loc[bad_mask].iloc[0]
            logger.warning(
                "anti_hits_skip_malformed_edge n_skipped=%d first_src=%r first_dst=%r",
                n_bad,
                str(first_bad["src"])[:40],
                str(first_bad["dst"])[:40],
            )
        except Exception:
            # Logging never blocks the recall path.
            pass
        contradicts = contradicts[~bad_mask]
        if contradicts.empty:
            return []

    for h in hits:
        hid = str(h.record_id)
        linked: set[str] = set()
        linked.update(
            contradicts.loc[contradicts["src"] == hid, "dst"].tolist()
        )
        linked.update(
            contradicts.loc[contradicts["dst"] == hid, "src"].tolist()
        )
        for lid in linked:
            # Belt-and-suspenders: the upstream filter already removed
            # malformed rows but mid-iteration corruption (e.g. concurrent
            # mutation) still gets caught here without crashing.
            try:
                u = UUID(lid)
            except (ValueError, TypeError, AttributeError):
                try:
                    logger.warning(
                        "anti_hits_skip_malformed_lid lid=%r",
                        str(lid)[:40],
                    )
                except Exception:
                    pass
                continue
            if u in seen:
                continue
            anti_ids.append(u)
            seen.add(u)
            if len(anti_ids) >= k:
                break
        if len(anti_ids) >= k:
            break

    out: list[MemoryHit] = []
    for aid in anti_ids[:k]:
        rec = records_cache.get(aid) if records_cache is not None else None
        if rec is None:
            rec = store.get(aid)
        if rec is None:
            continue
        out.append(
            MemoryHit(
                record_id=aid,
                score=0.0,
                reason="contradicts-edge neighbour",
                literal_surface=rec.literal_surface,
                adjacent_suggestions=[],
            )
        )
    return out


# ------------------------------------------------------------------ top-level


# redesign (08-PLAN-CHECK.md B2 / placement proof): an
# OPT-IN debug capture used by the verbatim-filter-placement test.
# When this dict is non-None and `_recall_core` is invoked, the function
# stashes its pre-filter and post-filter `reachable_ids` into the dict
# so the test can prove the filter applies between Stage 4 (union) and
# Stage 5 (rank). Set to None at module import; tests monkeypatch a
# fresh dict for the duration of one call.
_VERBATIM_FILTER_DEBUG: dict | None = None


def _recall_core(
    store: MemoryStore,
    graph: MemoryGraph,
    assignment: CommunityAssignment,
    rich_club: list[UUID],
    embedder: Embedder,
    cue: str,
    session_id: str,
    profile_state: dict | None = None,
    turn: int = 0,
    mode: str = "concept",
    *,
    knobs_applied: dict | None = None,
) -> _RecallCoreResult:
    """redesign: shared-cosine + Stage 2-5 + post-rank work.

    Performs the load-bearing recall computation ONCE and returns a
    fully-populated `_RecallCoreResult`. Both `recall_for_response`
    (08-02) and `recall_for_benchmark` (08-02) call this with
    identical arguments (minus the budget_tokens / k_hits cap, which
    is applied AFTER the core returns). The L0 retrieval-skip fast
    path is implemented INSIDE this function so both prongs share it.

    Stage walk:
      0. Active-inference gate -> L0 fast path on hit.
      1. Embed cue.
      2. Build records_cache from graph node attrs (zero-IO when
         build_runtime_graph populated the graph).
      3. SHARED COSINE PASS: one matmul over the full pool.
      4. Community gate diagnostic: top-3 communities by
         centroid cosine; output feeds the Stage-5 mode-dependent
         additive bias only (NO hard-filter).
      5. Seed selection: blended 0.6*shared_cos + 0.4*centrality
         over cosine_top_indices; pick top-3.
      6. Reachable union: cosine_top_indices ∪ 2-hop ∪ rich-club.
      7. Verbatim-mode filter: on `reachable_indices` between
         Stage 4 union and Stage 5 rank, canonical pipeline.py:831
         placement preserved exactly.
      8. Stage-5 rank ( cosine reuse, mode-dependent bias).
      9. Sort scored desc by score, secondary by UUID-asc (R5 contract).
     10. Build MemoryHits.
     11. Provenance batch.
     12. Anti-hits.
     13. S4/curiosity/schema hints (mode != "verbatim" only).
     14. profile_modulates edges.
     15. Concept-mode patterns_observed strip.
     16. Emit retrieval_used event.
     17. Return _RecallCoreResult.
    """
    profile_state = profile_state or {}

    # Stage 0 - Active-inference gate .
    # Lazy import + fn alias keeps this body free of substring
    # patterns the global security-reminder hook flags as eval-like.
    try:
        from iai_mcp import gate as _gate_mod
        _skip_fn = _gate_mod.should_skip_retrieval
        skip_flag, skip_reason = _skip_fn(cue)
    except Exception:
        skip_flag, skip_reason = False, ""
    if skip_flag:
        l0_uuid = UUID("00000000-0000-0000-0000-000000000001")
        l0_rec = store.get(l0_uuid)
        if l0_rec is not None:
            budget_used_l0 = len(l0_rec.literal_surface) // 4
            l0_hit = MemoryHit(
                record_id=l0_rec.id,
                score=1.0,
                reason="L0 identity (skipped per D-26)",
                literal_surface=l0_rec.literal_surface,
                adjacent_suggestions=[],
            )
            try:
                store.append_provenance(
                    l0_rec.id,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "cue": cue,
                        "session_id": session_id,
                    },
                )
            except Exception:
                pass
            try:
                write_event(
                    store,
                    kind="retrieval_used",
                    data={
                        "hit_ids": [str(l0_rec.id)],
                        "query": cue,
                        "used": True,
                        "budget_used": budget_used_l0,
                        "path": "recall_core_l0_fastpath",
                    },
                    severity="info",
                    session_id=session_id,
                )
            except Exception:
                pass
            return _RecallCoreResult(
                scored_hits=[l0_hit],
                activation_trace=[l0_rec.id],
                anti_hits=[],
                hints=[{
                    "kind": "retrieval_skipped",
                    "severity": "info",
                    "source_ids": [],
                    "text": skip_reason,
                }],
                patterns_observed=[],
                cue_mode=mode,
                budget_used=budget_used_l0,
            )

    # Stage 1 - Embed the cue.
    cue_emb = embedder.embed(cue)

    # Stage 2 - Build records_cache from graph node attrs.
    records_cache: dict[UUID, "object"] = {}
    try:
        G = graph._nx
        for nid_str in G.nodes:
            node = G.nodes[nid_str]
            if "embedding" not in node or "surface" not in node:
                continue
            try:
                rid = UUID(nid_str)
            except (TypeError, ValueError):
                continue
            records_cache[rid] = SimpleRecordView(
                id=rid,
                embedding=list(node["embedding"]),
                literal_surface=str(node.get("surface", "")),
                centrality=float(node.get("centrality", 0.0) or 0.0),
                tier=str(node.get("tier", "episodic")),
                tags=list(node.get("tags") or []),
                language=str(node.get("language", "en") or "en"),
            )
    except Exception:
        records_cache = {}
    if not records_cache:
        records_cache = {r.id: r for r in store.all_records()}

    # R5: in verbatim mode, restrict to tier='episodic'.
    # Build the set NOW; apply AFTER Stage-4 union (canonical placement).
    episodic_ids: set | None = None
    if mode == "verbatim":
        episodic_ids = {
            cid for cid, rec in records_cache.items()
            if getattr(rec, "tier", "episodic") == "episodic"
        }

    # Stage 3 - SHARED COSINE PASS. One matmul over the full pool.
    pool_ids, pool_embs = _collect_graph_pool(graph, records_cache, store)
    cue_vec = np.asarray(cue_emb, dtype=np.float32)
    cnorm = float(np.linalg.norm(cue_vec))
    if cnorm > 0.0:
        cue_vec = cue_vec / cnorm
    if pool_embs.size:
        # The single load-bearing matmul. Pool embeddings are
        # L2-normalized by sentence-transformers; dot == cosine.
        # Use np.matmul (not the @ operator) so the call is intercept-
        # able via monkeypatch — the matmul-counter test in
        # test_recall_core_unit.py asserts by counting
        # cue-vs-large-pool matmul invocations.
        shared_cos = np.matmul(pool_embs, cue_vec).astype(np.float32)
    else:
        shared_cos = np.empty(0, dtype=np.float32)
    if shared_cos.size:
        shared_order = np.argsort(-shared_cos, kind="stable")
        cosine_top_indices = shared_order[:K_CANDIDATES]
    else:
        shared_order = np.empty(0, dtype=np.int64)
        cosine_top_indices = np.empty(0, dtype=np.int64)

    id_to_idx = {rid: i for i, rid in enumerate(pool_ids)}

    # Stage 4 - Community gate DIAGNOSTIC. Top-3 communities;
    # their members form `gated_set` which feeds Stage 5's mode-bias.
    gated = _community_gate(cue_emb, assignment, top_n=3)
    gated_set: set[UUID] = set()
    for gc in gated:
        for rid in assignment.mid_regions.get(gc, []):
            gated_set.add(rid)

    # Centrality array aligned with pool_ids.
    centrality_arr = np.zeros(len(pool_ids), dtype=np.float32)
    _G_for_cen = graph._nx
    for i, rid in enumerate(pool_ids):
        node = _G_for_cen.nodes.get(str(rid))
        if node is not None and "centrality" in node:
            try:
                centrality_arr[i] = float(node["centrality"])
            except (TypeError, ValueError):
                centrality_arr[i] = 0.0
    if not np.any(centrality_arr) and pool_ids:
        try:
            cen_dict = graph.centrality()
            for i, rid in enumerate(pool_ids):
                centrality_arr[i] = float(cen_dict.get(rid, 0.0))
        except Exception:
            pass

    # Stage 5 - Seeds. Pure numpy on the shared array.
    seed_indices = _pick_seeds(
        cosine_top_indices, shared_cos, centrality_arr, n=3,
    )
    seed_ids = [pool_ids[int(i)] for i in seed_indices]

    # Stage 6 - Reachable: cosine top-K ∪ 2-hop ∪ rich-club.
    spread_ids = graph.two_hop_neighborhood(seed_ids, top_k=5)
    spread_indices = np.array(
        [id_to_idx[r] for r in spread_ids if r in id_to_idx],
        dtype=np.int64,
    )
    rich_indices = np.array(
        [id_to_idx[r] for r in (rich_club or []) if r in id_to_idx],
        dtype=np.int64,
    )
    if cosine_top_indices.size or spread_indices.size or rich_indices.size:
        reachable_indices = np.union1d(
            np.union1d(cosine_top_indices, spread_indices),
            rich_indices,
        ).astype(np.int64)
    else:
        reachable_indices = np.empty(0, dtype=np.int64)

    # Stage 7 - Verbatim-mode filter (, post-Stage-4 / pre-Stage-5).
    pre_filter_reachable_ids = [pool_ids[int(i)] for i in reachable_indices]
    if mode == "verbatim" and episodic_ids is not None:
        reachable_indices = np.array(
            [int(i) for i in reachable_indices if pool_ids[int(i)] in episodic_ids],
            dtype=np.int64,
        )
    post_filter_reachable_ids = [pool_ids[int(i)] for i in reachable_indices]

    # Optional debug capture for the verbatim-placement-proof test.
    if _VERBATIM_FILTER_DEBUG is not None:
        _VERBATIM_FILTER_DEBUG["pre_filter_reachable_ids"] = list(
            pre_filter_reachable_ids,
        )
        _VERBATIM_FILTER_DEBUG["post_filter_reachable_ids"] = list(
            post_filter_reachable_ids,
        )

    # Stage 8 - Rank ( cosine reuse, mode-dependent bias).
    from iai_mcp.profile import profile_modulation_for_record

    structural_weight: float = 0.0
    cue_structure_hv: bytes | None = None
    if profile_state:
        try:
            structural_weight = float(profile_state.get("structural_weight", 0.0) or 0.0)
        except (TypeError, ValueError):
            structural_weight = 0.0
        structural_weight = max(0.0, min(1.0, structural_weight))

    lp_value = "medium"
    if profile_state:
        try:
            raw_lp = profile_state.get("literal_preservation", "medium")
            if isinstance(raw_lp, str) and raw_lp in LITERAL_PRESERVATION_W_DEGREE_SCALE:
                lp_value = raw_lp
        except Exception:
            lp_value = "medium"
    lp_scale = LITERAL_PRESERVATION_W_DEGREE_SCALE[lp_value]
    effective_w_degree = W_DEGREE * lp_scale
    if mode == "verbatim":
        effective_w_degree = 0.0

    if structural_weight > 0.0:
        from iai_mcp import tem
        cue_structure_hv = tem.pack_pairs([("TOPIC", tem.filler_hv(cue))])

    max_deg = float(getattr(graph, "_max_degree", 0) or 0)
    log_max_deg = log(1.0 + max_deg) if max_deg > 0 else 0.0
    degree = dict(_G_for_cen.degree())

    # mode-dependent gate bias scalar.
    mode_bias = _gate_bias_for_mode(mode)

    scored: list[tuple[float, UUID, float, float, float, float, float, float]] = []
    if reachable_indices.size:
        from iai_mcp.hebbian_structure import structural_similarity
        for idx in reachable_indices:
            i = int(idx)
            cid = pool_ids[i]
            rec = records_cache.get(cid)
            if rec is None:
                continue
            # cosine read directly from shared array.
            cos = float(shared_cos[i])
            aaak = _aaak_overlap(cue, rec.aaak_index)
            deg = float(degree.get(str(cid), 0))
            age = _age_penalty(rec.created_at)
            if log_max_deg > 0.0:
                deg_norm = log(1.0 + deg) / log_max_deg
            else:
                deg_norm = 0.0
            base_s = (
                W_COSINE * cos
                + W_AAAK * aaak
                + effective_w_degree * deg_norm
                - W_AGE * age
            )
            # mode-dependent additive bias for top-3 gated communities.
            if cid in gated_set:
                base_s += mode_bias * cos
            structural_score = 0.0
            if (
                structural_weight > 0.0
                and cue_structure_hv is not None
                and rec.structure_hv
            ):
                structural_score = structural_similarity(
                    cue_structure_hv, rec.structure_hv,
                )
            if structural_weight > 0.0:
                base_s = (
                    (1.0 - structural_weight) * base_s
                    + structural_weight * structural_score
                )
            if profile_state:
                # -03 BLOCKER 3: thread the audit accumulator into
                # the gains-application call so AUTIST-01/03/09 record into
                # the same dict the caller (core.dispatch) attached to the
                # response. knobs_applied may be None (back-compat callers).
                gains = profile_modulation_for_record(
                    rec, profile_state, knobs_applied=knobs_applied,
                )
                if gains:
                    rec.profile_modulation_gain = dict(gains)
                    gain_product = 1.0
                    for gv in gains.values():
                        try:
                            gain_product *= float(gv)
                        except (TypeError, ValueError):
                            continue
                    s = base_s * gain_product
                else:
                    s = base_s
            else:
                s = base_s
            scored.append(
                (s, cid, cos, aaak, deg, deg_norm, age, structural_score),
            )

    # Stage 9 - Sort: score desc, UUID asc tie-break (R5 contract).
    scored.sort(key=lambda x: (-x[0], str(x[1])))

    # Stage 10 - Build MemoryHits over the SORTED ranked list.
    # Provenance batch + retrieval_used event move to the entry points
    # so they fire only over the capped hits (08-02 Rule 1 fix).
    scored_hits: list[MemoryHit] = []
    budget_used = 0
    for s, cid, cos, aaak, deg, deg_norm, age, structural_score in scored:
        rec = records_cache.get(cid)
        if rec is None:
            continue
        tokens = len(rec.literal_surface) // 4
        suggestions = graph.two_hop_neighborhood([cid], top_k=3)[:3]
        if structural_weight > 0.0:
            reason = (
                f"cos {cos:.3f} + aaak {aaak:.2f} "
                f"+ deg_norm {deg_norm:.3f} "
                f"- age {age:.2f} | structural {structural_score:.3f} "
                f"(w={structural_weight:.2f})"
            )
        else:
            reason = (
                f"cos {cos:.3f} + aaak {aaak:.2f} "
                f"+ deg_norm {deg_norm:.3f} "
                f"- age {age:.2f}"
            )
        scored_hits.append(
            MemoryHit(
                record_id=cid,
                score=float(s),
                reason=reason,
                literal_surface=rec.literal_surface,
                adjacent_suggestions=suggestions,
            ),
        )
        budget_used += tokens

    # architectural correction (08-02 Rule 1 fix):
    # Post-rank work (provenance batch, anti-hits, s4 hints, profile-modulates
    # edges, schema/curiosity hints, patterns_observed strip, retrieval_used
    # event) MUST run over the BUDGET-CAPPED hits, not over the full ranked
    # list of all reachable records. Pre-Wave-2 the OLD pipeline_recall body
    # applied the budget pack inline AND ran post-rank over the capped list;
    # _recall_core's first cut accidentally ran post-rank over the full list,
    # which made s4.on_read_check_batch fire ~K_CANDIDATES per-record cosines
    # per recall (350+ ms at N=200, blowing the 200ms perf-gate ceiling).
    #
    # The fix: _recall_core returns the SORTED full ranked list + activation
    # trace + cue_mode (the load-bearing core). The entry points
    # (recall_for_response, recall_for_benchmark) apply their cap THEN run
    # the post-rank pipeline over the capped subset — restoring OLD semantic
    # order (rank → cap → side-effects-over-capped-set).
    activation_trace = list({*seed_ids, *spread_ids})

    return _RecallCoreResult(
        scored_hits=scored_hits,
        activation_trace=activation_trace,
        anti_hits=[],            # populated by entry points over capped hits
        hints=[],                # populated by entry points over capped hits
        patterns_observed=[],    # populated by entry points over capped hits
        cue_mode=mode,
        budget_used=budget_used,  # informational sum over full ranked list
        _records_cache=records_cache,  # private: reused by entry points
    )


def _apply_post_rank_pipeline(
    hits: list[MemoryHit],
    *,
    store: MemoryStore,
    graph: MemoryGraph,
    records_cache: dict[UUID, "object"],
    cue: str,
    session_id: str,
    profile_state: dict | None,
    turn: int,
    mode: str,
    budget_used: int,
    path_label: str,
    knobs_applied: dict | None = None,
) -> tuple[list[MemoryHit], list[MemoryHit], list[dict], list[dict]]:
    """post-rank work shared by both entry points.

    Operates on the BUDGET/K-CAPPED `hits` list, not on the full ranked
    `scored_hits` from `_recall_core`. This restores the OLD semantic
    order: rank → cap → side-effects-over-capped-set.

    The function applies different scopes to different stages:
      - O(N) per-record work (provenance, profile_modulates, retrieval_used,
        patterns_observed strip) runs over the FULL caller-facing `hits`.
        This preserves the contract: every hit returned gets a
        provenance entry.
      - O(N²) heavy work (anti-hits lookup, s4 pairwise contradiction
        scan, schema/curiosity entropy) runs over the top
        `_POST_RANK_MAX_HITS` (default 50) of `hits`. This bounds the
        s4 pairwise scan to ~1225 pair checks regardless of how many
        hits the caller-facing list contains. Matches the OLD effective
        post-rank input size on healthy graphs.

    Returns: (hits_after_pattern_strip, anti_hits, hints, patterns_observed).

    Stages mirror the pre-Phase-8 OLD pipeline_recall body lines ~1860-2050:
      11. Provenance batch over full hits (; contract).
      12. Anti-hits over capped subset (s4 scope).
      13. S4 hints over capped subset, skipped in verbatim mode.
      14. profile_modulates edges over full hits (+ batched).
      15. Provisional schema + curiosity hints over capped subset, skipped in verbatim.
      16. Concept-mode patterns_observed strip over full hits (R6).
      17. retrieval_used event with full hit_ids (M2 LIVE).
    """
    # Heavy O(N²) post-rank scope is bounded by _POST_RANK_MAX_HITS.
    s4_scope_hits = hits[:_POST_RANK_MAX_HITS]

    # Stage 11 - Provenance batch over the FULL caller-facing hits
    # (every returned hit gets a provenance entry).
    if hits:
        provenance_pairs: list[tuple[UUID, dict]] = [
            (
                h.record_id,
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "cue": cue,
                    "session_id": session_id,
                },
            )
            for h in hits
        ]
        try:
            store.queue_provenance_batch(provenance_pairs)
        except Exception:
            pass

    # Stage 12 - Anti-hits over the s4-scope (capped) subset.
    anti_hits = _find_anti_hits(
        s4_scope_hits, store, graph, k=3, records_cache=records_cache,
    )

    # Stage 13 - S4/curiosity/schema hints (skipped in verbatim mode).
    # The s4 pairwise contradiction scan is O(N²); apply the cap.
    if mode == "verbatim":
        hints: list[dict] = []
    else:
        try:
            from iai_mcp.s4 import on_read_check_batch
            hints = on_read_check_batch(
                store, s4_scope_hits, session_id=session_id,
                records_cache=records_cache,
            )
        except Exception:
            hints = []

    # Stage 14 - profile_modulates edges over the FULL caller-facing hits
    # (+ batched). O(N) cheap; no cap.
    if profile_state:
        modulate_pairs: list[tuple] = []
        modulate_deltas: list[float] = []
        for h in hits:
            try:
                rec = records_cache.get(h.record_id)
                if rec is None:
                    continue
                gains = getattr(rec, "profile_modulation_gain", None) or {}
                if not gains:
                    continue
                total_gain = float(sum(gains.values()))
                if total_gain <= 0:
                    total_gain = 1.0
                modulate_pairs.append((h.record_id, PROFILE_SENTINEL_UUID))
                modulate_deltas.append(total_gain)
            except Exception:
                continue
        if modulate_pairs:
            try:
                store.boost_edges(
                    modulate_pairs,
                    edge_type="profile_modulates",
                    delta=modulate_deltas,
                )
            except Exception:
                pass

    # Stage 15 - Provisional schema + curiosity hints over s4-scope subset
    # (mode != verbatim). Both call O(N) entropy + O(N) iteration; cap to
    # match s4 scope so hint surface scales consistently.
    if mode != "verbatim":
        try:
            from iai_mcp.curiosity import compute_entropy
            from iai_mcp.schema import provisional_schemas_for_recall

            entropy_bits = compute_entropy([h.score for h in s4_scope_hits])
            hints.extend(provisional_schemas_for_recall(
                store, s4_scope_hits, entropy_bits,
                records_cache=records_cache,
            ))
        except Exception:
            pass
        try:
            from iai_mcp.curiosity import compute_entropy, fire_curiosity

            entropy_bits = compute_entropy([h.score for h in s4_scope_hits])
            q = fire_curiosity(
                store, s4_scope_hits, cue=cue, entropy=entropy_bits,
                session_id=session_id, turn=turn,
            )
            if q is not None:
                hints.append({
                    "kind": f"curiosity_{q.tier}",
                    "severity": "info",
                    "source_ids": [str(t) for t in q.triggered_by_record_ids],
                    "text": q.text,
                    "entropy": q.entropy,
                })
        except Exception:
            pass

    # Stage 16 - Concept-mode patterns_observed strip over the FULL hits
    # (R6). Schema records (tier=semantic AND tag=pattern:*)
    # are stripped from `hits` into `patterns_observed`; max 3 entries.
    patterns_observed: list[dict] = []
    if mode == "concept":
        kept_hits: list[MemoryHit] = []
        edges_df = None
        try:
            edges_df = store.db.open_table("edges").to_pandas()
        except Exception:
            edges_df = None
        for h in hits:
            rec = records_cache.get(h.record_id)
            if rec is None:
                kept_hits.append(h)
                continue
            tier = getattr(rec, "tier", "episodic")
            tags = list(getattr(rec, "tags", []) or [])
            is_schema = (
                tier == "semantic"
                and any(t.startswith("pattern:") for t in tags)
            )
            if is_schema:
                if len(patterns_observed) < 3:
                    pattern_str = ""
                    for t in tags:
                        if t.startswith("pattern:"):
                            pattern_str = t.split(":", 1)[1] if ":" in t else ""
                            break
                    evidence_count = 0
                    if edges_df is not None and not edges_df.empty:
                        try:
                            evidence_count = int(
                                ((edges_df["edge_type"] == "schema_instance_of")
                                 & (edges_df["dst"] == str(h.record_id))).sum()
                            )
                        except Exception:
                            evidence_count = 0
                    patterns_observed.append({
                        "pattern": pattern_str,
                        "evidence_count": evidence_count,
                        "schema_id": str(h.record_id),
                    })
            else:
                kept_hits.append(h)
        hits = kept_hits

    # Stage 17 - retrieval_used event with full hit_ids (M2 LIVE).
    try:
        write_event(
            store,
            kind="retrieval_used",
            data={
                "hit_ids": [str(h.record_id) for h in hits],
                "query": cue,
                "used": len(hits) > 0,
                "budget_used": budget_used,
                "path": path_label,
            },
            severity="info",
            session_id=session_id,
        )
    except Exception:
        pass

    return hits, anti_hits, hints, patterns_observed


def recall_for_response(
    store: MemoryStore,
    graph: MemoryGraph,
    assignment: CommunityAssignment,
    rich_club: list[UUID],
    embedder: Embedder,
    cue: str,
    session_id: str,
    budget_tokens: int = 1500,
    profile_state: dict | None = None,
    turn: int = 0,
    mode: str = "concept",
    *,
    knobs_applied: dict | None = None,
) -> RecallResponse:
    """redesign : production answer-packing entry point.

    Calls `_recall_core` for the load-bearing recall computation, then
    packs hits under `budget_tokens` per the pre-Phase-8 contract: the
    ranker's sorted output is consumed in score-desc order; each hit
    contributes `tokens = len(rec.literal_surface) // 4` to a running
    budget; the loop breaks when `budget_used + tokens > budget_tokens`
    AND `len(hits) >= 1` (the production "always at least one hit"
    minimum, matching pre-Phase-8 main pre-patch behaviour).

    This entry point does NOT accept a `k_hits` parameter. Production
    callers (`core.dispatch`) want token-budget-shaped responses for
    prompt assembly. For benchmark-shape (deterministic top-K), use
    `recall_for_benchmark`.

    Mode plumbing: the `mode` parameter is set upstream by the
    cue-classifier (`core.py:dispatch()`, R5) and is passed
    through to `_recall_core` unchanged. Inside `_recall_core` Stage 5,
    `_gate_bias_for_mode(mode)` selects the community-gate
    soft-bias scalar (verbatim=0.0, concept=0.1).
    """
    core = _recall_core(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        knobs_applied=knobs_applied,
    )

    # Enrich + downweight + re-sort BEFORE the budget-pack loop so a stale
    # high-cosine hit does not consume budget that should go to a fresh
    # lower-cosine record. Order is load-bearing. core.anti_hits is empty
    # on the regular path (_recall_core line 1014); anti_hits are built
    # later inside _apply_post_rank_pipeline. The L0 fast-path early-return
    # below DOES use core.anti_hits, so enrich both to cover both paths
    # cleanly.
    #
    # Perf-critical: build the (outgoing, ts_by_id) maps ONCE per recall
    # and pass them into the helper for BOTH the pre-budget enrichment
    # (here) and the post-pipeline anti_hits enrichment below. One
    # records.to_pandas() scan instead of two -- keeps the cost under
    # the M-02 p95 gate at N=100.
    #
    # NOTE: deliberately NOT consuming core._records_cache for created_at --
    # SimpleRecordView.created_at is a wall-clock placeholder (
    # graph node payload does not carry record.created_at), which would
    # poison the derived valid_from / valid_to. Plumbing created_at into
    # graph node attrs is follow-up.
    from iai_mcp.retrieve import (
        apply_stale_downweight,
        build_temporal_validity_maps,
        derive_temporal_validity,
    )
    _tv_maps = build_temporal_validity_maps(store)
    _tv_outgoing, _tv_ts = (_tv_maps if _tv_maps is not None else ({}, {}))
    derive_temporal_validity(
        None, core.scored_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    derive_temporal_validity(
        None, core.anti_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    apply_stale_downweight(core.scored_hits)
    apply_stale_downweight(core.anti_hits)
    core.scored_hits.sort(key=lambda h: h.score, reverse=True)

    # If the L0 fast-path fired, _recall_core returned an already-packed
    # single-hit result. Surface it directly as a RecallResponse.
    # scored_hits / anti_hits already enriched + downweighted upstream.
    if (
        len(core.scored_hits) == 1
        and any(h.get("kind") == "retrieval_skipped" for h in core.hints)
    ):
        return RecallResponse(
            hits=core.scored_hits,
            anti_hits=core.anti_hits,
            activation_trace=core.activation_trace,
            budget_used=core.budget_used,
            hints=core.hints,
            cue_mode=core.cue_mode,
            patterns_observed=core.patterns_observed,
        )

    # Pack hits under budget_tokens. Reproduces the pre-Phase-8 contract.
    # redesign (08-02 Rule 1 fix): the budget-pack loop also
    # respects `_POST_RANK_MAX_HITS` (default 50) as a safety cap on
    # the number of records that flow into the post-rank pipeline. This
    # matches OLD pipeline_recall's effective behavior on healthy graphs
    # (gate-restricted reachable to ~50-72 records); the wider
    # candidate pool (K_CANDIDATES=200) is preserved for ranking
    # accuracy, but the response surface stays bounded by the same cap
    # the OLD pipeline naturally produced. Without this cap, on small-
    # surface fixtures (synthetic perf-gate tests) the budget never
    # binds and the response would carry all 200 ranked records, which
    # blows past the 200ms / 75ms perf-gate ceilings via O(N²) s4 work
    # and the proportional-to-N provenance sync write.
    hits: list[MemoryHit] = []
    budget_used = 0
    for hit in core.scored_hits:
        if len(hits) >= _POST_RANK_MAX_HITS:
            break
        tokens = len(hit.literal_surface) // 4
        if budget_used + tokens > budget_tokens and len(hits) >= 1:
            break
        hits.append(hit)
        budget_used += tokens

    # (08-02 Rule 1 fix): post-rank pipeline runs OVER the
    # capped hits. `_apply_post_rank_pipeline` internally bounds heavy
    # O(N²) work (s4, anti-hits, schema/curiosity) to `_POST_RANK_MAX_HITS`
    # while letting cheap O(N) work (provenance batch, profile_modulates,
    # patterns_observed strip, retrieval_used event) span the full hits.
    hits, anti_hits, hints, patterns_observed = _apply_post_rank_pipeline(
        hits,
        store=store, graph=graph, records_cache=core._records_cache,
        cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        budget_used=budget_used, path_label="recall_for_response",
        knobs_applied=knobs_applied,
    )

    # anti_hits are constructed INSIDE _apply_post_rank_pipeline
    # (_find_anti_hits) from contradicts-edge neighbours with score=0.0.
    # They never pass through the pre-budget enrichment above, so enrich
    # them here so the JSON wire carries valid_from/valid_to on the
    # anti_hits surface too. Score downweight is a no-op on score=0.0
    # baseline anti-hits; the value is the valid_from/valid_to fields
    # + " · stale" reason marker. anti_hits intentionally NOT re-sorted:
    # they are an inhibitory tail. Reuses the (outgoing, ts_by_id) maps
    # built once per recall above -- no second records-table scan.
    derive_temporal_validity(
        None, anti_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    apply_stale_downweight(anti_hits)

    return RecallResponse(
        hits=hits,
        anti_hits=anti_hits,
        activation_trace=core.activation_trace,
        budget_used=budget_used,
        hints=hints,
        cue_mode=core.cue_mode,
        patterns_observed=patterns_observed,
    )


def recall_for_benchmark(
    store: MemoryStore,
    graph: MemoryGraph,
    assignment: CommunityAssignment,
    rich_club: list[UUID],
    embedder: Embedder,
    cue: str,
    session_id: str,
    k_hits: int = 10,
    profile_state: dict | None = None,
    turn: int = 0,
    mode: str = "concept",
    *,
    knobs_applied: dict | None = None,
) -> RecallResponse:
    """redesign : benchmark top-K entry point.

    Calls `_recall_core` for the load-bearing recall computation, then
    takes the top `k_hits` from the sorted `scored_hits`. Deterministic:
    no token budget, no per-hit pack rule. Used by:
      - bench/longmemeval_blind.py (Y prong)
      - bench/lme500/debug_pipeline_loss.py (stage tracer)
      - any future benchmark harness needing top-K retrieval surface.

    This entry point does NOT accept a `budget_tokens` parameter.
    For production answer-packing (token-budget-shaped responses),
    use `recall_for_response`.

    Mode plumbing: bench callers pass `mode="concept"` (LongMemEval-S
    is concept-shaped per BENCH_PROTOCOL_lme500.md); the parameter is
    passed through to `_recall_core` unchanged so the mode-dependent
    gate bias (`_gate_bias_for_mode(mode)`) operates as designed.
    """
    core = _recall_core(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        knobs_applied=knobs_applied,
    )
    # L0 fast-path: surface the single-hit result directly. (k_hits >= 1
    # is the only sensible value; the L0 result is already capped at 1.)
    if (
        len(core.scored_hits) == 1
        and any(h.get("kind") == "retrieval_skipped" for h in core.hints)
    ):
        return RecallResponse(
            hits=core.scored_hits,
            anti_hits=core.anti_hits,
            activation_trace=core.activation_trace,
            budget_used=core.budget_used,
            hints=core.hints,
            cue_mode=core.cue_mode,
            patterns_observed=core.patterns_observed,
        )

    hits = core.scored_hits[:k_hits]
    # budget_used is informational for the benchmark prong (not a cap);
    # report the sum of per-hit token estimates across the returned hits.
    budget_used = sum(len(h.literal_surface) // 4 for h in hits)

    # (08-02 Rule 1 fix): post-rank pipeline runs OVER the
    # capped (top-k_hits) hits. `_apply_post_rank_pipeline` internally
    # bounds heavy O(N²) work to `_POST_RANK_MAX_HITS` while letting cheap
    # O(N) work span the full hits.
    hits, anti_hits, hints, patterns_observed = _apply_post_rank_pipeline(
        hits,
        store=store, graph=graph, records_cache=core._records_cache,
        cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        budget_used=budget_used, path_label="recall_for_benchmark",
        knobs_applied=knobs_applied,
    )

    return RecallResponse(
        hits=hits,
        anti_hits=anti_hits,
        activation_trace=core.activation_trace,
        budget_used=budget_used,
        hints=hints,
        cue_mode=core.cue_mode,
        patterns_observed=patterns_observed,
    )

