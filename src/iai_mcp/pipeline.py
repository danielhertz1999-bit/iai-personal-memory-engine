"""Five-stage retrieval pipeline.

Stage 1 - Embed: bge-small(cue) -> 384d vector.
Stage 2 - Community gate: argmax cosine over centroids, keep top 3
         (primary + 2 neighbours via inter-community tunnel scores).
Stage 3 - Seeds: top-3 within gated communities by 0.6*cos + 0.4*centrality.
Stage 4 - 2-hop greedy spread, union with pre-fetched rich-club.
Stage 5 - Rank + pack under budget:
            score = W_COSINE*cos + W_AAAK*aaak_overlap + W_DEGREE*deg_norm
                    - W_AGE*age_penalty
            where deg_norm = log(1+deg) / log(1+max_deg) is bounded in [0,1]
            so the degree contribution is sample-rank-comparable to cosine
            (max_deg cached on graph._max_degree by build_runtime_graph).
            Multiplied by profile_modulation gain product if profile_state
            carries active knobs.
          Anti-hits from contradicts-edge neighbours of top hits (dual-route).

Rules enforced:
- Every hit appends a provenance entry (same as baseline retrieve.recall).
- literal_surface returned verbatim (never rewritten) from store.
- adjacent_suggestions populated per hit (cued recognition).

Profile modulation additions:
- profile_modulates edges: after ranking, active knob gains create
  profile_modulates edges from affected records -> PROFILE_SENTINEL_UUID.
- Curiosity hints: entropy-gated clarifying questions surfaced via
  RecallResponse.hints.
- Provisional schema hints: high-entropy recalls surface candidate schemas.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import log
from uuid import UUID

import numpy as np

from iai_mcp.community import CommunityAssignment
from iai_mcp.embed import Embedder
from iai_mcp.events import TELEMETRY_EMBED_NATIVE_FAILURE, write_event
from iai_mcp.exceptions import (
    RetrievalError, EmbeddingError, CommunityGateError, BudgetExceededError, StoreError,
    NativeError,
)
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryHit, RecallResponse

# Structured-log channel. Named ``logger`` (not ``log``) to avoid shadowing
# the ``math.log`` import used in the rank stage's degree normalisation.
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ helpers


@dataclass
class SimpleRecordView:
    """Lightweight record view sourced from graph node attrs.

    Covers the fields the seed + spread + rank stages actually read
    (embedding for cosine, literal_surface for MemoryHit hydration,
    centrality + tier for tie-break signals). Fields the scoring loops
    don't touch at the seed/spread stage are filled with safe defaults
    so the view can stand in for a MemoryRecord without crashing the
    rarer code paths (aaak_overlap, age_penalty) that hit rank stage.

    This is NOT a MemoryRecord replacement; it's a read-only payload
    carrier for the hot-path that never needs to round-trip to the store.
    Writes always go through store.insert / store.update / store.delete.
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
    # Fields touched by profile_modulation_for_record and rank-adjacent
    # paths. Safe defaults keep the rank stage shape-compatible with the
    # full MemoryRecord surface.
    tags: list = field(default_factory=list)
    language: str = "en"


def _read_record_payload(graph, rid: UUID, store: MemoryStore):
    """Graph-first record payload access.

    Reads from the MemoryGraph sidecar via ``graph.get_payload``. If the
    sidecar is missing the ``embedding`` field (race / partial-sync with the
    store), falls back to ``store.get(rid)`` so the recall path never
    crashes — just takes a small latency hit on that one node.

    ``graph`` accepts either a ``MemoryGraph`` instance (preferred) or a
    legacy NetworkX-shaped object for back-compat with older test fixtures
    that pass a raw networkx graph directly. Both paths funnel through the
    same fall-back semantics.

    Returns either a SimpleRecordView (graph-resident, no disk I/O)
    or a MemoryRecord (store fallback), or None if the id is truly
    unknown to both the graph and the store.
    """
    if rid is None:
        node = None
    elif hasattr(graph, "get_payload"):
        # MemoryGraph fast path — sidecar dict via public API.
        node = graph.get_payload(rid) or None
    else:
        # Legacy NetworkX-shaped object (test fixtures may still pass _nx).
        node = graph.nodes.get(str(rid)) if hasattr(graph, "nodes") else None
        node = dict(node) if node else None
    if node is not None and "embedding" in node and "surface" in node:
        # Empty/None surface OR a ``_decrypt_failed=True`` flag is a sentinel
        # for cache-miss-due-to-decrypt-failure. Fall through to store.get
        # which has its own retry semantics in crypto.py. A legitimately-
        # empty record round-trips correctly because store.get returns the
        # same empty literal_surface; the rare legitimate-empty case remains
        # correct because both paths produce the same output.
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
    except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
        logger.debug("read_record_payload_store_fallback_failed rid=%s: %s", rid, exc)
        return None

# Rank-stage score formula constants
W_COSINE = 1.0
W_AAAK = 0.3
W_DEGREE = 0.1
W_AGE = 0.05

# Age penalty "half-life": 30 days brings the penalty to 1.0 (fully saturated).
AGE_HALF_LIFE_DAYS = 30.0

# literal_preservation knob modulates the effective W_DEGREE in the rank-stage
# scoring formula. Keys MUST match the profile.py KnobSpec enum schema
# "enum:strong|medium|loose".
#
# Numeric mapping:
#   strong  = 0.3   tighten degree influence; verbatim wins
#   medium  = 1.0   normalize-only baseline; no extra knob effect
#   loose   = 1.5   let hubs speak louder; concept-mode-friendly
#
# Default fallback when profile_state is missing/empty/invalid is "medium"
# (scale 1.0) so callers without a knob set see the baseline behaviour.
LITERAL_PRESERVATION_W_DEGREE_SCALE: dict[str, float] = {
    "strong": 0.3,
    "medium": 1.0,
    "loose":  1.5,
}

# Candidate-pool size for the cosine top-K gate. K=200 is the empirical
# 99th-percentile gold rank from the LongMemEval-S v1 trace (worst-case qid
# had 12/12 gold inside cosine rank 1-200) plus 30% margin.
K_CANDIDATES: int = 200

# Mode-dependent community-gate soft-bias scalars.
#
# The community gate (Leiden communities + centroid cosine) is a categorical
# structure. Episodic recall (verbatim mode) is sparse and should NOT be
# weighted by categorical community membership; semantic recall (concept mode)
# benefits from a soft categorical hint:
#
#   verbatim mode -> 0.0  (categorical filtering degrades literal precision)
#   concept mode  -> 0.1  (soft +10% bonus to records in top-3 gated
#                          communities; a categorical hint without filtering)
#
# The bias is NEVER a hard filter; the candidate pool is always cosine
# top-K_CANDIDATES regardless of mode. `_gate_bias_for_mode(mode)` returns
# the appropriate scalar.
COMMUNITY_BIAS_VERBATIM: float = 0.0
COMMUNITY_BIAS_CONCEPT: float = 0.1

# Internal post-rank cap.
#
# K_CANDIDATES=200 widens what reaches Stage 5 ranking. Post-rank work
# (contradiction-detection pairwise scan, anti-hits lookup, profile_modulates
# edge writes, schema/curiosity hints) is O(N²) or O(N); running it over the
# full 200 candidates would breach perf-gate ceilings. Cap at 50 so s4's
# pairwise scan stays at 50*49/2 ≈ 1225 checks (vs ~20k at 200).
#
# The cap applies to side-effect computations only (hints, anti-hits,
# profile_modulates edges, schema, curiosity, retrieval_used event); the
# public `hits` list still respects the caller's budget contract.
_POST_RANK_MAX_HITS: int = 50


# Stage 8 historical-verbatim downweight. When the cue intent signals the
# user wants the "original" / pre-correction record, records that are the
# TARGET (dst) of a `contradicts` edge get this value subtracted from their
# final score so the contradicted (original) record ranks above its corrector.
#
# Direction note: per store.add_contradicts_edge(original, new) the edge is
# written as src=original, dst=new. The corrector is the dst. We build a
# contradicts-dst set from the outgoing dict's values and downweight cids
# that appear in that set.
#
# Default 0.25: the wrong record outscores gold by ~0.18 on benchmark probes;
# subtracting 0.25 flips the gap to +0.07 in favor of gold. Env override
# allows ops tuning without code change.
import os as _os_phase24  # noqa: E402 -- local alias to avoid os import collision
HISTORICAL_VERBATIM_DOWNWEIGHT: float = float(
    _os_phase24.environ.get("IAI_MCP_HISTORICAL_VERBATIM_DOWNWEIGHT", "0.25"),
)


def _build_contradicts_dst_set(
    contradicts_outgoing: dict[str, list[str]] | None,
) -> set[str]:
    """Build the set of record ids that are the TARGET (dst) of any contradicts edge.

    `contradicts_outgoing` is keyed by src (the original record id) with
    values = list of dst ids (the correctors). The DST set is the union
    of all values — this is the set the Stage 8 historical-verbatim
    downweight applies to.

    Returns an empty set when contradicts_outgoing is None or empty.
    """
    if not contradicts_outgoing:
        return set()
    dst_set: set[str] = set()
    for dsts in contradicts_outgoing.values():
        if dsts:
            dst_set.update(str(d) for d in dsts)
    return dst_set


def _gate_bias_for_mode(mode: str) -> float:
    """Mode-dependent community-gate soft-bias scalar.

    verbatim mode -> 0.0  (literal precision; categorical bias degrades recall)
    concept  mode -> 0.1  (soft categorical hint without hard filtering)

    Any unknown mode defaults to verbatim's 0.0 (conservative -- never
    accidentally bias toward categorical filtering on ambiguous mode).
    """
    return COMMUNITY_BIAS_CONCEPT if mode == "concept" else COMMUNITY_BIAS_VERBATIM


@dataclass
class _RecallCoreResult:
    """Shape returned by `_recall_core`.

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

    `scored_hits` is sorted by score descending (deterministic tie-break
    by UUID-asc as secondary key).

    `_records_cache` is a private field carrying the records_cache
    `_recall_core` built (graph-resident SimpleRecordViews + store
    fallback). Entry points reuse it for post-rank work to avoid
    duplicating the O(N) graph walk + store.all_records() scan.
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


# Deterministic sentinel UUID -- target of every profile_modulates edge.
# Individual gain breakdowns live on the record's profile_modulation_gain dict.
PROFILE_SENTINEL_UUID = UUID("00000000-0000-0000-0000-0000000000f1")


# --------------------------------------------------------------- math helpers


def _trigram_jaccard(a: str, b: str) -> float:
    """Trigram-set Jaccard similarity between two strings.

    Returns intersection/union of character trigram sets, 0.0 on empty.
    """
    if len(a) < 3 or len(b) < 3:
        return 0.0
    set_a = {a[i:i + 3] for i in range(len(a) - 2)}
    set_b = {b[i:i + 3] for i in range(len(b) - 2)}
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


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

    Whitespace + slash split applied symmetrically to both cue_text and
    aaak_index so "auth/login" tokenises consistently on either side.
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
    member_embeddings: dict[UUID, list[float]] | None = None,
) -> list[UUID]:
    """CONN-06: route cue to top-N communities.

    Two scoring modes (selected by `member_embeddings`):

    1. **Max-node cosine (B*)** — preferred. When `member_embeddings` is
       provided, each community is scored by ``max(cos(cue, m)) for m in
       members``. Per Fortunato 2010 (*Phys Reports*,
       DOI:10.1016/j.physrep.2009.11.002) and Mucha 2010 (*Science*,
       DOI:10.1126/science.1184819), centroid-cosine collapses
       specificity in high-dim spaces ("semantic mush"); max-node is the
       published-robust gate that immunises the retrieval pipeline from
       partition fragmentation (Fortunato 2010, Mucha 2010).

    2. **Centroid cosine (legacy)** — fallback when `member_embeddings`
       is None. Preserves the original behaviour for unit-test callers
       that construct `CommunityAssignment` without a records_cache;
       degenerate 1-member-per-community geometry makes the two scoring
       modes identical, so existing diagnostic tests stay green.

    Both modes are vectorized — one matmul over stacked centroids (legacy)
    or one matmul over stacked member embeddings + per-community max-reduce
    (B*). At N=5000 members over 100 communities the B* path stays under ~1 ms.

    Deterministic tie-break: stable sort by (-score, UUID-str).
    """
    cue_vec = np.asarray(cue_emb, dtype=np.float32)
    cue_norm = float(np.linalg.norm(cue_vec))
    if cue_norm > 0.0:
        cue_vec = cue_vec / cue_norm

    if member_embeddings is not None:
        return _community_gate_max_node(
            cue_vec, assignment, top_n, member_embeddings,
        )

    centroids = assignment.community_centroids
    if not centroids:
        return []
    cids = list(centroids.keys())
    mat = np.asarray(
        [centroids[c] for c in cids], dtype=np.float32
    )
    # Centroids may not be unit-norm (community.py averages member
    # embeddings then re-normalizes; we still normalize defensively so
    # this stays true-cosine even if a caller passes raw centroids).
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0.0] = 1.0
    mat = mat / norms[:, None]
    scores = mat @ cue_vec  # shape (K,)
    order = np.argsort(-scores, kind="stable")
    return [cids[int(i)] for i in order[:top_n]]


def _community_gate_max_node(
    cue_vec: np.ndarray,
    assignment: CommunityAssignment,
    top_n: int,
    member_embeddings: dict[UUID, list[float] | np.ndarray],
) -> list[UUID]:
    """B* max-node cosine helper for `_community_gate`.

    Stacks all member embeddings of all communities into one matrix,
    runs a single matmul against the cue, and reduces per-community
    via `np.maximum.reduceat` over community-boundary indices.

    Communities with no members in `member_embeddings` (e.g. sentinel
    UUIDs filtered out before the call) are skipped; an empty
    `mid_regions` falls back to the centroid path so the gate is never
    silently empty when centroids exist.

    Perf note: when the dict values are already `np.ndarray`, member
    stacking is a fast `np.stack` (sub-millisecond at N=5000). When
    values are Python lists (unit-test path), `np.asarray` triggers a
    per-float Python -> C cast that is ~30x slower; the production
    caller (`_recall_core`) converts to ndarray once at the records_cache
    build so the hot path stays vectorized.

    Determinism: stable lexical sort on (-score, str(UUID)).
    """
    mid_regions = assignment.mid_regions
    if not mid_regions:
        # No member->community mapping; can't run max-node. Degrade to
        # centroid path (legacy behaviour preserves the public contract:
        # _community_gate(cue, assignment, top_n) returns something
        # non-empty when centroids exist).
        return _community_gate(
            cue_vec.tolist(), assignment, top_n, member_embeddings=None,
        )

    # Build the per-community member-vector stack. Communities whose
    # members are all absent from `member_embeddings` are dropped (their
    # max score is undefined; including them with -inf would distort the
    # tie-break with UUID-str).
    cids: list[UUID] = []
    rows: list[np.ndarray] = []
    breaks: list[int] = []
    total = 0
    for cid, members in mid_regions.items():
        valid: list[np.ndarray] = []
        for m in members:
            emb = member_embeddings.get(m)
            if emb is None:
                continue
            # Cast list->ndarray here so np.stack below is fast even
            # when the caller passed Python lists.
            if not isinstance(emb, np.ndarray):
                emb = np.asarray(emb, dtype=np.float32)
            valid.append(emb)
        if not valid:
            continue
        cids.append(cid)
        breaks.append(total)
        total += len(valid)
        rows.extend(valid)

    if not rows:
        return []

    # np.stack over a list of ndarrays is fast (~0.2 ms at N=5000);
    # np.asarray on list-of-list is 30x slower.
    mat = np.stack(rows).astype(np.float32, copy=False)
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0.0] = 1.0
    mat = mat / norms[:, None]
    member_scores = mat @ cue_vec  # shape (total_members,)

    # Per-community max via reduceat: `breaks` are the start indices of
    # each community in the stacked matrix; reduceat over them yields
    # one max per community.
    comm_max = np.maximum.reduceat(member_scores, breaks)

    # Stable lexical tie-break by UUID string within tied score buckets.
    # numpy argsort with `kind="stable"` preserves input order, so we
    # pre-sort `cids` by str(uuid) and then stable-sort by -score.
    str_order = sorted(range(len(cids)), key=lambda i: str(cids[i]))
    lex_sorted_cids = [cids[i] for i in str_order]
    lex_sorted_scores = comm_max[str_order]
    score_order = np.argsort(-lex_sorted_scores, kind="stable")
    return [lex_sorted_cids[int(i)] for i in score_order[:top_n]]


def _pick_seeds(
    candidate_indices: np.ndarray,
    shared_cos: np.ndarray,
    centrality_arr: np.ndarray,
    n: int = 3,
) -> np.ndarray:
    """Seed selection over the shared cosine array.

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
    """Build the (ids, embeddings) pool over which the shared cosine pass operates.

    Reads embeddings in this order of preference:
      1. graph sidecar payload "embedding" (zero-IO; populated by
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
    for rid in graph.iter_nodes():
        emb: list[float] | None = None
        # Path 1: sidecar (cheapest, populated by build_runtime_graph).
        node_emb = graph.get_embedding(rid)
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
            except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
                logger.debug("collect_graph_pool_store_fallback_failed rid=%s: %s", rid, exc)
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


def _log_malformed_anti_edges(store: MemoryStore, hit_ids: "list[UUID]") -> None:
    """Scan contradicts edges for the given hit ids and log any with malformed
    src/dst values (non-UUID strings). Observability contract: operators learn
    which rows are corrupt so they can be repaired; incident_edges silently
    skips them.  Runs before the incident_edges call so the log is emitted
    from the iai_mcp.pipeline logger.
    """
    try:
        str_ids = [str(i) for i in hit_ids]
        ph = ", ".join("?" for _ in str_ids)
        sql = (  # nosemgrep: sql-injection
            f"SELECT src, dst FROM edges"  # noqa: S608
            f" WHERE (src IN ({ph}) OR dst IN ({ph}))"
            f" AND edge_type = 'contradicts'"
        )
        params: list = str_ids + str_ids
        with store.db._conn_lock:
            rows = store.db._conn.execute(sql, params).fetchall()
        for row in rows:
            src_s = str(row[0])
            dst_s = str(row[1])
            for val, label in ((src_s, "src"), (dst_s, "dst")):
                try:
                    UUID(val)
                except (ValueError, AttributeError):
                    logger.warning(
                        "anti_hits_skip_malformed_edge %s=%s",
                        label, val,
                    )
    except Exception:  # noqa: BLE001 -- observability is best-effort
        pass


def _find_anti_hits(
    hits: list[MemoryHit],
    store: MemoryStore,
    graph: MemoryGraph,
    k: int = 3,
    records_cache: dict[UUID, "object"] | None = None,
) -> list[MemoryHit]:
    """Anti-hits: contradicts-edge neighbours of top hits (dual-route).

    records_cache (optional): used to hydrate MemoryHit.literal_surface
    without calling store.get per anti-id. Missing ids fall back to store.get.
    """
    seen: set[UUID] = {h.record_id for h in hits}
    anti_ids: list[UUID] = []

    # Reroute: use incident_edges(hit_ids, edge_types=["contradicts"], top_k=None)
    # over the returned hit ids (UNCAPPED) instead of materialising the full
    # edges table.  This removes the O(N_edges) full-table scan from the
    # warm recall path.
    hit_ids = [h.record_id for h in hits]
    if not hit_ids:
        return []

    # Pre-scan: log malformed edges BEFORE calling incident_edges so the
    # structured-log observability contract is met at the pipeline layer
    # (incident_edges silently drops malformed rows; callers deserve a
    # WARNING so operators can repair the corrupted rows).
    _log_malformed_anti_edges(store, hit_ids)

    try:
        _contr_map = store.incident_edges(
            hit_ids, edge_types=["contradicts"], top_k=None,
        )
    except Exception as exc:  # noqa: BLE001 -- anti-hits is enrichment; degrade to []
        logger.debug("_find_anti_hits incident_edges failed: %s", exc)
        return []

    for h in hits:
        for (_nbr, _et, _wt) in _contr_map.get(h.record_id, []):
            if _nbr in seen:
                continue
            anti_ids.append(_nbr)
            seen.add(_nbr)
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
        _prov = (rec.provenance or [{}])[0]
        out.append(
            MemoryHit(
                record_id=aid,
                score=0.0,
                reason="contradicts-edge neighbour",
                literal_surface=rec.literal_surface,
                adjacent_suggestions=[],
                session_id=_prov.get("session_id"),
                captured_at=rec.created_at.isoformat() if rec.created_at else None,
            )
        )
    return out


# ------------------------------------------------------------------ top-level


# Last recall wall-clock latency (ms). If >2000ms, next recall reduces
# community gate to 1 and disables spread.
_last_recall_latency_ms: float = 0.0


# OPT-IN debug capture used by the verbatim-filter-placement test.
# When non-None, `_recall_core` stashes its pre-filter and post-filter
# `reachable_ids` into the dict so tests can verify filter placement
# between Stage 4 (union) and Stage 5 (rank). Set to None at module
# import; tests monkeypatch a fresh dict for the duration of one call.
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
    k_communities: int = 3,
    spread_hops: int = 2,
    # Optional cue intent + pre-computed contradicts_outgoing dict for the
    # Stage 8 historical-verbatim downweight. Both default to None for
    # back-compat with existing test callers; entry points populate them
    # via _classify_cue + the build_temporal_validity_maps helper.
    cue_intent: str | None = None,
    contradicts_outgoing: dict[str, list[str]] | None = None,
) -> _RecallCoreResult:
    """Shared-cosine + Stage 2-5 + post-rank work.

    Performs the load-bearing recall computation ONCE and returns a
    fully-populated `_RecallCoreResult`. Both `recall_for_response` and
    `recall_for_benchmark` call this with identical arguments (minus the
    budget_tokens / k_hits cap, applied AFTER the core returns). The L0
    retrieval-skip fast path is implemented INSIDE so both prongs share it.

    Stage walk:
      0. Active-inference gate -> L0 fast path on hit.
      1. Embed cue.
      2. Build records_cache from graph node attrs (zero-IO when
         build_runtime_graph populated the graph).
      3. SHARED COSINE PASS: one matmul over the full pool.
      4. Community gate diagnostic: top-3 communities by centroid cosine;
         output feeds Stage-5 mode-dependent additive bias only (no hard-filter).
      5. Seed selection: blended 0.6*shared_cos + 0.4*centrality; pick top-3.
      6. Reachable union: cosine_top_indices ∪ 2-hop ∪ rich-club.
      7. Verbatim-mode filter: on `reachable_indices` between Stage 4
         union and Stage 5 rank.
      8. Stage-5 rank (cosine reuse, mode-dependent community bias).
      9. Sort scored desc by score, secondary by UUID-asc.
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

    # Stage 0 - Active-inference gate.
    # Lazy import + fn alias keeps this body free of substring
    # patterns the global security-reminder hook flags as eval-like.
    try:
        from iai_mcp import gate as _gate_mod
        _skip_fn = _gate_mod.should_skip_retrieval
        skip_flag, skip_reason = _skip_fn(cue)
    except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
        logger.debug("active_inference_gate_failed: %s", exc)
        skip_flag, skip_reason = False, ""
    if skip_flag:
        l0_uuid = UUID("00000000-0000-0000-0000-000000000001")
        l0_rec = store.get(l0_uuid)
        if l0_rec is not None:
            budget_used_l0 = len(l0_rec.literal_surface) // 4
            _l0_prov = (l0_rec.provenance or [{}])[0]
            l0_hit = MemoryHit(
                record_id=l0_rec.id,
                score=1.0,
                reason="L0 identity (skipped)",
                literal_surface=l0_rec.literal_surface,
                adjacent_suggestions=[],
                session_id=_l0_prov.get("session_id"),
                captured_at=l0_rec.created_at.isoformat() if l0_rec.created_at else None,
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
            except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
                logger.debug("l0_provenance_append_failed: %s", exc)
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
            except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
                logger.debug("l0_retrieval_used_event_failed: %s", exc)
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
    try:
        cue_emb = embedder.embed(cue)
    except Exception as exc:
        write_event(
            store,
            TELEMETRY_EMBED_NATIVE_FAILURE,
            {
                "op_type": "recall_cue",
                "backend": "rust",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise NativeError(f"recall cue encode failed: {exc}") from exc

    # Stage 2 - Build records_cache from graph node sidecar payloads.
    records_cache: dict[UUID, "object"] = {}
    try:
        for rid in graph.iter_nodes():
            node = graph.get_payload(rid)
            if "embedding" not in node or "surface" not in node:
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
    except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
        logger.debug("records_cache_graph_build_failed: %s", exc)
        records_cache = {}
    if not records_cache:
        records_cache = {r.id: r for r in store.all_records()}

    # In verbatim mode, restrict to tier='episodic'.
    # Build the set NOW; apply AFTER Stage-4 union.
    episodic_ids: set | None = None
    if mode == "verbatim":
        episodic_ids = {
            cid for cid, rec in records_cache.items()
            if getattr(rec, "tier", "episodic") == "episodic"
        }

    # Stage 3 - SHARED COSINE PASS. One matmul over the full pool.
    _pool_t0 = time.perf_counter()
    pool_ids, pool_embs = _collect_graph_pool(graph, records_cache, store)
    _recall_pool_collection_ms = (time.perf_counter() - _pool_t0) * 1000.0
    cue_vec = np.asarray(cue_emb, dtype=np.float32)
    cnorm = float(np.linalg.norm(cue_vec))
    if cnorm > 0.0:
        cue_vec = cue_vec / cnorm
    if pool_embs.size:
        # The single load-bearing matmul. Pool embeddings are
        # L2-normalized by the Rust embedder; dot == cosine.
        # Use np.matmul (not the @ operator) so the call is
        # interceptable via monkeypatch in the matmul-counter test.
        shared_cos = np.matmul(pool_embs, cue_vec).astype(np.float32)
    else:
        shared_cos = np.empty(0, dtype=np.float32)
    if shared_cos.size:
        shared_order = np.argsort(-shared_cos, kind="stable")
        cosine_top_indices = shared_order[:K_CANDIDATES]
    else:
        shared_order = np.empty(0, dtype=np.int64)
        cosine_top_indices = np.empty(0, dtype=np.int64)

    # Arousal-budget A/B routing inside _recall_core. MD5(cue) deterministic
    # 50/50 split; IAI_MCP_AROUSAL_USE_SHADOW=1 forces 100% shadow.
    #
    # On arousal_real: derive RetrievalParams from a fresh ArousalState()
    # and apply all 3 fields:
    #   - rank_threshold: filter graph-derived secondary candidates (spread +
    #     rich_club) by shared_cos >= threshold. cosine_top_indices stays
    #     untouched so primary cosine candidates cannot be over-pruned.
    #   - max_hops: narrow spread_hops downward (never widens).
    #   - mode: bias adjust additive to _gate_bias_for_mode.
    # On arousal_shadow: no filter, no override, no bias adjust (baseline).
    # On arousal_skip: import or compute failure -> baseline + tag in telemetry.
    _arousal_cue_hash_bytes = hashlib.md5(str(cue).encode("utf-8")).digest()
    _arousal_cue_hash_hex = _arousal_cue_hash_bytes[:4].hex()
    if os.environ.get("IAI_MCP_AROUSAL_USE_SHADOW") == "1":
        _arousal_route = "arousal_shadow"
    else:
        _arousal_route = "arousal_real" if (_arousal_cue_hash_bytes[0] & 1) else "arousal_shadow"

    _arousal_level_for_telemetry: float = 0.5
    _arousal_mode_for_telemetry: str | None = None
    _arousal_max_hops_used: int = spread_hops
    _arousal_rank_threshold_used: float = 0.0
    _arousal_mode_bias_adjust: float = 0.0
    _arousal_budget_for_telemetry: int = 1500

    if _arousal_route == "arousal_real":
        try:
            from iai_mcp.arousal_budget import (
                ArousalState as _ArousalState,
                compute_retrieval_params as _compute_retrieval_params,
            )
            _arousal_state_local = _ArousalState()
            _arousal_params = _compute_retrieval_params(_arousal_state_local)
            _arousal_level_for_telemetry = float(_arousal_state_local.level)
            _arousal_mode_for_telemetry = _arousal_params.mode
            _arousal_budget_for_telemetry = int(_arousal_params.budget_tokens)
            # rank_threshold gates the graph-derived secondary candidates
            # (spread + rich_club). cosine_top_indices are by construction the
            # highest-cosine records; re-filtering them would be redundant and
            # could over-prune small healthy fixtures.
            _arousal_rank_threshold_used = float(_arousal_params.rank_threshold)
            # max_hops override: only narrows, never widens.
            _arousal_max_hops_used = int(min(int(_arousal_params.max_hops), spread_hops))
            spread_hops = _arousal_max_hops_used
            # mode bias adjust additive to _gate_bias_for_mode.
            _amode = _arousal_params.mode
            if _amode == "monotropic_tunnel":
                _arousal_mode_bias_adjust = -0.05
            elif _amode == "associative_dream":
                _arousal_mode_bias_adjust = +0.05
            else:  # "balanced" or unknown -> no adjust
                _arousal_mode_bias_adjust = 0.0
        except Exception as exc:  # noqa: BLE001 -- arousal hot-path fail-safe
            logger.debug("arousal_budget_real_route_failed: %s", exc)
            _arousal_route = "arousal_skip"
            # Restore defaults explicitly so accidental writes downstream
            # cannot mistake skip for real.
            _arousal_rank_threshold_used = 0.0
            _arousal_max_hops_used = spread_hops
            _arousal_mode_bias_adjust = 0.0

    id_to_idx = {rid: i for i, rid in enumerate(pool_ids)}

    # Stage 4 - Community gate DIAGNOSTIC. Top-N communities; their members
    # form `gated_set` which feeds Stage 5's mode-bias.
    # Pass member_embeddings to route through the max-node-cosine (B*) path.
    # `pool_embs` rows are reused as ndarray values so the gate's `np.stack`
    # stays sub-millisecond.
    gate_member_embeddings: dict[UUID, np.ndarray] = {
        pool_ids[i]: pool_embs[i]
        for i in range(len(pool_ids))
    }
    gated = _community_gate(
        cue_emb, assignment, top_n=k_communities,
        member_embeddings=gate_member_embeddings,
    )
    gated_set: set[UUID] = set()
    for gc in gated:
        for rid in assignment.mid_regions.get(gc, []):
            gated_set.add(rid)

    # Centrality array aligned with pool_ids. Sidecar lookup: returns 0.0
    # when the node is unknown OR when the centrality key is absent (the
    # legacy "key not in node" branch collapses into the same default).
    _centrality_t0 = time.perf_counter()
    centrality_arr = np.zeros(len(pool_ids), dtype=np.float32)
    for i, rid in enumerate(pool_ids):
        centrality_arr[i] = float(graph.get_centrality(rid))
    if not np.any(centrality_arr) and pool_ids:
        try:
            cen_dict = graph.centrality()
            for i, rid in enumerate(pool_ids):
                centrality_arr[i] = float(cen_dict.get(rid, 0.0))
        except Exception as exc:  # noqa: BLE001 -- emit diagnostic then re-raise as NativeError
            write_event(
                store,
                "recall_centrality_failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise NativeError(f"centrality recompute failed: {exc}") from exc
    _recall_centrality_ms = (time.perf_counter() - _centrality_t0) * 1000.0

    # Stage 5 - Seeds. Pure numpy on the shared array.
    seed_indices = _pick_seeds(
        cosine_top_indices, shared_cos, centrality_arr, n=3,
    )
    seed_ids = [pool_ids[int(i)] for i in seed_indices]

    # Stage 6 - Reachable: cosine top-K ∪ 2-hop ∪ rich-club.
    # Skip spread when spread_hops=0 (latency gate).
    spread_ids = graph.two_hop_neighborhood(seed_ids, top_k=5) if spread_hops > 0 else []
    spread_indices = np.array(
        [id_to_idx[r] for r in spread_ids if r in id_to_idx],
        dtype=np.int64,
    )
    rich_indices = np.array(
        [id_to_idx[r] for r in (rich_club or []) if r in id_to_idx],
        dtype=np.int64,
    )
    # rank_threshold application: filter graph-derived secondary candidates
    # (spread + rich_club) that don't meet the cosine bar.
    if _arousal_rank_threshold_used > 0.0 and shared_cos.size:
        if spread_indices.size:
            spread_indices = spread_indices[
                shared_cos[spread_indices] >= _arousal_rank_threshold_used
            ]
        if rich_indices.size:
            rich_indices = rich_indices[
                shared_cos[rich_indices] >= _arousal_rank_threshold_used
            ]
    if cosine_top_indices.size or spread_indices.size or rich_indices.size:
        reachable_indices = np.union1d(
            np.union1d(cosine_top_indices, spread_indices),
            rich_indices,
        ).astype(np.int64)
    else:
        reachable_indices = np.empty(0, dtype=np.int64)

    # Stage 7 - Verbatim-mode filter (post-Stage-6 / pre-Stage-8).
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

    # Stage 8 - Rank (cosine reuse, mode-dependent community bias).
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
        except (TypeError, ValueError, AttributeError) as exc:
            logger.debug("literal_preservation_parse_failed: %s", exc)
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
    # Sidecar-keyed degree dict: keys are str(uuid) to match the existing
    # downstream consumer (Stage 5 rank reads `degree.get(str(rid))`).
    # When the bounded assembler (core.py) has computed GLOBAL edge counts via
    # an uncapped incident_edges call, it attaches them as graph._global_degree
    # so this rank stage uses the same degree signal as the full-graph path
    # instead of the bounded sub-graph degree.  Falls back to graph.degrees()
    # when _global_degree is absent (full-graph path, recall_for_benchmark,
    # cold-structural-degrade — all correctly use graph-local degrees).
    _global_deg_override: "dict[str, int] | None" = getattr(graph, "_global_degree", None)
    if _global_deg_override:
        degree = _global_deg_override
    else:
        degree = {str(nid): deg for nid, deg in graph.degrees()}

    # Mode-dependent gate bias scalar + arousal mode adjust (additive).
    mode_bias = _gate_bias_for_mode(mode) + _arousal_mode_bias_adjust

    # Deterministic exact-token boost: scan records_cache for exact substring
    # presence. Guarantees lossless verbatim recall for rare tokens (hex,
    # UUIDs, code symbols) that embedding may miss.
    fts_hits: set[UUID] = set()
    if cue and len(cue) >= 4:
        cue_lower = cue.lower()
        for rid, rec in records_cache.items():
            if rec.literal_surface and cue_lower in rec.literal_surface.lower():
                fts_hits.add(rid)

    # Build the contradicts-dst set once before the per-record loop. Empty
    # when cue intent is not historical_verbatim — per-record check
    # short-circuits so the set is only computed when needed.
    contradicts_dst_set: set[str] = set()
    if cue_intent == "historical_verbatim":
        contradicts_dst_set = _build_contradicts_dst_set(contradicts_outgoing)

    # When the cue asks for the superseded (pre-correction) wording, the
    # corrector record carries the cue's salient keyword and so sits high in
    # cosine rank, while the keyword-less original drifts down behind
    # moderately-related records. Capture each corrector's pre-adjustment
    # score during the per-record loop so the original can be anchored to its
    # OWN corrector's score afterwards (association-based recall of the
    # superseded fact). Keyed by corrector record-id string. Empty on every
    # non-historical cue so the regular path pays nothing.
    corrector_base_score: dict[str, float] = {}

    scored: list[tuple[float, UUID, float, float, float, float, float, float]] = []
    if reachable_indices.size:
        from iai_mcp.hebbian_structure import structural_similarity
        for idx in reachable_indices:
            i = int(idx)
            cid = pool_ids[i]
            rec = records_cache.get(cid)
            if rec is None:
                continue
            # Cosine read directly from shared array.
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
            # Mode-dependent additive bias for top-3 gated communities.
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
                # Thread the audit accumulator into the gains-application call.
                # knobs_applied may be None (back-compat callers).
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
            # Stability-instability lift: unstable / recently-touched memories
            # get a small additive boost so newly-resolved contradictions
            # don't get drowned by older same-topic facts after sleep
            # consolidation.
            try:
                _stability = getattr(rec, "stability", 0.5) or 0.5
                _ig = (1.0 - min(float(_stability), 1.0)) * 0.1
                s += _ig
            except (TypeError, ValueError, AttributeError) as exc:
                logger.debug("stability_lift_failed: %s", exc)
            # Valence multiplier.
            _valence = getattr(rec, "valence", None) or 0.0
            if _valence > 0.0:
                s *= (1.0 + _valence)
            # N-gram boost: trigram Jaccard similarity doubles score when >0.3.
            if cue and rec.literal_surface and _trigram_jaccard(cue.lower(), rec.literal_surface.lower()) > 0.3:
                s *= 2.0
            # Deterministic exact-token boost: triples score for substring match.
            if fts_hits and cid in fts_hits:
                s *= 3.0
            # Stage 8 historical-verbatim corrector score capture: when cue
            # intent signals the user wants the ORIGINAL pre-correction record,
            # record the corrector's natural (unmodified) score so the anchor
            # pass below can position the original immediately BELOW the
            # corrector. The corrector is NOT downweighted — current-fact
            # primacy is preserved (the corrector keeps its natural high rank).
            # The anchor pass lifts the buried original to just below its own
            # corrector, so the corrector (current truth) ranks first and the
            # superseded original surfaces in second position (still in top-10).
            if cue_intent == "historical_verbatim" and contradicts_dst_set:
                if str(cid) in contradicts_dst_set:
                    corrector_base_score[str(cid)] = s
            scored.append(
                (s, cid, cos, aaak, deg, deg_norm, age, structural_score),
            )

    # Historical-verbatim anchor pass (reconsolidation recall): the superseded
    # original is the SRC of a `contradicts` edge whose DST is the corrector.
    # When the cue asks for the original wording, surface the original by
    # association with its corrector — set the original's score to the
    # corrector's natural score MINUS a tie-break epsilon, so the corrector
    # (current truth) ranks first and the original ranks immediately below it.
    # Current-fact primacy is preserved: the corrector keeps its full
    # cosine-driven score and the original is anchored just under it.
    # This is magnitude-free (inherits the corrector's cosine-appropriate
    # score) and topic-selective (each original anchors to its OWN corrector).
    # Fires only on cue_intent == "historical_verbatim"; the regular recall
    # path and post_flip cues are untouched.
    if (
        cue_intent == "historical_verbatim"
        and contradicts_outgoing
        and corrector_base_score
        and scored
    ):
        _ANCHOR_EPSILON = 1e-4
        # src_id_str -> highest corrector (dst) score in-set.
        anchor_target: dict[str, float] = {}
        for src_s, dsts in contradicts_outgoing.items():
            best: float | None = None
            for d in dsts or []:
                cs = corrector_base_score.get(str(d))
                if cs is not None and (best is None or cs > best):
                    best = cs
            if best is not None:
                # Place original just BELOW its corrector (current-fact primacy).
                anchor_target[str(src_s)] = best - _ANCHOR_EPSILON
        if anchor_target:
            for j, row in enumerate(scored):
                tgt = anchor_target.get(str(row[1]))
                if tgt is not None and row[0] < tgt:
                    scored[j] = (tgt,) + row[1:]

    # Stage 9 - Sort: score desc, UUID asc tie-break.
    scored.sort(key=lambda x: (-x[0], str(x[1])))

    # Stage 10 - Build MemoryHits over the SORTED ranked list.
    # Provenance batch + retrieval_used event move to the entry points
    # so they fire only over the capped hits.
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
        _prov = (rec.provenance or [{}])[0]
        scored_hits.append(
            MemoryHit(
                record_id=cid,
                score=float(s),
                reason=reason,
                literal_surface=rec.literal_surface,
                adjacent_suggestions=suggestions,
                session_id=_prov.get("session_id"),
                captured_at=rec.created_at.isoformat() if rec.created_at else None,
            ),
        )
        budget_used += tokens

    # Post-rank work MUST run over the budget-capped hits, not over the full
    # ranked list. _recall_core returns the sorted full list; entry points
    # apply their cap THEN run the post-rank pipeline over the capped subset.
    activation_trace = list({*seed_ids, *spread_ids})

    # Arousal-budget A/B telemetry. Buffered to avoid blocking the hot-path.
    try:
        _top_hit_id_for_telemetry: str | None = None
        if scored_hits:
            _top_hit_id_for_telemetry = str(scored_hits[0].record_id)
        write_event(
            store,
            kind="retrieval_arousal_ab",
            data={
                "cue_hash": _arousal_cue_hash_hex,
                "route": _arousal_route,
                "n_hits": len(scored_hits),
                "budget_tokens_used": _arousal_budget_for_telemetry,
                "max_hops_used": _arousal_max_hops_used,
                "rank_threshold_used": _arousal_rank_threshold_used,
                "arousal_level": _arousal_level_for_telemetry,
                "arousal_mode": _arousal_mode_for_telemetry,
                "top_hit_id": _top_hit_id_for_telemetry,
            },
            severity="info",
            session_id=session_id,
            buffered=True,
        )
    except Exception as exc:  # noqa: BLE001 -- telemetry must never crash recall
        logger.debug("retrieval_arousal_ab_emit_failed: %s", exc)

    # Per-recall latency telemetry — emitted at a tunable sample rate (env
    # IAI_MCP_RECALL_SAMPLE_RATE; default 0.1 = 1-in-10). Best-effort emit
    # wrapped in try/except so a hypothetical event-store outage cannot
    # cascade into a recall failure visible to the user.
    try:
        _sample_rate = float(os.environ.get("IAI_MCP_RECALL_SAMPLE_RATE", "0.1"))
    except (TypeError, ValueError):
        _sample_rate = 0.1
    if random.random() < _sample_rate:
        try:
            write_event(
                store,
                kind="recall_timing",
                data={
                    "centrality_ms": float(_recall_centrality_ms),
                    "sigma_ms": 0.0,  # sigma not invoked on the recall hot path
                    "pool_collection_ms": float(_recall_pool_collection_ms),
                    "n_nodes": int(len(pool_ids)),
                },
                severity="info",
                session_id=session_id,
            )
        except Exception as exc:  # noqa: BLE001 -- telemetry MUST NOT break recall
            logger.debug("recall_timing_emit_failed: %s", exc)

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
    contradicts_outgoing: dict[str, list[str]] | None = None,
) -> tuple[list[MemoryHit], list[MemoryHit], list[dict], list[dict]]:
    """Post-rank work shared by both entry points.

    Operates on the BUDGET/K-CAPPED `hits` list, not on the full ranked
    `scored_hits` from `_recall_core`. This restores the correct semantic
    order: rank → cap → side-effects-over-capped-set.

    The function applies different scopes to different stages:
      - O(N) per-record work (provenance, profile_modulates, retrieval_used,
        patterns_observed strip) runs over the FULL caller-facing `hits`.
        This ensures every hit returned gets a provenance entry.
      - O(N²) heavy work (anti-hits lookup, s4 pairwise contradiction
        scan, schema/curiosity entropy) runs over the top
        `_POST_RANK_MAX_HITS` (default 50) of `hits`. This bounds the
        s4 pairwise scan to ~1225 pair checks regardless of how many
        hits the caller-facing list contains. Matches the effective
        post-rank input size on healthy graphs.

    Returns: (hits_after_pattern_strip, anti_hits, hints, patterns_observed).

    Stages:
      11. Provenance batch over full hits.
      12. Anti-hits over capped subset (s4 scope).
      13. S4 hints over capped subset, skipped in verbatim mode.
      14. profile_modulates edges over full hits (batched).
      15. Provisional schema + curiosity hints over capped subset, skipped in verbatim.
      16. Concept-mode patterns_observed strip over full hits.
      17. retrieval_used event with full hit_ids.
    """
    # Heavy O(N²) post-rank scope is bounded by _POST_RANK_MAX_HITS.
    s4_scope_hits = hits[:_POST_RANK_MAX_HITS]

    # Stage 11 - Provenance batch over the FULL caller-facing hits.
    # Deferred to .deferred-provenance.jsonl — flushed by daemon WAKE handler.
    if hits:
        try:
            from iai_mcp.provenance_buffer import defer_provenance
            defer_provenance(
                store,
                [(h.record_id, cue, session_id) for h in hits],
            )
        except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
            logger.debug("provenance_defer_failed: %s", exc)

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
                contradicts_outgoing=contradicts_outgoing,
            )
        except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
            logger.debug("s4_on_read_check_batch_failed: %s", exc)
            hints = []

    # Stage 14 - profile_modulates edges over the FULL caller-facing hits.
    # O(N) cheap; no cap.
    # CC-E: chunk into ≤_BOOST_SMALL_BATCH calls so boost_edges always
    # takes the predicate-filtered small-batch fast path (store.py).  This
    # avoids the large-batch `edges.to_pandas()` full-table scan that fires
    # when >4 distinct (src,dst) pairs are coalesced in one call.  The edge
    # WRITE persists; the overlay epoch does NOT bump per-mutation (harmless).
    _BOOST_SMALL_BATCH: int = 4  # matches store.py's _SMALL_BATCH
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
            except (TypeError, ValueError, AttributeError) as exc:
                logger.debug("profile_modulate_per_hit_failed rid=%s: %s", h.record_id, exc)
                continue
        if modulate_pairs:
            try:
                # Chunk into ≤_BOOST_SMALL_BATCH slices so each slice takes
                # the fast predicate-filtered path in boost_edges rather than
                # the large-batch full edges.to_pandas() scan (CC-E).
                for _chunk_start in range(0, len(modulate_pairs), _BOOST_SMALL_BATCH):
                    _chunk_pairs = modulate_pairs[_chunk_start:_chunk_start + _BOOST_SMALL_BATCH]
                    _chunk_deltas = modulate_deltas[_chunk_start:_chunk_start + _BOOST_SMALL_BATCH]
                    try:
                        store.boost_edges(
                            _chunk_pairs,
                            edge_type="profile_modulates",
                            delta=_chunk_deltas,
                        )
                    except Exception as _chunk_exc:  # noqa: BLE001 — per-chunk degrade
                        logger.debug("boost_edges_chunk_failed: %s", _chunk_exc)
            except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
                logger.debug("boost_edges_profile_modulates_failed: %s", exc)

    # Stage 15 - Curiosity + schema hints DEFERRED to background overlay.
    # Previously computed synchronously (35ms). Now buffered as a deferred
    # event; the next recall or daemon tick surfaces them. Hot path cost: 0ms.
    if mode != "verbatim" and s4_scope_hits:
        try:
            write_event(
                store,
                kind="deferred_curiosity_input",
                data={
                    "hit_ids": [str(h.record_id) for h in s4_scope_hits[:10]],
                    "cue": cue[:200],
                    "session_id": session_id,
                },
                severity="info",
                session_id=session_id,
                buffered=True,
            )
        except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
            logger.debug("deferred_curiosity_input_event_failed: %s", exc)

    # Stage 16 - Concept-mode patterns_observed strip over the FULL hits.
    # Schema records (tier=semantic AND tag=pattern:*)
    # are stripped from `hits` into `patterns_observed`; max 3 entries.
    patterns_observed: list[dict] = []
    if mode == "concept":
        kept_hits: list[MemoryHit] = []
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
                    # Reroute: count schema_instance_of edges via bounded
                    # incident_edges on the schema record id (NOT full edges
                    # table to_pandas).
                    evidence_count = 0
                    try:
                        _schema_edges = store.incident_edges(
                            [h.record_id],
                            edge_types=["schema_instance_of"],
                            top_k=None,
                        )
                        evidence_count = sum(
                            len(v) for v in _schema_edges.values()
                        )
                    except Exception as exc:  # noqa: BLE001 — degradable evidence count
                        logger.debug("evidence_count_incident_edges_failed: %s", exc)
                        evidence_count = 0
                    patterns_observed.append({
                        "pattern": pattern_str,
                        "evidence_count": evidence_count,
                        "schema_id": str(h.record_id),
                    })
            else:
                kept_hits.append(h)
        hits = kept_hits

    # Stage 17 - retrieval_used event with full hit_ids.
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
            buffered=True,
        )
    except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
        logger.debug("retrieval_used_event_failed: %s", exc)

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
    arousal_state: dict | None = None,
    tv_maps: "tuple[dict, dict] | None" = None,
) -> RecallResponse:
    """Production answer-packing entry point.

    Calls `_recall_core` for the load-bearing recall computation, then
    packs hits under `budget_tokens`: the ranker's sorted output is
    consumed in score-desc order; each hit contributes
    `tokens = len(rec.literal_surface) // 4` to a running budget; the
    loop breaks when `budget_used + tokens > budget_tokens` AND
    `len(hits) >= 1` (always at least one hit).

    This entry point does NOT accept a `k_hits` parameter. Production
    callers want token-budget-shaped responses for prompt assembly. For
    benchmark-shape (deterministic top-K), use `recall_for_benchmark`.

    Mode plumbing: the `mode` parameter is set upstream by the
    cue-classifier (`core.py:dispatch()`) and is passed through to
    `_recall_core` unchanged. Inside `_recall_core` Stage 5,
    `_gate_bias_for_mode(mode)` selects the community-gate soft-bias
    scalar (verbatim=0.0, concept=0.1).
    """
    # Enactive auto-depth: reduce community gate and disable spread if
    # prior recall was slow (>2000ms). Self-regulating depth control.
    import time as _time
    global _last_recall_latency_ms
    _rfr_t0 = _time.perf_counter()

    # Arousal diagnostic log.
    if arousal_state:
        logger.debug(
            "arousal_recall: level=%.2f mode=%s budget=%d",
            arousal_state.get("level", 0.5),
            arousal_state.get("mode", "unknown"),
            budget_tokens,
        )

    _k_com = 1 if _last_recall_latency_ms > 2000 else 3
    _s_hops = 0 if _last_recall_latency_ms > 2000 else 2

    # Build (outgoing, ts_by_id) maps ONCE before _recall_core so the
    # Stage 8 historical-verbatim downweight can read contradicts_outgoing
    # without a second records-table scan. The same maps are reused below
    # for the temporal-validity enrichment / stale-downweight pass —
    # single records.to_pandas() per recall.
    #
    # mode is set upstream by the caller (core.dispatch passes the
    # cue-classifier's mode unchanged); intent is computed here from the
    # same classifier so production and bench paths share routing semantics.
    from iai_mcp.cue_router import _classify_cue
    from iai_mcp.retrieve import (
        apply_stale_downweight,
        build_temporal_validity_maps,
        derive_temporal_validity,
    )
    _cue_mode_unused, _cue_intent, _cue_label_unused = _classify_cue(cue)
    # Use pre-built candidate-scoped tv_maps when provided (bounded ANN path).
    # Fall back to build_temporal_validity_maps only when tv_maps is None
    # (non-ANN callers, recall_for_benchmark, etc.).
    if tv_maps is not None:
        _tv_outgoing, _tv_ts = tv_maps
    else:
        _tv_maps_built = build_temporal_validity_maps(store)
        _tv_outgoing, _tv_ts = (_tv_maps_built if _tv_maps_built is not None else ({}, {}))

    core = _recall_core(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        knobs_applied=knobs_applied,
        k_communities=_k_com,
        spread_hops=_s_hops,
        cue_intent=_cue_intent,
        contradicts_outgoing=_tv_outgoing,
    )

    # Enrich + downweight + re-sort BEFORE the budget-pack loop so a stale
    # high-cosine hit does not consume budget that should go to a fresh
    # lower-cosine record. Order is load-bearing.
    # core.anti_hits is empty on the regular path (_recall_core returns
    # them empty); anti_hits are built later inside _apply_post_rank_pipeline.
    # The L0 fast-path early-return below DOES use core.anti_hits, so enrich
    # both to cover both paths cleanly.
    #
    # The (outgoing, ts_by_id) maps are reused from the pre-_recall_core
    # build above — no second records-table scan.
    #
    # NOTE: deliberately NOT consuming core._records_cache for created_at —
    # SimpleRecordView.created_at is a wall-clock placeholder (graph node
    # payload does not carry record.created_at), which would poison the
    # derived valid_from / valid_to.
    derive_temporal_validity(
        None, core.scored_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    derive_temporal_validity(
        None, core.anti_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    apply_stale_downweight(core.scored_hits, cue_intent=_cue_intent)
    apply_stale_downweight(core.anti_hits, cue_intent=_cue_intent)
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

    # Pack hits under budget_tokens.
    # The budget-pack loop also respects `_POST_RANK_MAX_HITS` (default 50)
    # as a safety cap on the number of records that flow into the post-rank
    # pipeline. The wider candidate pool (K_CANDIDATES=200) is preserved for
    # ranking accuracy, but the response surface stays bounded by the same cap
    # that keeps O(N²) s4 work and provenance writes within perf bounds.
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

    # QUAL-02: recency union for just-written embedding_pending markers.
    # query_similar excludes embedding_pending rows (store.py WHERE clause).
    # UNION the ranked hits with store.recent_pending_markers (BOUNDED
    # index-backed filter-in-SQL, NOT all_records) so a record written
    # moments ago surfaces even before it is embedded.  Deduped by record_id
    # so a record that was both ranked and pending is not double-counted.
    try:
        _pending_n = max(10, len(hits))
        _pending_markers = store.recent_pending_markers(n=_pending_n)
        _ranked_ids: set = {h.record_id for h in hits}
        for _pm in _pending_markers:
            if _pm.id not in _ranked_ids:
                _ranked_ids.add(_pm.id)
                hits.append(MemoryHit(
                    record_id=_pm.id,
                    score=0.0,
                    reason="pending-recency",
                    literal_surface=_pm.literal_surface or "",
                    adjacent_suggestions=[],
                    session_id=(_pm.provenance[0].get("session_id") if _pm.provenance else None),
                    captured_at=(
                        _pm.created_at.isoformat() if _pm.created_at else None
                    ),
                ))
    except Exception as _pm_exc:  # noqa: BLE001 -- recency union is additive; never crash recall
        logger.debug("pending_markers_union_failed: %s", _pm_exc)

    # Provenance enrichment: when the hot-path records_cache contains
    # SimpleRecordView objects (graph-sidecar, no provenance), session_id
    # and captured_at are None on the MemoryHit. Enrich only the
    # budget-capped hits (O(K_budget), typically 1-10) via a targeted
    # store.get so the public recall surface carries originating session.
    # Wrapped in try/except — provenance is additive; never crash recall.
    for _h in hits:
        if _h.session_id is None:
            try:
                _full_rec = store.get(_h.record_id)
                if _full_rec is not None:
                    _h_prov = (_full_rec.provenance or [{}])[0]
                    _h.session_id = _h_prov.get("session_id")
                    _h.captured_at = (
                        _full_rec.created_at.isoformat()
                        if _full_rec.created_at else None
                    )
            except Exception as _exc:  # noqa: BLE001 -- additive enrichment, never crash recall
                logger.debug("hit_provenance_enrich_failed rid=%s: %s", _h.record_id, _exc)

    # Post-rank pipeline runs over the capped hits. Heavy O(N²) work
    # (s4, anti-hits, schema/curiosity) is bounded to `_POST_RANK_MAX_HITS`
    # while cheap O(N) work spans the full hits.
    hits, anti_hits, hints, patterns_observed = _apply_post_rank_pipeline(
        hits,
        store=store, graph=graph, records_cache=core._records_cache,
        cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        budget_used=budget_used, path_label="recall_for_response",
        knobs_applied=knobs_applied,
        contradicts_outgoing=_tv_outgoing,
    )

    # Final budget enforcement — recompute from the fully assembled `hits`
    # list (which may include pending-recency markers and may have had
    # schema records stripped by _apply_post_rank_pipeline Stage 16).
    # The pre-rank budget pack above is a first-pass cap that only counts
    # the scored hits; this pass enforces the cap over the final surface.
    # Contract: stop appending once the next hit would push total over
    # budget_tokens; always keep at least one hit; never truncate the
    # literal_surface of any individual record (lossless verbatim).
    # budget_used is re-derived from the capped list so the returned value
    # and the returned hits are consistent.
    if hits:
        _final_hits: list[MemoryHit] = []
        _final_budget = 0
        for _fh in hits:
            _fh_tokens = len(_fh.literal_surface) // 4
            if _final_budget + _fh_tokens > budget_tokens and _final_hits:
                break
            _final_hits.append(_fh)
            _final_budget += _fh_tokens
        hits = _final_hits
        budget_used = _final_budget

    # anti_hits are constructed inside _apply_post_rank_pipeline (Stage 12)
    # from contradicts-edge neighbours with score=0.0. They never pass
    # through the pre-budget enrichment above, so enrich them here so the
    # JSON wire carries valid_from/valid_to on the anti_hits surface too.
    # Score downweight is a no-op on score=0.0 baseline anti-hits; the
    # value is the valid_from/valid_to fields + " · stale" reason marker.
    # anti_hits intentionally NOT re-sorted: they are an inhibitory tail.
    # Reuses the (outgoing, ts_by_id) maps built once per recall above —
    # no second records-table scan.
    derive_temporal_validity(
        None, anti_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    apply_stale_downweight(anti_hits)

    # Enactive auto-depth: record wall-clock for next recall's gate.
    _last_recall_latency_ms = (_time.perf_counter() - _rfr_t0) * 1000

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
    """Benchmark top-K entry point.

    Calls `_recall_core` for the load-bearing recall computation, then
    takes the top `k_hits` from the sorted `scored_hits`. Deterministic:
    no token budget, no per-hit pack rule. Used by:
      - bench/longmemeval_blind.py (Y prong)
      - bench/lme500/debug_pipeline_loss.py (stage tracer)
      - any future benchmark harness needing top-K retrieval surface.

    This entry point does NOT accept a `budget_tokens` parameter.
    For production answer-packing (token-budget-shaped responses),
    use `recall_for_response`.

    Mode plumbing: bench callers pass `mode="concept"`; the parameter is
    passed through to `_recall_core` unchanged so the mode-dependent
    gate bias (`_gate_bias_for_mode(mode)`) operates as designed.

    Cue intent plumbing: bench harnesses historically hardcoded `mode="concept"`
    and never called `_classify_cue`. This function calls `_classify_cue`
    internally and passes the resulting intent into `_recall_core` so bench
    harnesses get intent routing without changing their mode semantics.
    """
    # Classify cue + build contradicts_outgoing once so Stage 8 can apply
    # the historical-verbatim downweight without a second store scan.
    from iai_mcp.cue_router import _classify_cue
    from iai_mcp.retrieve import build_temporal_validity_maps
    _cue_mode_unused, _cue_intent, _cue_label_unused = _classify_cue(cue)
    _tv_maps = build_temporal_validity_maps(store)
    _tv_outgoing, _tv_ts_unused = (_tv_maps if _tv_maps is not None else ({}, {}))

    core = _recall_core(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        knobs_applied=knobs_applied,
        cue_intent=_cue_intent,
        contradicts_outgoing=_tv_outgoing,
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

    # Post-rank pipeline runs over the capped (top-k_hits) hits. Heavy O(N²)
    # work is bounded to `_POST_RANK_MAX_HITS` while cheap O(N) work spans
    # the full hits. Pass `contradicts_outgoing` so the s4 hints branch sees
    # the same edge geometry the Stage 8 downweight used — single map shared
    # across both stages of one recall.
    hits, anti_hits, hints, patterns_observed = _apply_post_rank_pipeline(
        hits,
        store=store, graph=graph, records_cache=core._records_cache,
        cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        budget_used=budget_used, path_label="recall_for_benchmark",
        knobs_applied=knobs_applied,
        contradicts_outgoing=_tv_outgoing,
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

