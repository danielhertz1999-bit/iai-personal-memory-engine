"""Sleep-cycle replay.

Two phases:

- `run_light_consolidation` -- runs at every session_exit. Pure-local. NO LLM.
  FSRS tick on recently-recalled records. Sub-second. Always on.

- `run_heavy_consolidation` -- runs inside quiet window OR via MANUAL trigger
  (memory_consolidate MCP tool). A guard ladder gates any Tier-1 LLM path via
  `should_call_llm`; Tier-0 fallback is ALWAYS present (TF-IDF + cooccurrence
  summarisation). Creates `consolidated_from` edges linking semantic summary
  records to their source episodes. Runs FSRS edge decay sweep. Logs
  `cls_consolidation_run` event with mode=heavy, tier=tier0|tier1.

Scheduler (`should_run_heavy`):
- ACTIVITY (default): idle>=30min AND local time in quiet_window.
- TIME: strict cron at hour==3 local.
- MANUAL: never fires automatically.
- 48h max defer: if idle >= max_defer_hours, force-run regardless of window.

Decay sweep (`_decay_edges`):
- Only hebbian edges are decayed. contradicts / invariant_anchor /
  consolidated_from / schema_instance_of / temporal_next / curiosity_bridge /
  profile_modulates all survive forever by design.
- Edges > 90d stale: weight *= 0.9 ** (days - 90); prune if < ε (default 0.01).

Unification: the heavy cycle drives FSRS decay + summarisation +
schema-candidate surfacing in a single pass -- no duplicated IO.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from itertools import combinations
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from iai_mcp.aaak import enforce_language_tagged, generate_aaak_index
from iai_mcp.events import write_event
from iai_mcp.guard import BudgetLedger, RateLimitLedger, should_call_llm
from iai_mcp.store import EDGES_TABLE, MemoryStore, _uuid_literal
from iai_mcp.types import MemoryRecord


# ---------------------------------------------------------------- constants


class SleepMode(str, Enum):
    """trigger mode for heavy consolidation."""

    ACTIVITY = "activity"   # Idle-triggered (default). 30min idle + quiet window.
    TIME = "time"           # Strict cron at hour==3 local.
    MANUAL = "manual"       # Only via memory_consolidate tool.


@dataclass
class SleepConfig:
    """User-configurable sleep-cycle schedule knobs."""

    mode: SleepMode = SleepMode.ACTIVITY
    quiet_window: tuple[int, int] = (22, 6)   # local-hour start..end (wrap-around)
    require_idle_minutes: int = 30
    max_defer_hours: int = 48
    on_user_resume: str = "defer_remaining"
    light_on_exit: bool = True
    llm_enabled: bool = False                 # Tier 0 default
    llm_tier: int = 1                         # 1=Haiku-Batch, 2=Sonnet/Opus


DECAY_EPSILON: float = 0.01                   # prune threshold
DECAY_GRACE_DAYS: int = 90                    # no decay for edges <=90d old
DECAY_BASE: float = 0.9                       # weight *= 0.9^(days-90)
FSRS_STABILITY_BOOST: float = 0.2             # simple per-recall linear boost
CLUSTER_MIN_SIZE: int = 3                     # consolidation cluster threshold
# Hebbian LTP increment applied to existing edges between
# co-cluster members during heavy consolidation. Mirrors the LTD side (DECAY_*)
# so the graph strengthens frequently-co-retrieved associations during sleep,
# not only during explicit user-session pipeline_recall. Conservative delta --
# 10 consolidations bring a fresh edge from 0.05 to ~0.5 stable.
HEAVY_LTP_DELTA: float = 0.05


# ---------------------------------------------------------------- scheduler


def should_run_heavy(
    now_utc: datetime,
    last_activity_utc: datetime,
    config: SleepConfig,
    tz: ZoneInfo,
) -> tuple[bool, str]:
    """Trigger evaluator.

    Returns (ok, reason). reason is "" on success, a short diagnostic otherwise.

    The 48h deadline (config.max_defer_hours) overrides MANUAL, TIME, and
    ACTIVITY path-gates -- if the user has ignored the brain for 48h, we MUST
    consolidate before the next session starts. This is an S4 viability
    requirement.
    """
    idle_minutes = (now_utc - last_activity_utc).total_seconds() / 60.0

    # 48h force-run. Precedes MANUAL so a stuck manual-only deployment still
    # gets periodic consolidation.
    if idle_minutes >= config.max_defer_hours * 60:
        return True, f"max_defer_hours ({config.max_defer_hours}h) exceeded"

    if config.mode == SleepMode.MANUAL:
        return False, "manual-only mode"

    if config.mode == SleepMode.TIME:
        local = now_utc.astimezone(tz)
        ok = local.hour == 3
        return ok, f"TIME mode, local hour={local.hour}"

    # ACTIVITY mode from here on.
    if idle_minutes < config.require_idle_minutes:
        return False, f"idle < {config.require_idle_minutes}min"

    local = now_utc.astimezone(tz)
    start_h, end_h = config.quiet_window
    # Wrap-around window support: (22, 6) means 22-23 OR 0-5.
    if start_h > end_h:
        in_window = (local.hour >= start_h) or (local.hour < end_h)
    else:
        in_window = start_h <= local.hour < end_h
    if not in_window:
        return False, (
            f"outside quiet window {config.quiet_window}, "
            f"local hour={local.hour}"
        )
    return True, ""


# ---------------------------------------------------------------- FSRS bits


def _apply_fsrs(record: MemoryRecord, now: datetime) -> MemoryRecord:
    """Simple FSRS-inspired stability boost for recently-recalled records.

     scope: linear +0.2 per recall, capped at 1.0. Full FSRS (Woz et al
    2022) with per-difficulty retrievability modelling is.
    """
    if record.never_decay:
        return record
    record.stability = min(1.0, record.stability + FSRS_STABILITY_BOOST)
    record.last_reviewed = now
    return record


def _decay_edges(
    store: MemoryStore, epsilon: float = DECAY_EPSILON,
    plasticity_gain: float = 1.0,
) -> dict:
    """nightly sweep: decay stale hebbian + hebbian_structure edges, prune below e.

    Structure-edge LTP from hebbian_structure.strengthen_structure_edge decays
    under the SAME formula and grace period as content-edge hebbian (FSRS
    decay on structure edges is IDENTICAL to record-edge decay).

    pattern_separation_seed edges (pre-insert link layer) decay under
    the SAME formula and grace period as hebbian -- they are write-time seeds,
    not permanent anchors, so unused seeds prune like any other co-activation edge.

    hebbian_cluster_replay edges (REM-phase temporal-cluster Hebbian
    seeds) decay under the SAME formula and grace period as hebbian -- they are
    write-time temporal-coactivation seeds, not permanent anchors, so unused
    cluster-replay edges prune like any other co-activation edge.

    Other edge types (contradicts, invariant_anchor, consolidated_from,
    schema_instance_of, temporal_next, curiosity_bridge, profile_modulates)
    survive forever.
    """
    tbl = store.db.open_table(EDGES_TABLE)
    df = tbl.to_pandas()
    if df.empty:
        return {"decayed": 0, "pruned": 0}

    now = datetime.now(timezone.utc)
    decayed = 0
    pruned = 0

    # Include hebbian_structure in the sweep with identical formula.
    # Include pattern_separation_seed (pre-insert link layer).
    # Include hebbian_cluster_replay (temporal cluster Hebbian seeds).
    decayable_kinds = (
        "hebbian",
        "hebbian_structure",
        "pattern_separation_seed",
        "hebbian_cluster_replay",
    )
    hebbian = df[df["edge_type"].isin(decayable_kinds)]
    for _, row in hebbian.iterrows():
        # Per-row try/except ValueError so one poisoned row
        # cannot kill the entire sweep. _uuid_literal raises ValueError on any
        # non-RFC-4122 UUID string, preventing SQL predicate injection via a
        # corrupt or adversarial `src`/`dst` value.
        try:
            last = row["updated_at"]
            if last is None:
                continue
            # Coerce ISO TEXT / Timestamp / naive datetime -> tz-aware UTC datetime.
            try:
                py = last.to_pydatetime() if hasattr(last, "to_pydatetime") else last
            except (TypeError, ValueError, AttributeError):
                py = last
            if isinstance(py, str):
                try:
                    py = datetime.fromisoformat(py.replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    continue
            if not isinstance(py, datetime):
                continue
            if py.tzinfo is None:
                py = py.replace(tzinfo=timezone.utc)

            days = (now - py).total_seconds() / 86400.0
            if days <= DECAY_GRACE_DAYS:
                continue

            new_weight = float(row["weight"]) * (DECAY_BASE ** ((days - DECAY_GRACE_DAYS) * plasticity_gain))

            # fix: reject non-canonical UUID values BEFORE interpolation.
            src_lit = _uuid_literal(row["src"])
            dst_lit = _uuid_literal(row["dst"])
            edge_kind = str(row["edge_type"])
            if edge_kind not in decayable_kinds:
                # Belt-and-braces: should not happen given the.isin() above.
                continue
            if new_weight < epsilon:
                tbl.delete(
                    f"src = '{src_lit}' AND dst = '{dst_lit}' "
                    f"AND edge_type = '{edge_kind}'"
                )
                pruned += 1
            else:
                tbl.update(
                    where=(
                        f"src = '{src_lit}' AND dst = '{dst_lit}' "
                        f"AND edge_type = '{edge_kind}'"
                    ),
                    values={
                        "weight": float(new_weight),
                        "updated_at": now,
                    },
                )
                decayed += 1
        except ValueError:
            # Poisoned UUID shape -- skip this row, continue the sweep.
            continue

    return {"decayed": decayed, "pruned": pruned}


# ---------------------------------------------------------------- light phase


def run_light_consolidation(
    store: MemoryStore, session_id: str,
) -> dict:
    """light phase -- always on, pure local, no LLM.

    Runs at every session_exit. Nudges FSRS stability on records that were
    recalled in this session (identified by fresh provenance entry within the
    last hour). Writes one `cls_consolidation_run` event with mode=light.
    """
    now = datetime.now(timezone.utc)
    records = store.all_records()
    fsrs_ticked = 0

    for r in records:
        if r.never_decay:
            continue
        if not r.provenance:
            continue
        last_prov = r.provenance[-1]
        try:
            ts_str = last_prov.get("ts", "")
            prov_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if prov_ts.tzinfo is None:
                prov_ts = prov_ts.replace(tzinfo=timezone.utc)
            # Only tick records recalled within the last hour.
            if (now - prov_ts).total_seconds() < 3600:
                _apply_fsrs(r, now)
                # Persist the FSRS mutation so stability
                # and last_reviewed survive process restart. update_record
                # rewrites only the FSRS-relevant columns -- embedding,
                # provenance, tags etc. are left intact.
                store.update_record(r)
                fsrs_ticked += 1
        except (TypeError, ValueError, KeyError, AttributeError):
            # Provenance ts malformed -- ignore that record, don't fail the sweep.
            continue

    write_event(
        store,
        kind="cls_consolidation_run",
        data={
            "mode": "light",
            "fsrs_ticked": fsrs_ticked,
            "record_count": len(records),
        },
        severity="info",
        session_id=session_id,
    )
    return {
        "mode": "light",
        "fsrs_ticked": fsrs_ticked,
        "cooccurrence_updates": 0,  # populates real cooccurrence counts.
    }


# ---------------------------------------------------------------- heavy phase


def _build_hebbian_clusters(store: MemoryStore) -> list[list[UUID]]:
    """Find connected components in the hebbian edge graph with size >= CLUSTER_MIN_SIZE."""
    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    if edges_df.empty:
        return []
    hebbian = edges_df[edges_df["edge_type"] == "hebbian"]
    if hebbian.empty:
        return []

    adj: dict[UUID, set[UUID]] = {}
    for _, row in hebbian.iterrows():
        src = UUID(row["src"])
        dst = UUID(row["dst"])
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)

    visited: set[UUID] = set()
    clusters: list[list[UUID]] = []
    for node in list(adj.keys()):
        if node in visited:
            continue
        stack = [node]
        component: list[UUID] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            component.append(cur)
            for neigh in adj.get(cur, set()):
                if neigh not in visited:
                    stack.append(neigh)
        if len(component) >= CLUSTER_MIN_SIZE:
            clusters.append(component)
    return clusters


def _tier0_schema_surfacing(store: MemoryStore) -> list[dict]:
    """Tier-0 fallback schema candidate surfacing: tags appearing in >=3 records.

    Schema induction consumes these candidates.

    Rewritten on ``store.iter_record_columns(["tags_json"])``.
    No more full-store load + full-record decrypt -- only the ``tags_json`` column
    is read from disk; encrypted columns (literal_surface, provenance_json,
    profile_modulation_gain_json) are NEVER touched on this path. Saves ~16210
    AES-GCM operations + ~14.5 MB literal_surface materialisation + ~2.4 MB
    provenance_json materialisation + ~11.9 MB embedding materialisation per
    invocation on the production store.
    """
    tag_counts: dict[str, int] = {}
    record_count = 0
    for row in store.iter_record_columns(["tags_json"], batch_size=1024):
        record_count += 1
        tags_raw = row.get("tags_json") or "[]"
        try:
            tags = json.loads(tags_raw) if tags_raw else []
        except (TypeError, json.JSONDecodeError):
            tags = []
        for t in tags:
            # Skip language-qualifying raw:* and domain:* tags -- those are
            # classification metadata, not schema-candidate signals.
            if t.startswith("raw:") or t.startswith("domain:"):
                continue
            tag_counts[t] = tag_counts.get(t, 0) + 1
    if record_count < CLUSTER_MIN_SIZE:
        return []
    candidates: list[dict] = []
    for tag, count in tag_counts.items():
        if count >= 3:
            candidates.append(
                {
                    "pattern": f"tag:{tag}",
                    "confidence": min(1.0, count / 10.0),
                    "evidence_count": count,
                }
            )
    return candidates


def _create_semantic_summary(
    store: MemoryStore,
    cluster: list[MemoryRecord],
    summary_text: str,
    language: str,
) -> UUID:
    """Insert one semantic summary record + a consolidated_from edge to each source.

    The summary inherits the dominant language of the source cluster.
    detail_level=3 -> never_decay=True (auto-enforced by __post_init__).
    """
    # Lazy import -- embedder load is heavy; only needed when we actually summarise.
    from iai_mcp.embed import embedder_for_store

    emb = embedder_for_store(store).embed(summary_text)
    now = datetime.now(timezone.utc)
    summary_id = uuid4()
    summary = MemoryRecord(
        id=summary_id,
        tier="semantic",
        literal_surface=summary_text,
        aaak_index="",
        embedding=emb,
        community_id=None,
        centrality=0.0,
        detail_level=3,  # semantic summaries protected from decay
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=now,
        never_decay=True,
        never_merge=False,
        provenance=[
            {
                "ts": now.isoformat(),
                "cue": "cls_consolidation",
                "session_id": "system",
            }
        ],
        created_at=now,
        updated_at=now,
        tags=["semantic", "cls_summary"],
        language=language,
    )
    enforce_language_tagged(summary)
    summary.aaak_index = generate_aaak_index(summary)
    store.insert(summary)

    # Batch all consolidated_from edges into a single
    # boost_edges call (one merge_insert + one tbl.add at most). Previously
    # this loop emitted N store versions on the edges table for an N-source
    # cluster.
    pairs = [(summary_id, source.id) for source in cluster]
    if pairs:
        store.boost_edges(
            pairs,
            edge_type="consolidated_from",
            delta=1.0,
        )
    return summary_id


def _persist_tier1_schemas(
    store: MemoryStore,
    budget: "BudgetLedger",
    rate: "RateLimitLedger",
    llm_enabled: bool,
) -> "tuple[list, int]":
    """Extract: schema induction (Tier-1 pass-through) + auto-status persistence.

    Holds the EXACT legacy schema slice from run_heavy_consolidation:
    - Calls induce_schemas_tier1 (preserving its llm_health event emit).
    - Persists every status=="auto" candidate via persist_schema (creating
      schema_instance_of edges). pending_user_approval candidates are only
      logged (via induce_schemas_tier1's llm_health emission path).
    - Wraps the whole block in the legacy try/except boundary.

    Returns (candidates, persisted_count) where persisted_count is the number
    of auto-status candidates that were successfully persisted.

    Single-source invariant: both run_heavy_consolidation and the canonical
    _step_schema_mine call this helper — there is exactly ONE implementation
    of this slice.
    """
    persisted = 0
    candidates: list = []
    try:
        from iai_mcp.schema import (
            induce_schemas_tier1,
            persist_schema,
        )

        candidates = induce_schemas_tier1(
            store, budget=budget, rate=rate,
            llm_enabled=llm_enabled,
        )
        for cand in candidates:
            if cand.status == "auto":
                persist_schema(store, cand)
                persisted += 1
            # pending_user_approval candidates are only logged (via
            # induce_schemas_tier1's llm_health emission path).
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        write_event(
            store,
            kind="schema_induction_run",
            data={"error": str(exc), "status": "failed"},
            severity="warning",
            session_id="system",
        )
    return candidates, persisted


def _process_cluster_summaries(store: MemoryStore) -> int:
    """Extract: hebbian cluster-find → semantic summaries + consolidated_from edges + LTP.

    Holds the EXACT legacy cluster->summary->LTP slice from run_heavy_consolidation:
    - Builds the records_by_id map via a single store.all_records call for this slice.
    - Finds connected components of edge_type=="hebbian" with size >= CLUSTER_MIN_SIZE.
    - For each qualifying cluster: votes dominant language, builds summary text,
      inserts one semantic MemoryRecord + consolidated_from edges via
      _create_semantic_summary, then boosts cluster hebbian edges by HEAVY_LTP_DELTA.

    Returns summaries_created (number of semantic summary records inserted).

    Single-source invariant: both run_heavy_consolidation and the canonical
    _step_cluster_summary call this helper — there is exactly ONE implementation
    of this slice.

    Streaming invariant: this function owns the SINGLE store.all_records call for
    the cluster path. run_heavy_consolidation and the canonical step must NOT also
    call store.all_records directly for this purpose.
    """
    clusters = _build_hebbian_clusters(store)
    # Single materialisation: owns the ONLY store.all_records call for this slice.
    records_by_id = {r.id: r for r in store.all_records()}
    summaries_created = 0
    for cluster_ids in clusters:
        cluster_recs = [records_by_id[i] for i in cluster_ids if i in records_by_id]
        if len(cluster_recs) < CLUSTER_MIN_SIZE:
            continue
        # Dominant language vote among cluster members.
        langs = [r.language for r in cluster_recs if r.language]
        dom_lang = max(set(langs), key=langs.count) if langs else "en"
        # Tier-0 summary format: concatenated prefixes of cluster literals,
        # capped at 80 chars each + 5 members -- keeps the summary short and
        # keeps promises clean (summary is NEW content, sources intact).
        summary_text = (
            f"Cluster summary ({len(cluster_recs)} records, lang={dom_lang}): "
            + "; ".join(r.literal_surface[:80] for r in cluster_recs[:5])
        )
        _create_semantic_summary(store, cluster_recs, summary_text, dom_lang)
        summaries_created += 1

        # Hebbian LTP: strengthen existing hebbian edges between co-cluster
        # members. O(k^2) per cluster where k = cluster size.
        pairs_to_boost = list(combinations(cluster_ids, 2))
        if pairs_to_boost:
            store.boost_edges(
                pairs_to_boost,
                delta=HEAVY_LTP_DELTA,
                edge_type="hebbian",
            )
    return summaries_created


def _emit_cls_consolidation_run(
    store: MemoryStore,
    session_id: str,
    *,
    summaries_created: int,
    decay_result: dict,
    schema_candidates: int,
    schemas_induced: int,
    tier: str = "tier0",
    tier_eligible: str = "tier0",
    batch_submitted: bool = False,
) -> None:
    """Extract: write the cls_consolidation_run event.

    Holds the EXACT legacy write_event call from run_heavy_consolidation.
    Both run_heavy_consolidation and the canonical _sleep_pipeline.run call
    this helper — there is exactly ONE implementation of this emit.

    Args:
        store: MemoryStore instance.
        session_id: session identifier for the event (defaults to "system"
            in the canonical pipeline; the legacy path passes the
            daemon-specific session string).
        summaries_created: number of semantic summary records created.
        decay_result: nested dict with keys "decayed" and "pruned".
        schema_candidates: count of schema candidates (presence-only in the
            golden parity gate; the legacy source is
            _tier0_schema_surfacing which differs by construction from the
            canonical tag-pair count).
        schemas_induced: number of auto-status schemas actually persisted
            (legacy semantics — persisted count, NOT candidate count).
        tier: effective tier string (always "tier0" now).
        tier_eligible: tier eligibility string.
        batch_submitted: whether a batch was submitted (always False now).
    """
    write_event(
        store,
        kind="cls_consolidation_run",
        data={
            "mode": "heavy",
            "tier": tier,
            "tier_eligible": tier_eligible,
            "summaries_created": summaries_created,
            "decay_result": decay_result,
            "schema_candidates": schema_candidates,
            "schemas_induced": schemas_induced,
            "batch_submitted": batch_submitted,
        },
        severity="info",
        session_id=session_id,
    )


def run_heavy_consolidation(
    store: MemoryStore,
    session_id: str,
    config: SleepConfig,
    budget: BudgetLedger,
    rate: RateLimitLedger,
    has_api_key: bool = False,
) -> dict:
    """heavy phase -- cluster-find, summarise, decay-sweep, schema-surface.

    The Tier-1 gate is consulted at the top of the function. If
    `should_call_llm` returns False for any reason (llm_enabled=false, no API
    key, budget exceeded, ratelimit cooldown), the entire cycle falls back to
    Tier 0 -- local heuristic summarisation, zero network I/O. Every
    LLM-dependent path must degrade gracefully.

    Returns a dict with:
        mode: "heavy"
        tier: "tier0" | "tier1"
        summaries_created: int
        decay_result: {"decayed": int, "pruned": int}
        schema_candidates: list[dict]
    """
    now = datetime.now(timezone.utc)

    # Step 1: FSRS edge decay sweep (runs regardless of tier).
    decay_result = _decay_edges(store)

    # Step 2: Decide Tier 0 vs Tier 1. This is consulted BEFORE any API call;
    # even if Tier 1 is allowed, 's scope is Tier 0 summarisation
    # only. adds the actual Haiku Batch API call. The gate is here
    # so the event log reflects what WOULD have happened had Tier 1 been
    # implemented.
    llm_ok, _llm_reason = should_call_llm(
        budget=budget,
        rate=rate,
        llm_enabled=config.llm_enabled,
        has_api_key=has_api_key,
    )
    tier = "tier1" if llm_ok else "tier0"
    # The Anthropic Batch-API submit path was removed.
    # The Tier-1 critic now runs inline during REM RECONSOLIDATION via
    # `reconsolidation_critic.evaluate_batch_reconsolidation` (subscription-
    # billed `claude -p` subprocess, capped at 100 records/night). This
    # consolidation path stays Tier-0 from sleep.py's perspective.
    effective_tier = "tier0"
    batch_submitted = False

    # Step 3: cluster-find + summarise + cluster LTP.
    # Single-source: calls _process_cluster_summaries (the canonical
    # _step_cluster_summary calls the same helper; there is exactly ONE
    # implementation of this slice). The ONLY store.all_records call for
    # this path lives inside _process_cluster_summaries.
    # Regression test for the all_records-at-most-once invariant:
    # tests/test_sleep_consolidation_streaming.py
    #::test_run_heavy_consolidation_calls_all_records_at_most_once
    summaries_created = _process_cluster_summaries(store)

    # Step 4: Tier-0 schema candidate surfacing.
    schemas = _tier0_schema_surfacing(store)

    # Step 4b: schema induction batch run.
    # Tier-1 attempts the Haiku path; falls back to tier0.
    # auto-status candidates are persisted (creating schema_instance_of edges).
    # Single-source: calls _persist_tier1_schemas (the canonical _step_schema_mine
    # calls the same helper; there is exactly one implementation of this slice).
    _schema_candidates, schemas_induced = _persist_tier1_schemas(
        store, budget, rate, config.llm_enabled,
    )

    # Single-source: calls _emit_cls_consolidation_run (the canonical
    # _sleep_pipeline.run calls the same helper; there is exactly ONE
    # implementation of this event emit).
    _emit_cls_consolidation_run(
        store,
        session_id,
        summaries_created=summaries_created,
        decay_result=decay_result,
        schema_candidates=len(schemas),
        schemas_induced=schemas_induced,
        tier=effective_tier,
        tier_eligible=tier,
        batch_submitted=batch_submitted,
    )

    return {
        "mode": "heavy",
        "tier": effective_tier,
        "summaries_created": summaries_created,
        "decay_result": decay_result,
        "schema_candidates": schemas,
        "schemas_induced": schemas_induced,
    }
