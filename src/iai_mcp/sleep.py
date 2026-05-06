"""CLS sleep-cycle replay (MEM-07, D-16, D-19, D-29).

Two phases (dual-tier per D-16):

- `run_light_consolidation` -- runs at every session_exit. Pure-local. NO LLM.
  FSRS tick on recently-recalled records. Sub-second. Always on.

- `run_heavy_consolidation` -- runs inside quiet window OR via MANUAL trigger
  (memory_consolidate MCP tool). D-GUARD ladder gates any Tier-1 LLM path via
  `should_call_llm`; Tier-0 fallback is ALWAYS present (TF-IDF + cooccurrence
  summarisation). Creates `consolidated_from` edges linking semantic summary
  records to their source episodes. Runs FSRS edge decay sweep. Logs
  `cls_consolidation_run` event with mode=heavy, tier=tier0|tier1.

D-16 scheduler (`should_run_heavy`):
- ACTIVITY (default): idle>=30min AND local time in quiet_window.
- TIME: strict cron at hour==3 local.
- MANUAL: never fires automatically.
- 48h max defer: if idle >= max_defer_hours, force-run regardless of window.

D-19 decay sweep (`_decay_edges`):
- Only hebbian edges are decayed. contradicts / invariant_anchor /
  consolidated_from / schema_instance_of / temporal_next / curiosity_bridge /
  profile_modulates all survive forever (by design).
- Edges > 90d stale: weight *= 0.9 ** (days - 90); prune if < ε (default 0.01).

D-29 unification: heavy cycle drives FSRS decay + CLS summarisation +
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
    """D-16 trigger mode for heavy consolidation."""

    ACTIVITY = "activity"   # Idle-triggered (default). 30min idle + quiet window.
    TIME = "time"           # Strict cron at hour==3 local.
    MANUAL = "manual"       # Only via memory_consolidate tool.


@dataclass
class SleepConfig:
    """User-configurable sleep-cycle schedule knobs (D-16)."""

    mode: SleepMode = SleepMode.ACTIVITY
    quiet_window: tuple[int, int] = (22, 6)   # local-hour start..end (wrap-around)
    require_idle_minutes: int = 30
    max_defer_hours: int = 48
    on_user_resume: str = "defer_remaining"
    light_on_exit: bool = True
    llm_enabled: bool = False                 # Tier 0 default -- D-GUARD ladder step 1
    llm_tier: int = 1                         # 1=Haiku-Batch, 2=Sonnet/Opus


DECAY_EPSILON: float = 0.01                   # prune threshold
DECAY_GRACE_DAYS: int = 90                    # no decay for edges <=90d old
DECAY_BASE: float = 0.9                       # weight *= 0.9^(days-90)
FSRS_STABILITY_BOOST: float = 0.2             # simple per-recall linear boost
CLUSTER_MIN_SIZE: int = 3                     # CLS cluster threshold
# H-03: Hebbian LTP increment applied to existing edges between
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
    """D-16 trigger evaluator.

    Returns (ok, reason). reason is "" on success, a short diagnostic otherwise.

    The 48h deadline (config.max_defer_hours) overrides MANUAL, TIME, and
    ACTIVITY path-gates -- if the user has ignored the brain for 48h, we MUST
    consolidate before the next session starts. This is a cybernetic S4
    viability requirement (Beer VSM + Ashby ultrastability).
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
    2022) with per-difficulty retrievability modelling is Phase 3.
    """
    if record.never_decay:
        return record
    record.stability = min(1.0, record.stability + FSRS_STABILITY_BOOST)
    record.last_reviewed = now
    return record


def _decay_edges(
    store: MemoryStore, epsilon: float = DECAY_EPSILON,
) -> dict:
    """D-19 nightly sweep: decay stale hebbian + hebbian_structure edges, prune below e.

    CONN-05 D-TEM-04 extension: structure-edge LTP from
    hebbian_structure.strengthen_structure_edge decays under the SAME formula
    and grace period as content-edge hebbian (constitutional contract: FSRS
    decay on structure edges is IDENTICAL to record-edge decay).

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

    # include hebbian_structure in the sweep with identical formula.
    decayable_kinds = ("hebbian", "hebbian_structure")
    hebbian = df[df["edge_type"].isin(decayable_kinds)]
    for _, row in hebbian.iterrows():
        # CR-01: per-row try/except ValueError so one poisoned row
        # cannot kill the entire sweep. _uuid_literal raises ValueError on any
        # non-RFC-4122 UUID string, preventing SQL predicate injection via a
        # corrupt or adversarial `src`/`dst` value.
        try:
            last = row["updated_at"]
            if last is None:
                continue
            # Coerce naive -> UTC; pandas may drop tz on some backends.
            try:
                py = last.to_pydatetime() if hasattr(last, "to_pydatetime") else last
            except Exception:
                py = last
            if getattr(py, "tzinfo", None) is None:
                py = py.replace(tzinfo=timezone.utc)

            days = (now - py).total_seconds() / 86400.0
            if days <= DECAY_GRACE_DAYS:
                continue

            new_weight = float(row["weight"]) * (DECAY_BASE ** (days - DECAY_GRACE_DAYS))

            # CR-01 fix: reject non-canonical UUID values BEFORE interpolation.
            src_lit = _uuid_literal(row["src"])
            dst_lit = _uuid_literal(row["dst"])
            edge_kind = str(row["edge_type"])
            if edge_kind not in decayable_kinds:
                # Belt-and-braces: should not happen given the .isin() above.
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
    """D-16 light phase -- always on, pure local, no LLM.

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
                # H-01 fix: persist the FSRS mutation so stability
                # and last_reviewed survive process restart. update_record
                # rewrites only the FSRS-relevant columns -- embedding,
                # provenance, tags etc. are left intact.
                store.update_record(r)
                fsrs_ticked += 1
        except Exception:
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

    Plan 02-03's LEARN-03 schema induction consumes these candidates.

    W3: rewritten on ``store.iter_record_columns(["tags_json"])``.
    No more full-store load + full-record decrypt -- only the ``tags_json`` column
    is read from disk; encrypted columns (literal_surface, provenance_json,
    profile_modulation_gain_json) are NEVER touched on this path. Saves ~16210
    AES-GCM operations + ~14.5 MB literal_surface materialisation + ~2.4 MB
    provenance_json materialisation + ~11.9 MB embedding materialisation per
    invocation on a production-scale store.
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

    summary inherits dominant language of the source cluster.
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
    enforce_language_tagged(summary, detect=False)
    summary.aaak_index = generate_aaak_index(summary)
    store.insert(summary)

    # R3: batch all consolidated_from edges into a single
    # boost_edges call (one merge_insert + one tbl.add at most). Previously
    # this loop emitted N Lance versions on edges.lance for an N-source
    # cluster.
    pairs = [(summary_id, source.id) for source in cluster]
    if pairs:
        store.boost_edges(
            pairs,
            edge_type="consolidated_from",
            delta=1.0,
        )
    return summary_id


def run_heavy_consolidation(
    store: MemoryStore,
    session_id: str,
    config: SleepConfig,
    budget: BudgetLedger,
    rate: RateLimitLedger,
    has_api_key: bool = False,
) -> dict:
    """D-16 heavy phase -- cluster-find, summarise, decay-sweep, schema-surface.

    D-GUARD: the Tier-1 gate is consulted at the top of the function. If
    `should_call_llm` returns False for any reason (llm_enabled=false, no API
    key, budget exceeded, ratelimit cooldown), the entire cycle falls back to
    Tier 0 -- local heuristic summarisation, zero network I/O. This is the
    constitutional guarantee (D-GUARD): every LLM-dependent path
    must degrade gracefully.

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
    # even if Tier 1 is allowed, Plan 02-02's scope is Tier 0 summarisation
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
    # flips the Tier-1 switch by wiring the Batch API. The
    # gate is re-checked inside batch.submit_batch_consolidation so event
    # ordering matches prior plans. Tier-0 fallback remains unchanged.
    effective_tier = "tier0"
    batch_submitted = False
    if llm_ok:
        try:
            from iai_mcp.batch import submit_batch_consolidation

            # Summarise the workload before submission. scope:
            # the real cluster/schema task payload is populated post-hoc by
            # Phase 3; for now we submit placeholder tasks so the D-GUARD
            # side-effects (budget spend + events) fire on the correct path.
            tasks: list[dict] = [
                {
                    "task_id": f"sleep_cycle:{session_id}",
                    "prompt": "CLS consolidation batch",
                    "prompt_tok": 500,
                    "output_tok": 200,
                }
            ]
            ok_batch, _reason_batch, _results = submit_batch_consolidation(
                store, tasks, budget, rate,
                llm_enabled=config.llm_enabled,
            )
            if ok_batch:
                effective_tier = "tier1"
                batch_submitted = True
        except Exception as _exc:
            # Never block the Tier-0 fallback on batch errors.
            effective_tier = "tier0"

    # Step 3: cluster-find + summarise.
    clusters = _build_hebbian_clusters(store)
    # Phase 07.7-04 W4 (D-13/D-14/D-20 + amendment): single-materialisation
    # invariant. After Plan 07.7-03 W3 rewrites _tier0_schema_surfacing on
    # iter_record_columns and Plan 07.7-04 D-26-A/B migrate schema.py
    # induce_schemas_tier0 + persist_schema to iter_record_columns, this is
    # the ONLY all_records() call left inside run_heavy_consolidation. The
    # cluster-lookup primitive choice (switch this site to iter_records or
    # per-id store.get) is DEFERRED to with the rest of W6
    # (D-20 deferred). Regression test:
    #   tests/test_sleep_consolidation_streaming.py
    #   ::test_run_heavy_consolidation_calls_all_records_at_most_once
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

        # H-03: Hebbian LTP -- strengthen existing hebbian edges
        # between co-cluster members. Mirrors the LTD (_decay_edges) side so
        # the graph is not one-sided. Matches Woz 2022 SRS reinforcement on
        # co-retrieval. O(k^2) per cluster where k = cluster size; bounded by
        # the connected-components partition of hebbian adjacency.
        pairs_to_boost = list(combinations(cluster_ids, 2))
        if pairs_to_boost:
            store.boost_edges(
                pairs_to_boost,
                delta=HEAVY_LTP_DELTA,
                edge_type="hebbian",
            )

    # Step 4: Tier-0 schema candidate surfacing.
    schemas = _tier0_schema_surfacing(store)

    # Step 4b (Plan 02-03 LEARN-03 primary): schema induction batch run.
    # Tier-1 attempts the Haiku path via D-GUARD ladder; falls back to tier0.
    # auto-status candidates are persisted (creating schema_instance_of edges).
    schemas_induced = 0
    try:
        from iai_mcp.schema import (
            induce_schemas_tier1,
            persist_schema,
        )

        candidates = induce_schemas_tier1(
            store, budget=budget, rate=rate,
            llm_enabled=config.llm_enabled,
        )
        for cand in candidates:
            if cand.status == "auto":
                persist_schema(store, cand)
                schemas_induced += 1
            # pending_user_approval candidates are only logged (via
            # induce_schemas_tier1's llm_health emission path).
    except Exception as exc:
        write_event(
            store,
            kind="schema_induction_run",
            data={"error": str(exc), "status": "failed"},
            severity="warning",
            session_id=session_id,
        )

    write_event(
        store,
        kind="cls_consolidation_run",
        data={
            "mode": "heavy",
            "tier": effective_tier,
            "tier_eligible": tier,
            "summaries_created": summaries_created,
            "decay_result": decay_result,
            "schema_candidates": len(schemas),
            "schemas_induced": schemas_induced,
            "batch_submitted": batch_submitted,
        },
        severity="info",
        session_id=session_id,
    )

    return {
        "mode": "heavy",
        "tier": effective_tier,
        "summaries_created": summaries_created,
        "decay_result": decay_result,
        "schema_candidates": schemas,
        "schemas_induced": schemas_induced,
    }
