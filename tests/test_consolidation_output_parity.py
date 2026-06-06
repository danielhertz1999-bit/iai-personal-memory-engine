"""GOLDEN ARTIFACT-PARITY GATE — legacy run_heavy_consolidation vs canonical
matched steps produce IDENTICAL store-state on isolated tmp stores.

After extraction into shared helpers (single-source), 'parity' really means
'the canonical pipeline wires up and emits every legacy output via the same
helpers, with no double-run of decay or schema.' The assertions here prove:

  L1  decay/prune : matched post-run hebbian edge weights and prune set
  L2+L3 cluster  : count of NEW semantic records + consolidated_from edges
  L4  cluster LTP: hebbian weights between co-cluster members boosted equally
  L6  schema     : auto-status schemas persisted, schema_instance_of edges
  L7  cls event  : cls_consolidation_run payload value-asserted on
                   {mode, tier, summaries_created, decay_result, schemas_induced};
                   schema_candidates / tier_eligible / batch_submitted
                   are PRESENCE-ONLY (key exists) because schema_candidates'
                   legacy source (_tier0_schema_surfacing single-tag) differs
                   from the canonical tag-pair count by construction.

Matched-step ordering (LOAD-BEARING for L1/L4 parity — decay BEFORE cluster):
  DREAM_DECAY -> CLUSTER_SUMMARY -> SCHEMA_MINE

This test is the gate authorizing removal of the legacy _tick_body Steps 5-7.
It runs a MATCHED operation set (exactly the three legacy-equivalent steps),
NOT a full _sleep_pipeline.run (which includes 9 extra steps that would mutate
the inspected edges and break exact equality).

PART 3 runs a full pipeline separately to prove CLUSTER_SUMMARY is wired.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from iai_mcp.store import EDGES_TABLE, MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# ---------------------------------------------------------------------------
# Keyring isolation — same discipline as test_sleep_consolidation_streaming.py
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p),
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None),
    )
    yield fake


# ---------------------------------------------------------------------------
# Record factory (no PII; generic test data)
# ---------------------------------------------------------------------------


def _rec(
    text: str,
    tags: list[str] | None = None,
    tier: str = "episodic",
    language: str = "en",
    detail: int = 2,
    created_days_ago: int = 100,
) -> MemoryRecord:
    """Minimal MemoryRecord for parity-gate seeding."""
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    created = now - timedelta(days=created_days_ago)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=detail,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=(detail >= 3),
        never_merge=False,
        provenance=[],
        created_at=created,
        updated_at=created,
        tags=tags if tags is not None else [],
        language=language,
    )


# ---------------------------------------------------------------------------
# Deterministic seed helper
# ---------------------------------------------------------------------------


def _seed_store(store: MemoryStore) -> dict[str, Any]:
    """Populate store with records + edges that exercise every output class.

    Returns metadata: list of record ids inserted, cluster member ids,
    so the caller can verify edge/record parity without guessing.

    Design constraints:
    - At least one hebbian connected component of size >= CLUSTER_MIN_SIZE (3).
    - Tag co-occurrence that yields at least one status=="auto" schema candidate
      (evidence_count >= 5 for "auto" in induce_schemas_tier0).
    - Hebbian edges older than DECAY_GRACE_DAYS (90d) so decay/prune fires on L1.
    - Plasticity_gain=1.0 assumed (no user_model override).
    """
    from iai_mcp.sleep import CLUSTER_MIN_SIZE

    # --- cluster: 4 records forming a connected component ---
    # Use a shared tag PAIR repeated on >=9 records for schema auto-status.
    # induce_schemas_tier0 auto threshold: count >= 5 AND confidence >= 0.85
    # confidence = min(1.0, count/10.0) → need count >= 9 (confidence=0.9 >= 0.85).
    schema_tag_a = "topic-alpha"
    schema_tag_b = "topic-beta"
    cluster_recs = []
    for i in range(4):
        r = _rec(
            text=f"cluster record {i}",
            # Both schema tags + an extra for variety; the pair (a,b) will co-occur.
            tags=[schema_tag_a, schema_tag_b, f"extra-{i}"],
            created_days_ago=110,
        )
        store.insert(r)
        cluster_recs.append(r)

    # Extra records with the SAME pair (total 10 records with the pair for count=10).
    extra_recs = []
    for i in range(6):
        r = _rec(
            text=f"extra record {i}",
            tags=[schema_tag_a, schema_tag_b],
            created_days_ago=50,
        )
        store.insert(r)
        extra_recs.append(r)

    # Pair: another separate record (non-cluster, no schema tags).
    lone_rec = _rec(text="lone record", tags=["other"])
    store.insert(lone_rec)

    # Hebbian triangle + one more edge to ensure the 4 cluster records
    # are fully connected (size == 4 >= CLUSTER_MIN_SIZE==3).
    ids = [r.id for r in cluster_recs]
    store.boost_edges(
        [
            (ids[0], ids[1]),
            (ids[1], ids[2]),
            (ids[2], ids[3]),
            (ids[0], ids[3]),
        ],
        edge_type="hebbian",
        # weight=0.5 default; make old enough to be decay-eligible (>90d).
        delta=0.5,
    )
    # Manually age the edges to > 90 days so _decay_edges fires on them.
    # We do this by directly updating the updated_at timestamps.
    from iai_mcp.store import EDGES_TABLE
    from iai_mcp.hippo import HippoIntegrityError

    # Fixed backdate (not datetime.now) so the edge age is IDENTICAL for both
    # stores; combined with a pinned decay clock this makes the decay exponent
    # deterministic and the weight parity exact regardless of wall-clock load.
    old_ts = datetime(2025, 9, 28, tzinfo=timezone.utc).isoformat()
    tbl = store.db.open_table(EDGES_TABLE)
    for src_id, dst_id in [
        (ids[0], ids[1]),
        (ids[1], ids[2]),
        (ids[2], ids[3]),
        (ids[0], ids[3]),
    ]:
        try:
            tbl.update(
                where=f"edge_type='hebbian' AND ("
                      f"(src='{src_id}' AND dst='{dst_id}') OR "
                      f"(src='{dst_id}' AND dst='{src_id}')"
                      f")",
                values={"updated_at": old_ts},
            )
        except Exception:
            pass  # best-effort timestamp backdate

    return {
        "cluster_ids": ids,
        "extra_ids": [r.id for r in extra_recs],
        "lone_id": lone_rec.id,
        "schema_tag_a": schema_tag_a,
        "schema_tag_b": schema_tag_b,
    }


def _make_store(tmp_path: Path, suffix: str = "") -> MemoryStore:
    return MemoryStore(path=tmp_path / f"store{suffix}")


def _run_heavy(store: MemoryStore) -> dict:
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

    return run_heavy_consolidation(
        store,
        session_id="test-session",
        config=SleepConfig(llm_enabled=False),
        budget=BudgetLedger(store),
        rate=RateLimitLedger(store),
        has_api_key=False,
    )


def _make_pipeline(store: MemoryStore, tmp_path: Path, suffix: str = ""):
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.sleep_pipeline import SleepPipeline

    return SleepPipeline(
        store=store,
        lifecycle_state_path=tmp_path / f"lifecycle{suffix}.json",
        event_log=LifecycleEventLog(log_dir=tmp_path / f"logs{suffix}"),
    )


def _get_edges(store: MemoryStore, edge_type: str) -> list[dict]:
    tbl = store.db.open_table(EDGES_TABLE)
    df = tbl.to_pandas()
    if df.empty:
        return []
    return df[df["edge_type"] == edge_type].to_dict("records")


def _get_events(store: MemoryStore, kind: str) -> list[dict]:
    """Query events by kind via the canonical query_events API (handles AES-GCM decrypt)."""
    from iai_mcp.events import query_events
    try:
        rows = query_events(store, kind=kind, limit=100)
        return [{"kind": r["kind"], "data": r.get("data", {})} for r in rows]
    except Exception:
        return []


def _get_records_by_tier(store: MemoryStore, tier: str) -> list[dict]:
    recs = store.all_records()
    return [r for r in recs if r.tier == tier]


# ---------------------------------------------------------------------------
# PART 1: VALUE-PARITY — matched steps only (NOT a full pipeline run)
# ---------------------------------------------------------------------------


def test_parity_matched_steps_store_state(tmp_path, monkeypatch):
    """PART 1: Legacy run_heavy_consolidation and matched canonical steps
    (dream_decay -> cluster_summary -> schema_mine) produce IDENTICAL store
    state on isolated identically-seeded tmp stores.

    Order is LOAD-BEARING (decay-before-cluster for L1/L4 weight parity).
    """
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep_pipeline import SleepStep

    # Pin the decay clock so both runs see the SAME `now`. Both legacy and
    # canonical decay go through iai_mcp.sleep._decay_edges, which reads
    # datetime.now() via the module attribute; with a fixed instant and a fixed
    # edge backdate, the decay exponent is identical -> exact weight parity,
    # independent of the wall-clock gap between the two runs (which previously
    # leaked ~microdays into the exponent under load).
    import datetime as _dt

    class _FixedDT(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)

    monkeypatch.setattr("iai_mcp.sleep.datetime", _FixedDT)

    # Seed two identical stores.
    store_a = _make_store(tmp_path, "a")
    store_b = _make_store(tmp_path, "b")
    _seed_store(store_a)
    _seed_store(store_b)

    # --- Store A: legacy ---
    _run_heavy(store_a)

    # --- Store B: matched canonical steps in legacy order ---
    pipeline_b = _make_pipeline(store_b, tmp_path, "b")

    # Patch steps that we do NOT want to run (they would mutate beyond legacy scope).
    # We run only DREAM_DECAY, CLUSTER_SUMMARY, SCHEMA_MINE in legacy order.
    # All other steps are nooped.
    _noop = lambda ic: (True, {})
    _step_methods_backup = pipeline_b._step_methods  # read-only; we override per-step

    payloads_b: dict[SleepStep, dict] = {}
    for step in [SleepStep.DREAM_DECAY, SleepStep.CLUSTER_SUMMARY, SleepStep.SCHEMA_MINE]:
        done, payload = getattr(pipeline_b, f"_step_{step.name.lower()}")(None)
        assert done, f"step {step.name} returned done=False unexpectedly"
        payloads_b[step] = payload

    # --- L1: decay/prune parity ---
    # Two identically-seeded stores get different record UUIDs (each uuid4() call
    # is independent). Therefore we compare STRUCTURAL PROPERTIES (count + sorted
    # weight distribution) rather than exact (src, dst) UUID key pairs.
    edges_a_hebb = _get_edges(store_a, "hebbian")
    edges_b_hebb = _get_edges(store_b, "hebbian")

    weights_a = sorted(float(r["weight"]) for r in edges_a_hebb)
    weights_b = sorted(float(r["weight"]) for r in edges_b_hebb)

    assert len(weights_a) == len(weights_b), (
        f"L1 hebbian edge count: legacy={len(weights_a)}, canonical={len(weights_b)}"
    )
    for i, (wa, wb) in enumerate(zip(weights_a, weights_b)):
        assert abs(wa - wb) < 1e-6, (
            f"L1 hebbian weight mismatch at position {i}: legacy={wa}, canonical={wb}"
        )

    # --- L2+L3: cluster semantic records + consolidated_from edges ---
    sem_a = _get_records_by_tier(store_a, "semantic")
    sem_b = _get_records_by_tier(store_b, "semantic")

    # Filter to cls_summary tagged records (the ones created by cluster summarisation).
    cls_sum_a = [r for r in sem_a if "cls_summary" in (r.tags or [])]
    cls_sum_b = [r for r in sem_b if "cls_summary" in (r.tags or [])]
    assert len(cls_sum_a) == len(cls_sum_b), (
        f"L2 cluster semantic record count: legacy={len(cls_sum_a)}, canonical={len(cls_sum_b)}"
    )

    cf_a = _get_edges(store_a, "consolidated_from")
    cf_b = _get_edges(store_b, "consolidated_from")
    assert len(cf_a) == len(cf_b), (
        f"L3 consolidated_from edge count: legacy={len(cf_a)}, canonical={len(cf_b)}"
    )

    # --- L4: cluster LTP — hebbian edges between cluster members ---
    # The seed planted 4 cluster records with 4 hebbian edges.
    # After LTP both stores should have equal weights on those pairs.
    # (Already covered by the L1 key+weight check, but make explicit.)
    assert len(cls_sum_a) >= 1, "Expected at least one cluster summary to exercise L4"

    # --- L6: schema persistence ---
    si_a = _get_edges(store_a, "schema_instance_of")
    si_b = _get_edges(store_b, "schema_instance_of")
    # Discriminating check: the seed must yield at least one auto-status schema.
    # Without this, L6 parity is vacuous (0==0 proves nothing).
    assert len(si_a) >= 1, (
        f"Seed does not exercise L6 — schema_instance_of edge count = 0 in store A. "
        f"Adjust seed so the tag pair (topic-alpha, topic-beta) co-occurs on >=9 records "
        f"(auto threshold: count>=5 AND confidence=count/10>=0.85 -> need count>=9)."
    )
    assert len(si_b) >= 1, (
        f"Seed does not exercise L6 — schema_instance_of edge count = 0 in store B."
    )
    assert len(si_a) == len(si_b), (
        f"L6 schema_instance_of edge count: legacy={len(si_a)}, canonical={len(si_b)}"
    )


# ---------------------------------------------------------------------------
# PART 2: CLS-EVENT PARITY
# ---------------------------------------------------------------------------


def test_parity_cls_event(tmp_path):
    """PART 2: cls_consolidation_run parity via manually collected step payloads.

    PART 1 runs steps directly (bypassing _sleep_pipeline.run), so store B
    has no cls event yet — the event fires only at pipeline clean completion.
    We collect the step payloads from PART 1-style direct calls and invoke
    _emit_cls_consolidation_run manually on store B.

    VALUE-ASSERTED keys: {mode, tier, summaries_created, decay_result (nested),
    schemas_induced (persisted count)}.
    PRESENCE-ONLY keys: schema_candidates, tier_eligible, batch_submitted.
    """
    from iai_mcp.sleep import _emit_cls_consolidation_run
    from iai_mcp.sleep_pipeline import SleepStep

    store_a = _make_store(tmp_path, "a2")
    store_b = _make_store(tmp_path, "b2")
    _seed_store(store_a)
    _seed_store(store_b)

    # --- Store A: legacy (emits cls event internally) ---
    _run_heavy(store_a)

    # --- Store B: matched steps + manual emit ---
    pipeline_b = _make_pipeline(store_b, tmp_path, "b2")

    # Run matched steps in legacy order, collect payloads.
    done_decay, payload_decay = pipeline_b._step_dream_decay(None)
    done_cluster, payload_cluster = pipeline_b._step_cluster_summary(None)
    done_schema, payload_schema = pipeline_b._step_schema_mine(None)

    assert done_decay and done_cluster and done_schema

    # Reconstruct nested decay_result from DREAM_DECAY's flat payload.
    decay_result_b = {
        "decayed": int(payload_decay.get("decayed", 0)),
        "pruned": int(payload_decay.get("pruned", 0)),
    }
    summaries_created_b = int(payload_cluster.get("summaries_created", 0))
    # schemas_induced = PERSISTED count (schemas_persisted key from our extended step).
    schemas_induced_b = int(payload_schema.get("schemas_persisted", 0))
    schema_candidates_b = int(payload_schema.get("schemas_induced", 0))

    # Manually emit cls event on store B (same helper the pipeline uses).
    _emit_cls_consolidation_run(
        store_b,
        "test-session",
        summaries_created=summaries_created_b,
        decay_result=decay_result_b,
        schema_candidates=schema_candidates_b,
        schemas_induced=schemas_induced_b,
    )

    # --- Fetch cls events from both stores ---
    cls_a = _get_events(store_a, "cls_consolidation_run")
    cls_b = _get_events(store_b, "cls_consolidation_run")

    # Filter to "heavy" mode only.
    cls_a_heavy = [e for e in cls_a if e["data"].get("mode") == "heavy"]
    cls_b_heavy = [e for e in cls_b if e["data"].get("mode") == "heavy"]

    assert len(cls_a_heavy) >= 1, "Legacy store A missing cls_consolidation_run event"
    assert len(cls_b_heavy) >= 1, "Canonical store B missing cls_consolidation_run event"

    data_a = cls_a_heavy[-1]["data"]
    data_b = cls_b_heavy[-1]["data"]

    # VALUE-ASSERTED keys.
    for key in ["mode", "tier"]:
        assert data_a[key] == data_b[key], (
            f"cls key {key!r} mismatch: legacy={data_a[key]!r}, canonical={data_b[key]!r}"
        )
    assert data_a["summaries_created"] == data_b["summaries_created"], (
        f"cls summaries_created: legacy={data_a['summaries_created']}, "
        f"canonical={data_b['summaries_created']}"
    )
    # decay_result is nested in both; compare fields.
    dr_a = data_a.get("decay_result", {})
    dr_b = data_b.get("decay_result", {})
    if isinstance(dr_a, str):
        dr_a = json.loads(dr_a)
    if isinstance(dr_b, str):
        dr_b = json.loads(dr_b)
    assert int(dr_a.get("decayed", -1)) == int(dr_b.get("decayed", -1)), (
        f"cls decay_result.decayed: legacy={dr_a}, canonical={dr_b}"
    )
    assert int(dr_a.get("pruned", -1)) == int(dr_b.get("pruned", -1)), (
        f"cls decay_result.pruned: legacy={dr_a}, canonical={dr_b}"
    )
    assert data_a["schemas_induced"] == data_b["schemas_induced"], (
        f"cls schemas_induced: legacy={data_a['schemas_induced']}, "
        f"canonical={data_b['schemas_induced']}"
    )

    # PRESENCE-ONLY keys: must exist but values may differ.
    for key in ["schema_candidates", "tier_eligible", "batch_submitted"]:
        assert key in data_a, f"Legacy cls event missing presence-only key {key!r}"
        assert key in data_b, f"Canonical cls event missing presence-only key {key!r}"


# ---------------------------------------------------------------------------
# PART 3: Full pipeline run — CLUSTER_SUMMARY in completed_steps
# ---------------------------------------------------------------------------


def test_full_pipeline_includes_cluster_summary_step(tmp_path, monkeypatch):
    """PART 3: A full _sleep_pipeline.run includes CLUSTER_SUMMARY in
    completed_steps AND emits a cls_consolidation_run event with the
    value-asserted key set.

    This is NOT a parity comparison against legacy — it verifies CLUSTER_SUMMARY
    is wired into _STEP_ORDER and the pipeline runs it end-to-end.

    Steps that are irrelevant to this check and expensive/complex (reconsolidation
    critic, DMN reflection, erasure) are patched to no-ops so the test is fast
    and hermetic. CLUSTER_SUMMARY and the cls event are NOT patched.
    """
    from iai_mcp.sleep_pipeline import SleepPipeline, SleepStep

    store = _make_store(tmp_path, "full")
    _seed_store(store)
    pipeline = _make_pipeline(store, tmp_path, "full")

    # Patch the expensive/non-deterministic steps to no-ops.
    # We keep DREAM_DECAY, CLUSTER_SUMMARY, SCHEMA_MINE live (they are the
    # ones we want to verify). The rest are patched.
    _noop = lambda ic: (True, {})
    steps_to_noop = [
        SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
        SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
    ]
    for step in steps_to_noop:
        method_name = f"_step_{step.name.lower()}"
        monkeypatch.setattr(pipeline, method_name, _noop)

    result = pipeline.run()

    assert not result.get("interrupted"), "pipeline run was unexpectedly interrupted"
    assert result.get("failed_step") is None, (
        f"pipeline failed on step {result.get('failed_step')}: {result.get('error')}"
    )
    assert SleepStep.CLUSTER_SUMMARY in result["completed_steps"], (
        f"CLUSTER_SUMMARY missing from completed_steps: {result['completed_steps']}"
    )
    assert SleepStep.DREAM_DECAY in result["completed_steps"]
    assert SleepStep.SCHEMA_MINE in result["completed_steps"]

    # Assert cls_consolidation_run event was emitted by the pipeline.
    cls_events = _get_events(store, "cls_consolidation_run")
    heavy_cls = [e for e in cls_events if e["data"].get("mode") == "heavy"]
    assert len(heavy_cls) >= 1, (
        f"Full pipeline run did not emit cls_consolidation_run; "
        f"all cls events: {cls_events}"
    )

    data = heavy_cls[-1]["data"]
    for key in ["mode", "tier", "summaries_created", "decay_result", "schemas_induced"]:
        assert key in data, (
            f"cls_consolidation_run missing value-asserted key {key!r}: {data}"
        )
    for key in ["schema_candidates", "tier_eligible", "batch_submitted"]:
        assert key in data, (
            f"cls_consolidation_run missing presence-only key {key!r}: {data}"
        )
