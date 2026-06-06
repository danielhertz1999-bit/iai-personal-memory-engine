"""Regression tests for schema-bypass + memory-reconsolidation.

The tests pin the following acceptance contracts:

    schema_bypass column exists with default False; cosine probe
        toggles it to True only when max-cos >= threshold.
    labile_until column nullable by default; reinforce_record's new
        is_retrieval=True kwarg stamps it to now + LABILE_WINDOW_SEC;
        the default is_retrieval=False path leaves it untouched.
    SleepStep.RECONSOLIDATION=9 in REM phase between CLUSTER_REPLAY
        and CRISIS_RECLUSTER; _step_reconsolidation emits a single
        reconsolidation_pass event with the 4 keys (records_scanned,
        records_reconsolidated, critic_calls, dry_run_mode);
        cfg.reconsolidation_tier1=False short-circuits the critic loop.
    schema_bypass tagging never runs on the pattern_separation SKIP
        branch (only on GateAction.INSERT).
    Every one of the 5 IAI_MCP_RECONSOLIDATION_* env vars (and the
        SCHEMA_BYPASS_COS_THRESHOLD + LABILE_WINDOW_SEC siblings) fails
        loud with a ValueError naming the offending var.
    cfg.dry_run=True preserves no-mutation invariants on both the
        insert-side schema-bypass tagging AND the REM reconsolidation
        provenance/FSRS re-anchor — events still fire, no row mutates.

Two breadth-extra tests pin the critic surface:

    - PROMPT_TEMPLATE slot contract (verbatim template).
    - call_critic Tier-0 fallback returns 0.0 when the gate denies.

Synthetic stores use tmp_path with user_id='alice'. Critic is always
monkeypatched -- no live Anthropic SDK call from any test.
"""
# Standard-library imports first so optional iai_mcp.* imports fail loud
# with a clear ImportError if the package layout changes.
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iai_mcp.daemon import (
    ReconsolidationConfig,
    _load_reconsolidation_config,
)
from iai_mcp.events import query_events
from iai_mcp.lifecycle_state import default_state, save_state
from iai_mcp.reconsolidation_critic import PROMPT_TEMPLATE, call_critic
from iai_mcp.sleep_pipeline import (
    STEP_PHASE,
    SleepPhase,
    SleepPipeline,
    SleepStep,
)
from iai_mcp.store import RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Autouse fixture: pin IAI_MCP_STORE to tmp_path so per-test MemoryStore
# construction stays isolated from the user's real store, AND wipe
# every IAI_MCP_* env var reads so each test starts from
# defaults. Tests that need a specific value re-set after this fixture.
@pytest.fixture(autouse=True)
def _isolate_iai_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    for var in (
        "IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD",
        "IAI_MCP_LABILE_WINDOW_SEC",
        "IAI_MCP_RECONSOLIDATION_TIER1",
        "IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD",
        "IAI_MCP_RECONSOLIDATION_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)


# Build a minimal MemoryRecord. Embedding defaults to a tiny non-zero
# fill so the structure_hv autopop in MemoryStore.insert succeeds (a
# zero vector would also work but the non-zero default matches the
# precedent in the other sleep-pipeline tests).
def _make_record(
    *,
    embed_dim: int,
    literal: str = "alice prefers tea over coffee",
    embedding: list[float] | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=literal,
        aaak_index="",
        embedding=embedding if embedding is not None else [0.01] * embed_dim,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["t"],
    )


def _make_store(tmp_path: Path) -> MemoryStore:
    """Build a per-test MemoryStore rooted at tmp_path."""
    return MemoryStore(
        path=str(tmp_path / "iai-mcp-store"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )


# Stub the runtime_graph_cache.try_load surface so insert-time
# _maybe_tag_schema_bypass sees a controlled centroids dict.
# The helper imports runtime_graph_cache LAZILY inside the
# method body, so patching the module attribute is
# enough -- no re-import needed inside the test.
def _stub_centroids(
    monkeypatch: pytest.MonkeyPatch,
    centroids: dict[uuid.UUID, list[float]],
) -> None:
    """Patch try_load to yield (assignment, rich_club, node_payload, max_degree)
    with assignment.community_centroids = centroids."""
    assignment = SimpleNamespace(community_centroids=centroids)
    rich_club: list = []
    node_payload: dict = {}
    max_degree: int = 0
    fake = (assignment, rich_club, node_payload, max_degree)

    def _fake_try_load(store: Any) -> tuple:
        return fake

    monkeypatch.setattr(
        "iai_mcp.runtime_graph_cache.try_load", _fake_try_load,
    )


# ---------------------------------------------------------------------------
# Test 1: schema_bypass column exists + default False on plain insert
# ---------------------------------------------------------------------------


def test_R1_schema_bypass_column_exists_and_default_false(
    tmp_path: Path,
) -> None:
    """schema_bypass column is `bool`, defaults to False
    on a plain insert with empty centroids."""
    store = _make_store(tmp_path)
    tbl = store.db.open_table(RECORDS_TABLE)
    assert "schema_bypass" in tbl.schema.names
    field = tbl.schema.field("schema_bypass")
    # Lance schema may report bool() or bool_(); both expose.equals(pa.bool_())
    import pyarrow as pa
    assert field.type.equals(pa.bool_()), (
        f"schema_bypass column type must be bool, got {field.type}"
    )

    # Plain insert -- no centroids monkeypatch, cold cache -> empty dict
    # -> max_cos=0.0 -> tagged=False. Record persists with schema_bypass=False.
    rec = _make_record(embed_dim=store._embed_dim)
    store.insert(rec)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert bool(row["schema_bypass"]) is False


# ---------------------------------------------------------------------------
# Test 2: cosine probe toggles schema_bypass to True at/above threshold
# ---------------------------------------------------------------------------


def test_R1_schema_bypass_true_when_cosine_meets_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a populated centroid + a same-direction embedding the
    cos = 1.0 >= 0.85 path tags schema_bypass=True. An orthogonal second
    insert stays at False -- proves the probe actually computes."""
    # Force live mutation; pytest defaults dry_run=True.
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD", "0.85")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim

    # Unit centroid along axis 0: cos == 1.0 vs a same-direction embedding.
    centroid_axis0 = [1.0] + [0.0] * (embed_dim - 1)
    _stub_centroids(monkeypatch, {uuid.uuid4(): centroid_axis0})

    rec_aligned = _make_record(
        embed_dim=embed_dim,
        literal="aligned with centroid",
        embedding=list(centroid_axis0),
    )
    store.insert(rec_aligned)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row_a = df[df["id"] == str(rec_aligned.id)].iloc[0]
    assert bool(row_a["schema_bypass"]) is True, (
        "aligned-to-centroid insert must tag schema_bypass=True"
    )

    # Orthogonal embedding along axis 1: cos == 0.0 < 0.85 -> stays False.
    orthogonal = [0.0, 1.0] + [0.0] * (embed_dim - 2)
    rec_far = _make_record(
        embed_dim=embed_dim,
        literal="orthogonal to centroid",
        embedding=orthogonal,
    )
    store.insert(rec_far)
    df = tbl.to_pandas()
    row_b = df[df["id"] == str(rec_far.id)].iloc[0]
    assert bool(row_b["schema_bypass"]) is False, (
        "orthogonal insert must leave schema_bypass=False"
    )


# ---------------------------------------------------------------------------
# Test 3: labile_until stamped only on is_retrieval=True path
# ---------------------------------------------------------------------------


def test_R2_labile_until_set_by_reinforce_record_is_retrieval_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reinforce_record(rid, is_retrieval=True) writes
    labile_until = now + LABILE_WINDOW_SEC; default kwarg leaves it null
    AND a second is_retrieval=True call does not (in this test) clobber
    the prior stamp arbitrarily -- a second stamp is within the same
    window so we assert the timestamp stays inside the slop window."""
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_LABILE_WINDOW_SEC", "3600")

    store = _make_store(tmp_path)
    rec = _make_record(embed_dim=store._embed_dim)
    store.insert(rec)
    tbl = store.db.open_table(RECORDS_TABLE)

    # Pre-condition: labile_until null on a fresh insert.
    df = tbl.to_pandas()
    row_pre = df[df["id"] == str(rec.id)].iloc[0]
    assert row_pre["labile_until"] is None or (
        # pandas may yield NaT for nullable timestamp columns.
        hasattr(row_pre["labile_until"], "to_pydatetime")
        and str(row_pre["labile_until"]) == "NaT"
    ), f"fresh insert labile_until must be null/NaT, got {row_pre['labile_until']!r}"

    # is_retrieval=False (default) -- no labile stamp.
    store.reinforce_record(rec.id)
    df = tbl.to_pandas()
    row_default = df[df["id"] == str(rec.id)].iloc[0]
    val_default = row_default["labile_until"]
    is_null_default = val_default is None or str(val_default) == "NaT"
    assert is_null_default, (
        f"default reinforce_record must NOT stamp labile_until, "
        f"got {val_default!r}"
    )

    # is_retrieval=True -- labile_until stamped to now + 3600s.
    before = datetime.now(timezone.utc)
    store.reinforce_record(rec.id, is_retrieval=True)
    after = datetime.now(timezone.utc)
    df = tbl.to_pandas()
    row_post = df[df["id"] == str(rec.id)].iloc[0]
    stamped = row_post["labile_until"]
    # Normalise: pandas Timestamp, string, or datetime -> tz-aware datetime.
    if hasattr(stamped, "to_pydatetime"):
        stamped = stamped.to_pydatetime()
    if isinstance(stamped, str):
        # HippoDB returns ISO-8601 strings from TEXT columns.
        stamped = datetime.fromisoformat(stamped.replace("+00:00", "").rstrip("Z"))
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=timezone.utc)
    expected_low = before + timedelta(seconds=3600) - timedelta(seconds=10)
    expected_high = after + timedelta(seconds=3600) + timedelta(seconds=10)
    assert expected_low <= stamped <= expected_high, (
        f"labile_until={stamped} not in [{expected_low}, {expected_high}]"
    )


# ---------------------------------------------------------------------------
# Test 4: SleepStep enum + STEP_PHASE + _STEP_ORDER position
# ---------------------------------------------------------------------------


def test_R3_reconsolidation_step_in_enum_and_order() -> None:
    """Enum + order contract: RECONSOLIDATION=9; REM phase; placed
    between CLUSTER_REPLAY and CRISIS_RECLUSTER (inside the REM tail).

    USER_MODEL_UPDATE=10 sits between RECONSOLIDATION and CRISIS_RECLUSTER.
    The invariant "RECONSOLIDATION follows CLUSTER_REPLAY" still holds;
    RECONSOLIDATION now precedes USER_MODEL_UPDATE (which itself precedes
    CRISIS_RECLUSTER). The looser invariant "RECONSOLIDATION is inside the
    REM tail (after CLUSTER_REPLAY, before CRISIS_RECLUSTER)" remains.
    """
    assert SleepStep.RECONSOLIDATION.value == 9
    assert STEP_PHASE[SleepStep.RECONSOLIDATION] == SleepPhase.REM
    order = SleepPipeline._STEP_ORDER
    idx_recon = order.index(SleepStep.RECONSOLIDATION)
    idx_cluster = order.index(SleepStep.CLUSTER_REPLAY)
    idx_crisis = order.index(SleepStep.CRISIS_RECLUSTER)
    assert idx_recon == idx_cluster + 1, (
        f"RECONSOLIDATION must follow CLUSTER_REPLAY; "
        f"got idx_recon={idx_recon}, idx_cluster={idx_cluster}"
    )
    # USER_MODEL_UPDATE sits between RECONSOLIDATION and CRISIS_RECLUSTER,
    # so the invariant is the loose ordering: RECONSOLIDATION sits strictly
    # between CLUSTER_REPLAY and CRISIS_RECLUSTER inside the REM tail.
    assert idx_recon < idx_crisis, (
        f"RECONSOLIDATION must precede CRISIS_RECLUSTER; "
        f"got idx_recon={idx_recon}, idx_crisis={idx_crisis}"
    )


# ---------------------------------------------------------------------------
# Test 5: _step_reconsolidation emits reconsolidation_pass with 4 keys
# ---------------------------------------------------------------------------


def test_R3_step_body_emits_reconsolidation_pass_event_with_correct_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a labile record + Tier-1 ON + monkeypatched critic returning
    0.9 (>= default 0.5 threshold), _step_reconsolidation:
        - returns (True, payload) with records_scanned >= 1, records_reconsolidated >= 1
        - emits exactly one reconsolidation_pass event
        - event body has the 4 expected keys with correct types"""
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_TIER1", "true")
    monkeypatch.setenv("IAI_MCP_LABILE_WINDOW_SEC", "3600")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD", "0.5")

    store = _make_store(tmp_path)
    rec = _make_record(embed_dim=store._embed_dim, literal="alice loves haiku")
    store.insert(rec)

    # Stamp labile_until so the WHERE pushdown picks the record up.
    store.reinforce_record(rec.id, is_retrieval=True)

    # Critic stub: return high prediction error for every record in the
    # batch so the threshold gate fires.: monkeypatch target is
    # the batched evaluate_batch_reconsolidation (sleep_pipeline no longer
    # calls the legacy per-record call_critic surface).
    def _stub_batched(items: Any, *args: Any, **kwargs: Any) -> dict:
        return {rid: 0.9 for rid, _surface in items}

    monkeypatch.setattr(
        "iai_mcp.reconsolidation_critic.evaluate_batch_reconsolidation",
        _stub_batched,
    )

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    done, payload = pipeline._step_reconsolidation(interrupt_check=None)
    assert done is True
    assert payload["records_scanned"] >= 1
    assert payload["records_reconsolidated"] >= 1
    assert payload["dry_run"] is False

    events = query_events(store, kind="reconsolidation_pass", limit=10)
    assert len(events) == 1, (
        f"_step_reconsolidation must emit exactly one event, got {len(events)}"
    )
    body = events[0]["data"]
    expected_keys = {
        "records_scanned",
        "records_reconsolidated",
        "critic_calls",
        "dry_run_mode",
    }
    assert set(body.keys()) == expected_keys, (
        f"event body keys must be {expected_keys}, got {set(body.keys())}"
    )
    assert isinstance(body["records_scanned"], int)
    assert isinstance(body["records_reconsolidated"], int)
    assert isinstance(body["critic_calls"], int)
    assert isinstance(body["dry_run_mode"], bool)
    assert body["records_scanned"] >= 1
    assert body["records_reconsolidated"] >= 1
    assert body["critic_calls"] >= 1
    assert body["dry_run_mode"] is False


# ---------------------------------------------------------------------------
# Test 6: Tier-1 OFF skips the critic loop entirely
# ---------------------------------------------------------------------------


def test_R3_tier1_false_skips_critic_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cfg.reconsolidation_tier1=False skips the per-record
    critic loop. records_scanned still reflects the labile inventory, but
    critic_calls=0 and records_reconsolidated=0. Monkeypatched critic that
    raises on entry proves call_critic was never invoked."""
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_TIER1", "false")
    monkeypatch.setenv("IAI_MCP_LABILE_WINDOW_SEC", "3600")

    store = _make_store(tmp_path)
    rec = _make_record(embed_dim=store._embed_dim)
    store.insert(rec)
    store.reinforce_record(rec.id, is_retrieval=True)

    def _raise_critic(*args: Any, **kwargs: Any) -> dict:
        raise AssertionError(
            "evaluate_batch_reconsolidation must NOT be called when "
            "reconsolidation_tier1=False"
        )

    monkeypatch.setattr(
        "iai_mcp.reconsolidation_critic.evaluate_batch_reconsolidation",
        _raise_critic,
    )

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    done, payload = pipeline._step_reconsolidation(interrupt_check=None)
    assert done is True
    # No exception -> critic was correctly skipped.
    assert payload["records_reconsolidated"] == 0

    events = query_events(store, kind="reconsolidation_pass", limit=10)
    assert len(events) == 1
    body = events[0]["data"]
    assert body["critic_calls"] == 0
    assert body["records_reconsolidated"] == 0
    # The WHERE-pushdown still scans the labile record so the count is >= 1.
    assert body["records_scanned"] >= 1


# ---------------------------------------------------------------------------
# Test 7: SKIP branch does NOT tag schema_bypass on the existing record
# ---------------------------------------------------------------------------


def test_R4_schema_bypass_tagging_does_not_run_on_pattern_separation_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pattern_separation_gate fires SKIP, the existing
    record is reinforced (NOT re-inserted) and the schema-bypass cosine
    probe MUST NOT run on the original. With patsep DRY_RUN=false the SKIP
    path returns before the INSERT-gated tagger.

    We confirm the original record's schema_bypass column stays False even
    though we monkeypatch centroids to match the record's embedding (cos=1.0
    would normally tag) -- proving the tagger never ran on the SKIP branch.
    """
    # Force psep SKIP path to return early (not fall through to dry-run insert).
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD", "0.85")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim

    # Insert ONE record with NO centroids stub -> tagger sees empty centroids
    # -> schema_bypass stays False on the first insert.
    emb = [1.0] + [0.0] * (embed_dim - 1)
    rec_a = _make_record(
        embed_dim=embed_dim, literal="first record", embedding=emb,
    )
    store.insert(rec_a)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row_a_pre = df[df["id"] == str(rec_a.id)].iloc[0]
    assert bool(row_a_pre["schema_bypass"]) is False

    # Now stub centroids to force cos=1.0 on any subsequent INSERT branch.
    # The second insert is a near-dup (same embedding) -> patsep SKIP path
    # -> calls reinforce_record(existing_id) and returns early. The
    # INSERT-only tagger is NEVER reached.
    _stub_centroids(monkeypatch, {uuid.uuid4(): emb})

    rec_b = _make_record(
        embed_dim=embed_dim, literal="near-dup", embedding=emb,
    )
    store.insert(rec_b)  # SKIP path: caller-transparent merge into rec_a.

    df = tbl.to_pandas()
    # Only one row -- rec_b never got persisted (SKIP merged it).
    rows_a = df[df["id"] == str(rec_a.id)]
    assert len(rows_a) == 1
    row_a_post = rows_a.iloc[0]
    # The original record's schema_bypass MUST still be False -- the SKIP
    # branch does not re-run the centroid probe.
    assert bool(row_a_post["schema_bypass"]) is False, (
        "schema_bypass was unexpectedly toggled True on a SKIP branch; "
        "the centroid probe must only fire on GateAction.INSERT"
    )


# ---------------------------------------------------------------------------
# Test 8: every env var fails loud with a ValueError naming the var
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        ("IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD", "1.5"),
        ("IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD", "not-a-float"),
        ("IAI_MCP_LABILE_WINDOW_SEC", "-1"),
        ("IAI_MCP_LABILE_WINDOW_SEC", "0"),
        ("IAI_MCP_LABILE_WINDOW_SEC", "not-an-int"),
        ("IAI_MCP_RECONSOLIDATION_TIER1", "maybe"),
        ("IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD", "-0.1"),
        ("IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD", "1.1"),
        ("IAI_MCP_RECONSOLIDATION_DRY_RUN", "banana"),
    ],
)
def test_R5_invalid_env_var_raises_ValueError_naming_the_var(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    """Every malformed knob fails loud at daemon-boot via
    _load_reconsolidation_config(). The error message MUST name the
    offending env var so operators can act."""
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValueError, match=env_var):
        _load_reconsolidation_config()


# ---------------------------------------------------------------------------
# Test 9: dry_run skips the REM reconsolidation provenance update
# ---------------------------------------------------------------------------


def test_R6_dry_run_skips_all_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cfg.dry_run=True + Tier-1 ON + monkeypatched critic returning
    high error. Event still fires (with dry_run_mode=True); provenance
    untouched (no `reconsolidated_at` marker)."""
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "true")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_TIER1", "true")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD", "0.5")
    monkeypatch.setenv("IAI_MCP_LABILE_WINDOW_SEC", "3600")

    store = _make_store(tmp_path)
    rec = _make_record(embed_dim=store._embed_dim, literal="dry-run record")
    store.insert(rec)

    # We need labile_until > now in the row even though dry_run suppresses
    # the labile-write inside reinforce_record. Stamp it
    # directly via a one-off tbl.update so the WHERE pushdown picks it up.
    tbl = store.db.open_table(RECORDS_TABLE)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    tbl.update(
        where=f"id = '{str(rec.id)}'",
        values={"labile_until": future},
    )

    # stub for batched contract.
    def _stub_batched(items: Any, *args: Any, **kwargs: Any) -> dict:
        return {rid: 0.9 for rid, _surface in items}

    monkeypatch.setattr(
        "iai_mcp.reconsolidation_critic.evaluate_batch_reconsolidation",
        _stub_batched,
    )

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    done, payload = pipeline._step_reconsolidation(interrupt_check=None)
    assert done is True
    # Scan still runs.
    assert payload["records_scanned"] >= 1
    assert payload["dry_run"] is True

    # Event emitted with dry_run_mode=True.
    events = query_events(store, kind="reconsolidation_pass", limit=10)
    assert len(events) == 1
    body = events[0]["data"]
    assert body["dry_run_mode"] is True
    # critic_calls still increments under dry-run: the critic still runs
    # so operators observe its rate.
    assert body["critic_calls"] >= 1
    # records_reconsolidated counts the candidates that would have been
    # reconsolidated.
    assert body["records_reconsolidated"] >= 1

    # No actual mutation: round-trip the record and assert provenance has
    # no `reconsolidated_at` marker.
    fetched = store.get(rec.id)
    assert fetched is not None
    for entry in fetched.provenance:
        assert "reconsolidated_at" not in entry, (
            f"dry-run must NOT write provenance, got entry={entry!r}"
        )


# ---------------------------------------------------------------------------
# Test 10: dry_run suppresses the insert-side schema_bypass write
# ---------------------------------------------------------------------------


def test_R6_dry_run_skips_schema_bypass_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With cfg.dry_run=True the cosine probe still runs (event emitted
    with dry_run_mode=True, max_cos populated) but the schema_bypass
    attribute is NEVER set -- the persisted row stays at False."""
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "true")
    monkeypatch.setenv("IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD", "0.85")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim

    centroid = [1.0] + [0.0] * (embed_dim - 1)
    _stub_centroids(monkeypatch, {uuid.uuid4(): centroid})

    rec = _make_record(
        embed_dim=embed_dim,
        literal="aligned but dry-run",
        embedding=list(centroid),
    )
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert bool(row["schema_bypass"]) is False, (
        "dry-run schema-bypass must NOT mutate the row"
    )

    # Event still fires with dry_run_mode=True and tagged=False.
    events = query_events(store, kind="schema_bypass_pass", limit=10)
    assert len(events) >= 1
    body = events[0]["data"]
    assert body["dry_run_mode"] is True
    assert body["tagged"] is False
    # max_cos should reflect the actual probe (~1.0); use a permissive
    # lower bound so float precision doesn't bite.
    assert float(body["max_cos"]) >= 0.85


# ---------------------------------------------------------------------------
# Test 11: critic surface: PROMPT_TEMPLATE slot contract
# ---------------------------------------------------------------------------


def test_PROMPT_TEMPLATE_contains_required_slots() -> None:
    """prompt template must expose the two named slots and produce
    a non-empty string with no leftover braces after format()."""
    assert "{literal_surface}" in PROMPT_TEMPLATE
    assert "{current_summary}" in PROMPT_TEMPLATE
    formatted = PROMPT_TEMPLATE.format(
        literal_surface="x", current_summary="y",
    )
    assert isinstance(formatted, str) and len(formatted) > 0
    assert "{literal_surface}" not in formatted
    assert "{current_summary}" not in formatted
    # No stray un-substituted braces left behind.
    assert "{" not in formatted and "}" not in formatted


# ---------------------------------------------------------------------------
# Test 12: critic surface: Tier-0 fallback returns 0.0
# ---------------------------------------------------------------------------


def test_call_critic_tier0_fallback_returns_0_when_gate_denies(
    tmp_path: Path,
) -> None:
    """Tier-0 fallback contract: when llm_enabled=False and
    has_api_key=False the gate ladder denies the call and call_critic
    returns 0.0 unconditionally."""
    store = _make_store(tmp_path)
    err = call_critic(
        "memory text",
        "",
        store,
        llm_enabled=False,
        has_api_key=False,
    )
    assert err == 0.0


if __name__ == "__main__":  # pragma: no cover -- direct-run convenience
    import sys
    raise SystemExit(pytest.main([__file__, "-v"]))
