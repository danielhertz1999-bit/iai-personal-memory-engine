"""Regression tests for DMN Reflection Agent + Meta-Analyst.

The tests pin the following acceptance contracts:

    ReflectionAgent.synthesize returns a fresh semantic-tier MemoryRecord
        whose literal_surface includes the "Daily reflection..." framing and
        whose provenance carries synthesized_by="dmn_reflection".
    MetaAnalyst.snapshot counts known events correctly across kinds
        (memory_recall, memory_capture, sleep_step_completed/COMPACT_RECORDS,
        essential_variable_breach, erasure_agent_pass) and echoes
        window_hours + generated_at.
    SleepPipeline._step_dmn_reflection runs end-to-end against a real
        synthetic store under dry_run=false: returns (True, dict), inserts
        the synthesized semantic record, emits system_health_report event.
    Malformed env vars fail loud with ValueError naming the offending
        variable (parametrized across the 3 IAI_MCP_DMN_* / META_ANALYST_*
        env vars + multiple bad-value shapes).
    dry_run=true skips store.insert(synth_record) but DMN sub-passes
        still run and the system_health_report event still emits.
    meta_analyst_enabled=false suppresses the system_health_report event
        while ReflectionAgent still synthesizes + (with dry_run=false) inserts.

Synthetic stores use tmp_path with user_id='alice'. Fixture seed values use
'alice' / 'bob' / lorem-style labels -- never 'Alice'.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp.daemon import DmnConfig, _load_dmn_config
from iai_mcp.dmn_reflection import MetaAnalyst, ReflectionAgent
from iai_mcp.events import query_events, write_event
from iai_mcp.lifecycle_state import default_state, save_state
from iai_mcp.sleep_pipeline import SleepPipeline
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Autouse fixture: pin IAI_MCP_STORE to tmp_path so per-test MemoryStore
# construction stays isolated from the user's real store. Defensively
# wipe every IAI_MCP_DMN_* / META_ANALYST_* env var so each test starts from
# defaults (reflection_window_hours=24, meta_analyst_enabled=True,
# dry_run=True under pytest via PYTEST_CURRENT_TEST). Tests that need
# overrides re-set after this fixture.
@pytest.fixture(autouse=True)
def _isolate_iai_dmn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.setenv("IAI_MCP_KEYRING_BYPASS", "true")
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    for var in (
        "IAI_MCP_DMN_REFLECTION_WINDOW_HOURS",
        "IAI_MCP_META_ANALYST_ENABLED",
        "IAI_MCP_DMN_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)


# Build a minimal MemoryRecord with a per-record orthogonal-ish embedding
# axis so the pattern_separation_gate inside store.insert cannot collapse
# two synthetic records as near-duplicates.
def _make_record(
    *,
    embed_dim: int,
    literal_surface: str = "alice prefers tea over coffee",
    community_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
    embedding: list[float] | None = None,
) -> MemoryRecord:
    now = created_at if created_at is not None else datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=embedding if embedding is not None else [0.01] * embed_dim,
        community_id=community_id,
        centrality=0.5,
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


def _seed_records_for_communities(
    store: MemoryStore,
    community_assignments: list[tuple[str, uuid.UUID]],
) -> list[uuid.UUID]:
    """Insert one record per (literal, community_id) tuple.

    Returns the list of inserted record ids in insertion order. Distinct
    orthogonal-ish embedding axes (one cell per record) keep
    pattern_separation_gate from collapsing the seeded rows as near-
    duplicates (mirrors helper invariant;
    UUID-typed community_id discipline reused via the caller passing
    uuid.uuid4() values).
    """
    embed_dim = store._embed_dim
    ids: list[uuid.UUID] = []
    for i, (surface, cid) in enumerate(community_assignments):
        emb = [0.0] * embed_dim
        emb[i % embed_dim] = 1.0
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=surface,
            community_id=cid,
            embedding=emb,
        )
        store.insert(rec)
        ids.append(rec.id)
    return ids


# ---------------------------------------------------------------------------
# Test 1: ReflectionAgent.synthesize returns semantic-tier record
# ---------------------------------------------------------------------------


def test_reflection_synthesize_returns_semantic_record(
    tmp_path: Path,
) -> None:
    """With a synthetic 5-record store spread across two
    communities, ReflectionAgent.synthesize returns a fresh MemoryRecord
    at tier="semantic" whose literal_surface includes the "Daily
    reflection..." framing and whose provenance_json (via the provenance
    list[dict] on the dataclass) carries synthesized_by="dmn_reflection".

    Topic labels (first-50-chars of the most-recent record per community)
    appear in the literal_surface topics list when community_id is set --
    we seed two distinct community_ids and assert at least one topic
    fragment lands in the surface string."""
    store = _make_store(tmp_path)

    cid_alpha = uuid.uuid4()
    cid_beta = uuid.uuid4()
    # 5 records: 3 in alpha, 2 in beta. Short literals so first-50-chars
    # = full label.
    _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_alpha),
            ("alice topic alpha second", cid_alpha),
            ("alice topic alpha third", cid_alpha),
            ("bob topic beta", cid_beta),
            ("bob topic beta second", cid_beta),
        ],
    )

    synth = ReflectionAgent().synthesize(store, window_hours=24)

    # tier="semantic".
    assert synth.tier == "semantic", (
        f"synth.tier must be 'semantic'; got {synth.tier!r}"
    )

    # MemoryRecord shape: must round-trip through the dataclass.
    assert isinstance(synth, MemoryRecord)
    assert isinstance(synth.id, uuid.UUID)

    # literal_surface framing per ReflectionAgent.synthesize body.
    assert "Daily reflection" in synth.literal_surface, (
        f"literal_surface missing 'Daily reflection' framing; "
        f"got {synth.literal_surface!r}"
    )
    assert "top topics were" in synth.literal_surface, (
        f"literal_surface missing 'top topics were' marker; "
        f"got {synth.literal_surface!r}"
    )
    # Topic labels (first-50-chars of most-recent record per community)
    # should land in the surface string -- at least one of the seeded
    # community labels must appear.
    assert (
        "alice topic alpha" in synth.literal_surface
        or "bob topic beta" in synth.literal_surface
    ), (
        f"literal_surface missing seeded community labels; "
        f"got {synth.literal_surface!r}"
    )

    # provenance carries synthesized_by="dmn_reflection" + window_hours echo.
    assert isinstance(synth.provenance, list)
    assert len(synth.provenance) >= 1
    prov = synth.provenance[0]
    assert prov.get("synthesized_by") == "dmn_reflection", (
        f"provenance missing synthesized_by='dmn_reflection'; got {prov!r}"
    )
    assert prov.get("window_hours") == 24, (
        f"provenance missing window_hours echo; got {prov!r}"
    )
    assert isinstance(prov.get("topics"), list)
    assert isinstance(prov.get("captured_count"), int)
    assert isinstance(prov.get("recalled_count"), int)

    # Embedding is a placeholder zero-vector of the right dim.
    assert len(synth.embedding) == store._embed_dim
    assert all(v == 0.0 for v in synth.embedding), (
        "synthesised record must carry the zero-vector placeholder; "
        "next REM consolidation re-embeds"
    )


# ---------------------------------------------------------------------------
# Test 2: MetaAnalyst.snapshot counts known events correctly
# ---------------------------------------------------------------------------


def test_meta_analyst_snapshot_counts_correct(tmp_path: Path) -> None:
    """Seed the events table with a known mix of kinds and
    assert the snapshot dict carries exact counts per kind.

    Counts verified:
      * recall_count from memory_recall events
      * capture_count from memory_capture events
      * sleep_cycles_count from sleep_step_completed with data.step="COMPACT_RECORDS"
      * breach_count from essential_variable_breach events
      * erasure_count from erasure_agent_pass events

    Plus average_record_count_delta proxy (captures - erasures when at
    least one fires) + window_hours echo + generated_at ISO string."""
    store = _make_store(tmp_path)

    # Seed known event counts via write_event. Each call timestamps at
    # insertion (datetime.now(UTC)) so all events land inside the 24h
    # reflection window the snapshot scans.
    for _ in range(4):
        write_event(store, "memory_recall", {"cue": "alice"})
    for _ in range(3):
        write_event(store, "memory_capture", {"text": "bob fact"})
    for _ in range(2):
        # Only COMPACT_RECORDS sleep_step_completed events count toward
        # sleep_cycles_count per dmn_reflection.py L301-309.
        write_event(
            store,
            "sleep_step_completed",
            {"step": "COMPACT_RECORDS"},
        )
    # A non-COMPACT_RECORDS sleep_step_completed event must NOT count.
    write_event(
        store,
        "sleep_step_completed",
        {"step": "SCHEMA_MINE"},
    )
    for _ in range(1):
        write_event(
            store,
            "essential_variable_breach",
            {"variable": "richclub_density"},
        )
    for _ in range(5):
        write_event(store, "erasure_agent_pass", {"erased": 1})

    snap = MetaAnalyst().snapshot(store, window_hours=24)

    assert isinstance(snap, dict)
    assert snap["recall_count"] == 4, (
        f"recall_count mismatch: expected 4, got {snap['recall_count']}"
    )
    assert snap["capture_count"] == 3, (
        f"capture_count mismatch: expected 3, got {snap['capture_count']}"
    )
    assert snap["sleep_cycles_count"] == 2, (
        f"sleep_cycles_count mismatch (only COMPACT_RECORDS should "
        f"count): expected 2, got {snap['sleep_cycles_count']}"
    )
    assert snap["breach_count"] == 1, (
        f"breach_count mismatch: expected 1, got {snap['breach_count']}"
    )
    assert snap["erasure_count"] == 5, (
        f"erasure_count mismatch: expected 5, got {snap['erasure_count']}"
    )

    # Proxy: captures(3) - erasures(5) = -2.
    assert snap["average_record_count_delta"] == -2.0, (
        f"average_record_count_delta proxy mismatch: expected -2.0, "
        f"got {snap['average_record_count_delta']!r}"
    )

    # Echo + audit fields.
    assert snap["window_hours"] == 24
    assert isinstance(snap["generated_at"], str)
    # ISO-8601 parse round-trip sanity check.
    parsed = datetime.fromisoformat(snap["generated_at"])
    assert parsed.tzinfo is not None, (
        f"generated_at must be tz-aware ISO; got {snap['generated_at']!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: _step_dmn_reflection runs end-to-end (dry_run=false)
# ---------------------------------------------------------------------------


def test_dmn_reflection_step_runs_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instantiate SleepPipeline against a real synthetic
    store with dry_run=false; call _step_dmn_reflection directly; assert
    returns (True, dict) with both meta_analyst_emitted=True and
    reflection_synthesized=True; assert the synthesized record landed in
    the store (record count grew by 1); assert system_health_report event
    was emitted.

    CRITICAL: the outer try/except in _step_dmn_reflection swallows ALL
    exceptions into (True, {persist_error: True}); the explicit
    absence-of-persist_error assertion below catches silent
    fall-through to that path."""
    # Override pytest-aware default so the insert path actually fires.
    monkeypatch.setenv("IAI_MCP_DMN_DRY_RUN", "false")

    store = _make_store(tmp_path)
    cid_alpha = uuid.uuid4()
    cid_beta = uuid.uuid4()
    _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_alpha),
            ("alice topic alpha extended", cid_alpha),
            ("bob topic beta", cid_beta),
        ],
    )

    pre_count = len(store.all_records())

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store, lifecycle_state_path=lifecycle_path,
    )

    done, payload = pipeline._step_dmn_reflection(interrupt_check=None)
    assert done is True
    # Explicit happy-path assertion: outer try/except did NOT fire.
    assert "persist_error" not in payload, (
        f"_step_dmn_reflection fell through to the outer try/except "
        f"silent-failure path; payload={payload!r}"
    )
    assert payload.get("meta_analyst_emitted") is True, (
        f"payload must surface meta_analyst_emitted=True under default "
        f"enabled config; got {payload!r}"
    )
    assert payload.get("reflection_synthesized") is True, (
        f"payload must surface reflection_synthesized=True under "
        f"dry_run=false; got {payload!r}"
    )
    assert payload.get("dry_run_mode") is False, (
        f"payload must echo dry_run_mode=False (we set the env var); "
        f"got {payload!r}"
    )

    # Synthesised record landed in the store -- count grew by exactly 1.
    post_count = len(store.all_records())
    assert post_count == pre_count + 1, (
        f"store record count must grow by exactly 1 (the synthesised "
        f"semantic record); pre={pre_count}, post={post_count}"
    )

    # At least one synthesised semantic record present with the
    # dmn_reflection provenance marker.
    semantic_records = [
        r for r in store.all_records()
        if r.tier == "semantic"
        and any(
            (p or {}).get("synthesized_by") == "dmn_reflection"
            for p in (r.provenance or [])
        )
    ]
    assert len(semantic_records) >= 1, (
        f"no semantic record with synthesized_by='dmn_reflection' found; "
        f"all records: {[(r.tier, r.provenance) for r in store.all_records()]!r}"
    )

    # system_health_report event was emitted (MetaAnalyst path).
    health_events = query_events(
        store, kind="system_health_report", limit=10,
    )
    assert len(health_events) >= 1, (
        f"_step_dmn_reflection must emit system_health_report event "
        f"under meta_analyst_enabled=True; got {len(health_events)}"
    )
    body = health_events[0]["data"]
    assert "recall_count" in body
    assert "capture_count" in body
    assert "window_hours" in body
    assert body["dry_run_mode"] is False, (
        f"system_health_report body must echo dry_run_mode=False; "
        f"got {body!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: dry_run=true skips insert but events still emit
# ---------------------------------------------------------------------------


def test_dmn_dry_run_no_record_insert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With IAI_MCP_DMN_DRY_RUN=true the synthesised record
    is computed but NOT inserted; the store record count stays unchanged;
    the system_health_report event is still emitted (independent gate)."""
    # Explicit (the pytest-aware default would set this anyway, but
    # explicit is safer against future test-runner env-var leakage).
    monkeypatch.setenv("IAI_MCP_DMN_DRY_RUN", "true")

    store = _make_store(tmp_path)
    cid_alpha = uuid.uuid4()
    _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_alpha),
            ("alice topic alpha extended", cid_alpha),
        ],
    )

    pre_count = len(store.all_records())

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store, lifecycle_state_path=lifecycle_path,
    )

    done, payload = pipeline._step_dmn_reflection(interrupt_check=None)
    assert done is True
    assert "persist_error" not in payload, (
        f"_step_dmn_reflection fell through to silent-failure path "
        f"under dry_run=true; payload={payload!r}"
    )
    assert payload["dry_run_mode"] is True
    # reflection_synthesized tracks ACTUAL insert: under dry_run=true it
    # MUST be False even though the synthesise call still computed the
    # would-be record.
    assert payload["reflection_synthesized"] is False, (
        f"dry_run=true must leave reflection_synthesized=False; "
        f"got {payload!r}"
    )
    # MetaAnalyst path is independent of dry_run; meta_analyst_emitted
    # still flips True.
    assert payload["meta_analyst_emitted"] is True, (
        f"meta_analyst_emitted must stay True under dry_run=true "
        f"(independent gate); got {payload!r}"
    )

    # Store count unchanged.
    post_count = len(store.all_records())
    assert post_count == pre_count, (
        f"dry_run=true must NOT insert; pre={pre_count}, post={post_count}"
    )

    # system_health_report event STILL emitted (independent gate).
    health_events = query_events(
        store, kind="system_health_report", limit=10,
    )
    assert len(health_events) >= 1, (
        f"system_health_report must emit even under dry_run=true; "
        f"got {len(health_events)}"
    )


# ---------------------------------------------------------------------------
# Test 5: meta_analyst_enabled=false suppresses health-report event
# ---------------------------------------------------------------------------


def test_meta_analyst_disabled_skip_health_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With IAI_MCP_META_ANALYST_ENABLED=false, zero
    system_health_report events fire; ReflectionAgent still synthesises
    the record (independent gate); under dry_run=false the record lands
    in the store."""
    monkeypatch.setenv("IAI_MCP_META_ANALYST_ENABLED", "false")
    monkeypatch.setenv("IAI_MCP_DMN_DRY_RUN", "false")

    store = _make_store(tmp_path)
    cid_alpha = uuid.uuid4()
    _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_alpha),
            ("alice topic alpha extended", cid_alpha),
        ],
    )

    pre_count = len(store.all_records())

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store, lifecycle_state_path=lifecycle_path,
    )

    done, payload = pipeline._step_dmn_reflection(interrupt_check=None)
    assert done is True
    assert "persist_error" not in payload, (
        f"_step_dmn_reflection fell through to silent-failure path; "
        f"payload={payload!r}"
    )
    # MetaAnalyst gate closed -> no emit.
    assert payload["meta_analyst_emitted"] is False, (
        f"meta_analyst_emitted must be False under enabled=false; "
        f"got {payload!r}"
    )
    # Reflection runs regardless of meta_analyst_enabled (independent gate).
    assert payload["reflection_synthesized"] is True, (
        f"reflection still synthesises under meta_analyst_enabled=false "
        f"+ dry_run=false; got {payload!r}"
    )

    # Store grew by exactly 1 (the synthesised record).
    post_count = len(store.all_records())
    assert post_count == pre_count + 1, (
        f"store must grow by 1 under enabled=false+dry_run=false; "
        f"pre={pre_count}, post={post_count}"
    )

    # ZERO system_health_report events emitted under the closed gate.
    health_events = query_events(
        store, kind="system_health_report", limit=10,
    )
    assert len(health_events) == 0, (
        f"meta_analyst_enabled=false must suppress ALL "
        f"system_health_report emits; got {len(health_events)}"
    )


# ---------------------------------------------------------------------------
# Test 6: every malformed env var fails loud with ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        # int parse failure
        ("IAI_MCP_DMN_REFLECTION_WINDOW_HOURS", "abc"),
        # int parses but out of range [1, 720]
        ("IAI_MCP_DMN_REFLECTION_WINDOW_HOURS", "0"),
        ("IAI_MCP_DMN_REFLECTION_WINDOW_HOURS", "99999"),
        # vocab miss
        ("IAI_MCP_META_ANALYST_ENABLED", "not-a-bool"),
        ("IAI_MCP_META_ANALYST_ENABLED", "maybe"),
        ("IAI_MCP_DMN_DRY_RUN", "bogus"),
        ("IAI_MCP_DMN_DRY_RUN", "perhaps"),
    ],
)
def test_env_var_fail_loud_parametrized(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    """Every IAI_MCP_DMN_* / META_ANALYST_* env var with a
    malformed or out-of-range value raises ValueError whose message names
    the offending env var (so operators can grep the traceback). Default
    parse (no overrides) succeeds and returns the defaults."""
    monkeypatch.setenv(env_var, bad_value)

    with pytest.raises(ValueError) as excinfo:
        _load_dmn_config()

    # ValueError message must name the offending env var so operators
    # can locate the misconfiguration without reading source.
    assert env_var in str(excinfo.value), (
        f"ValueError must name the offending env var {env_var!r}; "
        f"got {excinfo.value!r}"
    )


# Default-parse sanity: with no overrides, _load_dmn_config returns
# the defaults (window=24h, meta_analyst_enabled=True, dry_run=True
# under pytest via PYTEST_CURRENT_TEST). Separate from the parametrized
# fail-loud test so a future default change surfaces in exactly one
# assertion site.
def test_dmn_config_defaults_under_pytest() -> None:
    """Defaults: window=24h, meta_analyst_enabled=True, dry_run=True
    (pytest-aware via PYTEST_CURRENT_TEST)."""
    cfg = _load_dmn_config()
    assert isinstance(cfg, DmnConfig)
    assert cfg.reflection_window_hours == 24
    assert cfg.meta_analyst_enabled is True
    # Under pytest the pytest-aware default flips dry_run to True
    # (reused).
    assert cfg.dry_run is True


# ---------------------------------------------------------------------------
# Test 8 -- reflection-input exclusion: prior reflection records must not
# be re-ingested as topics (no recursive self-nesting)
# ---------------------------------------------------------------------------


def test_reflection_does_not_re_ingest_prior_reflections(
    tmp_path: Path,
) -> None:
    """Prior reflection records must be excluded from the synthesize input set.

    After community-detection passes, synthetic reflection records can acquire
    a community_id and appear as community members. Without an exclusion filter
    they surface as topic labels whose first-50-chars starts with
    'Daily reflection: top topics were [Daily reflecti' — causing nested
    'Daily reflection: top topics were [Daily reflection: …]' strings in
    every subsequent reflection.

    This test:
      1. Seeds real episodic records with a community_id.
      2. Inserts a prior reflection record with synthesized_by='dmn_reflection'
         in its provenance AND the same community_id (simulating what
         crisis_recluster assigns in production).
      3. Runs ReflectionAgent().synthesize().
      4. Asserts the new reflection's literal_surface does NOT contain a
         nested 'Daily reflection:' inside the topics list.

    Negative-control note: reverting the exclusion filter in
    dmn_reflection.py (the provenance-check loop) makes this test RED —
    the prior reflection record would be included in in_window and its
    first-50-chars 'Daily reflection: top topics were […' would appear as
    a topic label inside the new reflection, producing the nested string.
    """
    store = _make_store(tmp_path)

    shared_cid = uuid.uuid4()

    # Seed real episodic records (the "genuine" community members).
    _seed_records_for_communities(
        store,
        [
            ("real user memory about something useful", shared_cid),
            ("another real memory from the user", shared_cid),
            ("a third real memory entry", shared_cid),
        ],
    )

    # Insert a prior reflection record that simulates having acquired a
    # community_id (as crisis_recluster would assign in production).
    # The literal_surface starts with 'Daily reflection: …' — without the
    # exclusion filter this would appear as the top-1-recency label for
    # shared_cid and produce nested output.
    embed_dim = store._embed_dim
    prior_reflection = MemoryRecord(
        id=uuid.uuid4(),
        tier="semantic",
        literal_surface=(
            "Daily reflection: top topics were "
            "[real user memory about something]; "
            "captured 5 turns; recalled 2 times."
        ),
        aaak_index="",
        embedding=[0.0] * embed_dim,
        community_id=shared_cid,  # simulates crisis_recluster assignment
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[
            {
                "synthesized_by": "dmn_reflection",
                "window_hours": 24,
                "topics": ["real user memory about something"],
                "captured_count": 5,
                "recalled_count": 2,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        ],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        language="en",
        tags=[],
    )
    store.insert(prior_reflection)

    # Run synthesize. Without the exclusion filter the prior reflection is
    # the most-recent record in shared_cid, so its first-50-chars label
    # ('Daily reflection: top topics were [real user m') would appear in the
    # new reflection's topics — producing the nested pattern.
    synth = ReflectionAgent().synthesize(store, window_hours=24)

    # The new reflection must carry the 'Daily reflection: …' frame at the
    # top level only — never nested inside the topics list.
    assert "Daily reflection" in synth.literal_surface, (
        "synthesized record must carry the 'Daily reflection' frame; "
        f"got {synth.literal_surface!r}"
    )
    # Core invariant: 'Daily reflection:' must NOT appear inside the topics
    # substring (i.e. no nested reflection label).
    topics_start = synth.literal_surface.find("top topics were [")
    topics_end = synth.literal_surface.find("]", topics_start)
    if topics_start >= 0 and topics_end > topics_start:
        topics_segment = synth.literal_surface[topics_start:topics_end + 1]
        assert "Daily reflection" not in topics_segment, (
            "prior reflection record was re-ingested as a topic label — "
            "nested 'Daily reflection:' found inside topics segment; "
            f"topics_segment={topics_segment!r}; "
            f"full literal_surface={synth.literal_surface!r}"
        )

    # The synthesized record must have synthesized_by='dmn_reflection' in
    # its own provenance (its own identity is never lost).
    prov = synth.provenance[0] if synth.provenance else {}
    assert prov.get("synthesized_by") == "dmn_reflection"
