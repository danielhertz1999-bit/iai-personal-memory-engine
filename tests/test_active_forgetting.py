"""Active forgetting regression suite.

Inline three-cohort synthetic store + five behavioural tests that pin
the ErasureAgent contract end-to-end on a real MemoryStore +
SleepPipeline.

Cohorts (total 19 records):

* **high_utility** — 5 records. centrality=0.5, last_reviewed within 30d,
  age=60d, pinned=False, never_decay=False. Should NEVER be tombstoned.
* **low_utility** — 10 records. centrality=0.005 (well below the 0.02
  default threshold), last_reviewed=None, age=60d, pinned=False,
  never_decay=False. SHOULD be tombstoned after one pass.
* **protected** — 4 records (2 pinned + 2 never_decay). Otherwise
  identical to low_utility (centrality=0.005, last_reviewed=None,
  age=60d). MUST be carved out across any number of passes.

Test functions:

1. ``test_low_utility_cohort_tombstoned_after_one_pass`` — mutation path.
2. ``test_protected_cohort_survives_multiple_passes`` — protected carve-out
   stays sticky across repeated invocations.
3. ``test_aged_tombstones_dropped_after_second_pass`` — TTL drop sweep.
4. ``test_dry_run_mode_emits_event_no_mutation`` — dry-run path.
5. ``test_erasure_event_body_shape_and_uniqueness`` — wiring proof.

All tests use the ``iai_home`` HOME-isolated fixture so production user
state is untouched. ``_utc_now`` is monkeypatched on the
``iai_mcp.sleep_pipeline`` module so cohort timestamps and TTL
fast-forward arithmetic stay deterministic. The autouse
``_crypto_passphrase_env`` fixture in ``tests/conftest.py`` covers the
AES-256-GCM events-table read path automatically.
"""
from __future__ import annotations

# Standard-library imports first so the optional `iai_mcp.*` imports below
# fail loud with a clear ImportError if the package layout changes.
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.daemon import _load_erasure_config
from iai_mcp.events import query_events
from iai_mcp.sleep_pipeline import SleepPipeline
from iai_mcp.store import RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord


# Fixed wall-clock anchor for the synthetic store. The ErasureAgent
# eligibility predicate compares created_at / last_reviewed against
# `_utc_now()`; monkeypatching `iai_mcp.sleep_pipeline._utc_now` to
# return this constant makes the cohort arithmetic deterministic.
FROZEN_NOW = datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc)

# Cohort sizes are baked into the acceptance criteria. Changing
# any of these breaks the count_quarantined assertion in
# test_low_utility_cohort_tombstoned_after_one_pass.
HIGH_UTILITY_N = 5
LOW_UTILITY_N = 10
PROTECTED_N = 4  # 2 pinned + 2 never_decay
TOTAL_N = HIGH_UTILITY_N + LOW_UTILITY_N + PROTECTED_N  # 19


# Helper: build a fully-populated MemoryRecord with cohort-specific
# salience features. Embedding values are stub (the eligibility predicate
# never inspects them); literal_surface uses generic example names per
# project the project convention ("never `Alice` as example data").
def _make_record(
    *,
    tier: str,
    centrality: float,
    pinned: bool,
    never_decay: bool,
    last_reviewed: datetime | None,
    created_at: datetime,
    embed_dim: int,
    literal_surface: str = "alice prefers tea over coffee",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=literal_surface,
        aaak_index="",
        embedding=[0.01] * embed_dim,
        community_id=None,
        centrality=centrality,
        detail_level=1,
        pinned=pinned,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=last_reviewed,
        # detail_level=1 keeps __post_init__ from auto-flipping never_decay
        # on us; the carve-out flag is therefore strictly the caller's
        # choice for protected-cohort wiring.
        never_decay=never_decay,
        never_merge=False,
        provenance=[],
        created_at=created_at,
        updated_at=created_at,
        language="en",
    )


# Build the 19-record three-cohort fixture against the supplied store.
# Returns a dict mapping cohort name to list of UUIDs so test bodies can
# slice the records dataframe by cohort without re-deriving membership.
def _build_three_cohort_store(
    store: MemoryStore, now: datetime,
) -> dict[str, list[UUID]]:
    age_60d = now - timedelta(days=60)
    recent_review = now - timedelta(days=7)
    embed_dim = store._embed_dim

    cohort_ids: dict[str, list[UUID]] = {
        "high_utility": [],
        "low_utility": [],
        "protected": [],
    }

    # High-utility: high centrality, freshly reviewed, age > 30d.
    for _ in range(HIGH_UTILITY_N):
        rec = _make_record(
            tier="episodic",
            centrality=0.5,
            pinned=False,
            never_decay=False,
            last_reviewed=recent_review,
            created_at=age_60d,
            embed_dim=embed_dim,
            literal_surface="alice's notes on graph topology stability",
        )
        store.insert(rec)
        cohort_ids["high_utility"].append(rec.id)

    # Low-utility: low centrality, no review, aged.
    for _ in range(LOW_UTILITY_N):
        rec = _make_record(
            tier="episodic",
            centrality=0.005,
            pinned=False,
            never_decay=False,
            last_reviewed=None,
            created_at=age_60d,
            embed_dim=embed_dim,
            literal_surface="bob mentioned the weather on a forgotten day",
        )
        store.insert(rec)
        cohort_ids["low_utility"].append(rec.id)

    # Protected: 2 pinned + 2 never_decay, otherwise low-utility-equivalent.
    for i in range(PROTECTED_N):
        is_pinned = i < 2
        is_never_decay = not is_pinned
        rec = _make_record(
            tier="episodic",
            centrality=0.005,
            pinned=is_pinned,
            never_decay=is_never_decay,
            last_reviewed=None,
            created_at=age_60d,
            embed_dim=embed_dim,
            literal_surface=(
                "alice locked this as a permanent reminder"
                if is_pinned
                else "bob marked this never_decay for the long memory"
            ),
        )
        store.insert(rec)
        cohort_ids["protected"].append(rec.id)

    return cohort_ids


# HOME-isolated fixture (mirrors tests/test_daemon_crash_loop_immunity.py
# L41-L52). Keyring backend is set to the fail-null backend so the
# crypto-key resolver lands on the passphrase env var rather than the
# production keyring; the autouse `_crypto_passphrase_env` from
# tests/conftest.py supplies that passphrase.
@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-phase11-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "hippo"))
    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


# Shared pipeline + store + cohorts fixture for Tests 1, 2, 3, 5.
# Test 4 (dry-run) explicitly skips this fixture and builds its own
# environment so the dry_run=True env-var override takes effect before
# any pipeline construction or step invocation.
@pytest.fixture
def pipeline(iai_home, tmp_path, monkeypatch):
    # Disable the pytest-aware dry-run default so the mutation
    # path actually runs. _load_erasure_config sees this env var on its
    # next call thanks to the call-on-demand contract.
    monkeypatch.setenv("IAI_MCP_ERASURE_DRY_RUN", "false")

    # Freeze the clock so cohort timestamps and the ErasureAgent's
    # `_utc_now()` agree. The monkeypatch target string is the module
    # attribute, so both the sleep_pipeline module read AND the
    # `_step_erasure_agent` reference resolve to the patched callable.
    monkeypatch.setattr(
        "iai_mcp.sleep_pipeline._utc_now", lambda: FROZEN_NOW,
    )

    store = MemoryStore()
    cohort_ids = _build_three_cohort_store(store, FROZEN_NOW)

    # Setup sanity guard: catch insertion failures (wrong embedding dim,
    # missing language field, etc.) at fixture time rather than at the
    # tombstone assertion. Cheaper to debug.
    tbl = store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == TOTAL_N, (
        f"three-cohort fixture must insert exactly {TOTAL_N} records, "
        f"got {tbl.count_rows()}"
    )

    pipe = SleepPipeline(
        store=store,
        lifecycle_state_path=tmp_path / "lifecycle_state.json",
    )
    return pipe, store, cohort_ids


# Helper: read the records table as a pandas dataframe and return the
# row dict for a single UUID, or None if absent. Encapsulates the
# `str(uuid)` lookup so test bodies stay readable.
def _row_for(df, rid: UUID) -> dict | None:
    sub = df[df["id"] == str(rid)]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


# Helper: True when `tombstoned_at` is non-null on the supplied row dict.
# The store returns pandas NaT for null timestamp columns; pd.isna covers
# both None and NaT.
def _is_tombstoned(row: dict) -> bool:
    import pandas as pd

    val = row.get("tombstoned_at")
    return val is not None and not pd.isna(val)


# ---------------------------------------------------------------------------
# Test 1 — low-utility tombstoned after one pass, high + protected untouched
# ---------------------------------------------------------------------------


def test_low_utility_cohort_tombstoned_after_one_pass(pipeline):
    """One `_step_erasure_agent` pass tombstones the 10 low-utility rows
    and leaves the 5 high-utility + 4 protected rows untouched.
    """
    pipe, store, cohort_ids = pipeline

    ok, payload = pipe._step_erasure_agent(None)
    assert ok is True, payload
    assert payload.get("dry_run") is False, (
        f"fixture should disable dry-run, got payload={payload}"
    )
    assert payload.get("count_quarantined") == LOW_UTILITY_N, (
        f"expected count_quarantined={LOW_UTILITY_N}, got {payload}"
    )

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    assert df.shape[0] == TOTAL_N, (
        f"tombstoning sets a column, must not delete rows; "
        f"expected {TOTAL_N}, got {df.shape[0]}"
    )

    # Every low-utility row must now have a tombstone.
    for rid in cohort_ids["low_utility"]:
        row = _row_for(df, rid)
        assert row is not None, f"low-utility row {rid} disappeared"
        assert _is_tombstoned(row), (
            f"low-utility row {rid} should be tombstoned, "
            f"got tombstoned_at={row.get('tombstoned_at')!r}"
        )

    # High-utility cohort: never tombstoned (above threshold + freshly reviewed).
    for rid in cohort_ids["high_utility"]:
        row = _row_for(df, rid)
        assert row is not None, f"high-utility row {rid} disappeared"
        assert not _is_tombstoned(row), (
            f"high-utility row {rid} should NOT be tombstoned, "
            f"got tombstoned_at={row.get('tombstoned_at')!r}"
        )

    # Protected cohort: absolute carve-out.
    for rid in cohort_ids["protected"]:
        row = _row_for(df, rid)
        assert row is not None, f"protected row {rid} disappeared"
        assert not _is_tombstoned(row), (
            f"protected row {rid} should NOT be tombstoned (R3 carve-out), "
            f"got tombstoned_at={row.get('tombstoned_at')!r} "
            f"pinned={row.get('pinned')} never_decay={row.get('never_decay')}"
        )


# ---------------------------------------------------------------------------
# Test 2 — protected cohort survives multiple ErasureAgent passes
# ---------------------------------------------------------------------------


def test_protected_cohort_survives_multiple_passes(pipeline):
    """Three sequential `_step_erasure_agent` invocations leave every
    protected row's `tombstoned_at` strictly NULL. Guards against any
    regression where the carve-out is applied only on first pass (e.g.
    accidental reliance on a per-run cache).
    """
    pipe, store, cohort_ids = pipeline

    for pass_idx in range(3):
        ok, payload = pipe._step_erasure_agent(None)
        assert ok is True, f"pass {pass_idx}: {payload}"

        tbl = store.db.open_table(RECORDS_TABLE)
        df = tbl.to_pandas()
        for rid in cohort_ids["protected"]:
            row = _row_for(df, rid)
            assert row is not None, (
                f"pass {pass_idx}: protected row {rid} disappeared"
            )
            assert not _is_tombstoned(row), (
                f"pass {pass_idx}: protected row {rid} was tombstoned; "
                f"R3 carve-out failed. "
                f"pinned={row.get('pinned')} never_decay={row.get('never_decay')}"
            )


# ---------------------------------------------------------------------------
# Test 3 — aged tombstones dropped from the records table after TTL fast-forward
# ---------------------------------------------------------------------------


def test_aged_tombstones_dropped_after_second_pass(pipeline, monkeypatch):
    """Cycle 1: `_step_erasure_agent` tombstones the low-utility cohort.
    Cycle 2: fast-forward the clock past `IAI_MCP_ERASURE_TOMBSTONE_TTL_SEC`,
    then run `_step_optimize_lance` — the tombstoned rows are physically
    dropped from the records table while high-utility + protected rows
    survive.
    """
    pipe, store, cohort_ids = pipeline

    # Freeze the canonical module's clock to FROZEN_NOW before erasure so
    # tombstoned_at is set to FROZEN_NOW (not actual wall-clock time).
    # The pipeline fixture patches iai_mcp.sleep_pipeline._utc_now (the shim),
    # but _step_erasure_agent reads the canonical module's _utc_now directly.
    monkeypatch.setattr(
        "iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: FROZEN_NOW,
    )

    ok, _ = pipe._step_erasure_agent(None)
    assert ok is True

    # Capture TTL from the active config (so this test stays correct
    # under env-var-overridden TTLs in future suites).
    cfg = _load_erasure_config()
    ttl = cfg.tombstone_ttl_sec

    # Fast-forward `_utc_now` past the tombstone TTL window. Both the
    # ErasureAgent and OPTIMIZE_LANCE handlers re-read this attribute
    # inside their bodies, so the patch applied here takes effect on
    # the next call.
    fast_forward = FROZEN_NOW + timedelta(seconds=ttl + 60)
    monkeypatch.setattr(
        "iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: fast_forward,
    )

    ok2, payload2 = pipe._step_optimize_lance(None)
    assert ok2 is True, payload2
    assert payload2.get("count_dropped_by_erasure") == LOW_UTILITY_N, (
        f"expected {LOW_UTILITY_N} drops, got {payload2}"
    )

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()

    # Every low-utility UUID is physically gone.
    surviving_ids = set(df["id"].tolist())
    for rid in cohort_ids["low_utility"]:
        assert str(rid) not in surviving_ids, (
            f"low-utility row {rid} should be dropped after TTL fast-forward, "
            f"but is still present"
        )

    # High-utility and protected rows all still present, untombstoned.
    for rid in cohort_ids["high_utility"]:
        row = _row_for(df, rid)
        assert row is not None, f"high-utility row {rid} disappeared"
        assert not _is_tombstoned(row)
    for rid in cohort_ids["protected"]:
        row = _row_for(df, rid)
        assert row is not None, f"protected row {rid} disappeared"
        assert not _is_tombstoned(row)

    # Final row count matches the surviving cohorts.
    assert df.shape[0] == HIGH_UTILITY_N + PROTECTED_N, (
        f"expected {HIGH_UTILITY_N + PROTECTED_N} surviving rows "
        f"({HIGH_UTILITY_N} high-utility + {PROTECTED_N} protected), "
        f"got {df.shape[0]}"
    )


# ---------------------------------------------------------------------------
# Test 4 — dry-run mode emits event but mutates nothing
# ---------------------------------------------------------------------------


def test_dry_run_mode_emits_event_no_mutation(
    iai_home, tmp_path, monkeypatch,
):
    """With `IAI_MCP_ERASURE_DRY_RUN=true` the event is emitted with the
    correct counts AND `dry_run_mode=True`, but no row is mutated
    (every row keeps `tombstoned_at IS NULL`). Pins the dry-run path.
    """
    # Explicit dry-run override. Note this test does NOT use the
    # `pipeline` fixture because that fixture sets DRY_RUN=false.
    monkeypatch.setenv("IAI_MCP_ERASURE_DRY_RUN", "true")
    monkeypatch.setattr(
        "iai_mcp.sleep_pipeline._utc_now", lambda: FROZEN_NOW,
    )

    store = MemoryStore()
    cohort_ids = _build_three_cohort_store(store, FROZEN_NOW)

    pipe = SleepPipeline(
        store=store,
        lifecycle_state_path=tmp_path / "lifecycle_state.json",
    )

    ok, payload = pipe._step_erasure_agent(None)
    assert ok is True, payload
    assert payload.get("dry_run") is True, (
        f"dry-run env var should land in payload, got {payload}"
    )
    assert payload.get("count_quarantined") == LOW_UTILITY_N, (
        f"dry-run must still count the eligibility set "
        f"(expected {LOW_UTILITY_N}), got {payload}"
    )

    # Zero mutation: every row still has tombstoned_at IS NULL.
    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    assert df.shape[0] == TOTAL_N
    for _, row in df.iterrows():
        assert not _is_tombstoned(row.to_dict()), (
            f"dry-run wrote a tombstone on row id={row.get('id')}; "
            f"mutation path must be inert when dry_run=True (R7)"
        )

    # Event surface: at least one erasure_agent_pass row with dry_run_mode=True.
    events = query_events(store, kind="erasure_agent_pass", limit=10)
    assert len(events) >= 1, (
        f"no erasure_agent_pass event emitted in dry-run mode, "
        f"got events={events}"
    )
    body = events[0]["data"]
    assert body.get("dry_run_mode") is True, body
    assert body.get("count_quarantined") == LOW_UTILITY_N, body
    # Sanity ping on cohort_ids so the fixture-builder regression
    # (e.g., empty low_utility list) is surfaced here rather than in
    # the count_quarantined assertion above.
    assert len(cohort_ids["low_utility"]) == LOW_UTILITY_N


# ---------------------------------------------------------------------------
# Test 5 — event body shape + uniqueness (wiring proof)
# ---------------------------------------------------------------------------


def test_erasure_event_body_shape_and_uniqueness(pipeline):
    """Exactly one `erasure_agent_pass` event per `_step_erasure_agent`
    invocation, body carries the 5 required typed fields, and
    the values match the synthetic fixture's known cohort sizes. The
    verifier-phase `events_query(kind='erasure_agent_pass')` smoke
    check rides on the same emit path proven here.
    """
    pipe, store, _ = pipeline

    ok, _ = pipe._step_erasure_agent(None)
    assert ok is True

    events = query_events(store, kind="erasure_agent_pass", limit=10)
    assert len(events) == 1, (
        f"exactly one erasure_agent_pass event per pass (R5), "
        f"got {len(events)} -> {events}"
    )

    body = events[0]["data"]

    # 5 required keys present.
    required_keys = {
        "count_quarantined",
        "count_dropped",
        "total_records_after",
        "threshold_used",
        "dry_run_mode",
    }
    missing = required_keys - set(body.keys())
    assert not missing, (
        f"erasure_agent_pass body missing required keys {sorted(missing)}; "
        f"got body={body}"
    )

    # Field types.
    assert isinstance(body["count_quarantined"], int), body
    assert isinstance(body["count_dropped"], int), body
    assert isinstance(body["total_records_after"], int), body
    assert isinstance(body["threshold_used"], float), body
    assert isinstance(body["dry_run_mode"], bool), body

    # Values pinned to the synthetic fixture.
    assert body["count_quarantined"] == LOW_UTILITY_N, body
    # No prior OPTIMIZE_LANCE pass has run, so count_dropped sources to 0.
    assert body["count_dropped"] == 0, body
    # Tombstoning doesn't delete rows — table count is unchanged by the step.
    assert body["total_records_after"] == TOTAL_N, body
    # Default IAI_MCP_ERASURE_CENTRALITY_THRESHOLD.
    assert body["threshold_used"] == 0.02, body
    # `pipeline` fixture explicitly disables dry-run.
    assert body["dry_run_mode"] is False, body
