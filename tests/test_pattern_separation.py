"""Pattern-separation regression suite.

Inline deterministic synthetic-embedding fixtures + six behavioural / static
tests that pin the pattern_separation_gate contract end-to-end on a real
MemoryStore. The gate runs INSIDE MemoryStore.insert, so every
test exercises the production write path — no SleepPipeline involved.

Deterministic embeddings: tests construct unit 4d vectors by hand so
cosine arithmetic is exact and cheap. The bge-small SDK is NEVER called from
this module — no `Embedder()` instantiation, no `sentence_transformers`
import, no `torch` import.

Test functions:

1. ``test_near_duplicate_cohort_collapses_to_reinforce`` — caller-transparent
   merge: 10 identical inserts collapse to 1 row, every call returns the same
   id, reinforce_record self-loop weight accumulates above the conservative
   floor.
2. ``test_link_seeding_creates_pattern_separation_seed_edge`` — positive
   case: cos=0.80 second insert seeds exactly one ``pattern_separation_seed``
   edge with weight ``link_initial_weight`` (default 0.10).
3. ``test_independent_insert_no_edge_seeded`` — negative case: cos<0.70
   second insert leaves the ``pattern_separation_seed`` edge set empty.
4. ``test_dry_run_preserves_regular_insert`` — emit-but-no-mutate: 10
   identical inserts under DRY_RUN=true produce 10 rows, zero seed edges,
   10 ``pattern_separation_pass`` events flagging ``dry_run_mode=True``.
5. ``test_event_body_shape_and_field_types`` — 8-field schema with per-field
   type checks.
6. ``test_gate_does_not_mutate_record_embedding`` — static + behavioural:
   no ``record.embedding =`` assignment exists inside either gate body OR
   insert body; gate call leaves ``rec.embedding`` value-unchanged even when
   matching hits exist.

All tests use the ``fresh_store`` fixture per test (tmp_path scope) so no
inter-test state leaks. The autouse ``_crypto_passphrase_env`` fixture in
``tests/conftest.py`` covers the AES-256-GCM events-table read path. The
autouse ``_reset_patsep_env`` fixture below clears every IAI_MCP_PATSEP_*
env var at test start so each test re-sets only what it needs.
"""
from __future__ import annotations

# Standard-library imports first so optional iai_mcp.* imports below fail
# loud with a clear ImportError if the package layout changes.
import inspect
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.daemon import PatSepConfig, _load_patsep_config
from iai_mcp.events import query_events
from iai_mcp.store import (
    EDGE_TYPES,
    EDGES_TABLE,
    RECORDS_TABLE,
    GateAction,
    MemoryStore,
)
from iai_mcp.types import MemoryRecord


# Module-level constants. EMBED_DIM=4 keeps the synthetic vectors short
# enough for hand-construction (just first 2 cells carry weight; cells 3-4
# stay 0 for the link cohort, OR carry a unit dim for the orthogonal cohort).
# Thresholds mirror the production defaults so the tests assert against the
# same values the production code reads at call time.
EMBED_DIM = 4
REFERENCE_EMBEDDING: list[float] = [1.0, 0.0, 0.0, 0.0]
NEAR_DUP_DEFAULT: float = 0.92
LINK_DEFAULT: float = 0.70
LINK_WEIGHT_DEFAULT: float = 0.10

# Canonical UUID regex for the event-body type check (test 5). Locked to
# RFC-4122 v4 shape: 8-4-4-4-12 lowercase hex.
_UUID_REGEX: re.Pattern[str] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


# Build a unit 4d vector whose dot-product against REFERENCE_EMBEDDING equals
# ``cos_target`` exactly. With unit-norm vectors the cosine collapses to the
# dot product, which equals the first component of ``v`` since REFERENCE has
# only the first cell non-zero. The second cell carries the remaining norm
# so the full vector stays on the unit sphere.
def _make_embedding_at_cosine(
    cos_target: float, embed_dim: int = EMBED_DIM,
) -> list[float]:
    if not (-1.0 <= cos_target <= 1.0):
        raise ValueError(
            f"cos_target must be in [-1.0, 1.0], got {cos_target}"
        )
    if embed_dim < 2:
        raise ValueError(
            f"embed_dim must be >= 2, got {embed_dim}"
        )
    residual = math.sqrt(max(0.0, 1.0 - cos_target * cos_target))
    return [cos_target, residual] + [0.0] * (embed_dim - 2)


# Build a fully-populated MemoryRecord with sensible defaults for every
# required field. ``literal_surface`` defaults to a generic ``alice...``
# string per project the project convention ("never `Alice` as example data"). The
# ``**overrides`` slot lets a test bump ``pinned=True`` or any other field
# without re-listing the full keyword set.
def _make_record(
    *,
    embedding: list[float],
    tier: str = "episodic",
    literal_surface: str = "alice prefers tea over coffee",
    **overrides,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    base = dict(
        id=uuid4(),
        tier=tier,
        literal_surface=literal_surface,
        aaak_index="",
        embedding=list(embedding),
        community_id=None,
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
    )
    base.update(overrides)
    return MemoryRecord(**base)


# Per-test MemoryStore. Follows the store-construction pattern:
# - path= + user_id= kwargs (NOT root=/embed_dim= which do not exist),
# - read_consistency_interval=timedelta(seconds=0) so tbl.count_rows() reflects
# the just-completed write (the tests assert against count_rows shortly
# after insert),
# - IAI_MCP_EMBED_DIM is set to 4 via the autouse env-reset fixture so the
# records table schema lands at 4d. Constructing the store without that
# env var would default to 384d (bge-small-en-v1.5 native).
def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        path=str(tmp_path / "iai-mcp"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )


# Static-source-check support. inspect.getsource returns the literal text
# of the method body, including any ``record.embedding =...`` assignment if
# one ever crept in. Test 6 greps both sources with a regex.
def _gate_body_source() -> str:
    return inspect.getsource(MemoryStore.pattern_separation_gate)


def _insert_body_source() -> str:
    return inspect.getsource(MemoryStore.insert)


# Autouse fixture order matters: pytest runs autouse fixtures alphabetically
# unless dependencies pin the order. _crypto_passphrase_env (from conftest.py)
# runs first by name; _reset_patsep_env below runs after. Both finalize at
# test teardown via monkeypatch unwinding.
@pytest.fixture(autouse=True)
def _reset_patsep_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clear all IAI_MCP_PATSEP_* env vars + pin the store path + embed dim.

    Five jobs:
    1. Delete every IAI_MCP_PATSEP_* override so each test starts from the
       canonical defaults (near=0.92, link=0.70, weight=0.10, k=8).
       Tests that need to override re-set after this fixture runs.
    2. Pin IAI_MCP_EMBED_DIM=4 so the per-test store schema lands at 4d. The
       global default is 384d (bge-small-en-v1.5);
       leaving it unset would make the 4d test vectors fail the
       ``len(record.embedding) != self._embed_dim`` guard inside insert().
    3. Pin IAI_MCP_STORE=tmp_path so per-test stores do not scribble on the
       user's real on-disk store. MemoryStore.__init__ reads IAI_MCP_STORE
       and OVERRIDES the path= kwarg if it is set; the user's shell has it set
       globally per project the project convention.
    4. Pin IAI_MCP_EMBED_MODEL absent so _resolve_embed_dim falls through to
       the IAI_MCP_EMBED_DIM=4 path even on machines where this is exported.
    5. Pin PYTEST_CURRENT_TEST absent? -- NO, leave it. Pytest sets it
       per-test; the gate's dry-run-default-true under pytest then activates
       only when the test does NOT explicitly setenv IAI_MCP_PATSEP_DRY_RUN.
    """
    for var in (
        "IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT",
        "IAI_MCP_PATSEP_TOP_K",
        "IAI_MCP_PATSEP_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(EMBED_DIM))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp"))


# Per-test fresh MemoryStore rooted at tmp_path. tmp_path is auto-cleaned
# by pytest; no explicit teardown.
@pytest.fixture
def fresh_store(tmp_path: Path) -> MemoryStore:
    return _make_store(tmp_path)


# ---------------------------------------------------------------------------
# Six regression tests.
# ---------------------------------------------------------------------------


# Helper: fetch every ``pattern_separation_pass`` event in INSERT ORDER
# (ts-ascending). ``query_events`` returns newest-first; the gate emits one
# event per ``insert()`` call so reversing the slice walks the test's call
# sequence back to chronological order. Limit deliberately wide so tests
# never miss an event.
def _events_in_insert_order(store: MemoryStore) -> list[dict]:
    events = query_events(store, kind="pattern_separation_pass", limit=1000)
    return list(reversed(events))


# ---------------------------------------------------------------------------
# Test 1 — near-duplicate cohort collapses to a single record + reinforce
# ---------------------------------------------------------------------------


def test_near_duplicate_cohort_collapses_to_reinforce(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Insert A, then 10 records with the IDENTICAL embedding. Every SKIP
    branch mutates ``record.id`` to ``A_id`` (caller-transparent merge)
    and reinforces the existing record via a self-loop ``hebbian`` edge.

    Pins the near-duplicate short-circuit end-to-end on the
    production MemoryStore.insert path.
    """
    # Disable the pytest-aware dry-run default so the mutation path runs.
    # _load_patsep_config sees this env var on its next call (CALL-ON-DEMAND
    # per).
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    A_id = rec_a.id
    fresh_store.insert(rec_a)

    # Drive 10 near-duplicate inserts. Each rec_dup gets a fresh uuid; the
    # SKIP path must mutate rec_dup.id to A_id BEFORE insert() returns so
    # the caller observes the merged-into id transparently.
    for _ in range(10):
        rec_dup = _make_record(embedding=REFERENCE_EMBEDDING)
        pre_id = rec_dup.id
        fresh_store.insert(rec_dup)
        assert rec_dup.id == A_id, (
            f"SKIP path must mutate rec.id to existing-record uuid; "
            f"got rec_dup.id={rec_dup.id}, expected A_id={A_id}"
        )
        assert rec_dup.id != pre_id, (
            f"caller-transparent merge: rec.id must change away from the "
            f"fresh uuid {pre_id} after a SKIP collapse"
        )

    # Exactly one row in the records table: A. The 10 dups merged in.
    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 1, (
        f"near-duplicate collapse must keep exactly 1 row; "
        f"got {tbl.count_rows()}"
    )

    # reinforce_record(A_id) was called 10 times — each call boosts the
    # self-loop hebbian edge by delta=0.1. 10 x 0.1 = 1.0 in exact
    # arithmetic but float32 accumulation in boost_edges (cur + accum_delta,
    # repeated) can land at ~0.99999... — soften the floor to 0.9 to avoid
    # float-boundary flake without losing the cohort-size signal.
    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    self_loops = edges_df[
        (edges_df["edge_type"] == "hebbian")
        & (edges_df["src"] == str(A_id))
        & (edges_df["dst"] == str(A_id))
    ]
    assert len(self_loops) >= 1, (
        f"reinforce_record must persist a self-loop hebbian edge; "
        f"got rows={edges_df.to_dict('records')}"
    )
    self_loop_weight = float(self_loops["weight"].iloc[0])
    assert self_loop_weight >= 0.9, (
        f"10 reinforces x delta=0.1 must accumulate to ~1.0; "
        f"got weight={self_loop_weight}"
    )

    # 11 pattern_separation_pass events total: 1 'insert' for A (empty store
    # at the time, no hits), 10 'skip' for the dups. Each skip references
    # A_id and reports cos >= 0.99 against the identical REFERENCE vector.
    events = _events_in_insert_order(fresh_store)
    assert len(events) == 11, (
        f"expected 11 pattern_separation_pass events (1 insert + 10 skip), "
        f"got {len(events)}"
    )
    first_body = events[0]["data"]
    assert first_body["action"] == "insert", first_body
    assert first_body["near_dup_hit_id"] is None, first_body
    for idx, ev in enumerate(events[1:], start=1):
        body = ev["data"]
        assert body["action"] == "skip", (
            f"event {idx} must be a skip; got body={body}"
        )
        assert body["near_dup_hit_id"] == str(A_id), (
            f"event {idx} must reference A_id={A_id} as the near-dup hit; "
            f"got body={body}"
        )
        assert body["dry_run_mode"] is False, body
        assert body["near_dup_cos"] is not None and body["near_dup_cos"] >= 0.99, body


# ---------------------------------------------------------------------------
# Test 2 — link-band insert seeds exactly one pattern_separation_seed edge
# ---------------------------------------------------------------------------


def test_link_seeding_creates_pattern_separation_seed_edge(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Insert A=REFERENCE, then B at cos=0.80 to A (strictly inside the link
    band [0.70, 0.92)). The gate must (a) keep B as a fresh insert, (b) seed
    a single ``pattern_separation_seed`` edge between A and B with weight
    ``link_initial_weight`` (default 0.10).

    Pins the link-band edge seeding end-to-end.
    """
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    A_id = rec_a.id
    fresh_store.insert(rec_a)

    rec_b = _make_record(embedding=_make_embedding_at_cosine(0.80))
    B_id = rec_b.id
    fresh_store.insert(rec_b)

    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 2, (
        f"link-band insert must NOT collapse; both A and B land. "
        f"Got {tbl.count_rows()} rows."
    )

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    seed_edges = edges_df[edges_df["edge_type"] == "pattern_separation_seed"]
    assert len(seed_edges) == 1, (
        f"expected exactly 1 pattern_separation_seed edge between A and B, "
        f"got {len(seed_edges)} -> {seed_edges.to_dict('records')}"
    )

    # boost_edges canonicalises (src, dst) via sorted tuple, so the edge's
    # endpoints come back as ``sorted([str(A_id), str(B_id)])``. Compare
    # against that canonical pair regardless of seeding direction.
    edge_row = seed_edges.iloc[0]
    actual_endpoints = tuple(sorted([edge_row["src"], edge_row["dst"]]))
    expected_endpoints = tuple(sorted([str(A_id), str(B_id)]))
    assert actual_endpoints == expected_endpoints, (
        f"seed-edge endpoints must canonicalise to {expected_endpoints}; "
        f"got {actual_endpoints}"
    )

    weight = float(edge_row["weight"])
    assert abs(weight - LINK_WEIGHT_DEFAULT) < 1e-6, (
        f"seed-edge weight must equal link_initial_weight={LINK_WEIGHT_DEFAULT} "
        f"to within 1e-6; got weight={weight}"
    )

    events = _events_in_insert_order(fresh_store)
    assert len(events) == 2, (
        f"expected 2 pattern_separation_pass events (one per insert), "
        f"got {len(events)}"
    )
    body_b = events[1]["data"]
    assert body_b["action"] == "insert", body_b
    assert body_b["edges_seeded"] == 1, body_b
    assert body_b["near_dup_hit_id"] is None, body_b


# ---------------------------------------------------------------------------
# Test 3 — independent insert (cos < link threshold) seeds no seed-edge
# ---------------------------------------------------------------------------


def test_independent_insert_no_edge_seeded(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Insert A=REFERENCE, then C orthogonal to A (cos(A,C)=0). The gate must
    NOT seed any ``pattern_separation_seed`` edge because every cosine sits
    strictly below the link threshold.

    Uses an orthogonal unit vector ([0, 0, 1, 0]) in a free dimension so the
    test stays robust against future store insertions that might land between
    A and C — there is no other record in this store anyway, but the
    construction is intentionally clean for downstream copy-paste reuse.

    Pins the negative case.
    """
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    fresh_store.insert(rec_a)

    # Orthogonal in a free dimension: dot(A, C) = 0 exactly. Stays a unit
    # vector so the store's cosine path returns cos=0.0 without floating-
    # point drift. cos=0.0 < link_threshold=0.70 so no seed edge fires.
    orthogonal = [0.0, 0.0, 1.0, 0.0]
    rec_c = _make_record(embedding=orthogonal)
    fresh_store.insert(rec_c)

    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 2, (
        f"independent insert must NOT collapse; both A and C land. "
        f"Got {tbl.count_rows()} rows."
    )

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    seed_edges = edges_df[edges_df["edge_type"] == "pattern_separation_seed"]
    assert len(seed_edges) == 0, (
        f"cos=0.0 < link_threshold=0.70 must NOT seed any seed-edge; "
        f"got {len(seed_edges)} -> {seed_edges.to_dict('records')}"
    )

    events = _events_in_insert_order(fresh_store)
    assert len(events) == 2, (
        f"expected 2 pattern_separation_pass events, got {len(events)}"
    )
    body_c = events[1]["data"]
    assert body_c["action"] == "insert", body_c
    assert body_c["edges_seeded"] == 0, body_c
    assert body_c["near_dup_hit_id"] is None, body_c


# ---------------------------------------------------------------------------
# Test 4 — dry-run mode emits 'skip' events but performs neither merge nor
# edge-seed; every same-vector insert lands as its own row
# ---------------------------------------------------------------------------


def test_dry_run_preserves_regular_insert(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under IAI_MCP_PATSEP_DRY_RUN=true, 10 identical inserts produce 10
    rows (no collapse), zero ``pattern_separation_seed`` edges (no edge
    seed), and 10 ``pattern_separation_pass`` events flagging
    ``dry_run_mode=True``. The first event is an 'insert' (empty store, no
    near-dup hit); the other 9 are 'skip' events that fell through to the
    regular insert under dry-run.

    Pins the dry-run emit-but-no-mutate path end-to-end.
    """
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "true")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    fresh_store.insert(rec_a)
    inserted_ids: set[str] = {str(rec_a.id)}

    for _ in range(9):
        rec_dup = _make_record(embedding=REFERENCE_EMBEDDING)
        pre_id = rec_dup.id
        fresh_store.insert(rec_dup)
        # Dry-run: rec.id MUST stay the fresh uuid (no merge mutation).
        assert rec_dup.id == pre_id, (
            f"dry-run R7: rec.id must NOT be mutated; "
            f"got {rec_dup.id}, expected fresh {pre_id}"
        )
        inserted_ids.add(str(rec_dup.id))

    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 10, (
        f"dry-run R7: all 10 inserts must land; got {tbl.count_rows()}"
    )

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    seed_edges = edges_df[edges_df["edge_type"] == "pattern_separation_seed"]
    assert len(seed_edges) == 0, (
        f"dry-run R7: zero seed edges; got {len(seed_edges)}"
    )

    events = _events_in_insert_order(fresh_store)
    assert len(events) == 10, (
        f"dry-run R7: exactly 10 pattern_separation_pass events; "
        f"got {len(events)}"
    )

    # First event: empty store at A's insert -> 'insert' branch, no hit.
    first_body = events[0]["data"]
    assert first_body["action"] == "insert", first_body
    assert first_body["near_dup_hit_id"] is None, first_body
    assert first_body["dry_run_mode"] is True, first_body

    # Following 9 events: 'skip' branch fell through under dry-run. The
    # near-dup hit can be ANY already-inserted record (the store's tiebreak
    # on identical cosines is not contractually stable when N>1), so accept
    # any inserted id and require cosine >= 0.99.
    for idx, ev in enumerate(events[1:], start=1):
        body = ev["data"]
        assert body["action"] == "skip", (
            f"event {idx}: dry-run must still report the SKIP decision; "
            f"got body={body}"
        )
        assert body["dry_run_mode"] is True, body
        assert body["near_dup_hit_id"] in inserted_ids, (
            f"event {idx}: near_dup_hit_id must reference an already-inserted "
            f"record; got {body['near_dup_hit_id']!r}, inserted={inserted_ids}"
        )
        assert body["near_dup_cos"] is not None and body["near_dup_cos"] >= 0.99, body


# ---------------------------------------------------------------------------
# Test 5 — event-body shape: exactly 8 keys, per-field type checks
# ---------------------------------------------------------------------------


def test_event_body_shape_and_field_types(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive both branches in one fixture (insert A, near-dup A2 at cos=1.0,
    link-target B at cos=0.80, independent C orthogonal to both). Walk every
    emitted ``pattern_separation_pass`` event and assert the body has EXACTLY
    the 8 documented keys with the correct types.

    Pins the event-body contract.
    """
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    A_id = rec_a.id
    fresh_store.insert(rec_a)

    rec_a2 = _make_record(embedding=REFERENCE_EMBEDDING)
    fresh_store.insert(rec_a2)  # SKIP-and-merge into A

    rec_b = _make_record(embedding=_make_embedding_at_cosine(0.80))
    B_id = rec_b.id
    fresh_store.insert(rec_b)  # link-band INSERT, edges_seeded=1

    # Orthogonal to BOTH A and B: cos(A, C) = 0, cos(B, C) = 0. Both below
    # link_threshold=0.70 so C lands as a clean independent INSERT with
    # edges_seeded=0. (Without the free-dimension trick — e.g. using
    # _make_embedding_at_cosine(0.40) — cos(B, C) would land at ~0.87 and
    # silently seed a B<->C edge, polluting the event body assertions.)
    rec_c = _make_record(embedding=[0.0, 0.0, 1.0, 0.0])
    fresh_store.insert(rec_c)

    events = _events_in_insert_order(fresh_store)
    assert len(events) == 4, (
        f"expected exactly 4 pattern_separation_pass events (1 per insert), "
        f"got {len(events)}"
    )

    expected_keys = {
        "action",
        "near_dup_hit_id",
        "near_dup_cos",
        "edges_seeded",
        "top_k_probed",
        "threshold_near_dup",
        "threshold_link",
        "dry_run_mode",
    }

    # Per-event strict shape + type checks. Walk every event so a regression
    # in any one branch surfaces here rather than at a downstream smoke.
    for idx, ev in enumerate(events):
        body = ev["data"]
        assert set(body.keys()) == expected_keys, (
            f"event {idx} body keys must equal {sorted(expected_keys)}; "
            f"got extra={set(body.keys()) - expected_keys}, "
            f"missing={expected_keys - set(body.keys())}, body={body}"
        )
        # Per-field type checks.
        assert isinstance(body["action"], str) and body["action"] in {"insert", "skip"}, body
        # near_dup_hit_id: None on the empty-store INSERT, canonical UUID string elsewhere.
        if body["near_dup_hit_id"] is not None:
            assert isinstance(body["near_dup_hit_id"], str), body
            assert _UUID_REGEX.match(body["near_dup_hit_id"]), (
                f"event {idx} near_dup_hit_id must be canonical UUID; "
                f"got {body['near_dup_hit_id']!r}"
            )
        # near_dup_cos: None when no hits, float otherwise. bool is a subclass
        # of int in Python; assert against numbers.Real-like by isinstance(..., float).
        if body["near_dup_cos"] is not None:
            assert isinstance(body["near_dup_cos"], float), body
        assert isinstance(body["edges_seeded"], int) and not isinstance(body["edges_seeded"], bool), body
        assert body["edges_seeded"] >= 0, body
        assert isinstance(body["top_k_probed"], int) and not isinstance(body["top_k_probed"], bool), body
        assert body["top_k_probed"] >= 0, body
        assert isinstance(body["threshold_near_dup"], float), body
        assert abs(body["threshold_near_dup"] - 0.92) < 1e-9, body
        assert isinstance(body["threshold_link"], float), body
        assert abs(body["threshold_link"] - 0.70) < 1e-9, body
        assert isinstance(body["dry_run_mode"], bool), body
        assert body["dry_run_mode"] is False, body

    # Cross-event invariants. Walk in insert order: A (empty store), A2 (skip
    # into A), B (link to A), C (orthogonal — fresh insert, no edges).
    a_body = events[0]["data"]
    assert a_body["action"] == "insert", a_body
    assert a_body["near_dup_hit_id"] is None, a_body
    assert a_body["edges_seeded"] == 0, a_body

    a2_body = events[1]["data"]
    assert a2_body["action"] == "skip", a2_body
    assert a2_body["near_dup_hit_id"] == str(A_id), a2_body
    assert a2_body["near_dup_cos"] is not None and a2_body["near_dup_cos"] >= 0.99, a2_body

    b_body = events[2]["data"]
    assert b_body["action"] == "insert", b_body
    assert b_body["edges_seeded"] == 1, b_body
    assert b_body["near_dup_hit_id"] is None, b_body

    c_body = events[3]["data"]
    assert c_body["action"] == "insert", c_body
    assert c_body["edges_seeded"] == 0, c_body
    assert c_body["near_dup_hit_id"] is None, c_body


# ---------------------------------------------------------------------------
# Test 6 — gate (and insert) MUST NOT mutate record.embedding
# ---------------------------------------------------------------------------


def test_gate_does_not_mutate_record_embedding(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two-layer pin:

    1. Static source check: greppable assignment ``record.embedding =...``
       must NOT appear in either ``MemoryStore.pattern_separation_gate`` body
       or ``MemoryStore.insert`` body. The regex tolerates whitespace before
       ``=`` so future formatting changes don't false-pass.
    2. Behavioural check: call ``pattern_separation_gate(rec)`` on an empty
       store AND on a store with a matching hit; ``rec.embedding`` value
       (list equality) must be unchanged after the call.
    """
    # (1) Static source check. Capture both bodies via inspect.getsource and
    # grep for assignment. The regex specifically matches ``record.embedding``
    # followed by zero-or-more whitespace and a literal ``=`` -- and EXCLUDES
    # ``==`` to avoid catching equality comparisons (which are read-only).
    assign_pattern = re.compile(r"record\.embedding\s*=(?!=)")
    gate_src = _gate_body_source()
    insert_src = _insert_body_source()
    gate_hits = assign_pattern.findall(gate_src)
    insert_hits = assign_pattern.findall(insert_src)
    assert gate_hits == [], (
        f"R4: pattern_separation_gate body MUST NOT assign to record.embedding; "
        f"found {len(gate_hits)} match(es) -> {gate_hits}"
    )
    assert insert_hits == [], (
        f"R4: MemoryStore.insert body MUST NOT assign to record.embedding; "
        f"found {len(insert_hits)} match(es) -> {insert_hits}"
    )

    # (2a) Behavioural: empty store + matching hit absent. The gate must
    # return (INSERT, []) without touching record.embedding.
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")
    rec_isolated = _make_record(embedding=REFERENCE_EMBEDDING)
    original_embedding = list(rec_isolated.embedding)  # defensive snapshot
    action_empty, payload_empty = fresh_store.pattern_separation_gate(rec_isolated)
    assert rec_isolated.embedding == original_embedding, (
        f"R4: gate on empty store mutated record.embedding; "
        f"original={original_embedding}, after={rec_isolated.embedding}"
    )
    assert action_empty == GateAction.INSERT, (
        f"empty store must return INSERT, got {action_empty}"
    )
    assert payload_empty == [], (
        f"empty store INSERT payload must be []; got {payload_empty}"
    )

    # (2b) Behavioural: store has a matching record. Insert A first via the
    # production path, then call the gate directly on a fresh near-duplicate
    # record. The gate must return (SKIP, A_id) without mutating
    # rec_new.embedding.
    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    fresh_store.insert(rec_a)

    rec_new = _make_record(embedding=REFERENCE_EMBEDDING)
    original_new_embedding = list(rec_new.embedding)
    action_hit, payload_hit = fresh_store.pattern_separation_gate(rec_new)
    assert rec_new.embedding == original_new_embedding, (
        f"R4: gate with matching hit mutated record.embedding; "
        f"original={original_new_embedding}, after={rec_new.embedding}"
    )
    assert action_hit == GateAction.SKIP, (
        f"matching hit must return SKIP, got {action_hit}"
    )
    assert isinstance(payload_hit, UUID), (
        f"SKIP payload must be the existing-record UUID; got {payload_hit!r}"
    )
    assert payload_hit == rec_a.id, (
        f"SKIP payload must equal A_id={rec_a.id}; got {payload_hit}"
    )
