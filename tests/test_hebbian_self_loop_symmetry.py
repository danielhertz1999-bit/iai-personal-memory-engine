"""Hebbian self-loop symmetrization contract.

Pre-fix asymmetry: `MemoryStore.insert` writes a `(rid, rid)` hebbian
self-loop only on the SKIP-dedup path (via `reinforce_record(existing_id)`
in store.py). The fresh-INSERT path does NOT write a self-loop. On a
bench store with 40 records, 30 were dedup-touched -> 30 self-loops; 10
were freshly inserted -> 0 self-loops. The 30/40 ratio is the dedup-rate
signature, NOT a deliberate semantic.

Post-fix: fresh-INSERT writes a hebbian self-loop at
`delta=link_initial_weight` (~0.1). Symmetric *presence* across all
records; Hebbian *weight* gradient preserved (dedup-touched records still
accumulate 10x weight vs fresh after 10 reinforces).

Test surface (6 cases):
1. `test_fresh_insert_writes_self_loop` — RED on un-patched source,
   GREEN after Task 2 (fresh-INSERT self-loop write).
2. `test_fresh_insert_self_loop_weight_matches_link_initial_weight` —
   weight = link_initial_weight on the single fresh insert.
3. `test_symmetry_post_migration_all_or_none` — backfill helper makes
   asymmetric -> all-records-have-self-loops state.
4. `test_symmetry_dry_run_reports_without_writing` — dry_run=True reports
   counts but does NOT mutate the edges table.
5. `test_phase11_1_dedup_test_smoke_post_fix` — explicit smoke that the
   dedup-path weight assertion in test_phase11_1_pattern_separation.py
   still holds post-fix (fresh 0.1 + 10x0.1 reinforces = ~1.1 >= 0.9).
6. `test_cli_subcommand_dry_run_smoke` — CLI handler smoke; dry_run
   returns 0 with JSON output containing the expected fields.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.store import (
    EDGES_TABLE,
    RECORDS_TABLE,
    MemoryStore,
)
from iai_mcp.types import MemoryRecord


# Constants mirror tests/test_phase11_1_pattern_separation.py.
# 4d synthetic vectors so cosine arithmetic stays exact + cheap; the
# patsep gate's near_dup/link thresholds (0.92/0.70 defaults) are
# easy to dodge with hand-crafted orthogonal vectors.
EMBED_DIM = 4
REFERENCE_EMBEDDING: list[float] = [1.0, 0.0, 0.0, 0.0]
LINK_INITIAL_WEIGHT_DEFAULT: float = 0.10


def _make_embedding_at_cosine(
    cos_target: float, embed_dim: int = EMBED_DIM,
) -> list[float]:
    """Build a unit `embed_dim`-d vector whose cosine to REFERENCE_EMBEDDING
    equals `cos_target`. Used to construct orthogonal vectors that bypass
    the patsep gate's near_dup + link bands."""
    if not (-1.0 <= cos_target <= 1.0):
        raise ValueError(f"cos_target out of range: {cos_target}")
    residual = math.sqrt(max(0.0, 1.0 - cos_target * cos_target))
    return [cos_target, residual] + [0.0] * (embed_dim - 2)


def _make_record(
    *,
    embedding: list[float],
    tier: str = "episodic",
    literal_surface: str = "alice prefers tea over coffee",
    **overrides,
) -> MemoryRecord:
    """Factory mirrors test_phase11_1_pattern_separation.py."""
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


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        path=str(tmp_path / "iai-mcp"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )


# Autouse: clear patsep env, force dry_run=false (so INSERT writes the
# self-loop), pin embed_dim=4 so the small synthetic vectors fit, scrub
# the store-path override the user's shell may export.
@pytest.fixture(autouse=True)
def _reset_patsep_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in (
        "IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT",
        "IAI_MCP_PATSEP_TOP_K",
    ):
        monkeypatch.delenv(var, raising=False)
    # CRITICAL: pytest sets PYTEST_CURRENT_TEST which makes patsep dry_run
    # default to True. Without this override the INSERT action emits the
    # event but writes no edges -> tests 1/2/3 silently false-negative.
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(EMBED_DIM))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp"))


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors test_capture_dedup_contract:41-53. Avoids real keyring hangs
    on the construction host when MemoryStore touches AES key acquisition."""
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )


@pytest.fixture
def fresh_store(tmp_path: Path) -> MemoryStore:
    return _make_store(tmp_path)


# Helper: count hebbian self-loops in the edges table.
def _self_loop_ids(store: MemoryStore) -> set[str]:
    edges_tbl = store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    if len(edges_df) == 0:
        return set()
    mask = (
        (edges_df["edge_type"] == "hebbian")
        & (edges_df["src"] == edges_df["dst"])
    )
    return set(edges_df.loc[mask, "src"].astype(str).tolist())


# ---------------------------------------------------------------------------
# Test 1 — fresh-INSERT writes a (rid, rid) hebbian self-loop
# ---------------------------------------------------------------------------


def test_fresh_insert_writes_self_loop(fresh_store: MemoryStore) -> None:
    """Five orthogonal fresh inserts produce five hebbian self-loops.

    PRE-FIX RED-witness: fresh-INSERT path does NOT write a self-loop;
    only the SKIP-dedup path (via reinforce_record) does. With 5
    orthogonal embeddings the gate stays INSERT for every record and
    NO reinforce_record call fires -> 0 self-loops in the edges table.

    POST-FIX (Task 2): fresh-INSERT writes (record.id, record.id) at
    delta=link_initial_weight on the same atomic write. 5 inserts -> 5
    self-loop rows.
    """
    record_ids: set[str] = set()
    # Use embeddings that are pairwise orthogonal (or near-orthogonal,
    # all well below link_threshold=0.70) so the patsep gate stays INSERT
    # for every record. Each vector lives on a distinct unit-axis.
    for i in range(5):
        emb = [0.0] * EMBED_DIM
        emb[i % EMBED_DIM] = 1.0
        # Tweak past axis 4 to avoid exact reuse if EMBED_DIM<5
        if i >= EMBED_DIM:
            emb[i % EMBED_DIM] = 0.5
            emb[(i + 1) % EMBED_DIM] = math.sqrt(1 - 0.25)
        rec = _make_record(embedding=emb, literal_surface=f"record-{i}")
        record_ids.add(str(rec.id))
        fresh_store.insert(rec)

    self_loops = _self_loop_ids(fresh_store)
    assert self_loops == record_ids, (
        f"fresh-INSERT path must write a (rid, rid) hebbian self-loop "
        f"per record. records={record_ids}, self_loops={self_loops}, "
        f"missing={record_ids - self_loops}"
    )


# ---------------------------------------------------------------------------
# Test 2 — fresh-INSERT self-loop weight == link_initial_weight
# ---------------------------------------------------------------------------


def test_fresh_insert_self_loop_weight_matches_link_initial_weight(
    fresh_store: MemoryStore,
) -> None:
    """A single fresh-INSERT produces a self-loop with weight ~0.1
    (link_initial_weight default). This is the weight
    choice: symmetric presence without erasing the Hebbian gradient
    between fresh and frequently-reinforced records.
    """
    rec = _make_record(embedding=REFERENCE_EMBEDDING)
    fresh_store.insert(rec)

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    self_loops = edges_df[
        (edges_df["edge_type"] == "hebbian")
        & (edges_df["src"] == str(rec.id))
        & (edges_df["dst"] == str(rec.id))
    ]
    assert len(self_loops) == 1, (
        f"exactly one self-loop expected after one fresh insert; "
        f"got {len(self_loops)}: {edges_df.to_dict('records')}"
    )
    weight = float(self_loops["weight"].iloc[0])
    # link_initial_weight default is 0.10 per daemon_config
    # _PATSEP_DEFAULT_LINK_INITIAL_WEIGHT; allow tiny float drift.
    assert abs(weight - LINK_INITIAL_WEIGHT_DEFAULT) < 1e-6, (
        f"fresh-INSERT self-loop weight must be link_initial_weight "
        f"(~{LINK_INITIAL_WEIGHT_DEFAULT}); got {weight}"
    )


# ---------------------------------------------------------------------------
# Test 3 — symmetrize_self_loops backfills missing self-loops (ALL or NONE)
# ---------------------------------------------------------------------------


def test_symmetry_post_migration_all_or_none(
    fresh_store: MemoryStore,
) -> None:
    """Construct a deliberately asymmetric pre-migration state by deleting
    self-loops from a subset of records, then verify symmetrize_self_loops
    backfills the missing ones.

    Post-Task-2 every fresh insert writes a self-loop, so we have to
    simulate the pre-fix asymmetry by deleting rows from the edges table
    after the fact. This mirrors what an upgraded store looks like:
    pre- records have self-loops only if they were dedup-touched;
    post- records all have self-loops. The migration brings the
    legacy records up to parity.
    """
    from iai_mcp.maintenance import symmetrize_self_loops

    # Step 1: insert 6 records with orthogonal embeddings (no dedup, all
    # fresh inserts -> each writes a self-loop post-Task-2).
    record_ids: list[str] = []
    for i in range(6):
        emb = [0.0] * EMBED_DIM
        emb[i % EMBED_DIM] = 1.0
        if i >= EMBED_DIM:
            emb[i % EMBED_DIM] = 0.5
            emb[(i + 1) % EMBED_DIM] = math.sqrt(1 - 0.25)
        rec = _make_record(embedding=emb, literal_surface=f"record-{i}")
        record_ids.append(str(rec.id))
        fresh_store.insert(rec)

    # Step 2: delete self-loops for the first 4 records to simulate the
    # pre- asymmetric state. Use edges_tbl.delete() with a where
    # predicate string per the store API.
    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    missing_ids = record_ids[:4]
    for rid in missing_ids:
        edges_tbl.delete(
            f"edge_type == 'hebbian' AND src == '{rid}' AND dst == '{rid}'"
        )

    # Sanity: pre-migration count should be 2 (records 4 + 5 only).
    pre_self_loops = _self_loop_ids(fresh_store)
    assert len(pre_self_loops) == 2, (
        f"setup invariant: 6 records - 4 deleted = 2 self-loops; "
        f"got {len(pre_self_loops)}: {pre_self_loops}"
    )

    # Step 3: run the migration in apply mode.
    result = symmetrize_self_loops(fresh_store, dry_run=False)

    assert result["dry_run"] is False, result
    assert result["records_total"] == 6, result
    assert result["self_loops_present"] == 2, result
    assert result["self_loops_pending"] == 4, result
    assert result["self_loops_inserted"] == 4, result

    # Step 4: post-migration symmetry — every record has a self-loop.
    post_self_loops = _self_loop_ids(fresh_store)
    assert post_self_loops == set(record_ids), (
        f"post-migration symmetry violated. records={set(record_ids)}, "
        f"self_loops={post_self_loops}, missing="
        f"{set(record_ids) - post_self_loops}"
    )


# ---------------------------------------------------------------------------
# Test 4 — dry_run reports counts WITHOUT writing
# ---------------------------------------------------------------------------


def test_symmetry_dry_run_reports_without_writing(
    fresh_store: MemoryStore,
) -> None:
    """dry_run=True returns the same counts dict but DOES NOT modify the
    edges table. Verifies the dry-run path is a pure read.
    """
    from iai_mcp.maintenance import symmetrize_self_loops

    # Same setup as Test 3: 6 fresh inserts, delete 4 self-loops, leave
    # 2 records (the last two) with self-loops present.
    record_ids: list[str] = []
    for i in range(6):
        emb = [0.0] * EMBED_DIM
        emb[i % EMBED_DIM] = 1.0
        if i >= EMBED_DIM:
            emb[i % EMBED_DIM] = 0.5
            emb[(i + 1) % EMBED_DIM] = math.sqrt(1 - 0.25)
        rec = _make_record(embedding=emb, literal_surface=f"record-{i}")
        record_ids.append(str(rec.id))
        fresh_store.insert(rec)

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    for rid in record_ids[:4]:
        edges_tbl.delete(
            f"edge_type == 'hebbian' AND src == '{rid}' AND dst == '{rid}'"
        )

    pre_self_loops_count = len(_self_loop_ids(fresh_store))
    assert pre_self_loops_count == 2

    result = symmetrize_self_loops(fresh_store, dry_run=True)

    assert result["dry_run"] is True, result
    assert result["records_total"] == 6, result
    assert result["self_loops_present"] == 2, result
    assert result["self_loops_pending"] == 4, result
    # CRITICAL: dry-run MUST NOT write — self_loops_inserted stays 0.
    assert result["self_loops_inserted"] == 0, result

    # Edges table unchanged: still 2 self-loops.
    post_self_loops_count = len(_self_loop_ids(fresh_store))
    assert post_self_loops_count == pre_self_loops_count, (
        f"dry_run must NOT mutate the edges table. "
        f"pre={pre_self_loops_count}, post={post_self_loops_count}"
    )


# ---------------------------------------------------------------------------
# Test 5 — dedup-path weight contract still holds (smoke regression)
# ---------------------------------------------------------------------------


def test_phase11_1_dedup_test_smoke_post_fix(
    fresh_store: MemoryStore,
) -> None:
    """Explicit smoke that the existing dedup-path self-loop weight
    contract (test_phase11_1_pattern_separation.py, assert
    weight >= 0.9 after 10 reinforces) STILL HOLDS.

    Math: post-Task-2 fresh-INSERT writes self-loop at 0.1. 10 SKIP-dedup
    inserts call reinforce_record(A_id) which adds 0.1 each. Final
    weight = 0.1 (fresh) + 10*0.1 (reinforces) = 1.1 >= 0.9. SAFE.

    The original test in test_phase11_1_pattern_separation.py is the
    authoritative gate; this is a redundant local smoke for fast
    iteration during development.
    """
    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    A_id = rec_a.id
    fresh_store.insert(rec_a)

    for _ in range(10):
        rec_dup = _make_record(embedding=REFERENCE_EMBEDDING)
        fresh_store.insert(rec_dup)

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    self_loops = edges_df[
        (edges_df["edge_type"] == "hebbian")
        & (edges_df["src"] == str(A_id))
        & (edges_df["dst"] == str(A_id))
    ]
    assert len(self_loops) >= 1, (
        f"dedup-touched record must have a self-loop; got "
        f"{edges_df.to_dict('records')}"
    )
    weight = float(self_loops["weight"].iloc[0])
    # Existing test_phase11_1 floor is 0.9. Post-fix expected ~1.1.
    assert weight >= 0.9, (
        f"dedup-target self-loop weight floor violated. "
        f"expected >= 0.9 (post-fix ~1.1); got {weight}"
    )


# ---------------------------------------------------------------------------
# Test 6 — CLI subcommand handler smoke (dry-run path)
# ---------------------------------------------------------------------------


def test_cli_subcommand_dry_run_smoke(
    fresh_store: MemoryStore,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """Smoke test for `cmd_maintenance_symmetrize_self_loops` handler.

    Constructs an asymmetric state, invokes the CLI handler with
    dry_run=True flag, asserts exit code 0 and JSON output shape.
    """
    from iai_mcp.cli import cmd_maintenance_symmetrize_self_loops

    # Setup: 6 records, delete 4 self-loops (same fixture as Test 3/4).
    record_ids: list[str] = []
    for i in range(6):
        emb = [0.0] * EMBED_DIM
        emb[i % EMBED_DIM] = 1.0
        if i >= EMBED_DIM:
            emb[i % EMBED_DIM] = 0.5
            emb[(i + 1) % EMBED_DIM] = math.sqrt(1 - 0.25)
        rec = _make_record(embedding=emb, literal_surface=f"record-{i}")
        record_ids.append(str(rec.id))
        fresh_store.insert(rec)

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    for rid in record_ids[:4]:
        edges_tbl.delete(
            f"edge_type == 'hebbian' AND src == '{rid}' AND dst == '{rid}'"
        )

    # The CLI handler resolves store_path -> opens its own MemoryStore.
    # Our fresh_store sits at tmp_path/iai-mcp; pass the same root.
    store_root = str(tmp_path / "iai-mcp")
    args = argparse.Namespace(
        store_path=store_root,
        dry_run=True,
        apply=False,
        yes=False,
    )

    exit_code = cmd_maintenance_symmetrize_self_loops(args)
    assert exit_code == 0, f"dry-run should return 0; got {exit_code}"

    captured = capsys.readouterr()
    # Stdout should contain JSON with the expected fields.
    payload = json.loads(captured.out)
    assert payload["dry_run"] is True, payload
    assert payload["records_total"] == 6, payload
    assert payload["self_loops_present"] == 2, payload
    assert payload["self_loops_pending"] == 4, payload
    assert payload["self_loops_inserted"] == 0, payload
