"""Tests for cleanup migration safety.

Behaviour covered:
- top-level `iai-mcp schema-cleanup` subcommand with `[--dry-run] [--apply]
  [--store-path PATH]`. Default mode is `--dry-run` for reversibility.
- `tier="semantic_pruned"` records remain in store indefinitely.
- `SEMANTIC_PRUNED_TIER` constant in `src/iai_mcp/types.py`.
- snapshot directory naming `~/.iai-mcp/lancedb-pre-cleanup-YYYYMMDDTHHMMSSZ`
  (UTC ISO-8601 basic format, no colons; filesystem-safe macOS + Linux).
- pytest under `tests/` (single file: `test_migrate_cleanup.py`).

Acceptance: N=12 known duplicates across 4 patterns →
`--dry-run` reports the diff without mutating; `--apply` snapshots
the LanceDB tables BEFORE any write, soft-deletes via tier rename to
`semantic_pruned`, reinforces incoming `schema_instance_of` edges
onto the keeper, emits `schema_cleanup_run` event, and is idempotent
(re-running on the migrated store reports zero changes).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest


# ---------------------------------------------------------------- helpers


def _rec(
    *,
    text: str = "t",
    tags: list[str] | None = None,
    language: str = "en",
    tier: str = "semantic",
    detail_level: int = 2,
    created_at: datetime | None = None,
):
    """Build a fresh MemoryRecord for fixtures (avoids loading the full embedder)."""
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    now = created_at or datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=detail_level,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or []),
        language=language,
    )


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch):
    """Avoid loading bge-m3 / bge-small during cleanup tests — perf hygiene."""
    from iai_mcp import embed as embed_mod
    from iai_mcp.types import EMBED_DIM

    class _FakeEmbedder:
        DIM = EMBED_DIM
        DEFAULT_DIM = EMBED_DIM
        DEFAULT_MODEL_KEY = "fake"

        def __init__(self, *args, **kwargs):
            self.DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield


# ---------------------------------------------------------------- SEMANTIC_PRUNED_TIER constant + TIER_ENUM extension


def test_semantic_pruned_tier_constant_and_enum_membership():
    """SEMANTIC_PRUNED_TIER is exported and present in TIER_ENUM."""
    from iai_mcp.types import SEMANTIC_PRUNED_TIER, TIER_ENUM

    assert SEMANTIC_PRUNED_TIER == "semantic_pruned", (
        "the constant value must be 'semantic_pruned' (used as a soft-delete "
        "sentinel by cleanup_schema_duplicates)."
    )
    assert SEMANTIC_PRUNED_TIER in TIER_ENUM, (
        "TIER_ENUM must include 'semantic_pruned' so MemoryRecord.__post_init__ "
        "tier validation accepts pruned rows when reading them back from the store."
    )


def test_memoryrecord_accepts_semantic_pruned_tier():
    """constructing a MemoryRecord with tier='semantic_pruned' succeeds."""
    rec = _rec(tier="semantic_pruned", text="pruned dup")
    # Should not raise.
    assert rec.tier == "semantic_pruned"


def test_memoryrecord_existing_tiers_still_accepted():
    """Negative-control: extending TIER_ENUM does not regress existing tier acceptance."""
    for tier in ("working", "episodic", "semantic", "procedural", "parametric"):
        rec = _rec(tier=tier, text=f"t-{tier}")
        assert rec.tier == tier


def test_memoryrecord_invalid_tier_still_raises():
    """Negative-control: garbage tier values still rejected after extension."""
    with pytest.raises(ValueError, match="invalid tier"):
        _rec(tier="garbage")


# ---------------------------------------------------------------- cleanup_schema_duplicates callable


def _seed_dup_store(
    tmp_path: Path,
    n_per_pattern: int = 4,
    n_patterns: int = 3,
    extra_singletons: int = 0,
):
    """Insert duplicate schema records DIRECTLY via store.insert(MemoryRecord(...)).

     made `persist_schema` idempotent so it would refuse to create the
    duplicate state we need for the test — the cleanup is a one-shot recovery for
    stores that accumulated duplicates BEFORE shipped.

    Each duplicate group also receives one inbound `schema_instance_of` edge
    from a freshly-inserted episodic evidence record, so the edge-reinforcement
    assertion has data to count.

    Returns (store, patterns) for downstream introspection.
    """
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    base = datetime.now(timezone.utc)
    patterns: list[str] = []

    for p_idx in range(n_patterns):
        pattern = f"tags:capture+role:user+p{p_idx}"
        patterns.append(pattern)
        # Insert N schema rows (oldest first so the first insert is the keeper).
        schema_ids = []
        for s_idx in range(n_per_pattern):
            sched_at = base + timedelta(seconds=p_idx * 60 + s_idx)
            sch = _rec(
                text=f"schema-p{p_idx}-i{s_idx}",
                tier="semantic",
                tags=[f"pattern:{pattern}", "schema"],
                created_at=sched_at,
            )
            store.insert(sch)
            schema_ids.append(sch.id)

            # One incoming schema_instance_of edge per schema row so each row
            # has at least one incident edge that needs redirecting onto the
            # keeper at cleanup time.
            ev = _rec(
                text=f"ev-p{p_idx}-i{s_idx}",
                tier="episodic",
                tags=["capture", "role:user"],
                created_at=sched_at,
            )
            store.insert(ev)
            store.boost_edges(
                [(ev.id, sch.id)],
                edge_type="schema_instance_of",
                delta=0.1,
            )

    # Add singletons (single-record patterns) that should be left alone.
    for s_idx in range(extra_singletons):
        pattern = f"singleton-p{s_idx}"
        patterns.append(pattern)
        sch = _rec(
            text=f"singleton-{s_idx}",
            tier="semantic",
            tags=[f"pattern:{pattern}", "schema"],
            created_at=base + timedelta(seconds=10000 + s_idx),
        )
        store.insert(sch)

    return store, patterns


def _count_semantic_pattern_records(store, pattern_tag_prefix: str = "pattern:") -> int:
    return sum(
        1
        for r in store.all_records()
        if r.tier == "semantic"
        and any(t.startswith(pattern_tag_prefix) for t in (r.tags or []))
    )


def _count_pruned(store) -> int:
    return sum(1 for r in store.all_records() if r.tier == "semantic_pruned")


def test_cleanup_dry_run_does_not_mutate_store(tmp_path):
    """--dry-run reports the diff and creates NO snapshot, mutates NO record."""
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, _patterns = _seed_dup_store(
        tmp_path, n_per_pattern=4, n_patterns=3
    )
    pre_semantic = _count_semantic_pattern_records(store)
    pre_pruned = _count_pruned(store)
    assert pre_semantic == 12  # 3 patterns x 4 dups each
    assert pre_pruned == 0

    # Sentinel: capture sibling listing of the IAI root before the run, so we
    # can assert no `lancedb-pre-cleanup-*` directory was created.
    siblings_pre = sorted(p.name for p in tmp_path.iterdir())

    summary = cleanup_schema_duplicates(store, apply=False)

    assert summary["mode"] == "dry-run"
    assert summary["groups"] == 3
    assert summary["keepers"] == 3
    assert summary["pruned"] == 9  # 3 patterns x (4-1) dups each
    assert summary["snapshot_dir"] is None

    # Store unchanged.
    assert _count_semantic_pattern_records(store) == 12
    assert _count_pruned(store) == 0

    # No snapshot directory created.
    siblings_post = sorted(p.name for p in tmp_path.iterdir())
    assert siblings_pre == siblings_post, (
        f"--dry-run must not create any sibling directories; saw new entries: "
        f"{set(siblings_post) - set(siblings_pre)}"
    )


def test_cleanup_apply_creates_snapshot_directory_before_writes(tmp_path):
    """--apply creates snapshot dir BEFORE soft-deletes; tables intact in copy.

    The snapshot is at `store.root / f'lancedb-pre-cleanup-{ts}'`
    (sibling of the inner `lancedb/` tables dir).
    """
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, _patterns = _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)
    summary = cleanup_schema_duplicates(store, apply=True)

    assert summary["mode"] == "apply"
    assert summary["snapshot_dir"] is not None

    snap = Path(summary["snapshot_dir"])
    assert snap.exists() and snap.is_dir()
    # naming: sibling of the `lancedb/` tables dir, prefixed `lancedb-pre-cleanup-`.
    assert snap.parent == Path(store.root)
    assert snap.name.startswith("lancedb-pre-cleanup-")
    # UTC ISO-8601 basic format suffix: YYYYMMDDTHHMMSSZ (16 chars).
    suffix = snap.name[len("lancedb-pre-cleanup-"):]
    assert len(suffix) == 16 and suffix.endswith("Z")

    # The snapshot is a copy of the storage dir; with HippoDB it contains
    # the hippo/ subdirectory (brain.sqlite3 + hnswlib index).
    snap_entries = {p.name for p in snap.iterdir()}
    assert snap_entries, f"snapshot must be non-empty, got: {snap_entries}"
    # Accept either HippoDB layout (brain.sqlite3 at snap top-level, copied from
    # hippo/ contents) or legacy LanceDB layout (.lance dirs).
    has_hippo = "brain.sqlite3" in snap_entries
    has_lance = any(e.endswith(".lance") for e in snap_entries)
    assert has_hippo or has_lance, (
        f"snapshot must contain brain.sqlite3 or .lance dirs; got: {snap_entries}"
    )


def test_cleanup_apply_soft_deletes_duplicates_via_tier_rename(tmp_path):
    """--apply leaves 1 keeper per pattern at tier='semantic'; the rest at 'semantic_pruned'."""
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, _patterns = _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)
    cleanup_schema_duplicates(store, apply=True)

    # 1 keeper per pattern × 3 patterns.
    assert _count_semantic_pattern_records(store) == 3
    # 3 dups per pattern × 3 patterns = 9 pruned.
    assert _count_pruned(store) == 9


def test_cleanup_apply_reinforces_edges_onto_keeper(tmp_path):
    """keeper inherits incoming schema_instance_of edges from duplicates."""
    from iai_mcp.migrate import cleanup_schema_duplicates
    from iai_mcp.store import EDGES_TABLE

    store, patterns = _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)

    # Pre-state: each schema row has 1 incident schema_instance_of edge.
    edges_pre = store.db.open_table(EDGES_TABLE).to_pandas()
    pre_total_sio = int(
        (edges_pre["edge_type"] == "schema_instance_of").sum()
    )
    assert pre_total_sio == 12  # 3 patterns x 4 schema rows x 1 edge each

    # Determine keepers (oldest record per pattern) BEFORE the cleanup runs so
    # we can locate them after.
    pattern_to_keeper_id = {}
    for p in patterns:
        recs = sorted(
            (
                r
                for r in store.all_records()
                if r.tier == "semantic"
                and f"pattern:{p}" in (r.tags or [])
            ),
            key=lambda r: r.created_at,
        )
        if recs:
            pattern_to_keeper_id[p] = recs[0].id

    summary = cleanup_schema_duplicates(store, apply=True)
    assert summary["edges_reinforced"] >= 9  # at least one per duplicate

    # For each pattern, keeper's incident schema_instance_of edge count
    # should equal the original cumulative count (4 per pattern).
    edges_post = store.db.open_table(EDGES_TABLE).to_pandas()
    for pattern, keeper_id in pattern_to_keeper_id.items():
        keeper_str = str(keeper_id)
        sio = edges_post[
            (edges_post["edge_type"] == "schema_instance_of")
            & ((edges_post["dst"] == keeper_str) | (edges_post["src"] == keeper_str))
        ]
        # 4 schema rows × 1 edge each = 4 inbound edges should now point to
        # the keeper (1 original + 3 redirected).
        assert len(sio) == 4, (
            f"pattern {pattern!r}: keeper {keeper_str[:8]} should have 4 "
            f"schema_instance_of edges (1 original + 3 redirected from dups), "
            f"got {len(sio)}"
        )


def test_cleanup_apply_keeper_is_oldest_per_pattern(tmp_path):
    """keeper selection preserves provenance ordering — oldest record wins."""
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, patterns = _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)

    # Identify expected keepers (oldest per pattern) BEFORE cleanup.
    expected_keeper_ids = {}
    for p in patterns:
        recs = sorted(
            (
                r
                for r in store.all_records()
                if r.tier == "semantic"
                and f"pattern:{p}" in (r.tags or [])
            ),
            key=lambda r: r.created_at,
        )
        expected_keeper_ids[p] = recs[0].id

    cleanup_schema_duplicates(store, apply=True)

    # After cleanup: per pattern, exactly one tier='semantic' record remains
    # AND it must be the oldest (the expected keeper id).
    for p, expected_id in expected_keeper_ids.items():
        survivors = [
            r
            for r in store.all_records()
            if r.tier == "semantic"
            and f"pattern:{p}" in (r.tags or [])
        ]
        assert len(survivors) == 1, (
            f"pattern {p!r}: expected exactly 1 keeper, got {len(survivors)}"
        )
        assert survivors[0].id == expected_id, (
            f"pattern {p!r}: keeper should be the oldest record "
            f"({str(expected_id)[:8]}), got {str(survivors[0].id)[:8]}"
        )


def test_cleanup_apply_skips_single_record_groups(tmp_path):
    """patterns with N=1 schema row are left untouched (not duplicates)."""
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, _patterns = _seed_dup_store(
        tmp_path, n_per_pattern=4, n_patterns=2, extra_singletons=2
    )
    # 2 patterns × 4 dups + 2 singletons = 10 semantic+pattern rows.
    assert _count_semantic_pattern_records(store) == 10

    summary = cleanup_schema_duplicates(store, apply=True)
    assert summary["groups"] == 2  # only the 2 dup groups, not the singletons
    assert summary["keepers"] == 2
    assert summary["pruned"] == 6  # 2 patterns × 3 dups

    # 2 singletons + 2 keepers = 4 semantic+pattern rows after cleanup.
    assert _count_semantic_pattern_records(store) == 4


def test_cleanup_emits_schema_cleanup_run_event(tmp_path):
    """audit trail: schema_cleanup_run event written with the summary payload."""
    from iai_mcp.events import query_events
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, _patterns = _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)
    cleanup_schema_duplicates(store, apply=True)

    events = query_events(store, kind="schema_cleanup_run")
    assert len(events) >= 1, "schema_cleanup_run event must be emitted"
    e = events[0]
    payload = e["data"]
    for required_key in (
        "mode",
        "groups",
        "keepers",
        "pruned",
        "edges_reinforced",
        "snapshot_dir",
    ):
        assert required_key in payload, (
            f"schema_cleanup_run event payload missing '{required_key}'"
        )
    assert payload["mode"] == "apply"
    assert payload["groups"] == 3
    assert payload["keepers"] == 3
    assert payload["pruned"] == 9


def test_cleanup_apply_is_idempotent_on_second_run(tmp_path):
    """re-running --apply on the migrated store reports zero work to do."""
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, _patterns = _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)

    # First pass: real work.
    summary1 = cleanup_schema_duplicates(store, apply=True)
    assert summary1["groups"] == 3
    assert summary1["pruned"] == 9

    # Second pass: store is already clean — pruned rows are at
    # tier='semantic_pruned' (not 'semantic'), so the dedup pass sees only
    # the 3 surviving keepers (one per pattern, N=1 each, no dups).
    summary2 = cleanup_schema_duplicates(store, apply=True)
    assert summary2["groups"] == 0, (
        f"second --apply must report 0 groups (idempotent), got {summary2}"
    )
    assert summary2["keepers"] == 0
    assert summary2["pruned"] == 0
    # Final post-state unchanged from the first pass.
    assert _count_semantic_pattern_records(store) == 3
    assert _count_pruned(store) == 9


# ---------------------------------------------------------------- iai-mcp schema-cleanup CLI subcommand


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """Invoke iai_mcp.cli.main(argv) under stdout capture; return (exit_code, output)."""
    import io
    from contextlib import redirect_stdout

    from iai_mcp.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            code = main(argv)
        except SystemExit as exc:
            # argparse exits via SystemExit — propagate the code.
            code = int(exc.code) if exc.code is not None else 0
    return code, buf.getvalue()


def test_cli_schema_cleanup_default_is_dry_run(tmp_path):
    """default mode is dry-run (Beer VSM S2 reversibility)."""
    _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)
    code, out = _run_cli(
        ["schema-cleanup", "--store-path", str(tmp_path)]
    )
    assert code == 0, f"CLI exited non-zero: {code!r}; output:\n{out}"
    assert "[dry-run]" in out, (
        f"default mode must report '[dry-run]' header; got:\n{out}"
    )
    # Reasonable summary output (counts visible).
    assert "groups" in out
    assert "keepers" in out
    assert "pruned" in out


def test_cli_schema_cleanup_apply_runs_end_to_end(tmp_path):
    """--apply performs the cleanup end-to-end and prints the snapshot dir."""
    from iai_mcp.store import MemoryStore

    _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)
    code, out = _run_cli(
        ["schema-cleanup", "--apply", "--store-path", str(tmp_path)]
    )
    assert code == 0, f"CLI exited non-zero: {code!r}; output:\n{out}"
    assert "[apply]" in out
    assert "snapshot" in out.lower()

    # Verify the store actually mutated — re-open and count.
    store = MemoryStore(path=tmp_path)
    assert _count_semantic_pattern_records(store) == 3
    assert _count_pruned(store) == 9


def test_cli_schema_cleanup_dry_run_and_apply_mutually_exclusive(tmp_path):
    """argparse mutually-exclusive group rejects --dry-run --apply combo."""
    _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=2)
    code, _out = _run_cli(
        [
            "schema-cleanup",
            "--dry-run",
            "--apply",
            "--store-path",
            str(tmp_path),
        ]
    )
    assert code != 0, (
        "--dry-run and --apply must be mutually exclusive (argparse-rejected)"
    )


def test_cli_schema_cleanup_honours_store_path_argument(tmp_path):
    """--store-path targets a synthetic store so the prod store is never touched."""
    # Two stores under the same temp tree: store_a has dups, store_b is empty.
    store_a_root = tmp_path / "a"
    store_b_root = tmp_path / "b"
    store_a_root.mkdir()
    store_b_root.mkdir()

    _seed_dup_store(store_a_root, n_per_pattern=4, n_patterns=2)

    # Cleanup against store_b should report 0 groups (empty store).
    code, out_b = _run_cli(
        ["schema-cleanup", "--store-path", str(store_b_root)]
    )
    assert code == 0
    assert "0" in out_b  # at least one count of 0 in the output

    # store_a still untouched (the b-cleanup hit a different path).
    from iai_mcp.store import MemoryStore

    store_a = MemoryStore(path=store_a_root)
    assert _count_semantic_pattern_records(store_a) == 8  # 2 patterns x 4 dups


def test_cli_schema_cleanup_argparse_contract():
    """argparse: schema-cleanup has --apply (default False) + --store-path."""
    from iai_mcp.cli import _build_parser

    p = _build_parser()
    ns = p.parse_args(["schema-cleanup", "--apply"])
    assert ns.cmd == "schema-cleanup"
    assert ns.apply is True
    assert ns.dry_run is False
    # Default --store-path is None (cmd_schema_cleanup falls back to ~/.iai-mcp).
    assert ns.store_path is None

    ns2 = p.parse_args(["schema-cleanup", "--dry-run"])
    assert ns2.dry_run is True
    assert ns2.apply is False

    ns3 = p.parse_args(["schema-cleanup", "--store-path", "/tmp/foo"])
    assert ns3.store_path == "/tmp/foo"
    # When neither --dry-run nor --apply is given, both flags default False;
    # cmd_schema_cleanup interprets this as the dry-run default.
    assert ns3.apply is False
    assert ns3.dry_run is False
