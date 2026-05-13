""" -> migration + encryption +
 TEM factorization (v3 -> v4 column rename + structure_hv fill).

(v1 -> v2):
  One-time batch migration that re-embeds every record with the
  configured embedder (bge-small-en-v1.5 by default per ; bge-m3
  remains opt-in via IAI_MCP_EMBED_MODEL), backfills the v2 fields with
  their defaults, detects language via langdetect on literal_surface
  for legacy provenance, and marks each record schema_version=2.

(v2 -> v3 data upgrade):
  In-place AES-256-GCM encryption of literal_surface / provenance_json /
  profile_modulation_gain_json on the records table, and data_json on the
  events table. Runs lazily via `migrate_encryption_v2_to_v3(store)` and
  is idempotent (skips rows that already carry the iai:enc:v1: prefix).

(v3 -> v4 TEM factorization):
  Renames the LanceDB records column `hd_vector_json` (pa.string(), JSON-
  encoded list[int]|None reservation slot from ) to `structure_hv`
  (pa.binary(), packed D=10000 BSC bits = 1250 bytes per row). For stores
  created on the new schema (the typical case after this plan ships), the
  column name is already correct; the migration just (a) backfills any row
  whose `structure_hv` is still empty bytes via `tem.bind_structure(record)`,
  and (b) bumps schema_version from 3 to 4. Idempotent: rows already at v4
  with a populated `structure_hv` are skipped.

Invariants preserved (constitutional):
- literal_surface is byte-for-byte preserved through ALL migrations.
- Provenance entries preserved.
- All flags (detail_level, pinned, never_merge, never_decay, etc.) unchanged.
- Tags list unchanged.
- CR-01: every WHERE/DELETE predicate routes through store._uuid_literal so
  injection content cannot ride a poisoned UUID.

Idempotent: records that are already schema_version=2 are skipped by v1->v2.
Records whose sensitive columns already start with iai:enc:v1: are skipped
by v2->v3. Records that are already schema_version=4 with a non-empty
structure_hv are skipped by v3->v4.

Resumable: each record is committed individually via delete + insert. If the
process crashes mid-batch, re-running picks up where it left off.

Emits events of kind='migration_v1_to_v2', 'migration_v2_to_v3', and
'migration_v3_to_v4' .

CLI wrappers:
  iai-mcp migrate --from=1 --to=2 [--dry-run]  # (v1 -> v2)
  iai-mcp migrate --from=2 --to=3 [--dry-run]  # (encryption)
  iai-mcp migrate --from=3 --to=4 [--dry-run]  # (TEM factorization)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from uuid import UUID

import pyarrow as pa

from iai_mcp.crypto import encrypt_field, is_encrypted
from iai_mcp.embed import Embedder
from iai_mcp.events import write_event
from iai_mcp.store import (
    EVENTS_TABLE,
    MemoryStore,
    RECORDS_TABLE,
    _uuid_literal,
)
from iai_mcp.types import (
    SCHEMA_VERSION_CURRENT,
    SCHEMA_VERSION_LEGACY,
    MemoryRecord,
)


log = logging.getLogger(__name__)


# / crash-safe reembed migration constants.
# `STAGING_TABLE` is the LanceDB table that receives re-embedded rows during
# of the four-phase flow (stage -> validate -> atomic swap ->
# deferred cleanup). `OLD_TABLE_PREFIX` is the timestamp-suffixed name of the
# rolled-aside original records table after a successful swap. `PROGRESS_FILE`
# sits next to the LanceDB store and lets `--resume` pick up at the last
# successfully-staged row index after a crash.
STAGING_TABLE = "records_v_new"
OLD_TABLE_PREFIX = "records_old_"
PROGRESS_FILE = "migration_progress.json"
# Prior-key AES recovery (tail-end mandate): disjoint from reembed staging so
# detect_partial_migration taxonomy stays unchanged.
CRYPTO_RECOVER_STAGING = "records_crypto_recover_stage"


def _db_table_names_set(db) -> set[str]:
    """LanceDB 0.30+ list_tables() paginated response vs legacy list."""
    res = db.list_tables()
    if hasattr(res, "tables"):
        return set(res.tables)
    return set(res)


def _detect_language(text: str) -> str:
    """Best-effort language detection; fall back to 'en' on low confidence."""
    text = (text or "").strip()
    if not text:
        return "en"
    try:
        from langdetect import DetectorFactory, detect_langs
        DetectorFactory.seed = 42
        cands = detect_langs(text)
        if cands and cands[0].prob >= 0.7:
            return cands[0].lang
    except Exception:
        pass
    return "en"


def migrate_v1_to_v2(
    store: MemoryStore,
    embedder: Optional[Embedder] = None,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Re-embed + language-tag + default-backfill every v1 record.

    Parameters
    ----------
    store:
        Open MemoryStore. Migration rewrites in-place via delete+insert per record.
    embedder:
        Embedder instance; defaults to Embedder() (bge-small-en-v1.5, 384d,
        per ). The store's records table schema must match the
        embedder's DIM; if they differ, the caller is responsible for using
        the appropriate model_key (e.g. legacy 1024d stores from the brief
        Phase-2 era should pass bge-m3 until the table schema is
        rebuilt down to 384d in a dedicated re-embed migration).
    dry_run:
        If True, counts what would be migrated without mutating the store.
    progress:
        Optional callable(idx, total) invoked before each record migration
        so CLI / external tooling can render a progress bar.

    Returns a dict with records_migrated / skipped / duration_sec / previous_model / new_model.
    """
    t0 = time.time()
    if embedder is not None:
        emb = embedder
    else:
        from iai_mcp.embed import embedder_for_store
        emb = embedder_for_store(store)

    all_records = store.all_records()
    v1_records = [r for r in all_records if r.schema_version == SCHEMA_VERSION_LEGACY]
    total = len(v1_records)
    migrated = 0

    for idx, record in enumerate(v1_records):
        if progress is not None:
            try:
                progress(idx, total)
            except Exception:
                pass

        new_lang = record.language if (record.language and record.language.strip()) else _detect_language(record.literal_surface)

        if dry_run:
            migrated += 1
            continue

        # Re-embed with the configured model (English-Only-Brain default,
        # ). If the embedder's DIM differs from the store's current
        # schema, insert will raise; callers on legacy 1024d stores from the
        # brief Phase-2 era must pass a matching model_key.
        new_embedding = emb.embed(record.literal_surface)

        updated = MemoryRecord(
            id=record.id,
            tier=record.tier,
            literal_surface=record.literal_surface,       # verbatim preserved
            aaak_index=record.aaak_index,
            embedding=new_embedding,
            structure_hv=record.structure_hv,
            community_id=record.community_id,
            centrality=record.centrality,
            detail_level=record.detail_level,
            pinned=record.pinned,
            stability=record.stability,
            difficulty=record.difficulty,
            last_reviewed=record.last_reviewed,
            never_decay=record.never_decay,
            never_merge=record.never_merge,
            provenance=record.provenance,
            created_at=record.created_at,
            updated_at=record.updated_at,
            tags=record.tags,
            language=new_lang,
            s5_trust_score=0.5,
            profile_modulation_gain={},
            schema_version=SCHEMA_VERSION_CURRENT,
        )
        # Delete old v1 row, insert new v2 row (LanceDB MVCC-safe).
        # fix: route record.id through _uuid_literal so the
        # DELETE predicate cannot carry SQL injection content, matching the
        # pattern already used in store.append_provenance / boost_edges.
        tbl = store.db.open_table(RECORDS_TABLE)
        tbl.delete(f"id = '{_uuid_literal(record.id)}'")
        store.insert(updated)
        migrated += 1

    duration_sec = time.time() - t0

    # Emit a single migration event even on dry-run so audit trails record
    # the planned scope (severity=info).
    if not dry_run and migrated > 0:
        write_event(
            store,
            kind="migration_v1_to_v2",
            data={
                "record_count": migrated,
                "duration_sec": duration_sec,
            },
            severity="info",
        )

    return {
        "records_migrated": migrated,
        "skipped": max(0, len(all_records) - total),
        "duration_sec": duration_sec,
        "previous_model": "bge-small-en-v1.5",
        "new_model": emb.model_key,
    }


def _records_schema_at_dim(dim: int) -> pa.Schema:
    """Build the records-table Arrow schema at an explicit embedding dim.

    Mirrors `MemoryStore._ensure_tables` lines 249-281 byte-for-byte except
    for the `embedding` column's `list_size=dim`. Inlined here because the
    staged-swap reembed migration needs to create `records_v_new` at a
    DIFFERENT dim from the live store's `_embed_dim` — `store._ensure_tables`
    is not parameterised on dim. / file-disjoint
    constraint forbids store.py changes; inlining is the conservative path.
    """
    return pa.schema(
        [
            ("id", pa.string()),
            ("tier", pa.string()),
            ("literal_surface", pa.string()),
            ("aaak_index", pa.string()),
            ("embedding", pa.list_(pa.float32(), dim)),
            ("structure_hv", pa.binary()),
            ("community_id", pa.string()),
            ("centrality", pa.float32()),
            ("detail_level", pa.int32()),
            ("pinned", pa.bool_()),
            ("stability", pa.float32()),
            ("difficulty", pa.float32()),
            ("last_reviewed", pa.timestamp("us", tz="UTC")),
            ("never_decay", pa.bool_()),
            ("never_merge", pa.bool_()),
            ("provenance_json", pa.string()),
            ("created_at", pa.timestamp("us", tz="UTC")),
            ("updated_at", pa.timestamp("us", tz="UTC")),
            ("tags_json", pa.string()),
            ("language", pa.string()),
            ("s5_trust_score", pa.float32()),
            ("profile_modulation_gain_json", pa.string()),
            ("schema_version", pa.int32()),
        ]
    )


def _progress_path(store: MemoryStore) -> Path:
    """Resolve the on-disk path of `migration_progress.json` for this store.

    Sits next to the LanceDB tables under `store.root` (the IAI root —
    parent of the `lancedb/` subdir, same convention as `daemon_state.py`
    and `cleanup_schema_duplicates`).
    """
    return Path(store.root) / PROGRESS_FILE


def _progress_read(store: MemoryStore) -> dict:
    """Self-healing reader for `migration_progress.json`.

    Returns `{}` on missing or malformed file — mirrors
    `daemon_state.load_state` lines 41-49 verbatim. Callers MUST tolerate an
    empty dict as "no checkpoint, start from row 0".
    """
    path = _progress_path(store)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _progress_write(store: MemoryStore, state: dict) -> None:
    """Atomic write for `migration_progress.json`.

    Verbatim copy of `daemon_state.save_state`'s tempfile + fsync +
    os.replace pattern — the project canon for atomic on-disk mutation.
    `os.replace` (not `os.rename`) per CONTEXT + project convention
    (cross-platform safety on Windows; preferred even on POSIX).
    """
    target = _progress_path(store)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".migration-progress.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _progress_clear(store: MemoryStore) -> None:
    """Drop the progress checkpoint if present. Idempotent."""
    path = _progress_path(store)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # Permission errors / odd FS states — don't crash the migration.
        pass


def _stage_record_to_table(
    store: MemoryStore,
    target_tbl,
    rec: MemoryRecord,
    new_embedding: list[float],
) -> None:
    """Append one re-embedded record to the staging table.

    Mirrors `store.insert`'s sync write path (the legacy branch at
    store.py:550-554) but targets an arbitrary table object instead of the
    hard-coded RECORDS_TABLE. `store._to_row` handles AES-GCM encryption of
    `literal_surface` / `provenance_json` / `profile_modulation_gain_json`
    with `AAD = _uuid_literal(record.id)`, so a record written through this
    helper round-trips through `store.get` after the atomic swap (same key,
    same AAD).

    `tem.bind_structure` is invoked when `structure_hv` is empty — preserves
    the autopoietic write-time fill from `store.insert` line 519-521 so a
    re-embedded record never lands in the staging table without a
    structural fingerprint.
    """
    if not rec.structure_hv:
        from iai_mcp.tem import bind_structure
        rec.structure_hv = bind_structure(rec)
    new_rec = MemoryRecord(
        id=rec.id,
        tier=rec.tier,
        literal_surface=rec.literal_surface,  # verbatim
        aaak_index=rec.aaak_index,
        embedding=new_embedding,
        structure_hv=rec.structure_hv,
        community_id=rec.community_id,
        centrality=rec.centrality,
        detail_level=rec.detail_level,
        pinned=rec.pinned,
        stability=rec.stability,
        difficulty=rec.difficulty,
        last_reviewed=rec.last_reviewed,
        never_decay=rec.never_decay,
        never_merge=rec.never_merge,
        provenance=rec.provenance,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        tags=rec.tags,
        language=rec.language,
        s5_trust_score=rec.s5_trust_score,
        profile_modulation_gain=rec.profile_modulation_gain,
        schema_version=rec.schema_version,
    )
    target_tbl.add([store._to_row(new_rec)])


def _stage_loop(
    store: MemoryStore,
    target_embedder,
    target_dim: int,
    target_tbl,
    source_iter,
    *,
    total: int,
    started_at_iso: str,
    started_idx: int = 0,
    already_staged_ids: Optional[set[str]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[int, list[str]]:
    """Run the per-row stage step over the source iterator.

    Re-embeds each source record under `target_embedder`, writes the new
    row to `target_tbl`, and atomically updates `migration_progress.json`
    after each successful row so a crash leaves the checkpoint pointing at
    the last successfully-staged record. Per-row exceptions are caught
    + structured-logged + counted (best-effort migration); KeyboardInterrupt
    and SystemExit propagate untouched so the caller (the live records
    table is intact in ) sees the kill.

    Returns `(staged_count, failures)`. `failures` is the list of
    record-id strings whose re-embedding raised a recoverable exception.
    """
    staged_count = 0
    failures: list[str] = []
    staged_ids: list[str] = list(already_staged_ids or [])
    skipped_set: set[str] = set(staged_ids)

    idx = started_idx
    for rec in source_iter:
        rec_id_str = str(rec.id)
        if rec_id_str in skipped_set:
            # Already in the staging table from a prior run.
            continue
        if progress is not None:
            try:
                progress(idx, total)
            except Exception:
                pass
        try:
            new_embedding = target_embedder.embed(rec.literal_surface)
            _stage_record_to_table(store, target_tbl, rec, new_embedding)
        except (KeyboardInterrupt, SystemExit):
            # Mid-flight kill: do not swallow. Records is intact;
            # records_v_new holds the partial set; progress file points
            # at the last successfully-staged row. The boot detector or
            # CLI rollback handles the cleanup.
            raise
        except Exception as exc:
            log.warning(
                "migrate_reembed_per_row_failed",
                extra={
                    "record_id": rec_id_str,
                    "error": str(exc)[:160],
                },
            )
            failures.append(rec_id_str)
            idx += 1
            continue

        staged_count += 1
        staged_ids.append(rec_id_str)
        # Atomic checkpoint write — every successful row.
        _progress_write(
            store,
            {
                "started_at": started_at_iso,
                "ts": int(time.time()),
                "row_index": idx,
                "last_rid": rec_id_str,
                "total": total,
                "target_dim": target_dim,
                "target_model_key": getattr(target_embedder, "model_key", "unknown"),
                "staged_ids": staged_ids,
                "failures": failures,
            },
        )
        idx += 1

    return staged_count, failures


def _lancedb_root(db) -> Path:
    """Resolve the on-disk root of the LanceDB connection.

    Tables live as `<name>.lance` directories under this root. Used by the
    filesystem-level atomic-swap fallback (LanceDB 0.30.2 OSS does NOT
    implement `db.rename_table` — calling it raises `NotImplementedError:
    rename_table is not supported in LanceDB OSS` despite the method
    existing on the connection object). The fallback uses `os.replace` on
    the table directories — POSIX `rename(2)` semantics on the same
    filesystem give us the atomicity LanceDB OSS withholds.
    """
    return Path(db.uri)


def _swap_tables_filesystem(db, *, source: str, dest: str) -> None:
    """Atomically rename `source.lance` -> `dest.lance` on disk.

    Uses `os.replace` (project canon, project convention prefers it over
    `os.rename` for cross-platform safety on Windows; on POSIX both are
    atomic on the same filesystem). The destination MUST be empty or
    absent (macOS/HFS+/APFS rejects `os.replace` onto a non-empty
    directory with `[Errno 66] Directory not empty`).

    Caller is responsible for ordering when swapping: rename A->A_old
    BEFORE renaming B->A so the destination slot is empty.
    """
    root = _lancedb_root(db)
    src_path = root / f"{source}.lance"
    dst_path = root / f"{dest}.lance"
    os.replace(src_path, dst_path)


def _validate_and_swap(
    store: MemoryStore,
    *,
    source_dim: int,
    target_dim: int,
    target_embedder,
    staged_count: int,
    failures: list[str],
    duration_sec: float,
) -> dict:
    """(validate) + (atomic swap) + event emit.

    Refuses to swap if staged < orig * 0.99 ( gross-mismatch guard).
    Emits `migration_reembed` BEFORE the rename so a crash mid-rename still
    leaves an audit trail. Swap uses filesystem-level `os.replace` on the
    table directories under `db.uri` (LanceDB 0.30.2 OSS raises
    `NotImplementedError` on `db.rename_table` despite exposing the
    method — verified at runtime against the pinned version). After the
    swap, `_embed_dim` is refreshed to target_dim so subsequent inserts
    pass the dim check.
    """
    orig = store.db.open_table(RECORDS_TABLE).count_rows()
    staged = store.db.open_table(STAGING_TABLE).count_rows()
    if orig > 0 and staged < orig * 0.99:
        log.error(
            "migrate_reembed_validate_failed",
            extra={
                "orig": orig,
                "staged": staged,
                "ratio": staged / max(orig, 1),
                "failures": len(failures),
            },
        )
        raise RuntimeError(
            f"reembed staging produced {staged}/{orig} rows "
            f"({staged/max(orig,1):.3%}); refusing to swap. Inspect tables "
            f"manually or run `iai-mcp migrate --rollback`."
        )

    # Emit BEFORE rename so the audit trail survives a mid-rename crash;
    # the rollback path is then triggered by the boot detector.
    try:
        write_event(
            store,
            kind="migration_reembed",
            data={
                "source_dim": source_dim,
                "target_dim": target_dim,
                "updated": staged_count,
                "duration_sec": duration_sec,
                "target_model_key": getattr(target_embedder, "model_key", "unknown"),
                "failures": len(failures),
            },
            severity="info",
        )
    except Exception:
        pass

    # — atomic swap via filesystem-level os.replace on the table
    # directories (LanceDB OSS doesn't implement rename_table — see
    # _swap_tables_filesystem docstring for evidence).
    ts = int(time.time())
    old_name = f"{OLD_TABLE_PREFIX}{ts}"
    # Step 1: records -> records_old_<ts> (slot is empty after, so step 2 is safe).
    _swap_tables_filesystem(store.db, source=RECORDS_TABLE, dest=old_name)
    # Step 2: records_v_new -> records.
    _swap_tables_filesystem(store.db, source=STAGING_TABLE, dest=RECORDS_TABLE)

    # Refresh the in-memory dim binding so subsequent store.insert calls
    # against the swapped table pass the dim check at store.py:514-517.
    store._embed_dim = target_dim

    # Drop the progress checkpoint — cleanup is handled at next
    # boot's detect_partial_migration -> needs_cleanup branch.
    _progress_clear(store)

    return {
        "source_dim": source_dim,
        "target_dim": target_dim,
        "updated": staged_count,
        "skipped": 0,
        "failures": len(failures),
        "duration_sec": duration_sec,
        "old_table": old_name,
    }


def migrate_reembed_to_current_dim(
    store: MemoryStore,
    target_embedder,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Crash-safe re-embed migration (/ four-phase flow).

    Closes V2-05: replaces the destructive drop-then-rebuild at the legacy
    line 300-305 with stage -> validate -> atomic swap -> deferred cleanup.
    A KeyboardInterrupt, kill, or power loss mid-flight leaves the original
    `records` table untouched; the boot-time detector
    (`detect_partial_migration`) refuses to advertise daemon-ready and
    surfaces a remediation prompt.

    (stage):
      - Drop any pre-existing `records_v_new` (defensive — should not
        normally exist; the boot detector catches a real partial state).
      - Create `records_v_new` at the post-migration schema (target_dim).
      - Stream rows from the live `records` table; re-embed each via
        `target_embedder.embed`; insert into `records_v_new` via the same
        AES-GCM-applying `_to_row` path as `store.insert`.
      - On every successful row, atomically update `migration_progress.json`
        with the row index + record id (resume anchor).
      - Per-row embed exceptions are logged + counted; KeyboardInterrupt /
        SystemExit propagates untouched.

    (validate):
      - `staged >= orig * 0.99` gate (allow up to 1% per-row failure).
      - Gross mismatch (< 99%) raises RuntimeError; both tables remain
        intact for inspection or `iai-mcp migrate --rollback`.

    (atomic swap):
      - LanceDB `db.rename_table(records, records_old_<ts>)` then
        `db.rename_table(records_v_new, records)`. Cross-platform safe —
        no filesystem-level `os.rename` (project convention prefers
        `os.replace`; LanceDB owns the table-rename atomicity here).
      - Emit `migration_reembed` BEFORE rename so audit trail survives
        a mid-rename crash.
      - Refresh `store._embed_dim = target_dim`.
      - Drop `migration_progress.json`.

    (deferred cleanup):
      - `records_old_<ts>` is RETAINED. Next boot's
        `detect_partial_migration` returns `needs_cleanup` and the daemon
        drops it before advertising ready. Gives the operator a one-cycle
        manual rollback window.

    Idempotency: same-dim same-model returns `no_op=True` without
    touching the store (preserves the legacy line-244-250 contract used
    by `tests/test_migrate_reembed_to_current_dim.py`).

    Preserves ( + full record fidelity):
      - `literal_surface` byte-for-byte (re-embedded but content unchanged).
      - `structure_hv` (TEM factorization independent of content embedding).
      - All flags, tags, language, schema_version, provenance,
        s5_trust_score, profile_modulation_gain, timestamps.

    Emits `kind='migration_reembed'` on success (data: source_dim,
    target_dim, updated, duration_sec, target_model_key, failures) AND
    on idempotent no-op runs (data.no_op = True).

    Parameters mirror the legacy signature for source-compat:
    `dry_run` short-circuits with a `would_update` count; `progress` is an
    optional callable invoked at each row before embedding.
    """
    t0 = time.time()

    source_dim = int(store.embed_dim)
    target_dim = int(target_embedder.DIM)
    started_at_iso = datetime.now(timezone.utc).isoformat()

    # — idempotency / dry-run / no-op fast paths.
    # Match the legacy contract at line 244-260 so the existing
    # tests/test_migrate_reembed_to_current_dim.py suite remains green.
    if source_dim == target_dim:
        # Emit a no-op event so case 5 (idempotency rerun) is witnessable.
        try:
            write_event(
                store,
                kind="migration_reembed",
                data={
                    "source_dim": source_dim,
                    "target_dim": target_dim,
                    "updated": 0,
                    "no_op": True,
                    "duration_sec": time.time() - t0,
                    "target_model_key": getattr(
                        target_embedder, "model_key", "unknown"
                    ),
                },
                severity="info",
            )
        except Exception:
            pass
        # `total` matches the legacy signature so the existing
        # test_reembed_idempotent_same_dim_no_op assertion holds:
        # `result["skipped"] == 2 or result.get("no_op") is True`.
        return {
            "source_dim": source_dim,
            "target_dim": target_dim,
            "updated": 0,
            "skipped": store.db.open_table(RECORDS_TABLE).count_rows(),
            "no_op": True,
            "duration_sec": time.time() - t0,
        }

    if dry_run:
        return {
            "source_dim": source_dim,
            "target_dim": target_dim,
            "would_update": store.db.open_table(RECORDS_TABLE).count_rows(),
            "duration_sec": time.time() - t0,
        }

    # — stage.
    # Defensive drop of any pre-existing staging table. A real partial
    # state is caught by `detect_partial_migration` at boot; if we got
    # here cleanly the staging table should not exist.
    if STAGING_TABLE in set(store.db.table_names()):
        store.db.drop_table(STAGING_TABLE)
    target_tbl = store.db.create_table(
        STAGING_TABLE, schema=_records_schema_at_dim(target_dim)
    )

    total = store.db.open_table(RECORDS_TABLE).count_rows()
    source_iter = store.iter_records()
    staged_count, failures = _stage_loop(
        store,
        target_embedder,
        target_dim,
        target_tbl,
        source_iter,
        total=total,
        started_at_iso=started_at_iso,
        progress=progress,
    )

    # (validate) + (atomic swap) + (deferred cleanup).
    duration_sec = time.time() - t0
    return _validate_and_swap(
        store,
        source_dim=source_dim,
        target_dim=target_dim,
        target_embedder=target_embedder,
        staged_count=staged_count,
        failures=failures,
        duration_sec=duration_sec,
    )


# ---------------------------------------------------------------------------
# / boot-time partial-migration detector + rollback /
# resume entry points. The detector runs at daemon boot BEFORE ready-state
# advertisement (see daemon.py main() — the wire-up makes the rollback
# handler actually fire, closing the V2-07 anti-pattern of declared-but-
# unwired knobs).
# ---------------------------------------------------------------------------


def detect_partial_migration(db) -> dict:
    """Inspect the LanceDB store for evidence of a crashed reembed migration.

    Returns a dict with `state` in:
      - "clean": no partial-migration tables present.
      - "needs_rollback": records_v_new present alongside records (mid-stage
        crash; original records intact, staging partial — recover by
        dropping staging or resuming).
      - "needs_cleanup": records_old_<ts> present alongside fresh records;
        successful swap from a prior boot — drop the old table.
      - "partial_swap_inconsistent": records_v_new present without records
        AND without any records_old_<ts> (catastrophic mid-swap state;
        manual recovery required).
      - "needs_rollback" (variant): records_v_new + records_old_<ts> both
        present, records absent — swap interrupted between renames; the
        old table is the rollback anchor.
      - "unknown": defensive default for shapes we didn't enumerate.

    Caller (daemon boot OR CLI subcommand) interprets state and acts. The
    pure-inspection contract (no side effects) lets boot-time integration
    bail out cleanly via `raise SystemExit(2)` while leaving the store
    untouched for operator inspection.
    """
    names = set(db.table_names())
    has_records = RECORDS_TABLE in names
    has_staging = STAGING_TABLE in names
    old_tables = sorted(n for n in names if n.startswith(OLD_TABLE_PREFIX))

    if not has_staging and not old_tables:
        return {"state": "clean"}

    if has_staging and not has_records and not old_tables:
        return {
            "state": "partial_swap_inconsistent",
            "staging": STAGING_TABLE,
            "old_tables": old_tables,
            "reason": (
                "records_v_new present but neither records nor records_old_<ts> "
                "exist; manual recovery required."
            ),
        }

    if has_staging and has_records:
        return {
            "state": "needs_rollback",
            "old_tables": old_tables,
            "reason": (
                "records_v_new present alongside records — staging did not "
                "complete; recover by dropping records_v_new (rollback) or "
                "resuming from migration_progress.json."
            ),
        }

    if not has_staging and has_records and old_tables:
        return {
            "state": "needs_cleanup",
            "old_tables": old_tables,
            "reason": "successful swap from prior boot; drop old tables.",
        }

    if has_staging and old_tables and not has_records:
        return {
            "state": "needs_rollback",
            "old_tables": old_tables,
            "reason": (
                "records_v_new + records_old_<ts> present, records absent — "
                "swap interrupted between renames; rollback from records_old_<ts>."
            ),
        }

    return {
        "state": "unknown",
        "has_records": has_records,
        "has_staging": has_staging,
        "old_tables": old_tables,
    }


def _decrypt_field_try_keys(
    ciphertext: str,
    record_id: UUID,
    keys: list[bytes],
) -> str:
    """Decrypt iai:enc:v1: field; try each key in order until one succeeds."""
    from cryptography.exceptions import InvalidTag

    from iai_mcp.crypto import decrypt_field

    if not is_encrypted(ciphertext):
        return str(ciphertext or "")
    ad = _uuid_literal(record_id).encode("ascii")
    last_exc: Exception | None = None
    for key in keys:
        if key is None or len(key) != 32:
            continue
        try:
            return decrypt_field(ciphertext, key, associated_data=ad)
        except (InvalidTag, ValueError) as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise ValueError("no valid keys supplied for decrypt")


def _memory_record_from_raw_row_multikey(
    store: MemoryStore,
    row: dict,
    keys: list[bytes],
) -> MemoryRecord:
    """Build MemoryRecord from a Lance row dict; decrypt with key fallbacks."""
    import pandas as pd

    from uuid import UUID as _UUID

    row_uuid = _UUID(row["id"])
    structure_raw = row.get("structure_hv")
    if structure_raw is None:
        structure_hv = b""
    elif isinstance(structure_raw, (bytes, bytearray)):
        structure_hv = bytes(structure_raw)
    else:
        structure_hv = b""

    community_raw = row.get("community_id") or ""
    community_id = _UUID(community_raw) if community_raw else None

    raw_version = row.get("schema_version")
    try:
        version_int = int(raw_version) if raw_version is not None else SCHEMA_VERSION_CURRENT
    except (TypeError, ValueError):
        version_int = SCHEMA_VERSION_CURRENT
    schema_version = version_int

    lang_raw = row.get("language")
    is_empty_language = lang_raw is None or (isinstance(lang_raw, str) and lang_raw == "")
    if is_empty_language and schema_version == 1:
        language = "__LEGACY_EMPTY__"
    elif is_empty_language:
        language = "en"
    else:
        language = str(lang_raw)

    s5_raw = row.get("s5_trust_score")
    s5_trust_score = float(s5_raw) if s5_raw is not None else 0.5

    gain_raw = row.get("profile_modulation_gain_json") or "{}"
    gain_plain = _decrypt_field_try_keys(str(gain_raw), row_uuid, keys)
    try:
        profile_modulation_gain = json.loads(gain_plain) or {}
    except (TypeError, json.JSONDecodeError):
        profile_modulation_gain = {}

    last_reviewed_raw = row.get("last_reviewed")
    try:
        last_reviewed = None if pd.isna(last_reviewed_raw) else last_reviewed_raw
    except (TypeError, ValueError):
        last_reviewed = last_reviewed_raw

    literal_raw = row.get("literal_surface", "")
    literal_plain = _decrypt_field_try_keys(str(literal_raw), row_uuid, keys)

    provenance_raw = row.get("provenance_json") or "[]"
    provenance_plain = _decrypt_field_try_keys(str(provenance_raw), row_uuid, keys)
    try:
        provenance_list = json.loads(provenance_plain) if provenance_plain else []
    except (TypeError, json.JSONDecodeError):
        provenance_list = []

    rec = MemoryRecord(
        id=row_uuid,
        tier=row.get("tier", "episodic"),
        literal_surface=literal_plain,
        aaak_index=row.get("aaak_index") or "",
        embedding=(
            list(row["embedding"])
            if row.get("embedding") is not None
            else []
        ),
        community_id=community_id,
        centrality=float(row.get("centrality", 0.0) or 0.0),
        detail_level=int(row.get("detail_level", 1)),
        pinned=bool(row.get("pinned", False)),
        stability=float(row.get("stability") or 0.0),
        difficulty=float(row.get("difficulty") or 0.0),
        last_reviewed=last_reviewed,
        never_decay=bool(row.get("never_decay", False)),
        never_merge=bool(row.get("never_merge", False)),
        provenance=provenance_list,
        created_at=row.get("created_at") or datetime.now(timezone.utc),
        updated_at=row.get("updated_at") or datetime.now(timezone.utc),
        tags=json.loads(row.get("tags_json") or "[]"),
        language=language,
        s5_trust_score=s5_trust_score,
        profile_modulation_gain=profile_modulation_gain,
        schema_version=schema_version,
        structure_hv=structure_hv,
    )
    if language == "__LEGACY_EMPTY__":
        rec.language = ""
    return rec


def migrate_crypto_recover_prior_key(
    store: MemoryStore,
    prior_key: bytes,
    *,
    dry_run: bool = False,
) -> dict:
    """Re-encrypt all records under the current file key using a prior AES key.

    Use when ``.crypto.key`` was rotated or replaced while rows still carry
    ciphertext from the old key (InvalidTag under the live key). Stages into
    ``records_crypto_recover_stage``, validates full row count, atomically
    swaps ``records`` aside (``records_old_<ts>``), promotes staging to
    ``records`` — same filesystem-rename pattern as reembed migration.

    Preconditions:
        - ``detect_partial_migration`` state is ``clean`` or ``needs_cleanup``
          (no in-flight ``records_v_new`` reembed).
        - ``prior_key`` is 32 raw bytes (same format as ``.crypto.key``).

    Idempotent: if every row decrypts with the **current** key alone, returns
    ``{"no_op": True, ...}`` without creating staging or swapping.

    Returns
    -------
    dict
        ``no_op``, ``records_staged``, ``duration_sec``, ``dry_run``, ``old_table`` (if any).
    """
    from cryptography.exceptions import InvalidTag

    from iai_mcp.crypto import KEY_BYTES

    if len(prior_key) != KEY_BYTES:
        raise ValueError(f"prior_key must be {KEY_BYTES} raw bytes")

    mig = detect_partial_migration(store.db)
    if mig["state"] not in ("clean", "needs_cleanup"):
        raise RuntimeError(
            "crypto recover requires a non-partial reembed state "
            f"(got {mig['state']!r}); resolve migrate --rollback/--resume first."
        )

    cur_key = store._key()
    key_chain = [cur_key, prior_key] if prior_key != cur_key else [cur_key]

    names = _db_table_names_set(store.db)
    if CRYPTO_RECOVER_STAGING in names:
        try:
            store.db.drop_table(CRYPTO_RECOVER_STAGING)
        except Exception as exc:
            raise RuntimeError(
                f"drop stale {CRYPTO_RECOVER_STAGING} failed: {exc}"
            ) from exc

    orig_tbl = store.db.open_table(RECORDS_TABLE)
    orig_count = int(orig_tbl.count_rows())
    if orig_count == 0:
        return {"no_op": True, "reason": "empty_store", "records_staged": 0, "dry_run": dry_run}

    df = orig_tbl.to_pandas()
    needs_prior = 0
    for _, r in df.iterrows():
        rid = UUID(str(r["id"]))
        lit = str(r.get("literal_surface") or "")
        if not is_encrypted(lit):
            continue
        try:
            _decrypt_field_try_keys(lit, rid, [cur_key])
        except (InvalidTag, ValueError):
            try:
                _decrypt_field_try_keys(lit, rid, [prior_key])
                needs_prior += 1
            except (InvalidTag, ValueError):
                raise RuntimeError(
                    f"record {rid}: literal_surface not decryptable with current "
                    "or prior key — run crypto redact-undecryptable or restore backup"
                ) from None

    if needs_prior == 0:
        return {
            "no_op": True,
            "reason": "all_rows_decrypt_with_current_key",
            "records_staged": 0,
            "dry_run": dry_run,
        }

    if dry_run:
        return {
            "no_op": False,
            "dry_run": True,
            "would_stage": orig_count,
            "rows_needing_prior_key": needs_prior,
        }

    schema = orig_tbl.schema
    staging_tbl = store.db.create_table(CRYPTO_RECOVER_STAGING, schema=schema)
    staged = 0
    t0 = time.time()
    for _, r in df.iterrows():
        row_dict = r.to_dict()
        rec = _memory_record_from_raw_row_multikey(store, row_dict, key_chain)
        staging_tbl.add([store._to_row(rec)])
        staged += 1

    if staged != orig_count:
        try:
            store.db.drop_table(CRYPTO_RECOVER_STAGING)
        except Exception:
            pass
        raise RuntimeError(
            f"staging row count mismatch: staged={staged} orig={orig_count}"
        )

    duration_sec = time.time() - t0
    try:
        write_event(
            store,
            kind="migration_crypto_recover",
            data={
                "records_staged": staged,
                "duration_sec": duration_sec,
                "rows_needed_prior_key": needs_prior,
            },
            severity="info",
        )
    except Exception:
        pass

    ts = int(time.time())
    old_name = f"{OLD_TABLE_PREFIX}{ts}"
    _swap_tables_filesystem(store.db, source=RECORDS_TABLE, dest=old_name)
    _swap_tables_filesystem(
        store.db, source=CRYPTO_RECOVER_STAGING, dest=RECORDS_TABLE
    )

    return {
        "no_op": False,
        "records_staged": staged,
        "duration_sec": duration_sec,
        "dry_run": False,
        "old_table": old_name,
        "rows_needed_prior_key": needs_prior,
    }


REDACT_UNDECRYPTABLE_MARKER = "<REDACTED: pre-2026-04-30 key rotation>"


def migrate_redact_undecryptable_records(store: MemoryStore) -> dict:
    """Replace literal_surface that cannot decrypt with ``REDACT_UNDECRYPTABLE_MARKER``.

    Preserves embeddings, tier, tags, provenance column bytes (best-effort:
    provenance_json is left unchanged — only literal_surface is redacted per
    mandate). Emits ``crypto_redaction`` per changed row. Idempotent.
    """
    from cryptography.exceptions import InvalidTag

    tbl = store.db.open_table(RECORDS_TABLE)
    if tbl.count_rows() == 0:
        return {"redacted": 0, "skipped_ok": 0, "skipped_plain": 0}

    df = tbl.to_pandas()
    redacted = 0
    skipped_ok = 0
    skipped_plain = 0
    for _, r in df.iterrows():
        rid = UUID(str(r["id"]))
        lit = str(r.get("literal_surface") or "")
        if not is_encrypted(lit):
            skipped_plain += 1
            continue
        try:
            plain = store._decrypt_for_record(rid, lit)
        except (InvalidTag, ValueError):
            plain = None
        if plain is not None:
            # Already decryptable (includes idempotent prior redaction).
            skipped_ok += 1
            continue
        prov_raw = str(r.get("provenance_json") or "[]")
        try:
            if is_encrypted(prov_raw):
                prov_plain = store._decrypt_for_record(rid, prov_raw)
            else:
                prov_plain = prov_raw
        except (InvalidTag, ValueError):
            prov_plain = "[]"
        gain_raw = str(r.get("profile_modulation_gain_json") or "{}")
        try:
            if is_encrypted(gain_raw):
                gain_plain = store._decrypt_for_record(rid, gain_raw)
            else:
                gain_plain = gain_raw
        except (InvalidTag, ValueError):
            gain_plain = "{}"
        new_lit = store._encrypt_for_record(rid, REDACT_UNDECRYPTABLE_MARKER)
        new_prov = store._encrypt_for_record(rid, prov_plain)
        new_gain = store._encrypt_for_record(rid, gain_plain)
        tbl.update(
            where=f"id = '{_uuid_literal(rid)}'",
            values={
                "literal_surface": new_lit,
                "provenance_json": new_prov,
                "profile_modulation_gain_json": new_gain,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        redacted += 1
        try:
            write_event(
                store,
                kind="crypto_redaction",
                data={"record_id": str(rid), "reason": "undecryptable_literal"},
                severity="warning",
            )
        except Exception:
            pass

    return {
        "redacted": redacted,
        "skipped_ok": skipped_ok,
        "skipped_plain": skipped_plain,
    }


def _rollback(db, store: MemoryStore) -> int:
    """Roll back a partial reembed migration. / .

    Behaviour by state (per `detect_partial_migration` taxonomy):
      - records present + records_v_new present (mid-stage crash):
        DROP records_v_new; records is intact, no rename needed.
      - records absent + records_old_<ts> present (mid-swap crash variant):
        Rename records_old_<newest_ts> -> records; drop records_v_new if
        present.
      - records present + records_old_<ts> present (deferred-cleanup state):
        Drop records_old_<ts> (treats rollback as "discard old snapshot"
        when the new table is already in place).
      - clean: no-op, return 0.

    Drops `migration_progress.json` if present.

    Returns 0 on success, 1 on user-correctable error (e.g. nothing to roll
    back to), 2 on unrecoverable.
    """
    names = set(db.table_names())
    has_records = RECORDS_TABLE in names
    has_staging = STAGING_TABLE in names
    old_tables = sorted(n for n in names if n.startswith(OLD_TABLE_PREFIX))

    try:
        # Mid-stage crash: drop the partial staging.
        if has_staging and has_records:
            db.drop_table(STAGING_TABLE)
            _progress_clear(store)
            log.info(
                "migrate_reembed_rollback_drop_staging",
                extra={"records_count": db.open_table(RECORDS_TABLE).count_rows()},
            )
            return 0

        # Mid-swap crash: restore from the newest old table.
        if not has_records and old_tables:
            newest_old = old_tables[-1]
            if has_staging:
                db.drop_table(STAGING_TABLE)
            # Filesystem-level rename: records_old_<ts>.lance -> records.lance.
            _swap_tables_filesystem(db, source=newest_old, dest=RECORDS_TABLE)
            # Refresh embed_dim from the restored table's schema
            # (mirrors store._ensure_tables lines 285-296).
            try:
                tbl = db.open_table(RECORDS_TABLE)
                emb_field = tbl.schema.field("embedding")
                actual_dim = getattr(emb_field.type, "list_size", None)
                if actual_dim and int(actual_dim) > 0:
                    store._embed_dim = int(actual_dim)
            except Exception:
                pass
            _progress_clear(store)
            log.info(
                "migrate_reembed_rollback_restore_old",
                extra={
                    "restored_from": newest_old,
                    "records_count": db.open_table(RECORDS_TABLE).count_rows(),
                },
            )
            return 0

        # Deferred-cleanup state: discard the old snapshot at the user's
        # request (rollback semantics here treat "discard old after
        # successful swap" as a valid operator action).
        if has_records and old_tables and not has_staging:
            for old in old_tables:
                try:
                    db.drop_table(old)
                except Exception as exc:
                    log.warning(
                        "migrate_reembed_rollback_drop_old_failed",
                        extra={"table": old, "error": str(exc)[:160]},
                    )
            _progress_clear(store)
            return 0

        # Clean state: nothing to roll back.
        if has_records and not has_staging and not old_tables:
            _progress_clear(store)
            return 0

        # Catastrophic: records absent + no old table to restore.
        log.error(
            "migrate_reembed_rollback_unrecoverable",
            extra={
                "has_records": has_records,
                "has_staging": has_staging,
                "old_tables": old_tables,
            },
        )
        return 2
    except Exception as exc:
        log.error(
            "migrate_reembed_rollback_failed",
            extra={"error": str(exc)[:200]},
        )
        return 1


def _resume(db, store: MemoryStore, target_embedder) -> int:
    """Resume a partial reembed migration from `migration_progress.json`.

    Reads the checkpoint to recover `staged_ids` and `target_dim`. Continues
    the staging loop over rows in the live `records` table that are NOT
    already in `staged_ids`. After staging completes, runs (validate)
    and (atomic swap), then drops the progress file.

    Returns 0 on success, 1 on user-correctable error (no progress file,
    target_dim mismatch with the embedder), 2 on unrecoverable.
    """
    progress_state = _progress_read(store)
    if not progress_state:
        log.error(
            "migrate_reembed_resume_no_progress_file",
            extra={"path": str(_progress_path(store))},
        )
        return 1

    target_dim = int(target_embedder.DIM)
    saved_target_dim = int(progress_state.get("target_dim") or 0)
    if saved_target_dim and saved_target_dim != target_dim:
        log.error(
            "migrate_reembed_resume_dim_mismatch",
            extra={
                "saved_target_dim": saved_target_dim,
                "embedder_dim": target_dim,
            },
        )
        return 1

    names = set(db.table_names())
    if RECORDS_TABLE not in names:
        log.error("migrate_reembed_resume_records_missing")
        return 2

    if STAGING_TABLE not in names:
        # Staging table was dropped (or never created). Re-create it at
        # the target dim and re-stage everything.
        target_tbl = db.create_table(
            STAGING_TABLE, schema=_records_schema_at_dim(target_dim)
        )
        already_staged: set[str] = set()
    else:
        target_tbl = db.open_table(STAGING_TABLE)
        already_staged = set(progress_state.get("staged_ids") or [])

    source_dim = int(store.embed_dim)
    started_at_iso = progress_state.get(
        "started_at", datetime.now(timezone.utc).isoformat()
    )
    total = db.open_table(RECORDS_TABLE).count_rows()
    last_idx = int(progress_state.get("row_index") or 0)

    t0 = time.time()
    try:
        staged_count, failures = _stage_loop(
            store,
            target_embedder,
            target_dim,
            target_tbl,
            store.iter_records(),
            total=total,
            started_at_iso=started_at_iso,
            started_idx=last_idx + 1,
            already_staged_ids=already_staged,
        )
    except (KeyboardInterrupt, SystemExit):
        # Re-kill mid-resume: progress file is up-to-date; another --resume
        # picks up where this one left off.
        raise
    except Exception as exc:
        log.error(
            "migrate_reembed_resume_stage_failed",
            extra={"error": str(exc)[:200]},
        )
        return 2

    # Combine prior-run staged count with this run's staged count for the
    # event payload — total updated rows is what the user/audit cares about.
    total_staged = len(already_staged) + staged_count

    duration_sec = time.time() - t0
    try:
        _validate_and_swap(
            store,
            source_dim=source_dim,
            target_dim=target_dim,
            target_embedder=target_embedder,
            staged_count=total_staged,
            failures=failures,
            duration_sec=duration_sec,
        )
    except RuntimeError as exc:
        log.error(
            "migrate_reembed_resume_validate_failed",
            extra={"error": str(exc)[:200]},
        )
        return 2
    return 0


# ---------------------------------------------------------------------------
# v2 -> v3 encryption migration
# ---------------------------------------------------------------------------


def _encrypt_or_passthrough(
    store: MemoryStore,
    record_id: UUID,
    value: str,
) -> tuple[str, bool]:
    """Encrypt `value` if it is plaintext; pass through if already encrypted.

    Returns (new_value, was_encrypted_now). `was_encrypted_now` is True only
    when the value flipped from plaintext to ciphertext on this call.
    """
    if is_encrypted(value):
        return value, False
    ad = _uuid_literal(record_id).encode("ascii")
    ct = encrypt_field(value or "", store._key(), associated_data=ad)
    return ct, True


def migrate_encryption_v2_to_v3(
    store: MemoryStore,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """One-shot encryption migration for .

    Scans both the records table and the events table; anything whose
    sensitive column currently lives as plaintext is re-encrypted in place.
    Idempotent: rows already carrying the iai:enc:v1: prefix are left alone.

    Records columns re-encrypted:
    - literal_surface (user content)
    - provenance_json (session cues + quotes)
    - profile_modulation_gain_json (learned per-user data)

    Events columns re-encrypted:
    - data_json (may contain quoted user content in some event kinds)

    Parameters
    ----------
    store: open MemoryStore (encryption key auto-loaded from keyring).
    dry_run: when True, count migrable rows without writing.
    progress: optional callback(idx, total) for CLI / external progress UIs.

    Returns a dict with record and event migration counts plus duration.

    preserved: encryption is lossless; decrypt + get() returns the
    exact same string bytes the caller originally stored.
    """
    t0 = time.time()
    result = {
        "records_migrated": 0,
        "events_migrated": 0,
        "records_scanned": 0,
        "events_scanned": 0,
        "duration_sec": 0.0,
    }

    # ----- records table sweep -----
    records_tbl = store.db.open_table(RECORDS_TABLE)
    records_df = records_tbl.to_pandas()
    result["records_scanned"] = int(len(records_df))

    records_updates: list[dict] = []
    record_total = len(records_df)
    for idx, (_, row) in enumerate(records_df.iterrows()):
        if progress is not None:
            try:
                progress(idx, record_total)
            except Exception:
                pass
        try:
            rid = UUID(str(row["id"]))
        except (ValueError, TypeError):
            continue

        literal_raw = row.get("literal_surface") or ""
        prov_raw = row.get("provenance_json") or "[]"
        gain_raw = row.get("profile_modulation_gain_json") or "{}"

        any_plaintext = any(
            not is_encrypted(v) for v in (literal_raw, prov_raw, gain_raw)
        )
        if not any_plaintext:
            continue  # Row fully encrypted already -- skip (idempotent).

        if dry_run:
            result["records_migrated"] += 1
            continue

        new_literal, _ = _encrypt_or_passthrough(store, rid, literal_raw)
        new_prov, _ = _encrypt_or_passthrough(store, rid, prov_raw)
        new_gain, _ = _encrypt_or_passthrough(store, rid, gain_raw)
        records_updates.append(
            {
                "id": _uuid_literal(rid),
                "literal_surface": new_literal,
                "provenance_json": new_prov,
                "profile_modulation_gain_json": new_gain,
            }
        )
        result["records_migrated"] += 1

    if not dry_run and records_updates:
        now = datetime.now(timezone.utc)
        import pyarrow as pa
        update_tbl = pa.table(
            {
                "id": [u["id"] for u in records_updates],
                "literal_surface": [u["literal_surface"] for u in records_updates],
                "provenance_json": [u["provenance_json"] for u in records_updates],
                "profile_modulation_gain_json": [
                    u["profile_modulation_gain_json"] for u in records_updates
                ],
                "updated_at": [now] * len(records_updates),
            }
        )
        try:
            records_tbl.merge_insert("id").when_matched_update_all().execute(update_tbl)
        except Exception:
            # Rule 1 fallback: per-id tbl.update when merge_insert is unavailable.
            for u in records_updates:
                try:
                    records_tbl.update(
                        where=f"id = '{u['id']}'",
                        values={
                            "literal_surface": u["literal_surface"],
                            "provenance_json": u["provenance_json"],
                            "profile_modulation_gain_json": u[
                                "profile_modulation_gain_json"
                            ],
                            "updated_at": now,
                        },
                    )
                except Exception:
                    continue

    # ----- events table sweep -----
    events_tbl = store.db.open_table(EVENTS_TABLE)
    events_df = events_tbl.to_pandas()
    result["events_scanned"] = int(len(events_df))

    events_updates: list[dict] = []
    for _, row in events_df.iterrows():
        data_raw = row.get("data_json") or "{}"
        if is_encrypted(data_raw):
            continue
        event_id = str(row["id"])
        if dry_run:
            result["events_migrated"] += 1
            continue
        ad = event_id.encode("ascii")
        new_data = encrypt_field(data_raw, store._key(), associated_data=ad)
        events_updates.append({"id": event_id, "data_json": new_data})
        result["events_migrated"] += 1

    if not dry_run and events_updates:
        for u in events_updates:
            try:
                events_tbl.update(
                    where=f"id = '{u['id']}'",
                    values={"data_json": u["data_json"]},
                )
            except Exception:
                continue

    result["duration_sec"] = time.time() - t0

    # ----- emit audit event -----
    if not dry_run and (
        result["records_migrated"] > 0 or result["events_migrated"] > 0
    ):
        write_event(
            store,
            kind="migration_v2_to_v3",
            data={
                "record_count": result["records_migrated"],
                "event_count": result["events_migrated"],
                "duration_sec": result["duration_sec"],
                "columns_encrypted": [
                    "records.literal_surface",
                    "records.provenance_json",
                    "records.profile_modulation_gain_json",
                    "events.data_json",
                ],
                "algorithm": "AES-256-GCM",
                "format": "iai:enc:v1:",
            },
            severity="info",
        )

    return result


# ---------------------------------------------------------------------------
# : v3 -> v4 TEM factorization migration
# ---------------------------------------------------------------------------


def migrate_hd_vector_to_structure_hv_v3_to_v4(
    store: MemoryStore,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """: rename `hd_vector_json` (pa.string) -> `structure_hv`
    (pa.binary) and backfill every record with a freshly-bound
    structural hypervector via tem.bind_structure().

    Idempotency contract:
        Rows that satisfy BOTH (a) schema_version >= 4 AND (b) non-empty
        structure_hv are skipped. Any row failing either condition is migrated.

    CR-01 / SQL-injection guard (carried over from 02-06 lesson):
        every WHERE / DELETE predicate routes through store._uuid_literal so
        a poisoned UUID cannot inject SQL content.

    Resumability:
        Each record is delete+insert'd individually; a crash mid-batch leaves
        a partially-migrated store that the next run picks up cleanly.

    :
        literal_surface is preserved byte-for-byte. The migration only touches
        structure_hv + schema_version on each row.

    LanceDB schema-rename note:
        For stores created on the new schema (the typical case after this plan
        ships) the column already exists as `structure_hv` (pa.binary()). For
        legacy stores still on the old `hd_vector_json` (pa.string()) schema,
        the rebuild is implicit -- store.insert() writes through the new
        schema, so the delete+insert per-row migration produces a fully-renamed
        table after one full sweep.

    Parameters
    ----------
    store: open MemoryStore.
    dry_run: when True, count migrable rows without writing.
    progress: optional callback(idx, total) for CLI / external progress UIs.

    Returns
    -------
    dict with keys: processed, updated, skipped, duration_ms,
                    column_renamed_from, column_renamed_to.
    """
    t0 = time.time()
    result: dict = {
        "processed": 0,
        "updated": 0,
        "skipped": 0,
        "duration_ms": 0.0,
        "column_renamed_from": "hd_vector_json",
        "column_renamed_to": "structure_hv",
    }

    # We use store.all_records() so the read path normalises legacy v3 rows
    # (with the old `hd_vector_json` column) into MemoryRecord instances with
    # an empty structure_hv -- giving the migration a uniform write surface.
    all_records = store.all_records()
    total = len(all_records)
    result["processed"] = total

    # Lazy import: tem.py is part of ; importing it at module top
    # would create a load-time cycle (migrate.py is imported by cli.py which
    # is imported by sometimes-called CLI tooling -- keep it lazy).
    from iai_mcp.tem import bind_structure
    from iai_mcp.types import (
        SCHEMA_VERSION_V4,
        STRUCTURE_HV_BYTES,
    )

    # Per-row delete+insert in the manner of migrate_v1_to_v2 (CR-01-safe).
    tbl = store.db.open_table(RECORDS_TABLE)
    for idx, record in enumerate(all_records):
        if progress is not None:
            try:
                progress(idx, total)
            except Exception:
                pass

        # Idempotency: already at v4 with a populated structure_hv -> skip.
        already_v4 = record.schema_version >= SCHEMA_VERSION_V4
        has_full_hv = (
            isinstance(record.structure_hv, (bytes, bytearray))
            and len(record.structure_hv) == STRUCTURE_HV_BYTES
        )
        if already_v4 and has_full_hv:
            result["skipped"] += 1
            continue

        if dry_run:
            result["updated"] += 1
            continue

        # Compute the canonical structure_hv if this row hasn't got one yet.
        # only structure_hv + schema_version mutate; literal_surface
        # and every other field flow through unchanged.
        if not has_full_hv:
            record.structure_hv = bind_structure(record)
        record.schema_version = SCHEMA_VERSION_V4

        # CR-01 guarded delete + insert. The _uuid_literal call sanitises the
        # UUID before it enters the WHERE predicate -- a poisoned UUID would
        # raise ValueError on canonical-form check, never reaching LanceDB.
        try:
            tbl.delete(f"id = '{_uuid_literal(record.id)}'")
        except Exception:
            # Diagnostic-only: a missing row still gets re-inserted below.
            pass
        store.insert(record)
        result["updated"] += 1

    result["duration_ms"] = (time.time() - t0) * 1000.0

    # Audit-event emission per the established convention (no-op on dry_run).
    if not dry_run and (result["updated"] > 0 or result["skipped"] > 0):
        write_event(
            store,
            kind="migration_v3_to_v4",
            data={
                "processed": result["processed"],
                "updated": result["updated"],
                "skipped": result["skipped"],
                "duration_ms": result["duration_ms"],
                "column_renamed_from": result["column_renamed_from"],
                "column_renamed_to": result["column_renamed_to"],
            },
            severity="info",
        )

    return result


# ---------------------------------------------------------------------------
# R8: cleanup migration for accumulated schema duplicates
# ---------------------------------------------------------------------------


def cleanup_schema_duplicates(
    store: MemoryStore,
    *,
    apply: bool = False,
    store_path: "Path | None" = None,
) -> dict:
    """Group semantic schema records by `pattern:*` tag; keep oldest; soft-delete the rest.

    R8: a one-shot reversible cleanup of duplicates that accumulated
    in the production store BEFORE made `persist_schema` idempotent.
    NOT a schema_version v-bump — this is a maintenance op that runs on
    demand, never automatically. Beer VSM S2 anti-oscillation + Ashby
    ultrastability mandate dry-run default + snapshot before write +
    soft-delete via tier rename + idempotency.

    Parameters
    ----------
    store : MemoryStore
        Open store (connected to the LanceDB directory under inspection).
    apply : bool
        False (default) -- dry-run, mutate nothing, return diff summary.
        True -- snapshot the LanceDB tables dir, reinforce edges, soft-delete
        duplicates by renaming their tier to "semantic_pruned" + flipping
        pinned/never_decay to False.
    store_path : Path | None
        IAI root directory (the path passed to MemoryStore(); contains the
        `lancedb/` subdir with the actual tables). When None, falls back to
        `store.root`. Snapshot lands at
        `store.root / f"lancedb-pre-cleanup-{ts}"` (sibling of `lancedb/`,
        per — recovery is `mv lancedb-pre-cleanup-{ts} lancedb`).

    Returns
    -------
    dict
        {
            "mode": "dry-run" | "apply",
            "groups": int,                      # patterns with N>1 duplicates
            "keepers": int,                     # one per group
            "pruned": int,                      # cumulative duplicates soft-deleted
            "edges_reinforced": int,            # incoming schema_instance_of edges redirected
            "snapshot_dir": str | None,         # set only on apply
        }
    """
    import shutil
    from pathlib import Path
    from datetime import datetime, timezone

    from iai_mcp.store import EDGES_TABLE
    from iai_mcp.types import SEMANTIC_PRUNED_TIER

    # --- 1. Discover pattern groups: tier='semantic' AND tag matches pattern:*
    groups: dict[str, list[MemoryRecord]] = {}
    try:
        all_records = store.all_records()
    except Exception:
        # Diagnostic-only: a read failure leaves the store untouched and
        # returns an empty summary instead of raising. Operators see the
        # empty result and can investigate.
        return {
            "mode": "apply" if apply else "dry-run",
            "groups": 0,
            "keepers": 0,
            "pruned": 0,
            "edges_reinforced": 0,
            "snapshot_dir": None,
        }

    for rec in all_records:
        if rec.tier != "semantic":
            continue
        pattern_tag = next(
            (t for t in (rec.tags or []) if t.startswith("pattern:")),
            None,
        )
        if pattern_tag is None or ":" not in pattern_tag:
            continue
        pattern = pattern_tag.split(":", 1)[1]
        groups.setdefault(pattern, []).append(rec)

    # Single-record groups are not duplicates -- nothing to do.
    dup_groups = {p: recs for p, recs in groups.items() if len(recs) > 1}

    # --- 2. Select keepers (oldest first per pattern) + identify duplicates
    keepers: list[MemoryRecord] = []
    duplicates: list[MemoryRecord] = []
    for pattern, recs in dup_groups.items():
        recs_sorted = sorted(recs, key=lambda r: r.created_at)
        keepers.append(recs_sorted[0])
        duplicates.extend(recs_sorted[1:])

    # --- 3. Plan edge redirects: count incoming schema_instance_of edges
    #         to duplicates so the dry-run can report what would be reinforced.
    edges_to_reinforce = 0
    try:
        edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
        dup_id_strs = {str(d.id) for d in duplicates}
        if dup_id_strs and "edge_type" in edges_df.columns:
            # boost_edges canonicalises (src, dst) to a sorted tuple, so the
            # duplicate appears in EITHER column. OR-count both columns —
            # each row has the dup in exactly one column, no double-count.
            mask = (
                (edges_df["edge_type"] == "schema_instance_of")
                & (
                    edges_df["dst"].isin(dup_id_strs)
                    | edges_df["src"].isin(dup_id_strs)
                )
            )
            edges_to_reinforce = int(mask.sum())
    except Exception:
        edges_to_reinforce = 0

    snapshot_dir: str | None = None

    if apply and (keepers or duplicates):
        # --- 4. Snapshot the LanceDB tables dir BEFORE any write.
        # store.root is the IAI root (contains lancedb/ subdir + state files).
        # The actual tables live at store.root / "lancedb"; the snapshot is a
        # sibling at store.root / f"lancedb-pre-cleanup-{ts}", so manual
        # recovery is `mv ~/.iai-mcp/lancedb-pre-cleanup-{ts} ~/.iai-mcp/lancedb`.
        iai_root = Path(store_path) if store_path is not None else Path(store.root)
        src_lancedb = iai_root / "lancedb"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap = iai_root / f"lancedb-pre-cleanup-{ts}"
        # If src_lancedb does not exist (e.g. legacy layout), fall back to
        # snapshotting the IAI root itself so the operator still has rollback.
        snapshot_source = src_lancedb if src_lancedb.exists() else iai_root
        shutil.copytree(snapshot_source, snap)
        snapshot_dir = str(snap)

        # --- 5. Build keeper lookup by pattern for the redirect step.
        keeper_by_pattern: dict[str, MemoryRecord] = {}
        for k in keepers:
            kp = next(
                (t for t in (k.tags or []) if t.startswith("pattern:")),
                None,
            )
            if kp and ":" in kp:
                keeper_by_pattern[kp.split(":", 1)[1]] = k

        # --- 6. Redirect edges: copy incoming schema_instance_of edges from
        # each duplicate onto its keeper BEFORE the duplicate's tier is renamed.
        # Edge reinforcement failure must NOT block the tier rename — the
        # operator can re-run cleanup to complete edge consolidation.
        try:
            edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
            for dup in duplicates:
                dp = next(
                    (t for t in (dup.tags or []) if t.startswith("pattern:")),
                    None,
                )
                if dp is None or ":" not in dp:
                    continue
                pattern = dp.split(":", 1)[1]
                keeper = keeper_by_pattern.get(pattern)
                if keeper is None or keeper.id == dup.id:
                    continue
                dup_str = str(dup.id)
                incoming_mask = (
                    (edges_df["edge_type"] == "schema_instance_of")
                    & ((edges_df["dst"] == dup_str) | (edges_df["src"] == dup_str))
                )
                incoming = edges_df[incoming_mask]
                if incoming.empty:
                    continue
                pairs: list[tuple[UUID, UUID]] = []
                for _, row in incoming.iterrows():
                    # Determine the OTHER side of the edge (the evidence node)
                    # — it's whichever column does NOT carry the duplicate's id.
                    other_str = (
                        row["src"] if row["dst"] == dup_str else row["dst"]
                    )
                    if other_str == dup_str:
                        # Self-edge sanity guard.
                        continue
                    try:
                        other_id = UUID(str(other_str))
                    except (TypeError, ValueError):
                        continue
                    pairs.append((other_id, keeper.id))
                if pairs:
                    store.boost_edges(
                        pairs,
                        edge_type="schema_instance_of",
                        delta=0.1,
                    )
        except Exception:
            # Diagnostic: see comment at section header.
            pass

        # --- 7. Soft-delete via tier rename: delete + re-insert each duplicate
        # with tier=semantic_pruned, pinned=False, never_decay=False.
        # Other fields preserved (literal_surface, embedding, provenance, etc.)
        # for reverse-migration recoverability.
        for dup in duplicates:
            try:
                store.delete(dup.id)
                pruned_rec = MemoryRecord(
                    id=dup.id,
                    tier=SEMANTIC_PRUNED_TIER,
                    literal_surface=dup.literal_surface,
                    aaak_index=dup.aaak_index,
                    embedding=dup.embedding,
                    community_id=dup.community_id,
                    centrality=dup.centrality,
                    detail_level=dup.detail_level,
                    pinned=False,                 # pruned rows are unpinned
                    stability=dup.stability,
                    difficulty=dup.difficulty,
                    last_reviewed=dup.last_reviewed,
                    never_decay=False,            # pruned rows can decay
                    never_merge=dup.never_merge,
                    provenance=dup.provenance,
                    created_at=dup.created_at,
                    updated_at=datetime.now(timezone.utc),
                    tags=dup.tags,
                    language=dup.language,
                    s5_trust_score=dup.s5_trust_score,
                    profile_modulation_gain=dup.profile_modulation_gain,
                    schema_version=dup.schema_version,
                    structure_hv=dup.structure_hv,
                )
                store.insert(pruned_rec)
            except Exception:
                # Per-record continuation: a single failed soft-delete must
                # not abort the rest of the batch. Operator can re-run.
                continue

    # --- 8. Emit summary event + return summary dict
    summary: dict = {
        "mode": "apply" if apply else "dry-run",
        "groups": len(dup_groups),
        "keepers": len(keepers),
        "pruned": len(duplicates),
        "edges_reinforced": int(edges_to_reinforce),
        "snapshot_dir": snapshot_dir,
    }
    try:
        write_event(
            store,
            kind="schema_cleanup_run",
            data=summary,
            severity="info",
            source_ids=[k.id for k in keepers[:5]] if keepers else None,
        )
    except Exception:
        # Diagnostic-only: an event-write failure must not invalidate the
        # cleanup itself.
        pass
    return summary
