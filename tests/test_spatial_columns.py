"""Regression tests for spatial-scaffold schema columns.

Pins the acceptance contract: records schema carries three nullable string
columns `wing`, `room`, `drawer`, and a half-migration-safe inline migration
adds them to pre-existing tables on store open.

    T1: Fresh-store open exposes wing/room/drawer in tbl.schema.names.
    T2: Pre-existing-store open (simulated by dropping the three columns)
        adds the columns back; second open is a no-op (idempotent).
    T3: Half-migrated store (only `wing` present) gets `room` + `drawer`
        added; `wing` stays put (column-presence checks are per-column).
    T4: Legacy rows inserted before the migration read back with
        wing=None, room=None, drawer=None.
    T5: `add_columns` exception is re-raised as RuntimeError naming the
        failing column set (FAIL-LOUD).

Fixtures are inline. Synthetic stores
use tmp_path; embedder is bypassed via the keyring stub so test runtime
stays in the single-digit-second budget.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pytest

from iai_mcp.store import RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord


# Autouse fixture: pin IAI_MCP_STORE to tmp_path so per-test MemoryStore
# construction stays isolated from the user's real store. Mirrors the
# shared fixture structure minus the spatial-specific env-var wipes
# (none of those vars affect schema migration).
@pytest.fixture(autouse=True)
def _isolate_iai_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.setenv("IAI_MCP_KEYRING_BYPASS", "true")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-pp")
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    # Defensively wipe the three spatial env vars so the legacy-row
    # assertion below (NULL wing/room/drawer) never gets clobbered by a
    # sibling test that left auto_tag enabled in the session env. With
    # auto_tag absent the `_load_spatial_config` default keeps the
    # `_maybe_spatial_tag` helper a no-op and the row lands with NULL
    # spatial fields exactly as the pre-spatial-scaffold behaviour described.
    monkeypatch.delenv("IAI_MCP_SPATIAL_AUTO_TAG", raising=False)
    monkeypatch.delenv("IAI_MCP_SPATIAL_DEFAULT_WING", raising=False)
    monkeypatch.delenv("IAI_MCP_SPATIAL_DRY_RUN", raising=False)


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "store"), user_id="alice")


def _make_record(*, embed_dim: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface="alice prefers tea over coffee",
        aaak_index="",
        embedding=[0.01] * embed_dim,
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


# ---------------------------------------------------------------------------
# T1: Fresh-store open exposes wing/room/drawer
# ---------------------------------------------------------------------------


def test_fresh_store_has_spatial_columns(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    names = s.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" in names, names
    assert "room" in names, names
    assert "drawer" in names, names


# ---------------------------------------------------------------------------
# T2: Pre-existing store gets columns added; second open is a no-op
# ---------------------------------------------------------------------------


def test_preexisting_store_migrates_spatial_columns(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    tbl = s.db.open_table(RECORDS_TABLE)
    # Simulate a pre-migration table by dropping the three columns.
    tbl.drop_columns(["wing", "room", "drawer"])
    names_pre = s.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" not in names_pre
    assert "room" not in names_pre
    assert "drawer" not in names_pre
    # Re-open: migration must add them back.
    s2 = MemoryStore(str(tmp_path / "store"), user_id="alice")
    names_post = s2.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" in names_post
    assert "room" in names_post
    assert "drawer" in names_post
    # Third open is a no-op (idempotent).
    s3 = MemoryStore(str(tmp_path / "store"), user_id="alice")
    names_post2 = s3.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" in names_post2
    assert "room" in names_post2
    assert "drawer" in names_post2


# ---------------------------------------------------------------------------
# T3: Half-migrated store -- only the missing columns are added
# ---------------------------------------------------------------------------


def test_half_migrated_store_adds_only_missing(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    tbl = s.db.open_table(RECORDS_TABLE)
    # Drop room+drawer but keep wing -- half-migrated state.
    tbl.drop_columns(["room", "drawer"])
    names_pre = s.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" in names_pre
    assert "room" not in names_pre
    assert "drawer" not in names_pre
    s2 = MemoryStore(str(tmp_path / "store"), user_id="alice")
    names_post = s2.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" in names_post  # untouched
    assert "room" in names_post
    assert "drawer" in names_post


# ---------------------------------------------------------------------------
# T4: Legacy rows read back with NULL spatial fields
# ---------------------------------------------------------------------------


def test_legacy_row_reads_as_null_spatial(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    tbl = s.db.open_table(RECORDS_TABLE)
    tbl.drop_columns(["wing", "room", "drawer"])
    # Re-open the store before inserting so the inline migration in
    # `_ensure_tables` re-adds the three columns. Because `_to_row` emits
    # `wing`/`room`/`drawer` keys (defaulting to NULL via the public
    # attribute getattr fallback), the stored schema MUST
    # carry the columns at insert time -- otherwise pyarrow rejects the
    # row with "field 'wing' does not exist in table schema". The
    # legacy-shape assertion remains: with `IAI_MCP_SPATIAL_AUTO_TAG`
    # unset (the autouse fixture clears it) the record carries no
    # spatial attrs and the row lands with NULL wing/room/drawer.
    s2 = MemoryStore(str(tmp_path / "store"), user_id="alice")
    rec = _make_record(embed_dim=s2._embed_dim)
    s2.insert(rec)
    tbl2 = s2.db.open_table(RECORDS_TABLE)
    df = tbl2.to_pandas()
    assert len(df) >= 1
    # All three columns must be present and NULL for the legacy row.
    assert "wing" in df.columns
    assert "room" in df.columns
    assert "drawer" in df.columns
    assert df["wing"].isna().all()
    assert df["room"].isna().all()
    assert df["drawer"].isna().all()


# ---------------------------------------------------------------------------
# T5: add_columns failure raises RuntimeError naming the failing column set
# ---------------------------------------------------------------------------


def test_add_columns_failure_raises_runtimeerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _make_store(tmp_path)
    tbl = s.db.open_table(RECORDS_TABLE)
    tbl.drop_columns(["wing", "room", "drawer"])

    # Monkey-patch HippoTable.add_columns to raise on the next call.
    # The migration in HippoDB._reconcile_columns routes spatial-column
    # additions through HippoTable.add_columns, so patching it here
    # exercises the failure-aggregation path.
    from iai_mcp.hippo import HippoTable

    original = HippoTable.add_columns

    def _raise(self: HippoTable, *args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated add_columns failure")

    monkeypatch.setattr(HippoTable, "add_columns", _raise)
    with pytest.raises(RuntimeError) as exc:
        MemoryStore(str(tmp_path / "store"), user_id="alice")
    msg = str(exc.value)
    # Error message must name at least one of the failing columns so an
    # operator can identify the migration that bailed out.
    assert any(name in msg for name in ("wing", "room", "drawer")), msg
    monkeypatch.setattr(HippoTable, "add_columns", original)
