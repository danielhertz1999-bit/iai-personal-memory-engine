"""Resource bound regression tests for HippoDB storage.

Tests verify storage layout correctness and that repeated open/close cycles
do not leak process memory.
"""
from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from iai_mcp.hippo import HippoDB
from iai_mcp.types import EMBED_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(seed: int) -> dict:
    """Return a minimal records-table row dict ready for HippoTable.add()."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(EMBED_DIM).astype(np.float32).tolist()
    return {
        "id": str(uuid.uuid4()),
        "tier": "episodic",
        "literal_surface": f"footprint test seed {seed}",
        "aaak_index": "",
        "embedding": vec,
        "structure_hv": b"",
        "community_id": None,
        "centrality": 0.0,
        "detail_level": 1,
        "pinned": False,
        "stability": 0.0,
        "difficulty": 0.0,
        "last_reviewed": None,
        "never_decay": False,
        "never_merge": False,
        "tombstoned_at": None,
        "schema_bypass": None,
        "labile_until": None,
        "provenance_json": "[]",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "tags_json": "[]",
        "language": "en",
        "s5_trust_score": 0.5,
        "profile_modulation_gain_json": "{}",
        "schema_version": 4,
        "wing": None,
        "room": None,
        "drawer": None,
        "valence": None,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hippo_dir_size_bounded_per_record(tmp_path: Path) -> None:
    """Total hippo/ directory size must be < FIXED_BASE + N * per_record_cap bytes.

    SQLite allocates pages in 4 KB blocks (minimum 32 pages = 128 KB default).
    hnswlib also has a fixed index header. The bound is:
      total < 256 KB fixed base + N * 8 KB per-record cap

    Float32 embedding BLOB: 384 * 4 = 1536 bytes.
    Full row including text columns + SQLite B-tree overhead + hnswlib vector
    entry: ~6–8 KB per record at small N; drops toward ~4 KB at large N.
    """
    n = 20
    db = HippoDB(tmp_path)
    try:
        tbl = db.open_table("records")
        rows = [_make_row(seed=8000 + i) for i in range(n)]
        tbl.add(rows)
    finally:
        db.close()

    hippo_dir = tmp_path / "hippo"
    total_bytes = sum(f.stat().st_size for f in hippo_dir.rglob("*") if f.is_file())
    fixed_base_bytes = 256 * 1024  # 256 KB for SQLite pages + hnswlib header
    per_record_cap = 8 * 1024      # 8 KB per record cap
    max_expected = fixed_base_bytes + n * per_record_cap
    assert total_bytes < max_expected, (
        f"hippo/ occupies {total_bytes} bytes for {n} records; "
        f"expected < {max_expected} bytes "
        f"(256 KB base + {n} * 8 KB/record)"
    )


def test_vector_blob_encoding_is_float32_not_float64(tmp_path: Path) -> None:
    """Embedding stored as SQLite BLOB must be exactly EMBED_DIM * 4 bytes (float32).

    If encoding were float64 the blob would be EMBED_DIM * 8 bytes — a 2x
    storage regression that would violate the byte-strict migration contract.
    """
    db = HippoDB(tmp_path)
    try:
        row = _make_row(seed=9001)
        tbl = db.open_table("records")
        tbl.add([row])
        row_id = row["id"]
    finally:
        db.close()

    db_path = tmp_path / "hippo" / "brain.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    result = conn.execute(
        "SELECT embedding FROM records WHERE id = ?", (row_id,)
    ).fetchone()
    conn.close()

    assert result is not None, "inserted row should be readable from SQLite"
    blob = bytes(result["embedding"])
    expected_len = EMBED_DIM * 4  # float32: 4 bytes per element
    assert len(blob) == expected_len, (
        f"embedding BLOB is {len(blob)} bytes; expected {expected_len} "
        f"(float32 = {EMBED_DIM} * 4). Got {len(blob) // EMBED_DIM} bytes/element "
        f"(float64 would be {EMBED_DIM * 8} bytes)"
    )


def test_no_leak_on_repeated_open_close(tmp_path: Path) -> None:
    """100 open+close cycles must not grow RSS by more than 10 MB.

    Each cycle: open HippoDB, insert 1 row, close. The test checks that
    accumulated RSS growth stays bounded (no file-handle / lock-fd leaks).
    """
    try:
        import resource
        rss_available = True
    except ImportError:
        rss_available = False

    def _get_rss_mb() -> float:
        if rss_available:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # macOS: ru_maxrss is in bytes; Linux: in kilobytes.
            import platform
            if platform.system() == "Darwin":
                return usage.ru_maxrss / (1024 * 1024)
            return usage.ru_maxrss / 1024
        return 0.0

    cycles = 100
    rss_before = _get_rss_mb()

    for i in range(cycles):
        db = HippoDB(tmp_path)
        try:
            tbl = db.open_table("records")
            tbl.add([_make_row(seed=7000 + i)])
        finally:
            db.close()

    rss_after = _get_rss_mb()

    if rss_available:
        growth_mb = rss_after - rss_before
        assert growth_mb < 10.0, (
            f"RSS grew by {growth_mb:.1f} MB over {cycles} open/close cycles; "
            f"expected < 10 MB (possible fd or lock leak)"
        )
