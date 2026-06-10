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


def _make_row(seed: int) -> dict:
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


def test_hippo_dir_size_bounded_per_record(tmp_path: Path) -> None:
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
    fixed_base_bytes = 256 * 1024
    per_record_cap = 8 * 1024
    max_expected = fixed_base_bytes + n * per_record_cap
    assert total_bytes < max_expected, (
        f"hippo/ occupies {total_bytes} bytes for {n} records; "
        f"expected < {max_expected} bytes "
        f"(256 KB base + {n} * 8 KB/record)"
    )


def test_vector_blob_encoding_is_float32_not_float64(tmp_path: Path) -> None:
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
    expected_len = EMBED_DIM * 4
    assert len(blob) == expected_len, (
        f"embedding BLOB is {len(blob)} bytes; expected {expected_len} "
        f"(float32 = {EMBED_DIM} * 4). Got {len(blob) // EMBED_DIM} bytes/element "
        f"(float64 would be {EMBED_DIM * 8} bytes)"
    )


def test_no_leak_on_repeated_open_close(tmp_path: Path) -> None:
    try:
        import resource
        rss_available = True
    except ImportError:
        rss_available = False

    def _get_rss_mb() -> float:
        if rss_available:
            usage = resource.getrusage(resource.RUSAGE_SELF)
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
