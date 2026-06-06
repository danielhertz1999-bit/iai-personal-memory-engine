"""Pask teachback: pre-injection contradiction verification."""
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore


def verify_hit_set(store: "MemoryStore", hit_record_ids: list[UUID]) -> dict:
    """Detect contradicts-edges within the given hit set.

    Returns dict with keys:
      - has_contradictions: bool
      - contradiction_pairs: list[tuple[str, str]] (UUIDs as str for JSON-safety)
      - teachback_summary: str
      - hit_count: int
    """
    hit_count = len(hit_record_ids)
    if hit_count < 2:
        # Trivially no internal contradictions possible with fewer than 2 hits.
        return {
            "has_contradictions": False,
            "contradiction_pairs": [],
            "teachback_summary": f"All {hit_count} memories appear mutually consistent.",
            "hit_count": hit_count,
        }

    hit_ids_str = [str(h) for h in hit_record_ids]
    # Lazy import: keeps this module free of daemon top-level coupling.
    from iai_mcp.store import EDGES_TABLE

    df = None
    try:
        tbl = store.db.open_table(EDGES_TABLE)
        # The store supports a SQL-style IN-list filter; quoting UUID strings is safe
        # because UUID format guarantees no single-quote injection vectors.
        id_list = ", ".join(f"'{i}'" for i in hit_ids_str)
        where = (
            f"edge_type = 'contradicts' "
            f"AND src IN ({id_list}) "
            f"AND dst IN ({id_list})"
        )
        df = tbl.search().where(where).to_pandas()
    except (OSError, RuntimeError, ValueError):
        df = None

    pairs: list[tuple[str, str]] = []
    if df is not None and len(df) > 0:
        for _, row in df.iterrows():
            pairs.append((str(row["src"]), str(row["dst"])))

    has_contradictions = len(pairs) > 0
    if has_contradictions:
        sample = pairs[0]
        summary = (
            f"WARNING: {len(pairs)} conflicting memory pair(s) surfaced "
            f"among {hit_count} hits — example: ({sample[0]}, {sample[1]})."
        )
    else:
        summary = f"All {hit_count} memories appear mutually consistent."

    return {
        "has_contradictions": has_contradictions,
        "contradiction_pairs": pairs,
        "teachback_summary": summary,
        "hit_count": hit_count,
    }
