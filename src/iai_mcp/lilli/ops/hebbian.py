from __future__ import annotations

import logging
import statistics
from itertools import combinations
from uuid import UUID

import numpy as np

from iai_mcp.store import MemoryStore
from iai_mcp.types import STRUCTURE_HV_DIM

log = logging.getLogger(__name__)


STRUCTURAL_SIMILARITY_THRESHOLD: float = 0.7


def structural_similarity(a: bytes, b: bytes) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    xor = np.bitwise_xor(aa, bb)
    try:
        ham_bits = int(np.bitwise_count(xor).sum())
    except AttributeError:
        ham_bits = int(np.unpackbits(xor).sum())
    return 1.0 - (ham_bits / STRUCTURE_HV_DIM)


def strengthen_structure_edge(
    store: MemoryStore,
    src_id: UUID,
    dst_id: UUID,
    gain: float = 1.0,
) -> dict[tuple[str, str], float]:
    return store.boost_edges(
        [(src_id, dst_id)],
        delta=float(gain),
        edge_type="hebbian_structure",
    )


def co_retrieval_trigger(
    store: MemoryStore,
    hits,
    *,
    threshold: float = STRUCTURAL_SIMILARITY_THRESHOLD,
    gain: float = 1.0,
) -> int:
    pairs: list[tuple[UUID, bytes]] = []
    for h in hits:
        rec_id = getattr(h, "record_id", None) or getattr(h, "id", None)
        if rec_id is None:
            continue
        hv = getattr(h, "structure_hv", None)
        if hv is None:
            rec = store.get(rec_id)
            if rec is None:
                continue
            hv = rec.structure_hv
        pairs.append((rec_id, hv or b""))

    fired = 0
    for (a_id, a_hv), (b_id, b_hv) in combinations(pairs, 2):
        if structural_similarity(a_hv, b_hv) >= threshold:
            try:
                strengthen_structure_edge(store, a_id, b_id, gain=gain)
                fired += 1
            except (OSError, RuntimeError, ValueError):
                continue
    return fired


SIMILARITY_WINDOW_MIN_SAMPLES: int = 10
SIMILARITY_WINDOW_STDDEV_THRESHOLD_DEFAULT: float = 0.05
_TELEMETRY_RANK_DEFICIENCY_KIND: str = "rank_deficiency_warning"


def monitor_similarity_window(
    store,
    window: list[float],
    *,
    threshold: float = SIMILARITY_WINDOW_STDDEV_THRESHOLD_DEFAULT,
) -> None:
    if store is None:
        return
    if len(window) < SIMILARITY_WINDOW_MIN_SAMPLES:
        return
    stddev = statistics.pstdev(window)
    if stddev >= threshold:
        return
    try:
        from iai_mcp import events
        kind = getattr(events, "TELEMETRY_RANK_DEFICIENCY", _TELEMETRY_RANK_DEFICIENCY_KIND)
        events.write_event(
            store,
            kind=kind,
            severity="warning",
            domain="lilli.ops.hebbian.monitor_similarity_window",
            data={
                "window_size": len(window),
                "stddev": float(stddev),
                "threshold": float(threshold),
                "mean": float(statistics.fmean(window)),
                "interpretation": (
                    "Sliding window of structural_similarity outputs has collapsed "
                    "stddev -- all hebbian comparisons are returning near-identical "
                    "scores. The HV space is likely rank-deficient for this content."
                ),
            },
        )
    except Exception:
        log.warning("rank_deficiency telemetry emit failed (non-fatal)", exc_info=True)
