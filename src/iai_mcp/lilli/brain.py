"""Brain -- top-level cognitive entry point.

Bundles all three tier backends, all eight ops, the cross-modal bridge, and
the profile registry. cognitive_mode is a fixed default of 'autistic' --
set unconditionally in __init__, not configurable.

Brain.recall(cue, *, limit=5, session_id='brain-recall') composes
iai_mcp.embed.embedder_for_store + iai_mcp.retrieve.recall -- the canonical
lilli-side entry-point for end-to-end seeded baseline recall.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from iai_mcp.lilli.tiers import bsc, fhrr, sparse_vsa
from iai_mcp.lilli.ops import (
    cleanup,
    consolidation,
    continual,
    decay,
    delta,
    orthogonalize,
    replay,
    separation,
)
from iai_mcp.lilli.crossmodal import embed_to_hv

log = logging.getLogger(__name__)


class Brain:
    """Top-level cognitive substrate. cognitive_mode is fixed at 'autistic'."""

    def __init__(self, hippo_conn: Any = None) -> None:
        # Fixed invariant -- NOT a configurable flag.
        self.cognitive_mode: str = "autistic"

        # Tier backends.
        self.bsc = bsc
        self.fhrr = fhrr
        self.sparse_vsa = sparse_vsa

        # Ops bundle.
        self.ops = SimpleNamespace(
            continual=continual,
            consolidation=consolidation,
            decay=decay,
            replay=replay,
            orthogonalize=orthogonalize,
            cleanup=cleanup,
            delta=delta,
            separation=separation,
        )

        # Cross-modal bundle.
        # to_embedding_neighbors lives in embed_to_hv (same module as from_embedding).
        self.crossmodal = SimpleNamespace(
            embed_to_hv=embed_to_hv,
            hv_to_neighbors=embed_to_hv.to_embedding_neighbors,
        )

        self.profile = SimpleNamespace()

        # Hippo connection (None acceptable for non-storage uses).
        self.hippo_conn = hippo_conn

        # Sleep-cycle dispatch bundle (REM / SWS / consolidation).
        # Imported locally to avoid loading cycle subpackage at module import time.
        from iai_mcp.lilli.cycle.orchestrator import (
            run_consolidation,
            run_rem,
            run_sws,
        )
        self.cycle = SimpleNamespace(
            run_rem=run_rem,
            run_sws=run_sws,
            run_consolidation=run_consolidation,
        )

    def recall(self, cue: str, *, limit: int = 5, session_id: str = "brain-recall") -> list:
        """Recall MemoryRecords matching the cue via the canonical retrieve.recall path.

        Thin wrapper: embeds the cue via embedder_for_store(self.hippo_conn), then
        calls iai_mcp.retrieve.recall(store, cue_embedding, cue_text, session_id, k_hits=limit).
        Returns the.hits attribute of the resulting RecallResponse (list[MemoryHit]).

        Args:
            cue: Free-text query.
            limit: Maximum hits to return (default 5; mapped to retrieve.recall's k_hits).
            session_id: Provenance session tag (default "brain-recall").

        Returns:
            response.hits -- list[MemoryHit] from iai_mcp.retrieve.recall.
            Empty list iff the store legitimately found nothing.

        Raises:
            RuntimeError: if self.hippo_conn is None -- Brain.recall is meaningless
                without a bound store.
        """
        if self.hippo_conn is None:
            raise RuntimeError(
                "Brain.recall requires hippo_conn (MemoryStore-like instance)"
            )
        from iai_mcp.embed import embedder_for_store
        from iai_mcp.retrieve import recall as _retrieve_recall

        embedder = embedder_for_store(self.hippo_conn)
        # Embedder.embed(text: str) -> list[float] -- see iai_mcp.embed.
        # Returns a normalised DIM-length list[float] directly.
        # Use embed(), not the native encoder's lower-level encode() method --
        # embed() is the iai-mcp Embedder wrapper's public interface.
        cue_embedding = embedder.embed(cue)
        response = _retrieve_recall(
            self.hippo_conn,
            cue_embedding=cue_embedding,
            cue_text=cue,
            session_id=session_id,
            k_hits=limit,
        )
        return response.hits

    def emit_telemetry(self, kind: str, data: dict) -> None:
        """Emit a telemetry event through write_event. No-op if hippo_conn is None."""
        if self.hippo_conn is None:
            return
        try:
            from iai_mcp.events import write_event
            write_event(self.hippo_conn, kind=kind, data=data)
        except Exception as exc:  # noqa: BLE001 -- telemetry must never crash callers
            log.warning("emit_telemetry failed for kind=%s: %s", kind, exc)

    def emit_rank_deficiency_warning(self, data: dict) -> None:
        """Emit a rank-deficiency telemetry event."""
        from iai_mcp.events import TELEMETRY_RANK_DEFICIENCY
        self.emit_telemetry(TELEMETRY_RANK_DEFICIENCY, data)

    def emit_role_saturation_warning(self, data: dict) -> None:
        """Emit a role-saturation telemetry event."""
        from iai_mcp.events import TELEMETRY_ROLE_SATURATION
        self.emit_telemetry(TELEMETRY_ROLE_SATURATION, data)

    def emit_codec_marker_missing(self, data: dict) -> None:
        """Emit a codec-marker-missing telemetry event."""
        from iai_mcp.events import TELEMETRY_CODEC_MARKER_MISSING
        self.emit_telemetry(TELEMETRY_CODEC_MARKER_MISSING, data)
