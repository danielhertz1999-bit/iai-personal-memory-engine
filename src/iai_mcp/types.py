"""Core types for IAI-MCP.

Source-of-truth schema for MemoryRecord.

The brain is **English-only**: the surface (Claude) translates
inbound text to English on the way in, and the records table stores the
English form. The schema retains the `language` ISO-639-1 column as a
historical marker on legacy rows; new records are tagged `"en"`.

Schema additions (backward-compatible for migration):
- language: str (ISO-639-1, required)
- s5_trust_score: float [0,1] (default 0.5 neutral prior)
- profile_modulation_gain: dict[str, float] (default {}) -- runtime gain
- schema_version: int (1 legacy | 2 current) -- migration marker

Codec metadata (V5 schema, additive):
- hv_tier: str (default "bsc") -- codec tier tag; one of HV_TIER_ENUM
- structure_hv_payload: bytes (default b"") -- variable-length bytes for
  non-BSC tier payloads; empty for BSC tier (which uses structure_hv)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


# English-only brain: the sole embed model is bge-small-en-v1.5 (384d),
# running via the Rust native extension (iai_mcp_native.embed). Users speak
# any language; Claude translates on the way in. No model-selection knob.
# Legacy 1024d stores remain readable via embedder_for_store(store);
# no forced migration of existing data.
DEFAULT_EMBED_DIM = 384        # bge-small-en-v1.5 native dimension
EMBED_DIM = DEFAULT_EMBED_DIM  # legacy alias for callers

# module-level schema-version constants
SCHEMA_VERSION_LEGACY = 1      # records predating migration
SCHEMA_VERSION_V2 = 2          # schema (language + s5_trust + profile gain)
SCHEMA_VERSION_V3 = 3          # encryption-at-rest data upgrade
SCHEMA_VERSION_V4 = 4          # TEM factorization (structure_hv: bytes)
SCHEMA_VERSION_V5 = 5          # Lilli HD/HDC codec metadata boundary (hv_tier + structure_hv_payload)
SCHEMA_VERSION_CURRENT = SCHEMA_VERSION_V5  # newest version: written to every new record; migration bumps older rows
SCHEMA_VERSION_ACCEPTED = frozenset({
    SCHEMA_VERSION_LEGACY,
    SCHEMA_VERSION_V2,
    SCHEMA_VERSION_V3,
    SCHEMA_VERSION_V4,
    SCHEMA_VERSION_V5,
})

# Structure/content factorization.
# Binary BSC hypervector at D=10000 bits, packed 8 bits/byte = 1250 bytes.
# `structure_hv` on MemoryRecord is a SEPARATE first-class field alongside `embedding`
# (NOT a "VSA retrieval layer over cosine"). Empty bytes = pre-migration sentinel.
STRUCTURE_HV_DIM: int = 10000
# STRUCTURE_HV_BYTES = 1250 is the BSC-tier byte count at D=10000. Other
# tiers use different byte counts — see lilli.tier_info(tier) for the
# canonical per-tier mapping. This constant remains as the back-compat
# anchor for the legacy `structure_hv` column.
STRUCTURE_HV_BYTES: int = STRUCTURE_HV_DIM // 8  # 1250 bytes packed (BSC tier)

# Codec tier enum for the Lilli HD/HDC layer.
# "bsc" — Binary Sparse Coding (existing D=10000 structure_hv field)
# "fhrr" — Fourier Holographic Reduced Representation (complex-valued)
# "sparse_vsa" — Sparse VSA (positional superposition encoding)
HV_TIER_ENUM: frozenset[str] = frozenset({"bsc", "fhrr", "sparse_vsa"})

# A sixth tier, semantic_pruned, is used by
# cleanup_schema_duplicates as a soft-delete sentinel for duplicate
# schema records (S2 anti-oscillation reversibility — pruned
# rows stay in the store and can be lifted back to "semantic" via a
# reverse migration; physical deletion is forbidden).
SEMANTIC_PRUNED_TIER: str = "semantic_pruned"
TIER_ENUM = frozenset({
    "working",
    "episodic",
    "semantic",
    "procedural",
    "parametric",
    SEMANTIC_PRUNED_TIER,
})


@dataclass
class MemoryRecord:
    """Canonical native-language memory record.

    Invariants:
    - `literal_surface` is always raw verbatim. The canonical
      form is English (Claude translates inbound surface text); legacy v2
      records may carry a non-English `language` tag and are read as-is.
    - Records with `detail_level >= 3` never decay.
    - Records with `never_merge=True` are skipped by ART gate (L0 guarantee).
    - `language` is a required ISO-639-1 tag; empty string is rejected.
    - `s5_trust_score` in [0, 1] (default 0.5 neutral prior, S5 identity kernel prep).
    - `schema_version` must be one of SCHEMA_VERSION_ACCEPTED (1–5).
    - `structure_hv` is empty bytes (pre-migration) OR exactly
      STRUCTURE_HV_BYTES (1250) bytes (TorchHD BSC binary at D=10000).
    - `hv_tier` must be one of HV_TIER_ENUM ("bsc", "fhrr", "sparse_vsa").
    - `structure_hv_payload` must be bytes (any length; empty for BSC tier).
    """

    # identity
    id: UUID                              # stable UUID4 at creation
    tier: str                             # "working" | "episodic" | "semantic" | "procedural" | "parametric" | "semantic_pruned"

    # content (invariant: raw verbatim in the user's language)
    literal_surface: str                  # raw verbatim; language tag below
    aaak_index: str                       # AAAK metadata line; default ""

    # retrieval features
    embedding: list[float]                # DIM from configured embedder (registry)

    # graph + salience
    community_id: UUID | None             # community assignment; None before first detection
    centrality: float                     # betweenness centrality; 0.0 default
    detail_level: int                     # 1..5; 5 = never summarize
    pinned: bool                          # user-pinned records (includes L0 identity)

    # FSRS schema fields (fields only; the decay scheduler lives elsewhere)
    stability: float                      # default 0.0
    difficulty: float                     # default 0.0
    last_reviewed: datetime | None        # default None
    never_decay: bool                     # auto-True when detail_level >= 3
    never_merge: bool                     # True for pinned L0

    # provenance (edge-based reconsolidation)
    provenance: list[dict[str, Any]]      # each entry: {"ts", "cue", "session_id"}

    # bookkeeping
    created_at: datetime
    updated_at: datetime

    # REQUIRED language field (keyword-only, no default).
    # Placed here (before default-valued fields) so dataclass init enforces it
    # as a required kwarg for every caller.
    language: str                         # ISO-639-1 tag (e.g. "en", "ru", "ja", "ar")

    # fields with defaults -- order must stay after required fields
    tags: list[str] = field(default_factory=list)
    s5_trust_score: float = 0.5           # neutral prior
    profile_modulation_gain: dict[str, float] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION_CURRENT
    # Structure/content factorization.
    # Binary BSC hypervector at D=10000 bits, packed to STRUCTURE_HV_BYTES (1250 bytes).
    # Empty bytes default = pre-migration / lazy-bind sentinel; tem.bind_structure
    # is called at insert time to fill it. SEPARATE first-class field alongside
    # `embedding` -- structural queries are peers of cosine, not a rerank layer.
    structure_hv: bytes = field(default=b"")
    # Lilli HD/HDC codec metadata boundary (V5 schema).
    # hv_tier tags which codec tier produced the hypervector above.
    # Defaults to "bsc" (BSC-tier, matches legacy structure_hv field).
    # Other tiers ("fhrr", "sparse_vsa") store their payload in
    # structure_hv_payload (variable-length); structure_hv stays empty.
    hv_tier: str = "bsc"
    structure_hv_payload: bytes = field(default=b"")
    # Deferred-embedding flag. 0 = embedding is valid; 1 = row was written
    # with a zero-vector placeholder pending daemon re-embed. Populated by
    # _from_row; omitted from _to_row (hippo INSERT paths set this directly).
    embedding_pending: int = 0

    def __post_init__(self) -> None:
        # Decay rule: high-detail records never decay, regardless of what
        # the caller passed (OFF for detail_level >= 3).
        if self.detail_level >= 3:
            self.never_decay = True
        # Tier validation -- fail fast on garbage input
        if self.tier not in TIER_ENUM:
            raise ValueError(
                f"invalid tier {self.tier!r}; must be one of {sorted(TIER_ENUM)}"
            )
        # language required non-empty ISO-639-1 tag.
        if not self.language or not isinstance(self.language, str):
            raise ValueError(
                "language is a required non-empty ISO-639-1 string field"
            )
        # s5_trust_score in [0, 1].
        if not (0.0 <= self.s5_trust_score <= 1.0):
            raise ValueError(
                f"s5_trust_score must be in [0, 1], got {self.s5_trust_score}"
            )
        # Migration marker: must be one of the accepted schema versions.
        if self.schema_version not in SCHEMA_VERSION_ACCEPTED:
            raise ValueError(
                f"schema_version must be one of {sorted(SCHEMA_VERSION_ACCEPTED)}, "
                f"got {self.schema_version}"
            )
        # structure_hv must be empty (pre-migration sentinel)
        # OR exactly STRUCTURE_HV_BYTES (1250) bytes for D=10000 BSC packed bits.
        if not isinstance(self.structure_hv, (bytes, bytearray)):
            raise ValueError(
                f"structure_hv must be bytes, got {type(self.structure_hv).__name__}"
            )
        if self.structure_hv and len(self.structure_hv) != STRUCTURE_HV_BYTES:
            raise ValueError(
                f"structure_hv must be empty (pre-migration) or exactly "
                f"{STRUCTURE_HV_BYTES} bytes (D={STRUCTURE_HV_DIM} BSC packed), "
                f"got {len(self.structure_hv)} bytes"
            )
        # Lilli HD/HDC codec tier validation.
        if self.hv_tier not in HV_TIER_ENUM:
            raise ValueError(
                f"hv_tier must be one of {sorted(HV_TIER_ENUM)}, got {self.hv_tier!r}; "
                f"HV_TIER_ENUM = {sorted(HV_TIER_ENUM)}"
            )
        if not isinstance(self.structure_hv_payload, (bytes, bytearray)):
            raise ValueError(
                f"structure_hv_payload must be bytes (expected bytes), "
                f"got {type(self.structure_hv_payload).__name__}"
            )


@dataclass
class MemoryHit:
    """Single retrieval result.

    `valid_from` and `valid_to` are
    DERIVED at recall time from the contradiction-edge graph, never stored
    on the underlying MemoryRecord (episodic write-once invariant
    preserved). None defaults
    preserve back-compat for callers (tests, bench harness,
    recall_for_benchmark) that don't run derivation.

    Semantic:
      valid_from = record.created_at when derivation runs; None on
                   back-compat paths.
      valid_to = oldest newer-contradicter's created_at; None if no
                   newer record points at this one via a contradicts edge.
    """

    record_id: UUID
    score: float                          # cosine + weighted bonuses
    reason: str                           # human-readable "cosine 0.87 + rich-club 0.05"
    literal_surface: str                  # verbatim content (returns content, not only id)
    adjacent_suggestions: list[UUID]      # cued-recognition suggestions
    # Derived temporal validity. Set by retrieve.derive_temporal_validity()
    # at recall time; None on paths that don't enrich (recall_for_benchmark,
    # any caller constructing MemoryHit directly without enrichment).
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    # Originating capture attribution — populated from provenance[0] at each
    # MemoryHit construction site. None when provenance is empty (anti-hit
    # edges, synthetic hits, back-compat callers).
    session_id: str | None = None
    captured_at: str | None = None   # ISO-8601 UTC from record.created_at


@dataclass
class RecallResponse:
    """Full response from memory_recall.

    `hints` carries per-recall S4 contradiction notices +
    S5 cooldown + provisional schema candidates. Each hint dict shape:
        {"kind": "s4_contradiction" | "s5_cooldown" | "provisional_schema",
         "severity": "info" | "warning",
         "source_ids": [str(UUID),...],
         "text": str,
         ...optional kind-specific fields}

    Two further fields carry backward-compatible defaults:
        cue_mode: str
            "verbatim" or "concept" — set by core.dispatch from the cue-router
            classifier (cue_router._classify_cue). Default "concept" preserves
            today's behaviour for callers constructing RecallResponse directly
            without a classified mode (existing 1100+ tests stay green).
        patterns_observed: list[dict]
            In concept mode, schema records (tier=semantic AND tag pattern:*)
            that would have ranked in top-K are surfaced here instead of in
            hits[]. Each entry: {"pattern": str, "evidence_count": int,
            "schema_id": str(UUID)}. Max 3 entries. Empty in verbatim mode
            (schema records excluded from candidate set entirely) and when
            no schemas were displaced. Default [] is back-compat.

    Design intent for the new fields: the verbatim and schema retrieval
    surfaces are kept distinct — patterns_observed[] gives the schema
    layer its own surface instead of mixing it into hits[].
    """

    hits: list[MemoryHit]                 # excitatory
    anti_hits: list[MemoryHit]            # inhibitory -- cosine match with opposing AAAK or contradicts edge
    activation_trace: list[UUID]          # node ids touched by 2-hop spread
    budget_used: int                      # tokens used by this response
    hints: list[dict] = field(default_factory=list)  # S4/S5/schema hints
    #: cue-router output + concept-mode schema-split surface.
    # Defaults preserve back-compat: callers that don't classify their cue
    # see cue_mode='concept' (matches today's mode-less behaviour) and
    # patterns_observed=[] (no displaced schemas).
    cue_mode: str = "concept"
    patterns_observed: list[dict] = field(default_factory=list)
    # Deterministic, unsampled anti-masking marker. Set True by
    # recall_for_response on the ANN-first success path; left False on
    # the core.py soft-fallback (retrieve.recall). Not a telemetry sample —
    # every JSON-RPC response carries this field so the Layer-1 gate can
    # prove the bounded ANN path was taken in every measured trial.
    ann_path_used: bool = False


@dataclass
class EdgeUpdate:
    """Result of memory_reinforce."""

    edges_boosted: int
    pairs: list[tuple[UUID, UUID]]
    # string keys for JSON serialisation ("uuid_a|uuid_b" -> weight)
    new_weights: dict[str, float]


@dataclass
class ReconsolidationReceipt:
    """Result of memory_contradict (edge-based)."""

    original_id: UUID
    new_record_id: UUID
    edge_type: str                        # "contradicts"
    ts: datetime
