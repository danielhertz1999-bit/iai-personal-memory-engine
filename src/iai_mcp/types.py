"""Core types for IAI-MCP.

Source-of-truth schema for MemoryRecord (canonical for IAI-MCP storage
drawer + PROJECT.md constitutional rules).

storage was English raw verbatim. amended
the schema to native-language storage. (2026-04-19)
reverted the brain to **English-Only**: the surface (Claude) translates
inbound text to English on the way in, and the records table stores the
English form. The schema retains the `language` ISO-639-1 column as a
historical marker on legacy rows; new records are tagged `"en"`.

schema additions (backward-compatible for migration):
- language: str (ISO-639-1, required)
- s5_trust_score: float [0,1] (default 0.5 neutral prior) -- prep
- profile_modulation_gain: dict[str, float] (default {})  -- runtime gain
- schema_version: int (1 legacy | 2 phase-2)              -- migration marker
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


# (2026-04-20): revert the Phase-2 deviation back to
# PROJECT.md line 125's original spec — bge-small-en-v1.5 (384d English-only).
# User directive 2026-04-19: brain stores English, surface translation is
# Claude's job. bge-m3 (1024d multilingual) remains selectable via the
# IAI_MCP_EMBED_MODEL env var or Embedder(model_key="bge-m3") kwarg;
# existing 1024d user stores stay readable via embedder_for_store(store)
# (commit 808e877). No forced migration of existing data.
DEFAULT_EMBED_DIM = 384        # bge-small-en-v1.5 native dimension (PROJECT.md line 125)
EMBED_DIM = DEFAULT_EMBED_DIM  # legacy alias for callers

# module-level constants (constitutional anchors)
SCHEMA_VERSION_LEGACY = 1      # pre-Phase-2 records before migration
SCHEMA_VERSION_V2 = 2          # schema (language + s5_trust + profile gain)
SCHEMA_VERSION_V3 = 3          # encryption-at-rest data upgrade
SCHEMA_VERSION_V4 = 4 # TEM factorization (structure_hv: bytes)
SCHEMA_VERSION_CURRENT = SCHEMA_VERSION_V4  # newest version: written to every new record; migration bumps older rows
SCHEMA_VERSION_ACCEPTED = frozenset({
    SCHEMA_VERSION_LEGACY,
    SCHEMA_VERSION_V2,
    SCHEMA_VERSION_V3,
    SCHEMA_VERSION_V4,
})

# TEM factorization (Whittington-Behrens 2020 Cell 183:1249-1263).
# Binary BSC hypervector at D=10000 bits, packed 8 bits/byte = 1250 bytes.
# `structure_hv` on MemoryRecord is a SEPARATE first-class field alongside `embedding`
# (NOT a "VSA retrieval layer over cosine"). Empty bytes = pre-migration sentinel.
STRUCTURE_HV_DIM: int = 10000
STRUCTURE_HV_BYTES: int = STRUCTURE_HV_DIM // 8  # 1250 bytes packed

# exactly five tiers per PROJECT.md Memory Core.
# adds a sixth: semantic_pruned, used by
# cleanup_schema_duplicates as a soft-delete sentinel for duplicate
# schema records (Beer VSM S2 anti-oscillation reversibility — pruned
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
    """Canonical memory record.

    Constitutional invariants:
    - `literal_surface` is always raw verbatim. Per the canonical
      form is English (Claude translates inbound surface text); legacy v2
      records may carry a non-English `language` tag and are read as-is.
    - Records with `detail_level >= 3` never decay (, ).
    - Records with `never_merge=True` are skipped by ART gate ( L0 guarantee).
    - `language` is a required ISO-639-1 tag; empty string is rejected.
    - `s5_trust_score` in [0, 1] (default 0.5 neutral prior, S5 identity kernel prep).
    - `schema_version` must be 1 (legacy) | 2 | 3 (-08 encryption) | 4 (-01 TEM).
    - `structure_hv` is empty bytes (pre-migration) OR exactly
      STRUCTURE_HV_BYTES (1250) bytes (TorchHD BSC binary at D=10000).
    """

    # identity
    id: UUID                              # stable UUID4 at creation
    tier: str                             # "working" | "episodic" | "semantic" | "procedural" | "parametric" | "semantic_pruned"

    # content (constitutional: raw verbatim in the user's language)
    literal_surface: str                  # raw verbatim; language tag below
    aaak_index: str # AAAK metadata line (populates; default "")

    # retrieval features
    embedding: list[float]                # DIM from configured embedder (D-02a registry)

    # graph + salience
    community_id: UUID | None # assigned by community detection; None in
    centrality: float # computed by graph analysis; 0.0 default
    detail_level: int                     # 1..5; 5 = never summarize (D-08 constitutional)
    pinned: bool                          # user-pinned records (includes L0 identity)

    # FSRS schema fields ( fields only; decay scheduler is )
    stability: float                      # default 0.0
    difficulty: float                     # default 0.0
    last_reviewed: datetime | None        # default None
    never_decay: bool # auto-True when detail_level >= 3
    never_merge: bool                     # True for pinned L0

    # provenance ( edge-based reconsolidation in )
    provenance: list[dict[str, Any]]      # each entry: {"ts", "cue", "session_id"}

    # bookkeeping
    created_at: datetime
    updated_at: datetime

    # REQUIRED language field (keyword-only, no default) -- constitutional.
    # Placed here (before default-valued fields) so dataclass init enforces it
    # as a required kwarg for every caller.
    language: str                         # ISO-639-1 tag (e.g. "en", "ru", "ja", "ar")

    # fields with defaults -- order must stay after required fields
    tags: list[str] = field(default_factory=list)
    s5_trust_score: float = 0.5           # prep; neutral prior
    profile_modulation_gain: dict[str, float] = field(default_factory=dict)  # D-11
    schema_version: int = SCHEMA_VERSION_CURRENT
    # TEM factorization (Whittington-Behrens 2020 Cell 183:1249-1263).
    # Binary BSC hypervector at D=10000 bits, packed to STRUCTURE_HV_BYTES (1250 bytes).
    # Empty bytes default = pre-migration / lazy-bind sentinel; tem.bind_structure
    # is called at insert time to fill it. SEPARATE first-class field alongside
    # `embedding` -- structural queries are peers of cosine, not a rerank layer.
    structure_hv: bytes = field(default=b"")

    def __post_init__(self) -> None:
        # rule from + PROJECT.md ("OFF for detail_level >= 3"):
        # high-detail records never decay, regardless of what caller passed.
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
                "language is a required non-empty ISO-639-1 string field "
                "(constitutional violation)"
            )
        # prep: s5_trust_score in [0, 1].
        if not (0.0 <= self.s5_trust_score <= 1.0):
            raise ValueError(
                f"s5_trust_score must be in [0, 1], got {self.s5_trust_score}"
            )
        # Migration marker: v1 (legacy) | v2 | v3 (encryption) | v4 (TEM).
        if self.schema_version not in SCHEMA_VERSION_ACCEPTED:
            raise ValueError(
                f"schema_version must be one of {sorted(SCHEMA_VERSION_ACCEPTED)}, "
                f"got {self.schema_version}"
            )
        # : structure_hv must be empty (pre-migration sentinel)
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


@dataclass
class MemoryHit:
    """Single retrieval result.

    `valid_from` and `valid_to` are DERIVED at recall time from the
    contradiction-edge graph, never stored on the underlying MemoryRecord
    (episodic write-once invariant preserved). None defaults preserve
    back-compat for callers (tests, bench harness, recall_for_benchmark)
    that don't run derivation.

    Semantic:
      valid_from = record.created_at when derivation runs; None on
                   back-compat paths.
      valid_to   = oldest newer-contradicter's created_at; None if no
                   newer record points at this one via a contradicts edge.
    """

    record_id: UUID
    score: float # cosine + weighted bonuses (fills full formula)
    reason: str                           # human-readable "cosine 0.87 + rich-club 0.05"
    literal_surface: str                  # verbatim content (returns content, not only id)
    adjacent_suggestions: list[UUID] # cued-recognition (populates)
    # Derived temporal validity. Set by retrieve.derive_temporal_validity()
    # at recall time; None on paths that don't enrich (recall_for_benchmark,
    # any caller constructing MemoryHit directly without enrichment).
    valid_from: datetime | None = None
    valid_to: datetime | None = None


@dataclass
class RecallResponse:
    """Full response from memory_recall (, ).

    `hints` carries per-recall S4 contradiction notices +
    S5 cooldown + provisional schema candidates. Each hint dict shape:
        {"kind": "s4_contradiction" | "s5_cooldown" | "provisional_schema",
         "severity": "info" | "warning",
         "source_ids": [str(UUID), ...],
         "text": str,
         ...optional kind-specific fields}

    adds two new fields with backward-compatible defaults:
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

    Constitutional framing for the new fields:
        - McClelland CLS: episodic and semantic stores are distinguishable;
          their retrieval surfaces should be too — patterns_observed[] gives
          the schema layer its own surface instead of mixing it into hits[].
        - Beer VSM S1 vs S4: operations (verbatim) live at S1; intelligence
          (schema) at S4. patterns_observed[] makes S4 visible WITHOUT
          collapsing it into S1.
    """

    hits: list[MemoryHit]                 # excitatory
    anti_hits: list[MemoryHit]            # inhibitory -- cosine match with opposing AAAK or contradicts edge
    activation_trace: list[UUID] # node ids touched by 2-hop spread (fills)
    budget_used: int                      # tokens used by this response
    hints: list[dict] = field(default_factory=list)  # S4/S5/schema hints
    # cue-router output + concept-mode schema-split surface.
    # Defaults preserve back-compat: callers that don't classify their cue
    # see cue_mode='concept' (matches today's mode-less behaviour) and
    # patterns_observed=[] (no displaced schemas).
    cue_mode: str = "concept"
    patterns_observed: list[dict] = field(default_factory=list)


@dataclass
class EdgeUpdate:
    """Result of memory_reinforce (MCP-02, )."""

    edges_boosted: int
    pairs: list[tuple[UUID, UUID]]
    # string keys for JSON serialisation ("uuid_a|uuid_b" -> weight)
    new_weights: dict[str, float]


@dataclass
class ReconsolidationReceipt:
    """Result of memory_contradict (MCP-03, edge-based in )."""

    original_id: UUID
    new_record_id: UUID
    edge_type: str                        # "contradicts"
    ts: datetime
