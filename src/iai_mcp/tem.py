"""TEM factorization — thin shim delegating to the BSC tier at D=10000.

Factorization of *structure* and *content* into binary BSC hypervectors.
Dense binary vectors packed 8 bits per byte, with bind = XOR (self-inverse),
bundle = per-bit majority vote.

This module re-exports the core BSC primitives from lilli.tiers.bsc at
D=10000 (1250 bytes per hypervector), preserving byte-for-byte fidelity with
the original implementation. Callers that use ROLE_VOCABULARY, role_hv,
filler_hv, bind, unbind, pack_pairs, or unpack_role see identical output.

Properties:
- BSC binary (NOT FHRR), D=10000.
- 15-20 role-filler pairs target; >= 95% unbind fidelity.
- Tensor-product binding (XOR self-inverse in the binary case).
- Hebbian LTP on structure edges mirrors content-edge behavior.
- bind/unbind is lossless wrt the codebook -- decode by nearest-neighbor
  Hamming-distance against known fillers.

Public API:
- ROLE_VOCABULARY: 18 fixed role symbols.
- role_hv(role): deterministic D=10000 binary codebook vector for a role symbol.
- filler_hv(value): deterministic hash-to-D=10000 of a string filler.
- bind(a, b) / unbind(bound, key): bytewise XOR (BSC binding self-inverse).
- pack_pairs(pairs): per-bit majority bundle of bound role-filler pairs.
- unpack_role(hv, role): unbind by role key; caller compares to filler codebook.
- bind_structure(record): derive role-filler pairs from MemoryRecord fields,
  return packed hypervector (1250 bytes).
- decay_structure_edge(stability, difficulty, dt_days): FSRS decay identical
  to the content-edge formula.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from iai_mcp.lilli.tiers.bsc import (
    BSC_ROLE_VOCABULARY as ROLE_VOCABULARY,
    bind,
    bundle as _bundle,
    filler_hv as _filler_hv,
    role_hv as _role_hv,
    unbind,
    unpack_role as _unpack_role,
)
from iai_mcp.types import STRUCTURE_HV_BYTES, STRUCTURE_HV_DIM

if TYPE_CHECKING:
    from iai_mcp.types import MemoryRecord

# D=10000 is the TEM contract — all public functions in this module operate
# at this dimension. STRUCTURE_HV_DIM must equal 10000 and STRUCTURE_HV_BYTES
# must equal 1250; this is enforced by types.py.
_D: int = STRUCTURE_HV_DIM


# ---------------------------------------------------------------- public API


def role_hv(role: str) -> bytes:
    """Deterministic D=10000 binary codebook vector for a role symbol.

    Uses the BSC tier codebook with D=10000. Any string is accepted; callers
    should use ROLE_VOCABULARY for the canonical 18 roles.
    """
    return _role_hv(role, D=_D)


def filler_hv(value: str) -> bytes:
    """Deterministic hash-to-D=10000 of a string filler."""
    return _filler_hv(value, D=_D)


def pack_pairs(pairs: list[tuple[str, bytes]]) -> bytes:
    """Bundle bound role-filler pairs via per-bit majority vote.

    Delegates to the BSC tier bundle() at D=10000. Empty pair list returns
    bytes(STRUCTURE_HV_BYTES) — the zero hypervector. Deterministic tiebreak:
    bit=1 on even ties.
    """
    return _bundle(pairs, D=_D)


def unpack_role(hv: bytes, role: str) -> bytes:
    """Unbind hv by role's hypervector. Returns a noisy filler hv; caller
    nearest-neighbour decodes against a known filler codebook."""
    return _unpack_role(hv, role, D=_D)


# ---------------------------------------------------------------- structure


def _bucket_datetime(dt) -> str:
    """Coarse temporal bucket for the WHEN role-filler -- ISO YYYY-MM-DD."""
    try:
        return dt.date().isoformat()
    except (AttributeError, TypeError, ValueError):
        return "unknown"


def bind_structure(record: "MemoryRecord") -> bytes:
    """Derive 15+ role-filler pairs from a MemoryRecord and pack to bytes.

    Deterministic per (record fields, structural identity). NOT a hash of the
    full record content -- only the structural attributes (tier, language,
    community, temporal bucket, schema_version, pinned, detail_level, leading
    tags, parent provenance). literal_surface is intentionally excluded (it
    is content, not structure -- structure and content are kept factorised).
    """
    pairs: list[tuple[str, bytes]] = []

    # Core six role-filler pairs:
    pairs.append(("WHEN", filler_hv(_bucket_datetime(record.created_at))))
    pairs.append(("WHERE", filler_hv(record.tier)))  # tier doubles as locale
    pairs.append(("ROLE", filler_hv(record.tier)))
    pairs.append(("PROJECT", filler_hv("iai-mcp")))
    pairs.append(("COMMUNITY_ID", filler_hv(str(record.community_id) if record.community_id else "none")))
    pairs.append(("TEMPORAL_POSITION", filler_hv(_bucket_datetime(record.created_at))))

    # Schema-side fillers (deterministic, queryable):
    pairs.append(("LANG", filler_hv(record.language or "en")))
    pairs.append(("TIER", filler_hv(record.tier)))
    pairs.append(("MODALITY", filler_hv("text")))
    pairs.append(("INTENT", filler_hv("episodic" if record.tier == "episodic" else "semantic")))
    pairs.append(("ACTOR", filler_hv("user")))
    pairs.append(("OBJECT", filler_hv(str(record.id))))
    pairs.append(("VALENCE", filler_hv("neutral")))
    pairs.append(("CERTAINTY", filler_hv(f"trust_{round(record.s5_trust_score, 1)}")))
    pairs.append(("SOURCE", filler_hv("pinned" if record.pinned else "drift")))

    # Content-adjacent fillers (still structural):
    leading_tag = (record.tags[0] if record.tags else "untagged")
    pairs.append(("TOPIC", filler_hv(str(leading_tag))))

    # Provenance hop -- session_id from latest provenance entry if any.
    sid = "no-session"
    if record.provenance:
        try:
            sid = str(record.provenance[-1].get("session_id") or "no-session")
        except (AttributeError, TypeError, KeyError, IndexError):
            sid = "no-session"
    pairs.append(("SESSION_ID", filler_hv(sid)))
    pairs.append(("PARENT_ID", filler_hv("root")))

    return pack_pairs(pairs)


# ---------------------------------------------------------------- decay


# Mirror the content-edge decay constants from sleep.py verbatim
# (cyclic-import safe; values are part of the decay contract).
_DECAY_GRACE_DAYS: int = 90
_DECAY_BASE: float = 0.9


def decay_structure_edge(stability: float, difficulty: float, dt_days: float) -> float:
    """FSRS decay multiplier for structure edges. Identical to content-edge
    formula: no decay during grace window, then
    ``weight *= 0.9 ** (days - 90)``. Returns the multiplier
    (1.0 = no decay; (0..1) decayed; <eps prune at caller).
    """
    age_days = max(0.0, float(dt_days))
    if age_days <= _DECAY_GRACE_DAYS:
        return 1.0
    return _DECAY_BASE ** (age_days - _DECAY_GRACE_DAYS)
