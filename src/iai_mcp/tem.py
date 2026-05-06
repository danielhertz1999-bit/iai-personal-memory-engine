"""Plan 03-01 CONN-05: TEM factorization (Whittington-Behrens 2020 Cell 183:1249-1263).

Tolman-Eichenbaum Machine factorization of *structure* and *content* into
binary BSC hypervectors at D=10000 (TorchHD semantics, packed to 1250 bytes).
Structure is bound with content via tensor product (binary XOR in BSC), and
multiple role-filler pairs are bundled via per-bit majority vote so a single
1250-byte hypervector carries 15-20 simultaneously-recoverable structural
attributes per record (D-TEM-02: unbind fidelity >= 0.95 at 15 pairs).

Constitutional fit:
- CONN-05 = TEM factorization. Structural queries are FIRST-CLASS peers of
  cosine queries in the retrieval pipeline. NOT a "VSA retrieval layer over
  cosine" -- structural and content signals merge in the ranker as siblings.
- D-TEM-01: BSC binary (NOT FHRR), D=10000.
- D-TEM-02: 15-20 role-filler pairs target; >= 95% unbind fidelity.
- D-TEM-03: tensor-product binding (XOR self-inverse in the binary case).
- D-TEM-04: Hebbian LTP on structure edges mirrors content-edge behavior
  (autopoiesis applied to structure).
- bind/unbind is lossless wrt the codebook -- decode by nearest-neighbor
  Hamming-distance against known fillers.

Implementation note (vs TorchHD direct usage): we operate on packed `bytes`
because (a) LanceDB's pa.binary() column type is the storage contract; (b)
1250 bytes per record is much cheaper than torch tensor materialisation on
every read; (c) bytewise XOR + np.unpackbits-based majority is faster than
the torch round-trip at our N. TorchHD BSC semantics are preserved bit-for-bit.

Public API:
- ROLE_VOCABULARY: 18 fixed role symbols (D-TEM Claude's Discretion).
- role_hv(role): deterministic D=10000 binary codebook vector for a role symbol.
- filler_hv(value): deterministic hash-to-D=10000 of a string filler.
- bind(a, b) / unbind(bound, key): bytewise XOR (BSC binding self-inverse).
- pack_pairs(pairs): per-bit majority bundle of bound role-filler pairs.
- unpack_role(hv, role): unbind by role key; caller compares to filler codebook.
- bind_structure(record): derive role-filler pairs from MemoryRecord fields,
  return packed hypervector (1250 bytes).
- decay_structure_edge(stability, difficulty, dt_days): FSRS decay identical
  to the content-edge formula (sleep.py: weight *= 0.9 ** (days - 90)).
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

from iai_mcp.types import STRUCTURE_HV_BYTES, STRUCTURE_HV_DIM

if TYPE_CHECKING:
    from iai_mcp.types import MemoryRecord


# D-TEM Claude's Discretion: 18 fixed role symbols. ORDER IS PART
# OF THE CONTRACT -- changing it breaks bind_structure's deterministic codebook.
ROLE_VOCABULARY: tuple[str, ...] = (
    "WHEN",
    "WHERE",
    "ROLE",
    "PROJECT",
    "COMMUNITY_ID",
    "TEMPORAL_POSITION",
    "ACTOR",
    "OBJECT",
    "INTENT",
    "MODALITY",
    "LANG",
    "SESSION_ID",
    "TIER",
    "VALENCE",
    "CERTAINTY",
    "SOURCE",
    "TOPIC",
    "PARENT_ID",
)


# ---------------------------------------------------------------- primitives


def _seed_from_str(prefix: str, value: str) -> int:
    """Stable per-string 64-bit seed (sha256 prefix). hash() is randomised
    per-process by default, so we use a deterministic digest instead."""
    digest = hashlib.sha256(f"{prefix}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _hv_from_seed(seed: int) -> bytes:
    """Generate a D=10000 binary hypervector packed to STRUCTURE_HV_BYTES."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=STRUCTURE_HV_DIM, dtype=np.uint8)
    return np.packbits(bits).tobytes()


# Precompute the 18-role codebook at import time. Same role -> same bytes
# across processes thanks to the deterministic sha256-prefixed seed.
_ROLE_HV_TABLE: dict[str, bytes] = {
    role: _hv_from_seed(_seed_from_str("tem-role-v1", role))
    for role in ROLE_VOCABULARY
}


def role_hv(role: str) -> bytes:
    """Deterministic D=10000 binary codebook vector for a role symbol.

    Uses the precomputed _ROLE_HV_TABLE for the 18 known roles; falls back
    to a fresh deterministic generation for any other role (still seeded
    on the role string, so callers can extend the vocabulary at their own
    risk -- ROLE_VOCABULARY is the canonical contract).
    """
    cached = _ROLE_HV_TABLE.get(role)
    if cached is not None:
        return cached
    return _hv_from_seed(_seed_from_str("tem-role-v1", role))


def filler_hv(value: str) -> bytes:
    """Deterministic hash-to-D=10000 of a string filler."""
    return _hv_from_seed(_seed_from_str("tem-filler-v1", value))


def bind(a: bytes, b: bytes) -> bytes:
    """BSC tensor-product binding: bytewise XOR. Self-inverse semantics."""
    if len(a) != len(b):
        raise ValueError(
            f"bind requires equal-length hypervectors, got {len(a)} and {len(b)}"
        )
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    return np.bitwise_xor(aa, bb).tobytes()


def unbind(bound: bytes, key: bytes) -> bytes:
    """XOR inverse of bind. Identical to bind() because XOR is self-inverse."""
    return bind(bound, key)


def pack_pairs(pairs: list[tuple[str, bytes]]) -> bytes:
    """Bundle bound role-filler pairs via per-bit majority vote.

    Deterministic tiebreak: bit=1 on even ties (`sums * 2 >= n`). This means
    a single (role, filler) pair recovers the filler exactly under unbind.
    """
    if not pairs:
        return bytes(STRUCTURE_HV_BYTES)  # empty bundle is the zero hv
    bound = []
    for role, filler in pairs:
        bound.append(np.frombuffer(bind(role_hv(role), filler), dtype=np.uint8))
    # Stack as (N, 1250) uint8, unpack to (N, 10000) bits, vote per column.
    stacked_bytes = np.stack(bound)  # shape (N, 1250)
    bits = np.unpackbits(stacked_bytes, axis=1).astype(np.int32)  # (N, 10000)
    sums = bits.sum(axis=0)
    n = len(pairs)
    # majority: bit=1 when more than half of inputs are 1; ties -> 1 (`>=`).
    voted = (sums * 2 >= n).astype(np.uint8)
    return np.packbits(voted).tobytes()


def unpack_role(hv: bytes, role: str) -> bytes:
    """Unbind hv by role's hypervector. Returns a noisy filler hv; caller
    nearest-neighbour decodes against a known filler codebook."""
    return unbind(hv, role_hv(role))


# ---------------------------------------------------------------- structure


def _bucket_datetime(dt) -> str:
    """Coarse temporal bucket for the WHEN role-filler -- ISO YYYY-MM-DD."""
    try:
        return dt.date().isoformat()
    except Exception:
        return "unknown"


def bind_structure(record: "MemoryRecord") -> bytes:
    """Derive 15+ role-filler pairs from a MemoryRecord and pack to bytes.

    Deterministic per (record fields, structural identity). NOT a hash of the
    full record content -- only the structural attributes (tier, language,
    community, temporal bucket, schema_version, pinned, detail_level, leading
    tags, parent provenance). literal_surface is intentionally excluded (it
    is content, not structure -- D-TEM-03 keeps the two factorised).
    """
    pairs: list[tuple[str, bytes]] = []

    # Constitutional 6 (D-TEM):
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
        except Exception:
            sid = "no-session"
    pairs.append(("SESSION_ID", filler_hv(sid)))
    pairs.append(("PARENT_ID", filler_hv("root")))

    return pack_pairs(pairs)


# ---------------------------------------------------------------- decay


# FSRS decay on structure edges is IDENTICAL to record-edge decay.
# Mirror sleep.py's _decay_edges constants verbatim instead of importing them
# (cyclic-import safe; values are part of the constitutional contract).
_DECAY_GRACE_DAYS: int = 90
_DECAY_BASE: float = 0.9


def decay_structure_edge(stability: float, difficulty: float, dt_days: float) -> float:
    """FSRS decay multiplier for structure edges. Identical to content-edge
    formula (sleep.py:21-26 + _decay_edges body): no decay during grace
    window, then `weight *= 0.9 ** (days - 90)`. Returns the multiplier
    (1.0 = no decay; (0..1) decayed; <eps prune at caller).
    """
    age_days = max(0.0, float(dt_days))
    if age_days <= _DECAY_GRACE_DAYS:
        return 1.0
    return _DECAY_BASE ** (age_days - _DECAY_GRACE_DAYS)
