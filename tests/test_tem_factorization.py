""" RED: TEM factorization (Whittington-Behrens 2020).

Verifies BSC binding/unbinding fidelity at D=10000 across 15 / 17 / 18
role-filler pairs (D-TEM-02 target). Constitutional invariants:

- Tensor-product bind is XOR-reversible (BSC self-inverse semantics).
- Pack/unpack maintains >= 95% unbind accuracy at 15 pairs.
- structure_hv is exactly STRUCTURE_HV_BYTES (1250 bytes) packed bits.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------- module surface


def test_role_vocabulary_has_18_entries() -> None:
    """D-TEM Claude's Discretion locks role count at 18 (covers WHEN/WHERE/...
    plus tier/lang/community/etc. structural attributes per MemoryRecord).
    """
    from iai_mcp.tem import ROLE_VOCABULARY

    assert isinstance(ROLE_VOCABULARY, tuple)
    assert len(ROLE_VOCABULARY) == 18
    # Constitutional minimum subset from CONTEXT.md D-TEM:
    for required in ("WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID", "TEMPORAL_POSITION"):
        assert required in ROLE_VOCABULARY, f"missing constitutional role {required!r}"


def test_role_hv_is_deterministic_and_correct_length() -> None:
    """Same role symbol always returns same bytes; length is STRUCTURE_HV_BYTES."""
    from iai_mcp.tem import role_hv
    from iai_mcp.types import STRUCTURE_HV_BYTES

    a = role_hv("WHEN")
    b = role_hv("WHEN")
    assert isinstance(a, bytes)
    assert len(a) == STRUCTURE_HV_BYTES
    assert a == b  # Deterministic codebook.

    c = role_hv("WHERE")
    assert c != a  # Different roles produce different hvs.


def test_filler_hv_is_deterministic_and_correct_length() -> None:
    """Same filler string always returns same bytes; length is STRUCTURE_HV_BYTES."""
    from iai_mcp.tem import filler_hv
    from iai_mcp.types import STRUCTURE_HV_BYTES

    a = filler_hv("2026-04-17")
    b = filler_hv("2026-04-17")
    assert isinstance(a, bytes)
    assert len(a) == STRUCTURE_HV_BYTES
    assert a == b


def test_bind_is_xor_reversible() -> None:
    """BSC tensor-product binding is bytewise XOR; XOR is self-inverse."""
    from iai_mcp.tem import bind, role_hv

    a = role_hv("WHEN")
    b = role_hv("PROJECT")
    bound = bind(a, b)
    assert isinstance(bound, bytes)
    assert len(bound) == len(a)
    # XOR self-inverse: bind(bind(a, b), b) == a
    assert bind(bound, b) == a
    assert bind(bound, a) == b


def test_unbind_inverts_bind() -> None:
    """unbind(bind(a, b), a) recovers b bit-for-bit."""
    from iai_mcp.tem import bind, role_hv, unbind

    a = role_hv("ROLE")
    b = role_hv("LANG")
    bound = bind(a, b)
    recovered = unbind(bound, a)
    assert recovered == b


# -------------------------------------------------------- fidelity at N pairs


def _fidelity_at(n_pairs: int) -> float:
    """Pack n_pairs role-filler pairs, then test unbind recovery against
    a known filler codebook of size 18. Returns matched / n_pairs in [0, 1]."""
    from iai_mcp.tem import (
        ROLE_VOCABULARY,
        bind,
        filler_hv,
        pack_pairs,
        role_hv,
        unbind,
    )

    # Deterministic seed=42-derived filler set (one filler per role, 18 total).
    fillers = [filler_hv(f"filler-seed42-{i}") for i in range(len(ROLE_VOCABULARY))]
    roles = list(ROLE_VOCABULARY[:n_pairs])
    pairs = [(roles[i], fillers[i]) for i in range(n_pairs)]
    packed = pack_pairs(pairs)
    assert isinstance(packed, bytes)

    # Hamming-distance helper.
    def hamming(x: bytes, y: bytes) -> int:
        return sum(bin(a ^ b).count("1") for a, b in zip(x, y))

    correct = 0
    for i, role in enumerate(roles):
        unbound = unbind(packed, role_hv(role))
        # Nearest-neighbour against the known filler codebook (size 18).
        best = min(range(len(fillers)), key=lambda j: hamming(unbound, fillers[j]))
        if best == i:
            correct += 1
    return correct / n_pairs


def test_unbind_fidelity_15_pairs() -> None:
    """D-TEM-02: at 15 role-filler pairs, unbind fidelity >= 0.95."""
    fidelity = _fidelity_at(15)
    assert fidelity >= 0.95, f"unbind fidelity at 15 pairs = {fidelity:.3f} < 0.95"


def test_unbind_fidelity_17_pairs() -> None:
    """D-TEM-02 secondary target: at 17 pairs, fidelity >= 0.92."""
    fidelity = _fidelity_at(17)
    assert fidelity >= 0.92, f"unbind fidelity at 17 pairs = {fidelity:.3f} < 0.92"


def test_unbind_fidelity_18_pairs() -> None:
    """D-TEM-02 outer bound: at 18 pairs (whole vocab), fidelity >= 0.90."""
    fidelity = _fidelity_at(18)
    assert fidelity >= 0.90, f"unbind fidelity at 18 pairs = {fidelity:.3f} < 0.90"
