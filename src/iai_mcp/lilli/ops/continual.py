"""Continual-learning HV ops -- APPROXIMATE incremental utilities, not true majority-vote bundling.

add_pair(hv, role, filler) is an XOR-overlay: ``hv XOR bind(role_hv(role), filler_hv(filler))``.
NOT true incremental majority-vote bundling -- majority-vote requires retaining accumulator
state per role-filler pair, which add_pair does not store. XOR-overlay preserves single-pair
recovery (unbind on a freshly XOR'd hv recovers the latest filler exactly) but loses
vote-margin information once >2 pairs accumulate.

update_role(hv, old_filler, role, new_filler) composes two add_pair operations to swap a
role's binding -- same approximation caveat applies.

For TRUE majority-vote bundling with the complete pair list, call lilli.tiers.bsc.bundle(pairs)
directly. These continual primitives serve streaming-update use cases where the full pair list
is unavailable; they are NOT a drop-in replacement for the canonical bundle path.

Module docstring note: add_pair uses XOR-merge for incremental binding -- semantically distinct
from bsc.bundle()'s majority vote. Use bsc.bundle() when you have the complete pair list;
use add_pair() when streaming pairs one at a time and you only need approximate retrieval.
"""
from __future__ import annotations

from iai_mcp.lilli.tiers import bsc


def empty_hv(*, D: int = bsc.LILLI_BSC_DEFAULT_DIM) -> bytes:
    """Return a zero-filled hypervector of dimension D.

    The all-zeros hv is the identity element for XOR: ``add_pair(empty_hv(),...) == bound_pair``.
    Useful as a blank starting slate before streaming in role-filler pairs via add_pair.

    Args:
        D: Hypervector dimensionality. Default is LILLI_BSC_DEFAULT_DIM (4096).

    Returns:
        Packed bytes of length D // 8, all zeros.
    """
    return bytes(D // 8)


def add_pair(
    hv: bytes,
    role: str,
    filler_value: str,
    *,
    D: int = bsc.LILLI_BSC_DEFAULT_DIM,
) -> bytes:
    """Approximate XOR-overlay binding utility for streaming updates.

    Returns ``hv XOR bind(role_hv(role), filler_hv(filler_value))``.

    APPROXIMATE -- NOT true incremental majority-vote bundling: majority-vote
    requires retaining accumulator state per role-filler pair, which this function
    does not store. XOR-overlay preserves single-pair recovery (unbind on a freshly
    XOR'd hv recovers the latest filler exactly) but loses vote-margin information
    once >2 pairs accumulate. For true majority-vote bundling, call
    ``lilli.tiers.bsc.bundle(pairs)`` with the complete pair list.

    Args:
        hv: Existing hypervector to update (packed bytes, length D // 8).
        role: Role symbol (e.g. ``"WHEN"``). Any string is accepted; canonical
                      roles are listed in ``bsc.BSC_ROLE_VOCABULARY``.
        filler_value: Filler value string to encode.
        D: Hypervector dimensionality. Default is LILLI_BSC_DEFAULT_DIM.

    Returns:
        New packed bytes of length D // 8 with the role-filler pair overlaid.
    """
    bound_pair = bsc.bind(bsc.role_hv(role, D=D), bsc.filler_hv(filler_value, D=D))
    return bsc.bind(hv, bound_pair)


def update_role(
    hv: bytes,
    old_filler_value: str,
    role: str,
    new_filler_value: str,
    *,
    D: int = bsc.LILLI_BSC_DEFAULT_DIM,
) -> bytes:
    """Replace an existing role's filler in-place by unbinding the old then binding the new.

    Composed as two add_pair operations:
    1. Remove old binding: ``hv XOR bind(role_hv(role), filler_hv(old_filler_value))``
    2. Add new binding: ``result XOR bind(role_hv(role), filler_hv(new_filler_value))``

    Because BSC bind is XOR (self-inverse), step 1 cancels the old pair and step 2
    introduces the new one. The same APPROXIMATE caveat as add_pair applies: this
    operation is exact for the targeted role but multi-pair vote-margin information
    accumulated before the update is irreversibly affected.

    Args:
        hv: Current hypervector (packed bytes, length D // 8).
        old_filler_value: Value to remove from the hv for this role.
        role: Role symbol to update.
        new_filler_value: New value to bind under this role.
        D: Hypervector dimensionality. Default is LILLI_BSC_DEFAULT_DIM.

    Returns:
        New packed bytes with the role's filler swapped.
    """
    hv_without_old = add_pair(hv, role, old_filler_value, D=D)
    return add_pair(hv_without_old, role, new_filler_value, D=D)
