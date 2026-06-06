"""Self-documenting per-tier metadata.

tier_info(tier_name) returns the canonical dict for each of the three tiers --
backend identifier, dimension D, bytes per stored hypervector, and use_case
description.
"""
from __future__ import annotations

from iai_mcp.lilli.tiers import bsc, fhrr, sparse_vsa

_TIER_REGISTRY: dict[str, dict] = {
    "bsc": bsc.TIER_INFO,
    "fhrr": fhrr.TIER_INFO,
    "sparse_vsa": sparse_vsa.TIER_INFO,
}


def tier_info(tier_name: str) -> dict:
    """Return a copy of the canonical metadata dict for the named tier.

    Args:
        tier_name: One of 'bsc', 'fhrr', 'sparse_vsa'.

    Returns:
        A copy of the tier's TIER_INFO dict with at least keys:
        backend, D, bytes_per_hv, use_case.

    Raises:
        ValueError: if tier_name is not recognised.
    """
    if tier_name not in _TIER_REGISTRY:
        raise ValueError(
            f"Unknown tier {tier_name!r}; expected one of {sorted(_TIER_REGISTRY)}"
        )
    return dict(_TIER_REGISTRY[tier_name])  # return a copy to prevent mutation


def list_tiers() -> list[str]:
    """Return sorted list of known tier names."""
    return sorted(_TIER_REGISTRY)
