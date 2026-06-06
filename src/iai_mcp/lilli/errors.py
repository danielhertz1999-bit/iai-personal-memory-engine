"""Lilli-namespace exception classes. Single home for errors raised by the lilli package
— tiers, ops, persistence, crossmodal, etc. Subclassing pattern: every lilli exception is
a subclass of an appropriate stdlib base (ValueError, RuntimeError, etc.) so callers that
catch the stdlib type still work.
"""
from __future__ import annotations


class BundleCapacityError(ValueError):
    """Raised when a tier's bundle operation receives more pairs than its capacity allows.

    Capacity is tier-specific: BSC default is D // 400. Subclassing ValueError so callers
    that catch the broader ValueError type continue to handle this case without changes.
    """
