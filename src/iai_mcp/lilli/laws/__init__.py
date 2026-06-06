"""Embedded laws -- structural slot reserved for future robotics integration.

No runtime hooks active in this release. The four laws (L0-L3) are documented
in README.md alongside the activation plan. Until a robotics integration plugs
Lilli into a real action loop, LAWS_ACTIVE remains False.
"""
from __future__ import annotations

LAWS_ACTIVE: bool = False
