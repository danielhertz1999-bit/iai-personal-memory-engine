"""Guard test: the in-process embedder construct budget stays at 1000 ms.

Warm in-process construction completes in roughly 37 ms (27× headroom).
A true cold-disk first-ever model load takes several seconds and degrades
cleanly to the bypass-safe recency floor within the fail-fast ceiling.
The 1000 ms value separates the two cases cleanly; re-tuning it without
updating the surrounding analysis would silently shift the latency contract.

These tests re-affirm (not retune) the constant and verify that the env
override parses correctly.  No store, no embedder, no daemon — hermetic.
"""
from __future__ import annotations

import os


def test_default_construct_budget_ms_is_1000():
    """The default construct+smoke-encode budget is 1000 ms.

    Re-affirms the locked value.  Changing this default shifts the
    latency contract — update the surrounding analysis before doing so.
    """
    from iai_mcp.semantic_recall import _DEFAULT_CONSTRUCT_BUDGET_MS
    assert _DEFAULT_CONSTRUCT_BUDGET_MS == 1000


def test_env_override_construct_budget_ms_parses(monkeypatch):
    """IAI_MCP_EMBED_CONSTRUCT_BUDGET_MS env var overrides the budget.

    Sets the env var, reads back the effective value via the helper, then
    restores the env.  Hermetic: monkeypatched env, no store or embedder
    constructed.
    """
    from iai_mcp.semantic_recall import _construct_budget_ms

    monkeypatch.setenv("IAI_MCP_EMBED_CONSTRUCT_BUDGET_MS", "500")
    assert _construct_budget_ms() == 500

    # Invalid value falls back to the default.
    monkeypatch.setenv("IAI_MCP_EMBED_CONSTRUCT_BUDGET_MS", "not-a-number")
    assert _construct_budget_ms() == 1000

    # Absent env var → default.
    monkeypatch.delenv("IAI_MCP_EMBED_CONSTRUCT_BUDGET_MS", raising=False)
    assert _construct_budget_ms() == 1000
