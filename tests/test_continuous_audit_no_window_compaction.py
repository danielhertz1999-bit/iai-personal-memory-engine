"""Asserts continuous_audit does NOT call optimize_hippo_storage outside
the canonical consolidation window, and that the S5/sigma read stages still
run normally.

The canonical sleep pipeline's OPTIMIZE_LANCE step is the sole driver of
Hippo storage compaction. The audit loop is a pure-read loop that must never
call optimize_hippo_storage directly.
"""
from __future__ import annotations

import asyncio

import pytest


def test_continuous_audit_never_calls_optimize_hippo_storage(monkeypatch):
    """optimize_hippo_storage must NOT be called from the audit loop."""
    from iai_mcp import identity_audit

    optimize_calls: list = []

    def fake_optimize(store):
        optimize_calls.append(store)
        return {}

    # Patch optimize_hippo_storage on the maintenance module — even if the
    # audit loop imported it by name, it would call through the same object.
    try:
        from iai_mcp import maintenance as _maint
        monkeypatch.setattr(_maint, "optimize_hippo_storage", fake_optimize)
    except (ImportError, AttributeError):
        pass

    # Also patch directly on identity_audit's module namespace in case it
    # retained a direct reference at import time.
    if hasattr(identity_audit, "optimize_hippo_storage"):
        monkeypatch.setattr(identity_audit, "optimize_hippo_storage", fake_optimize)

    s5_calls: list = []
    sigma_calls: list = []

    def fake_s5(store, window):
        s5_calls.append(store)

    def fake_sigma(store):
        sigma_calls.append(store)

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", fake_s5)
    monkeypatch.setattr(identity_audit, "compute_and_emit", fake_sigma)

    # Run several ticks with a very short interval.
    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(
                object(), shutdown, interval_sec=0.01,
            )
        )
        # Let it run ~4 ticks worth.
        await asyncio.sleep(0.06)
        shutdown.set()
        await task

    asyncio.run(runner())

    # Core assertion: compaction must NEVER be called from the audit loop.
    assert optimize_calls == [], (
        f"optimize_hippo_storage was called {len(optimize_calls)} time(s) "
        "from the audit loop — expected 0"
    )


def test_continuous_audit_read_stages_still_run(monkeypatch):
    """S5 drift anomaly detection and sigma topology snapshot must still run."""
    from iai_mcp import identity_audit

    s5_calls: list = []
    sigma_calls: list = []

    def fake_s5(store, window):
        s5_calls.append(store)

    def fake_sigma(store):
        sigma_calls.append(store)

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", fake_s5)
    monkeypatch.setattr(identity_audit, "compute_and_emit", fake_sigma)

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(
                object(), shutdown, interval_sec=0.01,
            )
        )
        # Allow at least 2 ticks.
        await asyncio.sleep(0.05)
        shutdown.set()
        await task

    asyncio.run(runner())

    assert len(s5_calls) >= 2, (
        f"detect_drift_anomaly was called {len(s5_calls)} time(s); expected >= 2"
    )
    assert len(sigma_calls) >= 2, (
        f"compute_and_emit was called {len(sigma_calls)} time(s); expected >= 2"
    )


def test_continuous_audit_module_has_no_optimize_import():
    """The audit module must not expose optimize_hippo_storage as an attribute.

    This ensures the Stage-3 compaction removal is complete at the import
    level, not just conditionally skipped at runtime.
    """
    from iai_mcp import identity_audit

    assert not hasattr(identity_audit, "optimize_hippo_storage"), (
        "identity_audit still exposes optimize_hippo_storage; "
        "the Stage-3 compaction removal is incomplete"
    )
    # The _last_optimize_completed_at bookkeeping variable must also be gone.
    assert not hasattr(identity_audit, "_last_optimize_completed_at"), (
        "identity_audit still carries _last_optimize_completed_at; "
        "the Stage-3 compaction removal is incomplete"
    )
