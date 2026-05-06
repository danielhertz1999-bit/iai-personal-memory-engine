"""Tests for iai_mcp.insight -- (D-13 Option A lucid moment).

Covers 12 behaviours (DAEMON-08):
1. Exactly ONE invoke_host_once call per generate_overnight_insight invocation.
2. Prompt text contains the Option A verbatim template fragments.
3. Top-3 recent schemas pulled from schema.induce_schemas_tier0 by confidence.
4. Top-1 surprise event from events query goes into the {surprise} slot.
5. Happy path: semantic L1 MemoryRecord with tag='overnight_insight' is inserted.
6. Budget pre-flight gate: can_spend False -> no subprocess call, ok=False.
7. host_disabled_after_billing_event True -> no subprocess call, ok=False.
8. Credentials gate fails -> no subprocess call, ok=False.
9. Budget is recorded with (tokens_in + tokens_out) after successful call.
10. cost_usd > 0 from invoke_host_once -> no record stored, ok=False.
11. Empty store -> placeholder pattern/surprise used, Claude STILL called.
12. write_event emits 'overnight_insight_generated' on success.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fresh store helper (mirrors pattern used in test_dream.py)
# ---------------------------------------------------------------------------


def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    from iai_mcp.store import MemoryStore
    return MemoryStore()


# ---------------------------------------------------------------------------
# Shared mocks: fake invoke_host_once + credentials gate + isolated state
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    from iai_mcp import daemon_state
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    return state_path


@pytest.fixture
def creds_ok(monkeypatch):
    """Force verify_credentials_subscription -> ok=True without touching disk."""
    monkeypatch.setattr(
        "iai_mcp.insight.verify_credentials_subscription",
        lambda: {"ok": True, "billing_type": "stripe_subscription"},
    )


@pytest.fixture
def mock_claude_ok(monkeypatch, creds_ok, isolated_state):
    """Mock invoke_host_once -> ok payload; record captured prompts + call count."""
    calls: list[dict] = []

    async def fake_invoke(prompt: str, *, model: str = "haiku"):
        calls.append({"prompt": prompt, "model": model})
        return {
            "ok": True,
            "data": {"result": "unifying insight text"},
            "tokens_in": 200,
            "tokens_out": 40,
            "cost_usd": 0.0,
        }

    monkeypatch.setattr("iai_mcp.insight.invoke_host_once", fake_invoke)
    return calls


# ---------------------------------------------------------------------------
# Test 1: exactly one invoke per call
# ---------------------------------------------------------------------------


def test_one_call_per_night(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.insight import generate_overnight_insight
    store = _fresh_store(tmp_path, monkeypatch)

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is True
    assert len(mock_claude_ok) == 1

    # Second call means a SECOND night -- also exactly one claude call.
    asyncio.run(generate_overnight_insight(store, "sess-B"))
    assert len(mock_claude_ok) == 2


# ---------------------------------------------------------------------------
# Test 2: verbatim prompt fragments
# ---------------------------------------------------------------------------


def test_prompt_template(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.insight import generate_overnight_insight
    store = _fresh_store(tmp_path, monkeypatch)
    asyncio.run(generate_overnight_insight(store, "sess-A"))

    prompt = mock_claude_ok[0]["prompt"]
    # Option A verbatim fragments.
    assert "3 locally-found patterns" in prompt
    assert "1 surprising episode" in prompt
    assert "unifying insight" in prompt
    assert "1-2 sentences" in prompt
    # Model default -- haiku preference.
    assert mock_claude_ok[0]["model"] == "haiku"


# ---------------------------------------------------------------------------
# Test 3: patterns pulled from schema induction
# ---------------------------------------------------------------------------


def test_patterns_from_schemas(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.insight import generate_overnight_insight
    from iai_mcp.schema import SchemaCandidate

    store = _fresh_store(tmp_path, monkeypatch)

    # Seed 5 candidates with distinct confidences -- top 3 by confidence
    # should show up in the prompt.
    fake_candidates = [
        SchemaCandidate(
            pattern=f"pattern-{i}",
            confidence=0.1 * (i + 1),
            evidence_count=3 + i,
        )
        for i in range(5)
    ]
    monkeypatch.setattr(
        "iai_mcp.insight.induce_schemas_tier0",
        lambda _store: fake_candidates,
    )

    asyncio.run(generate_overnight_insight(store, "sess-A"))
    prompt = mock_claude_ok[0]["prompt"]

    # Top 3 by confidence are patterns 4, 3, 2 (confidence 0.5, 0.4, 0.3).
    assert "pattern-4" in prompt
    assert "pattern-3" in prompt
    assert "pattern-2" in prompt


# ---------------------------------------------------------------------------
# Test 4: surprise event extraction
# ---------------------------------------------------------------------------


def test_surprise_from_events(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)

    fake_events = [
        {"kind": "art_gate_high_novelty",
         "data": {"summary": "UNEXPECTED-MARKER-ALPHA"}, "ts": "x"},
        {"kind": "routine_event", "data": {"summary": "boring"}, "ts": "y"},
    ]
    monkeypatch.setattr(
        "iai_mcp.insight.query_events",
        lambda _store, *, since=None, limit=1000: fake_events,
    )

    asyncio.run(generate_overnight_insight(store, "sess-A"))
    prompt = mock_claude_ok[0]["prompt"]
    assert "UNEXPECTED-MARKER-ALPHA" in prompt


# ---------------------------------------------------------------------------
# Test 5: record stored with L1 semantic tier + overnight_insight tag
# ---------------------------------------------------------------------------


def test_record_tag(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    inserted: list = []

    real_insert = store.insert

    def spy_insert(rec):
        inserted.append(rec)
        return real_insert(rec)

    monkeypatch.setattr(store, "insert", spy_insert)

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is True
    assert len(inserted) == 1
    rec = inserted[0]
    assert rec.tier == "semantic"
    assert rec.tag == "overnight_insight" or "overnight_insight" in (rec.tags or [])
    assert rec.literal_surface == "unifying insight text"


# ---------------------------------------------------------------------------
# Test 6: budget gate blocks the call
# ---------------------------------------------------------------------------


def test_budget_gate_blocks(tmp_path, monkeypatch, creds_ok, isolated_state):
    from iai_mcp.host_cli import BUDGET_STATE_KEY, DAILY_QUOTA_BUDGET_PCT, ESTIMATED_DAILY_TOKEN_CEILING
    from iai_mcp.daemon_state import save_state
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)

    calls: list = []

    async def fake_invoke(prompt, *, model="haiku"):
        calls.append(1)
        return {"ok": True, "data": {"result": "x"}, "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0}

    monkeypatch.setattr("iai_mcp.insight.invoke_host_once", fake_invoke)

    # Saturate daily + weekly so can_spend returns False.
    daily_cap = int(DAILY_QUOTA_BUDGET_PCT * ESTIMATED_DAILY_TOKEN_CEILING)
    save_state({BUDGET_STATE_KEY: {
        "daily_used_tokens": daily_cap,
        "weekly_buffer_used_tokens": 10_000_000,
        "last_reset_date": datetime.now(timezone.utc).date().isoformat(),
        "host_disabled": False,
        "host_disabled_reason": None,
    }})

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is False
    assert result["reason"] == "budget_exceeded"
    assert calls == []


# ---------------------------------------------------------------------------
# Test 7: C3 auto-disabled flag blocks the call
# ---------------------------------------------------------------------------


def test_host_disabled_blocks(tmp_path, monkeypatch, creds_ok, isolated_state):
    from iai_mcp.host_cli import BUDGET_STATE_KEY
    from iai_mcp.daemon_state import save_state
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)

    calls: list = []

    async def fake_invoke(prompt, *, model="haiku"):
        calls.append(1)
        return {"ok": True, "data": {"result": "x"}, "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0}

    monkeypatch.setattr("iai_mcp.insight.invoke_host_once", fake_invoke)
    save_state({BUDGET_STATE_KEY: {
        "daily_used_tokens": 0,
        "weekly_buffer_used_tokens": 0,
        "last_reset_date": datetime.now(timezone.utc).date().isoformat(),
        "host_disabled": True,
        "host_disabled_reason": "api_billing_detected",
    }})

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is False
    assert result["reason"] == "host_disabled_c3"
    assert calls == []


# ---------------------------------------------------------------------------
# Test 8: credentials gate blocks the call
# ---------------------------------------------------------------------------


def test_credentials_gate_blocks(tmp_path, monkeypatch, isolated_state):
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    calls: list = []

    async def fake_invoke(prompt, *, model="haiku"):
        calls.append(1)
        return {"ok": True}

    monkeypatch.setattr("iai_mcp.insight.invoke_host_once", fake_invoke)
    monkeypatch.setattr(
        "iai_mcp.insight.verify_credentials_subscription",
        lambda: {"ok": False, "reason": "not_subscription"},
    )

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is False
    assert result["reason"] == "credentials_check_failed"
    assert calls == []


# ---------------------------------------------------------------------------
# Test 9: budget is recorded
# ---------------------------------------------------------------------------


def test_budget_recorded(tmp_path, monkeypatch, mock_claude_ok, isolated_state):
    from iai_mcp.host_cli import BUDGET_STATE_KEY
    from iai_mcp.daemon_state import load_state
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    asyncio.run(generate_overnight_insight(store, "sess-A"))

    state = load_state()
    assert state[BUDGET_STATE_KEY]["daily_used_tokens"] == 240  # 200 + 40


# ---------------------------------------------------------------------------
# Test 10: api_billing_detected short-circuits storage
# ---------------------------------------------------------------------------


def test_api_billing_detected_no_store(tmp_path, monkeypatch, creds_ok, isolated_state):
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    inserted: list = []
    real_insert = store.insert
    monkeypatch.setattr(store, "insert", lambda r: inserted.append(r) or real_insert(r))

    async def fake_invoke(prompt, *, model="haiku"):
        return {
            "ok": False,
            "reason": "api_billing_detected",
            "cost_usd": 0.05,
            "data": {"result": "hostile"},
            "tokens_in": 100,
            "tokens_out": 20,
        }

    monkeypatch.setattr("iai_mcp.insight.invoke_host_once", fake_invoke)

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is False
    assert result["reason"] == "api_billing_detected"
    # Zero overnight_insight records stored.
    assert all(
        "overnight_insight" not in (getattr(r, "tags", []) or [])
        and getattr(r, "tag", None) != "overnight_insight"
        for r in inserted
    )


# ---------------------------------------------------------------------------
# Test 11: empty store still calls Claude (graceful degradation)
# ---------------------------------------------------------------------------


def test_empty_store_still_calls(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    # Force both pattern + surprise to come back empty.
    monkeypatch.setattr("iai_mcp.insight.induce_schemas_tier0", lambda _s: [])
    monkeypatch.setattr(
        "iai_mcp.insight.query_events",
        lambda _s, *, since=None, limit=1000: [],
    )

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is True
    assert len(mock_claude_ok) == 1
    prompt = mock_claude_ok[0]["prompt"]
    assert "[no patterns yet]" in prompt
    assert "[no surprise yet]" in prompt


# ---------------------------------------------------------------------------
# Test 12: overnight_insight_generated event emitted on success
# ---------------------------------------------------------------------------


def test_event_emitted(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.events import query_events
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    asyncio.run(generate_overnight_insight(store, "sess-A"))

    events = query_events(store, kind="overnight_insight_generated", limit=10)
    assert len(events) >= 1
    assert events[0]["data"].get("session_id") == "sess-A"
