"""Tests for iai_mcp.host_cli -- Task 1.

Covers 12 behaviours (DAEMON-07 + C3 constitutional):
1. invoke_host_once spawns `claude --bare -p ... --output-format json --max-turns 1
   --tools "" --no-session-persistence --model haiku` via create_subprocess_exec (argv).
2. ANTHROPIC_API_KEY / CLAUDE_API_KEY / CLAUDE_CODE_API_KEY scrubbed from child env.
3. Happy path -- returns ok=True with data, cost_usd, tokens_in, tokens_out.
4. Cost tripwire (bug #43333): cost_usd > 0 -> auto-disable Claude AND return ok=False.
5. 120s timeout -> terminate-then-kill escalation, returns ok=False reason=timeout.
6. Non-zero exit -> ok=False reason=nonzero_exit.
7. Malformed JSON stdout -> ok=False reason=unparseable_output.
8. verify_credentials_subscription gates on billingType=stripe_subscription.
9. BudgetTracker.can_spend -- daily cap + weekly buffer arithmetic.
10. BudgetTracker.reset_if_new_day -- local-midnight counter reset.
11. BudgetTracker.weekly_buffer_exceeded -- 7% ceiling.
12. Force-wake mid-call -- CancelledError triggers terminate->60s grace->kill
    escalation, returns force_wake_killed, does NOT re-raise.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect daemon_state.STATE_PATH to tmp_path for test isolation."""
    from iai_mcp import daemon_state
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    return state_path


@pytest.fixture
def fake_creds(tmp_path, monkeypatch):
    """Write a fake credentials.json and point host_cli at it."""
    creds = tmp_path / ".credentials.json"
    creds.write_text(json.dumps({"billingType": "stripe_subscription"}))
    from iai_mcp import host_cli
    monkeypatch.setattr(host_cli, "CREDENTIALS_PATH", creds)
    return creds


class _FakeProc:
    """Mock of an asyncio subprocess."""

    def __init__(
        self,
        stdout: bytes = b"{}",
        stderr: bytes = b"",
        returncode: int = 0,
        *,
        hang: bool = False,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.terminate_called = False
        self.kill_called = False

    async def communicate(self, input=None):  # noqa: ARG002
        if self._hang:
            await asyncio.sleep(3600)
        return (self._stdout, self._stderr)

    def terminate(self) -> None:
        self.terminate_called = True
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.kill_called = True
        if self.returncode is None:
            self.returncode = -9

    async def wait(self):
        return self.returncode


def _install_subprocess_mock(monkeypatch, proc: _FakeProc) -> dict:
    """Replace asyncio.create_subprocess_exec with an async callable that
    returns `proc` and captures its args/env for assertion."""
    capture: dict = {"args": None, "env": None, "kwargs": None}

    async def fake_spawn(*args, **kwargs):
        capture["args"] = args
        capture["env"] = kwargs.get("env")
        capture["kwargs"] = kwargs
        return proc

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_spawn)
    return capture


# ---------------------------------------------------------------------------
# Test 1: argv form + all required CLI flags
# ---------------------------------------------------------------------------


def test_invoke_uses_argv_and_required_flags(monkeypatch, fake_creds, isolated_state):
    from iai_mcp.host_cli import invoke_host_once

    proc = _FakeProc(stdout=json.dumps({
        "result": "ok",
        "cost_usd": 0,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }).encode("utf-8"))
    cap = _install_subprocess_mock(monkeypatch, proc)

    result = asyncio.run(invoke_host_once("hello", model="haiku"))

    assert result["ok"] is True
    args = cap["args"]
    assert args[0] == "claude"
    assert "--bare" in args
    assert "-p" in args
    assert "hello" in args
    assert "--output-format" in args and "json" in args
    assert "--max-turns" in args and "1" in args
    assert "--tools" in args
    assert "--no-session-persistence" in args
    assert "--model" in args and "haiku" in args


# ---------------------------------------------------------------------------
# Test 2: env scrubbing (C3 guard)
# ---------------------------------------------------------------------------


def test_env_scrubbed(monkeypatch, fake_creds, isolated_state):
    from iai_mcp.host_cli import invoke_host_once

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-hostile-1")
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-hostile-2")
    monkeypatch.setenv("CLAUDE_CODE_API_KEY", "sk-hostile-3")
    monkeypatch.setenv("KEEP_ME", "benign")

    proc = _FakeProc(stdout=json.dumps({
        "result": "ok", "cost_usd": 0, "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode("utf-8"))
    cap = _install_subprocess_mock(monkeypatch, proc)

    asyncio.run(invoke_host_once("hi", model="haiku"))

    env = cap["env"]
    assert env is not None
    for key in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "CLAUDE_CODE_API_KEY"):
        assert key not in env, f"C3 violation: {key} leaked to subprocess env"
    assert env.get("KEEP_ME") == "benign"


# ---------------------------------------------------------------------------
# Test 3: happy path
# ---------------------------------------------------------------------------


def test_happy_path_parses_tokens_and_cost(monkeypatch, fake_creds, isolated_state):
    from iai_mcp.host_cli import invoke_host_once

    payload = {
        "result": "unifying insight text",
        "cost_usd": 0,
        "usage": {"input_tokens": 150, "output_tokens": 40},
        "is_error": False,
        "session_id": "sess-x",
        "duration_ms": 500,
        "num_turns": 1,
    }
    proc = _FakeProc(stdout=json.dumps(payload).encode("utf-8"))
    _install_subprocess_mock(monkeypatch, proc)

    result = asyncio.run(invoke_host_once("hi", model="haiku"))
    assert result["ok"] is True
    assert result["cost_usd"] == 0.0
    assert result["tokens_in"] == 150
    assert result["tokens_out"] == 40
    assert result["data"]["result"] == "unifying insight text"


# ---------------------------------------------------------------------------
# Test 4: C3 auto-disable on cost_usd > 0 (bug #43333 tripwire)
# ---------------------------------------------------------------------------


def test_c3_auto_disable(monkeypatch, fake_creds, isolated_state):
    from iai_mcp.host_cli import BudgetTracker, invoke_host_once
    from iai_mcp.daemon_state import load_state

    payload = {
        "result": "billing detected text",
        "cost_usd": 0.05,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    proc = _FakeProc(stdout=json.dumps(payload).encode("utf-8"))
    _install_subprocess_mock(monkeypatch, proc)

    result = asyncio.run(invoke_host_once("hi", model="haiku"))
    assert result["ok"] is False
    assert result["reason"] == "api_billing_detected"
    assert result["cost_usd"] == 0.05

    tracker = BudgetTracker(load_state())
    assert tracker.host_disabled_after_billing_event() is True


# ---------------------------------------------------------------------------
# Test 5: timeout -> terminate -> kill escalation
# ---------------------------------------------------------------------------


def test_timeout_terminates_then_kills(monkeypatch, fake_creds, isolated_state):
    from iai_mcp import host_cli
    from iai_mcp.host_cli import invoke_host_once

    monkeypatch.setattr(host_cli, "HOST_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(host_cli, "TERMINATE_WAIT_SEC", 0.05)
    monkeypatch.setattr(host_cli, "KILL_WAIT_SEC", 0.05)

    proc = _FakeProc(hang=True, returncode=None)

    async def slow_wait():
        await asyncio.sleep(3600)
        return -9

    proc.wait = slow_wait  # type: ignore[assignment]
    _install_subprocess_mock(monkeypatch, proc)

    result = asyncio.run(invoke_host_once("hi", model="haiku"))
    assert result["ok"] is False
    assert result["reason"] == "timeout"
    assert proc.terminate_called is True
    assert proc.kill_called is True


# ---------------------------------------------------------------------------
# Test 6: non-zero exit
# ---------------------------------------------------------------------------


def test_nonzero_exit(monkeypatch, fake_creds, isolated_state):
    from iai_mcp.host_cli import invoke_host_once

    proc = _FakeProc(stdout=b"", stderr=b"subscription expired", returncode=1)
    _install_subprocess_mock(monkeypatch, proc)

    result = asyncio.run(invoke_host_once("hi", model="haiku"))
    assert result["ok"] is False
    assert result["reason"] == "nonzero_exit"
    assert result["exit_code"] == 1
    assert "subscription expired" in result["stderr"]


# ---------------------------------------------------------------------------
# Test 7: unparseable output
# ---------------------------------------------------------------------------


def test_unparseable_output(monkeypatch, fake_creds, isolated_state):
    from iai_mcp.host_cli import invoke_host_once

    proc = _FakeProc(stdout=b"not valid json at all", returncode=0)
    _install_subprocess_mock(monkeypatch, proc)

    result = asyncio.run(invoke_host_once("hi", model="haiku"))
    assert result["ok"] is False
    assert result["reason"] == "unparseable_output"


# ---------------------------------------------------------------------------
# Test 8: credentials.json gate
# ---------------------------------------------------------------------------


def test_credentials_gate(tmp_path, monkeypatch):
    from iai_mcp import host_cli
    from iai_mcp.host_cli import verify_credentials_subscription

    creds = tmp_path / ".credentials.json"
    monkeypatch.setattr(host_cli, "CREDENTIALS_PATH", creds)

    assert verify_credentials_subscription()["ok"] is False

    creds.write_text(json.dumps({"billingType": "api_key"}))
    r = verify_credentials_subscription()
    assert r["ok"] is False
    assert r["reason"] == "not_subscription"

    creds.write_text(json.dumps({"billingType": "stripe_subscription"}))
    r2 = verify_credentials_subscription()
    assert r2["ok"] is True
    assert r2["billing_type"] == "stripe_subscription"


# ---------------------------------------------------------------------------
# Test 9: BudgetTracker.can_spend arithmetic
# ---------------------------------------------------------------------------


def test_budget_cap(isolated_state):
    from iai_mcp.host_cli import (
        BUDGET_STATE_KEY,
        BudgetTracker,
        DAILY_QUOTA_BUDGET_PCT,
        ESTIMATED_DAILY_TOKEN_CEILING,
    )

    daily_cap = int(DAILY_QUOTA_BUDGET_PCT * ESTIMATED_DAILY_TOKEN_CEILING)

    state = {BUDGET_STATE_KEY: {
        "daily_used_tokens": daily_cap - 500,
        "weekly_buffer_used_tokens": 0,
        "last_reset_date": "2026-04-18",
        "host_disabled": False,
        "host_disabled_reason": None,
    }}
    assert BudgetTracker(state).can_spend(100) is True

    state2 = {BUDGET_STATE_KEY: {
        "daily_used_tokens": daily_cap,
        "weekly_buffer_used_tokens": 0,
        "last_reset_date": "2026-04-18",
        "host_disabled": False,
        "host_disabled_reason": None,
    }}
    assert BudgetTracker(state2).can_spend(ESTIMATED_DAILY_TOKEN_CEILING) is False

    state3 = {BUDGET_STATE_KEY: {
        "daily_used_tokens": 0,
        "weekly_buffer_used_tokens": 0,
        "last_reset_date": "2026-04-18",
        "host_disabled": True,
        "host_disabled_reason": "api_billing_detected",
    }}
    assert BudgetTracker(state3).can_spend(1) is False


# ---------------------------------------------------------------------------
# Test 10: reset_if_new_day
# ---------------------------------------------------------------------------


def test_reset_if_new_day(isolated_state):
    from iai_mcp.host_cli import BUDGET_STATE_KEY, BudgetTracker

    tz = ZoneInfo("Asia/Dubai")  # UTC+4
    state = {BUDGET_STATE_KEY: {
        "daily_used_tokens": 8000,
        "weekly_buffer_used_tokens": 0,
        "last_reset_date": "2026-04-17",
        "host_disabled": False,
        "host_disabled_reason": None,
    }}
    t = BudgetTracker(state)

    now_same_day = datetime(2026, 4, 17, 23, 0, tzinfo=tz)
    t.reset_if_new_day(now_same_day, tz)
    assert state[BUDGET_STATE_KEY]["daily_used_tokens"] == 8000

    now_new_day = datetime(2026, 4, 18, 1, 0, tzinfo=tz)
    t.reset_if_new_day(now_new_day, tz)
    assert state[BUDGET_STATE_KEY]["daily_used_tokens"] == 0
    assert state[BUDGET_STATE_KEY]["last_reset_date"] == "2026-04-18"


# ---------------------------------------------------------------------------
# Test 11: weekly buffer ceiling
# ---------------------------------------------------------------------------


def test_weekly_buffer_exceeded(isolated_state):
    from iai_mcp.host_cli import (
        BUDGET_STATE_KEY,
        BudgetTracker,
        ESTIMATED_DAILY_TOKEN_CEILING,
        WEEKLY_BUFFER_PCT,
    )

    weekly_cap = int(WEEKLY_BUFFER_PCT * ESTIMATED_DAILY_TOKEN_CEILING * 7)
    state_under = {BUDGET_STATE_KEY: {
        "daily_used_tokens": 0,
        "weekly_buffer_used_tokens": weekly_cap - 1,
        "last_reset_date": "2026-04-18",
        "host_disabled": False,
        "host_disabled_reason": None,
    }}
    assert BudgetTracker(state_under).weekly_buffer_exceeded() is False

    state_over = {BUDGET_STATE_KEY: {
        "daily_used_tokens": 0,
        "weekly_buffer_used_tokens": weekly_cap,
        "last_reset_date": "2026-04-18",
        "host_disabled": False,
        "host_disabled_reason": None,
    }}
    assert BudgetTracker(state_over).weekly_buffer_exceeded() is True


# ---------------------------------------------------------------------------
# Test 12: force-wake mid-Claude does not crash daemon (D-19 + Warning 8)
# ---------------------------------------------------------------------------


def test_force_wake_does_not_crash_daemon(monkeypatch, fake_creds, isolated_state):
    """CancelledError while awaiting claude -p must be handled cooperatively.
    invoke_host_once terminates the subprocess (60s grace -> kill) and returns
    a structured dict WITHOUT re-raising. Re-raising would propagate up and
    potentially crash the daemon scheduler."""
    from iai_mcp import host_cli
    from iai_mcp.host_cli import invoke_host_once

    monkeypatch.setattr(host_cli, "FORCE_WAKE_GRACE_SEC", 0.05)
    monkeypatch.setattr(host_cli, "KILL_WAIT_SEC", 0.05)

    proc = _FakeProc(hang=True, returncode=None)

    async def slow_wait():
        await asyncio.sleep(3600)
        return -9

    proc.wait = slow_wait  # type: ignore[assignment]
    _install_subprocess_mock(monkeypatch, proc)

    async def runner():
        task = asyncio.create_task(invoke_host_once("hi", model="haiku"))
        await asyncio.sleep(0)
        task.cancel()
        return await task

    result = asyncio.run(runner())
    assert isinstance(result, dict)
    assert result["ok"] is False
    assert result["reason"] == "force_wake_killed"
    assert proc.terminate_called is True
    assert proc.kill_called is True
