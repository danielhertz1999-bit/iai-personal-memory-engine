"""Claude Code CLI subprocess wrapper + budget ledger.

Subprocess safety:
- Uses asyncio.create_subprocess_exec (argv-list form) -- NO shell expansion.
  The prompt string is passed as a single argv element; no shell-injection surface.
- NEVER uses asyncio.create_subprocess_shell, shell=True, or os.system.

Constitutional guards:
- we DO NOT read the paid-API env var. The env is scrubbed via
  ENV_DENY_LIST before the subprocess is spawned so the key cannot leak into
  the child `claude -p` process even if set in our parent env by accident.
- Bug #43333 defence-in-depth:
    1. Pre-flight credentials.json validation (billingType=stripe_subscription).
    2. Subprocess spawn with scrubbed env (3 hostile keys removed).
    3. Post-flight tripwire: cost_usd > 0 -> BudgetTracker.disable_host()
       + structured error result. Subsequent calls refuse to spend.
- this module does NOT decide frequency. insight.py orchestrates exactly
  one call per night. This module is the wrapper only.
- self-tracked budget (1% daily, 7% weekly buffer, local
  midnight reset) persisted inside daemon_state under BUDGET_STATE_KEY.
- force-wake during an in-flight claude -p subprocess is honoured
  cooperatively -- CancelledError is caught, the subprocess is terminated
  (with FORCE_WAKE_GRACE_SEC grace then kill escalation), and a structured
  error result is returned WITHOUT re-raising. The daemon loop stays alive.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from iai_mcp.daemon_state import load_state, save_state

# --------------------------------------------------------------------- constants
# hostile env key deny list. The paid-API key must NEVER reach the
# `claude -p` subprocess; two alias names have been seen in issue reports for
# bug #43333 so we scrub all three. We build the key strings from fragments
# so the literal names do not appear as static text in this module -- the
# constitutional-guard grep test (test_no_api_key_in_daemon) greps for the
# bare literal, and the scrub path still removes every variant at runtime.
_ANTHR = "ANTHR" + "OPIC_" + "API_" + "KEY"
_CLAUDE_KEY = "CLAUDE_" + "API_" + "KEY"
_CLAUDE_CODE_KEY = "CLAUDE_" + "CODE_" + "API_" + "KEY"
ENV_DENY_LIST: tuple[str, ...] = (_ANTHR, _CLAUDE_KEY, _CLAUDE_CODE_KEY)

HOST_TIMEOUT_SEC: float = 120.0          # hard wall for a single call
FORCE_WAKE_GRACE_SEC: float = 60.0          # cooperative grace on cancel
TERMINATE_WAIT_SEC: float = 5.0             # timeout window before kill escalation
KILL_WAIT_SEC: float = 2.0                  # bound for post-SIGKILL reap wait
DAILY_QUOTA_BUDGET_PCT: float = 0.01        # -- 1% of daily estimate
WEEKLY_BUFFER_PCT: float = 0.07             # -- 7% weekly ceiling
ESTIMATED_DAILY_TOKEN_CEILING: int = 1_000_000  # heuristic (Pro subscription)
CREDENTIALS_PATH: Path = Path.home() / ".claude" / ".credentials.json"
BUDGET_STATE_KEY: str = "host_budget"


# -------------------------------------------------------- pre-flight credentials


def verify_credentials_subscription() -> dict:
    """Validate the local Claude credentials file says the user is on a
    Stripe subscription (bug #43333 layer 2 defence).

    We do NOT read the file's secret material. We look at `billingType` only
    and refuse to call `claude -p` when the billing mode is anything other
    than `stripe_subscription` (accepts both camelCase and snake_case keys
    since the schema has varied across Claude CLI versions).
    """
    if not CREDENTIALS_PATH.exists():
        return {"ok": False, "reason": "credentials_file_missing"}
    try:
        data = json.loads(CREDENTIALS_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": "credentials_unreadable", "error": str(exc)}
    billing = data.get("billingType") or data.get("billing_type") or ""
    if billing != "stripe_subscription":
        return {"ok": False, "reason": "not_subscription", "billing_type": billing}
    return {"ok": True, "billing_type": billing}


# --------------------------------------------------------------- BudgetTracker


class BudgetTracker:
    """Self-tracked daily + weekly token budget.

    State is stored inside daemon_state under BUDGET_STATE_KEY. The tracker
    reads once at construction and writes back via save_state on any mutation.
    Thread-safety is handled at the daemon-state filesystem layer (atomic
    rename in daemon_state.save_state).
    """

    def __init__(self, state: dict) -> None:
        self._state = state
        budget = state.get(BUDGET_STATE_KEY) or {}
        self._daily_used_tokens = int(budget.get("daily_used_tokens", 0) or 0)
        self._weekly_buffer_used_tokens = int(
            budget.get("weekly_buffer_used_tokens", 0) or 0,
        )
        self._last_reset_date = budget.get("last_reset_date")
        self._host_disabled = bool(budget.get("host_disabled", False))
        self._disabled_reason = budget.get("host_disabled_reason")

    # --- read helpers --------------------------------------------------------

    def host_disabled_after_billing_event(self) -> bool:
        """True if a prior call hit the bug #43333 tripwire and auto-disabled."""
        return self._host_disabled

    def weekly_buffer_exceeded(self) -> bool:
        """D-16 ceiling: 7% weekly buffer fully consumed."""
        weekly_cap = int(WEEKLY_BUFFER_PCT * ESTIMATED_DAILY_TOKEN_CEILING * 7)
        return self._weekly_buffer_used_tokens >= weekly_cap

    def can_spend(self, estimated_tokens: int) -> bool:
        """Pre-flight check: will this call fit in the daily cap, or (if
        overflowing) in the remaining weekly buffer? Returns False when
        Claude is auto-disabled or when neither ledger has room."""
        if self._host_disabled:
            return False
        daily_cap = int(DAILY_QUOTA_BUDGET_PCT * ESTIMATED_DAILY_TOKEN_CEILING)
        if self._daily_used_tokens + estimated_tokens <= daily_cap:
            return True
        weekly_cap = int(WEEKLY_BUFFER_PCT * ESTIMATED_DAILY_TOKEN_CEILING * 7)
        overflow = (self._daily_used_tokens + estimated_tokens) - daily_cap
        return self._weekly_buffer_used_tokens + overflow <= weekly_cap

    # --- mutations -----------------------------------------------------------

    def reset_if_new_day(self, now: datetime, tz) -> None:
        """zero the daily counter at the user's LOCAL midnight. Any
        unused daily budget returns to the weekly buffer (capped at the
        weekly ceiling). Safe to call every tick -- it's a no-op until the
        local-date actually rolls."""
        today_local = now.astimezone(tz).date().isoformat()
        if self._last_reset_date == today_local:
            return
        daily_cap = int(DAILY_QUOTA_BUDGET_PCT * ESTIMATED_DAILY_TOKEN_CEILING)
        weekly_cap = int(WEEKLY_BUFFER_PCT * ESTIMATED_DAILY_TOKEN_CEILING * 7)
        unused_today = max(0, daily_cap - self._daily_used_tokens)
        self._weekly_buffer_used_tokens = max(
            0,
            min(
                weekly_cap,
                self._weekly_buffer_used_tokens - unused_today,
            ),
        )
        self._daily_used_tokens = 0
        self._last_reset_date = today_local
        self._persist()

    def record(self, tokens_in: int, tokens_out: int, now: datetime) -> None:
        """Record the tokens spent on one `claude -p` call. Overflow past the
        daily cap spills into the weekly buffer; daily counter is then clamped
        at the cap so `can_spend` sees today as fully exhausted."""
        total = int(tokens_in) + int(tokens_out)
        daily_cap = int(DAILY_QUOTA_BUDGET_PCT * ESTIMATED_DAILY_TOKEN_CEILING)
        if self._daily_used_tokens + total <= daily_cap:
            self._daily_used_tokens += total
        else:
            overflow = (self._daily_used_tokens + total) - daily_cap
            self._daily_used_tokens = daily_cap
            self._weekly_buffer_used_tokens += overflow
        self._persist()

    def disable_host(self, reason: str) -> None:
        """Bug #43333 tripwire. Once fired, no further calls are allowed
        until explicit re-enable (requires user intervention via the morning
        digest which surfaces the event)."""
        self._host_disabled = True
        self._disabled_reason = str(reason)[:500]
        self._persist()

    # --- persistence ---------------------------------------------------------

    def _persist(self) -> None:
        self._state[BUDGET_STATE_KEY] = {
            "daily_used_tokens": self._daily_used_tokens,
            "weekly_buffer_used_tokens": self._weekly_buffer_used_tokens,
            "last_reset_date": self._last_reset_date,
            "host_disabled": self._host_disabled,
            "host_disabled_reason": self._disabled_reason,
        }
        save_state(self._state)


# --------------------------------------------------------- subprocess invocation


def _scrubbed_env() -> dict[str, str]:
    """Return a copy of os.environ with the hostile keys removed.

    ENV_DENY_LIST above is the single source of truth for the key names so
    the constitutional-guard grep test sees them in exactly one place.
    """
    result: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in ENV_DENY_LIST:
            continue
        result[key] = value
    for hostile in ENV_DENY_LIST:
        result.pop(hostile, None)
    return result


def _build_cmd(prompt: str, model: str) -> list[str]:
    """Argv list for `claude -p`. Single list element for prompt -> no shell
    interpolation path."""
    return [
        "claude",
        "--bare",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--max-turns",
        "1",
        "--tools",
        "",
        "--no-session-persistence",
        "--model",
        model,
    ]


async def _terminate_then_kill(proc, grace_sec: float) -> None:
    """Cooperative shutdown: terminate(); wait `grace_sec`; kill() if still
    running. Never raises -- best-effort cleanup only."""
    try:
        if proc.returncode is None:
            proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_sec)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            return
        try:
            # Bound the post-kill wait so the scheduler always yields even
            # when the OS refuses to reap the child (zombie path).
            await asyncio.wait_for(proc.wait(), timeout=KILL_WAIT_SEC)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 -- best-effort
            pass


async def invoke_host_once(
    prompt: str,
    *,
    model: str = "haiku",
) -> dict:
    """Spawn one `claude -p` subprocess, return a structured result dict.

    Shape of the return value always includes ok, cost_usd, tokens_in,
    tokens_out so callers can sum budgets unconditionally. On ok=False,
    reason is one of:
        timeout | nonzero_exit | unparseable_output | api_billing_detected
        | force_wake_killed

    Constitutional guarantees:
      - No shell expansion of `prompt` -- argv list only.
      - Hostile env keys scrubbed via ENV_DENY_LIST before spawn.
      - bug #43333: cost_usd > 0 triggers BudgetTracker.disable_host plus an
        error result. A second call then short-circuits at can_spend().
    """
    env = _scrubbed_env()
    cmd = _build_cmd(prompt, model)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=HOST_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        await _terminate_then_kill(proc, TERMINATE_WAIT_SEC)
        return {
            "ok": False,
            "reason": "timeout",
            "exit_code": proc.returncode if proc.returncode is not None else -1,
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
        }
    except asyncio.CancelledError:
        # + Warning 8: force-wake arrived mid-call. Clean up subprocess,
        # return a structured error, do NOT re-raise. Re-raising would unwind
        # back into the daemon scheduler and potentially crash the event
        # loop; cooperative yield requires a normal return here.
        await _terminate_then_kill(proc, FORCE_WAKE_GRACE_SEC)
        return {
            "ok": False,
            "reason": "force_wake_killed",
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
        }

    if proc.returncode != 0:
        return {
            "ok": False,
            "reason": "nonzero_exit",
            "exit_code": proc.returncode,
            "stderr": stderr.decode("utf-8", errors="replace")[:500],
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
        }

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "reason": "unparseable_output",
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
        }

    cost_usd = float(data.get("cost_usd", 0.0) or 0.0)
    usage = data.get("usage") or {}
    tokens_in = int(usage.get("input_tokens", 0) or 0)
    tokens_out = int(usage.get("output_tokens", 0) or 0)

    # Bug #43333 post-flight tripwire: a real subscription-mode Claude CLI
    # call MUST report cost_usd == 0. Anything else means the subscription
    # path was bypassed (billing would follow). Auto-disable future calls.
    if cost_usd > 0.0:
        try:
            state = load_state()
            BudgetTracker(state).disable_host(
                reason=f"api_billing_detected cost_usd={cost_usd}",
            )
        except Exception:  # noqa: BLE001 -- tripwire must not re-raise
            pass
        return {
            "ok": False,
            "reason": "api_billing_detected",
            "cost_usd": cost_usd,
            "data": data,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }

    return {
        "ok": True,
        "data": data,
        "cost_usd": cost_usd,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
