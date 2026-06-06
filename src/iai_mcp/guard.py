"""Graceful-degradation ladder before any LLM call.

Every LLM-dependent operation must pass through `should_call_llm`
BEFORE making an API call. The 7-step ladder:

1. sleep.llm_enabled=true? else Tier 0
2. API key present? else Tier 0
3. BudgetLedger daily cap OK? else Tier 0
4. BudgetLedger monthly cap OK? else Tier 0
5. RateLimitLedger: last 429 > 15 min ago? else Tier 0 this cycle
6. API call with retry(max=2, exp backoff) + timeout(60s) -- caller's job
7. On 429/400/401/5xx -> record in ledger, Tier 0 this cycle -- caller's job

Write & read paths (memory_recall/reinforce/contradict, profile_get/set,
session_start) NEVER block on LLM failure. LLM failures only reduce the QUALITY
of semantic consolidation, schema induction, and identity refinement.

Budget defaults: daily_usd_cap=$0.10, monthly_usd_cap=$3.00,
cooldown=15min, on_cap_hit=fallback_to_local.

BudgetLedger + RateLimitLedger persist in store tables (budget_ledger,
ratelimit_ledger) created by MemoryStore._ensure_tables.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from iai_mcp.store import BUDGET_TABLE, RATELIMIT_TABLE, MemoryStore


# Degradation-ladder defaults
BUDGET_DAILY_USD_DEFAULT = 0.10
BUDGET_MONTHLY_USD_DEFAULT = 3.00
RATELIMIT_COOLDOWN_MIN = 15


class BudgetLedger:
    """SQLite-backed daily + monthly USD spend tracker.

    Caps default to $0.10/day and $3.00/month. Both are advisory (no OS-level
    enforcement); caller inspects can_spend() before invoking an LLM API.
    """

    def __init__(
        self,
        store: MemoryStore,
        daily_usd_cap: float = BUDGET_DAILY_USD_DEFAULT,
        monthly_usd_cap: float = BUDGET_MONTHLY_USD_DEFAULT,
    ) -> None:
        self.store = store
        self.daily_cap = float(daily_usd_cap)
        self.monthly_cap = float(monthly_usd_cap)

    # ---- internal helpers

    def _today_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _this_month(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    # ---- queries

    def daily_used(self) -> float:
        """Sum of usd_spent rows for today (UTC)."""
        tbl = self.store.db.open_table(BUDGET_TABLE)
        df = tbl.to_pandas()
        if df.empty:
            return 0.0
        today = df[df["date"] == self._today_utc()]
        return float(today["usd_spent"].sum()) if not today.empty else 0.0

    def monthly_used(self) -> float:
        """Sum of usd_spent rows for the current month (UTC)."""
        tbl = self.store.db.open_table(BUDGET_TABLE)
        df = tbl.to_pandas()
        if df.empty:
            return 0.0
        mo = df[df["date"].str.startswith(self._this_month())]
        return float(mo["usd_spent"].sum()) if not mo.empty else 0.0

    def can_spend(self, usd: float) -> tuple[bool, str]:
        """Return (ok, reason). reason is "" on success."""
        daily = self.daily_used()
        if daily + float(usd) > self.daily_cap:
            return (
                False,
                f"daily cap exceeded (used {daily:.4f} + {float(usd):.4f} "
                f"> {self.daily_cap:.4f})",
            )
        monthly = self.monthly_used()
        if monthly + float(usd) > self.monthly_cap:
            return (
                False,
                f"monthly cap exceeded (used {monthly:.4f} + {float(usd):.4f} "
                f"> {self.monthly_cap:.4f})",
            )
        return True, ""

    # ---- writes

    def record_spend(self, usd: float, kind: str = "llm") -> None:
        """Persist a spend event to the ledger."""
        tbl = self.store.db.open_table(BUDGET_TABLE)
        tbl.add(
            [
                {
                    "date": self._today_utc(),
                    "usd_spent": float(usd),
                    "kind": kind,
                    "ts": datetime.now(timezone.utc),
                }
            ]
        )


class RateLimitLedger:
    """SQLite-backed 429 history with 15-min cooldown."""

    def __init__(
        self,
        store: MemoryStore,
        cooldown_minutes: int = RATELIMIT_COOLDOWN_MIN,
    ) -> None:
        self.store = store
        self.cooldown = timedelta(minutes=int(cooldown_minutes))

    def in_cooldown(self) -> bool:
        """True iff the most recent 429 was less than `cooldown_minutes` ago."""
        tbl = self.store.db.open_table(RATELIMIT_TABLE)
        df = tbl.to_pandas()
        if df.empty:
            return False
        latest = df["ts"].max()
        # Coerce ISO TEXT / Timestamp / naive datetime -> tz-aware UTC datetime.
        try:
            py = latest.to_pydatetime()
        except AttributeError:
            py = latest
        if isinstance(py, str):
            try:
                py = datetime.fromisoformat(py.replace("Z", "+00:00"))
            except (TypeError, ValueError):
                return False
        if not isinstance(py, datetime):
            return False
        if py.tzinfo is None:
            py = py.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - py) < self.cooldown

    def record_429(self, endpoint: str = "anthropic") -> None:
        """Record a 429 hit; subsequent in_cooldown() calls will see it."""
        tbl = self.store.db.open_table(RATELIMIT_TABLE)
        tbl.add(
            [
                {
                    "ts": datetime.now(timezone.utc),
                    "status_code": 429,
                    "endpoint": endpoint,
                }
            ]
        )


def should_call_llm(
    budget: BudgetLedger,
    rate: RateLimitLedger,
    llm_enabled: bool,
    has_api_key: bool,
    estimated_usd: float = 0.001,
) -> tuple[bool, str]:
    """7-step degradation ladder.

    Returns (ok, reason). reason is "ok" on success or a short diagnostic
    describing which ladder step blocked the call.

    Step ordering is fixed: changing the order without updating
    test_should_call_llm_ordering_* tests is a spec violation.
    """
    # Step 1: sleep.llm_enabled toggle.
    if not llm_enabled:
        return False, "sleep.llm_enabled=false"
    # Step 2: credentials.
    if not has_api_key:
        return False, "no api key"
    # Step 3 + 4: budget caps (daily, then monthly). can_spend tests both.
    ok, reason = budget.can_spend(estimated_usd)
    if not ok:
        return False, reason
    # Step 5: rate-limit cooldown.
    if rate.in_cooldown():
        return False, "ratelimit cooldown (last 429 < 15min)"
    # Steps 6-7 are caller's responsibility (retry + 429 recording).
    return True, "ok"
