from __future__ import annotations

from datetime import datetime, timedelta, timezone

from iai_mcp.store import BUDGET_TABLE, RATELIMIT_TABLE, MemoryStore


BUDGET_DAILY_USD_DEFAULT = 0.10
BUDGET_MONTHLY_USD_DEFAULT = 3.00
RATELIMIT_COOLDOWN_MIN = 15


class BudgetLedger:

    def __init__(
        self,
        store: MemoryStore,
        daily_usd_cap: float = BUDGET_DAILY_USD_DEFAULT,
        monthly_usd_cap: float = BUDGET_MONTHLY_USD_DEFAULT,
    ) -> None:
        self.store = store
        self.daily_cap = float(daily_usd_cap)
        self.monthly_cap = float(monthly_usd_cap)


    def _today_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _this_month(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")


    def daily_used(self) -> float:
        tbl = self.store.db.open_table(BUDGET_TABLE)
        df = tbl.to_pandas()
        if df.empty:
            return 0.0
        today = df[df["date"] == self._today_utc()]
        return float(today["usd_spent"].sum()) if not today.empty else 0.0

    def monthly_used(self) -> float:
        tbl = self.store.db.open_table(BUDGET_TABLE)
        df = tbl.to_pandas()
        if df.empty:
            return 0.0
        mo = df[df["date"].str.startswith(self._this_month())]
        return float(mo["usd_spent"].sum()) if not mo.empty else 0.0

    def can_spend(self, usd: float) -> tuple[bool, str]:
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


    def record_spend(self, usd: float, kind: str = "llm") -> None:
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

    def __init__(
        self,
        store: MemoryStore,
        cooldown_minutes: int = RATELIMIT_COOLDOWN_MIN,
    ) -> None:
        self.store = store
        self.cooldown = timedelta(minutes=int(cooldown_minutes))

    def in_cooldown(self) -> bool:
        tbl = self.store.db.open_table(RATELIMIT_TABLE)
        df = tbl.to_pandas()
        if df.empty:
            return False
        latest = df["ts"].max()
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
    if not llm_enabled:
        return False, "sleep.llm_enabled=false"
    if not has_api_key:
        return False, "no api key"
    ok, reason = budget.can_spend(estimated_usd)
    if not ok:
        return False, reason
    if rate.in_cooldown():
        return False, "ratelimit cooldown (last 429 < 15min)"
    return True, "ok"
