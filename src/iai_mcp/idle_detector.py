from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


_IOREG_BIN = "/usr/sbin/ioreg"

_PMSET_BIN = "/usr/bin/pmset"

_IOREG_TIMEOUT_SEC = 5

_PMSET_TIMEOUT_SEC = 10

_PMSET_TAIL_LINES = 200

_HID_IDLE_RE = re.compile(r'"HIDIdleTime"\s*=\s*(\d+)')

_PMSET_SLEEP_MARKERS = ("System Sleep", "Display is turned off")

_PMSET_DEFAULT_WINDOW_MIN = 5

_HID_IDLE_THRESHOLD_SEC = 30 * 60

_PMSET_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+([+-]\d{4})"
)

_PMSET_TS_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass
class IdleStatus:

    hid_idle_sec: int | None = None
    pmset_recent_sleep: bool = False
    available_signals: list[str] = field(default_factory=list)


class IdleDetector:


    def hid_idle_time_sec(self) -> int | None:
        try:
            result = subprocess.run(
                [_IOREG_BIN, "-c", "IOHIDSystem"],
                capture_output=True,
                text=True,
                timeout=_IOREG_TIMEOUT_SEC,
                check=False,
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return None
        except OSError:
            return None

        if result.returncode != 0:
            return None

        match = _HID_IDLE_RE.search(result.stdout or "")
        if match is None:
            return None
        try:
            ns = int(match.group(1))
        except ValueError:
            return None
        if ns < 0:
            return None
        return ns // 1_000_000_000


    def pmset_recent_sleep(
        self, window_min: int = _PMSET_DEFAULT_WINDOW_MIN
    ) -> bool:
        try:
            result = subprocess.run(
                [_PMSET_BIN, "-g", "log"],
                capture_output=True,
                text=True,
                timeout=_PMSET_TIMEOUT_SEC,
                check=False,
            )
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            return False
        except OSError:
            return False

        if result.returncode != 0:
            return False

        return self._scan_pmset_lines(result.stdout or "", window_min)

    @staticmethod
    def _scan_pmset_lines(stdout: str, window_min: int) -> bool:
        if window_min <= 0:
            return False
        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(minutes=window_min)

        lines = stdout.splitlines()
        tail = lines[-_PMSET_TAIL_LINES:] if len(lines) > _PMSET_TAIL_LINES else lines

        for line in tail:
            if not any(marker in line for marker in _PMSET_SLEEP_MARKERS):
                continue
            ts = _parse_pmset_timestamp(line)
            if ts is None:
                continue
            if ts >= cutoff:
                return True
        return False


    def sleep_eligible(self, heartbeat_idle_30min: bool) -> bool:
        if heartbeat_idle_30min:
            return True

        hid_idle = self.hid_idle_time_sec()
        if hid_idle is not None and hid_idle >= _HID_IDLE_THRESHOLD_SEC:
            return True

        return self.pmset_recent_sleep()


    def status(self) -> IdleStatus:
        hid_idle = self.hid_idle_time_sec()
        pmset_seen = self.pmset_recent_sleep()

        signals: list[str] = []
        if hid_idle is not None:
            signals.append("HIDIdleTime")
        if _pmset_responsive():
            signals.append("pmset")

        return IdleStatus(
            hid_idle_sec=hid_idle,
            pmset_recent_sleep=pmset_seen,
            available_signals=signals,
        )


def _parse_pmset_timestamp(line: str) -> datetime | None:
    m = _PMSET_TS_RE.match(line)
    if m is None:
        return None
    ts_str, offset_str = m.group(1), m.group(2)
    try:
        naive = datetime.strptime(ts_str, _PMSET_TS_FMT)
    except ValueError:
        return None
    sign = 1 if offset_str[0] == "+" else -1
    try:
        hours = int(offset_str[1:3])
        minutes = int(offset_str[3:5])
    except ValueError:
        return None
    offset = timedelta(hours=hours, minutes=minutes) * sign
    return (naive - offset).replace(tzinfo=timezone.utc)


def _pmset_responsive() -> bool:
    try:
        result = subprocess.run(
            [_PMSET_BIN, "-g"],
            capture_output=True,
            text=True,
            timeout=_PMSET_TIMEOUT_SEC,
            check=False,
        )
    except FileNotFoundError:
        return False
    except subprocess.TimeoutExpired:
        return False
    except OSError:
        return False
    return result.returncode == 0
