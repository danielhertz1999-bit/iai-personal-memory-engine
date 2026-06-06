"""Hardware-aware idle detector for the wake/sleep cycle.

Combines three hardware-grounded signals into a single ``sleep_eligible``
predicate the daemon's state machine consumes when deciding whether to
transition into a sleep cycle:

1. **Heartbeat-idle (30 min):** no FRESH wrapper heartbeats in the last 30
   minutes — supplied externally by ``HeartbeatScanner.heartbeat_idle_30min``.
2. **HIDIdleTime:** ``ioreg -c IOHIDSystem`` exposes nanoseconds since the
   last user input event. Convert ns→sec, compare against ``≥ 30 min``.
3. **pmset events:** macOS power-manager log entries for ``System Sleep`` or
   ``Display is turned off`` within the last ``window_min`` minutes.

``sleep_eligible`` is the **disjunction** of the three: any one signal is
sufficient. There is no wall-clock fallback, only hardware-grounded evidence
of inactivity.

Constraints:
- ALL subprocess calls use array form ``[bin, arg,...]`` with
  ``shell=False`` and a finite ``timeout``. NEVER ``shell=True``. NEVER
  f-string interpolation into command strings.
- Idle CPU near zero — this module is invoked on lifecycle TICK (every 30 s),
  not faster. ``pmset -g log`` can be slow (≈1 s) so we tail the last 200
  lines of output rather than re-parsing the entire log.
- macOS-only: ``ioreg`` and ``pmset`` are macOS binaries. On non-macOS the
  detector returns ``None`` / ``False`` gracefully.
- No new third-party dependencies — stdlib only.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


# Module-level constants -------------------------------------------------------

#: Absolute path to the macOS ``ioreg`` binary. Hard-coded to avoid PATH-based
#: hijacks (a planted ``ioreg`` in the user's PATH could feed us spoofed
#: HIDIdleTime values that would falsely keep the daemon awake or asleep).
_IOREG_BIN = "/usr/sbin/ioreg"

#: Absolute path to the macOS ``pmset`` binary. Same PATH-hijack rationale.
_PMSET_BIN = "/usr/bin/pmset"

#: Subprocess timeout for ``ioreg`` (seconds). The call is a straight kernel
#: registry dump and returns within ~50 ms on a healthy system; a 5 s ceiling
#: keeps a hung kernel-extension probe from blocking the lifecycle TICK.
_IOREG_TIMEOUT_SEC = 5

#: Subprocess timeout for ``pmset -g log``. ``pmset`` walks the system power
#: log and on a long-uptime machine can take ~1 s; 10 s ceiling.
_PMSET_TIMEOUT_SEC = 10

#: Number of trailing lines to scan from ``pmset -g log``. The log is
#: append-only and ordered by time, so the most-recent events are at the end.
#: 200 lines covers ~last 24 h on a typical workstation; the window check
#: filters by timestamp anyway.
_PMSET_TAIL_LINES = 200

#: Regex for the HIDIdleTime line. Format: ``"HIDIdleTime" = 12345678901``.
_HID_IDLE_RE = re.compile(r'"HIDIdleTime"\s*=\s*(\d+)')

#: Substrings that indicate a sleep / display-off event in pmset log output.
_PMSET_SLEEP_MARKERS = ("System Sleep", "Display is turned off")

#: Default window for ``pmset_recent_sleep`` (minutes): "in last 5 min".
_PMSET_DEFAULT_WINDOW_MIN = 5

#: Hardware-idle threshold for the disjunction in ``sleep_eligible`` —
#: ``HIDIdleTime ≥ 30 min`` is sufficient evidence of user inactivity.
_HID_IDLE_THRESHOLD_SEC = 30 * 60

#: Regex anchoring a pmset log line's leading timestamp. The format is
#: ``YYYY-MM-DD HH:MM:SS ±HHMM`` (e.g. ``2026-05-02 15:00:00 -0400``).
_PMSET_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+([+-]\d{4})"
)

#: Strptime pattern for the timestamp captured by ``_PMSET_TS_RE``.
_PMSET_TS_FMT = "%Y-%m-%d %H:%M:%S"


# Public dataclass -------------------------------------------------------------


@dataclass
class IdleStatus:
    """Snapshot of the L6 detector for the doctor row (n) display.

    Attributes:
        hid_idle_sec: Seconds since last user input, or ``None`` if ``ioreg``
            is unavailable or its output cannot be parsed.
        pmset_recent_sleep: True iff a System / Display Sleep event was seen
            within the configured window. False on parse failure or missing
            tool — biased toward "no recent sleep" so the doctor row reports
            a clean state rather than a false-positive sleep.
        available_signals: Subset of ``["HIDIdleTime", "pmset"]`` listing
            which hardware sources actually returned data on this probe.
            Empty list means we have no hardware grounding right now and
            the L6 disjunction must rely on the heartbeat-idle signal.
    """

    hid_idle_sec: int | None = None
    pmset_recent_sleep: bool = False
    available_signals: list[str] = field(default_factory=list)


# IdleDetector -----------------------------------------------------------------


class IdleDetector:
    """Hardware-grounded idle probe for the daemon state machine.

    Standalone module — wires this into the daemon's TICK so
    ``sleep_eligible`` gates the BEDTIME transition. Each public method
    can be called independently; ``status()`` aggregates them for the
    doctor row.
    """

    # ---- HIDIdleTime via ioreg --------------------------------------

    def hid_idle_time_sec(self) -> int | None:
        """Return seconds since last HID input, or ``None`` on any failure.

        Spawns ``/usr/sbin/ioreg -c IOHIDSystem`` (array form, ``shell=False``,
        5 s timeout, ``check=False``). Parses the first ``"HIDIdleTime" =
        <ns>`` match and integer-divides by 1e9. Any error path — missing
        tool, non-zero exit, parse miss, timeout — collapses to ``None`` so
        the caller treats the signal as absent rather than zero (zero would
        falsely imply "active right now").
        """
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

    # ---- pmset event detection --------------------------------------

    def pmset_recent_sleep(
        self, window_min: int = _PMSET_DEFAULT_WINDOW_MIN
    ) -> bool:
        """True iff a System/Display Sleep event was recorded in the window.

        Spawns ``/usr/bin/pmset -g log`` (array form, ``shell=False``, 10 s
        timeout, ``check=False``). Tails the last ``_PMSET_TAIL_LINES``
        lines of stdout, parses the leading timestamp, and reports True if
        any line within ``window_min`` minutes of "now" contains one of the
        ``_PMSET_SLEEP_MARKERS`` substrings.

        Failure modes (missing tool, non-zero exit, no parseable lines) all
        collapse to ``False`` — biased toward "no recent sleep" so an
        unavailable signal does not trigger the L6 disjunction on its own.
        """
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
        """Helper — pure-function scan over pmset log text.

        Split out for unit testing without subprocess mocking. Walks the
        last ``_PMSET_TAIL_LINES`` lines, returns True at the first match
        within the window. Parse failures on individual lines are skipped.
        """
        if window_min <= 0:
            return False
        # Build a UTC "now" once; pmset timestamps come with explicit ±HHMM
        # offsets so we convert each parsed timestamp to UTC for comparison.
        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(minutes=window_min)

        # Tail the last N lines so we don't re-scan a multi-megabyte log.
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

    # ---- Disjunction predicate consumed by the state machine --------

    def sleep_eligible(self, heartbeat_idle_30min: bool) -> bool:
        """L6 disjunction: any of three hardware-grounded signals is sufficient.

        Args:
            heartbeat_idle_30min: True iff no FRESH wrapper heartbeat in the
                last 30 minutes (supplied by
                ``HeartbeatScanner.heartbeat_idle_30min``).

        Returns:
            ``heartbeat_idle_30min OR (hid_idle_time_sec ≥ 30 min) OR
            pmset_recent_sleep()``. Short-circuits on the first True so a
            heartbeat-idle session does not pay for ``ioreg`` + ``pmset``
            spawns it does not need.
        """
        if heartbeat_idle_30min:
            return True

        hid_idle = self.hid_idle_time_sec()
        if hid_idle is not None and hid_idle >= _HID_IDLE_THRESHOLD_SEC:
            return True

        return self.pmset_recent_sleep()

    # ---- Aggregated snapshot for doctor row (n) ---------------------

    def status(self) -> IdleStatus:
        """Return an ``IdleStatus`` snapshot for the doctor checklist.

        Calls both probes regardless of disjunction short-circuit so the
        doctor surface always reflects the *actual* per-signal availability
        (a doctor that hides ``pmset`` whenever ``HIDIdleTime`` already
        triggers would not help the user diagnose a missing pmset log).
        """
        hid_idle = self.hid_idle_time_sec()
        pmset_seen = self.pmset_recent_sleep()

        signals: list[str] = []
        if hid_idle is not None:
            signals.append("HIDIdleTime")
        # pmset_recent_sleep returning False does not imply pmset is missing
        # — it only means no event in the window. We can't reliably tell
        # "tool present but quiet" from "tool absent" without re-spawning,
        # so we bias the doctor display toward listing pmset as available
        # whenever the call succeeded (i.e. did not raise / non-zero-exit).
        if _pmset_responsive():
            signals.append("pmset")

        return IdleStatus(
            hid_idle_sec=hid_idle,
            pmset_recent_sleep=pmset_seen,
            available_signals=signals,
        )


# Module-private helpers -------------------------------------------------------


def _parse_pmset_timestamp(line: str) -> datetime | None:
    """Return the leading timestamp of a pmset log line as UTC, or None.

    Matches ``YYYY-MM-DD HH:MM:SS ±HHMM`` at the start of the line. The
    ``±HHMM`` offset is parsed manually because ``%z`` on older Python
    builds is finicky with shorthand offsets — we apply the offset to a
    naive datetime and tag it as UTC.
    """
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
    # Treat naive timestamp as in the offset's local zone, then convert to
    # UTC by subtracting the offset.
    return (naive - offset).replace(tzinfo=timezone.utc)


def _pmset_responsive() -> bool:
    """Probe whether ``/usr/bin/pmset`` exists and exits 0 for a trivial call.

    Used by ``IdleDetector.status`` to populate ``available_signals``
    without inferring availability from the (legitimate) "no recent sleep"
    output. ``pmset -g`` (no subcommand) prints the current power state
    and exits 0 quickly; missing-binary or non-zero-exit ⇒ unavailable.
    """
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
