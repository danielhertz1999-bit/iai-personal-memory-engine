"""Direct CPU-feature probe for the AVX2 doctor row.

Independent of lancedb so the doctor can answer correctly even on a host
where ``import lancedb`` would deliver SIGILL on an illegal AVX2 opcode.
Python's ``try/except`` cannot catch SIGILL (it is a CPU-level signal that
bypasses the interpreter frame); the probe therefore consults OS-trusted
sources directly.

Probe order:
  - Linux: parse ``/proc/cpuinfo`` ``flags`` row for the token ``avx2``.
  - macOS Intel: ``sysctl -n machdep.cpu.leaf7_features`` for ``AVX2``.
  - macOS ARM (M-series): AVX2 is N/A; lancedb ARM64 builds use NEON, a
    separate failure surface. Return True.
  - Unknown platform: return True (assume present; the secondary
    ``import lancedb`` try/except in ``store.py`` catches Python-level
    failures if any).

A probe error returns True so the row defaults to non-blocking. A FAIL is
reserved for the case where we have a positive signal that AVX2 is missing
on a host where it matters (Linux x86 + macOS Intel).
"""
# Independent of lancedb so the doctor row can answer correctly even on a
# host where `import lancedb` would SIGILL on an illegal opcode.
from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


_SYSCTL_TIMEOUT_SEC = 2.0
_CPUINFO_PATH = "/proc/cpuinfo"
_LINUX_X86_MACHINES = frozenset({"x86_64", "amd64", "i686", "i386"})
_DARWIN_ARM_MACHINES = frozenset({"arm64", "aarch64"})


def has_avx2() -> bool:
    """Return True iff the host CPU advertises AVX2 (or AVX2 is N/A here).

    Never raises -- a probe error returns True so the row defaults to
    non-blocking. The doctor row is the surface that turns False into a
    FAIL with an actionable message; this helper only answers the
    underlying question "is AVX2 demonstrably absent on a host where it
    matters?".
    """
    system = platform.system()
    machine = platform.machine()

    if system == "Linux" and machine in _LINUX_X86_MACHINES:
        return _probe_linux_proc_cpuinfo()
    if system == "Darwin" and machine in {"x86_64", "amd64", "i686", "i386"}:
        return _probe_macos_intel_sysctl()
    if system == "Darwin" and machine in _DARWIN_ARM_MACHINES:
        # AVX2 not applicable on ARM; NEON path is unrelated.
        return True
    # Unknown platform (Windows, BSD, etc.): defer to store.py secondary
    # defense -- assume present, let the Python-level guard catch any
    # real failure.
    return True


def _probe_linux_proc_cpuinfo() -> bool:
    """Read /proc/cpuinfo and check the first ``flags`` row for ``avx2``."""
    path = Path(_CPUINFO_PATH)
    try:
        if not path.exists():
            return True  # file absent -> assume present; defer to secondary defense
        text = path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError) as exc:
        logger.debug("cpu_features: /proc/cpuinfo read failed: %s", exc)
        return True

    for line in text.splitlines():
        # The `flags` row appears once per logical CPU; the first row is
        # representative -- AVX2 cannot be enabled on some cores but not
        # others on a single physical socket.
        if line.startswith(("flags", "Features")):
            _, _, value = line.partition(":")
            tokens = value.split()
            return "avx2" in tokens
    # No flags row found -> assume present (defer to secondary defense).
    return True


def _probe_macos_intel_sysctl() -> bool:
    """Run ``sysctl -n machdep.cpu.leaf7_features`` and look for AVX2."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.leaf7_features"],
            capture_output=True,
            text=True,
            timeout=_SYSCTL_TIMEOUT_SEC,
            check=False,
        )
    except (
        OSError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as exc:
        logger.debug("cpu_features: sysctl probe failed: %s", exc)
        return True

    if result.returncode != 0:
        return True  # sysctl said no -- defer to secondary defense
    return "AVX2" in result.stdout.split()
