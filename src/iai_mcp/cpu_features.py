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
    system = platform.system()
    machine = platform.machine()

    if system == "Linux" and machine in _LINUX_X86_MACHINES:
        return _probe_linux_proc_cpuinfo()
    if system == "Darwin" and machine in {"x86_64", "amd64", "i686", "i386"}:
        return _probe_macos_intel_sysctl()
    if system == "Darwin" and machine in _DARWIN_ARM_MACHINES:
        return True
    return True


def _probe_linux_proc_cpuinfo() -> bool:
    path = Path(_CPUINFO_PATH)
    try:
        if not path.exists():
            return True
        text = path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError) as exc:
        logger.debug("cpu_features: /proc/cpuinfo read failed: %s", exc)
        return True

    for line in text.splitlines():
        if line.startswith(("flags", "Features")):
            _, _, value = line.partition(":")
            tokens = value.split()
            return "avx2" in tokens
    return True


def _probe_macos_intel_sysctl() -> bool:
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
        return True
    return "AVX2" in result.stdout.split()
