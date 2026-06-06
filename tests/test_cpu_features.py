"""/ D- unit tests for iai_mcp.cpu_features.has_avx2().

The probe is a direct CPU-feature check that does NOT depend on lancedb
importing — so the doctor row can answer correctly even on a host where
`import lancedb` would SIGILL. Tests cover four platform branches plus
fallback:

  1. Linux x86 with AVX2 in /proc/cpuinfo flags -> True.
  2. Linux x86 without AVX2 -> False.
  3. macOS Intel with AVX2 in sysctl machdep.cpu.leaf7_features -> True.
  4. macOS ARM (M-series) -> True unconditionally (AVX2 N/A; NEON unrelated).
  5. Unknown platform -> True (defer to secondary defense in store.py).

All tests use `monkeypatch` exclusively. No real /proc reads, no real
subprocess.run, so the suite is deterministic across hosts.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LINUX_CPUINFO_WITH_AVX2 = """\
processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 142
model name	: Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz
stepping	: 10
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single pti ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp md_clear flush_l1d arch_capabilities
"""

_LINUX_CPUINFO_WITHOUT_AVX2 = """\
processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 122
model name	: Intel(R) Celeron(R) N4020 CPU @ 1.10GHz
stepping	: 8
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes xsave rdrand lahf_lm 3dnowprefetch cpuid_fault pti ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase smep erms mpx rdseed smap clflushopt sha_ni xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts md_clear arch_capabilities
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_linux_proc_cpuinfo_with_avx2_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux x86 host with `avx2` in /proc/cpuinfo flags row returns True."""
    import iai_mcp.cpu_features as cf
    from pathlib import Path

    monkeypatch.setattr(cf.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cf.platform, "machine", lambda: "x86_64")

    def _fake_read_text(self, *a, **kw):
        if str(self) == "/proc/cpuinfo":
            return _LINUX_CPUINFO_WITH_AVX2
        raise FileNotFoundError(str(self))

    def _fake_exists(self):
        return str(self) == "/proc/cpuinfo"

    monkeypatch.setattr(Path, "read_text", _fake_read_text)
    monkeypatch.setattr(Path, "exists", _fake_exists)

    assert cf.has_avx2() is True, (
        "Linux x86 cpuinfo with avx2 flag must return True"
    )


def test_linux_proc_cpuinfo_without_avx2_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux x86 host lacking `avx2` in /proc/cpuinfo flags row returns False.

    This is the Celeron N4020 Gemini Lake case — the SIGILL-on-import host
    that motivated. The row must FAIL so the doctor can surface an
    actionable message before `import lancedb` crashes the daemon.
    """
    import iai_mcp.cpu_features as cf
    from pathlib import Path

    monkeypatch.setattr(cf.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cf.platform, "machine", lambda: "x86_64")

    def _fake_read_text(self, *a, **kw):
        if str(self) == "/proc/cpuinfo":
            return _LINUX_CPUINFO_WITHOUT_AVX2
        raise FileNotFoundError(str(self))

    def _fake_exists(self):
        return str(self) == "/proc/cpuinfo"

    monkeypatch.setattr(Path, "read_text", _fake_read_text)
    monkeypatch.setattr(Path, "exists", _fake_exists)

    assert cf.has_avx2() is False, (
        "Linux x86 cpuinfo without avx2 flag must return False"
    )


def test_macos_intel_sysctl_with_avx2_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS Intel host with AVX2 in `sysctl -n machdep.cpu.leaf7_features`."""
    import iai_mcp.cpu_features as cf

    monkeypatch.setattr(cf.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cf.platform, "machine", lambda: "x86_64")

    fake_result = MagicMock()
    fake_result.stdout = "AVX2 SMEP BMI2 ERMS INVPCID RDSEED ADX SMAP CLFSOPT IPT MPX RDPID SGX"
    fake_result.returncode = 0
    monkeypatch.setattr(cf.subprocess, "run", lambda *a, **kw: fake_result)

    assert cf.has_avx2() is True, (
        "macOS Intel sysctl output containing AVX2 must return True"
    )


def test_macos_arm_returns_true_assume_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS ARM (M-series) returns True; AVX2 is N/A on ARM (NEON path).

    LanceDB ARM64 builds use NEON instructions; AVX2 absence is meaningless
    on this architecture. The doctor row must not falsely FAIL on M-series
    Macs — instead it reports PASS with "AVX2 available (or N/A)".
    """
    import iai_mcp.cpu_features as cf

    monkeypatch.setattr(cf.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cf.platform, "machine", lambda: "arm64")

    assert cf.has_avx2() is True, (
        "macOS ARM (M-series) must return True (AVX2 N/A on ARM)"
    )


def test_fallback_unknown_platform_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown platform (Windows, BSD, etc.) returns True (assume present).

    The cpu_features module is best-effort; on a platform we don't probe
    we defer to the secondary defense in `store.py` (try/except wrap of
    `import lancedb`) to catch any actual failure.
    """
    import iai_mcp.cpu_features as cf

    monkeypatch.setattr(cf.platform, "system", lambda: "Windows")
    monkeypatch.setattr(cf.platform, "machine", lambda: "AMD64")

    assert cf.has_avx2() is True, (
        "Unknown platform must default to True (assume AVX2 present)"
    )
