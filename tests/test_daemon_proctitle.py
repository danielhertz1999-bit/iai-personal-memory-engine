"""Hermetic unit tests for daemon._set_process_title().

No daemon spawn — calls the helper directly.
No PII — uses generic names.
"""

import pytest
from setproctitle import getproctitle


# ---------------------------------------------------------------------------
# Happy-path: helper sets the OS-level process title
# ---------------------------------------------------------------------------

def test_set_process_title_sets_iai_lilli():
    """_set_process_title() sets the brand title while keeping the
    'iai_mcp.daemon' token so process-identification-by-cmdline still works."""
    from iai_mcp import daemon as _daemon

    original = getproctitle()
    try:
        _daemon._set_process_title()
        title = getproctitle()
        assert title == "iai lilli (iai_mcp.daemon)"
        # The module token MUST remain so the lockfile liveness check,
        # `daemon stop`, and doctor keep recognising the daemon by cmdline.
        assert "iai_mcp.daemon" in title
        assert title.startswith("iai lilli")
    finally:
        # Restore the test-runner's title so we don't pollute other tests.
        from setproctitle import setproctitle as _spt
        _spt(original)


# ---------------------------------------------------------------------------
# Fail-soft: an absent/broken setproctitle must never crash daemon boot
# ---------------------------------------------------------------------------

def test_set_process_title_fail_soft(monkeypatch):
    """A broken setproctitle must be swallowed — cosmetic failure must not propagate."""
    import setproctitle as _spt_mod
    from iai_mcp import daemon as _daemon

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated setproctitle breakage")

    monkeypatch.setattr(_spt_mod, "setproctitle", _raise)

    # Must return normally — must NOT raise.
    _daemon._set_process_title()
