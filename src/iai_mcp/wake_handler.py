"""Phase 10.5 L5 — daemon-side ``wake.signal`` consumer.

The TypeScript MCP wrapper (``mcp-wrapper/src/lifecycle.ts``) writes a
small marker file at ``~/.iai-mcp/wake.signal`` when:

* the wrapper boots and the daemon socket is unreachable, AND
* the platform is NOT macOS (so the wrapper cannot ``launchctl kickstart``
  the daemon directly), OR
* a kickstart attempt failed and the wrapper has fallen back to the
  cross-platform signal file path.

This module owns the daemon-side consume side of that signal. It is
**deliberately tiny**: read-and-delete on cold start, idempotent,
race-safe with a wrapper that may be writing a fresh signal mid-consume.
The wrapper's atomic-rename write semantics guarantee that ``read_text``
either sees the file fully or not at all; we never have to defend
against a torn read of the signal payload itself.

The placeholder integration in :func:`iai_mcp.daemon.main` calls
:meth:`WakeHandler.consume_wake_signal` once during startup. Phase 10.6
will dispatch the result into the lifecycle state machine's
``WAKE_SIGNAL`` event channel — until then this module is a write-once
hook so the wrapper's L5 path has somewhere to write to.

Constraints (carried from / 10.5 hard-rules):

- stdlib only — no third-party imports.
- macOS-first; non-macOS callers use this same path.
- Idempotent: a second ``consume_wake_signal()`` call returns ``False``
  cleanly without raising.
- Race-safe: a ``FileNotFoundError`` between the existence check and the
  unlink (concurrent wrapper writes a fresh signal that gets consumed
  before we re-stat) is swallowed and reported as "no pending wake".

Validates: WAKE-03, (Python-side consume half).
"""
from __future__ import annotations

from pathlib import Path


__all__ = ["WakeHandler"]


class WakeHandler:
    """Consume ``wake.signal`` markers written by the MCP wrapper.

    The handler holds the absolute path to the signal file. It does NOT
    create the directory; the wrapper is responsible for ensuring
    ``~/.iai-mcp/`` exists when it writes the signal. The daemon already
    creates this directory at boot via ``ProcessLock`` / ``MemoryStore``
    so by the time this handler is consulted the parent dir is present.
    """

    def __init__(self, wake_signal_path: Path) -> None:
        """Store the absolute path to the signal file.

        Args:
            wake_signal_path: Absolute path to ``wake.signal``. Caller is
                responsible for ``Path.expanduser()`` if a ``~`` was
                present in the input — production callers pass an
                already-expanded path.
        """
        self._wake_signal_path = wake_signal_path

    def consume_wake_signal(self) -> bool:
        """Atomically delete the signal file if present and return whether one existed.

        Returns:
            ``True`` if a signal was present and has been consumed, else
            ``False``. Idempotent — a second call after the first
            ``True`` returns ``False`` (file already gone).

        Race semantics:
            ``Path.unlink(missing_ok=False)`` is the atomic delete. If
            two consumers race (this should not happen in practice; the
            daemon is a singleton via ``ProcessLock``) the loser sees
            ``FileNotFoundError`` which we swallow and report as
            "no pending wake".
        """
        try:
            self._wake_signal_path.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            # Permission / FS error — surface as "no pending wake" rather
            # than raising, since the wake path must NEVER block daemon
            # boot. The wrapper will retry on its next boot if it still
            # cares.
            return False
        return True

    def has_pending_wake(self) -> bool:
        """Read-only check: does a wake signal currently exist?

        Used by the doctor row to surface pending-wake state without
        consuming it. Calling ``consume_wake_signal()`` after this method
        will return ``True`` iff this method returned ``True`` and no
        other consumer raced in between.
        """
        try:
            return self._wake_signal_path.is_file()
        except OSError:
            return False
