from __future__ import annotations

from pathlib import Path


__all__ = ["WakeHandler"]


class WakeHandler:

    def __init__(self, wake_signal_path: Path) -> None:
        self._wake_signal_path = wake_signal_path

    def consume_wake_signal(self) -> bool:
        try:
            self._wake_signal_path.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            return False
        return True

    def signal_wake(self) -> bool:
        """Create the wake signal so the next daemon boot enters WAKE.

        Symmetric counterpart to ``consume_wake_signal``. Whoever brings the
        daemon up from a hibernated (process-exited) state — the CLI
        start/install path, or the per-turn capture hook — must create this
        signal *before* the kickstart, so the booting daemon transitions
        HIBERNATION -> WAKE and serves its socket, instead of re-reading the
        persisted HIBERNATION state and immediately hibernate-exiting (which
        closes the socket and leaves recall unserved). Idempotent.
        """
        try:
            self._wake_signal_path.parent.mkdir(parents=True, exist_ok=True)
            self._wake_signal_path.touch()
        except OSError:
            return False
        return True

    def has_pending_wake(self) -> bool:
        try:
            return self._wake_signal_path.is_file()
        except OSError:
            return False
