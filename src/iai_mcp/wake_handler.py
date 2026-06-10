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

    def has_pending_wake(self) -> bool:
        try:
            return self._wake_signal_path.is_file()
        except OSError:
            return False
