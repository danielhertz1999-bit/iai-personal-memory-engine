"""Phase 10.5 — tests for :class:`iai_mcp.wake_handler.WakeHandler`.

Five-test matrix from CONTEXT 10.5:

- ``test_consume_wake_signal_when_present_deletes_and_returns_true``.
- ``test_consume_wake_signal_when_absent_returns_false``.
- ``test_consume_wake_signal_idempotent`` — second call returns False.
- ``test_has_pending_wake_read_only`` — does not delete the file.
- ``test_consume_atomic_no_race`` — concurrent wrapper-style writers and
  a single daemon-style consumer; no exception, end state coherent.

Tests use ``tmp_path`` for the signal file (no real ``~/.iai-mcp/``
involvement) so they are hermetic across machines and parallel runs.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from iai_mcp.wake_handler import WakeHandler


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def wake_signal_path(tmp_path: Path) -> Path:
    """Path to a wake.signal file under tmp_path (file does NOT exist yet)."""
    return tmp_path / "wake.signal"


def _write_signal(
    path: Path,
    payload: str = '{"requested_at":"2026-05-02T15:00:00Z"}',
    tmp_suffix: str = ".tmp",
) -> None:
    """Atomic write helper mirroring the wrapper's temp + rename semantics.

    The wrapper writes via ``fs.promises.writeFile(tmp)`` then
    ``fs.promises.rename(tmp, final)``; on POSIX that rename is atomic so
    the consumer either sees the file fully or not at all. The Python
    test mirrors this with ``Path.write_text`` followed by ``Path.rename``.

    The ``tmp_suffix`` parameter lets concurrent writer threads use
    distinct tmp filenames (mirroring the wrapper's per-pid-uuid scheme)
    so they don't collide on the staging path.
    """
    tmp = path.with_suffix(path.suffix + tmp_suffix)
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- tests


def test_consume_wake_signal_when_present_deletes_and_returns_true(
    wake_signal_path: Path,
) -> None:
    _write_signal(wake_signal_path)
    assert wake_signal_path.is_file()  # precondition

    handler = WakeHandler(wake_signal_path)
    assert handler.consume_wake_signal() is True
    assert not wake_signal_path.exists()


def test_consume_wake_signal_when_absent_returns_false(
    wake_signal_path: Path,
) -> None:
    assert not wake_signal_path.exists()  # precondition

    handler = WakeHandler(wake_signal_path)
    assert handler.consume_wake_signal() is False


def test_consume_wake_signal_idempotent(wake_signal_path: Path) -> None:
    _write_signal(wake_signal_path)

    handler = WakeHandler(wake_signal_path)
    assert handler.consume_wake_signal() is True
    # Second call must NOT raise — file already gone.
    assert handler.consume_wake_signal() is False
    # And once more for good measure: still False, still no exception.
    assert handler.consume_wake_signal() is False


def test_has_pending_wake_read_only(wake_signal_path: Path) -> None:
    _write_signal(wake_signal_path)

    handler = WakeHandler(wake_signal_path)
    # Read-only check — must NOT delete.
    assert handler.has_pending_wake() is True
    assert wake_signal_path.is_file()
    # Multiple reads still don't delete.
    assert handler.has_pending_wake() is True
    assert wake_signal_path.is_file()
    # Now consume; subsequent has_pending_wake reports False.
    assert handler.consume_wake_signal() is True
    assert handler.has_pending_wake() is False


def test_consume_atomic_no_race(wake_signal_path: Path) -> None:
    """Concurrent wrapper-style writers + one daemon-style consumer.

    Reproduces the wake-on-boot interleaving where the daemon is starting
    up while one or more wrappers are still writing fresh signals. The
    consumer must never raise — it either sees the file (returns True
    and deletes) or doesn't (returns False).
    """
    handler = WakeHandler(wake_signal_path)
    consumed_truthy_count = 0
    errors: list[BaseException] = []

    stop_writers = threading.Event()

    def writer_loop(writer_id: int) -> None:
        # Hammer atomic-rename writes for ~50 ms; ample time for the
        # consumer thread to fire several reads. Each writer uses a
        # unique tmp suffix so concurrent writers do NOT collide on the
        # staging path (mirrors the wrapper's per-pid-uuid scheme).
        suffix = f".tmp.w{writer_id}"
        try:
            for _ in range(200):
                if stop_writers.is_set():
                    return
                _write_signal(wake_signal_path, tmp_suffix=suffix)
        except BaseException as exc:  # pragma: no cover -- defensive
            errors.append(exc)

    def consumer_loop() -> None:
        nonlocal consumed_truthy_count
        try:
            for _ in range(200):
                if handler.consume_wake_signal():
                    consumed_truthy_count += 1
        except BaseException as exc:
            errors.append(exc)

    writers = [
        threading.Thread(target=writer_loop, args=(i,)) for i in range(3)
    ]
    consumer = threading.Thread(target=consumer_loop)
    for w in writers:
        w.start()
    consumer.start()
    consumer.join(timeout=10.0)
    stop_writers.set()
    for w in writers:
        w.join(timeout=10.0)

    # No thread raised. The consumer saw the signal at least once
    # (writers wrote 600 times). Final filesystem state is allowed to be
    # either present (a writer ran last) or absent (consumer ran last) —
    # both are valid steady states.
    assert errors == []
    assert consumed_truthy_count >= 1
