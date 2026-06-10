from __future__ import annotations

import threading
from pathlib import Path

import pytest

from iai_mcp.wake_handler import WakeHandler

@pytest.fixture
def wake_signal_path(tmp_path: Path) -> Path:
    return tmp_path / "wake.signal"

def _write_signal(
    path: Path,
    payload: str = '{"requested_at":"2026-05-02T15:00:00Z"}',
    tmp_suffix: str = ".tmp",
) -> None:
    tmp = path.with_suffix(path.suffix + tmp_suffix)
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)

def test_consume_wake_signal_when_present_deletes_and_returns_true(
    wake_signal_path: Path,
) -> None:
    _write_signal(wake_signal_path)
    assert wake_signal_path.is_file()

    handler = WakeHandler(wake_signal_path)
    assert handler.consume_wake_signal() is True
    assert not wake_signal_path.exists()

def test_consume_wake_signal_when_absent_returns_false(
    wake_signal_path: Path,
) -> None:
    assert not wake_signal_path.exists()

    handler = WakeHandler(wake_signal_path)
    assert handler.consume_wake_signal() is False

def test_consume_wake_signal_idempotent(wake_signal_path: Path) -> None:
    _write_signal(wake_signal_path)

    handler = WakeHandler(wake_signal_path)
    assert handler.consume_wake_signal() is True
    assert handler.consume_wake_signal() is False
    assert handler.consume_wake_signal() is False

def test_has_pending_wake_read_only(wake_signal_path: Path) -> None:
    _write_signal(wake_signal_path)

    handler = WakeHandler(wake_signal_path)
    assert handler.has_pending_wake() is True
    assert wake_signal_path.is_file()
    assert handler.has_pending_wake() is True
    assert wake_signal_path.is_file()
    assert handler.consume_wake_signal() is True
    assert handler.has_pending_wake() is False

def test_consume_atomic_no_race(wake_signal_path: Path) -> None:
    handler = WakeHandler(wake_signal_path)
    consumed_truthy_count = 0
    errors: list[BaseException] = []

    stop_writers = threading.Event()

    def writer_loop(writer_id: int) -> None:
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

    assert errors == []
    assert consumed_truthy_count >= 1
