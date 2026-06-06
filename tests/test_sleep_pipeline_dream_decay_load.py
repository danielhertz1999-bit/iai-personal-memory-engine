"""RED witness — DREAM_DECAY step must not raise AttributeError.

Bug: ``sleep_pipeline.py`` calls ``UserModel.load(self._store)`` but
``UserModel`` is a ``@dataclass`` with no ``load`` classmethod. The canonical
loader is the module-level ``user_model.load()`` (no arguments).

These tests fail on the unfixed code with:
    AttributeError: type object 'UserModel' has no attribute 'load'

They pass after sleep_pipeline.py imports the canonical module-level
``load`` and adds ``AttributeError`` to the defensive except tuple.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import iai_mcp.sleep as sleep_module
import iai_mcp.user_model as user_model_module
from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.sleep_pipeline import SleepPipeline
from iai_mcp.user_model import UserModel


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    """Isolated lifecycle_state.json path inside tmp_path."""
    return tmp_path / "lifecycle_state.json"


@pytest.fixture
def event_log(tmp_path: Path) -> LifecycleEventLog:
    """LifecycleEventLog rooted at an isolated tmp directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return LifecycleEventLog(log_dir=log_dir)


@pytest.fixture
def pipeline(state_path: Path, event_log: LifecycleEventLog) -> SleepPipeline:
    """SleepPipeline instance with stub store and isolated paths.

    ``store`` is None because ``_step_dream_decay`` only uses it as the
    first positional arg to ``_decay_edges``; the test patches
    ``_decay_edges`` to a spy so the real store path is never touched.
    """
    return SleepPipeline(
        store=None,
        lifecycle_state_path=state_path,
        event_log=event_log,
        quarantine_ttl_hours=24.0,
    )


def test_dream_decay_step_does_not_raise_attribute_error_on_user_model_load(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct invocation of _step_dream_decay must return cleanly.

    On the unfixed code the call ``UserModel.load(self._store)`` raises
    ``AttributeError`` which the except tuple does NOT catch, so the
    error propagates and the entire DREAM_DECAY step fails. Once the
    fix lands (use module-level ``load()`` + add AttributeError to the
    except), the step returns ``(True, {...})``.
    """
    # Patch the real _decay_edges to a no-op returning a dict shaped like
    # the production return so the assertion on the payload still holds.
    def _fake_decay(store: Any, plasticity_gain: float = 1.0) -> dict[str, int]:
        return {"decayed": 0, "pruned": 0}

    monkeypatch.setattr(sleep_module, "_decay_edges", _fake_decay)

    completed, payload = pipeline._step_dream_decay(interrupt_check=None)

    assert completed is True
    assert isinstance(payload, dict)
    assert payload.get("decayed") == 0
    assert payload.get("pruned") == 0


def test_dream_decay_step_reads_plasticity_gain_from_user_model(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plasticity_gain forwarded to _decay_edges comes from user_model.load().

    ``save()`` in user_model.py does not currently round-trip the
    ``plasticity_gain`` field (it is in the dataclass but not in the JSON
    payload), so a filesystem fixture round-trip would always observe
    ``1.0``. Patch the canonical loader directly to inject a non-default
    value; this is the path ``sleep_pipeline.py`` actually takes after the
    fix (``from iai_mcp.user_model import load``).
    """
    captured: dict[str, float] = {}

    def _fake_decay(store: Any, plasticity_gain: float = 1.0) -> dict[str, int]:
        captured["plasticity_gain"] = float(plasticity_gain)
        return {"decayed": 0, "pruned": 0}

    def _fake_load() -> UserModel:
        return UserModel(plasticity_gain=0.5)

    monkeypatch.setattr(sleep_module, "_decay_edges", _fake_decay)
    monkeypatch.setattr(user_model_module, "load", _fake_load)

    completed, _payload = pipeline._step_dream_decay(interrupt_check=None)

    assert completed is True
    assert captured.get("plasticity_gain") == pytest.approx(0.5)
