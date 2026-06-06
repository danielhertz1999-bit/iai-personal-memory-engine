"""Regression tests for StcConfig + _load_stc_config.

Pins the acceptance contracts (env-var typed bundle + CALL-ON-DEMAND fail-loud)
for the Synaptic Tagging-and-Capture (STC) temporal-association
config helper. This produces ONLY the dataclass + loader; later code
consumes it from PeriEventBuffer.trigger_stc and the daemon
singleton-wire site.

    - StcConfig is a frozen dataclass with 4 typed fields in the
            declared order (peri_event_buffer_size, peri_event_window_sec,
            strong_event_types, dry_run); mutation raises FrozenInstanceError.

    - _load_stc_config() reads the 4 IAI_MCP_* env vars on every
            invocation (no module cache), validates ranges, and
            raises ValueError naming the offending var on malformed input.
            Pytest-aware dry_run default fires when PYTEST_CURRENT_TEST
            is set and no explicit value is supplied.

Fixtures are inline.
No MemoryStore needed -- this is a pure-config test module.
"""
# Standard-library imports first so optional iai_mcp.* imports fail loud
# with a clear ImportError if the package layout changes.
from __future__ import annotations

import dataclasses

import pytest

from iai_mcp.daemon import (
    StcConfig,
    _load_stc_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Autouse fixture: wipe every IAI_MCP_* + PYTEST_CURRENT_TEST env var that
# the loader reads so each test starts from the defaults. Tests
# that need a specific value re-set after this fixture.
@pytest.fixture(autouse=True)
def _isolate_stc_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "IAI_MCP_PERI_EVENT_BUFFER_SIZE",
        "IAI_MCP_PERI_EVENT_WINDOW_SEC",
        "IAI_MCP_STC_STRONG_EVENT_TYPES",
        "IAI_MCP_STC_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# StcConfig dataclass shape
# ---------------------------------------------------------------------------


def test_T1_stcconfig_is_frozen_dataclass_with_four_typed_fields() -> None:
    """StcConfig is a frozen @dataclass with exactly 4 fields in
    declared order (peri_event_buffer_size:int, peri_event_window_sec:int,
    strong_event_types:frozenset[str], dry_run:bool)."""
    assert dataclasses.is_dataclass(StcConfig)
    field_names = [f.name for f in dataclasses.fields(StcConfig)]
    assert field_names == [
        "peri_event_buffer_size",
        "peri_event_window_sec",
        "strong_event_types",
        "dry_run",
    ], field_names

    cfg = StcConfig(
        peri_event_buffer_size=20,
        peri_event_window_sec=1800,
        strong_event_types=frozenset({"memory_capture"}),
        dry_run=False,
    )
    assert cfg.peri_event_buffer_size == 20
    assert cfg.peri_event_window_sec == 1800
    assert cfg.strong_event_types == frozenset({"memory_capture"})
    assert cfg.dry_run is False


def test_T1_stcconfig_is_immutable_frozen() -> None:
    """frozen=True -- attribute assignment raises
    dataclasses.FrozenInstanceError."""
    cfg = StcConfig(
        peri_event_buffer_size=20,
        peri_event_window_sec=1800,
        strong_event_types=frozenset({"memory_capture"}),
        dry_run=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.peri_event_buffer_size = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _load_stc_config() defaults path + pytest-aware dry_run
# ---------------------------------------------------------------------------


def test_T2_defaults_when_no_env_vars_set_under_pytest() -> None:
    """With all four IAI_MCP_STC_* vars absent and PYTEST_CURRENT_TEST
    set (auto by pytest), defaults are 20 / 1800 / {memory_capture,
    error_trace, user_correction} and dry_run=True (pytest-aware)."""
    cfg = _load_stc_config()
    assert cfg.peri_event_buffer_size == 20
    assert cfg.peri_event_window_sec == 1800
    assert cfg.strong_event_types == frozenset({
        "memory_capture", "error_trace", "user_correction",
    })
    # Pytest-aware default: PYTEST_CURRENT_TEST is set by the pytest
    # runner -> dry_run is True.
    assert cfg.dry_run is True


def test_T2_explicit_dry_run_false_overrides_pytest_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit IAI_MCP_STC_DRY_RUN=false beats the pytest-aware
    default branch."""
    monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", "false")
    cfg = _load_stc_config()
    assert cfg.dry_run is False


def test_T2_dry_run_vocab_accepts_synonyms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every documented true/false synonym parses without raising."""
    for v in ("true", "1", "yes", "on", "TRUE", "Yes"):
        monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", v)
        assert _load_stc_config().dry_run is True
    for v in ("false", "0", "no", "off", "FALSE", ""):
        monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", v)
        assert _load_stc_config().dry_run is False


def test_T2_strong_event_types_csv_strip_and_lowercase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CSV input is split on `,`, each token is trimmed and
    lowercased, and the result is a frozenset."""
    monkeypatch.setenv(
        "IAI_MCP_STC_STRONG_EVENT_TYPES",
        " Memory_Capture , ERROR_TRACE,  user_correction ",
    )
    cfg = _load_stc_config()
    assert cfg.strong_event_types == frozenset({
        "memory_capture", "error_trace", "user_correction",
    })
    assert isinstance(cfg.strong_event_types, frozenset)


def test_T2_call_on_demand_rereads_env_between_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every invocation re-reads os.environ. No module-level
    cache. monkeypatch.setenv between two calls must flip the value."""
    monkeypatch.setenv("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "5")
    assert _load_stc_config().peri_event_buffer_size == 5
    monkeypatch.setenv("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "100")
    assert _load_stc_config().peri_event_buffer_size == 100


# ---------------------------------------------------------------------------
# Fail-loud: invalid value -> ValueError naming the var
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        # peri_event_buffer_size: int in [1, 1000]
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "not-an-int"),
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "0"),
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "-1"),
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "1001"),
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "99999"),
        # peri_event_window_sec: int in [1, 86400]
        ("IAI_MCP_PERI_EVENT_WINDOW_SEC", "not-an-int"),
        ("IAI_MCP_PERI_EVENT_WINDOW_SEC", "0"),
        ("IAI_MCP_PERI_EVENT_WINDOW_SEC", "-100"),
        ("IAI_MCP_PERI_EVENT_WINDOW_SEC", "86401"),
        # strong_event_types: non-empty CSV with no empty tokens
        ("IAI_MCP_STC_STRONG_EVENT_TYPES", ""),
        ("IAI_MCP_STC_STRONG_EVENT_TYPES", ","),
        ("IAI_MCP_STC_STRONG_EVENT_TYPES", "a,,b"),
        ("IAI_MCP_STC_STRONG_EVENT_TYPES", "  ,  "),
        # dry_run: must be in documented vocab
        ("IAI_MCP_STC_DRY_RUN", "banana"),
        ("IAI_MCP_STC_DRY_RUN", "maybe"),
    ],
)
def test_T2_R5_invalid_env_var_raises_ValueError_naming_the_var(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    """Every malformed knob fails loud via _load_stc_config(). The error
    message MUST name the offending env var so operators can act."""
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValueError, match=env_var):
        _load_stc_config()
