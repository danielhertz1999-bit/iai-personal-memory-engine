"""BSC saturation guard + role-saturation telemetry tests.

Tests:
 1. test_max_bundle_pairs_default_D_4096
 2. test_max_bundle_pairs_D_10000
 3. test_max_bundle_pairs_D_2048
 4. test_max_bundle_pairs_tiny_D_at_least_one
 5. test_bundle_at_capacity_succeeds_D_4096
 6. test_bundle_raises_at_capacity_plus_one_D_4096
 7. test_bundle_capacity_error_is_value_error
 8. test_bundle_at_capacity_D_10000_succeeds_with_25_pairs
 9. test_bundle_below_warn_threshold_no_telemetry
10. test_bundle_above_warn_threshold_emits_telemetry
11. test_bundle_above_warn_threshold_no_store_no_emit
12. test_telemetry_kind_string_matches_events_module
13. test_bundle_over_cap_emits_then_raises
"""
from __future__ import annotations

from pathlib import Path

import pytest

from iai_mcp.lilli.errors import BundleCapacityError
from iai_mcp.lilli.tiers.bsc import (
    BSC_MAX_BUNDLE_PAIRS,
    _TELEMETRY_ROLE_SATURATION_KIND,
    _max_bundle_pairs,
    bundle,
    filler_hv,
    role_hv,
)


# ---------------------------------------------------------------------------
# Helper: open an isolated MemoryStore with crypto passphrase
# ---------------------------------------------------------------------------


def _open_store(tmpdir: str, monkeypatch: pytest.MonkeyPatch):
    """Open a MemoryStore in an isolated temp dir with keyring bypass.

    Env vars are set via monkeypatch so they are automatically reverted after
    each test, preventing env leaks into the rest of the suite.
    """
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_KEYRING_BYPASS", "true")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-saturation-pp")
    return MemoryStore(path=Path(tmpdir) / "store")


def _close_store(store) -> None:
    """Close store."""
    try:
        store.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1-4. _max_bundle_pairs / BSC_MAX_BUNDLE_PAIRS constants
# ---------------------------------------------------------------------------


def test_max_bundle_pairs_default_D_4096() -> None:
    assert BSC_MAX_BUNDLE_PAIRS == 10
    assert _max_bundle_pairs(4096) == 10


def test_max_bundle_pairs_D_10000() -> None:
    assert _max_bundle_pairs(10000) == 25


def test_max_bundle_pairs_D_2048() -> None:
    assert _max_bundle_pairs(2048) == 5


def test_max_bundle_pairs_tiny_D_at_least_one() -> None:
    # max(1, D // 400) floor: D=8 -> 8//400 = 0 -> max(1, 0) = 1
    assert _max_bundle_pairs(8) >= 1


# ---------------------------------------------------------------------------
# 5-8. Saturation boundary: success at cap, raise at cap+1
# ---------------------------------------------------------------------------


def test_bundle_at_capacity_succeeds_D_4096() -> None:
    """Bundle 10 pairs at D=4096 (= max cap) succeeds and returns 512 bytes."""
    roles = [
        "WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID",
        "TEMPORAL_POSITION", "ACTOR", "OBJECT", "INTENT", "MODALITY",
    ]
    assert len(roles) == 10 == BSC_MAX_BUNDLE_PAIRS
    pairs = [(r, filler_hv(f"v{i}")) for i, r in enumerate(roles)]
    result = bundle(pairs)
    assert isinstance(result, bytes)
    assert len(result) == 512


def test_bundle_raises_at_capacity_plus_one_D_4096() -> None:
    """Bundle 11 pairs at D=4096 (= cap+1) raises BundleCapacityError."""
    roles = [
        "WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID",
        "TEMPORAL_POSITION", "ACTOR", "OBJECT", "INTENT", "MODALITY", "LANG",
    ]
    assert len(roles) == 11
    pairs = [(r, filler_hv(f"v{i}")) for i, r in enumerate(roles)]
    with pytest.raises(BundleCapacityError) as exc_info:
        bundle(pairs)
    err_str = str(exc_info.value)
    assert "D=4096" in err_str
    assert "10" in err_str  # cap mentioned in message


def test_bundle_capacity_error_is_value_error() -> None:
    assert issubclass(BundleCapacityError, ValueError)


def test_bundle_at_capacity_D_10000_succeeds_with_25_pairs() -> None:
    """At D=10000, 25 pairs succeeds; 26 raises BundleCapacityError."""
    assert _max_bundle_pairs(10000) == 25
    # 25 pairs — must succeed
    pairs_25 = [(r if i < 18 else f"EXTRA_{i}", filler_hv(f"v{i}", D=10000))
                for i, r in enumerate(
                    list("WHEN WHERE ROLE PROJECT COMMUNITY_ID TEMPORAL_POSITION "
                         "ACTOR OBJECT INTENT MODALITY LANG SESSION_ID "
                         "TIER VALENCE CERTAINTY SOURCE TOPIC PARENT_ID".split())
                    + [f"EX{j}" for j in range(7)]
                )]
    assert len(pairs_25) == 25
    result = bundle(pairs_25, D=10000)
    assert len(result) == 1250

    # 26 pairs — must raise
    pairs_26 = pairs_25 + [("EXTRA_26", filler_hv("v26", D=10000))]
    with pytest.raises(BundleCapacityError):
        bundle(pairs_26, D=10000)


# ---------------------------------------------------------------------------
# 9. Below warn threshold: no telemetry
# ---------------------------------------------------------------------------


def test_bundle_below_warn_threshold_no_telemetry(tmp_path, monkeypatch) -> None:
    """7 pairs at D=4096 is below warn threshold (8 = ceil(0.8*10)). No event emitted."""
    from iai_mcp.events import query_events

    store = _open_store(str(tmp_path), monkeypatch)
    try:
        roles = ["WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID", "TEMPORAL_POSITION", "ACTOR"]
        assert len(roles) == 7
        pairs = [(r, filler_hv(f"v{i}")) for i, r in enumerate(roles)]
        bundle(pairs, store=store)
        events = query_events(store, kind="role_saturation_warning")
        assert len(events) == 0, f"Expected no saturation events at 7 pairs, got {len(events)}"
    finally:
        _close_store(store)


# ---------------------------------------------------------------------------
# 10. Above warn threshold: telemetry emitted
# ---------------------------------------------------------------------------


def test_bundle_above_warn_threshold_emits_telemetry(tmp_path, monkeypatch) -> None:
    """9 pairs at D=4096 (>= 8 = ceil(0.8*10), < 11 = over cap) emits telemetry."""
    from iai_mcp.events import query_events

    store = _open_store(str(tmp_path), monkeypatch)
    try:
        roles = [
            "WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID",
            "TEMPORAL_POSITION", "ACTOR", "OBJECT", "INTENT",
        ]
        assert len(roles) == 9
        pairs = [(r, filler_hv(f"v{i}")) for i, r in enumerate(roles)]
        result = bundle(pairs, store=store)
        assert isinstance(result, bytes), "bundle should succeed at 9 pairs"

        events = query_events(store, kind="role_saturation_warning")
        assert len(events) >= 1, f"Expected >= 1 saturation event at 9 pairs, got {len(events)}"

        # Verify payload data
        payload = events[0]["data"]
        assert payload.get("D") == 4096
        assert payload.get("n_pairs") == 9
        assert payload.get("max_pairs") == 10
    finally:
        _close_store(store)


# ---------------------------------------------------------------------------
# 11. store=None: no emit, no exception
# ---------------------------------------------------------------------------


def test_bundle_above_warn_threshold_no_store_no_emit() -> None:
    """9 pairs with store=None (default) does not raise and silently skips telemetry."""
    roles = [
        "WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID",
        "TEMPORAL_POSITION", "ACTOR", "OBJECT", "INTENT",
    ]
    pairs = [(r, filler_hv(f"v{i}")) for i, r in enumerate(roles)]
    # Must not raise — store is None (default), telemetry is silently skipped
    result = bundle(pairs)
    assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# 12. Telemetry kind string consistency check (may skip if 46-10 not shipped)
# ---------------------------------------------------------------------------


def test_telemetry_kind_string_matches_events_module() -> None:
    """When events.TELEMETRY_ROLE_SATURATION is defined, it must match bsc._TELEMETRY_ROLE_SATURATION_KIND."""
    from iai_mcp import events

    if not hasattr(events, "TELEMETRY_ROLE_SATURATION"):
        pytest.skip("events.TELEMETRY_ROLE_SATURATION not yet defined (46-10 has not shipped)")

    assert _TELEMETRY_ROLE_SATURATION_KIND == events.TELEMETRY_ROLE_SATURATION, (
        f"bsc._TELEMETRY_ROLE_SATURATION_KIND={_TELEMETRY_ROLE_SATURATION_KIND!r} "
        f"!= events.TELEMETRY_ROLE_SATURATION={events.TELEMETRY_ROLE_SATURATION!r}"
    )


# ---------------------------------------------------------------------------
# 13. EMIT-THEN-RAISE contract (Gate 18 contract)
# ---------------------------------------------------------------------------


def test_bundle_over_cap_emits_then_raises(tmp_path, monkeypatch) -> None:
    """11 pairs at D=4096 with store: emits telemetry BEFORE raising BundleCapacityError.

    This is the Gate 18 contract. The telemetry event MUST be observable in the
    store after catching the BundleCapacityError — proving emission happened first.
    """
    from iai_mcp.events import query_events

    store = _open_store(str(tmp_path), monkeypatch)
    try:
        roles = [
            "WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID",
            "TEMPORAL_POSITION", "ACTOR", "OBJECT", "INTENT", "MODALITY", "LANG",
        ]
        assert len(roles) == 11  # over cap of 10
        pairs = [(r, filler_hv(f"v{i}")) for i, r in enumerate(roles)]

        # Call MUST raise BundleCapacityError
        with pytest.raises(BundleCapacityError):
            bundle(pairs, store=store)

        # AFTER catching the exception, the telemetry event MUST be in the store.
        # This proves emit happened BEFORE the raise (the Gate 18 ordering contract).
        events = query_events(store, kind="role_saturation_warning")
        assert len(events) >= 1, (
            "EMIT-THEN-RAISE contract violated: telemetry event not found in store "
            "after catching BundleCapacityError. The emit must happen before the raise."
        )
    finally:
        _close_store(store)
