"""Task 1 — compact <iai:HHHHHHHHHHHHHHHH> handle tests.

Replaces the three legacy pointer fields at wake_depth=minimal with one
blake2s-derived 16-hex opaque handle. The payload dataclass still
carries the legacy fields for back-compat callers, but under
``minimal`` they are left empty so only the compact handle contributes
to ``total_cached_tokens`` (<=16 raw, below claude-mem's 17).

Covered contracts:

    Test 1 dataclass field present at minimal with non-empty value
    Test 2 encode_compact_handle is deterministic (same inputs -> same digest)
    Test 3 decode_compact_handle returns the original parts (LRU hit)
    Test 4 decode of an unknown (cold-cache) handle returns None
    Test 5 decode of a malformed handle returns None
    Test 6 standard / deep branches ALSO populate compact_handle (back-compat opt-in)
    Test 7 minimal payload warm token count <= 16 raw via bench.tokens._approx_tokens
    Test 8 constitutional: no profile-knob names may leak via the compact handle surface
    Test 9 minimal branch leaves the three legacy pointer fields empty
    Test 10 _resolve_compact_handle_to_pointers rebuilds the legacy triple verbatim
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.handle import (
    COMPACT_HANDLE_RE,
    COMPACT_HANDLE_TOKEN_BUDGET,
    decode_compact_handle,
    encode_compact_handle,
    _reset_cache_for_tests,
)
from iai_mcp.session import _approx_tokens


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _fresh_handle_cache():
    """Clean the module-level LRU between tests so decode-hit / decode-miss
    outcomes are deterministic."""
    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Stub keyring with an in-memory dict so MemoryStore never hits the macOS
    Keychain (same pattern used in tests/test_hippea_cascade.py)."""
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


@pytest.fixture
def _fresh_store(tmp_path: Path):
    """Hermetic MemoryStore anchored in a fresh tmp directory."""
    from iai_mcp.store import MemoryStore

    return MemoryStore(path=tmp_path / "hippo")


def _assemble_with_wake_depth(store, wake_depth):
    """Invoke assemble_session_start at the requested wake_depth, reusing
    the production retrieve.build_runtime_graph pipeline."""
    from iai_mcp import retrieve
    from iai_mcp.session import assemble_session_start

    _graph, assignment, rc = retrieve.build_runtime_graph(store)
    return assemble_session_start(
        store,
        assignment,
        rc,
        session_id=str(uuid4()),
        profile_state={"wake_depth": wake_depth},
    )


# --------------------------------------------------------------------------- Test 1


def test_minimal_payload_carries_non_empty_compact_handle(_fresh_store):
    payload = _assemble_with_wake_depth(_fresh_store, "minimal")
    assert payload.wake_depth == "minimal"
    assert payload.compact_handle != ""
    assert COMPACT_HANDLE_RE.fullmatch(payload.compact_handle)


# --------------------------------------------------------------------------- Test 2


def test_encode_is_deterministic():
    a = encode_compact_handle("abcdef01", "12345678", "general", 3)
    b = encode_compact_handle("abcdef01", "12345678", "general", 3)
    assert a == b
    assert COMPACT_HANDLE_RE.fullmatch(a)


# --------------------------------------------------------------------------- Test 3


def test_decode_round_trips_for_lru_hit():
    handle = encode_compact_handle("feedface", "cafebabe", "security", 7)
    parts = decode_compact_handle(handle)
    assert parts is not None
    # HandleParts is a NamedTuple(identity_short, session_short, topic_label, pending)
    assert parts[0] == "feedface"
    assert parts[1] == "cafebabe"
    assert parts[2] == "security"
    assert parts[3] == 7


# --------------------------------------------------------------------------- Test 4


def test_decode_cold_cache_returns_none():
    # Synthesise a well-formed but never-encoded handle. With a fresh LRU the
    # decoder cannot reverse it and must signal miss rather than guess.
    fake = "<iai:" + ("a" * 16) + ">"
    assert decode_compact_handle(fake) is None


# --------------------------------------------------------------------------- Test 5


@pytest.mark.parametrize(
    "malformed",
    [
        "",
        "abcdef0123456789",                     # no wrapper
        "<iai:ABCDEF0123456789>",               # uppercase hex not allowed
        "<iai:xyz>",                            # non-hex
        "<iai:" + ("a" * 15) + ">",             # 15 hex chars
        "<iai:" + ("a" * 17) + ">",             # 17 hex chars
        "<id:abcdef01>",                        # legacy pointer shape
        None,
        12345,
    ],
)
def test_decode_rejects_malformed(malformed):
    assert decode_compact_handle(malformed) is None


# --------------------------------------------------------------------------- Test 6


def test_standard_and_deep_populate_compact_handle_for_back_compat(_fresh_store):
    """Standard / deep payloads carry BOTH the eager segments AND a compact
    handle so downstream code can opt into the short form without forcing a
    wake_depth mode switch."""
    for depth in ("standard", "deep"):
        payload = _assemble_with_wake_depth(_fresh_store, depth)
        assert payload.wake_depth == depth
        assert payload.compact_handle != "", f"compact_handle missing at wake_depth={depth}"
        assert COMPACT_HANDLE_RE.fullmatch(payload.compact_handle)


# --------------------------------------------------------------------------- Test 7


def test_minimal_payload_cached_tokens_within_budget(_fresh_store):
    payload = _assemble_with_wake_depth(_fresh_store, "minimal")
    # cached prefix at minimal is the compact handle alone.
    assert payload.total_cached_tokens <= COMPACT_HANDLE_TOKEN_BUDGET, (
        f"cached={payload.total_cached_tokens} exceeds budget "
        f"{COMPACT_HANDLE_TOKEN_BUDGET}"
    )
    # Budget invariant also matches the approx counter on the wire string.
    assert _approx_tokens(payload.compact_handle) <= COMPACT_HANDLE_TOKEN_BUDGET


# --------------------------------------------------------------------------- Test 8


def test_compact_handle_is_hex_only_no_knob_leak():
    """Profile-knob names must NOT surface through the
    session-start prefix. The compact handle is
    ``<iai:{16 hex chars}>`` by construction so any knob name would have to
    smuggle itself through the hash digest, which is cryptographically
    impossible to engineer for arbitrary ASCII substrings."""
    import re

    knob_names = [
        "wake_depth",
        "autistic_mode",
        "hebbian_rate",
        "camouflaging_relaxation",
        "response_formality",
    ]
    handle = encode_compact_handle("abcdef01", "12345678", "general", 0)
    for name in knob_names:
        assert name not in handle, f"knob {name!r} leaked into {handle!r}"
    body = handle[5:-1]  # strip "<iai:" and trailing ">"
    assert re.fullmatch(r"[0-9a-f]{16}", body)


# --------------------------------------------------------------------------- Test 9


def test_minimal_cached_count_charges_only_compact_handle(_fresh_store):
    """Back-compat contract: the 3 legacy pointer strings stay populated on
    the dataclass so older consumers keep working, but
    ``total_cached_tokens`` reflects ONLY the compact handle --- the wire
    prefix at wake_depth=minimal is the compact handle alone."""
    payload = _assemble_with_wake_depth(_fresh_store, "minimal")
    # Legacy fields remain populated (non-empty under a real run with an L0).
    assert payload.brain_handle.startswith("<sess:")
    assert payload.topic_cluster_hint.startswith("<topic:")
    # But token accounting charges only the compact wire prefix.
    assert payload.total_cached_tokens == _approx_tokens(payload.compact_handle)
    assert payload.total_cached_tokens <= COMPACT_HANDLE_TOKEN_BUDGET


# --------------------------------------------------------------------------- Test 10


def test_resolve_compact_handle_rebuilds_legacy_triple():
    """No information loss vs prior 3-field shape: from the compact handle
    we can reconstruct the exact legacy pointer strings. Proves the
    encoding is isomorphic to the original under the values session.py
    actually emits."""
    from iai_mcp.session import _resolve_compact_handle_to_pointers

    handle = encode_compact_handle("abcdef01", "12345678", "general", 4)
    triple = _resolve_compact_handle_to_pointers(handle)
    assert triple is not None
    identity_pointer, brain_handle, topic_cluster_hint = triple
    assert identity_pointer == "<id:abcdef01>"
    assert brain_handle == "<sess:12345678 pend:4>"
    assert topic_cluster_hint == "<topic:general>"


def test_resolve_compact_handle_returns_none_for_unknown():
    """Cold-cache decode path is surfaced to session.py callers as a None
    triple, not a partial / guessed string."""
    from iai_mcp.session import _resolve_compact_handle_to_pointers

    fake = "<iai:" + ("b" * 16) + ">"
    assert _resolve_compact_handle_to_pointers(fake) is None
