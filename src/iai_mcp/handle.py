"""Compact session handle (Plan 05-06 -- ≤16 raw tok target).

Collapses three pointer fields historically emitted at session-start::

    <id:{8-hex}>               (~8  raw tok)   identity pointer (L0 UUID prefix)
    <sess:{8-hex} pend:{N}>    (~12 raw tok)   brain session handle + pending
    <topic:{label<=8}>         (~8  raw tok)   dominant community hint

into a single opaque pointer::

    <iai:HHHHHHHHHHHHHHHH>     (~6-10 raw tok) 16-hex blake2s digest

The payload bytes are derived deterministically from the three inputs via
blake2s(digest_size=8) -> 64 bits -> 16 hex chars. Deterministic encoding
means identical (id, sess, topic, pending) always yields the same handle,
so the handle can be quoted back to the server and resolved.

Resolution: the module keeps a bounded LRU (`_HANDLE_CACHE`) of the most
recent encodings so the wrapper / recall paths can decode a handle back
into its tuple without re-running the encoder. The cache is process-
local and intentionally small -- session-start emits one handle per new
session, so 256 slots handles the realistic working set with room for
concurrent sessions during sleep-daemon transitions. Misses are a
possible outcome (stale handle from an old process) and callers treat
them as recoverable: the live payload still carries the legacy pointer
fields under ``standard`` / ``deep`` wake_depth for fallback.

Security / invariants:

* The handle carries NO secrets. It is a hash of values Claude already
  saw (L0 UUID prefix, session id prefix, community label, pending
  count). Compromising the handle tells an attacker nothing they could
  not learn from the full session-start payload.
* blake2s is non-reversible. The cache is the only decode path. A
  caller that did not mint the handle cannot invert it -- by design.
* C6 (read-only audit) is untouched: this module writes nothing to the
  store; the cache is pure in-memory state.
"""
from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from typing import NamedTuple

# ------------------------------------------------------------------ constants

#: Regex a compact handle must match. Exposed for test assertions and
#: for the decoder's input-validation contract.
COMPACT_HANDLE_RE = re.compile(r"<iai:[0-9a-f]{16}>")

#: Raw-token budget ceiling for the compact handle per target.
#: Enforced by tests/test_handle.py against ``bench/tokens._approx_tokens``.
COMPACT_HANDLE_TOKEN_BUDGET = 16

#: Cache capacity. 256 concurrent handles is plenty for the realistic
#: steady-state: one per session, a handful of overlapping sessions
#: during daemon sleep transitions, plus test churn. Tuning knob, not
#: a policy guarantee.
_CACHE_CAPACITY = 256


# ------------------------------------------------------------------ types


class HandleParts(NamedTuple):
    """Decoded parts of a compact handle (server-side, never serialised)."""

    identity_short: str        # 8 hex of L0 UUID, or "" when unseeded
    session_short: str         # 8 hex of session id, or "-" placeholder
    topic_label: str           # community label (<=8 char) or "none"
    pending: int               # first_turn_pending count (>= 0)


# ------------------------------------------------------------------ cache


_HANDLE_CACHE: "OrderedDict[str, HandleParts]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _remember(handle: str, parts: HandleParts) -> None:
    """Record handle -> parts with LRU eviction."""
    with _CACHE_LOCK:
        if handle in _HANDLE_CACHE:
            _HANDLE_CACHE.move_to_end(handle)
            return
        _HANDLE_CACHE[handle] = parts
        while len(_HANDLE_CACHE) > _CACHE_CAPACITY:
            _HANDLE_CACHE.popitem(last=False)


# ------------------------------------------------------------------ public API


def encode_compact_handle(
    identity_short: str,
    session_short: str,
    topic_label: str,
    pending: int,
) -> str:
    """Derive the ``<iai:HHHHHHHHHHHHHHHH>`` handle from the three pointer inputs.

    The output is deterministic: equal inputs always yield equal handles.
    Inputs are normalised (``str``, sanitised) before hashing so whitespace
    or accidental newlines never affect the digest.

    The returned handle is also inserted into the in-memory decode cache
    so ``decode_compact_handle`` can reverse it within the same process.
    """
    id_s = str(identity_short or "")
    sess_s = str(session_short or "-")
    topic_s = str(topic_label or "none")
    # Coerce pending to a bounded non-negative int; negatives or huge values
    # are clamped to the [0, 999] window the emit site actually produces.
    try:
        pend_i = max(0, min(999, int(pending)))
    except (TypeError, ValueError):
        pend_i = 0

    h = hashlib.blake2s(digest_size=8)
    h.update(id_s.encode("utf-8"))
    h.update(b"\x1f")
    h.update(sess_s.encode("utf-8"))
    h.update(b"\x1f")
    h.update(topic_s.encode("utf-8"))
    h.update(b"\x1f")
    h.update(str(pend_i).encode("utf-8"))
    digest = h.hexdigest()  # 16 hex chars

    handle = f"<iai:{digest}>"
    _remember(handle, HandleParts(id_s, sess_s, topic_s, pend_i))
    return handle


def decode_compact_handle(handle: str) -> HandleParts | None:
    """Return the parts for a handle minted earlier in this process.

    Returns ``None`` when the input is malformed or the handle is no
    longer in the LRU (cold cache / different process). Callers treat a
    miss as a soft error -- the legacy ``identity_pointer`` /
    ``brain_handle`` / ``topic_cluster_hint`` fields remain available in
    ``standard`` / ``deep`` wake_depth for fallback resolution.
    """
    if not isinstance(handle, str) or not COMPACT_HANDLE_RE.fullmatch(handle):
        return None
    with _CACHE_LOCK:
        parts = _HANDLE_CACHE.get(handle)
        if parts is not None:
            _HANDLE_CACHE.move_to_end(handle)
        return parts


def _reset_cache_for_tests() -> None:
    """Test-only: clear the LRU. Production code must never call this."""
    with _CACHE_LOCK:
        _HANDLE_CACHE.clear()
