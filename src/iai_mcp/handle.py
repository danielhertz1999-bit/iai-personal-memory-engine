from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from typing import NamedTuple


COMPACT_HANDLE_RE = re.compile(r"<iai:[0-9a-f]{16}>")

COMPACT_HANDLE_TOKEN_BUDGET = 16

_CACHE_CAPACITY = 256


class HandleParts(NamedTuple):

    identity_short: str
    session_short: str
    topic_label: str
    pending: int


_HANDLE_CACHE: "OrderedDict[str, HandleParts]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _remember(handle: str, parts: HandleParts) -> None:
    with _CACHE_LOCK:
        if handle in _HANDLE_CACHE:
            _HANDLE_CACHE.move_to_end(handle)
            return
        _HANDLE_CACHE[handle] = parts
        while len(_HANDLE_CACHE) > _CACHE_CAPACITY:
            _HANDLE_CACHE.popitem(last=False)


def encode_compact_handle(
    identity_short: str,
    session_short: str,
    topic_label: str,
    pending: int,
) -> str:
    id_s = str(identity_short or "")
    sess_s = str(session_short or "-")
    topic_s = str(topic_label or "none")
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
    digest = h.hexdigest()

    handle = f"<iai:{digest}>"
    _remember(handle, HandleParts(id_s, sess_s, topic_s, pend_i))
    return handle


def decode_compact_handle(handle: str) -> HandleParts | None:
    if not isinstance(handle, str) or not COMPACT_HANDLE_RE.fullmatch(handle):
        return None
    with _CACHE_LOCK:
        parts = _HANDLE_CACHE.get(handle)
        if parts is not None:
            _HANDLE_CACHE.move_to_end(handle)
        return parts


def _reset_cache_for_tests() -> None:
    with _CACHE_LOCK:
        _HANDLE_CACHE.clear()
