from __future__ import annotations

import hashlib


HASH_LEN = 16
COMPONENTS = ("l0", "l1", "l2", "rich_club")


def hash_component(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8") if text is not None else b"").hexdigest()
    return h[:HASH_LEN]


def _component_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return str(value)


def build_delta(
    previous_hashes: dict[str, str],
    current_payload: dict,
) -> tuple[dict, dict[str, str]]:
    delta: dict = {}
    new_hashes: dict[str, str] = {}
    for key in COMPONENTS:
        value = current_payload.get(key)
        text = _component_text(value)
        h = hash_component(text)
        new_hashes[key] = h
        prev = previous_hashes.get(key) if previous_hashes else None
        if prev != h:
            delta[key] = value if value is not None else ""
    return delta, new_hashes


def apply_delta(previous: dict, delta: dict) -> dict:
    merged = dict(previous)
    for key, value in delta.items():
        merged[key] = value
    return merged
