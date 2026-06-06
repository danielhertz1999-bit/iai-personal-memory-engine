"""AAAK index generator + English-Only storage enforcement.

The English-Only Brain invariant: the surface (Claude) translates inbound
text to English on the way in; storage holds the English form. The
`language` column on MemoryRecord is retained for legacy compatibility on
older rows; new records default to "en". Embedding default is
bge-small-en-v1.5 (384d, English).

This module provides:

- `generate_aaak_index(record)` -- builds a `W:<wing>/R:<room>/E:<entities>/T:<tags>`
  metadata string from a MemoryRecord's tier, community_id and tags. The returned
  string is guaranteed to contain none of record.literal_surface.

- `parse_aaak_index(idx)` -- inverse of the generator, returning a
  {wing, room, entities, tags} dict. Round-trips the entities/tags lists.

- `enforce_language_tagged(record)` -- constitutional guard that raises
  ValueError if record.language is empty. The detect= parameter and any
  pure-Python language-detection path were removed once the English-only
  invariant became unconditional (Claude does the translation on the way
  in, so the brain never needs to guess the language of stored text).

- `enforce_english_raw(record)` -- legacy shim retained for backward compat.
  Preserves the original script-based Cyrillic/CJK rejection for records
  without a `raw:<lang>` tag.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iai_mcp.types import MemoryRecord

# --------------------------------------------------------------- script regex
# Covered: Cyrillic (Russian et al), Hiragana, Katakana, CJK Unified Ideographs.
# Sufficient for the historical raw:<lang> opt-in path; extend the alphabet
# list only if a genuine storage bug surfaces -- don't speculate.
CYRILLIC = re.compile(r"[\u0400-\u04FF]")          # U+0400..U+04FF
HIRAGANA_KATAKANA = re.compile(r"[\u3040-\u30FF]") # U+3040..U+30FF
CJK = re.compile(r"[\u4E00-\u9FFF]")               # U+4E00..U+9FFF Unified Ideographs


# ---------------------------------------------- tier -> wing alphabet
_TIER_TO_WING = {
    "working": "W",
    "episodic": "E",
    "semantic": "S",
    "procedural": "P",
    "parametric": "\u03a0",  # Pi glyph -- distinct from Latin P
}


def _wing_from_tier(tier: str) -> str:
    return _TIER_TO_WING.get(tier, "unknown")


def _room_from_community(record: "MemoryRecord") -> str:
    """First 8 chars of community UUID; "unknown" if community not yet assigned.

    L0/L1 pinned records may still have community_id=None (they're pinned
    by UUID, not graph position).
    """
    if record.community_id is None:
        return "unknown"
    return str(record.community_id)[:8]


def _entities_from_tags(tags: list[str]) -> str:
    """Up to 10 tags prefixed `entity:` (prefix stripped), joined by `,`.

    `"-"` if none found, so the generator output has a stable shape with
    exactly 3 `/` separators regardless of tag content.
    """
    ents = [t[len("entity:"):] for t in tags if t.startswith("entity:")][:10]
    if not ents:
        return "-"
    return ",".join(ents)


def _tagline(tags: list[str]) -> str:
    """Up to 10 non-entity tags joined by `,`. `"-"` if none."""
    non_ents = [t for t in tags if not t.startswith("entity:")][:10]
    if not non_ents:
        return "-"
    return ",".join(non_ents)


# ---------------------------------------------------------------- public API


def generate_aaak_index(record: "MemoryRecord") -> str:
    """Build the AAAK index string for a record.

    Format: `W:<wing>/R:<room>/E:<entities>/T:<tags>`

    Guarantees:
    - Exactly 3 `/` separators regardless of content.
    - Contains NO substring of `record.literal_surface`. Verified by
      `tests/test_aaak.py::test_aaak_index_does_not_contain_literal_surface`.
    - Deterministic: same record -> same index on repeat calls.
    """
    wing = _wing_from_tier(record.tier)
    room = _room_from_community(record)
    entities = _entities_from_tags(record.tags)
    tags = _tagline(record.tags)
    return f"W:{wing}/R:{room}/E:{entities}/T:{tags}"


def parse_aaak_index(idx: str) -> dict[str, list[str]]:
    """Inverse of generate_aaak_index. Returns wing/room/entities/tags lists.

    Each value is a list (even wing/room which are single strings) so callers
    have a uniform shape. Unknown keys are ignored. Empty-value `-` becomes [].
    """
    out: dict[str, list[str]] = {
        "wing": [],
        "room": [],
        "entities": [],
        "tags": [],
    }
    key_map = {"W": "wing", "R": "room", "E": "entities", "T": "tags"}
    for seg in idx.split("/"):
        if ":" not in seg:
            continue
        k, _, v = seg.partition(":")
        if k not in key_map:
            continue
        name = key_map[k]
        if v == "-" or v == "":
            out[name] = []
        else:
            # Wing/Room are single-token; entities/tags are comma-separated.
            if name in ("wing", "room"):
                out[name] = [v]
            else:
                out[name] = v.split(",")
    return out


def enforce_language_tagged(record: "MemoryRecord") -> None:
    """Guard: every record MUST carry a non-empty language tag.

    When `record.language` is a non-empty string, the guard passes
    unconditionally (the column is retained for legacy compatibility on
    older rows; new records default to "en" under the English-Only Brain
    invariant).

    When `record.language` is empty/missing, raises ValueError. There is
    no auto-detection path: the surface (Claude) translates inbound text
    to English on the way in, so the brain never needs to guess the
    language of stored text. Callers that need a default should set
    `record.language = "en"` explicitly before calling this guard.
    """
    if record.language and isinstance(record.language, str) and record.language.strip():
        return  # already tagged; accept

    raise ValueError(
        "constitutional violation: record.language is required and must be "
        "non-empty. Set record.language='en' (or the appropriate code on "
        "legacy rows) before calling this guard."
    )


def enforce_english_raw(record: "MemoryRecord") -> None:
    """Legacy script-based guard retained for backward compatibility.

    Semantics (preserved byte-for-byte for backward compatibility):
    - `raw:<lang>` tag present on record -> accept (explicit raw capture)
    - literal_surface contains Cyrillic / Hiragana / Katakana / CJK codepoints
      and no `raw:<lang>` tag -> raise ValueError("constitutional...")
    - else -> accept

    The modern guard is `enforce_language_tagged`; downstream callers that
    just need a language-tag presence check should import that directly.
    This shim is kept so the existing test fixtures (tests/test_aaak.py,
    tests/test_provenance.py) continue to assert the exact rejection
    behaviour they documented.
    """
    text = record.literal_surface or ""
    has_non_english = bool(
        CYRILLIC.search(text)
        or HIRAGANA_KATAKANA.search(text)
        or CJK.search(text)
    )
    if not has_non_english:
        return

    # Caller opted in via `raw:<lang>` tag -> accept.
    if any(t.startswith("raw:") for t in record.tags):
        return

    raise ValueError(
        "constitutional violation: literal_surface contains non-English "
        "characters; storage must be English raw verbatim. "
        "Add 'raw:<lang>' tag to declare explicit raw capture."
    )
