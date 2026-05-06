"""AAAK index generator + English-Only storage enforcement.

Phase 1 constitutional rule:
    Storage is raw verbatim English always. AAAK is a RETRIEVAL VIEW only.

Phase 2 (superseded):
    Storage was briefly amended to raw verbatim in the user's original language.
    Every MemoryRecord carries an ISO-639-1 `language` tag retained as a column
    on legacy rows from that era.

Plan 05-08 (2026-04-19) restored the English-Only Brain (D-08 spirit):
    The surface (Claude) translates inbound text to English; storage holds the
    English form. The `language` column is retained for legacy compatibility;
    new records default to "en". Embedding default is bge-small-en-v1.5 (384d,
    English) per Plan 05-08.

This module provides:

- `generate_aaak_index(record)` -- builds a `W:<wing>/R:<room>/E:<entities>/T:<tags>`
  metadata string from a MemoryRecord's tier, community_id and tags. The returned
  string is guaranteed to contain none of record.literal_surface.

- `parse_aaak_index(idx)` -- inverse of the generator, returning a
  {wing, room, entities, tags} dict. Round-trips the entities/tags lists.

- `enforce_language_tagged(record, detect=False)` -- guard.
  Raises ValueError if record.language is empty and detect is False. When
  detect=True, runs langdetect on literal_surface; mutates record.language
  with the detected code if confidence >= 0.7, else raises. Empty text with
  detect=True defaults to "en" without raising.

- `enforce_english_raw(record)` -- shim retained for backward compat.
  Delegates to enforce_language_tagged for records with a language tag set;
  preserves Cyrillic/CJK rejection for records without one unless
  `raw:<lang>` tag is present.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iai_mcp.types import MemoryRecord

# constitutional: confidence threshold below which langdetect refuses.
LANGDETECT_MIN_CONFIDENCE = 0.7


# --------------------------------------------------------------- script regex
# Covered: Cyrillic (Russian et al), Hiragana, Katakana, CJK Unified Ideographs.
# Sufficient for (the three scripts the project explicitly documents
# as needing `raw:<lang>` handling). Extend the alphabet list in only
# if a genuine storage bug surfaces -- don't speculate.
CYRILLIC = re.compile(r"[\u0400-\u04FF]")          # U+0400..U+04FF
HIRAGANA_KATAKANA = re.compile(r"[\u3040-\u30FF]") # U+3040..U+30FF
CJK = re.compile(r"[\u4E00-\u9FFF]")               # U+4E00..U+9FFF Unified Ideographs


# ---------------------------------------------- tier -> wing alphabet (TOK-10)
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

    Plan 02 assigns community_id; Plan 03 L0/L1 pinned records may still have
    community_id=None (they're pinned by UUID, not graph position).
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
    """Build the AAAK index string for a record (D-08, TOK-10).

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


def enforce_language_tagged(
    record: "MemoryRecord",
    *,
    detect: bool = False,
) -> None:
    """D-08a constitutional: every Phase-2+ record MUST carry a language tag.

    When record.language is a non-empty string, the guard passes unconditionally
    (the column is retained for legacy compatibility; the English-Only Brain
    pivot in means new records default to "en").

    When record.language is empty/missing and detect is False, raises
    ValueError("constitutional violation: ...") because storage is
    tag-addressable -- not defaulting to English.

    When detect=True and language is empty:
    - If literal_surface is empty/whitespace, sets language="en" and returns.
    - Else runs langdetect; if top candidate has probability >= 0.7 (D-08a
      threshold), mutates record.language with the detected code.
    - If langdetect fails or confidence < 0.7, raises ValueError.

    The seed for langdetect's DetectorFactory is fixed at 42 so the same text
    always produces the same language code across runs.
    """
    if record.language and isinstance(record.language, str) and record.language.strip():
        return  # already tagged; accept

    if not detect:
        raise ValueError(
            "constitutional violation: record.language is required. "
            "Pass detect=True to auto-detect via langdetect."
        )

    text = record.literal_surface or ""
    if not text.strip():
        record.language = "en"  # empty -> default en
        return

    try:
        from langdetect import DetectorFactory, detect_langs
        DetectorFactory.seed = 42  # determinism
        candidates = detect_langs(text)
    except Exception as e:
        raise ValueError(
            f"constitutional violation: langdetect failed on record text: {e}"
        )

    if not candidates or candidates[0].prob < LANGDETECT_MIN_CONFIDENCE:
        top = candidates[0] if candidates else None
        raise ValueError(
            f"constitutional violation: langdetect confidence too low "
            f"(<{LANGDETECT_MIN_CONFIDENCE}); top candidate={top}"
        )

    record.language = candidates[0].lang


def enforce_english_raw(record: "MemoryRecord") -> None:
    """Phase 1 shim -- preserves the original script-based guard.

    semantics (retained byte-for-byte for backward compatibility):
    - `raw:<lang>` tag present on record -> accept (explicit raw capture)
    - literal_surface contains Cyrillic / Hiragana / Katakana / CJK codepoints
      and no `raw:<lang>` tag -> raise ValueError("constitutional ...")
    - else -> accept

    The guard is exposed as `enforce_language_tagged`. Downstream
    plans that want native-language storage should import that directly
    instead of this shim. This function is kept so the test fixtures
    (tests/test_aaak.py, tests/test_provenance.py) continue to assert the
    exact rejection behaviour they documented.
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
        "characters; storage must be English raw verbatim (D-08, TOK-10). "
        "Add 'raw:<lang>' tag to declare explicit raw capture."
    )
