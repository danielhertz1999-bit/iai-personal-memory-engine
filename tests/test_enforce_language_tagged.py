"""Tests for `enforce_language_tagged` (constitutional language-tag guard).

The legacy `enforce_english_raw` gated storage to English-only by script
codepoints. The modern guard only checks that the language tag is non-empty;
under the English-Only Brain invariant the surface translates inbound text
to English on the way in, so the brain never needs to guess.

`enforce_english_raw` is retained as a backward-compat shim.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


def _rec(text: str, language: str = "", tags: list[str] | None = None) -> MemoryRecord:
    """Build a MemoryRecord with an overridable language tag.

    When language="" we would normally fail __post_init__, but we need to
    exercise the "missing tag" enforcement path. So we set a placeholder
    language="XX" when the caller asks for empty and the guard will fail
    accordingly via its own checks.
    """
    # For tests that probe missing language, pass "XX" (still valid non-empty)
    # and then zero it out on the record after construction.
    actual_lang = language if language else "XX"
    r = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=list(tags) if tags else [],
        language=actual_lang,
    )
    if not language:
        # Post-construction: simulate "record missing language" for the guard.
        r.language = ""
    return r


# ---------------------------------------------------- enforce_language_tagged


def test_enforce_language_tagged_accepts_english_with_tag():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("hello world", language="en")
    enforce_language_tagged(r)  # should not raise


def test_enforce_language_tagged_accepts_russian_with_tag():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("привет мир", language="ru")
    enforce_language_tagged(r)


def test_enforce_language_tagged_accepts_japanese_with_tag():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("こんにちは", language="ja")
    enforce_language_tagged(r)


def test_enforce_language_tagged_accepts_arabic_with_tag():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("مرحبا بالعالم", language="ar")
    enforce_language_tagged(r)


def test_enforce_language_tagged_rejects_missing_language():
    """record.language="" must raise — the guard has no auto-detect path."""
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("some text", language="")  # simulates un-tagged record
    with pytest.raises(ValueError) as exc:
        enforce_language_tagged(r)
    assert "constitutional" in str(exc.value).lower()


def test_enforce_language_tagged_caller_sets_default_explicitly():
    """Callers that want a default set it explicitly before calling the guard."""
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("hello world", language="")
    r.language = "en"  # caller's explicit default
    enforce_language_tagged(r)  # should not raise
    assert r.language == "en"


# ------------------------------------------------ enforce_english_raw shim


def test_enforce_english_raw_still_importable():
    """Backward compat: the guard is still a valid import."""
    from iai_mcp.aaak import enforce_english_raw

    assert callable(enforce_english_raw)


def test_enforce_english_raw_with_language_tag_still_phase1_semantics():
    """The shim preserves semantics: even with language='ru' set,
    untagged Cyrillic literal_surface WITHOUT 'raw:<lang>' tag still raises.

     callers who want native-language storage should call
    `enforce_language_tagged` instead of this shim.
    """
    from iai_mcp.aaak import enforce_english_raw

    r = _rec("привет мир", language="ru")
    with pytest.raises(ValueError):
        enforce_english_raw(r)


def test_enforce_english_raw_still_blocks_untagged_cyrillic():
    """behaviour preserved for untagged records (language="")."""
    from iai_mcp.aaak import enforce_english_raw

    r = _rec("привет мир", language="")
    with pytest.raises(ValueError) as exc:
        enforce_english_raw(r)
    assert "constitutional" in str(exc.value).lower()


def test_enforce_english_raw_accepts_cyrillic_with_raw_tag():
    """raw:<lang> tag exception still works through the shim."""
    from iai_mcp.aaak import enforce_english_raw

    r = _rec("привет мир", language="", tags=["raw:ru"])
    enforce_english_raw(r)


def test_enforce_english_raw_accepts_pure_english():
    from iai_mcp.aaak import enforce_english_raw

    r = _rec("hello world", language="")
    enforce_english_raw(r)
