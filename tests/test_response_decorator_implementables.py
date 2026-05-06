"""Phase 07.12-01: AUTIST-05/10/14 implementables — visible response delta.

Closes the second leg of "не оставлять пустышки": each helper now produces
a measurable, test-asserted mutation when its knob value flips.

CONTEXT freezes the substitution tables — these tests assert the
EXACT strings, no executor invention permitted.

CONTEXT BLOCKER 1 fix: the True-branch fixture uses the
PRODUCTION DICT SHAPE from core.py:1178 (NOT a bare bool). The helper
gate is `if not response.get("first_turn_recall"): return` — shape-
agnostic truthy presence check.
"""
import copy

from iai_mcp.response_decorator import apply_profile


def _hit(literal: str, suggestions: list[str] | None = None) -> dict:
    """Build a synthetic hit dict matching _hit_to_json shape (core.py:712-719)."""
    return {
        "record_id": "00000000-0000-0000-0000-000000000001",
        "score": 0.5,
        "reason": "test",
        "literal_surface": literal,
        "adjacent_suggestions": suggestions or [],
    }


def _resp(hits: list[dict], **extra) -> dict:
    """Build a synthetic response dict with optional top-level fields."""
    base = {"hits": hits}
    base.update(extra)
    return base


def _first_turn_recall_dict() -> dict:
    """Production dict shape from core.py:1178 — used as the truthy
    fixture value for AUTIST-10's True-branch test.

    Shape verified against live source 2026-04-30:
        response["first_turn_recall"] = {
            "hits": [...],
            "budget_tokens": 400,
            "budget_used": ...,
            "warm_lru_size": ...,
            "warm_lru_source": ...,
        }
    """
    return {
        "hits": [],
        "budget_tokens": 400,
        "budget_used": 0,
        "warm_lru_size": 0,
        "warm_lru_source": "none",
    }


# ---- demand_avoidance_tolerance ----------------------------------

def test_pda_tolerance_collaborative_softens_imperatives() -> None:
    """CONTEXT frozen table: 'Try X' → 'You could try X', etc.

    Substitution applies ONLY to first-word imperative match; mid-sentence
    imperatives are NOT touched.
    """
    response = _resp([
        _hit("orig", suggestions=[
            "Try refactoring X",
            "Do the migration",
            "Use bge-small embedder",
            "Run pytest -q",
            "If you Try refactoring later, beware",  # mid-sentence — NOT touched
        ]),
    ])
    profile = {"demand_avoidance_tolerance": "collaborative"}
    apply_profile(response, profile)
    assert response["hits"][0]["adjacent_suggestions"] == [
        "You could try refactoring X",
        "Consider the migration",
        "Try using bge-small embedder",
        "Try running pytest -q",
        "If you Try refactoring later, beware",
    ]


def test_pda_tolerance_avoidant_prepends_fyi() -> None:
    """CONTEXT avoidant mode prepends 'FYI: ' to every entry."""
    response = _resp([_hit("orig", suggestions=["Try X", "Run Y", "ad-hoc note"])])
    profile = {"demand_avoidance_tolerance": "avoidant"}
    apply_profile(response, profile)
    assert response["hits"][0]["adjacent_suggestions"] == [
        "FYI: Try X",
        "FYI: Run Y",
        "FYI: ad-hoc note",
    ]


def test_pda_tolerance_neutral_no_op() -> None:
    """CONTEXT neutral mode bypasses — byte-equal to input.

    Disable (default=True) for isolation; assertion is on the
    surface only.
    """
    suggestions = ["Try X", "Run Y", "ad-hoc note"]
    response = _resp([_hit("orig", suggestions=list(suggestions))])
    snapshot = copy.deepcopy(response)
    profile = {
        "demand_avoidance_tolerance": "neutral",
        "scene_construction_scaffold": False,
    }
    apply_profile(response, profile)
    # Plan 07.12-03: apply_profile now adds _knobs_applied; strip it before
    # the byte-equality check so the / mutation surfaces
    # are isolated.
    response.pop("_knobs_applied", None)
    assert response == snapshot


# ---- inertia_awareness -------------------------------------------

def test_inertia_awareness_first_turn_prefixes_resume() -> None:
    """CONTEXT BLOCKER 1 fix: when knob=True AND first_turn_recall
    is truthy (dict OR bool — production path uses the dict at core.py:1178),
    prefix top-1 hit's literal_surface with 'Resuming from your last session: '.

    Fixture uses the production dict shape — shape-agnostic gate must
    treat it as truthy.
    """
    response = _resp(
        [_hit("orig hit 1"), _hit("orig hit 2")],
        first_turn_recall=_first_turn_recall_dict(),
    )
    profile = {"inertia_awareness": True}
    apply_profile(response, profile)
    assert response["hits"][0]["literal_surface"] == (
        "Resuming from your last session: orig hit 1"
    )
    # Second hit untouched — only top-1 gets the cue.
    assert response["hits"][1]["literal_surface"] == "orig hit 2"


def test_inertia_awareness_subsequent_turn_no_op() -> None:
    """CONTEXT when first_turn_recall is absent (subsequent turn),
    no prefix even when knob=True.

    Disable (default=True) for isolation; assertion is on the
    literal_surface only.
    """
    response = _resp([_hit("orig hit")])  # no first_turn_recall key
    snapshot = copy.deepcopy(response)
    profile = {
        "inertia_awareness": True,
        "scene_construction_scaffold": False,
    }
    apply_profile(response, profile)
    # Plan 07.12-03: strip _knobs_applied for byte-equality isolation.
    response.pop("_knobs_applied", None)
    assert response == snapshot


def test_inertia_awareness_off_no_op() -> None:
    """CONTEXT knob=False → no prefix even on first turn (with the
    production dict-shaped first_turn_recall present).

    Disable (default=True) for isolation.
    """
    response = _resp(
        [_hit("orig hit")],
        first_turn_recall=_first_turn_recall_dict(),
    )
    snapshot = copy.deepcopy(response)
    profile = {
        "inertia_awareness": False,
        "scene_construction_scaffold": False,
    }
    apply_profile(response, profile)
    # Plan 07.12-03: strip _knobs_applied for byte-equality isolation.
    response.pop("_knobs_applied", None)
    assert response == snapshot


# ---- scene_construction_scaffold ---------------------------------

def test_scene_construction_attaches_hint_when_true() -> None:
    """CONTEXT + PATTERNS option-3 reconciliation: drop tier filter,
    attach _scene_hint to EVERY hit when knob=True."""
    response = _resp([_hit("h1"), _hit("h2")])
    profile = {"scene_construction_scaffold": True}
    apply_profile(response, profile)
    for hit in response["hits"]:
        assert "_scene_hint" in hit, hit
        assert hit["_scene_hint"]["advice"] == (
            "use as scaffold for autobiographical reconstruction"
        )
        # session_id / captured_at are None when not present on the hit dict.
        assert hit["_scene_hint"]["session_id"] is None
        assert hit["_scene_hint"]["captured_at"] is None


def test_scene_construction_no_hint_when_false() -> None:
    """CONTEXT knob=False → no _scene_hint key on any hit."""
    response = _resp([_hit("h1"), _hit("h2")])
    profile = {"scene_construction_scaffold": False}
    apply_profile(response, profile)
    for hit in response["hits"]:
        assert "_scene_hint" not in hit, hit
