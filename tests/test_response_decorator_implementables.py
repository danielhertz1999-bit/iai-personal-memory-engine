import copy

from iai_mcp.response_decorator import apply_profile

def _hit(literal: str, suggestions: list[str] | None = None) -> dict:
    return {
        "record_id": "00000000-0000-0000-0000-000000000001",
        "score": 0.5,
        "reason": "test",
        "literal_surface": literal,
        "adjacent_suggestions": suggestions or [],
    }

def _resp(hits: list[dict], **extra) -> dict:
    base = {"hits": hits}
    base.update(extra)
    return base

def _first_turn_recall_dict() -> dict:
    return {
        "hits": [],
        "budget_tokens": 400,
        "budget_used": 0,
        "warm_lru_size": 0,
        "warm_lru_source": "none",
    }

def test_pda_tolerance_collaborative_softens_imperatives() -> None:
    response = _resp([
        _hit("orig", suggestions=[
            "Try refactoring X",
            "Do the migration",
            "Use bge-small embedder",
            "Run pytest -q",
            "If you Try refactoring later, beware",
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
    response = _resp([_hit("orig", suggestions=["Try X", "Run Y", "ad-hoc note"])])
    profile = {"demand_avoidance_tolerance": "avoidant"}
    apply_profile(response, profile)
    assert response["hits"][0]["adjacent_suggestions"] == [
        "FYI: Try X",
        "FYI: Run Y",
        "FYI: ad-hoc note",
    ]

def test_pda_tolerance_neutral_no_op() -> None:
    suggestions = ["Try X", "Run Y", "ad-hoc note"]
    response = _resp([_hit("orig", suggestions=list(suggestions))])
    snapshot = copy.deepcopy(response)
    profile = {
        "demand_avoidance_tolerance": "neutral",
        "scene_construction_scaffold": False,
    }
    apply_profile(response, profile)
    response.pop("_knobs_applied", None)
    assert response == snapshot

def test_inertia_awareness_first_turn_prefixes_resume() -> None:
    response = _resp(
        [_hit("orig hit 1"), _hit("orig hit 2")],
        first_turn_recall=_first_turn_recall_dict(),
    )
    profile = {"inertia_awareness": True}
    apply_profile(response, profile)
    assert response["hits"][0]["literal_surface"] == (
        "Resuming from your last session: orig hit 1"
    )
    assert response["hits"][1]["literal_surface"] == "orig hit 2"

def test_inertia_awareness_subsequent_turn_no_op() -> None:
    response = _resp([_hit("orig hit")])
    snapshot = copy.deepcopy(response)
    profile = {
        "inertia_awareness": True,
        "scene_construction_scaffold": False,
    }
    apply_profile(response, profile)
    response.pop("_knobs_applied", None)
    assert response == snapshot

def test_inertia_awareness_off_no_op() -> None:
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
    response.pop("_knobs_applied", None)
    assert response == snapshot

def test_scene_construction_attaches_hint_when_true() -> None:
    response = _resp([_hit("h1"), _hit("h2")])
    profile = {"scene_construction_scaffold": True}
    apply_profile(response, profile)
    for hit in response["hits"]:
        assert "_scene_hint" in hit, hit
        assert hit["_scene_hint"]["advice"] == (
            "use as scaffold for autobiographical reconstruction"
        )
        assert hit["_scene_hint"]["session_id"] is None
        assert hit["_scene_hint"]["captured_at"] is None

def test_scene_construction_no_hint_when_false() -> None:
    response = _resp([_hit("h1"), _hit("h2")])
    profile = {"scene_construction_scaffold": False}
    apply_profile(response, profile)
    for hit in response["hits"]:
        assert "_scene_hint" not in hit, hit
