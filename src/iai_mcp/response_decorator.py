"""Server-side profile knob decorator.

`apply_profile(response, profile)` mutates a response dict in place based on
the 11 sealed profile knobs. Every per-knob helper is silent-fail so a
malformed knob value can never break the response path.

Invariant: this module is pure-local Python. NO paid-API SDK import.
NO API-key env read. The static grep guard
`test_no_api_key_in_response_decorator` enforces the invariant at CI time.

Knob names never cross the MCP wire. They are read from the per-process
profile state, applied to the response here, and the result goes back over
JSON-RPC free of any knob identifiers.

Helper layout (10 dispatch helpers — one per AUTIST knob the decorator
mutates; wake_depth has no helper here, see end note):
- _apply_formality_relaxation (AUTIST-13 camouflaging_relaxation)
- _apply_monotropic_focus (AUTIST-01 monotropism_depth)
- _apply_literal_preservation (AUTIST-04)
- _apply_masking_off (AUTIST-06)
- _apply_task_support (AUTIST-07)
- _apply_scene_construction (AUTIST-14)
- _apply_dunn_quadrant (AUTIST-03)
- _apply_pda_tolerance (AUTIST-05 demand_avoidance_tolerance)
- _apply_interest_boost (AUTIST-09)
- _apply_inertia_awareness (AUTIST-10)

wake_depth (MCP-12) affects the session-start payload, not the response
shape, so it gets no helper here.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


#: HELPER_TO_KNOB_ID maps each apply_profile helper (and the
# upstream-gains / session-start virtual keys) to its knob requirement ID.
# Used by the dispatch loop to populate response['_knobs_applied'] with
# file:symbol provenance for every helper invocation. The table contains:
# - 8 helper-keyed entries (the AUTIST helpers wired in apply_profile that
# produce response-level mutations)
# - 2 upstream-gains entries (AUTIST-03 dunn_quadrant, AUTIST-09 interest_boost)
# — provenance strings are written by profile.py:profile_modulation_for_record;
# the dispatch loop ignores these virtual keys (HELPER_TO_KNOB_ID.get(...)
# returns None for them when keyed by helper name).
# - 1 session-start entry (MCP-12 wake_depth) — provenance points into
# session.py:assemble_session_start; written by core.dispatch.
#
# DO NOT re-add removed-knob keys (AUTIST-02 sensory_channel_weights,
# AUTIST-08 event_vs_time_cue, AUTIST-11 alexithymia_accommodation,
# AUTIST-12 double_empathy) — deleted them from the registry.
HELPER_TO_KNOB_ID: dict[str, str] = {
    # --- helper-keyed entries (8) — recorded by the dispatch loop -----------
    "_apply_monotropic_focus": "AUTIST-01",       # monotropism_depth
    "_apply_literal_preservation": "AUTIST-04",   # literal_preservation
    "_apply_pda_tolerance": "AUTIST-05",          # demand_avoidance_tolerance
    "_apply_masking_off": "AUTIST-06",            # masking_off
    "_apply_task_support": "AUTIST-07",           # task_support
    "_apply_inertia_awareness": "AUTIST-10",      # inertia_awareness
    "_apply_formality_relaxation": "AUTIST-13",   # camouflaging_relaxation
    "_apply_scene_construction": "AUTIST-14",     # scene_construction_scaffold
    # --- upstream-gains entries (2) — recorded by profile.py via the kwarg --
    # These are virtual lookup keys (NOT helper names). The dispatch loop's
    # HELPER_TO_KNOB_ID.get(helper_name) returns None for the existing pass-
    # through helpers _apply_dunn_quadrant / _apply_interest_boost because
    # those helpers are NOT in this table — the AUTHORITATIVE provenance for
    # the gain is profile.profile_modulation_for_record, written
    # by the upstream accumulator.
    "dunn_quadrant": "AUTIST-03",                 # via profile.py
    "interest_boost": "AUTIST-09",                # via profile.py
    # --- session-start entry (1) — recorded by core.dispatch ---------------
    # MCP-12 wake_depth is operator-facing; the seed entry is set in
    # core.dispatch when the session-start path runs. Provenance points
    # into assemble_session_start (wake_depth = state.get(...)).
    "wake_depth": "MCP-12",
}


def apply_profile(response: dict, profile: dict) -> dict:
    """Apply the 10 dispatch profile knobs to ``response`` in place.

    Contract:
    - Returns the same response for chainability.
    - Never raises. Each per-knob helper has its own try/except AND the
      central dispatch wraps every helper call with an outer guard so a
      monkey-patched or mis-named helper cannot break the hot path.
    - Malformed profile state is tolerated (unexpected types, missing keys).
    - No MCP-side knob names are added to the response.

     telemetry: emits response['_knobs_applied'] — a dict
    mapping knob requirement IDs (e.g., 'AUTIST-01') to deterministic
    file:symbol provenance strings. Future code-readers can audit, per
    response, which knobs actually mutated which fields.

    The accumulator is preserved across upstream paths: any entries
    seeded by core.dispatch BEFORE apply_profile runs (typically the
    upstream-gains entries for AUTIST-03 / AUTIST-09 and the wake_depth
    seed for MCP-12) survive — the dispatch loop only ADDS entries via
    helper-keyed lookup, never overwrites the dict shape.
    """
    if not isinstance(response, dict) or not isinstance(profile, dict):
        return response

    # Preserve any upstream-seeded entries.
    # core.dispatch seeds knobs_applied for AUTIST-03 / AUTIST-09 (via
    # profile_modulation_for_record) + MCP-12 wake_depth before this
    # function runs. We extend, never overwrite the dict reference held
    # by core.dispatch.
    pre_seeded = response.get("_knobs_applied")
    if isinstance(pre_seeded, dict):
        applied: dict[str, str] = pre_seeded
    else:
        applied = {}

    # Outer guard per helper call — tolerates a helper that was monkey-patched
    # to raise (seen in test_pre_existing_keys_untouched_on_exception) or an
    # accidental helper rewrite that skips the inner try/except.
    for helper in (
        _apply_formality_relaxation,
        _apply_monotropic_focus,
        _apply_literal_preservation,
        _apply_masking_off,
        _apply_task_support,
        _apply_scene_construction,
        _apply_dunn_quadrant,
        _apply_pda_tolerance,
        _apply_interest_boost,
        _apply_inertia_awareness,
    ):
        helper_raised = False
        try:
            helper(response, profile)
        except Exception as exc:
            logger.debug("profile helper %s failed: %s", helper.__name__, exc)
            helper_raised = True  # silent-fail per — no audit entry
        if helper_raised:
            continue
        helper_name = helper.__name__
        knob_id = HELPER_TO_KNOB_ID.get(helper_name)
        if knob_id is None:
            # Unmapped helper (e.g., _apply_dunn_quadrant, _apply_interest_boost
            # — their provenance lives in profile.py via the upstream gains
            # accumulator). Skip rather than corrupt the audit.
            continue
        provenance = f"response_decorator.py:{helper_name}"
        # No-op markers for mode-gate sites ("consulted and chose to do nothing"
        # vs "knob is dead").
        if helper_name == "_apply_pda_tolerance":
            mode = profile.get("demand_avoidance_tolerance", "collaborative")
            if mode == "neutral":
                provenance = f"{provenance}:no-op (mode=neutral)"
        elif helper_name == "_apply_inertia_awareness":
            if not profile.get("inertia_awareness", False):
                provenance = f"{provenance}:no-op (knob=False)"
            elif not response.get("first_turn_recall"):
                provenance = f"{provenance}:no-op (subsequent turn)"
        elif helper_name == "_apply_scene_construction":
            if not profile.get("scene_construction_scaffold", True):
                provenance = f"{provenance}:no-op (knob=False)"
        applied[knob_id] = provenance

    response["_knobs_applied"] = applied
    # wake_depth is the operator-facing knob; it drives session-start payload
    # shape, not response content. No helper here by design. Its
    # entry (MCP-12) is seeded by core.dispatch before apply_profile runs.
    return response


# ---------------------------------------------------------- per-knob helpers
# Each helper MUST be wrapped in try/except Exception: pass — a malformed
# profile knob value cannot break the hot recall path.


def _apply_formality_relaxation(response: dict, profile: dict) -> None:
    """AUTIST-13 camouflaging_relaxation > 0.5 -> rewrite surface_text toward
    informal register.

    The transform here is intentionally minimal (just strips trailing
    "Sir"/"Madam" honorifics). The weekly pass owns the heavy lift; this
    hook ensures response-time consistency.
    """
    try:
        level = float(profile.get("camouflaging_relaxation", 0.0))
        if level <= 0.5:
            return
        for hit in response.get("hits", []) or []:
            if not isinstance(hit, dict):
                continue
            text = hit.get("literal_surface") or hit.get("surface_text")
            if not isinstance(text, str):
                continue
            # Drop stale honorifics if present (best-effort).
            stripped = text
            for honorific in (" Sir.", " Sir,", " Madam.", " Madam,"):
                stripped = stripped.replace(honorific, ".")
            if "surface_text" in hit:
                hit["surface_text"] = stripped
            # Leave literal_surface byte-exact.
    except (ValueError, TypeError, KeyError) as exc:
        logger.debug("_apply_formality_relaxation: %s", exc)


def _apply_monotropic_focus(response: dict, profile: dict) -> None:
    """AUTIST-01 monotropism_depth per domain -> narrow top-k to dominant.

    When any domain in monotropism_depth has depth > 0.7, hits carrying a
    non-matching domain tag are down-ranked to the tail of the list. The
    transform is conservative: we reorder, never delete.
    """
    try:
        md = profile.get("monotropism_depth")
        if not isinstance(md, dict) or not md:
            return
        hot_domains = {d for d, depth in md.items() if _as_float(depth, 0.0) > 0.7}
        if not hot_domains:
            return
        hits = response.get("hits")
        if not isinstance(hits, list) or not hits:
            return
        def _key(h):
            if not isinstance(h, dict):
                return 1
            tags = h.get("tags") or []
            for t in tags:
                if isinstance(t, str) and t.startswith("domain:"):
                    return 0 if t.split(":", 1)[1] in hot_domains else 1
            return 1
        hits.sort(key=_key)
    except (ValueError, TypeError, KeyError) as exc:
        logger.debug("_apply_monotropic_focus: %s", exc)


def _apply_literal_preservation(response: dict, profile: dict) -> None:
    """AUTIST-04: strong -> keep literal_surface byte-exact (default); loose
    -> surface_text may be summarised. literal_surface is
    never mutated.
    """
    try:
        mode = profile.get("literal_preservation", "strong")
        if mode not in ("strong", "medium", "loose"):
            return
        # No-op by design: the hook exists for future summarisation logic but
        # must never mutate literal_surface per C5.
    except (ValueError, TypeError, KeyError) as exc:
        logger.debug("_apply_literal_preservation: %s", exc)


def _apply_masking_off(response: dict, profile: dict) -> None:
    """AUTIST-06: masking_off True -> strip performative empathy filler."""
    try:
        if not profile.get("masking_off", True):
            return
        filler = (
            "Great question! ",
            "Certainly! ",
            "Of course! ",
        )
        for hit in response.get("hits", []) or []:
            if not isinstance(hit, dict):
                continue
            txt = hit.get("surface_text")
            if isinstance(txt, str):
                for f in filler:
                    if txt.startswith(f):
                        hit["surface_text"] = txt[len(f):]
                        break
    except (ValueError, TypeError, KeyError) as exc:
        logger.debug("_apply_masking_off: %s", exc)


def _apply_task_support(response: dict, profile: dict) -> None:
    """AUTIST-07: cued_recognition -> adjacent_suggestions populated (no-op
    here because retrieve.recall already emits them); blank_recall -> strip
    suggestions to force free recall.
    """
    try:
        mode = profile.get("task_support", "cued_recognition")
        if mode != "blank_recall":
            return
        for hit in response.get("hits", []) or []:
            if isinstance(hit, dict) and "adjacent_suggestions" in hit:
                hit["adjacent_suggestions"] = []
    except (ValueError, TypeError, KeyError) as exc:
        logger.debug("_apply_task_support: %s", exc)


def _apply_scene_construction(response: dict, profile: dict) -> None:
    """AUTIST-14: scene_construction_scaffold autobiographical reconstruction
    hint.

    The hit dict from _hit_to_json does NOT carry tier/session_id/captured_at,
    so we drop the tier filter. When knob=True, attach
    _scene_hint to EVERY hit; downstream consumers ignore the hint on
    non-episodic content without harm. The 'advice' string is fixed —
    no LLM call.

    When False: no _scene_hint key added (test asserts absence).
    """
    try:
        if not profile.get("scene_construction_scaffold", True):
            return
        for hit in response.get("hits", []) or []:
            if not isinstance(hit, dict):
                continue
            hit["_scene_hint"] = {
                "session_id": hit.get("session_id"),
                "captured_at": hit.get("captured_at"),
                "advice": "use as scaffold for autobiographical reconstruction",
            }
    except (ValueError, TypeError, KeyError) as exc:
        logger.debug("_apply_scene_construction: %s", exc)


def _apply_dunn_quadrant(response: dict, profile: dict) -> None:
    """AUTIST-03: dunn_quadrant -> precision is upstream; no-op here."""
    try:
        _ = profile.get("dunn_quadrant", "neutral")
    except (ValueError, TypeError, KeyError) as exc:
        logger.debug("_apply_dunn_quadrant: %s", exc)


def _apply_pda_tolerance(response: dict, profile: dict) -> None:
    """AUTIST-05: demand_avoidance_tolerance lexical softener.

    - collaborative (default): replace leading imperatives in each
      adjacent_suggestion entry per the frozen substitution table. Only
      first-word matches; mid-sentence
      imperatives are NOT touched (avoids false positives in code blocks).
    - avoidant: prepend 'FYI: ' to every adjacent_suggestion entry.
    - neutral: bypass.
    """
    try:
        mode = profile.get("demand_avoidance_tolerance", "collaborative")
        if mode == "neutral":
            return
        if mode == "avoidant":
            for hit in response.get("hits", []) or []:
                if not isinstance(hit, dict):
                    continue
                suggestions = hit.get("adjacent_suggestions")
                if not isinstance(suggestions, list):
                    continue
                hit["adjacent_suggestions"] = [
                    f"FYI: {entry}" for entry in suggestions
                ]
            return
        if mode == "collaborative":
            # Frozen substitution table.
            substitutions: tuple[tuple[str, str], ...] = (
                ("Try ", "You could try "),
                ("Do ", "Consider "),
                ("Use ", "Try using "),
                ("Run ", "Try running "),
            )
            for hit in response.get("hits", []) or []:
                if not isinstance(hit, dict):
                    continue
                suggestions = hit.get("adjacent_suggestions")
                if not isinstance(suggestions, list):
                    continue
                rewritten: list = []
                for entry in suggestions:
                    if not isinstance(entry, str):
                        rewritten.append(entry)
                        continue
                    new_entry = entry
                    for prefix, replacement in substitutions:
                        if entry.startswith(prefix):
                            new_entry = replacement + entry[len(prefix):]
                            break
                    rewritten.append(new_entry)
                hit["adjacent_suggestions"] = rewritten
    except (ValueError, TypeError, KeyError) as exc:
        logger.debug("_apply_pda_tolerance: %s", exc)


def _apply_interest_boost(response: dict, profile: dict) -> None:
    """AUTIST-09: interest_boost > 0 -> amplify hits in interest domains.
    Applied during scoring, not at response rewrite time; no-op here.
    """
    try:
        _ = profile.get("interest_boost", 0.0)
    except (ValueError, TypeError, KeyError) as exc:
        logger.debug("_apply_interest_boost: %s", exc)


def _apply_inertia_awareness(response: dict, profile: dict) -> None:
    """AUTIST-10: inertia_awareness session-resumption cue.

    The upstream hook sets response["first_turn_recall"]
    to a DICT, not a bool. The gate MUST be a shape-agnostic truthy check —
    `is True` equality would silent-no-op in production.

    When knob=True AND response["first_turn_recall"] is truthy (set by
    _first_turn_recall_hook on the first turn of a
    session), prepend a one-line resumption cue to the top-1 hit's
    literal_surface. The text is fixed (not LLM-generated) for determinism.

    The per-recall fallback is explicitly forbidden: if the
    first_turn_recall flag is unreliable, escalate via checkpoint rather
    than silently re-introducing recall-noise.

    Subsequent turns OR knob=False → no transform; literal_surface stays
    byte-exact.
    """
    try:
        if not profile.get("inertia_awareness", False):
            return
        # Truthy presence check — shape-agnostic (works for dict OR bool).
        # The upstream hook sets this to a dict on the first turn; the truthy
        # check covers both production (dict) and any test path (bool).
        if not response.get("first_turn_recall"):
            return
        hits = response.get("hits") or []
        if not hits:
            return
        top = hits[0]
        if not isinstance(top, dict):
            return
        literal = top.get("literal_surface")
        if not isinstance(literal, str):
            return
        top["literal_surface"] = f"Resuming from your last session: {literal}"
    except (ValueError, TypeError, KeyError) as exc:
        logger.debug("_apply_inertia_awareness: %s", exc)


# ----------------------------------------------------------------- utilities
def _as_float(value, default: float) -> float:
    """Coerce ``value`` to float; return ``default`` on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
