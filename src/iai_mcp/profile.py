"""11-knob profile registry ( + wake_depth, removals).

activated the Phase-2 autistic-kernel knobs. flipped
AUTIST-13 camouflaging_relaxation to live. appended the sealed
operator-facing knob `wake_depth` — selects session-start payload size
(minimal = <=30 raw tok lazy handle; standard = Phase-1 1388 tok eager dump;
deep = <=2000 tok expanded rich_club). REMOVED 4 dead KnobSpec
entries (AUTIST-02 sensory_channel_weights, event_vs_time_cue,
AUTIST-11 alexithymia_accommodation, double_empathy) — none was
read in any production scoring/response path; double_empathy was promoted
to a passive system invariant in CLAUDE.md, event_vs_time_cue was documented
as a deferred future capability.

Registry shape:
- 10 live autistic-kernel knobs (AUTIST-01,03,04,05,06,07,09,10,13,14)
- 1 live Phase-5 operator knob (MCP-12 wake_depth, default "minimal")
- 0 deferred

The registry is a module-level frozen-dataclass dict so
   1. `assert len(PROFILE_KNOBS) == 11`
   2. test_profile.py can grep exact knob names in order
   3. Session-start assembler reads the live subset in O(1)

Schema validation covers:
- `enum:a|b|c`            -- value must be exactly one of the listed tokens
- `bool`                  -- isinstance(value, bool)
- `int_range:lo..hi`      -- integer in [lo, hi] inclusive
- `float_range:lo..hi`    -- float in [lo, hi] inclusive
- `dict:<keytype>:<valuetype>` -- per-key recursive validation
                                  (e.g. `dict:str:float_range:0.0..1.0`)
- anything else           -- reject (typo guard)

runtime-gain mechanism exposed via two helpers:
- bayesian_update: weighted ensemble posterior update
- profile_modulation_for_record: per-record edge-weight gain dict
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# --------------------------------------------------------------------- schema
@dataclass(frozen=True)
class KnobSpec:
    """Static spec for one autistic-kernel knob."""

    name: str
    phase: int                 # 1 | 2 | 3
    default: Any               # Phase-1 default, or Phase-2/3 placeholder default
    description: str
    value_schema: str          # "enum:a|b|c" | "bool" | "int_range:0..5" | "float_range:0.0..1.0"
    requirement_id: str        # AUTIST-01..14


# ------------------------------------------------------------------ registry
# 11 sealed knobs: 10 autistic-kernel + wake_depth
# (removed sensory_channel_weights, AUTIST-08
# event_vs_time_cue, alexithymia_accommodation, double_empathy).
# flipped 9 Phase-2 knobs to phase=1.
# flipped camouflaging_relaxation to phase=1.
# appended wake_depth (MCP-12, operator-facing).
PROFILE_KNOBS: dict[str, KnobSpec] = {
    "monotropism_depth": KnobSpec(
        "monotropism_depth",
        1,
        {},  # per-domain dict; empty default (unknown domains -> no gain)
        "Monotropism depth per domain (voluntary tunnel; HIPPEA precision)",
        "dict:str:float_range:0.0..1.0",
        "AUTIST-01",
    ),
    "dunn_quadrant": KnobSpec(
        "dunn_quadrant",
        1,
        "neutral",
        "Sensory threshold x regulation posture (Dunn four-quadrant; "
        "drives HIPPEA precision weighting at runtime)",
        "enum:neutral|low-registration|seeking|sensitive|avoiding",
        "AUTIST-03",
    ),
    "literal_preservation": KnobSpec(
        "literal_preservation",
        1,
        "strong",
        "Verbatim vs semantic summary (raw always retained)",
        "enum:strong|medium|loose",
        "AUTIST-04",
    ),
    "demand_avoidance_tolerance": KnobSpec(
        "demand_avoidance_tolerance",
        1,
        "collaborative",
        "PDA-aware collaborative phrasing vs imperative",
        "enum:collaborative|neutral|imperative",
        "AUTIST-05",
    ),
    "masking_off": KnobSpec(
        "masking_off",
        1,
        True,
        "No small-talk, no performative empathy, literal pragmatics",
        "bool",
        "AUTIST-06",
    ),
    "task_support": KnobSpec(
        "task_support",
        1,
        "cued_recognition",
        "Blank-recall vs cued-recognition with adjacent suggestions (Bowler)",
        "enum:blank_recall|cued_recognition",
        "AUTIST-07",
    ),
    "interest_boost": KnobSpec(
        "interest_boost",
        1,
        0.0,
        "Salience amplification adjacent to monotropism domains",
        "float_range:0.0..1.0",
        "AUTIST-09",
    ),
    "inertia_awareness": KnobSpec(
        "inertia_awareness",
        1,
        False,
        "Ambient passive capture in high-inertia windows",
        "bool",
        "AUTIST-10",
    ),
    "camouflaging_relaxation": KnobSpec(
        "camouflaging_relaxation",
        1,
        0.0,
        "Detect over-formal writing, gradually relax formality (live)",
        "float_range:0.0..1.0",
        "AUTIST-13",
    ),
    "scene_construction_scaffold": KnobSpec(
        "scene_construction_scaffold",
        1,
        True,
        "Scene-construction scaffold intensity for episodic encoding",
        "bool",
        "AUTIST-14",
    ),
    # D5-06: 15th sealed knob (operator-facing, not autistic-kernel).
    # wake_depth drives session-start payload size. minimal (default) = ≤30 raw
    # tok pointer handle (lazy; brain stays server-side); standard = Phase-1
    # 1388 tok eager dump (back-compat per D5-10); deep = ≤2000 tok expanded
    # rich_club. Set via existing profile_get_set tool; no new MCP surface.
    "wake_depth": KnobSpec(
        "wake_depth",
        1,  # phase — live in (counts toward PHASE_1_LIVE)
        "minimal",
        (
            "Session-start payload size: minimal=<=30 raw (lazy, default), "
            "standard=Phase-1 eager (back-compat), deep=<=2000 (full)"
        ),
        "enum:minimal|standard|deep",
        "MCP-12",
    ),
}


PHASE_1_LIVE: frozenset[str] = frozenset(
    {name for name, spec in PROFILE_KNOBS.items() if spec.phase == 1}
)
PHASE_2_DEFERRED: frozenset[str] = frozenset(
    {name for name, spec in PROFILE_KNOBS.items() if spec.phase == 2}
)
PHASE_3_DEFERRED: frozenset[str] = frozenset(
    {name for name, spec in PROFILE_KNOBS.items() if spec.phase == 3}
)


# : 11-knob shape is load-bearing. Enforced at import time.
# History:
# - flipped the 9 Phase-2 knobs to phase=1 (PHASE_1_LIVE=13).
# - FLIPPED camouflaging_relaxation to phase=1 (PHASE_1_LIVE=14).
# - APPENDS wake_depth as the 15th sealed knob (PHASE_1_LIVE=15).
# - REMOVES 4 dead KnobSpec entries (AUTIST-02 sensory,
#   event_vs_time_cue, alexithymia, double_empathy).
#   Final shape: 10 AUTIST + 1 wake_depth = 11 sealed knobs.
assert len(PROFILE_KNOBS) == 11, (
    ": 10 autistic-kernel knobs + wake_depth = 11 sealed entries"
)
assert len(PHASE_1_LIVE) == 11, (
    ": 10 autistic-kernel knobs + wake_depth are live"
)
assert len(PHASE_2_DEFERRED) == 0, "empties PHASE_2_DEFERRED"
assert len(PHASE_3_DEFERRED) == 0, "PHASE_3_DEFERRED emptied"


# Bayesian signal weights (LEARN-01)
SIGNAL_WEIGHT: dict[str, float] = {
    "implicit": 0.3,
    "inferred": 0.5,
    "explicit": 1.0,
}


# profile sentinel UUID -- target node for every profile_modulates edge.
# Deterministic so the edges table can be scanned without a side table. The
# UUID is ff-nonsense so no record ever collides with it.
PROFILE_SENTINEL_UUID_STR = "00000000-0000-0000-0000-0000000000f1"


# --------------------------------------------------------------------- state
def default_state() -> dict[str, Any]:
    """Initial per-process state: the live knobs with defaults.

    Deferred knobs do not appear in state because profile_set rejects them;
    profile_get on a deferred knob returns status/phase/requirement_id directly
    from the registry.
    """
    return {
        name: spec.default
        for name, spec in PROFILE_KNOBS.items()
        if spec.phase == 1
    }


# ---------------------------------------------------------------- validation
def _validate(schema: str, value: Any) -> tuple[bool, str]:
    """Return (ok, reason). Reason empty on success.

    extends the validators to support `dict:<keytype>:<valuetype>`
    via recursive per-key validation. Unknown schemas (typos) are rejected.
    """
    if schema == "bool":
        # Note: `isinstance(True, int)` is True in Python, so check bool first.
        if isinstance(value, bool):
            return True, ""
        return False, f"value must be bool, got {type(value).__name__}"

    if schema.startswith("enum:"):
        allowed = schema[len("enum:"):].split("|")
        if value in allowed:
            return True, ""
        return False, f"value {value!r} not in enum {allowed}"

    if schema.startswith("int_range:"):
        bounds = schema[len("int_range:"):]
        try:
            lo_s, hi_s = bounds.split("..")
            lo, hi = int(lo_s), int(hi_s)
        except (ValueError, TypeError):
            return False, f"malformed int_range schema {schema!r}"
        if isinstance(value, bool):
            return False, "value must be int, got bool"
        if not isinstance(value, int):
            return False, f"value must be int, got {type(value).__name__}"
        if value < lo or value > hi:
            return False, f"value {value} out of range [{lo}, {hi}]"
        return True, ""

    if schema.startswith("float_range:"):
        bounds = schema[len("float_range:"):]
        try:
            lo_s, hi_s = bounds.split("..")
            lo, hi = float(lo_s), float(hi_s)
        except (ValueError, TypeError):
            return False, f"malformed float_range schema {schema!r}"
        if isinstance(value, bool):
            return False, "value must be float, got bool"
        if not isinstance(value, (int, float)):
            return False, f"value must be float, got {type(value).__name__}"
        v = float(value)
        if v < lo or v > hi:
            return False, f"value {v} out of range [{lo}, {hi}]"
        return True, ""

    if schema.startswith("dict:"):
        body = schema[len("dict:"):]
        key_type, _, val_type = body.partition(":")
        if not val_type:
            return False, f"malformed dict schema {schema!r}"
        if not isinstance(value, dict):
            return False, f"value must be dict, got {type(value).__name__}"
        for k, v in value.items():
            if key_type == "str" and not isinstance(k, str):
                return False, f"dict key must be str, got {type(k).__name__}"
            ok, reason = _validate(val_type, v)
            if not ok:
                return False, f"in key {k!r}: {reason}"
        return True, ""

    # Unknown schema -> reject (covers accidental typos in KnobSpec.value_schema).
    return False, f"unknown value_schema {schema!r}"


# ------------------------------------------------------------- public surface
def profile_get(knob: str | None, state: dict[str, Any]) -> dict:
    """Read a knob (or the full registry surface).

    - knob=None -> full registry: {live: {11}, deferred: {0}, total_knobs: 11}.
    - knob in PHASE_1_LIVE -> {"knob": n, "value": state[n]}.
    - knob in deferred (P3) -> status/phase/requirement_id payload.
    - unknown knob -> {"knob": n, "status": "unknown"}.

    : total_knobs is 11 (10 AUTIST + wake_depth) after AUTIST-02/08/11/12 removal.
    """
    if knob is None:
        live = {
            n: state.get(n, PROFILE_KNOBS[n].default)
            for n in sorted(PHASE_1_LIVE)
        }
        deferred = {}
        for n in sorted(PHASE_2_DEFERRED | PHASE_3_DEFERRED):
            spec = PROFILE_KNOBS[n]
            deferred[n] = {
                "status": "not-yet-implemented",
                "phase": spec.phase,
                "requirement_id": spec.requirement_id,
                "description": spec.description,
            }
        return {"live": live, "deferred": deferred, "total_knobs": 11}

    if knob in PHASE_1_LIVE:
        spec = PROFILE_KNOBS[knob]
        return {"knob": knob, "value": state.get(knob, spec.default)}

    if knob in PROFILE_KNOBS:
        spec = PROFILE_KNOBS[knob]
        return {
            "knob": knob,
            "status": "not-yet-implemented",
            "phase": spec.phase,
            "requirement_id": spec.requirement_id,
        }

    return {"knob": knob, "status": "unknown"}


def profile_set(
    knob: str,
    value: Any,
    state: dict[str, Any],
    *,
    store: "object | None" = None,
) -> dict:
    """Write a live knob. Rejects unknown/deferred/invalid-value writes.

    Rule priority:
      1. unknown knob  -> {"status": "error", "reason": "unknown knob"}
      2. Phase-2 knob -> {"status": "error", "reason": "deferred to "}
         (empties this set but the branch is retained for safety.)
      3. Phase-3 knob -> {"status": "error", "reason": "deferred to "}
      4. schema fail   -> {"status": "error", "reason": <validator message>}
      5. success       -> mutates state; returns {"status": "ok", knob, value}

    (M4 LIVE prerequisite): when ``store`` is provided AND the
    write actually changes the value, emit ``kind='profile_updated'`` so
    M4 profile-variance can be computed live. No-op writes (old == new) do
    NOT emit (avoid event flood). The ``store`` kwarg is optional so old
    callers (e.g. core.dispatch profile_set branch) keep working unchanged.
    """
    if knob not in PROFILE_KNOBS:
        return {"status": "error", "reason": "unknown knob", "knob": knob}

    spec = PROFILE_KNOBS[knob]
    if spec.phase == 2:
        return {
            "status": "error",
            "reason": "deferred to ",
            "knob": knob,
            "requirement_id": spec.requirement_id,
        }
    if spec.phase == 3:
        return {
            "status": "error",
            "reason": "deferred to ",
            "knob": knob,
            "requirement_id": spec.requirement_id,
        }

    ok, reason = _validate(spec.value_schema, value)
    if not ok:
        return {
            "status": "error",
            "reason": reason,
            "knob": knob,
            "schema": spec.value_schema,
        }

    old_value = state.get(knob, spec.default)
    state[knob] = value

    # M4 LIVE: emit only on actual change to avoid no-op flood.
    if store is not None and old_value != value:
        try:
            from datetime import datetime, timezone
            from iai_mcp.events import write_event
            write_event(
                store,
                kind="profile_updated",
                data={
                    "knob": knob,
                    "old": old_value,
                    "new": value,
                    "requirement_id": spec.requirement_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                severity="info",
            )
        except Exception:
            # Diagnostic only: never block the profile_set on emit failure.
            pass

    return {"status": "ok", "knob": knob, "value": value}


# ---------------------------------------------------------------- Bayesian


def bayesian_update(
    knob: str,
    signal: str,
    observed: Any,
    state: dict,
    posterior: dict,
) -> tuple[Any, dict]:
    """ weighted-ensemble posterior update on a knob value.

    Conjugate-prior form per schema type:
      - bool        -> Beta(alpha, beta); alpha += w*obs, beta += w*(1-obs)
                       New value is the Beta mode (alpha > beta -> True).
      - enum        -> Dirichlet(alphas); alphas[obs] += w
                       New value is argmax(alphas).
      - float_range -> Normal mean via weighted running average
      - int_range   -> rounded weighted running average
      - dict:...    -> per-key recursive update (observed must also be a dict)

    Returns (new_value, new_posterior). `posterior` is a dict keyed by knob
    name with an internal per-knob sub-dict carrying alpha/beta/alphas/mean/n.
    """
    w = SIGNAL_WEIGHT.get(signal, 0.0)
    if w == 0.0:
        return state.get(knob, PROFILE_KNOBS[knob].default if knob in PROFILE_KNOBS else None), posterior

    spec = PROFILE_KNOBS.get(knob)
    if spec is None:
        return state.get(knob), posterior

    sch = spec.value_schema
    p = dict(posterior)
    kp = dict(p.get(knob, {}))

    current = state.get(knob, spec.default)

    if sch == "bool":
        alpha = float(kp.get("alpha", 1.0))
        beta = float(kp.get("beta", 1.0))
        if observed is True:
            alpha += w
        elif observed is False:
            beta += w
        else:
            # Invalid observation for bool; degrade silently.
            return current, p
        kp["alpha"] = alpha
        kp["beta"] = beta
        new_value = alpha >= beta
    elif sch.startswith("enum:"):
        allowed = sch[len("enum:"):].split("|")
        alphas: dict[str, float] = dict(kp.get("alphas", {}))
        if observed not in allowed:
            return current, p
        alphas[observed] = alphas.get(observed, 1.0) + w
        kp["alphas"] = alphas
        # Seed with current as implicit prior boost if no entries yet.
        if current in allowed and current not in alphas:
            alphas[current] = alphas.get(current, 1.0) + 0.001
        new_value = max(alphas.keys(), key=lambda k: alphas[k])
    elif sch.startswith("float_range:"):
        # Weighted running mean.
        try:
            obs_f = float(observed)
        except (TypeError, ValueError):
            return current, p
        prev_sum = float(kp.get("weighted_sum", float(current) if isinstance(current, (int, float)) else 0.0))
        prev_wts = float(kp.get("total_weight", 0.0))
        new_sum = prev_sum + w * obs_f
        new_wts = prev_wts + w
        mean = new_sum / new_wts if new_wts > 0 else obs_f
        # Clamp to the schema range.
        bounds = sch[len("float_range:"):]
        lo_s, hi_s = bounds.split("..")
        lo, hi = float(lo_s), float(hi_s)
        mean = max(lo, min(hi, mean))
        kp["weighted_sum"] = new_sum
        kp["total_weight"] = new_wts
        kp["mean"] = mean
        new_value = mean
    elif sch.startswith("int_range:"):
        try:
            obs_f = float(observed)
        except (TypeError, ValueError):
            return current, p
        prev_sum = float(kp.get("weighted_sum", float(current) if isinstance(current, (int, float)) else 0.0))
        prev_wts = float(kp.get("total_weight", 0.0))
        new_sum = prev_sum + w * obs_f
        new_wts = prev_wts + w
        mean = new_sum / new_wts if new_wts > 0 else obs_f
        bounds = sch[len("int_range:"):]
        lo_s, hi_s = bounds.split("..")
        lo, hi = int(lo_s), int(hi_s)
        new_value = max(lo, min(hi, int(round(mean))))
        kp["weighted_sum"] = new_sum
        kp["total_weight"] = new_wts
        kp["mean"] = mean
    elif sch.startswith("dict:"):
        # Per-key recursive update. `observed` must be dict-of-same-shape.
        if not isinstance(observed, dict):
            return current, p
        body = sch[len("dict:"):]
        _key_type, _, val_type = body.partition(":")
        per_key_posts: dict[str, dict] = dict(kp.get("per_key", {}))
        current_dict: dict = dict(current) if isinstance(current, dict) else {}
        for k, v in observed.items():
            # Mini-recursion: synthesise a float-style update for the inner value.
            sub_spec = val_type
            sub_kp = dict(per_key_posts.get(k, {}))
            if sub_spec.startswith("float_range:"):
                try:
                    obs_f = float(v)
                except (TypeError, ValueError):
                    continue
                prev_sum = float(sub_kp.get("weighted_sum", float(current_dict.get(k, 0.0))))
                prev_wts = float(sub_kp.get("total_weight", 0.0))
                new_sum = prev_sum + w * obs_f
                new_wts = prev_wts + w
                mean = new_sum / new_wts if new_wts > 0 else obs_f
                bounds = sub_spec[len("float_range:"):]
                lo_s, hi_s = bounds.split("..")
                lo, hi = float(lo_s), float(hi_s)
                mean = max(lo, min(hi, mean))
                sub_kp["weighted_sum"] = new_sum
                sub_kp["total_weight"] = new_wts
                sub_kp["mean"] = mean
                per_key_posts[k] = sub_kp
                current_dict[k] = mean
        kp["per_key"] = per_key_posts
        new_value = current_dict
    else:
        return current, p

    p[knob] = kp
    state[knob] = new_value
    return new_value, p


# ---------------------------------------------------------------- gain


def profile_modulation_for_record(
    record,
    profile_state: dict,
    *,
    knobs_applied: dict | None = None,
) -> dict[str, float]:
    """Compute edge-weight gain dict for a record.

    Returned gains are multiplicative (>=1.0 means amplify, <1.0 means damp).
    Keys match the knob name. Empty dict means no active modulation.

    Current gain sources:
    - `monotropism_depth`: gain = 1.0 + depth for the record's domain tag.
    - `interest_boost`: gain = 1.0 + boost (amplifies every record).
    - `dunn_quadrant`: seeking -> 1.2, avoiding -> 0.8, else no entry.
    - `special_interest_amplification`: extension (no-op here).

    The record's own `profile_modulation_gain` dict is NOT mutated here; the
    caller (pipeline_recall) copies the gains onto the record cache after
    computing them.

    -03: when ``knobs_applied`` is provided (a dict), records
    / / provenance strings into it whenever
    the corresponding gain branch fires. The accumulator is owned by the
    caller (typically core.dispatch); this function mutates it in place,
    pass-by-reference — never reassigns, never returns it.

    BLOCKER 3 (CONTEXT , 2026-04-30): provenance strings MUST contain
    'profile.py' so the production-path integration test can prove the
    upstream-gains accumulator is wired in this file (not stubbed elsewhere).
    Back-compat: callers that don't pass the kwarg behave exactly as before.
    """
    gains: dict[str, float] = {}

    # Monotropism depth per domain tag.
    md = profile_state.get("monotropism_depth", {})
    if isinstance(md, dict) and md:
        for tag in (record.tags or []):
            if tag.startswith("domain:"):
                dom = tag.split(":", 1)[1]
                if dom in md:
                    depth = md[dom]
                    try:
                        gains["monotropism_depth"] = 1.0 + float(depth)
                    except (TypeError, ValueError):
                        pass
                    if knobs_applied is not None:
                        knobs_applied["AUTIST-01"] = (
                            "profile.py:profile_modulation_for_record:monotropism_depth"
                        )
                    break

    # Interest boost amplifies any record. (verified line range: 613-616)
    ib = profile_state.get("interest_boost", 0.0)
    try:
        if float(ib) > 0:
            gains["interest_boost"] = 1.0 + float(ib)
            if knobs_applied is not None:
                knobs_applied["AUTIST-09"] = (
                    "profile.py:profile_modulation_for_record:interest_boost"
                )
    except (TypeError, ValueError):
        pass

    # Dunn quadrant posture. (verified line range: 621-625)
    dq = profile_state.get("dunn_quadrant")
    if dq == "seeking":
        gains["dunn_quadrant"] = 1.2
        if knobs_applied is not None:
            knobs_applied["AUTIST-03"] = (
                "profile.py:profile_modulation_for_record:dunn_quadrant=seeking"
            )
    elif dq == "avoiding":
        gains["dunn_quadrant"] = 0.8
        if knobs_applied is not None:
            knobs_applied["AUTIST-03"] = (
                "profile.py:profile_modulation_for_record:dunn_quadrant=avoiding"
            )

    return gains
