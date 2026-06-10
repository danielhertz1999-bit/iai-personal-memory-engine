from __future__ import annotations

import math


THETA_SKIP = 0.2

TRIVIAL_SHORT_CUES: frozenset[str] = frozenset({
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
    "yes", "no", "sure", ".", "!", "?",
})


def expected_free_energy_reduction(cue: str) -> float:
    if not cue:
        return 0.0
    stripped = cue.strip()
    if len(stripped) < 3:
        return 0.0

    normalised = stripped.lower().rstrip(".!?").strip()
    if normalised in TRIVIAL_SHORT_CUES:
        return 0.1

    tokens = [t for t in stripped.split() if t]
    unique = len({t.lower() for t in tokens})
    if unique <= 1:
        return 0.25
    value = math.log2(1 + unique) * 0.5
    return min(2.0, float(value))


def should_skip_retrieval(cue: str) -> tuple[bool, str]:
    if not cue or len(cue.strip()) < 3:
        return True, "very short cue (<3 chars); no discriminable signal"

    value = expected_free_energy_reduction(cue)
    if value < THETA_SKIP:
        return True, (
            f"trivial cue (EFE {value:.3f} bits < theta {THETA_SKIP})"
        )
    return False, ""
