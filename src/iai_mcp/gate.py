"""TOK-06 active-inference retrieval gate (Plan 02-04 Task 2, D-26).

Skip full pipeline_recall when the expected free-energy reduction for the
current cue is below THETA_SKIP bits. Trivial cues (greetings, "thanks",
single characters) short-circuit to an L0-only response, saving 200-500
tokens per trivial turn.

The heuristic uses a simple token-count proxy for EFE:
- Empty / sub-3-char cues: 0.0 bits (no signal).
- Greetings ("hi", "hello", "thanks", "ok") in the fixed trivial set: 0.1 bits.
- Single-token cues not in the trivial set: 0.25 bits (above threshold -- 
  one rare/novel token can still justify a retrieval).
- General cues: min(2.0, log2(1 + unique_token_count) * 0.5).

Phase 2 note: this is an approximation. can replace with a real
embedding-distance-to-prior computation once the write policy is active.
"""
from __future__ import annotations

import math


# threshold (bits).
THETA_SKIP = 0.2

# Fixed-EFE trivial cues. Matched case-insensitively against stripped punctuation.
TRIVIAL_SHORT_CUES: frozenset[str] = frozenset({
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
    "yes", "no", "sure", ".", "!", "?",
})


# ---------------------------------------------------------- EFE computation


def expected_free_energy_reduction(cue: str) -> float:
    """Estimate the expected free-energy reduction for `cue` (bits).

    - Empty or <3 chars  -> 0.0 (below threshold; skip)
    - Fixed trivial set  -> 0.1 (below threshold; skip)
    - Single non-trivial -> 0.25 (above threshold; proceed)
    - General formula    -> min(2.0, log2(1 + unique_token_count) * 0.5)
    """
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
        # Single token not in trivial set -- rare/novel token MAY be a proper
        # noun, code identifier, or keyword. Stay above threshold.
        return 0.25
    value = math.log2(1 + unique) * 0.5
    return min(2.0, float(value))


# ---------------------------------------------------------- skip decision


def should_skip_retrieval(cue: str) -> tuple[bool, str]:
    """Return (skip, reason) per D-26.

    reason is a short English diagnostic suitable for a RecallResponse hint.
    """
    if not cue or len(cue.strip()) < 3:
        return True, "very short cue (<3 chars); no discriminable signal"

    value = expected_free_energy_reduction(cue)
    if value < THETA_SKIP:
        return True, (
            f"trivial cue (EFE {value:.3f} bits < theta {THETA_SKIP})"
        )
    return False, ""
