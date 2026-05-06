"""OPS-07 prompt-injection shield (D-30, D-31) -- Plan 02-05.

Three-tier deployment per D-31:
    HARD_BLOCK     -> L0 identity + S5 invariant writes (reject on detection)
    FLAG_FOR_REVIEW -> profile updates (flag + warn, write proceeds)
    LOG_ONLY        -> content records (log only, allow)

D-30 threat model (three severities):
  - Direct override (e.g. "forget X, now Y") -> HARD BLOCK via signal words
  - Gradual drift (subtle lies over weeks)   -> DETECT via trajectory M4 anomaly
                                                 (see s5.detect_drift_anomaly)
  - Data poisoning (intentional false write) -> MITIGATE via ART vigilance
                                                 + user-approval UX

Global-product mandate: signal words cover 7+ languages
(en + ru + ja + ar + de + fr + es + zh) at minimum. The module exports
`SHIELD_LANGUAGES_SUPPORTED` as the authoritative set; downstream acceptance
tests grep against it.

The shield is a PURE LOCAL filter: no LLM call, no network. Detection uses
case-insensitive substring matching against curated signal-word lists. The
tier policy is additive: warning signals escalate to critical in the
HARD_BLOCK tier (L0 is sacred).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

from iai_mcp.events import write_event


# ------------------------------------------------------------ constitutional constants

# Confidence thresholds for the shield verdict. Confidence is a simple signal:
# matched_count / TOTAL_BASELINE -- used for downstream analytics, not the
# tier-policy gate. The tier enum + match count drives the action.
SHIELD_SIGNAL_WORDS_MAX_CONFIDENCE: float = 0.9  # upper bound reported on any match
SHIELD_FLAG_CONFIDENCE: float = 0.6              # reported when matches are warning-only

# global-product mandate: 7+ languages supported.
SHIELD_LANGUAGES_SUPPORTED: frozenset[str] = frozenset({
    "en", "ru", "ja", "ar", "de", "fr", "es", "zh",
})

# gradual-drift detection threshold -- used by s5.detect_drift_anomaly
# but declared here so the single authoritative constant sits alongside the
# other shield thresholds (downstream greps one file).
DRIFT_M4_ANOMALY_SIGMA: float = 3.0


# ------------------------------------------------------------ signal-word catalogues

# English critical signal words: classic prompt-injection imperatives.
SIGNAL_WORDS_CRITICAL_EN: list[str] = [
    "forget", "override", "ignore previous", "you are now",
    "from now on", "system:", "admin:", "instruction:",
    "disregard", "new instructions", "ignore previous instructions",
]

# English warning signals: softer but still suspicious rephrasings.
SIGNAL_WORDS_WARNING_EN: list[str] = [
    "different", "instead", "actually", "update",
]

# Per-language critical signal words (D-02a mandate).
# Keys are ISO-639-1 codes; values are minimal strictly-imperative tokens.
# Conservative by design: false positives on legitimate non-English chatter are
# worse than false negatives at this tier (users have multiple layers of
# defence; the shield is one slice of defence-in-depth).
SIGNAL_WORDS_CRITICAL_BY_LANG: dict[str, list[str]] = {
    "ru": [
        "забудь", "забыть", "игнорируй",
        "отмени", "сбрось", "новые инструкции",
        "теперь ты", "с этого момента",
    ],
    "ja": [
        "忘れて", "無視", "リセット",
        "新しい指示", "これから", "今から",
    ],
    "ar": [
        "انسى", "تجاهل",
        "إعادة تعيين", "تعليمات جديدة", "أنت الآن",
    ],
    "de": [
        "vergiss", "ignoriere", "überschreibe",
        "neue anweisungen", "ab jetzt",
    ],
    "fr": [
        "oublie", "ignore",
        "remplace", "nouvelles instructions",
    ],
    "es": [
        "olvida", "ignora",
        "sobrescribe", "nuevas instrucciones",
    ],
    "zh": [
        "忘记", "忽略", "重置",
        "新指令", "从现在开始",
    ],
}


# ------------------------------------------------------------ enums + types


class ShieldTier(str, Enum):
    """D-31 three-tier deployment."""

    HARD_BLOCK = "hard_block"          # L0 identity + S5 invariants
    FLAG_FOR_REVIEW = "flag"           # profile updates
    LOG_ONLY = "log"                   # content records


@dataclass
class ShieldVerdict:
    """Result of evaluating injection risk for a single text blob."""

    tier: ShieldTier
    detected: bool
    matched_patterns: list[str] = field(default_factory=list)
    severity: str = "info"             # "info" | "warning" | "critical"
    action: str = "log_allow"          # "reject" | "flag" | "log_allow"
    reason: str = ""
    language: str | None = None
    confidence: float = 0.0


# ------------------------------------------------------------ private helpers


def _signal_lists_for_language(
    lang: str | None,
) -> tuple[list[str], list[str]]:
    """Return (critical, warning) lists for the given language.

    English signals are ALWAYS included (prompt-injection attempts are often
    copy-pasted English regardless of the user's native language). When a
    `lang` is given AND supported, its per-language critical list is appended.
    """
    critical = list(SIGNAL_WORDS_CRITICAL_EN)
    warning = list(SIGNAL_WORDS_WARNING_EN)
    if lang and lang in SIGNAL_WORDS_CRITICAL_BY_LANG:
        critical.extend(SIGNAL_WORDS_CRITICAL_BY_LANG[lang])
    return critical, warning


def _match_patterns(text: str, patterns: list[str]) -> list[str]:
    """Return the subset of patterns present in the (lowercased) text.

    For Latin-script patterns we lowercase both sides. For non-ASCII scripts
    (Cyrillic, Hiragana, CJK, Arabic) lowercasing is either identity-preserving
    (CJK has no case) or handled uniformly by str.lower() which is safe for
    our lists.
    """
    t = (text or "").lower()
    out: list[str] = []
    for p in patterns:
        if p.lower() in t:
            out.append(p)
    return out


# ------------------------------------------------------------ public API


def evaluate_injection_risk(
    text: str,
    tier: ShieldTier,
    target_language: str | None = None,
) -> ShieldVerdict:
    """Core shield detection (pure function, no side effects).

    Tier escalation policy:
      HARD_BLOCK       -- any critical OR warning match -> reject (severity critical)
      FLAG_FOR_REVIEW  -- any match -> flag (severity warning)
      LOG_ONLY         -- any match -> log_allow (severity info)
      no match         -- detected=False, action=log_allow
    """
    critical_list, warning_list = _signal_lists_for_language(target_language)
    matched_critical = _match_patterns(text, critical_list)
    matched_warning = _match_patterns(text, warning_list)
    all_matched = matched_critical + matched_warning

    if not all_matched:
        return ShieldVerdict(
            tier=tier,
            detected=False,
            matched_patterns=[],
            severity="info",
            action="log_allow",
            reason="no signal patterns detected",
            language=target_language,
            confidence=0.0,
        )

    # Confidence: 0.9 when any critical match, 0.6 when warning-only.
    confidence = (
        SHIELD_SIGNAL_WORDS_MAX_CONFIDENCE
        if matched_critical
        else SHIELD_FLAG_CONFIDENCE
    )

    if tier == ShieldTier.HARD_BLOCK:
        return ShieldVerdict(
            tier=tier,
            detected=True,
            matched_patterns=all_matched,
            severity="critical",
            action="reject",
            reason=(
                f"injection signals detected in HARD_BLOCK tier: {all_matched}"
            ),
            language=target_language,
            confidence=confidence,
        )
    if tier == ShieldTier.FLAG_FOR_REVIEW:
        return ShieldVerdict(
            tier=tier,
            detected=True,
            matched_patterns=all_matched,
            severity="warning",
            action="flag",
            reason=f"injection signals detected in FLAG tier: {all_matched}",
            language=target_language,
            confidence=confidence,
        )
    # LOG_ONLY
    return ShieldVerdict(
        tier=tier,
        detected=True,
        matched_patterns=all_matched,
        severity="info",
        action="log_allow",
        reason=f"injection signals detected in LOG tier: {all_matched}",
        language=target_language,
        confidence=confidence,
    )


def apply_shield(
    store: Any,  # MemoryStore
    record: Any,  # MemoryRecord (avoids import cycle with types)
    tier: ShieldTier,
    session_id: str = "-",
) -> ShieldVerdict:
    """Evaluate + emit event (side-effectful wrapper).

    Event kind is determined by the tier policy:
      - reject    -> kind="shield_rejection" (severity critical)
      - flag      -> kind="shield_flag"      (severity warning)
      - log_allow -> kind="shield_log"       (severity info, ONLY on detection)

    No event is emitted when the verdict is "not detected" -- no signal, no
    noise in the events table.
    """
    verdict = evaluate_injection_risk(
        record.literal_surface or "",
        tier,
        target_language=record.language or None,
    )
    if verdict.detected:
        kind_map = {
            "reject": "shield_rejection",
            "flag": "shield_flag",
            "log_allow": "shield_log",
        }
        event_kind = kind_map.get(verdict.action, "shield_log")
        # Clip matched patterns payload so the events table does not grow
        # unbounded on adversarial input.
        matched_clipped = [str(p)[:80] for p in verdict.matched_patterns[:10]]
        record_id = record.id
        source_ids: list[UUID] = []
        if isinstance(record_id, UUID):
            source_ids = [record_id]
        write_event(
            store,
            kind=event_kind,
            data={
                "record_id": str(record_id) if record_id is not None else None,
                "tier": verdict.tier.value,
                "matched": matched_clipped,
                "language": record.language,
                "action": verdict.action,
                "confidence": verdict.confidence,
            },
            severity=verdict.severity,
            session_id=session_id,
            source_ids=source_ids,
        )
    return verdict


__all__ = [
    "DRIFT_M4_ANOMALY_SIGMA",
    "SHIELD_FLAG_CONFIDENCE",
    "SHIELD_LANGUAGES_SUPPORTED",
    "SHIELD_SIGNAL_WORDS_MAX_CONFIDENCE",
    "SIGNAL_WORDS_CRITICAL_BY_LANG",
    "SIGNAL_WORDS_CRITICAL_EN",
    "SIGNAL_WORDS_WARNING_EN",
    "ShieldTier",
    "ShieldVerdict",
    "apply_shield",
    "evaluate_injection_risk",
]
