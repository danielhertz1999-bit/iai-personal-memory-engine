from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

from iai_mcp.events import write_event


SHIELD_SIGNAL_WORDS_MAX_CONFIDENCE: float = 0.9
SHIELD_FLAG_CONFIDENCE: float = 0.6

SHIELD_LANGUAGES_SUPPORTED: frozenset[str] = frozenset({
    "en", "ru", "ja", "ar", "de", "fr", "es", "zh",
})

DRIFT_M4_ANOMALY_SIGMA: float = 3.0


SIGNAL_WORDS_CRITICAL_EN: list[str] = [
    "forget", "override", "ignore previous", "you are now",
    "from now on", "system:", "admin:", "instruction:",
    "disregard", "new instructions", "ignore previous instructions",
]

SIGNAL_WORDS_WARNING_EN: list[str] = [
    "different", "instead", "actually", "update",
]

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


class ShieldTier(str, Enum):

    HARD_BLOCK = "hard_block"
    FLAG_FOR_REVIEW = "flag"
    LOG_ONLY = "log"


@dataclass
class ShieldVerdict:

    tier: ShieldTier
    detected: bool
    matched_patterns: list[str] = field(default_factory=list)
    severity: str = "info"
    action: str = "log_allow"
    reason: str = ""
    language: str | None = None
    confidence: float = 0.0


def _signal_lists_for_language(
    lang: str | None,
) -> tuple[list[str], list[str]]:
    critical = list(SIGNAL_WORDS_CRITICAL_EN)
    warning = list(SIGNAL_WORDS_WARNING_EN)
    if lang and lang in SIGNAL_WORDS_CRITICAL_BY_LANG:
        critical.extend(SIGNAL_WORDS_CRITICAL_BY_LANG[lang])
    return critical, warning


def _match_patterns(text: str, patterns: list[str]) -> list[str]:
    t = (text or "").lower()
    out: list[str] = []
    for p in patterns:
        if p.lower() in t:
            out.append(p)
    return out


def evaluate_injection_risk(
    text: str,
    tier: ShieldTier,
    target_language: str | None = None,
) -> ShieldVerdict:
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
    store: Any,
    record: Any,
    tier: ShieldTier,
    session_id: str = "-",
) -> ShieldVerdict:
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
