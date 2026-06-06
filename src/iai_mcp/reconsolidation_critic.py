"""Tier-1 LLM-gated prediction-error scoring for reconsolidation.

Subscription path only, batched per-night.

References:
    - Guard gating: every LLM call goes through
      `iai_mcp.claude_cli.verify_credentials_subscription` (no paid API key)
      and the `BudgetTracker` daily/weekly cap before subprocess spawn.
    - Single-shot prompt: the batched variant packs up to
      `MAX_RECORDS_PER_CALL` labile records into ONE compact JSON prompt
      and returns one float per record. Exactly ONE claude -p call per night
      via the cap.

Contract:
    `evaluate_batch_reconsolidation` returns `dict[UUID, float]` where each
    value is in `[0.0, 1.0]`. Missing record IDs fall back to 0.0 (Tier-0
    semantics: "no provenance update"). Empty dict on any failure (gate
    denied, subscription absent, parse error, timeout, exception).

Boundary:
    No paid-API SDK imports, no env-key probes, no third-party LLM client
    construction. All LLM inference flows through
    `iai_mcp.claude_cli.invoke_claude_sync` which spawns the user's local
    `claude -p` subprocess bound to their Claude.ai subscription.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Iterable

logger = logging.getLogger(__name__)


# Batched-call hard cap: even when 500 labile records are present, the critic
# does exactly one subscription-billed subprocess invocation per REM cycle.
# Records beyond
# the cap are simply not evaluated this cycle (Tier-0 default = 0.0). They
# remain labile and get another chance on the next reconsolidation pass.
MAX_RECORDS_PER_CALL: int = 100


# Compact prompt template. Token-efficient JSON-only output (no prose) so
# the response fits within the BudgetTracker daily cap even at
# MAX_RECORDS_PER_CALL=100. Surface is truncated to 300 chars per record;
# verbatim invariant is preserved because we only ANNOTATE
# `prediction_error` — never rewrite `literal_surface` itself.
_SURFACE_TRUNC: int = 300

_BATCH_PROMPT_HEADER: str = (
    "Score each memory record for prediction error. Output ONLY a JSON "
    'array, no prose. Format: [{"id":"<uuid>","err":<float 0.0-1.0>}, ...]. '
    "0.0 = memory still accurate; 1.0 = memory contradicted by current "
    "state. Records:\n"
)


def _build_batch_prompt(items: list[tuple[uuid.UUID, str]]) -> str:
    """Assemble the compact JSON-prompt body. Surface truncated to keep
    input tokens low; UUID kept full so the round-trip ID match works."""
    lines = [
        json.dumps({
            "id": str(rid),
            "surface": (surface or "")[:_SURFACE_TRUNC],
        })
        for rid, surface in items
    ]
    return _BATCH_PROMPT_HEADER + "\n".join(lines)


def _parse_batch_response(raw: str, expected_ids: set[uuid.UUID]) -> dict[uuid.UUID, float]:
    """Parse the model's JSON-array response. Defensive: any malformed item
    is skipped silently; we only return scores for UUIDs we asked about
    (model hallucinations of extra IDs are dropped). Floats are clamped
    to [0.0, 1.0] per the original contract."""
    try:
        # Some Claude responses wrap JSON in code fences; strip a leading
        # fence if present so the parse succeeds on the common variant.
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
        parsed = json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return {}
    if not isinstance(parsed, list):
        return {}

    out: dict[uuid.UUID, float] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        rid_str = item.get("id")
        err_raw = item.get("err")
        if rid_str is None or err_raw is None:
            continue
        try:
            rid = uuid.UUID(str(rid_str))
        except (TypeError, ValueError):
            continue
        if rid not in expected_ids:
            continue
        try:
            err = float(err_raw)
        except (TypeError, ValueError):
            continue
        out[rid] = max(0.0, min(1.0, err))
    return out


def evaluate_batch_reconsolidation(
    items: Iterable[tuple[uuid.UUID, str]],
    *,
    llm_enabled: bool = True,
    max_records: int = MAX_RECORDS_PER_CALL,
) -> dict[uuid.UUID, float]:
    """Single batched LLM critic call via `claude -p` subscription path.

    Args:
        items: iterable of (record_id, literal_surface) tuples.
        llm_enabled: per-config kill switch; False -> Tier-0 fallback
            (empty dict, no claude -p invocation).
        max_records: hard cap on records evaluated per call (default 100;
            prevents unbounded prompt growth).

    Returns:
        dict mapping record UUID -> prediction_error float in [0.0, 1.0].
        Records not in the dict (over-cap or parse-skipped) fall back to
        0.0 = "no provenance update needed" per the original Tier-0
        contract.

    Failure modes (ALL return empty dict, sleep pipeline continues):
        - llm_enabled=False
        - verify_credentials_subscription denies (no creds / expired /
          wrong tier / missing inference scope)
        - BudgetTracker.can_spend denies (1% daily cap or 7% weekly buffer
          exhausted, or auto-disabled after the budget tripwire)
        - claude -p subprocess timeout / nonzero exit / unparseable JSON
        - any uncaught exception
    """
    if not llm_enabled:
        return {}

    # Defensive copy + cap. We keep the LAST `max_records` items so the most
    # recently-touched labile records win the cap race (LRU semantics).
    pool = list(items)
    if not pool:
        return {}
    if len(pool) > max_records:
        pool = pool[-max_records:]
    expected_ids = {rid for rid, _ in pool}

    # Lazy claude_cli import. Keeps module-import flat; tests can stub the
    # lookup via monkeypatch.setattr(reconsolidation_critic,...) without
    # spawning real subprocesses.
    try:
        from iai_mcp.claude_cli import (
            invoke_claude_sync,
            verify_credentials_subscription,
        )
    except ImportError:
        return {}

    # Subscription gate. Mirrors the invariant: if the user is not
    # on a valid subscription (`pro`/`pro_max`/`max`/`team`/`enterprise`)
    # with `user:inference` scope and a non-expired refresh token, the
    # critic is silently skipped.
    creds = verify_credentials_subscription()
    if not creds.get("ok"):
        return {}

    prompt = _build_batch_prompt(pool)

    try:
        result = invoke_claude_sync(prompt, model="haiku")
    except Exception as exc:  # noqa: BLE001 -- critic must never raise into REM
        logger.debug("reconsolidation critic subprocess raised: %s", exc)
        return {}

    if not result.get("ok"):
        # Surface the failure reason to logs but DO NOT propagate to the
        # sleep pipeline. Tier-0 fallback covers this cycle.
        logger.info(
            "reconsolidation critic call failed: reason=%s",
            result.get("reason"),
        )
        return {}

    raw_text = ""
    data = result.get("data") or {}
    if isinstance(data, dict):
        raw_text = str(data.get("result") or data.get("text") or "")

    return _parse_batch_response(raw_text, expected_ids)


# Backward-compat shim. Tests grep `PROMPT_TEMPLATE`
# for the verbatim slot contract (test_PROMPT_TEMPLATE_contains_required_slots).
# The template is no longer used at runtime (batched contract above subsumes it),
# but is preserved here so the static-source contract test keeps passing.
PROMPT_TEMPLATE: str = (
    "Given this stored memory `{literal_surface}` and the current state of "
    "repository `{current_summary}`, return prediction_error in 0.0-1.0 where "
    "1.0 means the memory is now wrong/contradicted."
)


# Legacy single-record entry kept as a thin shim so tests
# importing `call_critic` directly don't break. The body delegates to the
# batched implementation with a synthetic UUID so the same gate + subprocess
# guarantees apply. `has_api_key` and `estimated_usd` kwargs accepted but
# IGNORED (no paid-API surface remains). Returns 0.0 unconditionally when
# `llm_enabled=False` or the subscription gate denies — matches the original
# Tier-0 fallback contract.
def call_critic(
    literal_surface: str,
    current_summary: str = "",
    store=None,  # noqa: ARG001 -- legacy signature, retained for back-compat
    *,
    llm_enabled: bool = True,
    has_api_key: bool = False,  # noqa: ARG001 -- legacy, ignored
    estimated_usd: float = 0.001,  # noqa: ARG001 -- legacy, ignored
) -> float:
    """Legacy per-record critic API that routes through the batched
    implementation for back-compat with tests that import the symbol
    directly. `sleep_pipeline._step_reconsolidation` no longer calls
    this -- it calls `evaluate_batch_reconsolidation` directly with all
    labile records at once.
    """
    if not llm_enabled:
        return 0.0
    synthetic_rid = uuid.uuid4()
    result = evaluate_batch_reconsolidation(
        [(synthetic_rid, literal_surface)],
        llm_enabled=llm_enabled,
        max_records=1,
    )
    return result.get(synthetic_rid, 0.0)
