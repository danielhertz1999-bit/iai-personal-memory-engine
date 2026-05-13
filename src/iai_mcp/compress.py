""" LLMLingua-2 compression (Task 2, ).

Compression is allowed ONLY on retrieval views and summaries, NEVER on raw
content. Enforcement lives in `is_compressible`:

Forbidden:
- pinned records (includes L0 identity)
- invariant_anchor records (s5_trust_score >= 0.9)
- user-tagged raw: records (raw:en, raw:ru, ...)
- normal episodic records (default reject; literal_surface is constitutional
  per )

Allowed:
- records tagged cls_summary (CLS consolidation output)
- records tagged schema (LEARN-03 induction output)
- records tagged session_summary

Runtime fallback: when `llmlingua` is not installed, `compress_llmlingua2`
returns the input unchanged and emits an llm_health event. This keeps the
Tier-0 path green on minimal installs (CI, fresh user machines).

Constants:
- COMPRESSION_TARGET_L2 = 0.5 (community descriptors)
- COMPRESSION_TARGET_SUMMARY = 0.3 (session summaries)
"""
from __future__ import annotations

import threading
from typing import Any

from iai_mcp.events import write_event


# ratio targets.
COMPRESSION_TARGET_L2 = 0.5
COMPRESSION_TARGET_SUMMARY = 0.3

# threshold -- records at or above this trust score are invariant anchors.
INVARIANT_TRUST_THRESHOLD = 0.9


# ----------------------------------------------------------- scope gate


def is_compressible(record) -> tuple[bool, str]:
    """Return (allowed, reason) for a given MemoryRecord.

    Reason is a short English diagnostic consumed only in tests / debug logs.
    """
    if getattr(record, "pinned", False):
        return False, "pinned record (D-14 L0 / user-pinned)"

    trust = getattr(record, "s5_trust_score", 0.5)
    try:
        if float(trust) >= INVARIANT_TRUST_THRESHOLD:
            return False, (
                f"invariant anchor (trust={float(trust):.2f} >= "
                f"{INVARIANT_TRUST_THRESHOLD}); forbids compression"
            )
    except (TypeError, ValueError):
        pass

    tags = getattr(record, "tags", None) or []
    for tag in tags:
        if tag.startswith("raw:"):
            return False, f"raw-tagged record ({tag}); user flagged as raw"

    # Explicit allowlist.
    allow_tags = {"cls_summary", "schema", "session_summary"}
    for tag in tags:
        if tag in allow_tags:
            return True, ""

    return False, "literal_surface constitutional (D-25 default deny)"


# ----------------------------------------------------------- llmlingua loader


_LLMLINGUA_LOCK = threading.Lock()
_LLMLINGUA_CACHE: dict[str, Any] = {}


def _load_llmlingua2():
    """Lazy-load llmlingua's PromptCompressor (LLMLingua-2 model).

    Returns the compressor instance on success; None if the package is absent
    or fails to instantiate. Callers log a fallback event and passthrough.
    """
    with _LLMLINGUA_LOCK:
        if "instance" in _LLMLINGUA_CACHE:
            return _LLMLINGUA_CACHE["instance"]
        try:
            from llmlingua import PromptCompressor  # type: ignore
        except Exception:
            _LLMLINGUA_CACHE["instance"] = None
            return None
        try:
            # Device auto-detection: CUDA if available (Linux GPU), else MPS on
            # Apple Silicon (torch.backends.mps), else CPU. llmlingua's default
            # assumes CUDA which breaks on macOS ARM64.
            import torch  # type: ignore
            if torch.cuda.is_available():
                device_map = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device_map = "mps"
            else:
                device_map = "cpu"
            # microsoft/llmlingua-2-xlm-roberta-large-meetingbank (default in
            # llmlingua>=0.2). Although this compressor is multilingual-capable,
            # the IAI-MCP brain itself is English-only; the
            # multilingual support is incidental and only matters for the
            # opt-in bge-m3 path.
            compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device_map=device_map,
            )
        except Exception:
            _LLMLINGUA_CACHE["instance"] = None
            return None
        _LLMLINGUA_CACHE["instance"] = compressor
        return compressor


# ----------------------------------------------------------- core compression


def compress_llmlingua2(
    text: str,
    target_ratio: float = 0.5,
    store=None,
) -> str:
    """Compress `text` to approximately `target_ratio` of original tokens.

    On any failure (package missing, model load error, runtime exception):
    - Return `text` unchanged (passthrough).
    - If `store` is provided, emit an llm_health event of kind
      'compression_fallback' with severity='warning'.

    scope is the caller's responsibility (is_compressible must be
    consulted BEFORE reaching this function).
    """
    if not text:
        return text

    compressor = _load_llmlingua2()
    if compressor is None:
        if store is not None:
            try:
                write_event(
                    store,
                    kind="llm_health",
                    data={
                        "component": "compress_llmlingua2",
                        "tier": "fallback",
                        "reason": "llmlingua package unavailable or model load failed",
                    },
                    severity="warning",
                )
            except Exception:
                pass
        return text

    try:
        result = compressor.compress_prompt(text, rate=float(target_ratio))
        if isinstance(result, dict):
            return str(result.get("compressed_prompt", text))
        return str(result)
    except Exception as exc:  # pragma: no cover -- runtime failure passthrough
        if store is not None:
            try:
                write_event(
                    store,
                    kind="llm_health",
                    data={
                        "component": "compress_llmlingua2",
                        "tier": "fallback",
                        "error": str(exc),
                    },
                    severity="warning",
                )
            except Exception:
                pass
        return text


def compress_l2_descriptor(descriptor: str, store=None) -> str:
    """Compress an L2 community descriptor ( target ratio 0.5)."""
    return compress_llmlingua2(
        descriptor, target_ratio=COMPRESSION_TARGET_L2, store=store,
    )


def compress_summary(summary: str, store=None) -> str:
    """Compress a session summary ( target ratio 0.3)."""
    return compress_llmlingua2(
        summary, target_ratio=COMPRESSION_TARGET_SUMMARY, store=store,
    )
