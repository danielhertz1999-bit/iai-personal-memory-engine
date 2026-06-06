"""Typed exception hierarchy for IAI-MCP.

: Replace bare except-Exception with narrowed, loggable exceptions.
Organized by subsystem. All inherit from IAIMCPError for catch-all at boundaries.
"""

from __future__ import annotations


class IAIMCPError(Exception):
    """Base for all IAI-MCP exceptions. Catch at daemon boundary only."""


class NativeError(IAIMCPError):
    """A mandatory native-extension operation (encode or graph compute) failed.

    Raised instead of swallowing so callers can distinguish a native-runtime
    failure from a soft infrastructure error (graph-build cache miss,
    community-detection OOM, etc.) that is safe to degrade gracefully.
    Carry the original exception as ``__cause__`` via ``raise NativeError(...)
    from original``.
    """


# --- Store subsystem ---


class StoreError(IAIMCPError):
    """Base for storage-layer failures."""


class StoreInsertError(StoreError):
    """Failed to insert a record into the store."""


class StoreQueryError(StoreError):
    """Failed to query records from the store."""


class StoreSchemaError(StoreError):
    """Schema mismatch or migration required."""


class StoreConcurrencyError(StoreError):
    """Concurrent access conflict (lock contention, WAL)."""


class StoreCorruptionError(StoreError):
    """Data integrity violation detected."""


# --- Sleep pipeline subsystem ---


class SleepPipelineError(IAIMCPError):
    """Base for sleep/REM pipeline failures."""


class SleepStepError(SleepPipelineError):
    """A named pipeline step failed."""

    def __init__(self, step_name: str, cause: Exception | None = None):
        self.step_name = step_name
        self.cause = cause
        super().__init__(f"Sleep step {step_name!r} failed: {cause}")


class SleepCheckpointError(SleepPipelineError):
    """Checkpoint read/write failure."""


class SleepQuarantineError(SleepPipelineError):
    """Record quarantined after repeated failures."""


# --- Retrieval subsystem ---


class RetrievalError(IAIMCPError):
    """Base for retrieval/recall failures."""


class EmbeddingError(RetrievalError):
    """Embedding model failed to produce vector."""


class CommunityGateError(RetrievalError):
    """Leiden community gate produced no candidates."""


class BudgetExceededError(RetrievalError):
    """Token budget exceeded during packing."""


# --- Daemon lifecycle ---


class LifecycleError(IAIMCPError):
    """Base for daemon lifecycle failures."""


class LifecycleTransitionError(LifecycleError):
    """Invalid state transition attempted."""


class DaemonTickError(LifecycleError):
    """Non-fatal error within a daemon tick (logged, not propagated)."""


# --- Capture subsystem ---


class CaptureError(IAIMCPError):
    """Base for ambient capture failures."""


class CaptureDeduplicationError(CaptureError):
    """Dedup check failed (not a duplicate — the check itself errored)."""


class CaptureDrainError(CaptureError):
    """Deferred capture drain failure."""


# --- Crypto subsystem ---


class CryptoError(IAIMCPError):
    """Base for encryption/decryption failures."""


class CryptoDecryptError(CryptoError):
    """Failed to decrypt a record or bank file."""


class CryptoKeyMissing(CryptoError):
    """Expected key material not found."""


# --- VSM / algedonic bypass ---


class AlgedonicSignal(IAIMCPError):
    """S1→S5 escalation signal when a subsystem detects critical failure.

    Not a crash — this is a cybernetic control signal that triggers
    S5 to evaluate whether the system's identity/viability is at risk.
    """

    def __init__(self, subsystem: str, severity: str, detail: str):
        self.subsystem = subsystem
        self.severity = severity
        self.detail = detail
        super().__init__(f"[{subsystem}] algedonic({severity}): {detail}")
