
from __future__ import annotations


class IAIMCPError(Exception):
    pass


class NativeError(IAIMCPError):
    pass


class StoreError(IAIMCPError):
    pass


class StoreInsertError(StoreError):
    pass


class StoreQueryError(StoreError):
    pass


class StoreSchemaError(StoreError):
    pass


class StoreConcurrencyError(StoreError):
    pass


class StoreCorruptionError(StoreError):
    pass


class SleepPipelineError(IAIMCPError):
    pass


class SleepStepError(SleepPipelineError):

    def __init__(self, step_name: str, cause: Exception | None = None):
        self.step_name = step_name
        self.cause = cause
        super().__init__(f"Sleep step {step_name!r} failed: {cause}")


class SleepCheckpointError(SleepPipelineError):
    pass


class SleepQuarantineError(SleepPipelineError):
    pass


class RetrievalError(IAIMCPError):
    pass


class EmbeddingError(RetrievalError):
    pass


class CommunityGateError(RetrievalError):
    pass


class BudgetExceededError(RetrievalError):
    pass


class LifecycleError(IAIMCPError):
    pass


class LifecycleTransitionError(LifecycleError):
    pass


class DaemonTickError(LifecycleError):
    pass


class CaptureError(IAIMCPError):
    pass


class CaptureDeduplicationError(CaptureError):
    pass


class CaptureDrainError(CaptureError):
    pass


class CryptoError(IAIMCPError):
    pass


class CryptoDecryptError(CryptoError):
    pass


class CryptoKeyMissing(CryptoError):
    pass


class AlgedonicSignal(IAIMCPError):

    def __init__(self, subsystem: str, severity: str, detail: str):
        self.subsystem = subsystem
        self.severity = severity
        self.detail = detail
        super().__init__(f"[{subsystem}] algedonic({severity}): {detail}")
