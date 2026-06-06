"""Embedding layer -- single English Rust embedder.

The sole backend is the Rust native extension (iai_mcp_native.embed.Embedder),
which runs bge-small-en-v1.5 (384d, English-only) with no Python fallback.
A missing or broken native module causes the daemon and MCP server to fail
loud at startup via the native_guard, not here.

MODEL_REGISTRY contains exactly one entry (bge-small-en-v1.5). There is no
model-selection environment variable and no PyTorch path.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from iai_mcp_native import embed as _rust


logger = logging.getLogger(__name__)


MODEL_REGISTRY: dict[str, dict] = {
    "bge-small-en-v1.5": {"hf": "BAAI/bge-small-en-v1.5", "dim": 384},
}
DEFAULT_MODEL_KEY = "bge-small-en-v1.5"

# Write-side quantization knob. Default (env unset) keeps the fp32 path
# byte-identical; int8 is exposed via the additive Embedder.embed_quantized()
# surface only. Extensible: future modes (e.g. "fp16") can be added to this set.
VALID_QUANTIZE_MODES: set[str] = {"int8"}

# Module-level counter incremented by _encode_one on every native encode
# exception. Process-wide (not per-Embedder instance).
embed_failure_total: int = 0


def _resolve_model_key(model_key: str | None = None) -> str:
    if model_key is not None:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"unknown embed model key {model_key!r}; valid: {sorted(MODEL_REGISTRY)}"
            )
        return model_key
    return DEFAULT_MODEL_KEY


def _resolve_quantize_mode() -> str | None:
    """Read ``IAI_MCP_EMBED_QUANTIZE`` env var; return mode or None.

    Empty string or unset -> ``None`` (fp32 default -- unchanged behavior).
    ``"int8"`` (case-sensitive, lower-case only) -> ``"int8"``.
    Any other non-empty value -> ``ValueError``. NO silent fallback.

    Case-sensitivity choice: lower-case only. ``"INT8"`` is rejected so the
    knob value matches the canonical mode token used in storage metadata.
    """
    raw = os.environ.get("IAI_MCP_EMBED_QUANTIZE", "")
    if not raw:
        return None
    if raw not in VALID_QUANTIZE_MODES:
        raise ValueError(
            f"IAI_MCP_EMBED_QUANTIZE={raw!r} is not a valid quantization mode; "
            f"valid: {sorted(VALID_QUANTIZE_MODES)} or unset for fp32 default"
        )
    return raw


@dataclass(frozen=True)
class QuantizedVector:
    """int8-quantized embedding with per-vector min/max calibration metadata.

    Reconstruct the fp32 approximation via:

        fp32[i] ≈ (values[i] - zero_point) * scale

    Per-vector calibration is used because BGE per-dim values cluster
    narrowly around 0 but are NOT confined to [-1, 1] despite L2
    normalization. A global codebook would waste resolution; per-vector
    min/max maps each vector's full dynamic range onto the [-128, 127]
    int8 codebook, preserving cos >= 0.99 round-trip on real probes.
    """

    values: list[int]   # length == dim; each in [-128, 127] (signed int8)
    scale: float        # per-vector scale = (vmax - vmin) / 255
    zero_point: int     # per-vector zero-point in the int8 codebook
    dim: int            # convenience; equals len(values)


def _quantize_int8(vec: list[float]) -> QuantizedVector:
    """Per-vector min/max int8 quantization of a fp32 embedding vector.

    Inverse: ``fp32[i] ≈ (values[i] - zero_point) * scale``. Empirically
    preserves cos >= 0.99 on real bge-small-en-v1.5 outputs.
    """
    arr = np.asarray(vec, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    # Degenerate case: all-equal vector -> scale=1.0, zero_point=0, values=zeros.
    # In practice BGE never produces this, but guard anyway so the helper is
    # total over the input space.
    if vmax == vmin:
        return QuantizedVector(
            values=[0] * len(vec), scale=1.0, zero_point=0, dim=len(vec)
        )
    scale = (vmax - vmin) / 255.0
    # Map fp32 vmin -> -128, fp32 vmax -> 127, define zero_point so the
    # inverse (values[i] - zero_point) * scale recovers fp32.
    zero_point = int(round(-vmin / scale)) - 128
    quantized = np.round(arr / scale).astype(np.int32) + zero_point
    quantized = np.clip(quantized, -128, 127).astype(np.int8)
    return QuantizedVector(
        values=[int(x) for x in quantized.tolist()],
        scale=float(scale),
        zero_point=int(zero_point),
        dim=len(vec),
    )


class Embedder:
    """English-Only Brain embedder backed by the Rust native extension.

    Uses bge-small-en-v1.5 (384d, English-only) exclusively. There is no
    PyTorch fallback and no backend selection option. A runtime encode failure
    increments the module-level embed_failure_total counter, logs an error
    breadcrumb, and raises -- it never silently returns an off-distribution
    vector.

    Backward compatibility:
    - ``Embedder.DIM`` is kept as a class attribute aliased to the default model
      dimension so earlier tests that reference ``Embedder.DIM`` still work; new
      code should prefer ``Embedder().DIM`` (instance attr) for correctness.
    - ``Embedder.DEFAULT_MODEL`` is the HF id of the English model (bge-small-en-v1.5).
    """

    DEFAULT_MODEL_KEY: str = DEFAULT_MODEL_KEY
    DEFAULT_DIM: int = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["dim"]
    # Legacy class-level attributes (test compatibility).
    # New code should construct Embedder() and read.DIM from the instance.
    DEFAULT_MODEL: str = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["hf"]
    DIM: int = DEFAULT_DIM

    def __init__(
        self,
        model_key: str | None = None,
        *,
        model_name: str | None = None,
    ) -> None:
        """Initialise an Embedder.

        Parameters
        ----------
        model_key:
            Logical key from MODEL_REGISTRY. The only valid value is
            "bge-small-en-v1.5"; any other key raises ValueError. If None,
            resolves to the default ("bge-small-en-v1.5").
        model_name:
            Legacy parameter: full HuggingFace repo id. Prefer model_key for
            new code. If both are provided, model_key wins.

        The Rust native extension is the sole, mandatory embed runtime. No
        PyTorch path exists. A missing or broken native module is caught at
        process startup by native_guard._require_native() before this
        constructor is ever called.
        """
        if model_key is None and model_name is not None:
            # Reverse-lookup: find the key whose hf matches this name.
            match = next(
                (k for k, v in MODEL_REGISTRY.items() if v["hf"] == model_name),
                None,
            )
            if match is None:
                raise ValueError(
                    f"model_name {model_name!r} is not in MODEL_REGISTRY; "
                    f"valid hf ids: {[v['hf'] for v in MODEL_REGISTRY.values()]}"
                )
            key = match
        else:
            key = _resolve_model_key(model_key)
        self.model_key: str = key
        spec = MODEL_REGISTRY[key]
        self.model_name: str = spec["hf"]
        self.DIM: int = int(spec["dim"])  # instance attr overrides class attr

        # Unconditional Rust native construction. The native extension is the
        # sole mandatory embed runtime; there is no PyTorch path.
        self._model = _rust.Embedder()
        self._backend: str = "rust"

        # Write-side quantization knob. Read once at construction so an
        # invalid value fails loud at startup rather than later at first
        #.embed_quantized() call. Does NOT change.embed() /.embed_batch()
        # behavior -- the int8 surface is exposed exclusively via
        #.embed_quantized().
        self._quantize_mode: str | None = _resolve_quantize_mode()

    def _encode_one(self, text: str) -> list[float]:
        """Single encode call routed through the observability chokepoint.

        All encode calls from embed() and embed_batch() pass through here.
        On a native encode failure: increments the module-level
        embed_failure_total counter, logs an error breadcrumb, then re-raises
        (no fallback -- fail loud).
        """
        global embed_failure_total
        try:
            return self._model.encode(text)
        except Exception as exc:
            embed_failure_total += 1
            logger.error(
                "native embed encode failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            raise

    def embed(self, text: str) -> list[float]:
        """Encode a single string to a DIM-length list[float]. Normalised, deterministic."""
        return self._encode_one(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch-encode preserving input order. Returns N vectors for N inputs.

        Uses a per-item Rust encode loop routed through the _encode_one
        chokepoint, so a failure in the batch loop also increments
        embed_failure_total and logs a breadcrumb.
        """
        return [self._encode_one(t) for t in texts]

    def embed_quantized(self, text: str) -> QuantizedVector:
        """Encode ``text`` and return an int8-quantized vector + metadata.

        Always available regardless of env var -- the env var gates init-time
        validation (so an invalid value fails loud at startup), not method
        availability. For ambient ergonomics, callers that opt in via
        ``IAI_MCP_EMBED_QUANTIZE=int8`` should branch on ``self._quantize_mode``.
        """
        fp32 = self.embed(text)
        return _quantize_int8(fp32)


def embedder_for_store(store) -> "Embedder":
    """Store-aware Embedder factory.

    Returns the single English Rust embedder (bge-small-en-v1.5, 384d).
    When the store carries an embed_dim attribute the factory checks that it
    matches the English model dimension. Stores built against a non-384d
    dim (e.g. a legacy 1024d store) fall through to the default Embedder().

    The try/except TypeError shim preserves compatibility with test stubs that
    monkeypatch Embedder with a zero-arg callable.
    """
    target_dim = getattr(store, "embed_dim", None)
    if target_dim is None:
        return Embedder()
    preferred = {384: "bge-small-en-v1.5"}
    key = preferred.get(int(target_dim))
    try:
        if key is not None and key in MODEL_REGISTRY:
            return Embedder(model_key=key)
        for reg_key, spec in MODEL_REGISTRY.items():
            if int(spec["dim"]) == int(target_dim):
                return Embedder(model_key=reg_key)
    except TypeError:
        pass
    return Embedder()
