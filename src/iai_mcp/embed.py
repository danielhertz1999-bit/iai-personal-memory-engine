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

VALID_QUANTIZE_MODES: set[str] = {"int8"}

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

    values: list[int]
    scale: float
    zero_point: int
    dim: int


def _quantize_int8(vec: list[float]) -> QuantizedVector:
    arr = np.asarray(vec, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax == vmin:
        return QuantizedVector(
            values=[0] * len(vec), scale=1.0, zero_point=0, dim=len(vec)
        )
    scale = (vmax - vmin) / 255.0
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

    DEFAULT_MODEL_KEY: str = DEFAULT_MODEL_KEY
    DEFAULT_DIM: int = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["dim"]
    DEFAULT_MODEL: str = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["hf"]
    DIM: int = DEFAULT_DIM

    def __init__(
        self,
        model_key: str | None = None,
        *,
        model_name: str | None = None,
    ) -> None:
        if model_key is None and model_name is not None:
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
        self.DIM: int = int(spec["dim"])

        self._model = _rust.Embedder()
        self._backend: str = "rust"

        self._quantize_mode: str | None = _resolve_quantize_mode()

    def _encode_one(self, text: str) -> list[float]:
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
        return self._encode_one(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(t) for t in texts]

    def embed_quantized(self, text: str) -> QuantizedVector:
        fp32 = self.embed(text)
        return _quantize_int8(fp32)


def embedder_for_store(store) -> "Embedder":
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
