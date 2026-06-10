from __future__ import annotations

import os
from typing import Callable, TypeVar

import pytest

T = TypeVar("T")


def skip_if_loaded(threshold_per_core: float = 1.5) -> None:
    try:
        load1 = os.getloadavg()[0] / (os.cpu_count() or 1)
    except (OSError, AttributeError):
        return
    if load1 > threshold_per_core:
        pytest.skip(
            f"machine load {load1:.2f}/core > {threshold_per_core} — perf bench skipped"
        )


def best_of_n(fn: Callable[[], T], n: int = 3) -> T:
    if n < 1:
        raise ValueError(f"best_of_n needs n >= 1, got {n}")
    return min(fn() for _ in range(n))
