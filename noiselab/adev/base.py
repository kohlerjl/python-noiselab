from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

TDoubleArray = NDArray[np.double]
TIntArray = NDArray[np.int64]


def sample_logspace(num: int, *,
                    base: float,
                    subspacing: int | Iterable[int] = 1,
                    dtype: np.dtype = np.int64) -> TIntArray:
    if isinstance(subspacing, int):
        subspacing = np.logspace(0, 1, num=subspacing + 1, base=base)

    if num <= 0:
        return np.array([], dtype=dtype)

    # Include sub-spaced intervals in final decade, even if not complete
    num_base = math.ceil(math.log(num / 2) / math.log(base))
    base_spacing = [base ** n for n in range(num_base)]
    spacing = np.unique(np.array([round(m * base) for base in base_spacing for m in subspacing], dtype=dtype))
    return spacing[spacing < num / 2]  # Drop any intervals greater or equal to half the sample size


def get_strides(dt: float, num: int, taus: str | Iterable[float], dtype: np.dtype = np.int64) -> TIntArray:
    if not isinstance(taus, str):
        return np.array(np.round(np.array(taus) / dt), dtype=dtype)
    if taus.startswith('octave'):
        subspace_str = taus[len('octave'):]
        subspacing: int = int(subspace_str) if subspace_str else 1
        return sample_logspace(num, base=2, subspacing=subspacing, dtype=dtype)
    if taus.startswith('decade'):
        subspace_str = taus[len('decade'):]
        # Match default decade spacing in allantools and Stable32
        subspacing: int | list[int] = int(subspace_str) if subspace_str else [1, 2, 4]
        return sample_logspace(num, base=10, subspacing=subspacing, dtype=dtype)

    raise ValueError(f"Invalid taus: {taus!r}. Must be one of: 'octave[n]' or 'decade[n]'")


def integrate_samples(x: TDoubleArray) -> TDoubleArray:
    # Subtract mean to avoid loss of precision from accumulator overflow in integrated values
    x = x - np.mean(x)
    sum_x = np.empty(len(x) + 1, dtype=x.dtype)
    sum_x[0] = 0
    np.cumsum(x, out=sum_x[1:])
    return sum_x
