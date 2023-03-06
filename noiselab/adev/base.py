from __future__ import annotations

import logging
import math
import typing

import numpy as np

logger = logging.getLogger(__name__)


def sample_logspace(num, base, subspacing: int | typing.Sequence[int] = 1, dtype=np.int64):
    if isinstance(subspacing, int):
        subspacing = np.logspace(0, 1, num=subspacing + 1, base=base)

    if num <= 0:
        return np.array([], dtype=dtype)

    # Include sub-spaced intervals in final decade, even if not complete
    num_base = math.ceil(math.log(num / 2) / math.log(base))
    base_spacing = [base ** n for n in range(num_base)]
    spacing = np.unique(np.array([round(m * base) for base in base_spacing for m in subspacing], dtype=dtype))
    return spacing[spacing < num / 2]  # Drop any intervals greater or equal to half the sample size


def get_strides(dt: float, num: int, taus: str | typing.Sequence[float], dtype=np.int64):
    if not isinstance(taus, str):
        return np.array(np.round(np.array(taus) / dt), dtype=dtype)
    elif taus.startswith('octave'):
        prefix_len = len('octave')
        if len(taus) > prefix_len:
            subspacing = int(taus[prefix_len:])
        else:
            subspacing = 1
        return sample_logspace(num, base=2, subspacing=subspacing, dtype=dtype)
    elif taus.startswith('decade'):
        prefix_len = len('decade')
        if len(taus) > prefix_len:
            subspacing = int(taus[prefix_len:])

        else:
            subspacing = [1, 2, 4]  # Match default decade spacing in allantools and Stable32
        return sample_logspace(num, base=10, subspacing=subspacing, dtype=dtype)
    else:
        raise ValueError(f"Invalid taus: {taus!r}. Must be one of: 'octave[n]' or 'decade[n]'")


def integrate_samples(x: np.ndarray) -> np.ndarray:
    # Subtract mean to avoid loss of precision from accumulator overflow in integrated values
    x = x - np.mean(x)
    sum_x = np.empty(len(x) + 1, dtype=x.dtype)
    sum_x[0] = 0
    np.cumsum(x, out=sum_x[1:])
    return sum_x
