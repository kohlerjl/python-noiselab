from __future__ import annotations

import logging
from collections import abc
from typing import Literal

import numpy as np

from ._adev import oavar_integrated
from .base import TDoubleArray, TIntArray, get_strides, integrate_samples

logger = logging.getLogger(__name__)


def oavar(data: np.ndarray, dt: float, *,
          taus: str | abc.Iterable[int] = 'octave',
          data_type: Literal['averaged', 'integrated'] = 'averaged') -> tuple[TDoubleArray, TDoubleArray, TIntArray]:

    if data_type in ('averaged', 'freq'):
        x = integrate_samples(data)
        norm = 1.0
    elif data_type in ('integrated', 'phase'):
        x = data
        norm = dt**2
    else:
        raise ValueError(f"Invalid data_type: {data_type!r}. "
                         f"Must be one of: 'averaged', 'integrated', 'freq', or 'phase'.")

    num_intervals = len(x) - 1
    strides = get_strides(dt, num=num_intervals, taus=taus)

    logger.debug('Estimating AVAR over M = %d intervals on strides n = %r', num_intervals, strides.tolist())

    out = np.empty(shape=len(strides), dtype=np.double)
    if len(strides) > 0:
        oavar_integrated(x, strides=strides, out=out)

    num_terms = num_intervals - 2 * strides + 1
    return strides * dt, out / norm, num_terms


def oadev(data: np.ndarray, dt: float, *,
          taus: str | abc.Iterable[int] = 'octave',
          data_type: Literal['averaged', 'integrated'] = 'averaged') -> tuple[TDoubleArray, TDoubleArray, TIntArray]:
    actual_taus, avar, num_terms = oavar(data, dt, taus=taus, data_type=data_type)
    return actual_taus, np.sqrt(avar), num_terms
