from __future__ import annotations

import logging
import typing

import numpy as np

from .base import get_strides, integrate_samples
from ._adev import oavar_integrated

logger = logging.getLogger(__name__)


def oavar(data: np.ndarray, dt: float, *,
          taus: str | typing.Sequence[int] = 'octave',
          data_type='averaged') -> (np.ndarray, np.ndarray, np.ndarray):

    if data_type == 'averaged' or data_type == 'freq':
        x = integrate_samples(data)
        norm = 1
    elif data_type == 'integrated' or data_type == 'phase':
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
    return strides*dt, out/norm, num_terms


def oadev(data: np.ndarray, dt: float, *,
          taus: str | typing.Sequence[int] = 'octave',
          data_type='averaged') -> (np.ndarray, np.ndarray, np.ndarray):
    taus, avar, num_terms = oavar(data, dt, taus=taus, data_type=data_type)
    return taus, np.sqrt(avar), num_terms
