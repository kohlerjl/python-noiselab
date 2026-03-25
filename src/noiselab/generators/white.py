from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import DTypeLike

from .base import GeneratorBase, TShape


class WhitenoiseProcess(GeneratorBase):
    def __init__(
        self,
        *,
        psd: float = 1.0,
        mean: float = 0.0,
        shape: TShape = 1,
        dtype: DTypeLike = np.double,
        rng: Generator | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(shape=shape, dtype=dtype, rng=rng, seed=seed)

        self._psd = psd
        self.mean = mean

    def sample(self, num: int, *, dt: float) -> np.ndarray:
        size = self.sample_size(num)
        out = np.sqrt(self._psd / dt) * self.rng.standard_normal(
            size=size, dtype=self.dtype,
        )

        if self.mean:
            out += self.mean
        return out

    def auto_correlation(self, tau: np.ndarray, dt: float) -> np.ndarray:
        return np.where(np.isclose(tau, 0), self._psd / dt, 0)

    def psd(self, f: np.ndarray) -> np.ndarray:
        return np.full_like(f, self._psd)

    def avar(self, tau: np.ndarray) -> np.ndarray:
        return self._psd / tau
