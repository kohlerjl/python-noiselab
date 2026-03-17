from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import DTypeLike

from .base import GeneratorBase, TShape


class WienerProcess(GeneratorBase):
    def __init__(
        self,
        *,
        diffusion_rate: float,
        mean: float = 0.0,
        init_var: float = 0.0,
        shape: TShape = 1,
        dtype: DTypeLike = np.double,
        rng: Generator | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(shape=shape, dtype=dtype, rng=rng, seed=seed)
        self.diffusion_rate = diffusion_rate
        self.mean = mean
        self.init_var = init_var

        self._state = self.mean + np.sqrt(init_var) * self.rng.standard_normal(
            size=self.shape, dtype=self.dtype,
        )

    def reset(self, init: float | None = None, *, seed: int | None = None) -> None:
        super().reset(init=init, seed=seed)
        if init is not None:
            self._state = np.broadcast_to(init, self.shape)
        else:
            self._state = self.mean + np.sqrt(self.init_var) * self.rng.standard_normal(
                size=self.shape, dtype=self.dtype,
            )

    def sample(self, num: int, *, dt: float) -> np.ndarray:
        size = self.sample_size(num)
        dx = self.rng.standard_normal(size=size, dtype=self.dtype)
        x = self._state + np.sqrt(dt * self.diffusion_rate) * np.cumsum(dx, axis=0)
        self._state = x[-1, ...]

        return x

    def psd(self, f: np.ndarray) -> np.ndarray:
        return self.diffusion_rate / (2 * np.pi * f) ** 2

    def avar(self, tau: np.ndarray) -> np.ndarray:
        return self.diffusion_rate * tau / 3
