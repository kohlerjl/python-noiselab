from __future__ import annotations

from .base import GeneratorBase

import numpy as np
from numpy.random import Generator


class WienerProcess(GeneratorBase):

    def __init__(self, diffusion_rate: float, mean=0.0, init_var=0.0, rng: Generator = None, seed=None):
        super().__init__(rng=rng, seed=seed)
        self.diffusion_rate = diffusion_rate
        self.mean = mean
        self.init_var = init_var

        self._state = self.mean + np.sqrt(init_var) * self.rng.standard_normal(1)

    def reset(self, init=None, seed=None):
        with self._lock:
            super().reset(seed=seed)
            if init is not None:
                self._state = init
            else:
                self._state = self.mean + np.sqrt(self.init_var) * self.rng.standard_normal(1)

    def sample(self, num: int, dt: float) -> np.ndarray:
        dx = self.rng.standard_normal(size=num, dtype=np.double)
        with self._lock:
            x = self._state + np.sqrt(dt * self.diffusion_rate) * np.cumsum(dx)
            self._state = x[-1]
            return x

    def psd(self, f: np.ndarray) -> np.ndarray:
        return self.diffusion_rate / (2*np.pi*f)**2
