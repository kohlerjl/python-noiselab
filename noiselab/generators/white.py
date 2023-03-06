from .base import GeneratorBase

import numpy as np


class WhitenoiseProcess(GeneratorBase):

    def __init__(self, psd=1.0, mean=0.0, rng=None, seed=None):
        super().__init__(rng=rng, seed=seed)

        self.psd = psd
        self.mean = mean

    def sample(self, num: int, dt: float) -> np.ndarray:
        out = np.sqrt(self.psd) * np.random.normal(num)
        if self.mean:
            out += self.mean
        return out

    def auto_correlation(self, tau: np.ndarray, dt: float) -> np.ndarray:
        return np.where(np.isclose(tau, 0), self.psd/dt, 0)

    def psd(self, f: np.ndarray) -> np.ndarray:
        return np.full_like(f, self.psd)
