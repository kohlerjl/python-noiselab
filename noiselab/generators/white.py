import numpy as np
from numpy.random import Generator

from .base import GeneratorBase


class WhitenoiseProcess(GeneratorBase):

    def __init__(self, *, psd: float = 1.0, mean: float = 0.0, rng: Generator | None = None, seed: int | None = None):
        super().__init__(rng=rng, seed=seed)

        self.psd = psd
        self.mean = mean

    def sample(self, num: int, dt: float) -> np.ndarray:
        out = np.sqrt(self.psd) * np.random.normal(num)
        if self.mean:
            out += self.mean
        return out

    def auto_correlation(self, tau: np.ndarray, dt: float) -> np.ndarray:
        return np.where(np.isclose(tau, 0), self.psd / dt, 0)

    def psd(self, f: np.ndarray) -> np.ndarray:
        return np.full_like(f, self.psd)
