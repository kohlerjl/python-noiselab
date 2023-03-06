import numpy as np

from .base import GeneratorBase
from ._markov import integrate_markov


class MarkovProcess(GeneratorBase):

    def __init__(self, diffusion_rate=None, correlation_time=None, var=None, mean=0.0, rng=None, seed=None):
        super().__init__(rng=rng, seed=seed)

        if sum(x is not None for x in [diffusion_rate, correlation_time, var]) != 2:
            raise ValueError("Exactly two of diffusion_rate, correlation_time, or var must be specified.")

        if diffusion_rate is not None:
            self.diffusion_rate = diffusion_rate
        else:
            self.diffusion_rate = var * 2 / correlation_time

        if correlation_time is not None:
            self.correlation_time = correlation_time
        else:
            self.correlation_time = 2 * var / diffusion_rate

        if var is not None:
            self.var = var
        else:
            self.var = diffusion_rate * correlation_time / 2

        self.mean = mean

        self._state = np.sqrt(self.var) * self.rng.standard_normal(1, dtype=np.double)

    def reset(self, init=None, seed=None):
        with self. _lock:
            super().reset(seed=seed)

            if init is not None:
                self._state = init - self.mean
            else:
                self._state = np.sqrt(self.var) * self.rng.standard_normal(1, dtype=np.double)

    def sample(self, num: int, dt: float) -> np.ndarray:
        dW = self.rng.standard_normal(num, dtype=np.double)
        out = np.empty(shape=num, dtype=np.float64)
        with self._lock:
            integrate_markov(dW, out, y0=self._state, D2=np.sqrt(dt * self.diffusion_rate), G=dt/self.correlation_time)
            self._state = out[-1]

        if self.mean:
            out += self.mean

        return out

    def auto_correlation(self, tau: np.ndarray) -> np.ndarray:
        return self.var * np.exp(-np.abs(tau) / self.correlation_time)

    def psd(self, f: np.ndarray) -> np.ndarray:
        x = 2*np.pi*self.correlation_time*f
        return self.diffusion_rate * self.correlation_time**2 / (x**2 + 1)
