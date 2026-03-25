from __future__ import annotations

import numpy as np

from ._markov import integrate_markov
from .base import GeneratorBase, TShape


class MarkovProcess(GeneratorBase):
    def __init__(
        self,
        *,
        diffusion_rate: float | None = None,
        correlation_time: float | None = None,
        var: float | None = None,
        mean: float = 0.0,
        shape: TShape = 1,
        dtype: np.dtype = np.double,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(shape=shape, dtype=dtype, rng=rng, seed=seed)

        if diffusion_rate is not None:
            self.diffusion_rate: float = float(diffusion_rate)
        elif var is not None and correlation_time is not None:
            self.diffusion_rate = float(var * 2 / correlation_time)
        else:
            raise ValueError(
                "Exactly two of diffusion_rate, correlation_time, or var must be specified.",
            )

        if correlation_time is not None:
            self.correlation_time: float = float(correlation_time)
        elif var is not None and diffusion_rate is not None:
            self.correlation_time = float(2 * var / diffusion_rate)
        else:
            raise ValueError(
                "Exactly two of diffusion_rate, correlation_time, or var must be specified.",
            )

        if var is not None:
            self.var: float = float(var)
        elif diffusion_rate is not None and correlation_time is not None:
            self.var = float(diffusion_rate * correlation_time / 2)
        else:
            raise ValueError(
                "Exactly two of diffusion_rate, correlation_time, or var must be specified.",
            )

        self.mean = mean

        self._state = np.sqrt(self.var) * self.rng.standard_normal(
            self.shape, dtype=self.dtype,
        )

    def reset(self, init: float | None = None, *, seed: int | None = None) -> None:
        super().reset(init=init, seed=seed)

        if init is not None:
            self._state = np.broadcast_to(init - self.mean, self.shape)
        else:
            self._state = np.sqrt(self.var) * self.rng.standard_normal(
                self.shape, dtype=self.dtype,
            )

    def sample(self, num: int, *, dt: float) -> np.ndarray:
        size = self.sample_size(num)
        dW = self.rng.standard_normal(size=size, dtype=self.dtype)
        out = np.empty(shape=size, dtype=self.dtype)
        # TODO vectorize
        integrate_markov(
            dW,
            out,
            y0=self._state.item(),
            D2=np.sqrt(dt * self.diffusion_rate),
            G=dt / self.correlation_time,
        )
        self._state = out[-1]

        if self.mean:
            out += self.mean

        return out

    def auto_correlation(self, tau: np.ndarray) -> np.ndarray:
        return self.var * np.exp(-np.abs(tau) / self.correlation_time)

    def psd(self, f: np.ndarray) -> np.ndarray:
        x = 2 * np.pi * self.correlation_time * f
        return self.diffusion_rate * self.correlation_time**2 / (x**2 + 1)

    def avar(self, tau: np.ndarray) -> np.ndarray:
        x = tau / self.correlation_time
        return (
            2
            * self.diffusion_rate
            * self.correlation_time
            * (4 * np.exp(-x) - np.exp(-2 * x) - 3 + 2 * x)
            / (4 * x**2)
        )
