from __future__ import annotations

import time
import typing as typ

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

TShape = typ.Union[int, typ.Sequence[int]]


class GeneratorBase:
    def __init__(
        self,
        *,
        shape: TShape = 1,
        dtype: npt.DTypeLike = np.double,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        self.shape: tuple[int, ...] = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype: np.dtype = np.dtype(dtype)

        if rng is None:
            if seed is None:
                seed = time.time_ns()
            self.rng: Generator = np.random.default_rng(seed)
        else:
            self.rng = rng

    def reset(self, init: float | None = None, *, seed: int | None = None) -> None:
        if seed is None:
            seed = time.time_ns()

        # Construct new rng from seed
        self.rng = Generator(self.rng.bit_generator.__class__(seed=seed))

    def sample(self, num: int, *, dt: float) -> np.ndarray:
        raise NotImplementedError()

    def sample_size(self, num: int) -> int | tuple[int, ...]:
        return num if self.shape == (1,) else (num, *self.shape)
