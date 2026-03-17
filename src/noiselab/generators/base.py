from __future__ import annotations

import threading
import time

import numpy as np
from numpy.random import Generator


class GeneratorBase:

    def __init__(self, *, rng: Generator | None = None, seed: int | None = None):
        if rng is None:
            if seed is None:
                seed = time.time_ns()
            self.rng: Generator = np.random.default_rng(seed)
        else:
            self.rng = rng

        self._lock = threading.RLock()

    def reset(self, init: float | None = None, *, seed: int | None = None) -> None:
        if seed is None:
            seed = time.time_ns()

        # Construct new rng from seed
        self.rng = Generator(self.rng.bit_generator.__class__(seed=seed))

    def sample(self, num: int, dt: float) -> np.ndarray:
        raise NotImplementedError()
