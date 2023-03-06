import numpy as np

from .base import GeneratorBase
from ._relaxation import RelaxationGenerator


class RelaxationProcess(GeneratorBase):

    def __init__(self, alpha: float, f_min: float, f_max: float, mean=0.0, gaussianity=10.0, decay_max=20.0,
                                rng: np.random.Generator = None, seed: int = None, cache_size=65535):
        super().__init__(rng, seed)

        self.mean = mean
        self.gen = RelaxationGenerator(alpha, f_min, f_max, gaussianity=gaussianity, decay_max=decay_max,
                                       rng=self.rng.bit_generator, cache_size=cache_size)

    def __getattr__(self, item):
        return getattr(self.gen, item)

    def reset(self, init=None, seed=None):
        with self. _lock:
            super().reset(seed=seed)
            # TODO implement reset
            raise NotImplementedError()

    def sample(self, num: int, dt: float) -> np.ndarray:
        with self._lock:
            out = self.gen.get_samples(num=num, dt=dt)
        if self.mean:
            out += self.mean
        return out

