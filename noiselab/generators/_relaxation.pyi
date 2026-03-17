import typing as typ

import numpy as np

class RelaxationGenerator:
    transition_rate: float
    mean_inv_lambda: float
    fill_time: float
    skewness: float
    gaussianity: float

    lambda_min: float
    lambda_max: float
    beta0: float

    lb_min: float
    lb_max: float

    list_length: int
    mean_list_length: int
    max_list_length: int

    t_last: float

    rng: np.random.BitGenerator

    cache_size: int

    def __init__(
        self,
        alpha: float,
        f_min: float,
        f_max: float,
        gaussianity: float = 10.0,
        decay_max: float = 20.0,
        rng: np.random.BitGenerator | None = None,
        seed: int | None = None,
        cache_size: int = 2**16,
    ) -> None: ...

    def skip(self, dt: float) -> None: ...

    def get_samples(
        self, num: int, dt: float,

    ) -> np.ndarray[typ.Any, np.dtype[np.double]]: ...

    def psd(
        self, f: np.ndarray[typ.Any, np.dtype[np.double]],
    ) -> np.ndarray[typ.Any, np.dtype[np.double]]: ...
