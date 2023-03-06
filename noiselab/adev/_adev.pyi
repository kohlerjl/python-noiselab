import typing as typ
import numpy as np

def oavar_integrated(sum_x: np.ndarray, strides: np.ndarray, out: np.ndarray) -> None: ...

class RealtimeADEV:

    dt: float
    record_size: int
    total_samples: int

    taus: np.ndarray
    accumulators: Accumulators
    counts: Counts
    record: Record

    def __init__(self, dt: float, *, taus: str | typ.Sequence[float] = 'octave', max_tau=-1.0, data_type='averaged'): ...

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> float: ...

    @property
    def record_length(self) -> int: ...

    def append(self, x: float) -> None: ...

    def extend(self, data: np.ndarray) -> None: ...

    def avar(self) -> np.ndarray: ...


class Counts:
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> int: ...


class Accumulators:
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> float: ...

class Record:
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> float: ...
