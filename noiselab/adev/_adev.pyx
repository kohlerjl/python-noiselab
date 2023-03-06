#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
#distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport ceil

from .base import get_strides

cdef double NaN = np.nan

cpdef void oavar_integrated(double [::1] x, Py_ssize_t [::1] strides, double [::1] out) nogil:

    cdef Py_ssize_t N = len(x)  # Integrated data is one longer than interval number, N = M + 1
    cdef Py_ssize_t num_strides = len(strides)

    cdef double accumulator
    cdef Py_ssize_t idx_stride, num_terms, idx, n

    for idx_stride in range(num_strides):
        n = strides[idx_stride]

        num_terms = N - 2*n
        if n < 1 or num_terms <= 0:
            out[idx_stride] = NaN
            continue

        accumulator = 0
        for idx in range(num_terms):
            accumulator += (x[idx + 2*n] - 2*x[idx + n] + x[idx])**2
        out[idx_stride] = accumulator / (2 * (<double> n)*(<double> n) * (<double> num_terms))  # Convert to double first, to avoid integer overflow


cdef class RealtimeADEV:
    cdef readonly double dt

    cdef bint integrated

    cdef Py_ssize_t num_strides
    cdef Py_ssize_t* strides
    cdef double* _accumulators
    cdef Py_ssize_t* _counts

    # Circular buffer for signal memory
    cdef readonly Py_ssize_t total_samples
    cdef readonly Py_ssize_t record_size
    cdef double* _record
    cdef Py_ssize_t idx_read
    cdef Py_ssize_t idx_write

    cdef readonly np.ndarray taus
    cdef readonly Accumulators accumulators
    cdef readonly Counts counts
    cdef readonly Record record

    def __init__(self, double dt, *, taus, double max_tau = -1, str data_type='averaged'):
        self.dt = dt

        if data_type == 'averaged':
            self.integrated = False
        elif data_type == 'integrated':
            self.integrated = True
        else:
            raise ValueError(f"Invalid data_type: {data_type!r}. Must be one of: 'averaged' or 'integrated'")

        cdef Py_ssize_t max_record
        if max_tau > 0:
            max_record = <Py_ssize_t> (2*ceil(max_tau / dt) + 1)
        else:
            max_record = 2**21

        cdef Py_ssize_t [::1] strides = get_strides(dt=dt, num=max_record, taus=taus)
        self.num_strides = len(strides)

        self.taus = np.zeros(shape=self.num_strides, dtype=np.double)
        self.strides = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*self.num_strides)
        if not self.strides:
            raise MemoryError()

        self._accumulators = <double*> malloc(sizeof(double)*self.num_strides)
        if not self._accumulators:
            raise MemoryError()

        self._counts = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*self.num_strides)
        if not self._counts:
            raise MemoryError()

        cdef Py_ssize_t max_stride = 0
        for idx in range(self.num_strides):
            if strides[idx] < 1:
                raise ValueError("Stride values must be positive and non-zero")

            self.taus[idx] = self.dt * strides[idx]
            self.strides[idx] = strides[idx]
            self._accumulators[idx] = 0.0
            self._counts[idx] = 0
            max_stride = max(max_stride, strides[idx])

        self.record_size = 2*max_stride + 1
        self._record = <double*> malloc(sizeof(double)*self.record_size)
        if not self._record:
            raise MemoryError()

        self.idx_read = 0
        self.idx_write = 0
        self.total_samples = 0

        self.counts = Counts(self)
        self.record = Record(self)
        self.accumulators = Accumulators(self)

        if not self.integrated:
            self._write(0)

    def __dealloc__(self):
        if self.strides != NULL:
            free(self.strides)

        if self._accumulators != NULL:
            free(self._accumulators)

        if self._counts != NULL:
            free(self._counts)

        if self._record != NULL:
            free(self._record)

    def __len__(self) -> int:
        return self.num_strides

    def __getitem__(self, Py_ssize_t idx) -> float:
        if idx < 0 or idx >= self.num_strides:
            raise IndexError("index out of range")
        return self._calc_avar(idx)

    cdef inline void _write(self, double x) nogil:
        cdef Py_ssize_t idx_next = (self.idx_write + 1) % self.record_size
        if self.idx_read == idx_next:
            # overwrite oldest sample
            self.idx_read = (self.idx_read + 1) % self.record_size
        self._record[self.idx_write] = x
        self.idx_write = idx_next
        self.total_samples += 1

    cdef inline double _read(self, Py_ssize_t idx) nogil:
        if idx < 0:
            return NaN
        if idx >= self._length():
            return NaN
        return self._record[(self.idx_write - 1 - idx + self.record_size) % self.record_size]

    cdef inline Py_ssize_t _length(self) nogil:
        return (self.idx_write - self.idx_read + self.record_size) % self.record_size

    cdef inline void _append(self, double x) nogil:
        if not self.integrated:
            # Calculated integrated sample
            # TODO avoid loss of precision from accumulator overflow?
            x = self._read(0) + x
        else:
            x / self.dt

        cdef double x1, x2
        cdef Py_ssize_t n
        cdef Py_ssize_t record_length
        record_length = self._length()
        for idx in range(self.num_strides):
            n = self.strides[idx]
            if record_length < 2*n:
                continue

            x1 = self._read(n-1)
            x2 = self._read(2*n-1)

            self._accumulators[idx] += ((x - 2*x1 + x2)/(<double> n))**2/2
            self._counts[idx] += 1

        self._write(x)

    cpdef void append(self, double x):
        self._append(x)

    cpdef void extend(self, double [::1] data):
        cdef Py_ssize_t num_samples = len(data)
        cdef Py_ssize_t idx_sample
        with nogil:
            for idx_sample in range(num_samples):
                self._append(data[idx_sample])

    cdef inline double _calc_avar(self, Py_ssize_t idx) nogil:
        # Convert to double first, to avoid integer overflow
        if self._counts[idx] <= 1:
            return NaN
        return self._accumulators[idx]/self._counts[idx]

    def avar(self) -> np.ndarray:
        cdef np.ndarray[double, ndim=1] avar = np.empty(shape=self.num_strides, dtype=np.double)
        for idx in range(self.num_strides):
            avar[idx] = self._calc_avar(idx)
        return avar


cdef class Counts:
    cdef RealtimeADEV parent

    def __init__(self, RealtimeADEV parent):
        self.parent = parent

    def __len__(self) -> int:
        return self.parent.num_strides

    def __getitem__(self, Py_ssize_t idx) -> int:
        if idx < 0 or idx >= self.parent.num_strides:
            raise IndexError("index out of range")
        return self.parent._counts[idx]


cdef class Record:
    cdef RealtimeADEV parent

    def __init__(self, RealtimeADEV parent):
        self.parent = parent

    def __len__(self) -> int:
        return self.parent._length()

    def __getitem__(self, Py_ssize_t idx) -> float:
        if idx < 0 or idx >= self.parent._length():
            raise IndexError("index out of range")
        return self.parent._read(idx)


cdef class Accumulators:
    cdef RealtimeADEV parent

    def __init__(self, RealtimeADEV parent):
        self.parent = parent

    def __len__(self) -> int:
        return self.parent.num_strides

    def __getitem__(self, Py_ssize_t idx) -> float:
        if idx < 0 or idx >= self.parent.num_strides:
            raise IndexError("index out of range")
        return self.parent._accumulators[idx]
