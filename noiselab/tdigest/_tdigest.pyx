#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
#distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from libc.errno cimport EDOM
import numbers
import numpy as np
cimport numpy as np

cdef extern from "tdigest.h" nogil:
    struct td_histogram:
        double compression
        double min
        double max
        int cap
        int merged_nodes
        int unmerged_nodes
        long long total_compressions
        long long merged_weight
        long long unmerged_weight
        double *nodes_mean
        long long *nodes_weight

    ctypedef td_histogram td_histogram_t

    td_histogram_t *td_new(double compression)
    int td_init(double compression, td_histogram_t **result)
    void td_free(td_histogram_t *h)
    void td_reset(td_histogram_t *h)
    int td_add(td_histogram_t *h, double val, long long weight)
    int td_compress(td_histogram_t *h)
    int td_merge(td_histogram_t *h, td_histogram_t *other)
    double td_cdf(td_histogram_t *h, double x)
    double td_quantile(td_histogram_t *h, double q)
    int td_quantiles(td_histogram_t *h, const double *quantiles, double *values, size_t length)
    double td_trimmed_mean(td_histogram_t *h, double leftmost_cut, double rightmost_cut)
    double td_trimmed_mean_symmetric(td_histogram_t *h, double proportion_to_cut)
    int td_compression(td_histogram_t *h)
    long long td_size(td_histogram_t *h)
    int td_centroid_count(td_histogram_t *h)
    double td_min(td_histogram_t *h)
    double td_max(td_histogram_t *h)
    const long long *td_centroids_weight(td_histogram_t *h)
    const double *td_centroids_mean(td_histogram_t *h)
    long long td_centroids_weight_at(td_histogram_t *h, int pos)
    double td_centroids_mean_at(td_histogram_t *h, int pos)


cdef class TDigest(object):
    cdef td_histogram *td

    def __cinit__(self, double compression = 100.0):
        if td_init(compression, &self.td):
            raise RuntimeError("Failed to allocate TDigest structure")

    def __dealloc__(self):
        if self.td is not NULL:
            td_free(self.td)
            self.td = NULL

    @property
    def compression(self):
        return self.td.compression

    @property
    def size(self):
        return td_size(self.td)

    @property
    def centroid_count(self):
        return td_centroid_count(self.td)

    @property
    def min(self):
        return self.td.min

    @property
    def max(self):
        return self.td.max

    def reset(self):
        td_reset(self.td)

    cpdef void compress(self):
        cdef int res
        res = td_compress(self.td)
        if res == EDOM:
            raise RuntimeError("Overflow")

    def add(self, double val, long long weight=1):
        cdef int res
        res = td_add(self.td, val, weight)
        if res == EDOM:
            raise RuntimeError("Overflow")

    def extend(self, double [::1] values not None, long long [::1] weights = None):
        cdef Py_ssize_t idx
        cdef int res = 0
        if weights is not None:
            if values.shape[0] != weights.shape[0]:
                raise ValueError("Length of values and weights do not match")

            with nogil:
                for idx in range(values.shape[0]):
                    res = td_add(self.td, values[idx], weights[idx])
                    if res:
                        break
        else:
            with nogil:
                for idx in range(values.shape[0]):
                    res = td_add(self.td, values[idx], 1)
                    if res:
                        break

        if res == EDOM:
            raise RuntimeError("Overflow")

    def merge(self, TDigest other):
        cdef int res
        res = td_merge(self.td, other.td)
        if res == EDOM:
            raise RuntimeError("Overflow")

    def cdf(self, double x):
        return td_cdf(self.td, x)

    def cdfs(self, double [::1] x):
        cdef np.ndarray[np.double_t, ndim=1] result
        cdef Py_ssize_t num, idx

        num = x.shape[0]
        result = np.empty(shape=num, dtype=np.double)
        for idx in range(num):
            result[idx] = td_cdf(self.td, x[idx])
        return result

    def quantile(self, double q):
        return td_quantile(self.td, q)

    def quantiles(self, double [::1] qs not None):
        cdef Py_ssize_t num
        cdef np.ndarray[np.double_t, ndim=1] values

        num = len(qs)
        values = np.empty(shape=num, dtype=np.double)
        td_quantiles(self.td, &qs[0], &values[0], num)
        return values

    def trimmed_mean(self, double left, double right):
        return td_trimmed_mean(self.td, left, right)

    def trimmed_mean_symmetric(self, double trim):
        return td_trimmed_mean_symmetric(self.td, trim)

    def centroids(self):
        cdef long long count
        cdef np.ndarray[np.int64_t, ndim=1] weights
        cdef np.ndarray[np.double_t, ndim=1] means

        self.compress()

        count = td_centroid_count(self.td)
        weights = np.empty(shape=count, dtype=np.int64)
        means = np.empty(shape=count, dtype=np.double)
        for idx in range(count):
            weights[idx] = self.td.nodes_weight[idx]
            means[idx] = self.td.nodes_mean[idx]
        return means, weights
