#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
#distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cpdef void integrate_markov(double [::1] dW, double [::1] out, double D2, double G, double y0) nogil:
    cdef Py_ssize_t num = len(dW)
    cdef double last = y0
    for idx in range(num):
        last = last * (1 - G) + D2 * dW[idx]
        out[idx] = last
