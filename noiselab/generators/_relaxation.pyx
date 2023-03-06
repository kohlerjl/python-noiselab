#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
#distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import time
import numpy as np
import logging
from scipy import special

cimport numpy as np
from libc.math cimport pow, sqrt, log, exp
from libc.stdlib cimport malloc, free

from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random cimport bitgen_t, BitGenerator
from numpy.random.c_distributions cimport random_standard_exponential_fill_f, random_standard_uniform_fill_f

logger = logging.getLogger(__name__)


cdef struct node:
    node *next
    double t_start
    double t_finish
    double decay_rate

cdef class RelaxationGenerator:

    cdef public double transition_rate
    cdef double tau
    cdef public double mean_inv_lambda
    cdef public double fill_time

    cdef double average
    cdef double stddev
    cdef public double skewness
    cdef public double gaussianity

    cdef public double lambda_min
    cdef public double lambda_max
    cdef double decay_max  # number of 1/e time constants to calculate for each transition
    cdef double beta
    cdef public double beta0

    cdef public double lb_min
    cdef public double lb_max
    cdef double delta_lb
    cdef double binv

    cdef node *list
    cdef public Py_ssize_t list_length
    cdef public Py_ssize_t mean_list_length
    cdef public Py_ssize_t max_list_length

    cdef double t_transition  # time of last transition
    cdef public double t_last  # time of last sample
    cdef double last_signal  # Last signal value

    # Random number generator
    cdef public BitGenerator rng
    cdef bitgen_t *bitgen

    # Input noise sample cache
    cdef public Py_ssize_t cache_size 
    cdef Py_ssize_t exp_index
    cdef float* exp_cache
    cdef Py_ssize_t uni_index
    cdef float* uni_cache

    def __init__(self, double alpha, double f_min, double f_max, double gaussianity = 10.0, double decay_max = 20.0,
                                BitGenerator rng = None, seed = None, Py_ssize_t cache_size=2**16):
        if alpha < 0 or alpha > 4:
            raise ValueError("Invalid noise exponent alpha. Only 0 <= alpha <=4 is supported")

        self.list = NULL
        self.list_length = 0
        self.max_list_length = 0

        self.t_transition = 0
        self.t_last = 0
        self.last_signal = 0

        self.gaussianity = gaussianity
        self.lambda_min = 2*np.pi*f_min
        self.lambda_max = 2*np.pi*f_max
        self.decay_max = decay_max
        self.beta = alpha-1.

        if rng is None:
            if seed is None:
                seed = time.time_ns()
            self.rng = np.random.default_rng(seed=seed).bit_generator
        else:
            self.rng = rng

        capsule = self.rng.capsule
        # Optional check that the capsule is from a BitGenerator
        if not PyCapsule_IsValid(capsule, "BitGenerator"):
            raise ValueError("Invalid rng type. Must be BitGenerator instance.")
        # Cast the pointer
        self.bitgen = <bitgen_t *> PyCapsule_GetPointer(capsule, "BitGenerator")

        self.cache_size = cache_size
        self.exp_index = cache_size    
        self.uni_index = cache_size

        self.exp_cache = <float*> malloc(sizeof(float)*self.cache_size)
        if not self.exp_cache:
            raise MemoryError()

        self.uni_cache = <float*> malloc(sizeof(float)*self.cache_size)
        if not self.uni_cache:
            raise MemoryError()

        if not self._init_generator():
            raise MemoryError()

    cdef bint _init_generator(self) nogil:
        if self.beta <= 1:
            self.beta0 = self.beta
        elif self.beta > 1 and self.beta <=3:
            self.beta0 = self.beta - 2.

        if self.lambda_max == 0 or self.lambda_max <= self.lambda_min or self.beta0 == 1:
            self.lambda_max = self.lambda_min
            self.mean_inv_lambda = 1./(self.lambda_min)
            self.beta0 = 1
        else:
            if self.beta0 == 0:
                self.mean_inv_lambda = log(self.lambda_max / self.lambda_min)/(self.lambda_max - self.lambda_min)
            else:
                self.mean_inv_lambda = -((1.-self.beta0)/self.beta0) * (pow(self.lambda_max, -self.beta0) - pow(self.lambda_min,-self.beta0)) \
                                                                / (pow(self.lambda_max, 1.-self.beta0) - pow(self.lambda_min,1.-self.beta0))

        self.transition_rate = self.gaussianity/self.mean_inv_lambda
        self.tau = 1. / self.transition_rate

        self.average = self.transition_rate * self.mean_inv_lambda
        self.stddev = sqrt(self.average / 2.0)
        self.skewness = pow(2.0, 1.5)/(3.0 * sqrt(self.average))

        self.lb_min = pow(self.lambda_min, 1.-self.beta0)
        self.lb_max = pow(self.lambda_max, 1.-self.beta0)
        self.delta_lb = self.lb_max - self.lb_min
        if self.lambda_max != self.lambda_min:
            self.binv = 1./(1.-self.beta0)
        else:
            self.binv = 0.

        self.mean_list_length = <int> (self.decay_max*self.average)
        self.fill_time = self.decay_max / self.lambda_min    
        if not self._advance(self.fill_time):
            return False

        self.t_last = self.fill_time
        self._clean_list(self.t_transition)
        return True

    def __dealloc__(self):
        cdef node *current
        while self.list != NULL:
            current = self.list
            self.list = self.list.next
            free(current)

        if self.exp_cache != NULL:
            free(self.exp_cache)

        if self.uni_cache != NULL:
            free(self.uni_cache)

    cdef inline bint _append(self, double t_start, double t_finish, double decay_rate) nogil:
        cdef node* newnode = <node *> malloc(sizeof(node))
        if not newnode:
            return False

        newnode.t_start = t_start
        newnode.t_finish = t_finish
        newnode.decay_rate = decay_rate
        newnode.next = self.list
        self.list = newnode
        self.list_length += 1;
        self.max_list_length = max(self.list_length, self.max_list_length)

        return True

    cdef void _clean_list(self, double t) nogil:
        cdef node *next
        cdef node *current

        # Free spent nodes at the head of the list, advancing head pointer
        while self.list != NULL and t > self.list.t_finish:
            current = self.list
            self.list = self.list.next
            free(current)

        if self.list == NULL:
            self.list_length = 0
            return

        cdef Py_ssize_t k = 1
        current = self.list
        next = current.next
        # Iterate through and remove remaining spent nodes
        while next != NULL:
            if t > next.t_finish:
                current.next = next.next
                free(next)
            else:
                k += 1
                current = next
            next = current.next

        self.list_length = k

    cdef inline float _generate_exponential(self) nogil:
        if self.exp_index == self.cache_size:
            random_standard_exponential_fill_f(self.bitgen, self.cache_size, <double*> self.exp_cache)
            self.exp_index = 0

        cdef float value = self.exp_cache[self.exp_index]
        self.exp_index = self.exp_index + 1
        return value

    cdef inline float _generate_uniform(self) nogil:
        if self.uni_index == self.cache_size:
            random_standard_uniform_fill_f(self.bitgen, self.cache_size, self.uni_cache)
            self.uni_index = 0

        cdef float value = self.uni_cache[self.uni_index]
        self.uni_index = self.uni_index + 1
        return value

    cdef inline bint _advance(self, double t) nogil:
        cdef double decay_rate
        cdef double t_finish

        while self.t_transition < t:
            # Generate new transition nodes until requested time, t
            self.t_transition += self.tau*self._generate_exponential()
            if self.beta0 != 1:
                decay_rate = pow(self.lb_min + self.delta_lb*self._generate_uniform(), self.binv)
            else:
                decay_rate = self.lambda_min

            t_finish = self.t_transition + self.decay_max/decay_rate
            if t_finish <= t:
                # Ignore transition if it finishes before t
                continue

            if not self._append(self.t_transition, t_finish, decay_rate):
                return False

        return True

    cdef inline bint _sample(self, double t) nogil:
        cdef double raw_signal

        if not self._advance(t):
            return False

        raw_signal = self._response(t)
        self.last_signal = (raw_signal - self.average)/self.stddev

        self.t_last = t
        return True

    cdef inline double _response(self, double t) nogil:
        cdef double signal = 0
        cdef node *current = self.list
        cdef node *next

        while current != NULL:
            if t > current.t_start:
                signal += exp(-(current.decay_rate)*(t - current.t_start))

            next = current.next
            # Remove any spent nodes
            while next != NULL and t > next.t_finish:
                current.next = next.next
                free(next)
                self.list_length -= 1
                next = current.next

            current = current.next
        
        return signal

    cdef inline bint _sample_integrated(self, double t) nogil:
        cdef double raw_signal

        if not self._advance(t):
            return False

        raw_signal = self._integrated_response(self.t_last, t)
        self.last_signal = self.last_signal + (raw_signal - self.average*(t-self.t_last))/self.stddev

        self.t_last = t
        return True

    cdef inline double _integrated_response(self, double t1, double t2) nogil:
        cdef double sum
        cdef double dt, dt1, dt2, tk, lk;
        cdef node *current = self.list
        cdef node *next

        if t2 <= t1:
            # return a nonzero value only if t2 > t1
            return 0
        
        dt = t2 - t1
        sum = 0

        while current != NULL:
            tk = current.t_start
            lk = current.decay_rate
            dt1 = t1 - tk
            dt2 = t2 - tk

            if dt1 > 0 and dt2 > 0:
                sum += exp(-lk*dt1)*(1. - exp(-lk*dt))/lk
            elif dt1 <= 0 and dt2 > 0:
                sum += (1. - exp(-lk*dt2))/lk

            next = current.next
            # Remove any spent nodes
            while next != NULL and t1 > next.t_finish:
                current.next = next.next
                free(next)
                self.list_length -= 1
                next = current.next

            current = current.next

        return sum;

    cpdef skip(self, double dt):
        with self.rng.lock:
            self.t_last += dt

            if not self._advance(self.t_last):
                raise MemoryError()


    cpdef np.ndarray[np.float64_t] get_samples(self, unsigned long num, double dt):
        cdef double t

        t = self.t_last

        # Memoryview on a NumPy array
        cdef np.ndarray[np.float64_t] py_arr = np.empty((num,), dtype=np.double)
        cdef double [:] arr = py_arr

        with self.rng.lock, nogil:
            if self.beta <=1:
                # beta is in the (-1,1) range

                for idx in range(num):
                    t += dt
                    if not self._sample(t):
                        raise MemoryError()
                    arr[idx] = self.last_signal
            else:
                # beta is in the (1,3) range, and we must use the integrated response

                for idx in range(num):
                    t += dt
                    if not self._sample_integrated(t):
                        raise MemoryError()
                    arr[idx] = self.last_signal

        return py_arr

    cpdef np.ndarray[np.float64_t] psd(self, np.ndarray[np.float64_t] f):
        cdef np.ndarray[np.float64_t] w = 2*np.pi*f
        cdef double scale = self.mean_inv_lambda/2*(self.lb_max - self.lb_min)
        return 1/scale/w**2 * (
                self.lb_max*special.hyp2f1(1, (1-self.beta0)/2, (3-self.beta0)/2, -self.lambda_max**2/w**2)
                - self.lb_min*special.hyp2f1(1, (1-self.beta0)/2, (3-self.beta0)/2, -self.lambda_min**2/w**2)
            )
