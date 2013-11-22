from cython cimport boundscheck
from cython cimport cdivision
from cython cimport Py_ssize_t
from cython cimport sizeof
from cython cimport wraparound
from cython.view cimport array
from libc.math cimport M_PI
from libc.stdlib cimport malloc
from libc.stdlib cimport free

from greenop cimport GreenOperator

cdef str INVALID_N_MSG = 'length of n must be {0} (was {1})'
cdef str INVALID_H_MSG = 'h must be > 0 (was {0})'
cdef str INVALID_B_MSG = 'shape of b must be ({0},) [was ({1},)]'

cdef class TruncatedGreenOperator:
    cdef readonly GreenOperator green
    cdef readonly double h
    #TODO Make this readable from Python
    cdef Py_ssize_t *n
    cdef double[::1] k
    cdef double two_pi_over_h

    @cdivision(True)
    def __cinit__(self, GreenOperator green, tuple n not None, double h):
        cdef Py_ssize_t d = len(n)
        if d != green.mat.dim:
            raise ValueError(INVALID_N_MSG.format(green.mat.dim, d))
        if h <= 0.:
            raise ValueError(INVALID_H_MSG.format(h))
        self.green = green
        self.h = h
        self.two_pi_over_h = 2. * M_PI / h
        self.k = array(shape=(d,), itemsize=sizeof(double), format='d')
        self.n = <Py_ssize_t *> malloc(d * sizeof(Py_ssize_t))
        cdef int i
        for i in range(d):
            #TODO Check for sign of shape[i]
            self.n[i] = n[i]

    def __dealloc__(self):
        free(self.n)

    cdef inline void check_b(self, Py_ssize_t[:] b) except *:
    # TODO Improve quality of checks and error messages.
        if b.shape[0] != self.green.mat.dim:
            raise ValueError(INVALID_B_MSG.format(self.green.mat.dim,
                                                  b.shape[0]))
        cdef Py_ssize_t i, ni, bi
        for i in range(self.green.mat.dim):
            ni = self.n[i]
            bi = b[i]
            if (bi < 0) or (bi >= ni):
                raise ValueError('')

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    cdef void update(self, Py_ssize_t[:] b):
        cdef:
            Py_ssize_t i, ni, bi
            double s
        for i in range(self.green.mat.dim):
            ni = self.n[i]
            bi = b[i]
            s = self.two_pi_over_h / <double> ni
            if 2 * bi > ni:
                self.k[i] = s * (bi - ni)
            else:
                self.k[i] = s * bi

    cpdef double[:] apply_single_freq(self,
                                      Py_ssize_t[:] b,
                                      double[:] tau,
                                      double[:] eta=None):
        self.check_b(b)
        self.update(b)
        return self.green.apply(self.k, tau, eta)

    def apply_all_freqs(self, tau, eta=None):
        pass

    cpdef double[:, :] asarray(self, Py_ssize_t[:] b, double[:, :] a=None):
        self.update(b)
        return self.green.asarray(self.k, a)

cdef class TruncatedGreenOperator2D(TruncatedGreenOperator):

    @boundscheck(False)
    def apply_all_freqs(self, tau, eta=None):
        cdef double[:, :, :] tau_mv = tau
        cdef double[:, :, :] eta_mv = eta
        cdef int n0 = self.n[0]
        cdef int n1 = self.n[1]
        cdef Py_ssize_t b0, b1
        cdef Py_ssize_t b[2]
        for b1 in range(n1):
            b[1] = b1
            for b0 in range(n0):
                b[0] = b0
                self.update(b)
                self.green.apply(self.k,
                                 tau_mv[b1, b0, :],
                                 eta_mv[b1, b0, :])
        return eta_mv
