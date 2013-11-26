from cython cimport boundscheck
from cython cimport cdivision
from cython cimport Py_ssize_t
from cython cimport sizeof
from cython cimport wraparound
from cython.view cimport array
from libc.math cimport M_PI
from libc.stdlib cimport malloc
from libc.stdlib cimport free

from checkarray cimport create_or_check_shape_2d
from greenop cimport GreenOperator

cdef str INVALID_N_MSG = 'length of n must be {0} (was {1})'
cdef str INVALID_H_MSG = 'h must be > 0 (was {0})'
cdef str INVALID_B_MSG = 'shape of b must be ({0},) [was ({1},)]'

cdef class TruncatedGreenOperator:
    cdef readonly GreenOperator green
    cdef readonly double h
    # Size of the space of symmetric, second-rank tensors
    cdef int sym
    #TODO Make this readable from Python
    cdef Py_ssize_t *n
    cdef double* k
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
        self.sym = (d * (d + 1)) / 2
        self.n = <Py_ssize_t *> malloc(d * sizeof(Py_ssize_t))
        self.k = <double *> malloc(d * sizeof(double))
        cdef int i
        for i in range(d):
            #TODO Check for sign of shape[i]
            self.n[i] = n[i]

    def __dealloc__(self):
        free(self.n)
        free(self.k)

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
        self.green.c_apply(self.k, tau, eta)
        return eta

    def apply_all_freqs(self, tau, eta=None):
        pass

    cpdef double[:, :] asarray(self, Py_ssize_t[:] b, double[:, :] out=None):
        self.update(b)
        out = create_or_check_shape_2d(out, self.sym, self.sym)
        self.green.c_as_array(self.k, out)
        return out

cdef class TruncatedGreenOperator2D(TruncatedGreenOperator):
    cdef inline void check_grid_shape(self,
                                      double[:, :, :] a,
                                      name) except *:
        if ((a.shape[0] != self.n[1])
            or (a.shape[1] != self.n[0])
            or (a.shape[2] != self.sym)):
            raise ValueError('shape of {0} must be ({1}, {2}) [was ({3}, {4})]'
                             .format(name,
                                     self.n[1], self.n[0],
                                     a.shape[0], a.shape[1]))

    @boundscheck(False)
    cdef inline double[:, :, :]  capply_all_freqs(self,
                                                  double[:, :, :] tau,
                                                  double[:, :, :] eta):
        cdef int n0 = self.n[0]
        cdef int n1 = self.n[1]
        cdef Py_ssize_t b0, b1
        cdef Py_ssize_t b[2]
        for b1 in range(n1):
            b[1] = b1
            for b0 in range(n0):
                b[0] = b0
                self.update(b)
                self.green.c_apply(self.k, tau[b1, b0, :], eta[b1, b0, :])
        return eta

    def apply_all_freqs(self, tau, eta=None):
        cdef double[:, :, :] tau_mv = tau
        self.check_grid_shape(tau_mv, 'tau')

        cdef double[:, :, :] eta_mv = eta
        if eta is not None:
            self.check_grid_shape(eta_mv, 'eta')
        else:
            eta_mv = array(shape=(self.n[1], self.n[0], self.sym),
                           itemsize=sizeof(double),
                           format='d')
        return self.capply_all_freqs(tau_mv, eta_mv)
