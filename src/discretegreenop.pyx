from cython cimport boundscheck
from cython cimport cdivision
from cython cimport Py_ssize_t
from cython cimport sizeof
from cython cimport wraparound
from cython.view cimport array
from libc.math cimport M_PI
from libc.stdlib cimport malloc
from libc.stdlib cimport free

from checkarray cimport create_or_check_shape_1d
from checkarray cimport create_or_check_shape_2d
from checkarray cimport create_or_check_shape_3d
from checkarray cimport create_or_check_shape_4d
from checkarray cimport check_shape_1d
from checkarray cimport check_shape_3d
from checkarray cimport check_shape_4d
from greenop cimport GreenOperator

cdef str INVALID_N_MSG = 'length of n must be {0} (was {1})'
cdef str INVALID_H_MSG = 'h must be > 0 (was {0})'
cdef str INVALID_B_MSG = 'shape of b must be ({0},) [was ({1},)]'

def create(green, n, h):
    if green.dim == 2:
        return TruncatedGreenOperator2D(green, n, h)
    elif green.dim == 3:
        return TruncatedGreenOperator3D(green, n, h)
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(green.dim))

cdef class TruncatedGreenOperator:
    cdef readonly GreenOperator green
    cdef readonly double h
    cdef readonly int dim
    cdef readonly int nrows
    cdef readonly int ncols
    #TODO Make this readable from Python
    cdef Py_ssize_t *n
    cdef double* k
    cdef double two_pi_over_h

    def __cinit__(self, GreenOperator green, n, double h):
        cdef Py_ssize_t d = len(n)
        if d != green.mat.dim:
            raise ValueError(INVALID_N_MSG.format(green.mat.dim, d))
        if h <= 0.:
            raise ValueError(INVALID_H_MSG.format(h))
        self.green = green
        self.h = h
        self.two_pi_over_h = 2. * M_PI / h
        self.dim = d
        self.nrows = green.nrows
        self.ncols = green.ncols
        self.n = <Py_ssize_t *> malloc(d * sizeof(Py_ssize_t))
        self.k = <double *> malloc(d * sizeof(double))
        cdef int i
        for i in range(d):
            #TODO Check for sign of shape[i]
            self.n[i] = n[i]

    def __dealloc__(self):
        free(self.n)
        free(self.k)

    cdef inline void check_b(self, Py_ssize_t[::1] b) except *:
        if b.shape[0] != self.dim:
            raise ValueError('invalid shape: expected ({0},), actual ({1},)'
                             .format(self.dim, b.shape[0]))
        cdef Py_ssize_t i, ni, bi
        for i in range(self.dim):
            ni = self.n[i]
            bi = b[i]
            if (bi < 0) or (bi >= ni):
                raise ValueError('index must be >= 0 and < {0} (was {1})'
                                 .format(ni, bi))

    @cdivision(True)
    cdef inline void update(self, Py_ssize_t *b):
        cdef:
            Py_ssize_t i, ni, bi
            double s
        for i in range(self.dim):
            ni = self.n[i]
            bi = b[i]
            s = self.two_pi_over_h / <double> ni
            if 2 * bi > ni:
                self.k[i] = s * (bi - ni)
            else:
                self.k[i] = s * bi

    cdef void c_as_array(self, Py_ssize_t *b, double[:, :] out):
        self.update(b)
        self.green.c_as_array(self.k, out)

    @boundscheck(False)
    @wraparound(False)
    def as_array(self, Py_ssize_t[::1] b, double[:, :] out=None):
        self.check_b(b)
        out = create_or_check_shape_2d(out, self.nrows, self.ncols)
        self.c_as_array(&b[0], out)
        return out

    cdef void c_apply_single_freq(self, Py_ssize_t *b,
                                  double[:] tau, double[:] eta):
        self.update(b)
        self.green.c_apply(self.k, tau, eta)

    @boundscheck(False)
    @wraparound(False)
    def apply_single_freq(self, Py_ssize_t[::1] b,
                          double[:] tau, double[:] eta=None):
        self.check_b(b)
        check_shape_1d(tau, self.ncols)
        eta = create_or_check_shape_1d(eta, self.nrows)
        self.c_apply_single_freq(&b[0], tau, eta)
        return eta

    def apply_all_freqs(self, tau, eta=None):
        pass

cdef class TruncatedGreenOperator2D(TruncatedGreenOperator):

    @boundscheck(False)
    @wraparound(False)
    cdef inline void c_apply_all_freqs(self,
                                       double[:, :, :] tau,
                                       double[:, :, :] eta):
        cdef int n0 = self.n[0]
        cdef int n1 = self.n[1]
        cdef Py_ssize_t b0, b1, b[2]
        for b1 in range(n1):
            b[1] = b1
            for b0 in range(n0):
                b[0] = b0
                self.update(b)
                self.green.c_apply(self.k, tau[b0, b1, :], eta[b0, b1, :])

    def apply_all_freqs(self, tau, eta=None):
        check_shape_3d(tau, self.n[0], self.n[1], self.ncols)
        eta = create_or_check_shape_3d(eta, self.n[0], self.n[1], self.nrows)
        self.c_apply_all_freqs(tau, eta)
        return eta

    def convolve(self, tau, eta, transform):
        cdef int n0 = self.n[0]
        cdef int n1 = self.n[1]
        cdef Py_ssize_t b0, b1, b[2]
        """
        dft_tau = np.empty(transform.cshape + (self.ncols,), dtype=np.float64)
        dft_tau = transform.r2c(tau)
        """

cdef class TruncatedGreenOperator3D(TruncatedGreenOperator):

    @boundscheck(False)
    @wraparound(False)
    cdef inline void c_apply_all_freqs(self,
                                       double[:, :, :, :] tau,
                                       double[:, :, :, :] eta):
        cdef int n0 = self.n[0]
        cdef int n1 = self.n[1]
        cdef int n2 = self.n[2]
        cdef Py_ssize_t b0, b1, b2, b[3]
        for b2 in range(n2):
            b[2] = b2
            for b1 in range(n1):
                b[1] = b1
                for b0 in range(n0):
                    b[0] = b0
                    self.update(b)
                    self.green.c_apply(self.k,
                                       tau[b0, b1, b2, :],
                                       eta[b0, b1, b2, :])

    def apply_all_freqs(self, tau, eta=None):
        check_shape_4d(tau, self.n[0], self.n[1], self.n[2], self.ncols)
        eta = create_or_check_shape_4d(eta, self.n[0], self.n[1], self.n[2],
                                       self.nrows)
        self.c_apply_all_freqs(tau, eta)
        return eta
