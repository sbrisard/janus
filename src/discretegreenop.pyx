from cython cimport boundscheck
from cython cimport cdivision
from cython cimport embedsignature
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
from fft.serial._serial_fft cimport _RealFFT2D

cdef str INVALID_B_MSG = 'shape of b must be ({0},) [was ({1},)]'

def create(green, n, h, transform=None):
    if green.dim == 2:
        return TruncatedGreenOperator2D(green, n, h, transform)
    elif green.dim == 3:
        return TruncatedGreenOperator3D(green, n, h, transform)
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(green.dim))

@embedsignature(True)
cdef class TruncatedGreenOperator:

    """

    Parameters
    ----------
    green:
        The underlying continuous green operator.
    shape:
        The shape of the spatial grid used to discretize the Green operator.
    h: float
        The size of each cell of the grid.
    transform:
        The FFT object to be used to carry out discrete Fourier transforms.

    Attributes
    ----------
    green:
        The underlying continuous green operator.
    shape:
        The shape of the spatial grid used to discretize the Green operator.
    h: float
        The size of each cell of the grid.
    dim: int
        The dimension of the physical space.
    nrows: int
        The number of rows of the Green tensor, for each frequency (dimension
        of the space of polarizations).
    ncols: int
        The number of columns of the Green tensor, for each frequency (dimension
        of the space of strains).
    """
    cdef readonly GreenOperator green
    cdef readonly tuple shape
    cdef readonly double h
    cdef readonly int dim
    cdef readonly int nrows
    cdef readonly int ncols

    cdef Py_ssize_t *n
    cdef double* k
    cdef double two_pi_over_h

    def __cinit__(self, GreenOperator green, shape, double h, transform=None):
        self.dim = len(shape)
        if self.dim != green.mat.dim:
            raise ValueError('length of shape must be {0} (was {1})'
                             .format(green.mat.dim, self.dim))
        if h <= 0.:
            raise ValueError('h must be > 0 (was {0})'.format(h))
        self.green = green
        self.h = h
        self.two_pi_over_h = 2. * M_PI / h
        self.nrows = green.nrows
        self.ncols = green.ncols
        self.k = <double *> malloc(self.dim * sizeof(double))

        self.n = <Py_ssize_t *> malloc(self.dim * sizeof(Py_ssize_t))
        cdef int i
        cdef Py_ssize_t ni
        for i in range(self.dim):
            ni = shape[i]
            if ni < 0:
                raise ValueError('shape[{0}] must be > 0 (was {1})'
                                 .format(i, ni))
            self.n[i] = shape[i]
        self.shape = tuple(shape)

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
    cdef Py_ssize_t n0, n1
    cdef _RealFFT2D transform
    cdef tuple dft_tau_shape, dft_eta_shape

    def __cinit__(self, GreenOperator green, shape, double h, transform=None):
        self.transform = transform
        if self.transform is not None:
            if ((self.transform.shape[0] != shape[0]) or
                (self.transform.shape[1] != shape[1])):
                raise ValueError('shape of transform must be {0} [was {1}]'
                                 .format(self.shape, transform.shape))
        self.n0 = self.n[0]
        self.n1 = self.n[1]
        self.dft_tau_shape = (self.transform.csize0, self.transform.csize1,
                              self.ncols)
        self.dft_eta_shape = (self.transform.csize0, self.transform.csize1,
                              self.nrows)

    @boundscheck(False)
    @wraparound(False)
    cdef inline void c_apply_all_freqs(self,
                                       double[:, :, :] tau,
                                       double[:, :, :] eta):
        cdef Py_ssize_t b0, b1, b[2]
        for b1 in range(self.n1):
            b[1] = b1
            for b0 in range(self.n0):
                b[0] = b0
                self.update(b)
                self.green.c_apply(self.k, tau[b0, b1, :], eta[b0, b1, :])

    def apply_all_freqs(self, tau, eta=None):
        check_shape_3d(tau, self.n0, self.n1, self.ncols)
        eta = create_or_check_shape_3d(eta, self.n0, self.n1, self.nrows)
        self.c_apply_all_freqs(tau, eta)
        return eta

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    def convolve(self, tau, eta=None):
        cdef double[:, :, :] tau_as_mv = tau
        check_shape_3d(tau_as_mv, self.transform.rsize0,
                       self.transform.rsize1, self.ncols)
        eta = create_or_check_shape_3d(eta, self.transform.rsize0,
                                       self.transform.rsize1, self.nrows)

        cdef double[:, :, :] dft_tau = array(self.dft_tau_shape,
                                             sizeof(double), 'd')
        cdef double[:, :, :] dft_eta = array(self.dft_eta_shape,
                                             sizeof(double), 'd')

        # Compute DFT of tau
        cdef int i
        for i in range(self.ncols):
            self.transform.r2c(tau_as_mv[:, :, i], dft_tau[:, :, i])

        # Apply Green operator frequency-wise
        cdef Py_ssize_t n0 = dft_tau.shape[0]
        cdef Py_ssize_t n1 = dft_tau.shape[1] / 2
        cdef Py_ssize_t b0, b1, b[2]

        for b1 in range(n1):
            b[1] = b1
            for b0 in range(n0):
                b[0] = b0 + self.transform.offset0
                self.update(b)
                # Apply Green operator to real part
                self.green.c_apply(self.k,
                                   dft_tau[b0, 2 * b1, :],
                                   dft_eta[b0, 2 * b1, :])
                # Apply Green operator to imaginary part
                self.green.c_apply(self.k,
                                   dft_tau[b0, 2 * b1 + 1, :],
                                   dft_eta[b0, 2 * b1 + 1, :])

        # Compute inverse DFT of eta
        for i in range(self.nrows):
            self.transform.c2r(dft_eta[:, :, i], eta[:, :, i])

        return eta

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
