# TODO do not use pointers, but memoryviews.
from cython cimport boundscheck
from cython cimport cdivision
from cython cimport Py_ssize_t
from cython cimport sizeof
from cython cimport wraparound
from cython.view cimport array
from libc.math cimport M_PI
from libc.stdlib cimport malloc
from libc.stdlib cimport free

from janus.fft.serial._serial_fft cimport _RealFFT2D
from janus.fft.serial._serial_fft cimport _RealFFT3D
from janus.greenop cimport GreenOperator
from janus.utils.checkarray cimport create_or_check_shape_1d
from janus.utils.checkarray cimport create_or_check_shape_2d
from janus.utils.checkarray cimport create_or_check_shape_3d
from janus.utils.checkarray cimport create_or_check_shape_4d
from janus.utils.checkarray cimport check_shape_1d
from janus.utils.checkarray cimport check_shape_3d
from janus.utils.checkarray cimport check_shape_4d

def create(green, n, h, transform=None):
    if green.dim == 2:
        return TruncatedGreenOperator2D(green, n, h, transform)
    elif green.dim == 3:
        return TruncatedGreenOperator3D(green, n, h, transform)
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(green.dim))


cdef class DiscreteGreenOperator:

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

    cdef int *n

    def __cinit__(self, GreenOperator green, shape, double h, transform=None):
        self.dim = len(shape)
        if self.dim != green.mat.dim:
            raise ValueError('length of shape must be {0} (was {1})'
                             .format(green.mat.dim, self.dim))
        if h <= 0.:
            raise ValueError('h must be > 0 (was {0})'.format(h))
        self.green = green
        self.h = h
        self.nrows = green.nrows
        self.ncols = green.ncols

        self.n = <int *> malloc(self.dim * sizeof(int))
        cdef int i
        cdef int ni
        for i in range(self.dim):
            ni = shape[i]
            if ni < 0:
                raise ValueError('shape[{0}] must be > 0 (was {1})'
                                 .format(i, ni))
            self.n[i] = shape[i]
        self.shape = tuple(shape)

    def __dealloc__(self):
        free(self.n)

    cdef inline void check_b(self, int[::1] b) except *:
        if b.shape[0] != self.dim:
            raise ValueError('invalid shape: expected ({0},), actual ({1},)'
                             .format(self.dim, b.shape[0]))
        cdef int i, ni, bi
        for i in range(self.dim):
            ni = self.n[i]
            bi = b[i]
            if (bi < 0) or (bi >= ni):
                raise ValueError('index must be >= 0 and < {0} (was {1})'
                                 .format(ni, bi))

    cdef void c_as_array(self, int *b, double[:, :] out):
        raise NotImplementedError

    cdef void c_apply(self, int *b, double[:] tau, double[:] eta):
        raise NotImplementedError

    @boundscheck(False)
    @wraparound(False)
    def as_array(self, int[::1] b, double[:, :] out=None):
        self.check_b(b)
        out = create_or_check_shape_2d(out, self.nrows, self.ncols)
        self.c_as_array(&b[0], out)
        return out

    @boundscheck(False)
    @wraparound(False)
    def apply(self, int[::1] b, double[:] tau, double[:] eta=None):
        self.check_b(b)
        check_shape_1d(tau, self.ncols)
        eta = create_or_check_shape_1d(eta, self.nrows)
        self.c_apply(&b[0], tau, eta)
        return eta


cdef class TruncatedGreenOperator(DiscreteGreenOperator):

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

    cdef double* k
    cdef double two_pi_over_h

    def __cinit__(self, GreenOperator green, shape, double h, transform=None):
        self.two_pi_over_h = 2. * M_PI / h
        self.k = <double *> malloc(self.dim * sizeof(double))

    def __dealloc__(self):
        free(self.k)

    @cdivision(True)
    cdef inline void update(self, int *b):
        cdef:
            int i, ni, bi
            double s
        for i in range(self.dim):
            ni = self.n[i]
            bi = b[i]
            s = self.two_pi_over_h / <double> ni
            if 2 * bi > ni:
                self.k[i] = s * (bi - ni)
            else:
                self.k[i] = s * bi

    cdef void c_as_array(self, int *b, double[:, :] out):
        self.update(b)
        self.green.c_as_array(self.k, out)

    cdef void c_apply(self, int *b, double[:] tau, double[:] eta):
        self.update(b)
        self.green.c_apply(self.k, tau, eta)


cdef class TruncatedGreenOperator2D(TruncatedGreenOperator):
    cdef int n0, n1
    cdef double s0, s1
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
        self.dft_tau_shape = (self.transform.cshape0, self.transform.cshape1,
                              self.ncols)
        self.dft_eta_shape = (self.transform.cshape0, self.transform.cshape1,
                              self.nrows)
        self.s0 = 2. * M_PI / (self.h * self.n0)
        self.s1 = 2. * M_PI / (self.h * self.n1)

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    def convolve(self, tau, eta=None):
        cdef double[:, :, :] tau_as_mv = tau
        check_shape_3d(tau_as_mv, self.transform.rshape0,
                       self.transform.rshape1, self.ncols)
        eta = create_or_check_shape_3d(eta, self.transform.rshape0,
                                       self.transform.rshape1, self.nrows)

        cdef double[:, :, :] dft_tau = array(self.dft_tau_shape,
                                             sizeof(double), 'd')
        cdef double[:, :, :] dft_eta = array(self.dft_eta_shape,
                                             sizeof(double), 'd')

        cdef int i

        # Compute DFT of tau
        for i in range(self.ncols):
            self.transform.r2c(tau_as_mv[:, :, i], dft_tau[:, :, i])

        # Apply Green operator frequency-wise
        cdef int n0 = dft_tau.shape[0]
        cdef int n1 = dft_tau.shape[1] / 2
        cdef int i0, i1, b0, b1
        cdef double k[2]

        for i0 in range(n0):
            b0 = i0 + self.transform.offset0
            if 2 * b0 > self.n0:
                k[0] = self.s0 * (b0 - self.n0)
            else:
                k[0] = self.s0 * b0

            i1 = 0
            for b1 in range(n1):
                # At this point, i1 = 2 * b1
                if i1 > self.n1:
                    k[1] = self.s1 * (b1 - self.n1)
                else:
                    k[1] = self.s1 * b1

                # Apply Green operator to real part
                self.green.c_apply(k,
                                   dft_tau[i0, i1, :],
                                   dft_eta[i0, i1, :])
                i1 += 1
                # Apply Green operator to imaginary part
                self.green.c_apply(k,
                                   dft_tau[i0, i1, :],
                                   dft_eta[i0, i1, :])
                i1 += 1

        # Compute inverse DFT of eta
        for i in range(self.nrows):
            self.transform.c2r(dft_eta[:, :, i], eta[:, :, i])

        return eta


cdef class TruncatedGreenOperator3D(TruncatedGreenOperator):
    cdef int n0, n1, n2
    cdef double s0, s1, s2
    cdef _RealFFT3D transform
    cdef tuple dft_tau_shape, dft_eta_shape

    def __cinit__(self, GreenOperator green, shape, double h, transform=None):
        self.transform = transform
        if self.transform is not None:
            if ((self.transform.shape[0] != shape[0]) or
                (self.transform.shape[1] != shape[1]) or
                (self.transform.shape[2] != shape[2])):
                raise ValueError('shape of transform must be {0} [was {1}]'
                                 .format(self.shape, transform.shape))
        self.n0 = self.n[0]
        self.n1 = self.n[1]
        self.n2 = self.n[2]
        self.dft_tau_shape = (self.transform.cshape0, self.transform.cshape1,
                              self.transform.cshape2, self.ncols)
        self.dft_eta_shape = (self.transform.cshape0, self.transform.cshape1,
                              self.transform.cshape2, self.nrows)
        self.s0 = 2. * M_PI / (self.h * self.n0)
        self.s1 = 2. * M_PI / (self.h * self.n1)
        self.s2 = 2. * M_PI / (self.h * self.n2)

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    def convolve(self, tau, eta=None):
        cdef double[:, :, :, :] tau_as_mv = tau
        check_shape_4d(tau_as_mv,
                       self.transform.rshape0,
                       self.transform.rshape1,
                       self.transform.rshape2,
                       self.ncols)
        eta = create_or_check_shape_4d(eta,
                                       self.transform.rshape0,
                                       self.transform.rshape1,
                                       self.transform.rshape2,
                                       self.nrows)

        cdef double[:, :, :, :] dft_tau = array(self.dft_tau_shape,
                                                sizeof(double), 'd')
        cdef double[:, :, :, :] dft_eta = array(self.dft_eta_shape,
                                                sizeof(double), 'd')

        cdef int i

        # Compute DFT of tau
        for i in range(self.ncols):
            self.transform.r2c(tau_as_mv[:, :, :, i], dft_tau[:, :, :, i])

        # Apply Green operator frequency-wise
        cdef int n0 = dft_tau.shape[0]
        cdef int n1 = dft_tau.shape[1]
        cdef int n2 = dft_tau.shape[2] / 2
        cdef int i0, i2, b0, b1, b2
        cdef double k[3]

        for i0 in range(n0):
            b0 = i0 + self.transform.offset0
            if 2 * b0 > self.n0:
                k[0] = self.s0 * (b0 - self.n0)
            else:
                k[0] = self.s0 * b0

            for b1 in range(n1):
                if 2 * b1 > self.n1:
                    k[1] = self.s1 * (b1 - self.n1)
                else:
                    k[1] = self.s1 * b1

                i2 = 0
                for b2 in range(n2):
                    # At this point, i2 = 2 * b2
                    if i2 > self.n2:
                        k[2] = self.s2 * (b2 - self.n2)
                    else:
                        k[2] = self.s2 * b2

                    # Apply Green operator to real part
                    self.green.c_apply(k,
                                       dft_tau[i0, b1, i2, :],
                                       dft_eta[i0, b1, i2, :])
                    i2 += 1
                    # Apply Green operator to imaginary part
                    self.green.c_apply(k,
                                       dft_tau[i0, b1, i2, :],
                                       dft_eta[i0, b1, i2, :])
                    i2 += 1

        # Compute inverse DFT of eta
        for i in range(self.nrows):
            self.transform.c2r(dft_eta[:, :, :, i], eta[:, :, :, i])

        return eta
