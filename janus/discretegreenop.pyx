from cython cimport boundscheck
from cython cimport cdivision
from cython cimport sizeof
from cython cimport wraparound
from cython.view cimport array
from libc.math cimport M_PI
from libc.math cimport cos
from libc.stdlib cimport malloc
from libc.stdlib cimport free

from janus.fft.serial._serial_fft cimport _RealFFT2D
from janus.fft.serial._serial_fft cimport _RealFFT3D
from janus.greenop cimport AbstractGreenOperator
from janus.operators cimport AbstractStructuredOperator2D
from janus.operators cimport AbstractStructuredOperator3D
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

cdef class DiscreteGreenOperator2D(AbstractStructuredOperator2D):
    cdef readonly AbstractGreenOperator green
    cdef readonly double h
    cdef int global_shape0
    cdef int offset0
    # s[i] = 2 * pi / (h * n[i]),
    # where n[i] is the size of the *global* grid in the direction i.
    cdef double s0, s1
    cdef _RealFFT2D transform
    cdef tuple dft_tau_shape, dft_eta_shape

    def __cinit__(self, AbstractGreenOperator green, shape, double h,
                  _RealFFT2D transform=None):
        if len(shape) != 2:
            raise ValueError('length of shape must be 2 (was {0})'
                             .format(len(shape)))
        if green.mat.dim != 2:
            raise ValueError('continuous green operator must operate in '
                             'a 2D space')
        if h <= 0.:
            raise ValueError('h must be > 0 (was {0})'.format(h))

        self.green = green
        self.ishape2 = green.isize
        self.oshape2 = green.osize
        self.h = h

        if transform is not None:
            if transform.shape != shape:
                raise ValueError('shape of transform must be {0} [was {1}]'
                                 .format(shape, transform.shape))
            self.dft_tau_shape = (transform.cshape0, transform.cshape1,
                                  self.ishape2)
            self.dft_eta_shape = (transform.cshape0, transform.cshape1,
                                  self.oshape2)
            self.global_shape0 = transform.shape[0]
            self.offset0 = transform.offset0
            self.ishape0 = transform.rshape0
            self.ishape1 = transform.rshape1
            self.oshape0 = self.ishape0
            self.oshape1 = self.ishape1
        else:
            self.global_shape0 = shape[0]
            self.offset0 = 0
            self.ishape0 = shape[0]
            self.ishape1 = shape[1]
            self.oshape0 = self.ishape0
            self.oshape1 = self.ishape1
            if self.ishape0 < 0:
                raise ValueError('shape[0] must be > 0 (was {0})'
                                 .format(self.ishape0))
            if self.ishape1 < 0:
                raise ValueError('shape[1] must be > 0 (was {0})'
                                 .format(self.ishape1))

        self.transform = transform
        self.ishape = (self.ishape0, self.ishape1, self.ishape2)
        self.oshape = (self.oshape0, self.oshape1, self.oshape2)
        self.s0 = 2. * M_PI / (self.h * self.global_shape0)
        self.s1 = 2. * M_PI / (self.h * self.ishape1)

    cdef void c_set_frequency(self, int[:] b):
        raise NotImplementedError

    def set_frequency(self, int[:] b):
        if b.shape[0] != 2:
            raise ValueError('invalid shape: expected (2,), actual ({0},)'
                             .format(b.shape[0]))

        cdef int b0 = b[0]
        if (b0 < self.offset0) or (b0 >= self.offset0 + self.ishape0):
            raise ValueError('index must be >= {0} and < {1} (was {2})'
                             .format(self.offset0,
                                     self.offset0 + self.ishape0,
                                     b0))

        cdef int b1 = b[1]
        if (b1 < 0) or (b1 >= self.ishape1):
            raise ValueError('index must be >= 0 and < {0} (was {1})'
                             .format(self.ishape1, b1))

        self.c_set_frequency(b)

    cdef void c_to_memoryview(self, double[:, :] out):
        raise NotImplementedError

    def to_memoryview(self, double[:, :] out=None):
        out = create_or_check_shape_2d(out, self.oshape2, self.ishape2)
        self.c_to_memoryview(out)
        return out

    cdef void c_apply_by_freq(self, double[:] tau, double[:] eta):
        raise NotImplementedError

    def apply_by_freq(self, double[:] tau, double[:] eta=None):
        check_shape_1d(tau, self.ishape2)
        eta = create_or_check_shape_1d(eta, self.oshape2)
        self.c_apply_by_freq(tau, eta)
        return eta

cdef class DiscreteGreenOperator3D(AbstractStructuredOperator3D):

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
    osize: int
        The number of rows of the Green tensor, for each frequency (dimension
        of the space of polarizations).
    isize: int
        The number of columns of the Green tensor, for each frequency (dimension
        of the space of strains).
    """
    cdef readonly AbstractGreenOperator green
    cdef readonly double h
    cdef int global_shape0
    cdef int offset0
    # s[i] = 2 * pi / (h * n[i]),
    # where n[i] is the size of the grid in the direction i.
    cdef double s0, s1, s2
    cdef _RealFFT3D transform
    cdef tuple dft_tau_shape, dft_eta_shape

    def __cinit__(self, AbstractGreenOperator green, shape, double h,
                  _RealFFT3D transform=None):
        if len(shape) != 3:
            raise ValueError('length of shape must be 3 (was {0})'
                             .format(len(shape)))
        if green.mat.dim != 3:
            raise ValueError('continuous green operator must operate in '
                             'a 3D space')
        if h <= 0.:
            raise ValueError('h must be > 0 (was {0})'.format(h))

        self.green = green
        self.ishape3 = green.isize
        self.oshape3 = green.osize
        self.h = h


        if transform is not None:
            if transform.shape != shape:
                raise ValueError('shape of transform must be {0} [was {1}]'
                                 .format(shape, transform.shape))
            self.dft_tau_shape = (transform.cshape0, transform.cshape1,
                                  transform.cshape2, self.ishape3)
            self.dft_eta_shape = (transform.cshape0, transform.cshape1,
                                  transform.cshape2, self.oshape3)
            self.global_shape0 = transform.shape[0]
            self.offset0 = transform.offset0
            self.ishape0 = transform.rshape0
            self.ishape1 = transform.rshape1
            self.ishape2 = transform.rshape2
            self.oshape0 = self.ishape0
            self.oshape1 = self.ishape1
            self.oshape2 = self.ishape2
        else:
            self.global_shape0 = shape[0]
            self.offset0 = 0
            self.ishape0 = shape[0]
            self.ishape1 = shape[1]
            self.ishape2 = shape[2]
            self.oshape0 = self.ishape0
            self.oshape1 = self.ishape1
            self.oshape2 = self.ishape2
            if self.ishape0 < 0:
                raise ValueError('shape[0] must be > 0 (was {0})'
                                 .format(self.ishape0))
            if self.ishape1 < 0:
                raise ValueError('shape[1] must be > 0 (was {0})'
                                 .format(self.ishape1))
            if self.ishape2 < 0:
                raise ValueError('shape[2] must be > 0 (was {0})'
                                 .format(self.ishape2))

        self.transform = transform
        self.ishape = (self.ishape0, self.ishape1, self.ishape2, self.ishape3)
        self.oshape = (self.oshape0, self.oshape1, self.oshape2, self.oshape3)
        self.s0 = 2. * M_PI / (self.h * self.global_shape0)
        self.s1 = 2. * M_PI / (self.h * self.ishape1)
        self.s2 = 2. * M_PI / (self.h * self.ishape2)

    cdef void c_set_frequency(self, int[:] b):
        raise NotImplementedError

    def set_frequency(self, int[:] b):
        if b.shape[0] != 3:
            raise ValueError('invalid shape: expected (3,), actual ({0},)'
                             .format(b.shape[0]))

        cdef int b0 = b[0]
        if (b0 < self.offset0) or (b0 >= self.offset0 + self.ishape0):
            raise ValueError('index must be >= {0} and < {1} (was {2})'
                             .format(self.offset0,
                                     self.offset0 + self.ishape0,
                                     b0))

        cdef int b1 = b[1]
        if (b1 < 0) or (b1 >= self.ishape1):
            raise ValueError('index must be >= 0 and < {0} (was {1})'
                             .format(self.ishape1, b1))

        cdef int b2 = b[2]
        if (b2 < 0) or (b2 >= self.ishape2):
            raise ValueError('index must be >= 0 and < {0} (was {1})'
                             .format(self.ishape2, b2))

        self.c_set_frequency(b)

    cdef void c_to_memoryview(self, double[:, :] out):
        raise NotImplementedError

    def to_memoryview(self, double[:, :] out=None):
        out = create_or_check_shape_2d(out, self.oshape3, self.ishape3)
        self.c_to_memoryview(out)
        return out

    cdef void c_apply_by_freq(self, double[:] tau, double[:] eta):
        raise NotImplementedError

    def apply_by_freq(self, double[:] tau, double[:] eta=None):
        check_shape_1d(tau, self.ishape3)
        eta = create_or_check_shape_1d(eta, self.oshape3)
        self.c_apply_by_freq(tau, eta)
        return eta


cdef class TruncatedGreenOperator2D(DiscreteGreenOperator2D):

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
    osize: int
        The number of rows of the Green tensor, for each frequency (dimension
        of the space of polarizations).
    isize: int
        The number of columns of the Green tensor, for each frequency (dimension
        of the space of strains).
    """

    cdef double[:] k

    def __cinit__(self, AbstractGreenOperator green, shape, double h,
                  transform=None):
        self.k = array(shape=(2,), itemsize=sizeof(double), format='d')

    @boundscheck(False)
    @wraparound(False)
    cdef void c_set_frequency(self, int[:] b):
        cdef b0 = b[0]
        if 2 * b0 > self.global_shape0:
            self.k[0] = self.s0 * (b0 - self.global_shape0)
        else:
            self.k[0] = self.s0 * b0
        cdef b1 = b[1]
        if 2 * b1 > self.ishape1:
            self.k[1] = self.s1 * (b1 - self.ishape1)
        else:
            self.k[1] = self.s1 * b1
        self.green.set_frequency(self.k)

    cdef void c_to_memoryview(self, double[:, :] out):
        self.green.c_to_memoryview(out)

    cdef void c_apply_by_freq(self, double[:] tau, double[:] eta):
        self.green.c_apply(tau, eta)

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    cdef void c_apply(self, double[:, :, :] x, double[:, :, :] y):
        cdef double[:, :, :] dft_x = array(self.dft_tau_shape,
                                           sizeof(double), 'd')
        cdef double[:, :, :] dft_y = array(self.dft_eta_shape,
                                           sizeof(double), 'd')
        cdef int i

        # Compute DFT of x
        for i in range(self.ishape2):
            self.transform.r2c(x[:, :, i], dft_x[:, :, i])

        # Apply Green operator frequency-wise
        cdef int n0 = dft_x.shape[0]
        cdef int n1 = dft_x.shape[1] / 2
        cdef int i0, i1, b0, b1

        for i0 in range(n0):
            b0 = i0 + self.offset0
            if 2 * b0 > self.global_shape0:
                self.k[0] = self.s0 * (b0 - self.global_shape0)
            else:
                self.k[0] = self.s0 * b0

            i1 = 0
            for b1 in range(n1):
                # At this point, i1 = 2 * b1
                if i1 > self.ishape1:
                    self.k[1] = self.s1 * (b1 - self.ishape1)
                else:
                    self.k[1] = self.s1 * b1

                # Apply Green operator to real part
                self.green.c_set_frequency(self.k)
                self.green.c_apply(dft_x[i0, i1, :],
                                   dft_y[i0, i1, :])
                i1 += 1
                # Apply Green operator to imaginary part
                self.green.c_apply(dft_x[i0, i1, :],
                                   dft_y[i0, i1, :])
                i1 += 1

        # Compute inverse DFT of y
        for i in range(self.oshape2):
            self.transform.c2r(dft_y[:, :, i], y[:, :, i])


cdef class TruncatedGreenOperator3D(DiscreteGreenOperator3D):
    cdef double[:] k

    def __cinit__(self, AbstractGreenOperator green, shape, double h,
                  transform=None):
        self.k = array(shape=(3,), itemsize=sizeof(double), format='d')

    @boundscheck(False)
    @wraparound(False)
    cdef void c_set_frequency(self, int[:] b):
        cdef b0 = b[0]
        if 2 * b0 > self.global_shape0:
            self.k[0] = self.s0 * (b0 - self.global_shape0)
        else:
            self.k[0] = self.s0 * b0
        cdef b1 = b[1]
        if 2 * b1 > self.ishape1:
            self.k[1] = self.s1 * (b1 - self.ishape1)
        else:
            self.k[1] = self.s1 * b1
        cdef b2 = b[2]
        if 2 * b2 > self.ishape2:
            self.k[2] = self.s2 * (b2 - self.ishape2)
        else:
            self.k[2] = self.s2 * b2
        self.green.set_frequency(self.k)

    cdef void c_to_memoryview(self, double[:, :] out):
        self.green.c_to_memoryview(out)

    cdef void c_apply_by_freq(self, double[:] tau, double[:] eta):
        self.green.c_apply(tau, eta)

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    cdef void c_apply(self, double[:, :, :, :] x, double[:, :, :, :] y):
        cdef double[:, :, :, :] dft_x = array(self.dft_tau_shape,
                                              sizeof(double), 'd')
        cdef double[:, :, :, :] dft_y = array(self.dft_eta_shape,
                                              sizeof(double), 'd')
        cdef int i

        # Compute DFT of x
        for i in range(self.ishape3):
            self.transform.r2c(x[:, :, :, i], dft_x[:, :, :, i])

        # Apply Green operator frequency-wise
        cdef int n0 = dft_x.shape[0]
        cdef int n1 = dft_x.shape[1]
        cdef int n2 = dft_x.shape[2] / 2
        cdef int i0, i2, b0, b1, b2

        for i0 in range(n0):
            b0 = i0 + self.offset0
            if 2 * b0 > self.global_shape0:
                self.k[0] = self.s0 * (b0 - self.global_shape0)
            else:
                self.k[0] = self.s0 * b0

            for b1 in range(n1):
                if 2 * b1 > self.ishape1:
                    self.k[1] = self.s1 * (b1 - self.ishape1)
                else:
                    self.k[1] = self.s1 * b1

                i2 = 0
                for b2 in range(n2):
                    # At this point, i2 = 2 * b2
                    if i2 > self.ishape2:
                        self.k[2] = self.s2 * (b2 - self.ishape2)
                    else:
                        self.k[2] = self.s2 * b2

                    # Apply Green operator to real part
                    self.green.set_frequency(self.k)
                    self.green.c_apply(dft_x[i0, b1, i2, :],
                                       dft_y[i0, b1, i2, :])
                    i2 += 1
                    # Apply Green operator to imaginary part
                    self.green.c_apply(dft_x[i0, b1, i2, :],
                                       dft_y[i0, b1, i2, :])
                    i2 += 1

        # Compute inverse DFT of y
        for i in range(self.oshape3):
            self.transform.c2r(dft_y[:, :, :, i], y[:, :, :, i])


cdef class FilteredGreenOperator2D(DiscreteGreenOperator2D):
    cdef double g00, g01, g02
    cdef double g11, g12
    cdef double g22

    # Cached arrays to store the four terms of the weighted sum defining the
    # filtered Green operator.
    cdef double[:] k1, k2, k3, k4
    cdef double[:, :] g1, g2, g3, g4

    def __cinit__(self, AbstractGreenOperator green, shape, double h,
                  transform=None):
        shape = (2,)
        self.k1 = array(shape, sizeof(double), 'd')
        self.k2 = array(shape, sizeof(double), 'd')
        self.k3 = array(shape, sizeof(double), 'd')
        self.k4 = array(shape, sizeof(double), 'd')

        shape = (green.osize, green.isize)
        self.g1 = array(shape, sizeof(double), 'd')
        self.g2 = array(shape, sizeof(double), 'd')
        self.g3 = array(shape, sizeof(double), 'd')
        self.g4 = array(shape, sizeof(double), 'd')

    @boundscheck(False)
    @wraparound(False)
    cdef void c_set_frequency(self, int[:] b):
        cdef int b0 = b[0]
        cdef int b1 = b[1]
        cdef double k, w, w1, w2, w3, w4

        # Computation of the first component of k1, k2, k3, k4 and the first
        # factor of the corresponding weights.
        k = self.s0 * (b0 - self.global_shape0)
        w = cos(0.25 * self.h * k)
        w *= w
        self.k1[0] = k
        self.k2[0] = k
        w1 = w
        w2 = w

        k = self.s0 * b0
        w = cos(0.25 * self.h * k)
        w *= w
        self.k3[0] = k
        self.k4[0] = k
        w3 = w
        w4 = w

        # Computation of the second component of k1, k2, k3, k4 and the second
        # factor of the corresponding weights.
        k = self.s1 * (b1 - self.ishape1)
        w = cos(0.25 * self.h * k)
        w *= w
        self.k1[1] = k
        self.k3[1] = k
        w1 *= w
        w3 *= w

        k = self.s1 * b1
        w = cos(0.25 * self.h * k)
        w *= w
        self.k2[1] = k
        self.k4[1] = k
        w2 *= w
        w4 *= w

        self.green.c_set_frequency(self.k1)
        self.green.c_to_memoryview(self.g1)
        self.green.c_set_frequency(self.k2)
        self.green.c_to_memoryview(self.g2)
        self.green.c_set_frequency(self.k3)
        self.green.c_to_memoryview(self.g3)
        self.green.c_set_frequency(self.k4)
        self.green.c_to_memoryview(self.g4)

        self.g00 = (w1 * self.g1[0, 0] + w2 * self.g2[0, 0]
                    + w3 * self.g3[0, 0] + w4 * self.g4[0, 0])
        self.g01 = (w1 * self.g1[0, 1] + w2 * self.g2[0, 1]
                    + w3 * self.g3[0, 1] + w4 * self.g4[0, 1])
        self.g02 = (w1 * self.g1[0, 2] + w2 * self.g2[0, 2]
                    + w3 * self.g3[0, 2] + w4 * self.g4[0, 2])
        self.g11 = (w1 * self.g1[1, 1] + w2 * self.g2[1, 1]
                    + w3 * self.g3[1, 1] + w4 * self.g4[1, 1])
        self.g12 = (w1 * self.g1[1, 2] + w2 * self.g2[1, 2]
                    + w3 * self.g3[1, 2] + w4 * self.g4[1, 2])
        self.g22 = (w1 * self.g1[2, 2] + w2 * self.g2[2, 2]
                    + w3 * self.g3[2, 2] + w4 * self.g4[2, 2])

    @boundscheck(False)
    @wraparound(False)
    cdef void c_to_memoryview(self, double[:, :] out):
        out[0, 0] = self.g00
        out[0, 1] = self.g01
        out[0, 2] = self.g02
        out[1, 0] = self.g01
        out[1, 1] = self.g11
        out[1, 2] = self.g12
        out[2, 0] = self.g02
        out[2, 1] = self.g12
        out[2, 2] = self.g22

    @boundscheck(False)
    @wraparound(False)
    cdef void c_apply_by_freq(self, double[:] x, double[:] y):
        cdef double x0, x1, x2, y0, y1, y2
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        y[0] = self.g00 * x0 + self.g01 * x1 + self.g02 * x2
        y[1] = self.g01 * x0 + self.g11 * x1 + self.g12 * x2
        y[2] = self.g02 * x0 + self.g12 * x1 + self.g22 * x2

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    cdef void c_apply(self, double[:, :, :] x, double[:, :, :] y):
        cdef int[:] b = array((2,), sizeof(int), 'i')
        cdef double[:, :, :] dft_x = array(self.dft_tau_shape,
                                           sizeof(double), 'd')
        cdef double[:, :, :] dft_y = array(self.dft_eta_shape,
                                           sizeof(double), 'd')
        cdef int i

        # Compute DFT of x
        for i in range(self.ishape2):
            self.transform.r2c(x[:, :, i], dft_x[:, :, i])

        # Apply Green operator frequency-wise
        cdef int n0 = dft_x.shape[0]
        cdef int n1 = dft_x.shape[1] / 2
        cdef int i0, i1, b0, b1

        for i0 in range(n0):
            b[0] = i0 + self.offset0
            i1 = 0
            for b1 in range(n1):
                b[1] = b1
                self.c_set_frequency(b)

                # Apply Green operator to real part
                self.c_apply_by_freq(dft_x[i0, i1, :], dft_y[i0, i1, :])
                i1 += 1
                # Apply Green operator to imaginary part
                self.c_apply_by_freq(dft_x[i0, i1, :], dft_y[i0, i1, :])
                i1 += 1

        # Compute inverse DFT of y
        for i in range(self.oshape2):
            self.transform.c2r(dft_y[:, :, i], y[:, :, i])
