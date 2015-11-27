from cython cimport boundscheck
from cython cimport cdivision
from cython cimport sizeof
from cython cimport wraparound
from cython.view cimport array
from libc.math cimport M_PI
from libc.math cimport cos, sin

from janus.fft.serial._serial_fft cimport _RealFFT2D
from janus.fft.serial._serial_fft cimport _RealFFT3D
from janus.operators cimport AbstractLinearOperator
from janus.operators cimport AbstractStructuredOperator2D
from janus.operators cimport AbstractStructuredOperator3D
from janus.utils.checkarray cimport create_or_check_shape_1d
from janus.utils.checkarray cimport create_or_check_shape_2d
from janus.utils.checkarray cimport create_or_check_shape_3d
from janus.utils.checkarray cimport create_or_check_shape_4d
from janus.utils.checkarray cimport check_shape_1d
from janus.utils.checkarray cimport check_shape_3d
from janus.utils.checkarray cimport check_shape_4d

def truncated(green, n, h, transform=None):
    if green.dim == 2:
        return TruncatedGreenOperator2D(green, n, h, transform)
    elif green.dim == 3:
        return TruncatedGreenOperator3D(green, n, h, transform)
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(green.dim))

def filtered(green, n, h, transform=None):
    if green.dim == 2:
        return FilteredGreenOperator2D(green, n, h, transform)
    elif green.dim == 3:
        return FilteredGreenOperator3D(green, n, h, transform)
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(green.dim))


cdef class AbstractGreenOperator(AbstractLinearOperator):

    """This class defines Abstract periodic Green operators.

    Green operators are defined as linear operators, which are
    supplemented with a `set_frequency()` method.

    The __repr__() method assumes that the attribute mat stores the
    underlying material.

    Attributes:
        dim: the dimension of the physical space

    """

    cdef void c_set_frequency(self, double[:] k):
        """Set the current wave-vector of this Green operator.

        Any subsequent call to e.g. c_apply(), c_to_memoryview() are
        performed with the specified value of k.

        Concrete implementations of this method are not required to
        perform any test on the validity (size) of k.

        Args:
            k: the wave-vector (memoryview of float64)

        """
        raise NotImplementedError

    def set_frequency(self, double[:] k):
        """Set the current wave-vector of this Green operator.

        Any subsequent call to e.g. apply(), to_memoryview() are
        performed with the specified value of k.

        Concrete implementations of this method are not required to
        perform any test on the validity (size) of k.

        Args:
            k: the wave-vector (memoryview of float64)

        """
        check_shape_1d(k, self.dim)
        self.c_set_frequency(k)

    def __repr__(self):
        return 'Green Operator({0})'.format(self.mat)


cdef class DiscreteGreenOperator2D(AbstractStructuredOperator2D):
    cdef readonly AbstractGreenOperator green
    cdef readonly double h
    cdef int global_shape0
    cdef int offset0
    # s[i] = 2 * pi / (h * n[i]),
    # where n[i] is the size of the *global* grid in the direction i.
    cdef double s0, s1
    cdef readonly _RealFFT2D transform
    cdef double[:, :, :] dft_x, dft_y

    def __cinit__(self, AbstractGreenOperator green, shape, double h,
                  _RealFFT2D transform=None):
        if len(shape) != 2:
            raise ValueError('length of shape must be 2 (was {0})'
                             .format(len(shape)))
        if green.dim != 2:
            raise ValueError('continuous green operator must operate in '
                             'a 2D space')
        if h <= 0.:
            raise ValueError('h must be > 0 (was {0})'.format(h))

        self.green = green
        self.h = h

        cdef int shape0, shape1
        if transform is not None:
            if transform.global_ishape != shape:
                raise ValueError('shape of transform must be {0} [was {1}]'
                                 .format(shape, transform.global_ishape))
            self.dft_x = array((transform.oshape0, transform.oshape1,
                                green.isize), sizeof(double), 'd')
            if green.osize == green.isize:
                self.dft_y = self.dft_x
            else:
                self.dft_y = array((transform.oshape0, transform.oshape1,
                                    green.osize), sizeof(double), 'd')
            self.global_shape0 = transform.global_ishape[0]
            self.offset0 = transform.offset0
            shape0 = transform.ishape0
            shape1 = transform.ishape1
        else:
            self.dft_x = None
            self.dft_y = None
            self.global_shape0 = shape[0]
            self.offset0 = 0
            shape0 = shape[0]
            shape1 = shape[1]

        self.transform = transform
        self.init_shapes(shape0, shape1, green.isize, green.osize)
        self.s0 = 2. * M_PI / (self.h * self.global_shape0)
        self.s1 = 2. * M_PI / (self.h * self.shape1)

    cdef void c_set_frequency(self, int[:] b):
        raise NotImplementedError

    def set_frequency(self, int[:] b):
        if b.shape[0] != 2:
            raise ValueError('invalid shape: expected (2,), actual ({0},)'
                             .format(b.shape[0]))

        cdef int b0 = b[0]
        if (b0 < self.offset0) or (b0 >= self.offset0 + self.shape0):
            raise ValueError('index must be >= {0} and < {1} (was {2})'
                             .format(self.offset0,
                                     self.offset0 + self.shape0,
                                     b0))

        cdef int b1 = b[1]
        if (b1 < 0) or (b1 >= self.shape1):
            raise ValueError('index must be >= 0 and < {0} (was {1})'
                             .format(self.shape1, b1))

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

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    cdef void c_apply(self, double[:, :, :] x, double[:, :, :] y):
        cdef int[:] b = array((2,), sizeof(int), 'i')
        cdef int i

        # Compute DFT of x
        for i in range(self.ishape2):
            self.transform.r2c(x[:, :, i], self.dft_x[:, :, i])

        # Apply Green operator frequency-wise
        cdef int n0 = self.dft_x.shape[0]
        cdef int n1 = self.dft_x.shape[1] / 2
        cdef int i0, i1, b0, b1

        for i0 in range(n0):
            b[0] = i0 + self.offset0
            i1 = 0
            for b1 in range(n1):
                b[1] = b1
                self.c_set_frequency(b)

                # Apply Green operator to real part
                self.c_apply_by_freq(self.dft_x[i0, i1, :],
                                     self.dft_y[i0, i1, :])
                i1 += 1
                # Apply Green operator to imaginary part
                self.c_apply_by_freq(self.dft_x[i0, i1, :],
                                     self.dft_y[i0, i1, :])
                i1 += 1

        # Compute inverse DFT of y
        for i in range(self.oshape2):
            self.transform.c2r(self.dft_y[:, :, i], y[:, :, i])


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
    cdef readonly _RealFFT3D transform
    cdef double[:, :, :, :] dft_x, dft_y

    def __cinit__(self, AbstractGreenOperator green, shape, double h,
                  _RealFFT3D transform=None):
        if len(shape) != 3:
            raise ValueError('length of shape must be 3 (was {0})'
                             .format(len(shape)))
        if green.dim != 3:
            raise ValueError('continuous green operator must operate in '
                             'a 3D space')
        if h <= 0.:
            raise ValueError('h must be > 0 (was {0})'.format(h))

        self.green = green
        self.h = h

        cdef int shape0, shape1, shape2
        if transform is not None:
            if transform.global_ishape != shape:
                raise ValueError('shape of transform must be {0} [was {1}]'
                                 .format(shape, transform.global_ishape))
            self.dft_x = array((transform.oshape0, transform.oshape1,
                                transform.oshape2, green.isize),
                               sizeof(double), 'd')
            if green.osize == green.isize:
                self.dft_y = self.dft_x
            else:
                self.dft_y = array((transform.oshape0, transform.oshape1,
                                    transform.oshape2, green.osize),
                                   sizeof(double), 'd')
            self.global_shape0 = transform.global_ishape[0]
            self.offset0 = transform.offset0
            shape0 = transform.ishape0
            shape1 = transform.ishape1
            shape2 = transform.ishape2
        else:
            self.dft_x = None
            self.dft_y = None
            self.global_shape0 = shape[0]
            self.offset0 = 0
            shape0 = shape[0]
            shape1 = shape[1]
            shape2 = shape[2]

        self.transform = transform
        self.init_shapes(shape0, shape1, shape2, green.isize, green.osize)
        self.s0 = 2. * M_PI / (self.h * self.global_shape0)
        self.s1 = 2. * M_PI / (self.h * self.shape1)
        self.s2 = 2. * M_PI / (self.h * self.shape2)

    cdef void c_set_frequency(self, int[:] b):
        raise NotImplementedError

    def set_frequency(self, int[:] b):
        if b.shape[0] != 3:
            raise ValueError('invalid shape: expected (3,), actual ({0},)'
                             .format(b.shape[0]))

        cdef int b0 = b[0]
        if (b0 < self.offset0) or (b0 >= self.offset0 + self.shape0):
            raise ValueError('index must be >= {0} and < {1} (was {2})'
                             .format(self.offset0,
                                     self.offset0 + self.shape0,
                                     b0))

        cdef int b1 = b[1]
        if (b1 < 0) or (b1 >= self.shape1):
            raise ValueError('index must be >= 0 and < {0} (was {1})'
                             .format(self.shape1, b1))

        cdef int b2 = b[2]
        if (b2 < 0) or (b2 >= self.shape2):
            raise ValueError('index must be >= 0 and < {0} (was {1})'
                             .format(self.shape2, b2))

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

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    cdef void c_apply(self, double[:, :, :, :] x, double[:, :, :, :] y):
        cdef int[:] b = array((3,), sizeof(int), 'i')
        cdef int i

        # Compute DFT of x
        for i in range(self.ishape3):
            self.transform.r2c(x[:, :, :, i], self.dft_x[:, :, :, i])

        # Apply Green operator frequency-wise
        cdef int n0 = self.dft_x.shape[0]
        cdef int n1 = self.dft_x.shape[1]
        cdef int n2 = self.dft_x.shape[2] / 2
        cdef int i0, i1, i2, b2

        for i0 in range(n0):
            b[0] = i0 + self.offset0
            for i1 in range(n1):
                b[1] = i1
                i2 = 0
                for b2 in range(n2):
                    b[2] = b2
                    self.c_set_frequency(b)

                    # Apply Green operator to real part
                    self.c_apply_by_freq(self.dft_x[i0, i1, i2, :],
                                         self.dft_y[i0, i1, i2, :])
                    i2 += 1
                    # Apply Green operator to imaginary part
                    self.c_apply_by_freq(self.dft_x[i0, i1, i2, :],
                                         self.dft_y[i0, i1, i2, :])
                    i2 += 1

        # Compute inverse DFT of y
        for i in range(self.oshape3):
            self.transform.c2r(self.dft_y[:, :, :, i], y[:, :, :, i])


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
        cdef int b0 = b[0]
        if 2 * b0 > self.global_shape0:
            self.k[0] = self.s0 * (b0 - self.global_shape0)
        else:
            self.k[0] = self.s0 * b0
        cdef int b1 = b[1]
        if 2 * b1 > self.shape1:
            self.k[1] = self.s1 * (b1 - self.shape1)
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
        cdef int i

        # Compute DFT of x
        for i in range(self.ishape2):
            self.transform.r2c(x[:, :, i], self.dft_x[:, :, i])

        # Apply Green operator frequency-wise
        cdef int n0 = self.dft_x.shape[0]
        cdef int n1 = self.dft_x.shape[1] / 2
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
                if i1 > self.shape1:
                    self.k[1] = self.s1 * (b1 - self.shape1)
                else:
                    self.k[1] = self.s1 * b1

                # Apply Green operator to real part
                self.green.c_set_frequency(self.k)
                self.green.c_apply(self.dft_x[i0, i1, :],
                                   self.dft_y[i0, i1, :])
                i1 += 1
                # Apply Green operator to imaginary part
                self.green.c_apply(self.dft_x[i0, i1, :],
                                   self.dft_y[i0, i1, :])
                i1 += 1

        # Compute inverse DFT of y
        for i in range(self.oshape2):
            self.transform.c2r(self.dft_y[:, :, i], y[:, :, i])


cdef class TruncatedGreenOperator3D(DiscreteGreenOperator3D):
    cdef double[:] k

    def __cinit__(self, AbstractGreenOperator green, shape, double h,
                  transform=None):
        self.k = array(shape=(3,), itemsize=sizeof(double), format='d')

    @boundscheck(False)
    @wraparound(False)
    cdef void c_set_frequency(self, int[:] b):
        cdef int b0 = b[0]
        if 2 * b0 > self.global_shape0:
            self.k[0] = self.s0 * (b0 - self.global_shape0)
        else:
            self.k[0] = self.s0 * b0
        cdef int b1 = b[1]
        if 2 * b1 > self.shape1:
            self.k[1] = self.s1 * (b1 - self.shape1)
        else:
            self.k[1] = self.s1 * b1
        cdef int b2 = b[2]
        if 2 * b2 > self.shape2:
            self.k[2] = self.s2 * (b2 - self.shape2)
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
        cdef int i

        # Compute DFT of x
        for i in range(self.ishape3):
            self.transform.r2c(x[:, :, :, i], self.dft_x[:, :, :, i])

        # Apply Green operator frequency-wise
        cdef int n0 = self.dft_x.shape[0]
        cdef int n1 = self.dft_x.shape[1]
        cdef int n2 = self.dft_x.shape[2] / 2
        cdef int i0, i2, b0, b1, b2

        for i0 in range(n0):
            b0 = i0 + self.offset0
            if 2 * b0 > self.global_shape0:
                self.k[0] = self.s0 * (b0 - self.global_shape0)
            else:
                self.k[0] = self.s0 * b0

            for b1 in range(n1):
                if 2 * b1 > self.shape1:
                    self.k[1] = self.s1 * (b1 - self.shape1)
                else:
                    self.k[1] = self.s1 * b1

                i2 = 0
                for b2 in range(n2):
                    # At this point, i2 = 2 * b2
                    if i2 > self.shape2:
                        self.k[2] = self.s2 * (b2 - self.shape2)
                    else:
                        self.k[2] = self.s2 * b2

                    # Apply Green operator to real part
                    self.green.set_frequency(self.k)
                    self.green.c_apply(self.dft_x[i0, b1, i2, :],
                                       self.dft_y[i0, b1, i2, :])
                    i2 += 1
                    # Apply Green operator to imaginary part
                    self.green.c_apply(self.dft_x[i0, b1, i2, :],
                                       self.dft_y[i0, b1, i2, :])
                    i2 += 1

        # Compute inverse DFT of y
        for i in range(self.oshape3):
            self.transform.c2r(self.dft_y[:, :, :, i], y[:, :, :, i])


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
        k = self.s1 * (b1 - self.shape1)
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

        self.g00 = (w1 * self.g1[0, 0] + w2 * self.g2[0, 0] +
                    w3 * self.g3[0, 0] + w4 * self.g4[0, 0])
        self.g01 = (w1 * self.g1[0, 1] + w2 * self.g2[0, 1] +
                    w3 * self.g3[0, 1] + w4 * self.g4[0, 1])
        self.g02 = (w1 * self.g1[0, 2] + w2 * self.g2[0, 2] +
                    w3 * self.g3[0, 2] + w4 * self.g4[0, 2])
        self.g11 = (w1 * self.g1[1, 1] + w2 * self.g2[1, 1] +
                    w3 * self.g3[1, 1] + w4 * self.g4[1, 1])
        self.g12 = (w1 * self.g1[1, 2] + w2 * self.g2[1, 2] +
                    w3 * self.g3[1, 2] + w4 * self.g4[1, 2])
        self.g22 = (w1 * self.g1[2, 2] + w2 * self.g2[2, 2] +
                    w3 * self.g3[2, 2] + w4 * self.g4[2, 2])

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


cdef class FilteredGreenOperator3D(DiscreteGreenOperator3D):
    cdef double g00, g01, g02, g03, g04, g05
    cdef double g11, g12, g13, g14, g15
    cdef double g22, g23, g24, g25
    cdef double g33, g34, g35
    cdef double g44, g45
    cdef double g55

    cdef double[:] k1, k2, k3, k4, k5, k6, k7, k8
    cdef double[:, :] g1, g2, g3, g4, g5, g6, g7, g8

    def __cinit__(self, AbstractGreenOperator green, shape, double h,
                  transform=None):
        shape = (3,)
        self.k1 = array(shape, sizeof(double), 'd')
        self.k2 = array(shape, sizeof(double), 'd')
        self.k3 = array(shape, sizeof(double), 'd')
        self.k4 = array(shape, sizeof(double), 'd')
        self.k5 = array(shape, sizeof(double), 'd')
        self.k6 = array(shape, sizeof(double), 'd')
        self.k7 = array(shape, sizeof(double), 'd')
        self.k8 = array(shape, sizeof(double), 'd')

        shape = (green.osize, green.isize)
        self.g1 = array(shape, sizeof(double), 'd')
        self.g2 = array(shape, sizeof(double), 'd')
        self.g3 = array(shape, sizeof(double), 'd')
        self.g4 = array(shape, sizeof(double), 'd')
        self.g5 = array(shape, sizeof(double), 'd')
        self.g6 = array(shape, sizeof(double), 'd')
        self.g7 = array(shape, sizeof(double), 'd')
        self.g8 = array(shape, sizeof(double), 'd')

    @boundscheck(False)
    @wraparound(False)
    cdef void c_set_frequency(self, int[:] b):
        cdef int b0 = b[0]
        cdef int b1 = b[1]
        cdef int b2 = b[2]
        cdef double k, w, w1, w2, w3, w4, w5, w6, w7, w8

        k = self.s0 * (b0 - self.global_shape0)
        w = cos(0.25 * self.h * k)
        w *= w
        self.k1[0] = k
        self.k2[0] = k
        self.k3[0] = k
        self.k4[0] = k
        w1 = w
        w2 = w
        w3 = w
        w4 = w

        k = self.s0 * b0
        w = cos(0.25 * self.h * k)
        w *= w
        self.k5[0] = k
        self.k6[0] = k
        self.k7[0] = k
        self.k8[0] = k
        w5 = w
        w6 = w
        w7 = w
        w8 = w

        k = self.s1 * (b1 - self.shape1)
        w = cos(0.25 * self.h * k)
        w *= w
        self.k1[1] = k
        self.k2[1] = k
        self.k5[1] = k
        self.k6[1] = k
        w1 *= w
        w2 *= w
        w5 *= w
        w6 *= w

        k = self.s1 * b1
        w = cos(0.25 * self.h * k)
        w *= w
        self.k3[1] = k
        self.k4[1] = k
        self.k7[1] = k
        self.k8[1] = k
        w3 *= w
        w4 *= w
        w7 *= w
        w8 *= w

        k = self.s2 * (b2 - self.shape2)
        w = cos(0.25 * self.h * k)
        w *= w
        self.k1[2] = k
        self.k3[2] = k
        self.k5[2] = k
        self.k7[2] = k
        w1 *= w
        w3 *= w
        w5 *= w
        w7 *= w

        k = self.s2 * b2
        w = cos(0.25 * self.h * k)
        w *= w
        self.k2[2] = k
        self.k4[2] = k
        self.k6[2] = k
        self.k8[2] = k
        w2 *= w
        w4 *= w
        w6 *= w
        w8 *= w

        self.green.c_set_frequency(self.k1)
        self.green.c_to_memoryview(self.g1)
        self.green.c_set_frequency(self.k2)
        self.green.c_to_memoryview(self.g2)
        self.green.c_set_frequency(self.k3)
        self.green.c_to_memoryview(self.g3)
        self.green.c_set_frequency(self.k4)
        self.green.c_to_memoryview(self.g4)
        self.green.c_set_frequency(self.k5)
        self.green.c_to_memoryview(self.g5)
        self.green.c_set_frequency(self.k6)
        self.green.c_to_memoryview(self.g6)
        self.green.c_set_frequency(self.k7)
        self.green.c_to_memoryview(self.g7)
        self.green.c_set_frequency(self.k8)
        self.green.c_to_memoryview(self.g8)

        self.g00 = (w1 * self.g1[0, 0] + w2 * self.g2[0, 0] +
                    w3 * self.g3[0, 0] + w4 * self.g4[0, 0] +
                    w5 * self.g5[0, 0] + w6 * self.g6[0, 0] +
                    w7 * self.g7[0, 0] + w8 * self.g8[0, 0])
        self.g01 = (w1 * self.g1[0, 1] + w2 * self.g2[0, 1] +
                    w3 * self.g3[0, 1] + w4 * self.g4[0, 1] +
                    w5 * self.g5[0, 1] + w6 * self.g6[0, 1] +
                    w7 * self.g7[0, 1] + w8 * self.g8[0, 1])
        self.g02 = (w1 * self.g1[0, 2] + w2 * self.g2[0, 2] +
                    w3 * self.g3[0, 2] + w4 * self.g4[0, 2] +
                    w5 * self.g5[0, 2] + w6 * self.g6[0, 2] +
                    w7 * self.g7[0, 2] + w8 * self.g8[0, 2])
        self.g03 = (w1 * self.g1[0, 3] + w2 * self.g2[0, 3] +
                    w3 * self.g3[0, 3] + w4 * self.g4[0, 3] +
                    w5 * self.g5[0, 3] + w6 * self.g6[0, 3] +
                    w7 * self.g7[0, 3] + w8 * self.g8[0, 3])
        self.g04 = (w1 * self.g1[0, 4] + w2 * self.g2[0, 4] +
                    w3 * self.g3[0, 4] + w4 * self.g4[0, 4] +
                    w5 * self.g5[0, 4] + w6 * self.g6[0, 4] +
                    w7 * self.g7[0, 4] + w8 * self.g8[0, 4])
        self.g05 = (w1 * self.g1[0, 5] + w2 * self.g2[0, 5] +
                    w3 * self.g3[0, 5] + w4 * self.g4[0, 5] +
                    w5 * self.g5[0, 5] + w6 * self.g6[0, 5] +
                    w7 * self.g7[0, 5] + w8 * self.g8[0, 5])
        self.g11 = (w1 * self.g1[1, 1] + w2 * self.g2[1, 1] +
                    w3 * self.g3[1, 1] + w4 * self.g4[1, 1] +
                    w5 * self.g5[1, 1] + w6 * self.g6[1, 1] +
                    w7 * self.g7[1, 1] + w8 * self.g8[1, 1])
        self.g12 = (w1 * self.g1[1, 2] + w2 * self.g2[1, 2] +
                    w3 * self.g3[1, 2] + w4 * self.g4[1, 2] +
                    w5 * self.g5[1, 2] + w6 * self.g6[1, 2] +
                    w7 * self.g7[1, 2] + w8 * self.g8[1, 2])
        self.g13 = (w1 * self.g1[1, 3] + w2 * self.g2[1, 3] +
                    w3 * self.g3[1, 3] + w4 * self.g4[1, 3] +
                    w5 * self.g5[1, 3] + w6 * self.g6[1, 3] +
                    w7 * self.g7[1, 3] + w8 * self.g8[1, 3])
        self.g14 = (w1 * self.g1[1, 4] + w2 * self.g2[1, 4] +
                    w3 * self.g3[1, 4] + w4 * self.g4[1, 4] +
                    w5 * self.g5[1, 4] + w6 * self.g6[1, 4] +
                    w7 * self.g7[1, 4] + w8 * self.g8[1, 4])
        self.g15 = (w1 * self.g1[1, 5] + w2 * self.g2[1, 5] +
                    w3 * self.g3[1, 5] + w4 * self.g4[1, 5] +
                    w5 * self.g5[1, 5] + w6 * self.g6[1, 5] +
                    w7 * self.g7[1, 5] + w8 * self.g8[1, 5])
        self.g22 = (w1 * self.g1[2, 2] + w2 * self.g2[2, 2] +
                    w3 * self.g3[2, 2] + w4 * self.g4[2, 2] +
                    w5 * self.g5[2, 2] + w6 * self.g6[2, 2] +
                    w7 * self.g7[2, 2] + w8 * self.g8[2, 2])
        self.g23 = (w1 * self.g1[2, 3] + w2 * self.g2[2, 3] +
                    w3 * self.g3[2, 3] + w4 * self.g4[2, 3] +
                    w5 * self.g5[2, 3] + w6 * self.g6[2, 3] +
                    w7 * self.g7[2, 3] + w8 * self.g8[2, 3])
        self.g24 = (w1 * self.g1[2, 4] + w2 * self.g2[2, 4] +
                    w3 * self.g3[2, 4] + w4 * self.g4[2, 4] +
                    w5 * self.g5[2, 4] + w6 * self.g6[2, 4] +
                    w7 * self.g7[2, 4] + w8 * self.g8[2, 4])
        self.g25 = (w1 * self.g1[2, 5] + w2 * self.g2[2, 5] +
                    w3 * self.g3[2, 5] + w4 * self.g4[2, 5] +
                    w5 * self.g5[2, 5] + w6 * self.g6[2, 5] +
                    w7 * self.g7[2, 5] + w8 * self.g8[2, 5])
        self.g33 = (w1 * self.g1[3, 3] + w2 * self.g2[3, 3] +
                    w3 * self.g3[3, 3] + w4 * self.g4[3, 3] +
                    w5 * self.g5[3, 3] + w6 * self.g6[3, 3] +
                    w7 * self.g7[3, 3] + w8 * self.g8[3, 3])
        self.g34 = (w1 * self.g1[3, 4] + w2 * self.g2[3, 4] +
                    w3 * self.g3[3, 4] + w4 * self.g4[3, 4] +
                    w5 * self.g5[3, 4] + w6 * self.g6[3, 4] +
                    w7 * self.g7[3, 4] + w8 * self.g8[3, 4])
        self.g35 = (w1 * self.g1[3, 5] + w2 * self.g2[3, 5] +
                    w3 * self.g3[3, 5] + w4 * self.g4[3, 5] +
                    w5 * self.g5[3, 5] + w6 * self.g6[3, 5] +
                    w7 * self.g7[3, 5] + w8 * self.g8[3, 5])
        self.g44 = (w1 * self.g1[4, 4] + w2 * self.g2[4, 4] +
                    w3 * self.g3[4, 4] + w4 * self.g4[4, 4] +
                    w5 * self.g5[4, 4] + w6 * self.g6[4, 4] +
                    w7 * self.g7[4, 4] + w8 * self.g8[4, 4])
        self.g45 = (w1 * self.g1[4, 5] + w2 * self.g2[4, 5] +
                    w3 * self.g3[4, 5] + w4 * self.g4[4, 5] +
                    w5 * self.g5[4, 5] + w6 * self.g6[4, 5] +
                    w7 * self.g7[4, 5] + w8 * self.g8[4, 5])
        self.g55 = (w1 * self.g1[5, 5] + w2 * self.g2[5, 5] +
                    w3 * self.g3[5, 5] + w4 * self.g4[5, 5] +
                    w5 * self.g5[5, 5] + w6 * self.g6[5, 5] +
                    w7 * self.g7[5, 5] + w8 * self.g8[5, 5])

    @boundscheck(False)
    @wraparound(False)
    cdef void c_to_memoryview(self, double[:, :] out):
        out[0][0] = self.g00
        out[0][1] = self.g01
        out[0][2] = self.g02
        out[0][3] = self.g03
        out[0][4] = self.g04
        out[0][5] = self.g05
        out[1][0] = self.g01
        out[1][1] = self.g11
        out[1][2] = self.g12
        out[1][3] = self.g13
        out[1][4] = self.g14
        out[1][5] = self.g15
        out[2][0] = self.g02
        out[2][1] = self.g12
        out[2][2] = self.g22
        out[2][3] = self.g23
        out[2][4] = self.g24
        out[2][5] = self.g25
        out[3][0] = self.g03
        out[3][1] = self.g13
        out[3][2] = self.g23
        out[3][3] = self.g33
        out[3][4] = self.g34
        out[3][5] = self.g35
        out[4][0] = self.g04
        out[4][1] = self.g14
        out[4][2] = self.g24
        out[4][3] = self.g34
        out[4][4] = self.g44
        out[4][5] = self.g45
        out[5][0] = self.g05
        out[5][1] = self.g15
        out[5][2] = self.g25
        out[5][3] = self.g35
        out[5][4] = self.g45
        out[5][5] = self.g55

    @boundscheck(False)
    @wraparound(False)
    cdef void c_apply_by_freq(self, double[:] x, double[:] y):
        cdef double x0, x1, x2, x3, x4, x5
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        x3 = x[3]
        x4 = x[4]
        x5 = x[5]
        y[0] = (self.g00 * x0 + self.g01 * x1 + self.g02 * x2 +
                self.g03 * x3 + self.g04 * x4 + self.g05 * x5)
        y[1] = (self.g01 * x0 + self.g11 * x1 + self.g12 * x2 +
                self.g13 * x3 + self.g14 * x4 + self.g15 * x5)
        y[2] = (self.g02 * x0 + self.g12 * x1 + self.g22 * x2 +
                self.g23 * x3 + self.g24 * x4 + self.g25 * x5)
        y[3] = (self.g03 * x0 + self.g13 * x1 + self.g23 * x2 +
                self.g33 * x3 + self.g34 * x4 + self.g35 * x5)
        y[4] = (self.g04 * x0 + self.g14 * x1 + self.g24 * x2 +
                self.g34 * x3 + self.g44 * x4 + self.g45 * x5)
        y[5] = (self.g05 * x0 + self.g15 * x1 + self.g25 * x2 +
                self.g35 * x3 + self.g45 * x4 + self.g55 * x5)


cdef class FiniteDifferences2D(DiscreteGreenOperator2D):
    cdef double[:] k

    def __cinit__(self, AbstractGreenOperator green, shape, double h,
                  transform=None):
        self.k = array(shape=(2,), itemsize=sizeof(double), format='d')

    @boundscheck(False)
    @wraparound(False)
    cdef void c_set_frequency(self, int[:] b):
        cdef double phi0 = 0.5*self.s0*self.h*b[0]
        cdef double phi1 = 0.5*self.s1*self.h*b[1]
        self.k[0] = sin(phi0)*cos(phi1)
        self.k[1] = cos(phi0)*sin(phi1)
        self.green.set_frequency(self.k)

    cdef void c_to_memoryview(self, double[:, :] out):
        self.green.c_to_memoryview(out)

    cdef void c_apply_by_freq(self, double[:] tau, double[:] eta):
        self.green.c_apply(tau, eta)


cdef class FiniteDifferences3D(DiscreteGreenOperator3D):
    cdef double[:] k

    def __cinit__(self, AbstractGreenOperator green, shape, double h,
                  transform=None):
        self.k = array(shape=(3,), itemsize=sizeof(double), format='d')

    @boundscheck(False)
    @wraparound(False)
    cdef void c_set_frequency(self, int[:] b):
        cdef double phi0 = 0.5*self.s0*self.h*b[0]
        cdef c0 = cos(phi0)
        cdef s0 = sin(phi0)
        cdef double phi1 = 0.5*self.s1*self.h*b[1]
        cdef c1 = cos(phi1)
        cdef s1 = sin(phi1)
        cdef double phi2 = 0.5*self.s2*self.h*b[2]
        cdef c2 = cos(phi2)
        cdef s2 = sin(phi2)
        self.k[0] = s0*c1*c2
        self.k[1] = c0*s1*c2
        self.k[2] = c0*c1*s2
        self.green.set_frequency(self.k)

    cdef void c_to_memoryview(self, double[:, :] out):
        self.green.c_to_memoryview(out)

    cdef void c_apply_by_freq(self, double[:] tau, double[:] eta):
        self.green.c_apply(tau, eta)


def willot2015(green, n, h, transform=None):
    if green.dim == 2:
        return FiniteDifferences2D(green, n, h, transform)
    elif green.dim == 3:
        return FiniteDifferences3D(green, n, h, transform)
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(green.dim))
