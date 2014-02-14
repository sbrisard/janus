from cython cimport boundscheck
from cython cimport cdivision
from cython cimport wraparound
from cython.view cimport array
from libc.math cimport M_SQRT2

from janus.matprop cimport IsotropicLinearElasticMaterial as Material
from janus.operators cimport AbstractLinearOperator
from janus.utils.checkarray cimport check_shape_1d


def create(mat):
    if mat.dim == 2:
        return GreenOperator2D(mat)
    elif mat.dim == 3:
        return GreenOperator3D(mat)


cdef class AbstractGreenOperator(AbstractLinearOperator):

    @cdivision(True)
    def __cinit__(self, Material mat):
        cdef int sym = (mat.dim * (mat.dim + 1)) / 2
        self.init_sizes(sym, sym)
        self.dim = mat.dim
        self.mat = mat
        cdef double g = mat.g
        cdef double nu = mat.nu
        self.daux1 = 1.0 / g
        self.daux2 = 0.5 / (g * (1.0 - nu))
        self.daux3 = 0.25 / g
        self.daux4 = 0.5 / g

    cdef void c_set_frequency(self, double[:] k):
        """Set the current wave-vector of this Green operator.

        Any subsequent call to e.g. :func:`c_apply`,
        :func:`c_to_memoryview` are performed with the specified value
        of `k`.

        Concrete implementations of this method are not required to
        perform any test on the validity (size) of `k`.

        Parameters
        ----------
        k : memoryview of float64
            The current wave-vector.

        """
        raise NotImplementedError

    def set_frequency(self, double[:] k):
        """Set the current wave-vector of this Green operator.

        Any subsequent call to e.g. :func:`apply`, :func:`to_memoryview`
        are performed with the specified value of `k`.

        Parameters
        ----------
        k : memoryview of float64
            The current wave-vector.

        """
        check_shape_1d(k, self.dim)
        self.c_set_frequency(k)

    def __repr__(self):
        return 'Green Operator({0})'.format(self.mat)

cdef class GreenOperator2D(AbstractGreenOperator):

    cdef double g00, g01, g02, g11, g12, g22

    def __cinit__(self, Material mat):
        if (mat.dim != 2):
            raise ValueError('plane strain material expected')

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    cdef inline void c_set_frequency(self, double[:] k):
        cdef double k0 = k[0]
        cdef double k1 = k[1]
        cdef double k0k0 = k0 * k0
        cdef double k1k1 = k1 * k1
        cdef double s, k0k1
        if k0k0 + k1k1 == 0.:
            self.g00 = 0.
            self.g01 = 0.
            self.g02 = 0.
            self.g11 = 0.
            self.g12 = 0.
            self.g22 = 0.
        else:
            s = 1.0 / (k0k0 + k1k1)
            k0k0 *= s
            k1k1 *= s
            k0k1 = s * k0 * k1
            self.g00 = k0k0 * (self.daux1 - self.daux2 * k0k0)
            self.g11 = k1k1 * (self.daux1 - self.daux2 * k1k1)
            dummy = -self.daux2 * k0k0 * k1k1
            self.g01 = dummy
            self.g22 = 2 * (self.daux3 + dummy)
            self.g02 = M_SQRT2 * k0k1 * (self.daux4 - self.daux2 * k0k0)
            self.g12 = M_SQRT2 * k0k1 * (self.daux4 - self.daux2 * k1k1)

    @boundscheck(False)
    @wraparound(False)
    cdef void c_apply(self, double[:] x, double[:] y):
        cdef double x0, x1, x2, y0, y1, y2
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        y[0] = self.g00 * x0 + self.g01 * x1 + self.g02 * x2
        y[1] = self.g01 * x0 + self.g11 * x1 + self.g12 * x2
        y[2] = self.g02 * x0 + self.g12 * x1 + self.g22 * x2

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


cdef class GreenOperator3D(AbstractGreenOperator):

    cdef:
        double g00, g01, g02, g03, g04, g05, g11, g12, g13, g14, g15
        double g22, g23, g24, g25, g33, g34, g35, g44, g45, g55

    def __cinit__(self, Material mat):
        if (mat.dim != 3):
            raise ValueError('3D material expected')

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    cdef inline void c_set_frequency(self, double[:] k):
        cdef double k0 = k[0]
        cdef double k1 = k[1]
        cdef double k2 = k[2]
        cdef double k0k0 = k0 * k0
        cdef double k1k1 = k1 * k1
        cdef double k2k2 = k2 * k2
        cdef double s, k1k2, k2k0, k0k1
        if k0k0 + k1k1 +k2k2 == 0.:
            self.g00 = 0.
            self.g01 = 0.
            self.g02 = 0.
            self.g03 = 0.
            self.g04 = 0.
            self.g05 = 0.
            self.g11 = 0.
            self.g12 = 0.
            self.g13 = 0.
            self.g14 = 0.
            self.g15 = 0.
            self.g22 = 0.
            self.g23 = 0.
            self.g24 = 0.
            self.g25 = 0.
            self.g33 = 0.
            self.g34 = 0.
            self.g35 = 0.
            self.g44 = 0.
            self.g45 = 0.
            self.g55 = 0.
            return
        else:
            s = 1.0 / (k0k0 + k1k1 + k2k2)
            k0k0 *= s
            k1k1 *= s
            k2k2 *= s
            k1k2 = s * k1 * k2
            k2k0 = s * k2 * k0
            k0k1 = s * k0 * k1

            self.g00 = k0k0 * (self.daux1 - self.daux2 * k0k0)
            self.g11 = k1k1 * (self.daux1 - self.daux2 * k1k1)
            self.g22 = k2k2 * (self.daux1 - self.daux2 * k2k2)
            self.g33 = 2. * (self.daux3 * (k1k1 + k2k2)
                             - self.daux2 * k1k1 * k2k2)
            self.g44 = 2. * (self.daux3 * (k2k2 + k0k0)
                             - self.daux2 * k2k2 * k0k0)
            self.g55 = 2. * (self.daux3 * (k0k0 + k1k1)
                             - self.daux2 * k0k0 * k1k1)
            self.g01 = -self.daux2 * k0k0 * k1k1
            self.g02 = -self.daux2 * k0k0 * k2k2
            self.g03 = -M_SQRT2 * self.daux2 * k0k0 * k1k2
            self.g04 = M_SQRT2 * k2k0 * (self.daux4 - self.daux2 * k0k0)
            self.g05 = M_SQRT2 * k0k1 * (self.daux4 - self.daux2 * k0k0)
            self.g12 = -self.daux2 * k1k1 * k2k2
            self.g13 = M_SQRT2 * k1k2 * (self.daux4 - self.daux2 * k1k1)
            self.g14 = -M_SQRT2 * self.daux2 * k1k1 * k2k0
            self.g15 = M_SQRT2 * k0k1 * (self.daux4 - self.daux2 * k1k1)
            self.g23 = M_SQRT2 * k1k2 * (self.daux4 - self.daux2 * k2k2)
            self.g24 = M_SQRT2 * k2k0 * (self.daux4 - self.daux2 * k2k2)
            self.g25 = -M_SQRT2 * self.daux2 * k2k2 * k0k1
            self.g34 = 2 * k0k1 * (self.daux3 - self.daux2 * k2k2)
            self.g35 = 2 * k2k0 * (self.daux3 - self.daux2 * k1k1)
            self.g45 = 2 * k1k2 * (self.daux3 - self.daux2 * k0k0)

    @boundscheck(False)
    @wraparound(False)
    cdef void c_apply(self, double[:] x, double[:] y):
        cdef double x0, x1, x2, x3, x4, x5
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        x3 = x[3]
        x4 = x[4]
        x5 = x[5]
        y[0] = (self.g00 * x0 + self.g01 * x1 + self.g02 * x2
                + self.g03 * x3 + self.g04 * x4 + self.g05 * x5)
        y[1] = (self.g01 * x0 + self.g11 * x1 + self.g12 * x2
                + self.g13 * x3 + self.g14 * x4 + self.g15 * x5)
        y[2] = (self.g02 * x0 + self.g12 * x1 + self.g22 * x2
                + self.g23 * x3 + self.g24 * x4 + self.g25 * x5)
        y[3] = (self.g03 * x0 + self.g13 * x1 + self.g23 * x2
                + self.g33 * x3 + self.g34 * x4 + self.g35 * x5)
        y[4] = (self.g04 * x0 + self.g14 * x1 + self.g24 * x2
                + self.g34 * x3 + self.g44 * x4 + self.g45 * x5)
        y[5] = (self.g05 * x0 + self.g15 * x1 + self.g25 * x2
                + self.g35 * x3 + self.g45 * x4 + self.g55 * x5)

    @boundscheck(False)
    @wraparound(False)
    cdef void c_to_memoryview(self, double[:, :] out):
        out[0, 0] = self.g00
        out[0, 1] = self.g01
        out[0, 2] = self.g02
        out[0, 3] = self.g03
        out[0, 4] = self.g04
        out[0, 5] = self.g05
        out[1, 0] = self.g01
        out[1, 1] = self.g11
        out[1, 2] = self.g12
        out[1, 3] = self.g13
        out[1, 4] = self.g14
        out[1, 5] = self.g15
        out[2, 0] = self.g02
        out[2, 1] = self.g12
        out[2, 2] = self.g22
        out[2, 3] = self.g23
        out[2, 4] = self.g24
        out[2, 5] = self.g25
        out[3, 0] = self.g03
        out[3, 1] = self.g13
        out[3, 2] = self.g23
        out[3, 3] = self.g33
        out[3, 4] = self.g34
        out[3, 5] = self.g35
        out[4, 0] = self.g04
        out[4, 1] = self.g14
        out[4, 2] = self.g24
        out[4, 3] = self.g34
        out[4, 4] = self.g44
        out[4, 5] = self.g45
        out[5, 0] = self.g05
        out[5, 1] = self.g15
        out[5, 2] = self.g25
        out[5, 3] = self.g35
        out[5, 4] = self.g45
        out[5, 5] = self.g55
