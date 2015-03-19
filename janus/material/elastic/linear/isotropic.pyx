"""This module defines isotropic, linear elastic materials.

Materials defined in this module can operate on physical space of
dimension dim = 2 or 3. dim = 2 refers to plane strain elasticity, while
dim = 3 refers to classical 3D elasticity.

To create a plane stress material with shear modulus mu and Poisson
ratio nu, a plane strain material should be created, with the same shear
modulus mu, and fictitious Poisson ratio nu / (1 + nu).

"""

from cython cimport boundscheck
from cython cimport cdivision
from cython cimport wraparound
from libc.math cimport M_SQRT2

from janus.green cimport AbstractGreenOperator


def create(g, nu, dim=3):
    """Create a new isotropic, linear and elastic material.

    Args:
        g: the shear modulus
        nu: the Poisson ratio
        dim: the dimension of the physical space (default 3)

    Returns:
        A new instance of IsotropicLinearElasticMaterial.

    """
    return IsotropicLinearElasticMaterial(g, nu, dim)


@cdivision(True)
cpdef double poisson_from_bulk_and_shear_moduli(double k, double g, int dim=3):
    """Compute the Poisson ratio from the bulk and shear moduli.

    No checks are performed on the positivity of the bulk and shear
    moduli, or on the validity of the returned Poisson ratio (which
    should lie between -1 and 1/2).

    Args:
        k: the bulk modulus
        g: the shear modulus
        dim: the dimension of the physical space (default 3). Must be 2
            (plane strain elasticity) or 3 (3D elasticity).


    Returns:
        The Poisson ratio.

    """
    if dim == 2:
        return (k - g) / (2. * k)
    elif dim == 3:
        return (3 * k - 2 * g) / (2 * (3 * k + g))
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(dim))


cdef class IsotropicLinearElasticMaterial:

    """Isotropic, linear and elastic materials.

    Args:
        g: the shear modulus
        nu: the Poisson ratio
        dim: the dimension of the physical space (default 3)

    Attributes:
        k: the bulk modulus (read-only)
        g: the shear modulus (read-only)
        nu: the Poisson ratio (read-only)
        dim: the dimension of the physical space (read-only)

    """

    @cdivision(True)
    def __cinit__(self, double g, double nu, int dim=3):
        if g < 0.0:
            raise ValueError('g must be >= 0 (was {0})'.format(g))
        if dim == 2 and nu > 0.5:
            raise ValueError('nu must be <= 1/2 (was {0})'.format(nu))
        if dim == 3 and (nu < -1.0 or nu > 0.5):
            raise ValueError('nu must be >= -1 and <= 1/2 (was {0})'.format(nu))
        if dim != 2 and dim != 3:
            raise ValueError('dim must be 2 or 3 (was {0})'.format(dim))
        self.g = g
        self.nu = nu
        self.dim = dim
        if dim == 2:
            self.k = g / (1. - 2. * nu)
        elif dim == 3:
            self.k = 2. / 3. * (1. + nu) / (1. - 2. * nu) * g
        else:
            raise RuntimeError('Inconsistent state')

    def __repr__(self):
        return ('IsotropicLinearElasticMaterial'
                '(g={0}, nu={1}, dim={2})').format(self.g,
                                                   self.nu,
                                                   self.dim)

    def green_operator(self):
        """Return a new Green operator for this material.

        The returned Green operator is the periodic Green operator for
        strains, which is given in Fourier space.
        """
        if self.dim == 2:
            return _GreenOperatorForStrains2D(self)
        elif self.dim == 3:
            return _GreenOperatorForStrains3D(self)


cdef class _GreenOperatorForStrains(AbstractGreenOperator):

    """Periodic Green operator for strains.

    Instances of this class are associated to isotropic, linear elastic
    materials.

    This class factorizes code common to the 2D and 3D implementations.

    Args:
        mat: the underlying material

    Attributes:
        mat: the underlying material

    """

    # The auxiliary variables daux1 to daux4 are defined as follows
    #   daux1 = 1 / g
    #   daux2 = 1 / [2 * g * (1 - nu)]
    #   daux3 = 1 / (4 * g)
    #   daux4 = 1 / (2 * g)
    # where g (resp. nu) is the shear modulus (resp. Poisson ratio) of the
    # reference material.
    cdef double daux1, daux2, daux3, daux4
    cdef readonly IsotropicLinearElasticMaterial mat

    @cdivision(True)
    def __cinit__(self, IsotropicLinearElasticMaterial mat):
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


cdef class _GreenOperatorForStrains2D(_GreenOperatorForStrains):

    """Periodic Green operator for strains (2D implementation). """

    cdef double g00, g01, g02, g11, g12, g22

    @cdivision(True)
    def __cinit__(self, IsotropicLinearElasticMaterial mat):
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


cdef class _GreenOperatorForStrains3D(_GreenOperatorForStrains):

    """Periodic Green operator for strains (3D implementation). """

    cdef:
        double g00, g01, g02, g03, g04, g05, g11, g12, g13, g14, g15
        double g22, g23, g24, g25, g33, g34, g35, g44, g45, g55

    def __cinit__(self, IsotropicLinearElasticMaterial mat):
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
