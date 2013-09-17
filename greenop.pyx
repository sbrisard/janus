from cython cimport boundscheck
from libc.math cimport sqrt

from matprop cimport IsotropicLinearElasticMaterial as Material

cdef double SQRT_TWO = sqrt(2.)

cdef class GreenOperator2d:

    """This class defines the periodic Green operator for 2d (plane
    strain elasticity.

    """

    cdef:
        readonly int dim, sym
        readonly Material mat
        double daux1, daux2, daux3, daux4

    cdef double daux1, daux2, daux3, daux4

    
    def __cinit__(self, Material mat):
        self.dim = 2
        self.sym = 3
        if (self.dim != mat.dim):
            raise ValueError('plane strain material expected')
        self.mat = mat
        cdef double mu = mat.g
        cdef double nu = mat.nu
        self.daux1 = 1.0 / mu
        self.daux2 = 0.5 / (mu * (1.0 - nu))
        self.daux3 = 0.25 / mu
        self.daux4 = 0.5 / mu

    @boundscheck(False)
    cpdef double[::1] apply(self,
                            double[::1] k,
                            double[::1] tau,
                            double[::1] eps):

        # The following tests are necessary, since bounds checks are removed.
        if k.shape[0] != self.dim:
            raise IndexError('shape of k must be ({0},)'.format(self.dim))

        if tau.shape[0] != self.sym:
            raise IndexError('shape of tau must be ({0},)'.format(self.sym))

        if eps.shape[0] != self.sym:
            raise IndexError('shape of eps must be ({0},)'.format(self.sym))

        cdef double kx = k[0]
        cdef double ky = k[1]
        cdef double kxkx = kx * kx
        cdef double kyky = ky * ky
        if kxkx + kyky == 0.:
            eps[0] = 0.
            eps[1] = 0.
            eps[2] = 0.
            return eps

        cdef double s = 1.0 / (kxkx + kyky)
        kxkx *= s
        kyky *= s
        cdef double kxky = s * kx * ky

        cdef double m00 = kxkx * (self.daux1 - self.daux2 * kxkx)
        cdef double m11 = kyky * (self.daux1 - self.daux2 * kyky)

        cdef double dummy = -self.daux2 * kxkx * kyky
        cdef double m01 = dummy
        cdef double m22 = 2 * (self.daux3 + dummy)
        cdef double m02 = SQRT_TWO * kxky * (self.daux4 - self.daux2 * kxkx)
        cdef double m12 = SQRT_TWO * kxky * (self.daux4 - self.daux2 * kyky)

        eps[0] = m00 * tau[0] + m01 * tau[1] + m02 * tau[2]
        eps[1] = m01 * tau[0] + m11 * tau[1] + m12 * tau[2]
        eps[2] = m02 * tau[0] + m12 * tau[1] + m22 * tau[2]

        return eps
