import numpy as np
cimport numpy as np

from cython cimport boundscheck
from libc.math cimport sqrt

from matprop cimport IsotropicLinearElasticMaterial as Material

cdef double SQRT_TWO = sqrt(2.)

cdef class GreenOperator2d:

    """This class defines the periodic Green operator for 2d (plane
    strain) elasticity.

    Parameters
    ----------
    mat : IsotropicLinearElasticMaterial
        Reference material.

    Attributes
    ----------
    dim : int
        Dimension of the physical space.
    mat : IsotropicLinearElasticMaterial
        Reference material.
    sym : int
        Dimension of the space on which this object operates (space of
        the second-rank, symmetric tensors).
    daux1 : double
        Value of `1 / g`, where `g` is the shear modulus of the reference
        material.
    daux2 : double
        Value of `1 / 2 / g / (1 - nu)`, where `g` (resp. `nu`) is the
        shear modulus (resp. Poisson ratio) of the reference material.
    daux3 : double
        Value of `1 / 4 / g`, where `g` is the shear modulus of the
        reference material.
    daux4 : double
        Value of `1 / g`, where `g` is the shear modulus of the
        reference material.

    """

    cdef:
        readonly int dim
        readonly Material mat
        int sym
        double daux1, daux2, daux3, daux4

    def __cinit__(self, Material mat):
        self.dim = 2
        self.sym = 3
        if (self.dim != mat.dim):
            raise ValueError('plane strain material expected')
        self.mat = mat
        cdef double g = mat.g
        cdef double nu = mat.nu
        self.daux1 = 1.0 / g
        self.daux2 = 0.5 / (g * (1.0 - nu))
        self.daux3 = 0.25 / g
        self.daux4 = 0.5 / g

    @boundscheck(False)
    cpdef double[:] apply(self, double[:] k, double[:] tau,
                          double[:] eps = None):
        """ Apply the Green operator to the specified prestress.

        Parameters
        ----------
        k : array_like
            The wave-vector.
        tau : array_like
            The value of the prestress for the specified mode `k`.
        eps : array_like, optional
            The result of the operation `Gamma(k) : tau`. Strictly
            speaking, `eps` is the opposite of a strain.

        Returns
        -------
        eps : array_like
            The result of the linear operation `Gamma(k) : tau`.
        """

        # The following tests are necessary, since bounds checks are removed.
        if k.shape[0] != self.dim:
            raise IndexError('shape of k must be ({0},)'.format(self.dim))

        if tau.shape[0] != self.sym:
            raise IndexError('shape of tau must be ({0},)'.format(self.sym))

        if eps is not None:
            if eps.shape[0] != self.sym:
                raise IndexError('shape of eps must be ({0},)'.format(self.sym))
        else:
            eps = np.empty((self.sym,), dtype = np.float64)

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
