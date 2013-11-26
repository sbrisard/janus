from cython cimport boundscheck
from cython cimport cdivision
from cython.view cimport array
from libc.math cimport M_SQRT2

from checkarray cimport check_shape_1d
from checkarray cimport create_or_check_shape_1d
from checkarray cimport create_or_check_shape_2d
from matprop cimport IsotropicLinearElasticMaterial as Material

def create(mat):
    if mat.dim == 2:
        return GreenOperator2d(mat)
    elif mat.dim == 3:
        return GreenOperator3d(mat)

cdef class GreenOperator:

    @cdivision(True)
    def __cinit__(self, Material mat):
        self.dim = mat.dim
        self.sym = (mat.dim * (mat.dim + 1)) / 2
        self.mat = mat
        cdef double g = mat.g
        cdef double nu = mat.nu
        self.daux1 = 1.0 / g
        self.daux2 = 0.5 / (g * (1.0 - nu))
        self.daux3 = 0.25 / g
        self.daux4 = 0.5 / g

    cdef void capply(self, double *k, double[:] tau, double[:] eta):
        """apply(k, tau, eta)

        Apply the Green operator to the specified prestress. C version of the
        ``apply`` method, which performs no checks on the parameters.

        Parameters
        ----------
        k : pointer to double
            Wave-vector.
        tau : array_like
            Value of the prestress for the specified mode `k`.
        eta : array_like, optional
            Result of the operation `Gamma(k) : tau`.

        Returns
        -------
        eta : array_like
            The result of the linear operation `Gamma(k) : tau`.
        """
        pass

    def apply(self, double[::1] k, double[:] tau, double[:] eta=None):
        """apply(k, tau, eta=None)

        Apply the Green operator to the specified prestress.

        Parameters
        ----------
        k : array_like
            Wave-vector.
        tau : array_like
            Value of the prestress for the specified mode `k`.
        eta : array_like, optional
            Result of the operation `Gamma(k) : tau`.

        Returns
        -------
        eta : array_like
            The result of the linear operation `Gamma(k) : tau`.
        """
        pass

    def asarray(self, double[::1] k, double[:, :] g=None):
        """asarray(k, g=None)

        Return the array representation of the Green operator for the
        specified wave vector. Uses the Mandel-Voigt convention.

        Parameters
        ----------
        k : array_like
            Wave-vector.
        g : array_like, optional
            Matrix, to be updated.

        Returns
        -------
        g : array_like
            Matrix of the Green operator.
        """
        pass

cdef class GreenOperator2d(GreenOperator):

    cdef double m00, m01, m02, m11, m12, m22

    def __cinit__(self, Material mat):
        if (mat.dim != 2):
            raise ValueError('plane strain material expected')

    @boundscheck(False)
    @cdivision(True)
    cdef void _update(self, double kx, double ky):
        """Compute the coefficients of the underlying matrix for the specified
        value of the wave vector.

        No checks are performed on the parameters.

        Parameters
        ----------
        kx : double
            x-component of the wave-vector.
        ky : double
            y-component of the wave-vector.

        """
        cdef double kxkx = kx * kx
        cdef double kyky = ky * ky
        cdef double s, kxky
        if kxkx + kyky == 0.:
            self.m00 = 0.
            self.m01 = 0.
            self.m02 = 0.
            self.m11 = 0.
            self.m12 = 0.
            self.m22 = 0.
        else:
            s = 1.0 / (kxkx + kyky)
            kxkx *= s
            kyky *= s
            kxky = s * kx * ky
            self.m00 = kxkx * (self.daux1 - self.daux2 * kxkx)
            self.m11 = kyky * (self.daux1 - self.daux2 * kyky)
            dummy = -self.daux2 * kxkx * kyky
            self.m01 = dummy
            self.m22 = 2 * (self.daux3 + dummy)
            self.m02 = M_SQRT2 * kxky * (self.daux4 - self.daux2 * kxkx)
            self.m12 = M_SQRT2 * kxky * (self.daux4 - self.daux2 * kyky)

    @boundscheck(False)
    cdef inline void _capply(self, double kx, double ky,
                             double[:] tau, double[:] eta):
        self._update(kx, ky)
        eta[0] = self.m00 * tau[0] + self.m01 * tau[1] + self.m02 * tau[2]
        eta[1] = self.m01 * tau[0] + self.m11 * tau[1] + self.m12 * tau[2]
        eta[2] = self.m02 * tau[0] + self.m12 * tau[1] + self.m22 * tau[2]

    @boundscheck(False)
    cdef void capply(self, double* k, double[:] tau, double[:] eta):
        self._capply(k[0], k[1], tau, eta)

    def apply(self, double[::1] k, double[:] tau, double[:] eta=None):
        check_shape_1d(k, self.dim)
        check_shape_1d(tau, self.sym)
        eta = create_or_check_shape_1d(eta, self.sym)
        self._capply(k[0], k[1], tau, eta)
        return eta

    @boundscheck(False)
    def asarray(self, double[::1] k, double[:, :] g=None):
        check_shape_1d(k, self.dim)
        g = create_or_check_shape_2d(g, self.sym, self.sym)
        self._update(k[0], k[1])
        g[0, 0] = self.m00
        g[0, 1] = self.m01
        g[0, 2] = self.m02
        g[1, 0] = self.m01
        g[1, 1] = self.m11
        g[1, 2] = self.m12
        g[2, 0] = self.m02
        g[2, 1] = self.m12
        g[2, 2] = self.m22
        return g

cdef class GreenOperator3d(GreenOperator):

    cdef:
        double m00, m01, m02, m03, m04, m05, m11, m12, m13, m14, m15
        double m22, m23, m24, m25, m33, m34, m35, m44, m45, m55

    def __cinit__(self, Material mat):
        if (mat.dim != 3):
            raise ValueError('3Dmaterial expected')

    @boundscheck(False)
    @cdivision(True)
    cdef void _update(self, double kx, double ky, double kz):
        """Compute the coefficients of the underlying matrix for the
        specified value of the wave vector.

        Parameters
        ----------
        kx : double
            x-component of the wave-vector.
        ky : double
            y-component of the wave-vector.
        kz : double
            z-component of the wave-vector.

        """
        cdef double kxkx = kx * kx
        cdef double kyky = ky * ky
        cdef double kzkz = kz * kz
        cdef double s, kykz, kzkx, kxky
        if kxkx + kyky +kzkz == 0.:
            self.m00 = 0.
            self.m01 = 0.
            self.m02 = 0.
            self.m03 = 0.
            self.m04 = 0.
            self.m05 = 0.
            self.m11 = 0.
            self.m12 = 0.
            self.m13 = 0.
            self.m14 = 0.
            self.m15 = 0.
            self.m22 = 0.
            self.m23 = 0.
            self.m24 = 0.
            self.m25 = 0.
            self.m33 = 0.
            self.m34 = 0.
            self.m35 = 0.
            self.m44 = 0.
            self.m45 = 0.
            self.m55 = 0.
            return
        else:
            s = 1.0 / (kxkx + kyky + kzkz)
            kxkx *= s
            kyky *= s
            kzkz *= s
            kykz = s * ky * kz
            kzkx = s * kz * kx
            kxky = s * kx * ky

            self.m00 = kxkx * (self.daux1 - self.daux2 * kxkx)
            self.m11 = kyky * (self.daux1 - self.daux2 * kyky)
            self.m22 = kzkz * (self.daux1 - self.daux2 * kzkz)
            self.m33 = 2. * (self.daux3 * (kyky + kzkz)
                             - self.daux2 * kyky * kzkz)
            self.m44 = 2. * (self.daux3 * (kzkz + kxkx)
                             - self.daux2 * kzkz * kxkx)
            self.m55 = 2. * (self.daux3 * (kxkx + kyky)
                             - self.daux2 * kxkx * kyky)
            self.m01 = -self.daux2 * kxkx * kyky
            self.m02 = -self.daux2 * kxkx * kzkz
            self.m03 = -M_SQRT2 * self.daux2 * kxkx * kykz
            self.m04 = M_SQRT2 * kzkx * (self.daux4 - self.daux2 * kxkx)
            self.m05 = M_SQRT2 * kxky * (self.daux4 - self.daux2 * kxkx)
            self.m12 = -self.daux2 * kyky * kzkz
            self.m13 = M_SQRT2 * kykz * (self.daux4 - self.daux2 * kyky)
            self.m14 = -M_SQRT2 * self.daux2 * kyky * kzkx
            self.m15 = M_SQRT2 * kxky * (self.daux4 - self.daux2 * kyky)
            self.m23 = M_SQRT2 * kykz * (self.daux4 - self.daux2 * kzkz)
            self.m24 = M_SQRT2 * kzkx * (self.daux4 - self.daux2 * kzkz)
            self.m25 = -M_SQRT2 * self.daux2 * kzkz * kxky
            self.m34 = 2 * kxky * (self.daux3 - self.daux2 * kzkz)
            self.m35 = 2 * kzkx * (self.daux3 - self.daux2 * kyky)
            self.m45 = 2 * kykz * (self.daux3 - self.daux2 * kxkx)

    @boundscheck(False)
    cdef inline void _capply(self, double kx, double ky, double kz,
                             double[:] tau, double[:] eta):
        self._update(kx, ky, kz)
        eta[0] = (self.m00 * tau[0] + self.m01 * tau[1] + self.m02 * tau[2]
                  + self.m03 * tau[3] + self.m04 * tau[4] + self.m05 * tau[5])
        eta[1] = (self.m01 * tau[0] + self.m11 * tau[1] + self.m12 * tau[2]
                  + self.m13 * tau[3] + self.m14 * tau[4] + self.m15 * tau[5])
        eta[2] = (self.m02 * tau[0] + self.m12 * tau[1] + self.m22 * tau[2]
                  + self.m23 * tau[3] + self.m24 * tau[4] + self.m25 * tau[5])
        eta[3] = (self.m03 * tau[0] + self.m13 * tau[1] + self.m23 * tau[2]
                  + self.m33 * tau[3] + self.m34 * tau[4] + self.m35 * tau[5])
        eta[4] = (self.m04 * tau[0] + self.m14 * tau[1] + self.m24 * tau[2]
                  + self.m34 * tau[3] + self.m44 * tau[4] + self.m45 * tau[5])
        eta[5] = (self.m05 * tau[0] + self.m15 * tau[1] + self.m25 * tau[2]
                  + self.m35 * tau[3] + self.m45 * tau[4] + self.m55 * tau[5])

    @boundscheck(False)
    cdef void capply(self, double *k, double[:] tau, double[:] eta):
        self._capply(k[0], k[1], k[2], tau, eta)

    @boundscheck(False)
    def apply(self, double[::1] k, double[:] tau, double[:] eta=None):
        check_shape_1d(k, self.dim)
        check_shape_1d(tau, self.sym)
        eta = create_or_check_shape_1d(eta, self.sym)
        self._capply(k[0], k[1], k[2], tau, eta)
        return eta

    @boundscheck(False)
    cpdef double[:, :] asarray(self, double[::1] k, double[:, :] g=None):
        check_shape_1d(k, self.dim)
        g = create_or_check_shape_2d(g, self.sym, self.sym)
        self._update(k[0], k[1], k[2])
        g[0, 0] = self.m00
        g[0, 1] = self.m01
        g[0, 2] = self.m02
        g[0, 3] = self.m03
        g[0, 4] = self.m04
        g[0, 5] = self.m05
        g[1, 0] = self.m01
        g[1, 1] = self.m11
        g[1, 2] = self.m12
        g[1, 3] = self.m13
        g[1, 4] = self.m14
        g[1, 5] = self.m15
        g[2, 0] = self.m02
        g[2, 1] = self.m12
        g[2, 2] = self.m22
        g[2, 3] = self.m23
        g[2, 4] = self.m24
        g[2, 5] = self.m25
        g[3, 0] = self.m03
        g[3, 1] = self.m13
        g[3, 2] = self.m23
        g[3, 3] = self.m33
        g[3, 4] = self.m34
        g[3, 5] = self.m35
        g[4, 0] = self.m04
        g[4, 1] = self.m14
        g[4, 2] = self.m24
        g[4, 3] = self.m34
        g[4, 4] = self.m44
        g[4, 5] = self.m45
        g[5, 0] = self.m05
        g[5, 1] = self.m15
        g[5, 2] = self.m25
        g[5, 3] = self.m35
        g[5, 4] = self.m45
        g[5, 5] = self.m55
        return g
