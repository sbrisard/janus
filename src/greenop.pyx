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
        return GreenOperator2D(mat)
    elif mat.dim == 3:
        return GreenOperator3D(mat)

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

    def apply(self, double[:] k, double[:] tau, double[:] eta=None):
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

    def asarray(self, double[:] k, double[:, :] g=None):
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

cdef class GreenOperator2D(GreenOperator):

    cdef double g00, g01, g02, g11, g12, g22

    def __cinit__(self, Material mat):
        if (mat.dim != 2):
            raise ValueError('plane strain material expected')

    @boundscheck(False)
    @cdivision(True)
    cdef inline void _update(self, double kx, double ky):
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
            self.g00 = 0.
            self.g01 = 0.
            self.g02 = 0.
            self.g11 = 0.
            self.g12 = 0.
            self.g22 = 0.
        else:
            s = 1.0 / (kxkx + kyky)
            kxkx *= s
            kyky *= s
            kxky = s * kx * ky
            self.g00 = kxkx * (self.daux1 - self.daux2 * kxkx)
            self.g11 = kyky * (self.daux1 - self.daux2 * kyky)
            dummy = -self.daux2 * kxkx * kyky
            self.g01 = dummy
            self.g22 = 2 * (self.daux3 + dummy)
            self.g02 = M_SQRT2 * kxky * (self.daux4 - self.daux2 * kxkx)
            self.g12 = M_SQRT2 * kxky * (self.daux4 - self.daux2 * kyky)

    @boundscheck(False)
    cdef inline void _capply(self, double kx, double ky,
                             double[:] tau, double[:] eta):
        self._update(kx, ky)
        eta[0] = self.g00 * tau[0] + self.g01 * tau[1] + self.g02 * tau[2]
        eta[1] = self.g01 * tau[0] + self.g11 * tau[1] + self.g12 * tau[2]
        eta[2] = self.g02 * tau[0] + self.g12 * tau[1] + self.g22 * tau[2]

    @boundscheck(False)
    cdef void capply(self, double* k, double[:] tau, double[:] eta):
        self._capply(k[0], k[1], tau, eta)

    def apply(self, double[:] k, double[:] tau, double[:] eta=None):
        check_shape_1d(k, self.dim)
        check_shape_1d(tau, self.sym)
        eta = create_or_check_shape_1d(eta, self.sym)
        self._capply(k[0], k[1], tau, eta)
        return eta

    @boundscheck(False)
    def asarray(self, double[:] k, double[:, :] out=None):
        check_shape_1d(k, self.dim)
        out = create_or_check_shape_2d(out, self.sym, self.sym)
        self._update(k[0], k[1])
        out[0, 0] = self.g00
        out[0, 1] = self.g01
        out[0, 2] = self.g02
        out[1, 0] = self.g01
        out[1, 1] = self.g11
        out[1, 2] = self.g12
        out[2, 0] = self.g02
        out[2, 1] = self.g12
        out[2, 2] = self.g22
        return out

cdef class GreenOperator3D(GreenOperator):

    cdef:
        double g00, g01, g02, g03, g04, g05, g11, g12, g13, g14, g15
        double g22, g23, g24, g25, g33, g34, g35, g44, g45, g55

    def __cinit__(self, Material mat):
        if (mat.dim != 3):
            raise ValueError('3D material expected')

    @boundscheck(False)
    @cdivision(True)
    cdef inline void _update(self, double kx, double ky, double kz):
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
            s = 1.0 / (kxkx + kyky + kzkz)
            kxkx *= s
            kyky *= s
            kzkz *= s
            kykz = s * ky * kz
            kzkx = s * kz * kx
            kxky = s * kx * ky

            self.g00 = kxkx * (self.daux1 - self.daux2 * kxkx)
            self.g11 = kyky * (self.daux1 - self.daux2 * kyky)
            self.g22 = kzkz * (self.daux1 - self.daux2 * kzkz)
            self.g33 = 2. * (self.daux3 * (kyky + kzkz)
                             - self.daux2 * kyky * kzkz)
            self.g44 = 2. * (self.daux3 * (kzkz + kxkx)
                             - self.daux2 * kzkz * kxkx)
            self.g55 = 2. * (self.daux3 * (kxkx + kyky)
                             - self.daux2 * kxkx * kyky)
            self.g01 = -self.daux2 * kxkx * kyky
            self.g02 = -self.daux2 * kxkx * kzkz
            self.g03 = -M_SQRT2 * self.daux2 * kxkx * kykz
            self.g04 = M_SQRT2 * kzkx * (self.daux4 - self.daux2 * kxkx)
            self.g05 = M_SQRT2 * kxky * (self.daux4 - self.daux2 * kxkx)
            self.g12 = -self.daux2 * kyky * kzkz
            self.g13 = M_SQRT2 * kykz * (self.daux4 - self.daux2 * kyky)
            self.g14 = -M_SQRT2 * self.daux2 * kyky * kzkx
            self.g15 = M_SQRT2 * kxky * (self.daux4 - self.daux2 * kyky)
            self.g23 = M_SQRT2 * kykz * (self.daux4 - self.daux2 * kzkz)
            self.g24 = M_SQRT2 * kzkx * (self.daux4 - self.daux2 * kzkz)
            self.g25 = -M_SQRT2 * self.daux2 * kzkz * kxky
            self.g34 = 2 * kxky * (self.daux3 - self.daux2 * kzkz)
            self.g35 = 2 * kzkx * (self.daux3 - self.daux2 * kyky)
            self.g45 = 2 * kykz * (self.daux3 - self.daux2 * kxkx)

    @boundscheck(False)
    cdef inline void _capply(self, double kx, double ky, double kz,
                             double[:] tau, double[:] eta):
        self._update(kx, ky, kz)
        eta[0] = (self.g00 * tau[0] + self.g01 * tau[1] + self.g02 * tau[2]
                  + self.g03 * tau[3] + self.g04 * tau[4] + self.g05 * tau[5])
        eta[1] = (self.g01 * tau[0] + self.g11 * tau[1] + self.g12 * tau[2]
                  + self.g13 * tau[3] + self.g14 * tau[4] + self.g15 * tau[5])
        eta[2] = (self.g02 * tau[0] + self.g12 * tau[1] + self.g22 * tau[2]
                  + self.g23 * tau[3] + self.g24 * tau[4] + self.g25 * tau[5])
        eta[3] = (self.g03 * tau[0] + self.g13 * tau[1] + self.g23 * tau[2]
                  + self.g33 * tau[3] + self.g34 * tau[4] + self.g35 * tau[5])
        eta[4] = (self.g04 * tau[0] + self.g14 * tau[1] + self.g24 * tau[2]
                  + self.g34 * tau[3] + self.g44 * tau[4] + self.g45 * tau[5])
        eta[5] = (self.g05 * tau[0] + self.g15 * tau[1] + self.g25 * tau[2]
                  + self.g35 * tau[3] + self.g45 * tau[4] + self.g55 * tau[5])

    @boundscheck(False)
    cdef void capply(self, double *k, double[:] tau, double[:] eta):
        self._capply(k[0], k[1], k[2], tau, eta)

    @boundscheck(False)
    def apply(self, double[:] k, double[:] tau, double[:] eta=None):
        check_shape_1d(k, self.dim)
        check_shape_1d(tau, self.sym)
        eta = create_or_check_shape_1d(eta, self.sym)
        self._capply(k[0], k[1], k[2], tau, eta)
        return eta

    @boundscheck(False)
    def asarray(self, double[:] k, double[:, :] out=None):
        check_shape_1d(k, self.dim)
        out = create_or_check_shape_2d(out, self.sym, self.sym)
        self._update(k[0], k[1], k[2])
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
        return out
