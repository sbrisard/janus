from cython cimport boundscheck
from cython.view cimport array
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
        double m00, m01, m02, m11, m12, m22

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
    cdef void update(self, double[:] k):
        """`update(k)`
    
        Compute the coefficients of the underlying matrix for the
        specified value of the wave vector.

        Parameters
        ----------
        k : array_like
            Wave-vector.
        
        """
        # The following tests are necessary, since bounds checks are removed.
        if k.shape[0] != self.dim:
            raise IndexError('shape of k must be ({0},)'.format(self.dim))

        cdef double kx = k[0]
        cdef double ky = k[1]
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
            self.m02 = SQRT_TWO * kxky * (self.daux4 - self.daux2 * kxkx)
            self.m12 = SQRT_TWO * kxky * (self.daux4 - self.daux2 * kyky)

    @boundscheck(False)
    cpdef double[:] apply(self, double[:] k, double[:] tau,
                          double[:] eps = None):
        """
        `apply(k, tau, eps = None)`
                          
        Apply the Green operator to the specified prestress.

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
        if tau.shape[0] != self.sym:
            raise IndexError('shape of tau must be ({0},)'.format(self.sym))

        if eps is not None:
            if eps.shape[0] != self.sym:
                raise IndexError('shape of eps must be ({0},)'.format(self.sym))
        else:
            eps = array(shape=(self.sym,), itemsize=sizeof(double), format='d')

        self.update(k)
        eps[0] = self.m00 * tau[0] + self.m01 * tau[1] + self.m02 * tau[2]
        eps[1] = self.m01 * tau[0] + self.m11 * tau[1] + self.m12 * tau[2]
        eps[2] = self.m02 * tau[0] + self.m12 * tau[1] + self.m22 * tau[2]

        return eps

    @boundscheck(False)
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

        if g is not None:
            if g.shape[0] != self.sym or g.shape[1] != self.sym:
                raise IndexError('shape of g must be ({0}, {0})'
                                 .format(self.sym, self.sym))
        else:
            g = array(shape=(self.sym, self.sym),
                      itemsize=sizeof(double), format='d')
        self.update(k)
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

cdef class GreenOperator3d:

    """This class defines the periodic Green operator for 3d elasticity.

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
        Value of `1 / 2 / g`, where `g` is the shear modulus of the
        reference material.

    """

    cdef:
        readonly int dim
        readonly Material mat
        int sym
        double daux1, daux2, daux3, daux4
        double m00, m01, m02, m03, m04, m05, m11, m12, m13, m14, m15
        double m22, m23, m24, m25, m33, m34, m35, m44, m45, m55

    def __cinit__(self, Material mat):
        self.dim = 3
        self.sym = 6
        if (self.dim != mat.dim):
            raise ValueError('3d material expected')
        self.mat = mat
        cdef double g = mat.g
        cdef double nu = mat.nu
        self.daux1 = 1.0 / g
        self.daux2 = 0.5 / (g * (1.0 - nu))
        self.daux3 = 0.25 / g
        self.daux4 = 0.5 / g

    @boundscheck(False)
    cdef void update(self, double[:] k):
        """`update(k)`
    
        Compute the coefficients of the underlying matrix for the
        specified value of the wave vector.

        Parameters
        ----------
        k : array_like
            Wave-vector.
        
        """

        # The following tests are necessary, since bounds checks are removed.
        if k.shape[0] != self.dim:
            raise IndexError('shape of k must be ({0},)'.format(self.dim))

        cdef double kx = k[0]
        cdef double ky = k[1]
        cdef double kz = k[2]
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
            #print(self.m00)
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
            self.m03 = -SQRT_TWO * self.daux2 * kxkx * kykz
            self.m04 = SQRT_TWO * kzkx * (self.daux4 - self.daux2 * kxkx)
            self.m05 = SQRT_TWO * kxky * (self.daux4 - self.daux2 * kxkx)
            self.m12 = -self.daux2 * kyky * kzkz
            self.m13 = SQRT_TWO * kykz * (self.daux4 - self.daux2 * kyky)
            self.m14 = -SQRT_TWO * self.daux2 * kyky * kzkx
            self.m15 = SQRT_TWO * kxky * (self.daux4 - self.daux2 * kyky)
            self.m23 = SQRT_TWO * kykz * (self.daux4 - self.daux2 * kzkz)
            self.m24 = SQRT_TWO * kzkx * (self.daux4 - self.daux2 * kzkz)
            self.m25 = -SQRT_TWO * self.daux2 * kzkz * kxky
            self.m34 = 2 * kxky * (self.daux3 - self.daux2 * kzkz)
            self.m35 = 2 * kzkx * (self.daux3 - self.daux2 * kyky)
            self.m45 = 2 * kykz * (self.daux3 - self.daux2 * kxkx)

    @boundscheck(False)
    cpdef double[:] apply(self, double[:] k, double[:] tau,
                          double[:] eps = None):
        """
        `apply(k, tau, eps = None)`
                          
        Apply the Green operator to the specified prestress.

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
        if tau.shape[0] != self.sym:
            raise IndexError('shape of tau must be ({0},)'.format(self.sym))

        if eps is not None:
            if eps.shape[0] != self.sym:
                raise IndexError('shape of eps must be ({0},)'.format(self.sym))
        else:
            eps = array(shape=(self.sym,), itemsize=sizeof(double), format='d')

        self.update(k)
        eps[0] = (self.m00 * tau[0] + self.m01 * tau[1] + self.m02 * tau[2]
                  + self.m03 * tau[3] + self.m04 * tau[4] + self.m05 * tau[5])
        eps[1] = (self.m01 * tau[0] + self.m11 * tau[1] + self.m12 * tau[2]
                  + self.m13 * tau[3] + self.m14 * tau[4] + self.m15 * tau[5])
        eps[2] = (self.m02 * tau[0] + self.m12 * tau[1] + self.m22 * tau[2]
                  + self.m23 * tau[3] + self.m24 * tau[4] + self.m25 * tau[5])
        eps[3] = (self.m03 * tau[0] + self.m13 * tau[1] + self.m23 * tau[2]
                  + self.m33 * tau[3] + self.m34 * tau[4] + self.m35 * tau[5])
        eps[4] = (self.m04 * tau[0] + self.m14 * tau[1] + self.m24 * tau[2]
                  + self.m34 * tau[3] + self.m44 * tau[4] + self.m45 * tau[5])
        eps[5] = (self.m05 * tau[0] + self.m15 * tau[1] + self.m25 * tau[2]
                  + self.m35 * tau[3] + self.m45 * tau[4] + self.m55 * tau[5])
        return eps

    @boundscheck(False)
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
        
        if g is not None:
            if g.shape[0] != self.sym or g.shape[1] != self.sym:
                raise IndexError('shape of g must be ({0}, {0})'
                                 .format(self.sym, self.sym))
        else:
            g = array(shape=(self.sym, self.sym),
                      itemsize=sizeof(double), format='i')
        self.update(k)
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
