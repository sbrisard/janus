from cython cimport boundscheck
from cython cimport cdivision
from cython cimport wraparound
from cython.view cimport array

from janus.material.elastic.linear.isotropic cimport IsotropicLinearElasticMaterial as Material
from janus.operators cimport AbstractLinearOperator
from janus.utils.checkarray cimport check_shape_1d


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
