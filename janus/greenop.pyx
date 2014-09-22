from cython cimport boundscheck
from cython cimport cdivision
from cython cimport wraparound
from cython.view cimport array

from janus.material.elastic.linear.isotropic cimport IsotropicLinearElasticMaterial as Material
from janus.operators cimport AbstractLinearOperator
from janus.utils.checkarray cimport check_shape_1d


cdef class AbstractGreenOperator(AbstractLinearOperator):

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
