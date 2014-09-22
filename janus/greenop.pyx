from cython cimport boundscheck
from cython cimport cdivision
from cython cimport wraparound
from cython.view cimport array

from janus.material.elastic.linear.isotropic cimport IsotropicLinearElasticMaterial as Material
from janus.operators cimport AbstractLinearOperator
from janus.utils.checkarray cimport check_shape_1d


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
