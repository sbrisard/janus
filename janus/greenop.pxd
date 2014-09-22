from cython.view cimport array
from janus.operators cimport AbstractLinearOperator

cdef class AbstractGreenOperator(AbstractLinearOperator):

    """This class defines the periodic Green operator for linear elasticity.

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

    """

    cdef readonly int dim

    cdef void c_set_frequency(self, double[:] k)
