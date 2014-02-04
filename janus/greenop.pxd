from cython.view cimport array
from janus.matprop cimport IsotropicLinearElasticMaterial as Material
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
    cdef readonly Material mat

    # The auxiliary variables daux1 to daux4 are defined as follows
    #   daux1 = 1 / g
    #   daux2 = 1 / [2 * g * (1 - nu)]
    #   daux3 = 1 / (4 * g)
    #   daux4 = 1 / (2 * g)
    # where g (resp. nu) is the shear modulus (resp. Poisson ratio) of the
    # reference material.
    cdef double daux1, daux2, daux3, daux4

    cdef void c_set_frequency(self, double[:] k)
