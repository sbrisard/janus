from cython.view cimport array
from matprop cimport IsotropicLinearElasticMaterial as Material

cdef class GreenOperator:

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
    # Dimension of the space on which this object operates (space of the
    # second-rank, symmetric tensors).
    cdef int sym
    # Let g (resp. nu) be the shear modulus (resp. Poisson ratio) of the
    # reference material. Then
    #   daux1 = 1 / g
    #   daux2 = 1 / [2 * g * (1 - nu)]
    #   daux3 = 1 / (4 * g)
    #   daux4 = 1 / (2 * g)
    cdef double daux1, daux2, daux3, daux4

    cdef void update(self, double[:] k)
    cpdef double[:] apply(self, double[:] k, double[:] tau, double[:] eps=*)
