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

    # The auxiliary variables daux1 to daux4 are defined as follows
    #   daux1 = 1 / g
    #   daux2 = 1 / [2 * g * (1 - nu)]
    #   daux3 = 1 / (4 * g)
    #   daux4 = 1 / (2 * g)
    # where g (resp. nu) is the shear modulus (resp. Poisson ratio) of the
    # reference material.
    cdef double daux1, daux2, daux3, daux4

    cdef void capply(self, double *k, double[:] tau, double[:] eta)
    cdef void c_as_array(self, double *k, double[:, :] out)
