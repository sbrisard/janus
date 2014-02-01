from cython.view cimport array
from janus.matprop cimport IsotropicLinearElasticMaterial as Material

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
    nrows : int
        Number of rows of the matrix representation of the Green operator.
    ncols : int
        Number of columns of the matrix representation of the Green operator.
    mat : IsotropicLinearElasticMaterial
        Reference material.

    """

    cdef readonly int dim, nrows, ncols
    cdef readonly Material mat

    # The auxiliary variables daux1 to daux4 are defined as follows
    #   daux1 = 1 / g
    #   daux2 = 1 / [2 * g * (1 - nu)]
    #   daux3 = 1 / (4 * g)
    #   daux4 = 1 / (2 * g)
    # where g (resp. nu) is the shear modulus (resp. Poisson ratio) of the
    # reference material.
    cdef double daux1, daux2, daux3, daux4

    cdef void c_apply(self, double[:] k, double[:] tau, double[:] eta)
    cdef void c_as_array(self, double[:] k, double[:, :] out)
