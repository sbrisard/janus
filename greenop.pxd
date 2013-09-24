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

    cdef readonly int dim
    cdef readonly Material mat
    cdef int sym
    cdef double daux1, daux2, daux3, daux4

    cdef void check_k(self, double[:] k) except *
    cdef void check_tau(self, double[:] tau) except *    
    cdef void check_eps(self, double[:] eps) except *
    cdef double[:] pre_apply(self, double[:] k, double[:] tau, double[:] eps)
    cdef double[:, :] pre_asarray(self, double[:] k, double[:, :] g) except *
    cdef void update(self, double[:] k)
    cpdef double[:] apply(self, double[:] k, double[:] tau, double[:] eps=*)
