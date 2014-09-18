cdef class IsotropicLinearElasticMaterial:
    cdef readonly int dim
    cdef readonly double k
    cdef readonly double g
    cdef readonly double nu
