# The signature of the static function allowing application of a fourth rank,
# isotropic tensor to a second rank, symmetric tensor.
ctypedef void (*isotropic4_apply_t)(double, double, double[:], double[:])

cdef class IsotropicFourthRankTensor:
    cdef int dim, sym
    cdef readonly double sph, dev
    cdef double s
    cdef isotropic4_apply_t static_c_apply

    cdef inline void c_apply(self, double[:] x, double[:] y)

cdef class IsotropicFourthRankTensor2D:
    cdef int dim
    cdef readonly double sph, dev
    cdef double s

    cdef inline void c_apply(self, double[:] x, double[:] y)

cdef class IsotropicFourthRankTensor3D:
    cdef int dim
    cdef readonly double sph, dev
    cdef double s

    cdef inline void c_apply(self, double[:] x, double[:] y)
