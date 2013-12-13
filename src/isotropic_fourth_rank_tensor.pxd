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
