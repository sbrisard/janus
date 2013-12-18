cdef class FourthRankIsotropicTensor:
    cdef int dim, sym
    cdef readonly double sph, dev
    cdef double tr
    cdef void c_apply(self, double[:] x, double[:] y)

