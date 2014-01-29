from janus.utils.operators cimport Operator

cdef class FourthRankIsotropicTensor(Operator):
    cdef readonly int dim
    cdef readonly double sph, dev
    cdef double tr
    cdef void c_apply(self, double[:] x, double[:] y)

