cdef class Operator:
    cdef readonly int nrows, ncols

    cdef void c_apply(self, double[:] x, double[:] y)
