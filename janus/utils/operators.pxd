cdef class Operator:
    cdef readonly int nrows, ncols

    cdef void c_apply(self, double[:] x, double[:] y)

cdef class AbstractStructuredOperator2D:
    cdef readonly int ishape0, ishape1, ishape2
    cdef readonly int oshape0, oshape1, oshape2

    cdef void c_apply(self, double[:, :, :] x, double[:, :, :] y)
