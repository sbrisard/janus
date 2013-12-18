


cdef class LocalOperator2D:
    cdef readonly tuple shape
    cdef readonly int dim
    cdef object[:, :] ops
    cdef Py_ssize_t n0, n1

    def __cinit__(self, object[:, :] ops):
        # TODO correct this line
        self.shape = None
        self.dim = 2
        self.n0 = ops.shape[0]
        self.n1 = ops.shape[1]
        # TODO this is potentially dangerous. Take a deep copy instead
        self.ops = ops

    def apply(self, double[:, :, :] x, double[:, :, :] y=None):
        cdef Py_ssize_t i0, i1
        for i0 in range(self.n0):
            for i1 in range(self.n1):
                self.ops[i0, i1].apply(x[i0, i1, :], y[i0, i1, :])
        return y
