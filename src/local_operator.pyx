from cpython cimport PyObject
from cython cimport boundscheck
from cython cimport wraparound

from checkarray cimport check_shape_3d
from checkarray cimport create_or_check_shape_3d

from interfaces cimport Operator

cdef class LocalOperator2D:
    cdef readonly tuple shape
    cdef readonly int dim
    cdef object[:, :] op
    cdef Py_ssize_t n0, n1

    def __cinit__(self, object[:, :] op):
        # TODO correct this line
        self.shape = None
        self.dim = 2
        self.n0 = op.shape[0]
        self.n1 = op.shape[1]
        # TODO this is potentially dangerous. Take a deep copy instead
        self.op = op

    @boundscheck(False)
    @wraparound(False)
    cdef c_apply(self, double[:, :, :] x, double[:, :, :] y):
        cdef int i0, i1
        cdef Operator op

        for i0 in range(self.n0):
            for i1 in range(self.n1):
                # TODO This should be optimized
                op = self.op[i0, i1]
                op.c_apply(x[i0, i1, :], y[i0, i1, :])
                #self.op[i0, i1].apply(x[i0, i1, :], y[i0, i1, :])

    @boundscheck(False)
    @wraparound(False)
    def apply(self, double[:, :, :] x, double[:, :, :] y=None):
        # TODO How to specify shape of input and output?
        #check_shape_3d(x, n0, n1,
        self.c_apply(x, y)
        return y
