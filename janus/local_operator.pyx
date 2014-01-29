from cpython cimport PyObject
from cython cimport boundscheck
from cython cimport wraparound

from janus.utils.checkarray cimport check_shape_3d
from janus.utils.checkarray cimport create_or_check_shape_3d
from janus.utils.operators cimport Operator

cdef class BlockDiagonalOperator2D:
    cdef readonly tuple global_shape, local_shape
    cdef readonly int dim
    cdef Operator[:, :] op
    cdef Py_ssize_t n0, n1
    cdef int nrows, ncols

    def __cinit__(self, Operator[:, :] op):
        self.dim = 2
        self.n0 = op.shape[0]
        self.n1 = op.shape[1]
        self.global_shape = (self.n0, self.n1)
        self.nrows = op[0, 0].nrows
        self.ncols = op[0, 0].ncols
        self.local_shape = (self.nrows, self.ncols)
        self.op = op.copy()
        
        cdef int i0, i1, nrows, ncols
        for i0 in range(self.n0):
            for i1 in range(self.n1):
                nrows = op[i0, i1].nrows
                ncols = op[i0, i1].ncols
                if nrows != self.nrows or ncols != self.ncols:
                    raise ValueError('invalid of block operator: '
                                     'expected ({0}, {1}), '
                                     'actual ({2}, {3})'.format(self.nrows,
                                                                self.ncols,
                                                                nrows, ncols))

    @boundscheck(False)
    @wraparound(False)
    cdef c_apply(self, double[:, :, :] x, double[:, :, :] y):
        cdef int i0, i1
        cdef Operator op

        for i0 in range(self.n0):
            for i1 in range(self.n1):
                op = self.op[i0, i1]
                op.c_apply(x[i0, i1, :], y[i0, i1, :])

    @boundscheck(False)
    @wraparound(False)
    def apply(self, double[:, :, :] x, double[:, :, :] y=None):
        check_shape_3d(x, self.n0, self.n1, self.ncols)
        y = create_or_check_shape_3d(y, self.n0, self.n1, self.nrows)
        self.c_apply(x, y)
        return y
