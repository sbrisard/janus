from cpython cimport bool

cdef void check_shape_1d(double[:] a, int n0) except *
cdef double[:] create_or_check_shape_1d(double[:] a, int n0) except *

cdef void check_shape_2d(double[:, :] a, int n0, int n1) except *
cdef double[:, :] create_or_check_shape_2d(double[:, :] a,
                                           int n0, int n1) except *
