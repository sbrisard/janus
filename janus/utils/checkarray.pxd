from cpython cimport bool

#
# Arrays of int
#
cdef void check_shape_1i(int[:] a, int n0) except *

#
# Arrays of double
#
cdef void check_shape_1d(double[:] a, int n0) except *
cdef double[:] create_or_check_shape_1d(double[:] a, int n0) except *

cdef void check_shape_2d(double[:, :] a, int n0, int n1) except *
cdef double[:, :] create_or_check_shape_2d(double[:, :] a,
                                           int n0, int n1) except *

cdef void check_shape_3d(double[:, :, :] a,
                         int n0, int n1, int n2) except *
cdef double[:, :, :] create_or_check_shape_3d(double[:, :, :] a,
                                              int n0, int n1, int n2) except *

cdef void check_shape_4d(double[:, :, :, :] a,
                         int n0, int n1, int n2, int n3) except *
cdef double[:, :, :, :] create_or_check_shape_4d(double[:, :, :, :] a,
                                                 int n0, int n1, int n2, int n3) except *
