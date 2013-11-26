from cpython cimport bool

#
# Arrays of Py_ssize_t
#
cdef void check_shape_ssize_t_1D(Py_ssize_t[:] a, int n0) except *

#
# Arrays of double
#
cdef void check_shape_1d(double[:] a, int n0) except *
cdef double[:] create_or_check_shape_1d(double[:] a, int n0) except *

cdef void check_shape_2d(double[:, :] a, int n0, int n1) except *
cdef double[:, :] create_or_check_shape_2d(double[:, :] a,
                                           int n0, int n1) except *
