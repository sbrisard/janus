cimport cython

from cython.view cimport array
from fftw cimport *
from libc.stddef cimport ptrdiff_t

cdef inline int padding(int n):
    if n % 2 == 0:
        return 2
    else:
        return 1

cdef class RealFFT2D:
    cdef:
        int padding
        ptrdiff_t rsize0, rsize1, csize0, csize1
        double *buffer
        fftw_plan plan_r2c, plan_c2r
        readonly ptrdiff_t offset0
        readonly tuple shape, rshape, cshape

    cdef inline void check_real_array(self, double[:, :] r) except *
    cdef inline void check_complex_array(self, double[:, :] c) except *
    cdef inline copy_to_buffer(self, double[:, :] a,
                               ptrdiff_t n0, ptrdiff_t n1, int padding)
    cdef inline copy_from_buffer(self, double[:, :] a,
                                 ptrdiff_t n0, ptrdiff_t n1, int padding)
    cpdef double[:, :] r2c(self, double[:, :] r, double[:, :] c=*)
    cpdef double[:, :] c2r(self, double[:, :] c, double[:, :] r=*)
