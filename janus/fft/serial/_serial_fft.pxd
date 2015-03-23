from janus.fft.serial.fftw cimport *
from libc.stddef cimport ptrdiff_t

cdef inline int padding(int n):
    if n % 2 == 0:
        return 2
    else:
        return 1

cdef class _RealFFT2D:
    cdef:
        int padding
        ptrdiff_t ishape0, ishape1, oshape0, oshape1
        double *buffer
        double scaling
        fftw_plan plan_r2c, plan_c2r
        readonly ptrdiff_t isize, osize, offset0, idispl, odispl
        readonly tuple shape, ishape, oshape, global_oshape

    cdef inline void check_real_array(self, double[:, :] r) except *
    cdef inline void check_complex_array(self, double[:, :] c) except *
    cdef inline copy_to_buffer(self, double[:, :] a,
                               ptrdiff_t n0, ptrdiff_t n1, int padding)
    cdef inline copy_from_buffer(self, double[:, :] a,
                                 ptrdiff_t n0, ptrdiff_t n1, int padding,
                                 double scaling)
    cpdef double[:, :] r2c(self, double[:, :] r, double[:, :] c=*)
    cpdef double[:, :] c2r(self, double[:, :] c, double[:, :] r=*)

cdef class _RealFFT3D:
    cdef:
        int padding
        ptrdiff_t ishape0, ishape1, ishape2, oshape0, oshape1, oshape2
        double *buffer
        double scaling
        fftw_plan plan_r2c, plan_c2r
        readonly ptrdiff_t isize, osize, offset0, idispl, odispl
        readonly tuple shape, ishape, oshape, global_oshape

    cdef inline void check_real_array(self, double[:, :, :] r) except *
    cdef inline void check_complex_array(self, double[:, :, :] c) except *
    cdef inline copy_to_buffer(self, double[:, :, :] a,
                               ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                               int padding)
    cdef inline copy_from_buffer(self, double[:, :, :] a,
                                 ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                 int padding, double scaling)
    cpdef double[:, :, :] r2c(self, double[:, :, :] r,
                              double[:, :, :] c=*)
    cpdef double[:, :, :] c2r(self, double[:, :, :] c,
                              double[:, :, :] r=*)
