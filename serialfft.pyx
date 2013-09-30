from fftw cimport *

import numpy as np
cimport numpy as np
np.import_array()

cdef int padding(int n):
    if n % 2 == 0:
        return 2
    else:
        return 1

cdef class SerialFFT2D:
    cdef:
        double *buffer
        fftw_plan plan_r2c
        # TODO specifiy type of elements in tuple
        # TODO these attributes are not visible in Python.
        tuple ishape, oshape
        int padding

    def __cinit__(self, tuple shape):
        # TODO check that shape is a pair
        cdef int n0 = shape[0]
        cdef int n1 = shape[1]
        self.ishape = shape
        self.oshape = n0, n1 / 2 + 1
        self.padding = padding(n1)

        cdef size_t n = 2 * n0 * (n1 / 2 + 1)
        self.buffer = fftw_alloc_real(n)
        self.plan_r2c = fftw_plan_dft_r2c_2d(n0, n1,
                                             <double *> self.buffer,
                                             <fftw_complex *> self.buffer,
                                             FFTW_ESTIMATE)
        
    def __dealloc__(self):
        fftw_free(self.buffer)
        fftw_destroy_plan(self.plan_r2c)

    def r2c(self,
            np.ndarray[np.float64_t, ndim=2] ain,
            np.ndarray[np.complex128_t, ndim=2] aout = None):
        cdef int i, j, offset
        # TODO declare type of self.ishape so that loop gets optimized.
        offset = 0
        for i in range(self.ishape[0]):
            for j in range(self.ishape[1]):
                self.buffer[offset] = ain[i, j]
                offset += 1
            offset += self.padding
        
        fftw_execute(self.plan_r2c)
        if aout is None:
            aout = np.empty(self.oshape, np.complex128)
        # TODO declare type of self.ishape so that loop gets optimized.
        offset = 0
        for i in range(self.oshape[0]):
            for j in range(self.oshape[1]):
                aout[i, j] = self.buffer[offset] + 1j * self.buffer[offset + 1]
                offset += 2
        return aout
        
    def in_shape(self):
        return self.ishape
                
    def out_shape(self):
        return self.oshape
