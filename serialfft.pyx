from fftw cimport *
from cpython cimport bool
cimport cython
from cython.view cimport array

cdef int SIZEOF_DOUBLE = sizeof(double)
cdef int SIZEOF_COMPLEX = 2 * sizeof(double)
cdef str INVALID_REAL_ARRAY_SHAPE = 'shape of real array must be {0} [was ({1}, {2})]'
cdef str INVALID_COMPLEX_ARRAY_SHAPE = 'shape of complex array must be {0} [was ({1}, {2})]'

cdef int padding(int n):
    if n % 2 == 0:
        return 2
    else:
        return 1

cdef class SerialRealFFT2D:
    cdef:
        double *buffer
        fftw_plan plan_r2c, plan_c2r
        readonly tuple rshape, cshape
        int padding
        int rsize0, rsize1, csize0, csize1

    @cython.boundscheck(False)
    def __cinit__(self, int n0, int n1):
        self.rsize0 = n0
        self.rsize1 = n1
        self.rshape = self.rsize0, self.rsize1
        self.csize0 = self.rsize0
        self.csize1 = 2 * (self.rsize1 / 2 + 1)
        self.cshape = self.csize0, self.csize1
        self.padding = padding(self.rsize1)

        self.buffer = fftw_alloc_real(self.csize0 * self.csize1)
        self.plan_r2c = fftw_plan_dft_r2c_2d(self.rsize0, self.rsize1,
                                             <double *> self.buffer,
                                             <fftw_complex *> self.buffer,
                                             FFTW_ESTIMATE)
        self.plan_c2r = fftw_plan_dft_c2r_2d(self.rsize0, self.rsize1,
                                             <fftw_complex *> self.buffer,
                                             <double *> self.buffer,
                                             FFTW_ESTIMATE)
        
    def __dealloc__(self):
        fftw_free(self.buffer)
        fftw_destroy_plan(self.plan_r2c)
        fftw_destroy_plan(self.plan_c2r)

    @cython.boundscheck(False)
    cdef inline void check_real_array(self, double[:, :] a) except *:
        if a.shape[0] != self.rsize0 or a.shape[1] != self.rsize1:
            raise ValueError(INVALID_REAL_ARRAY_SHAPE.format(self.rshape,
                                                             a.shape[0],
                                                             a.shape[1]))

    @cython.boundscheck(False)
    cdef inline void check_complex_array(self, double[:, :] a) except *:
        if a.shape[0] != self.csize0 or a.shape[1] != self.csize1:
            raise ValueError(INVALID_COMPLEX_ARRAY_SHAPE.format(self.cshape,
                                                                a.shape[0],
                                                                a.shape[1]))

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline copy_to_buffer(self, double[:, :] a,
                               int n0, int n1, int padding):
        cdef:
            int s0, s1, i0, i1
            double *pbuf, *row, *cell
        s0 = a.strides[0] / SIZEOF_DOUBLE
        s1 = a.strides[1] / SIZEOF_DOUBLE
        pbuf = self.buffer
        row = &a[0, 0]
        for i0 in range(n0):
            cell = row 
            for i1 in range(n1):
                pbuf[0] = cell[0]
                pbuf += 1
                cell += s1
            row += s0
            pbuf += padding

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline copy_from_buffer(self, double[:, :] a,
                                 int n0, int n1, int padding):
        cdef:
            int s0, s1, i0, i1
            double *pbuf, *row, *cell
        s0 = a.strides[0] / SIZEOF_DOUBLE
        s1 = a.strides[1] / SIZEOF_DOUBLE
        pbuf = self.buffer
        row = &a[0, 0]
        for i0 in range(n0):
            cell = row
            for i1 in range(n1):
                cell[0] = pbuf[0]
                pbuf += 1
                cell += s1
            row += s0
            pbuf += padding

    cpdef double[:, :] r2c(self, double[:, :] r, double[:, :] c = None):
        self.check_real_array(r)
        if c is None:
             c = array(shape=self.cshape, itemsize=SIZEOF_DOUBLE, format='d')
        else:
             self.check_complex_array(c)

        self.copy_to_buffer(r, self.rsize0, self.rsize1, self.padding)
        fftw_execute(self.plan_r2c)
        self.copy_from_buffer(c, self.csize0, self.csize1, 0)
        
        return c
       
    cpdef double[:, :] c2r(self, double[:, :] c, double[:, :] r = None):
        self.check_complex_array(c)
        if r is None:
             r = array(shape=self.rshape, itemsize=SIZEOF_DOUBLE, format='d')
        else:
             self.check_real_array(r)

        self.copy_to_buffer(c, self.csize0, self.csize1, 0)
        fftw_execute(self.plan_c2r)
        self.copy_from_buffer(r, self.rsize0, self.rsize1, self.padding)
        
        return r

