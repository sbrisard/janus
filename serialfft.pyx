cimport cython

cdef int SIZEOF_DOUBLE = sizeof(double)
cdef int SIZEOF_COMPLEX = 2 * sizeof(double)
cdef str INVALID_REAL_ARRAY_SHAPE = 'shape of real array must be {0} [was ({1}, {2})]'
cdef str INVALID_COMPLEX_ARRAY_SHAPE = 'shape of complex array must be {0} [was ({1}, {2})]'

#@cython.internal
cdef class RealFFT2D:
    @cython.boundscheck(False)
    def __cinit__(self, ptrdiff_t n0, ptrdiff_t n1,
                  ptrdiff_t n0_loc, ptrdiff_t offset0):
        self.rsize0 = n0_loc
        self.rsize1 = n1
        self.csize0 = n0_loc
        self.csize1 = 2 * (n1 / 2 + 1)
        self.offset0 = offset0
        self.padding = padding(n1)
        self.shape = n0, n1
        self.rshape = self.rsize0, self.rsize1
        self.cshape = self.csize0, self.csize1

    def __dealloc__(self):
        fftw_free(self.buffer)
        fftw_destroy_plan(self.plan_r2c)
        fftw_destroy_plan(self.plan_c2r)

    @cython.boundscheck(False)
    cdef inline void check_real_array(self, double[:, :] r) except *:
        if r.shape[0] != self.rsize0 or r.shape[1] != self.rsize1:
            raise ValueError(INVALID_REAL_ARRAY_SHAPE.format(self.rshape,
                                                             r.shape[0],
                                                             r.shape[1]))

    @cython.boundscheck(False)
    cdef inline void check_complex_array(self, double[:, :] c) except *:
        if c.shape[0] != self.csize0 or c.shape[1] != self.csize1:
            raise ValueError(INVALID_COMPLEX_ARRAY_SHAPE.format(self.cshape,
                                                                c.shape[0],
                                                                c.shape[1]))

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline copy_to_buffer(self, double[:, :] a,
                               ptrdiff_t n0, ptrdiff_t n1, int padding):
        cdef:
            ptrdiff_t s0, s1, i0, i1
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
                                 ptrdiff_t n0, ptrdiff_t n1, int padding):
        cdef:
            ptrdiff_t s0, s1, i0, i1
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

cpdef create_serial_real_fft(ptrdiff_t n0, ptrdiff_t n1):

    cdef RealFFT2D fft = RealFFT2D(n0, n1, n0, 0)
    fft.buffer = fftw_alloc_real(2 * n0 * (n1 / 2 + 1))
    fft.plan_r2c = fftw_plan_dft_r2c_2d(n0, n1,
                                        fft.buffer, <fftw_complex *> fft.buffer,
                                        FFTW_ESTIMATE)
    fft.plan_c2r = fftw_plan_dft_c2r_2d(n0, n1,
                                        <fftw_complex *> fft.buffer, fft.buffer,
                                        FFTW_ESTIMATE)
    return fft
    
cdef class SerialRealFFT3D:
    cdef:
        int rsize0, rsize1, rsize2, csize0, csize1, csize2, padding
        double *buffer
        fftw_plan plan_r2c, plan_c2r
        readonly tuple rshape, cshape

    @cython.boundscheck(False)
    def __cinit__(self, int n0, int n1, int n2):
        self.rsize0 = n0
        self.rsize1 = n1
        self.rsize2 = n2
        self.rshape = self.rsize0, self.rsize1, self.rsize2
        self.csize0 = self.rsize0
        self.csize1 = self.rsize1
        self.csize2 = 2 * (self.rsize2 / 2 + 1)
        self.cshape = self.csize0, self.csize1, self.csize2
        self.padding = padding(self.rsize2)

        self.buffer = fftw_alloc_real(self.csize0 * self.csize1 * self.csize2)
        self.plan_r2c = fftw_plan_dft_r2c_3d(n0, n1, n2,
                                             <double *> self.buffer,
                                             <fftw_complex *> self.buffer,
                                             FFTW_ESTIMATE)
        self.plan_c2r = fftw_plan_dft_c2r_3d(n0, n1, n2,
                                             <fftw_complex *> self.buffer,
                                             <double *> self.buffer,
                                             FFTW_ESTIMATE)
        
    def __dealloc__(self):
        fftw_free(self.buffer)
        fftw_destroy_plan(self.plan_r2c)
        fftw_destroy_plan(self.plan_c2r)

    @cython.boundscheck(False)
    cdef inline void check_real_array(self, double[:, :, :] r) except *:
        if (r.shape[0] != self.rsize0 or r.shape[1] != self.rsize1
            or r.shape[2] != self.rsize2):
            # TODO Adapt error message
            raise ValueError(INVALID_REAL_ARRAY_SHAPE.format(self.rshape,
                                                             r.shape[0],
                                                             r.shape[1]))

    @cython.boundscheck(False)
    cdef inline void check_complex_array(self, double[:, :, :] c) except *:
        if (c.shape[0] != self.csize0 or c.shape[1] != self.csize1
                or c.shape[2] != self.csize2):
            # TODO Adapt error message
            raise ValueError(INVALID_COMPLEX_ARRAY_SHAPE.format(self.cshape,
                                                                c.shape[0],
                                                                c.shape[1]))

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline copy_to_buffer(self, double[:, :, :] a,
                               int n0, int n1, int n2, int padding):
        cdef:
            int s0, s1, s2, i0, i1, i2
            double *pbuf, *pslice, *prow, *pcell
        s0 = a.strides[0] / SIZEOF_DOUBLE
        s1 = a.strides[1] / SIZEOF_DOUBLE
        s2 = a.strides[2] / SIZEOF_DOUBLE
        pbuf = self.buffer
        pslice = &a[0, 0, 0]
        for i0 in range(n0):
            prow = pslice 
            for i1 in range(n1):
                pcell = prow
                for i2 in range(n2):
                    pbuf[0] = pcell[0]
                    pbuf += 1
                    pcell += s2
                pbuf += padding
                prow += s1
            pslice += s0

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline copy_from_buffer(self, double[:, :, :] a,
                                 int n0, int n1, int n2, int padding):
        cdef:
            int s0, s1, s2, i0, i1, i2
            double *pbuf, *pslice, *prow, *pcell
        s0 = a.strides[0] / SIZEOF_DOUBLE
        s1 = a.strides[1] / SIZEOF_DOUBLE
        s2 = a.strides[2] / SIZEOF_DOUBLE
        pbuf = self.buffer
        pslice = &a[0, 0, 0]
        for i0 in range(n0):
            prow = pslice 
            for i1 in range(n1):
                pcell = prow
                for i2 in range(n2):
                    pcell[0] = pbuf[0]
                    pbuf += 1
                    pcell += s2
                pbuf += padding
                prow += s1
            pslice += s0

    cpdef double[:, :, :] r2c(self, double[:, :, :] r,
                              double[:, :, :] c = None):
        self.check_real_array(r)
        if c is None:
             c = array(shape=self.cshape, itemsize=SIZEOF_DOUBLE, format='d')
        else:
             self.check_complex_array(c)

        self.copy_to_buffer(r, self.rsize0, self.rsize1, self.rsize2,
                            self.padding)
        fftw_execute(self.plan_r2c)
        self.copy_from_buffer(c, self.csize0, self.csize1, self.csize2, 0)
        
        return c
       
    cpdef double[:, :, :] c2r(self, double[:, :, :] c,
                              double[:, :, :] r = None):
        self.check_complex_array(c)
        if r is None:
             r = array(shape=self.rshape, itemsize=SIZEOF_DOUBLE, format='d')
        else:
             self.check_real_array(r)

        self.copy_to_buffer(c, self.csize0, self.csize1, self.csize2, 0)
        fftw_execute(self.plan_c2r)
        self.copy_from_buffer(r, self.rsize0, self.rsize1, self.rsize2,
                              self.padding)
        
        return r

