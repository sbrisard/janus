cimport cython

from cython.view cimport array

cdef int SIZEOF_DOUBLE = sizeof(double)
cdef int SIZEOF_COMPLEX = 2 * sizeof(double)
cdef str INVALID_REAL_ARRAY_SHAPE = 'shape of real array must be {0} [was ({1}, {2})]'
cdef str INVALID_COMPLEX_ARRAY_SHAPE = 'shape of complex array must be {0} [was ({1}, {2})]'

def create_real(shape):
    if len(shape) == 2:
        return create_real_2D(shape[0], shape[1])
    elif len(shape) == 3:
        return create_real_3D(shape[0], shape[1], shape[2])
    else:
        msg = 'length of shape can be 2 or 3 (was {0})'
        raise ValueError(msg.format(len(shape)))

cdef create_real_2D(ptrdiff_t n0, ptrdiff_t n1):

    cdef _RealFFT2D fft = _RealFFT2D(n0, n1, n0, 0)
    fft.buffer = fftw_alloc_real(2 * n0 * (n1 / 2 + 1))
    fft.plan_r2c = fftw_plan_dft_r2c_2d(n0, n1,
                                        fft.buffer, <fftw_complex *> fft.buffer,
                                        FFTW_ESTIMATE)
    fft.plan_c2r = fftw_plan_dft_c2r_2d(n0, n1,
                                        <fftw_complex *> fft.buffer, fft.buffer,
                                        FFTW_ESTIMATE)
    return fft

cdef create_real_3D(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2):

    cdef _RealFFT3D fft = _RealFFT3D(n0, n1, n2, n0, 0)
    fft.buffer = fftw_alloc_real(fft.cshape0 * fft.cshape1 * fft.cshape2)
    fft.plan_r2c = fftw_plan_dft_r2c_3d(n0, n1, n2,
                                        fft.buffer, <fftw_complex *> fft.buffer,
                                        FFTW_ESTIMATE)
    fft.plan_c2r = fftw_plan_dft_c2r_3d(n0, n1, n2,
                                        <fftw_complex *> fft.buffer, fft.buffer,
                                        FFTW_ESTIMATE)
    return fft

cdef class _RealFFT2D:

    @cython.boundscheck(False)
    def __cinit__(self, ptrdiff_t n0, ptrdiff_t n1,
                  ptrdiff_t n0_loc, ptrdiff_t offset0):
        self.rshape0 = n0_loc
        self.rshape1 = n1
        self.cshape0 = n0_loc
        self.cshape1 = 2 * (n1 / 2 + 1)
        self.offset0 = offset0
        self.padding = padding(n1)
        self.shape = n0, n1
        self.rshape = self.rshape0, self.rshape1
        self.cshape = self.cshape0, self.cshape1
        self.scaling = 1. / <double> (n0 * n1)

    def __dealloc__(self):
        fftw_free(self.buffer)
        fftw_destroy_plan(self.plan_r2c)
        fftw_destroy_plan(self.plan_c2r)

    @cython.boundscheck(False)
    cdef inline void check_real_array(self, double[:, :] r) except *:
        if r.shape[0] != self.rshape0 or r.shape[1] != self.rshape1:
            raise ValueError(INVALID_REAL_ARRAY_SHAPE.format(self.rshape,
                                                             r.shape[0],
                                                             r.shape[1]))

    @cython.boundscheck(False)
    cdef inline void check_complex_array(self, double[:, :] c) except *:
        if c.shape[0] != self.cshape0 or c.shape[1] != self.cshape1:
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
            double *pbuf
            double *row
            double *cell
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
                                 ptrdiff_t n0, ptrdiff_t n1, int padding,
                                 double scaling):
        cdef:
            ptrdiff_t s0, s1, i0, i1
            double *pbuf
            double *row
            double *cell
        s0 = a.strides[0] / SIZEOF_DOUBLE
        s1 = a.strides[1] / SIZEOF_DOUBLE
        pbuf = self.buffer
        row = &a[0, 0]
        if scaling == 1.:
            for i0 in range(n0):
                cell = row
                for i1 in range(n1):
                    cell[0] = pbuf[0]
                    pbuf += 1
                    cell += s1
                row += s0
                pbuf += padding
        else:
            for i0 in range(n0):
                cell = row
                for i1 in range(n1):
                    cell[0] = scaling * pbuf[0]
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

        self.copy_to_buffer(r, self.rshape0, self.rshape1, self.padding)
        fftw_execute(self.plan_r2c)
        self.copy_from_buffer(c, self.cshape0, self.cshape1, 0, 1.)

        return c

    cpdef double[:, :] c2r(self, double[:, :] c, double[:, :] r = None):
        self.check_complex_array(c)
        if r is None:
             r = array(shape=self.rshape, itemsize=SIZEOF_DOUBLE, format='d')
        else:
             self.check_real_array(r)

        self.copy_to_buffer(c, self.cshape0, self.cshape1, 0)
        fftw_execute(self.plan_c2r)
        self.copy_from_buffer(r, self.rshape0, self.rshape1, self.padding,
                              self.scaling)

        return r

cdef class _RealFFT3D:

    @cython.boundscheck(False)
    def __cinit__(self, ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                  ptrdiff_t n0_loc, ptrdiff_t offset0):
        self.rshape0 = n0_loc
        self.rshape1 = n1
        self.rshape2 = n2
        self.cshape0 = n0_loc
        self.cshape1 = n1
        self.cshape2 = 2 * (n2 / 2 + 1)
        self.offset0 = offset0
        self.padding = padding(self.rshape2)
        self.shape = n0, n1, n2
        self.rshape = self.rshape0, self.rshape1, self.rshape2
        self.cshape = self.cshape0, self.cshape1, self.cshape2
        self.scaling = 1. / <double> (n0 * n1 * n2)

    def __dealloc__(self):
        fftw_free(self.buffer)
        fftw_destroy_plan(self.plan_r2c)
        fftw_destroy_plan(self.plan_c2r)

    @cython.boundscheck(False)
    cdef inline void check_real_array(self, double[:, :, :] r) except *:
        if (r.shape[0] != self.rshape0 or r.shape[1] != self.rshape1
            or r.shape[2] != self.rshape2):
            # TODO Adapt error message
            raise ValueError(INVALID_REAL_ARRAY_SHAPE.format(self.rshape,
                                                             r.shape[0],
                                                             r.shape[1]))

    @cython.boundscheck(False)
    cdef inline void check_complex_array(self, double[:, :, :] c) except *:
        if (c.shape[0] != self.cshape0 or c.shape[1] != self.cshape1
                or c.shape[2] != self.cshape2):
            # TODO Adapt error message
            raise ValueError(INVALID_COMPLEX_ARRAY_SHAPE.format(self.cshape,
                                                                c.shape[0],
                                                                c.shape[1]))

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline copy_to_buffer(self, double[:, :, :] a,
                               ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                               int padding):
        cdef:
            int s0, s1, s2, i0, i1, i2
            double *pbuf
            double *pslice
            double *prow
            double *pcell
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
                                 ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                 int padding, double scaling):
        cdef:
            int s0, s1, s2, i0, i1, i2
            double *pbuf
            double *pslice
            double *prow
            double *pcell
        s0 = a.strides[0] / SIZEOF_DOUBLE
        s1 = a.strides[1] / SIZEOF_DOUBLE
        s2 = a.strides[2] / SIZEOF_DOUBLE
        pbuf = self.buffer
        pslice = &a[0, 0, 0]
        if scaling == 1.:
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
        else:
            for i0 in range(n0):
                prow = pslice
                for i1 in range(n1):
                    pcell = prow
                    for i2 in range(n2):
                        pcell[0] = scaling * pbuf[0]
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

        self.copy_to_buffer(r, self.rshape0, self.rshape1, self.rshape2,
                            self.padding)
        fftw_execute(self.plan_r2c)
        self.copy_from_buffer(c, self.cshape0, self.cshape1, self.cshape2, 0, 1.)

        return c

    cpdef double[:, :, :] c2r(self, double[:, :, :] c,
                              double[:, :, :] r = None):
        self.check_complex_array(c)
        if r is None:
             r = array(shape=self.rshape, itemsize=SIZEOF_DOUBLE, format='d')
        else:
             self.check_real_array(r)

        self.copy_to_buffer(c, self.cshape0, self.cshape1, self.cshape2, 0)
        fftw_execute(self.plan_c2r)
        self.copy_from_buffer(r, self.rshape0, self.rshape1, self.rshape2,
                              self.padding, self.scaling)


        return r
