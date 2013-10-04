#from libc.stddef cimport ptrdiff_t
cimport cython

from fftw cimport *
from fftw_mpi cimport *
from serialfft cimport RealFFT2D
from serialfft cimport padding

def init():
    fftw_mpi_init()

cdef class ParallelRealFFT2D(RealFFT2D):
    cdef readonly ptrdiff_t offset0

    @cython.boundscheck(False)
    def __cinit__(self, ptrdiff_t n0, ptrdiff_t n1):
        # TODO Pass any MPI communicator
        cdef ptrdiff_t size = 2 * fftw_mpi_local_size_2d(n0, n1 / 2 + 1,
                                                          mpi.MPI_COMM_WORLD,
                                                          &self.rsize0,
                                                          &self.offset0) 
        self.rsize1 = n1
        self.rshape = self.rsize0, self.rsize1
        self.csize0 = self.rsize0
        self.csize1 = 2 * (n1 / 2 + 1)
        self.cshape = self.csize0, self.csize1
        self.padding = padding(self.rsize1)

        self.buffer = fftw_alloc_real(size)
        self.plan_r2c = fftw_mpi_plan_dft_r2c_2d(n0, n1,
                                                 <double *> self.buffer,
                                                 <fftw_complex *> self.buffer,
                                                 mpi.MPI_COMM_WORLD,
                                                 FFTW_ESTIMATE)
        self.plan_c2r = fftw_mpi_plan_dft_c2r_2d(n0, n1,
                                                 <fftw_complex *> self.buffer,
                                                 <double *> self.buffer,
                                                 mpi.MPI_COMM_WORLD,
                                                 FFTW_ESTIMATE)
        

