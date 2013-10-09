cimport cython

from fftw cimport *
from fftw_mpi cimport *
from mpi4py cimport MPI
from serialfft cimport RealFFT2D
from serialfft cimport padding

def init():
    fftw_mpi_init()

cpdef create_parallel_real_fft(ptrdiff_t n0, ptrdiff_t n1, MPI.Comm comm):
    cdef ptrdiff_t n0_loc, offset0
    cdef ptrdiff_t size = 2 * fftw_mpi_local_size_2d(n0, n1 / 2 + 1,
                                                          comm.ob_mpi,
                                                          &n0_loc,
                                                          &offset0) 

    cdef RealFFT2D fft = RealFFT2D(n0, n1, n0_loc, offset0)
    fft.buffer = fftw_alloc_real(size)
    fft.plan_r2c = fftw_mpi_plan_dft_r2c_2d(n0, n1,
                                                 <double *> fft.buffer,
                                                 <fftw_complex *> fft.buffer,
                                                 comm.ob_mpi,
                                                 FFTW_ESTIMATE)
    fft.plan_c2r = fftw_mpi_plan_dft_c2r_2d(n0, n1,
                                            <fftw_complex *> fft.buffer,
                                            <double *> fft.buffer,
                                            comm.ob_mpi,
                                            FFTW_ESTIMATE)

    return fft

