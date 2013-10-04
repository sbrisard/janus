from fftw cimport *
from libc.stddef cimport ptrdiff_t
import mpi4py
from mpi4py cimport mpi_c as mpi
#cimport mpi4py.mpi_c as mpi

cdef extern from 'fftw3-mpi.h' nogil:    
    void fftw_mpi_init()
    ptrdiff_t fftw_mpi_local_size_2d(ptrdiff_t n0,
                                     ptrdiff_t n1,
                                     mpi.MPI_Comm comm,
                                     ptrdiff_t *local_n0,
                                     ptrdiff_t *local_0_start)
    fftw_plan fftw_mpi_plan_dft_r2c_2d(ptrdiff_t n0, ptrdiff_t n1,
                                       double *input, fftw_complex *output,
                                       mpi.MPI_Comm comm, unsigned flags)
    fftw_plan fftw_mpi_plan_dft_c2r_2d(ptrdiff_t n0, ptrdiff_t n1,
                                       fftw_complex *input, double *output,
                                       mpi.MPI_Comm comm, unsigned flags)

